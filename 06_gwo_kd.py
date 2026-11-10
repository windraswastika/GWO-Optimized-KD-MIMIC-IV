"""
06_gwo_kd.py — Stage 6: GWO-guided Knowledge Distillation (Section 2.8–2.10)
=============================================================================
Uses the trained teacher + GWO to find optimal KD hyperparameters,
then distils the student and fine-tunes with hard labels.

Pipeline:
  1. Load teacher (canonical best checkpoint)
  2. GWO search: N_w=10 wolves × I_max=10 iterations × ≥5 seeds
     Fitness = val AUROC of student trained with proposed (T, α_kd, β_kd, s)
  3. Full student distillation with GWO-optimal params (all epochs)
  4. Hard-label fine-tuning (lr=1e-4, 10 epochs)
  5. Save checkpoint + efficiency metrics

Loss (Equation from Section 2.9):
  L = α_kd · BCE(σ(z_s), y)
    + (1 - α_kd) · T² · KL(softmax(z_t/T) ‖ softmax(z_s/T))
    + β_kd · MSE(r(f_s), f_t)

where r(·) is a learned linear regressor mapping f_s dim → f_t dim.

Run:
    python src/06_gwo_kd.py
    python src/06_gwo_kd.py --teacher_ckpt checkpoints/teacher_tcn_best.pt
"""

import sys, os, argparse, time, json
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
from pathlib import Path

from torch.utils.data import DataLoader, WeightedRandomSampler

import config as cfg
from utils.data_utils import (
    make_dataloader, load_split_arrays,
    SepsisWindowDataset, compute_class_weights,
    measure_latency_cpu, model_size_mb, count_params
)
from utils.metrics import auroc, dual_threshold_eval, bootstrap_auroc_ci
from utils.gwo import run_gwo_multiseed, decode_position
from models.teacher import build_teacher
from models.students import build_student


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def get_device():
    # NOTE: a CPU-only + multithreaded-wolf-evaluation variant was tried here
    # to speed up GWO search. Measured result: with n_workers>1 threads, GIL
    # contention over the many small Python-level ops in the training loop
    # serialised everything (~105% aggregate CPU over a 1h45m test producing
    # zero completed evaluations) — worse than the original single-threaded
    # MPS run. Reverted to the original auto-detected device; do not retry
    # ThreadPoolExecutor-based wolf parallelism for this workload without a
    # true multi-process design (and note: naive multiprocessing would
    # replicate the 6.7GB train dataset per worker process — check memory
    # budget before attempting that).
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_shared_loader(
    dataset,
    batch_size: int,
    shuffle: bool = False,
    sample_weights: torch.Tensor | None = None,
) -> DataLoader:
    """
    Build a fresh, lightweight DataLoader around an already-in-memory Dataset.
    Cheap to call repeatedly (no disk I/O / tensor copy), and safe to call
    concurrently from multiple threads since each call yields an independent
    DataLoader/iterator over the same read-only Dataset.
    """
    if sample_weights is not None:
        sampler = WeightedRandomSampler(
            weights=sample_weights, num_samples=len(sample_weights), replacement=True
        )
        return DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                          num_workers=0, pin_memory=False)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=0, pin_memory=False)


# =============================================================================
# A100 Optimisation — GPU-resident batch iterator
# =============================================================================
_GPU_CACHE: dict = {}   # id(dataset) → (X_gpu, y_gpu), loaded once

class GPUBatcher:
    """
    Drop-in DataLoader replacement for GWO fitness evaluations.

    Strategy:
      1. First call moves dataset tensors to GPU VRAM (one-time, ~2 s).
      2. All subsequent calls (per-wolf, per-iteration) reuse the cached
         GPU tensors — zero CPU→GPU transfer overhead.
      3. Batch size 4096 fills A100 compute units far better than 256.

    Memory budget on A100 80 GB:
      train  4.66M × 12 × 21 × float32 ≈ 4.7 GB
      val    0.99M × 12 × 21 × float32 ≈ 1.0 GB
      ─────────────────────────────────────────
      total                             ≈ 5.7 GB  (well within 80 GB)
    """
    def __init__(self, dataset, batch_size: int, device: torch.device,
                 weights=None, shuffle: bool = False):
        global _GPU_CACHE
        key = id(dataset)
        if key not in _GPU_CACHE:
            _GPU_CACHE[key] = (dataset.X.to(device), dataset.y.to(device))
        self.X, self.y = _GPU_CACHE[key]
        self.n       = len(self.y)
        self.bs      = batch_size
        self.device  = device
        self.w       = weights.to(device) if weights is not None else None
        self.shuffle = shuffle

    def __iter__(self):
        if self.w is not None:                          # weighted sampling
            idx = torch.multinomial(self.w, self.n, replacement=True)
        elif self.shuffle:
            idx = torch.randperm(self.n, device=self.device)
        else:
            idx = torch.arange(self.n, device=self.device)
        for s in range(0, self.n, self.bs):
            b = idx[s : s + self.bs]
            yield self.X[b], self.y[b]                 # already on GPU

    def __len__(self):
        import math
        return math.ceil(self.n / self.bs)



# =============================================================================
# Feature regressor r(f_s) → f_t   (learned projection for feature-KD)
# =============================================================================

class FeatureRegressor(nn.Module):
    """Linear projection to align student feature space with teacher's."""
    def __init__(self, s_dim: int, t_dim: int):
        super().__init__()
        self.proj = nn.Linear(s_dim, t_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


# =============================================================================
# KD loss (Section 2.9)
# =============================================================================

def kd_loss(
    z_s     : torch.Tensor,   # student logits   (B,)
    z_t     : torch.Tensor,   # teacher logits   (B,)
    f_s_proj: torch.Tensor,   # projected student features (B, t_feat_dim)
    f_t     : torch.Tensor,   # teacher features (B, t_feat_dim)
    y       : torch.Tensor,   # ground truth labels (B,)
    T       : float,
    alpha   : float,
    beta    : float,
    pos_weight: torch.Tensor,
) -> torch.Tensor:
    """
    Combined KD loss:
      L = α · BCE(σ(z_s), y)
        + (1-α) · T² · KL(p_t ‖ p_s)
        + β · MSE(f_s_proj, f_t.detach())
    """
    # (1) Task loss: BCE
    bce_fn  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    L_bce   = bce_fn(z_s, y)

    # (2) KL divergence (soft targets)
    # For binary classification: treat logits as 2-class [negative, positive]
    logits_t = torch.stack([-z_t, z_t], dim=1) / T
    logits_s = torch.stack([-z_s, z_s], dim=1) / T
    p_t      = F.softmax(logits_t, dim=1).detach()
    log_p_s  = F.log_softmax(logits_s, dim=1)
    L_kl     = F.kl_div(log_p_s, p_t, reduction="batchmean") * (T ** 2)

    # (3) Feature alignment: MSE between projected student and teacher features
    L_feat   = F.mse_loss(f_s_proj, f_t.detach())

    return alpha * L_bce + (1 - alpha) * L_kl + beta * L_feat


# =============================================================================
# Quick-train for GWO fitness evaluation (few epochs only)
# =============================================================================

def quick_train_student(
    student  : nn.Module,
    regressor: nn.Module,
    teacher  : nn.Module,
    params   : dict,
    device   : torch.device,
    n_epochs : int = 5,
    pos_weight: torch.Tensor = None,
    train_dataset=None,
    val_dataset=None,
    train_weights: torch.Tensor = None,
) -> float:
    """
    Fast student training (n_epochs) for GWO fitness evaluation.
    Returns val AUROC.
    """
    T       = params["T"]
    alpha   = params["alpha_kd"]
    beta    = params["beta_kd"]

    if pos_weight is None:
        pos_weight = torch.tensor(1.0, dtype=torch.float32).to(device)

    # A100: GPU-resident batching, 16× larger batch than default
    _BS = 4096
    train_loader = GPUBatcher(train_dataset, _BS, device, weights=train_weights)
    val_loader   = GPUBatcher(val_dataset,   _BS, device)

    optimizer = torch.optim.Adam(
        list(student.parameters()) + list(regressor.parameters()),
        lr=cfg.TEACHER_LR
    )

    student.train()
    regressor.train()
    teacher.eval()

    for ep in range(n_epochs):
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            with torch.no_grad():
                z_t, f_t = teacher(X_b)
            z_s, f_s = student(X_b)
            f_s_proj = regressor(f_s)
            loss = kd_loss(z_s, z_t, f_s_proj, f_t, y_b, T, alpha, beta, pos_weight)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(student.parameters()) + list(regressor.parameters()), 1.0)
            optimizer.step()

    # Evaluate
    student.eval()
    all_scores, all_labels = [], []
    with torch.no_grad():
        for X_b, y_b in val_loader:
            X_b = X_b.to(device)
            z_s, _ = student(X_b)
            prob = torch.sigmoid(z_s).cpu().numpy()
            all_scores.append(prob)
            all_labels.append(y_b.cpu().numpy())  # cpu() safe for GPU tensors

    y_score = np.concatenate(all_scores)
    y_true  = np.concatenate(all_labels)
    return auroc(y_true, y_score)


# =============================================================================
# Full student distillation + fine-tuning
# =============================================================================

def full_distil(
    student  : nn.Module,
    regressor: nn.Module,
    teacher  : nn.Module,
    params   : dict,
    device   : torch.device,
    pos_weight: torch.Tensor,
    train_dataset=None,
    val_dataset=None,
    train_weights: torch.Tensor = None,
) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Full KD training + hard-label fine-tuning.
    Returns: (best_val_auroc, y_true_val, y_score_val)
    """
    T, alpha, beta = params["T"], params["alpha_kd"], params["beta_kd"]
    log(f"  Full distillation: T={T:.1f}  α={alpha:.2f}  β={beta:.3f}")

    # A100: GPU-resident batching, 16× larger batch than default
    _BS = 4096
    train_loader = GPUBatcher(train_dataset, _BS, device, weights=train_weights)
    val_loader   = GPUBatcher(val_dataset,   _BS, device)

    optimizer = torch.optim.Adam(
        list(student.parameters()) + list(regressor.parameters()),
        lr=cfg.TEACHER_LR
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5, min_lr=1e-6
    )

    best_val_auc  = -1.0
    best_ckpt_buf = None
    patience_cnt  = 0
    teacher.eval()

    for epoch in range(1, cfg.KD_MAX_EPOCHS + 1):
        student.train(); regressor.train()
        ep_loss = 0.0; n_batches = 0
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            with torch.no_grad():
                z_t, f_t = teacher(X_b)
            z_s, f_s = student(X_b)
            f_s_proj = regressor(f_s)
            loss = kd_loss(z_s, z_t, f_s_proj, f_t, y_b, T, alpha, beta, pos_weight)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(student.parameters()) + list(regressor.parameters()), 1.0)
            optimizer.step()
            ep_loss += loss.item(); n_batches += 1

        # Validate
        student.eval()
        scores, labels = [], []
        with torch.no_grad():
            for X_b, y_b in val_loader:
                z_s, _ = student(X_b.to(device))
                scores.append(torch.sigmoid(z_s).cpu().numpy())
                labels.append(y_b.cpu().numpy())
        y_score = np.concatenate(scores)
        y_true  = np.concatenate(labels)
        val_auc = auroc(y_true, y_score)
        scheduler.step(val_auc)

        log(f"  KD Epoch {epoch:3d} | loss {ep_loss/n_batches:.4f} | val AUROC {val_auc:.4f}")

        if val_auc > best_val_auc + 1e-5:
            best_val_auc  = val_auc
            best_ckpt_buf = {
                "student"   : {k: v.cpu().clone() for k, v in student.state_dict().items()},
                "regressor" : {k: v.cpu().clone() for k, v in regressor.state_dict().items()},
            }
            patience_cnt = 0
            log(f"    ★ Best val AUROC: {val_auc:.4f}")
        else:
            patience_cnt += 1
            if patience_cnt >= cfg.TEACHER_PATIENCE:
                log(f"  Early stopping at epoch {epoch}")
                break

    # Restore best
    student.load_state_dict(best_ckpt_buf["student"])
    regressor.load_state_dict(best_ckpt_buf["regressor"])
    student.to(device); regressor.to(device)

    # ── Hard-label fine-tuning (lr=1e-4, 10 epochs) ───────────────────────
    log(f"\n  Hard-label fine-tuning (lr={cfg.KD_FINETUNE_LR}, {cfg.KD_FINETUNE_EPOCHS} epochs) …")
    bce_fn   = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    ft_optim = torch.optim.Adam(student.parameters(), lr=cfg.KD_FINETUNE_LR)

    for ep in range(cfg.KD_FINETUNE_EPOCHS):
        student.train()
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            z_s, _ = student(X_b)
            loss = bce_fn(z_s, y_b)
            ft_optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            ft_optim.step()

    # Final val AUROC
    student.eval()
    scores, labels = [], []
    with torch.no_grad():
        for X_b, y_b in val_loader:
            z_s, _ = student(X_b.to(device))
            scores.append(torch.sigmoid(z_s).cpu().numpy())
            labels.append(y_b.cpu().numpy())  # cpu() safe for GPU tensors (GPUBatcher)
    y_score_ft = np.concatenate(scores)
    y_true_ft  = np.concatenate(labels)
    ft_auc = auroc(y_true_ft, y_score_ft)
    log(f"  Post fine-tune val AUROC: {ft_auc:.4f}")

    final_auc = max(best_val_auc, ft_auc)
    return final_auc, y_true_ft, y_score_ft


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher_ckpt", default=str(cfg.CHECKPOINTS_DIR / "teacher_best.pt"))
    parser.add_argument("--gwo_n_wolves", type=int, default=cfg.GWO_N_WOLVES)
    parser.add_argument("--gwo_i_max",   type=int, default=cfg.GWO_I_MAX)
    parser.add_argument("--seeds",       type=int, nargs="+", default=cfg.META_SEEDS)
    parser.add_argument("--gwo_workers", type=int, default=None,
                         help="Concurrent wolf evaluations per GWO iteration "
                              "(default: min(n_wolves, cpu_count))")
    args = parser.parse_args()

    log("=" * 60)
    log("Stage 06 — GWO Knowledge Distillation")
    log("=" * 60)

    device = get_device()
    log(f"  Device: {device}")

    # Wolf-level thread parallelism was tried and reverted (see get_device()
    # note) — GIL contention negated it entirely. Default back to serial
    # evaluation; --gwo_workers is kept for future experimentation only.
    n_workers = args.gwo_workers or 1
    log(f"  GWO parallel workers: {n_workers}")

    # ── Load teacher ──────────────────────────────────────────────────────────
    ckpt_path = Path(args.teacher_ckpt)
    if not ckpt_path.exists():
        log(f"ERROR: Teacher checkpoint not found: {ckpt_path}")
        log("  Run: python src/04_teacher.py first")
        return

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    teacher_name = ckpt["model_name"]
    n_features   = ckpt["n_features"]
    teacher = build_teacher(teacher_name, n_features=n_features).to(device)
    teacher.load_state_dict(ckpt["state_dict"])
    teacher.eval()
    log(f"  Teacher: {teacher_name.upper()} ({count_params(teacher):,} params)")
    log(f"  Teacher val AUROC: {ckpt['val_auroc']:.4f}")

    # ── Class weights ─────────────────────────────────────────────────────────
    _, y_train = load_split_arrays("train")
    n_pos = y_train.sum(); n_neg = len(y_train) - n_pos
    pos_weight = torch.tensor(min(n_neg / n_pos, 10.0), dtype=torch.float32).to(device)

    # ── Shared datasets (loaded once from disk, reused across every GWO
    #    fitness call and the final full distillation — avoids re-reading
    #    the .npy windows on every one of the ~hundreds of evaluations) ──────
    X_train, y_train_arr = load_split_arrays("train")
    X_val,   y_val_arr   = load_split_arrays("val")
    train_dataset = SepsisWindowDataset(X_train, y_train_arr)
    val_dataset   = SepsisWindowDataset(X_val, y_val_arr)
    w_pos = len(y_train_arr) / (2 * n_pos) if n_pos > 0 else 1.0
    w_neg = len(y_train_arr) / (2 * n_neg) if n_neg > 0 else 1.0
    train_weights = torch.from_numpy(
        np.where(y_train_arr == 1, w_pos, w_neg)
    ).float()

    # ── GWO fitness function ──────────────────────────────────────────────────
    log(f"\n[GWO] N_w={args.gwo_n_wolves}  I_max={args.gwo_i_max}  seeds={args.seeds}")
    log(f"  Fitness = val AUROC after {5} epochs KD")
    log(f"  Search space: T∈[1,15]  α∈[0.1,0.9]  β∈[0.01,0.5]  s∈{{1,2,3}}")

    def fitness(params: dict) -> float:
        """GWO fitness: val AUROC after quick student distillation."""
        s_idx    = params["s"]
        student  = build_student(s_idx, n_features=n_features).to(device)
        reg      = FeatureRegressor(student.feature_dim, teacher.feature_dim).to(device)
        auc      = quick_train_student(student, reg, teacher, params, device,
                                       n_epochs=5, pos_weight=pos_weight,
                                       train_dataset=train_dataset, val_dataset=val_dataset,
                                       train_weights=train_weights)
        log(f"    → s={s_idx} T={params['T']:.1f} α={params['alpha_kd']:.2f} "
            f"β={params['beta_kd']:.3f}  val AUROC={auc:.4f}")
        return auc

    best_params, best_score, all_results = run_gwo_multiseed(
        fitness_fn=fitness,
        seeds=args.seeds,
        n_wolves=args.gwo_n_wolves,
        i_max=args.gwo_i_max,
        verbose=True,
        n_workers=n_workers,
    )

    # Full distillation runs single-threaded (one job) — let torch use all cores.
    torch.set_num_threads(os.cpu_count() or 1)

    log(f"\n[GWO] Optimal params: {best_params}  fitness={best_score:.4f}")

    # Save GWO results
    gwo_rows = []
    for seed, params, score, history in all_results:
        for h in history:
            gwo_rows.append({"seed": seed, **h})
    gwo_df = pd.DataFrame(gwo_rows)
    gwo_path = cfg.RESULTS_DIR / "gwo_history.csv"
    gwo_df.to_csv(gwo_path, index=False)
    log(f"  GWO history → {gwo_path}")

    # ── Full distillation with optimal params ─────────────────────────────────
    log("\n[Full KD] Training student with GWO-optimal params …")
    s_idx   = best_params["s"]
    student = build_student(s_idx, n_features=n_features).to(device)
    reg     = FeatureRegressor(student.feature_dim, teacher.feature_dim).to(device)
    log(f"  Student S{s_idx} | params: {count_params(student):,} | feat_dim: {student.feature_dim}")

    val_auc, y_true_val, y_score_val = full_distil(
        student, reg, teacher, best_params, device, pos_weight,
        train_dataset=train_dataset, val_dataset=val_dataset,
        train_weights=train_weights,
    )

    # ── Test evaluation ───────────────────────────────────────────────────────
    log("\n[Test evaluation]")
    test_loader = make_dataloader("test", batch_size=cfg.KD_BATCH_SIZE,
                                  shuffle=False, num_workers=0)
    student.eval()
    scores, labels = [], []
    with torch.no_grad():
        for X_b, y_b in test_loader:
            z_s, _ = student(X_b.to(device))
            scores.append(torch.sigmoid(z_s).cpu().numpy())
            labels.append(y_b.numpy())
    y_true_test  = np.concatenate(labels)
    y_score_test = np.concatenate(scores)

    dual = dual_threshold_eval(y_true_test, y_score_test, cfg.EVAL_SENSITIVITY_TARGET)
    auc_pt, auc_lo, auc_hi = bootstrap_auroc_ci(y_true_test, y_score_test, n_resamples=1000)

    from utils.metrics import knowledge_retention, print_dual_threshold_summary
    kr = knowledge_retention(dual["auroc"], ckpt["val_auroc"])

    log(f"\n── GWO-KD Student (S{s_idx}) Test Results ──")
    log(f"  Test AUROC:  {auc_pt:.4f} [{auc_lo:.4f}–{auc_hi:.4f}]")
    log(f"  AUPRC:       {dual['auprc']:.4f}")
    log(f"  KR:          {kr:.2f}%")
    print_dual_threshold_summary(f"S{s_idx} GWO-KD", dual)

    # Efficiency metrics (on CPU)
    latency = measure_latency_cpu(student.cpu(), n_features=n_features)
    size_mb = model_size_mb(student)
    log(f"\n  CPU latency: {latency:.2f} ms/sample")
    log(f"  Model size:  {size_mb:.4f} MB")
    log(f"  Params:      {count_params(student):,}")

    # ── Save checkpoint ───────────────────────────────────────────────────────
    ckpt_out = cfg.CHECKPOINTS_DIR / f"student_gwo_kd_s{s_idx}.pt"
    torch.save({
        "model_name"     : f"s{s_idx}",
        "n_features"     : n_features,
        "state_dict"     : student.state_dict(),
        "gwo_params"     : best_params,
        "val_auroc"      : val_auc,
        "test_auroc"     : auc_pt,
        "test_auroc_ci"  : [auc_lo, auc_hi],
        "kr"             : kr,
        "latency_ms_cpu" : latency,
        "size_mb"        : size_mb,
        "n_params"       : count_params(student),
        "teacher_name"   : teacher_name,
        "teacher_val_auroc": ckpt["val_auroc"],
    }, ckpt_out)
    log(f"\n  Checkpoint → {ckpt_out}")

    # ── Summary row for results table ─────────────────────────────────────────
    result_row = {
        "model"          : f"GWO-KD (S{s_idx})",
        "teacher"        : teacher_name,
        "n_params"       : count_params(student),
        "size_mb"        : round(size_mb, 4),
        "latency_ms_cpu" : round(latency, 2),
        "test_auroc"     : round(auc_pt, 4),
        "test_auroc_ci"  : f"[{auc_lo:.4f}–{auc_hi:.4f}]",
        "test_auprc"     : round(dual["auprc"], 4),
        "kr_pct"         : round(kr, 2),
        "youden_sens"    : round(dual["youden_sensitivity"], 3),
        "youden_spec"    : round(dual["youden_specificity"], 3),
        "youden_f1"      : round(dual["youden_f1"], 3),
        "clin_sens"      : round(dual["clin_sensitivity"], 3),
        "clin_spec"      : round(dual["clin_specificity"], 3),
        "clin_f1"        : round(dual["clin_f1"], 3),
        "gwo_T"          : best_params["T"],
        "gwo_alpha"      : best_params["alpha_kd"],
        "gwo_beta"       : best_params["beta_kd"],
    }
    pd.DataFrame([result_row]).to_csv(
        cfg.RESULTS_DIR / "gwo_kd_result.csv", index=False)

    log("\n[Stage 06 COMPLETE] ✓")
    log("  → Run next: python src/07_baselines.py")


if __name__ == "__main__":
    main()
