"""
05_screening.py — Stage 5: Student Backbone Screening (Section 2.7)
====================================================================
Evaluates all student candidates before GWO to identify the Pareto frontier
of (val AUROC, model size, CPU latency). This informs GWO search — though GWO
also searches over 's' (student index), this screen gives interpretable results.

Students screened:
  S1 — TinyLSTM         (32 hidden)               [candidate, used in GWO-KD]
  S2 — 1D-CNN           (32→16→8 filters)          [candidate, used in GWO-KD]
  S3 — CompactGRU       (32, 16 hidden)            [candidate, used in GWO-KD]
  S4 — 1D-MobileNet     (depthwise-separable)      [screened only, Section 2.7.2]
  S5 — Pruned LightGRU  (48,24 -> 60% pruned)       [screened only, Section 2.7.2]

S4/S5 exist to justify EXCLUDING other prevailing lightweight 1D backbones from
the GWO-KD candidate set (S1-S3): they are trained and screened here under the
identical protocol, but never used in distillation. Each candidate is checked
against edge constraints (size < 1 MB, latency < 10 ms/sample CPU) and against
Pareto-dominance on (AUROC, size, latency); excluded candidates are reported
with the specific criterion they fail.

Trained WITHOUT knowledge distillation (vanilla hard-label training).

Run:
    python src/05_screening.py
"""

import sys, os, argparse, time
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datetime import datetime
from pathlib import Path

import config as cfg
from utils.data_utils import (
    make_dataloader, load_split_arrays,
    compute_class_weights, measure_latency_cpu,
    model_size_mb, count_params
)
from utils.metrics import auroc, auprc, dual_threshold_eval, bootstrap_auroc_ci
from models.students import build_student, STUDENT_REGISTRY


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def count_flops(model: nn.Module, n_features: int, window: int = cfg.WINDOW_SIZE_HOURS) -> int:
    """
    Analytic FLOPs (multiply-adds x2, the standard convention) for one window,
    for the module types used in this project. Self-contained — no thop/
    ptflops dependency — so it works on any machine this pipeline runs on.
    Conv1d assumes 'same' padding (kernel//2, stride=1) as used throughout,
    so output length == window.
    """
    flops = 0
    for m in model.modules():
        if isinstance(m, nn.Linear):
            flops += 2 * m.in_features * m.out_features
        elif isinstance(m, nn.Conv1d):
            macs = m.kernel_size[0] * (m.in_channels // m.groups) * m.out_channels * window
            flops += 2 * macs
        elif isinstance(m, nn.LSTM):
            macs_per_step = 4 * (m.input_size * m.hidden_size + m.hidden_size ** 2)
            flops += 2 * macs_per_step * window
        elif isinstance(m, nn.GRU):
            macs_per_step = 3 * (m.input_size * m.hidden_size + m.hidden_size ** 2)
            flops += 2 * macs_per_step * window
    return flops


def pareto_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add 'screening_verdict': the specific reason a candidate is excluded
    (hard constraint fail, or Pareto-dominated by another compliant candidate
    on AUROC/size/latency simultaneously), or 'retained' if on the frontier.
    Addresses Section 2.7.2: "excluded backbones are reported with the
    specific criterion they fail."
    """
    verdicts = []
    for i, row in df.iterrows():
        if not row["size_ok"]:
            verdicts.append(f"FAIL: size {row['size_mb']:.4f} MB > {cfg.MAX_MODEL_SIZE_MB} MB limit")
            continue
        if not row["latency_ok"]:
            verdicts.append(f"FAIL: latency {row['latency_ms_cpu']:.2f} ms > {cfg.MAX_LATENCY_MS} ms limit")
            continue
        dominated_by = None
        for j, other in df.iterrows():
            if i == j or not (other["size_ok"] and other["latency_ok"]):
                continue
            geq = (other["test_auroc"] >= row["test_auroc"]
                   and other["size_mb"] <= row["size_mb"]
                   and other["latency_ms_cpu"] <= row["latency_ms_cpu"])
            strict = (other["test_auroc"] > row["test_auroc"]
                      or other["size_mb"] < row["size_mb"]
                      or other["latency_ms_cpu"] < row["latency_ms_cpu"])
            if geq and strict:
                dominated_by = other["student"]
                break
        verdicts.append(f"Pareto-dominated by {dominated_by}" if dominated_by
                         else "retained (Pareto-frontier)")
    df = df.copy()
    df["screening_verdict"] = verdicts
    return df


def finetune_after_prune(
    student: nn.Module,
    device: torch.device,
    pos_weight: torch.Tensor,
    epochs: int,
    lr: float = 1e-4,
) -> None:
    """Short hard-label fine-tune to recover accuracy lost to magnitude pruning."""
    train_loader = make_dataloader("train", batch_size=cfg.KD_BATCH_SIZE,
                                   weighted_sampler=True, num_workers=0)
    optimizer = torch.optim.Adam(student.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    student.train()
    for ep in range(epochs):
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            logit, _ = student(X_b)
            loss = criterion(logit, y_b)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()
        log(f"  [S5 fine-tune] epoch {ep+1}/{epochs} done")


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train_student_vanilla(
    student   : nn.Module,
    device    : torch.device,
    pos_weight: torch.Tensor,
    max_epochs: int = 30,
    patience  : int = 5,
    name      : str = "S?",
) -> tuple[float, np.ndarray, np.ndarray]:
    """Train student with BCE loss only (no KD). Returns (best_val_auroc, y_true, y_score)."""
    train_loader = make_dataloader("train", batch_size=cfg.KD_BATCH_SIZE,
                                   weighted_sampler=True, num_workers=0)
    val_loader   = make_dataloader("val", batch_size=cfg.KD_BATCH_SIZE,
                                   shuffle=False, num_workers=0)

    optimizer = torch.optim.Adam(student.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3, min_lr=1e-6)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_val_auc = -1.0
    best_state   = None
    patience_cnt = 0
    y_true_best  = None
    y_score_best = None

    for epoch in range(1, max_epochs + 1):
        student.train()
        ep_loss = 0.0; n_b = 0
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            logit, _ = student(X_b)
            loss = criterion(logit, y_b)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()
            ep_loss += loss.item(); n_b += 1

        student.eval()
        scores, labels = [], []
        with torch.no_grad():
            for X_b, y_b in val_loader:
                logit, _ = student(X_b.to(device))
                scores.append(torch.sigmoid(logit).cpu().numpy())
                labels.append(y_b.numpy())
        y_score = np.concatenate(scores)
        y_true  = np.concatenate(labels)
        val_auc = auroc(y_true, y_score)
        scheduler.step(val_auc)

        log(f"  [{name}] Epoch {epoch:2d} | loss {ep_loss/n_b:.4f} | val AUROC {val_auc:.4f}")

        if val_auc > best_val_auc + 1e-5:
            best_val_auc = val_auc
            best_state   = {k: v.cpu().clone() for k, v in student.state_dict().items()}
            y_true_best  = y_true
            y_score_best = y_score
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                log(f"  [{name}] Early stop at epoch {epoch}")
                break

    student.load_state_dict(best_state)
    return best_val_auc, y_true_best, y_score_best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_epochs", type=int, default=30)
    parser.add_argument("--students", type=int, nargs="+", default=[1, 2, 3, 4, 5],
                         choices=[1, 2, 3, 4, 5],
                         help="Which student indices to (re-)run, e.g. --students 5 "
                              "to redo only S5 after a code fix without retraining "
                              "S1-S4. NOTE: this OVERWRITES screening_summary.csv "
                              "with only the selected students' rows unless you "
                              "merge with a previous run's CSV yourself.")
    args = parser.parse_args()

    log("=" * 60)
    log("Stage 05 — Student Backbone Screening")
    log("=" * 60)

    device = get_device()
    log(f"  Device: {device}")

    X_sample   = np.load(cfg.PROCESSED_DIR / "windows_features.npy", mmap_mode="r")
    n_features = X_sample.shape[2]
    log(f"  n_features: {n_features}")

    _, y_train = load_split_arrays("train")
    n_pos = y_train.sum(); n_neg = len(y_train) - n_pos
    pos_weight = torch.tensor(min(n_neg / n_pos, 10.0), dtype=torch.float32).to(device)

    log(f"  pos_weight (capped): {pos_weight.item():.2f}")

    test_loader = make_dataloader("test", batch_size=cfg.KD_BATCH_SIZE,
                                  shuffle=False, num_workers=0)

    STUDENT_NAMES = {
        1: "TinyLSTM", 2: "1D-CNN", 3: "CompactGRU",
        4: "MobileNet1D", 5: "PrunedLightGRU",
    }

    rows = []
    for s_idx in args.students:
        name = STUDENT_NAMES[s_idx]
        log(f"\n{'─'*50}")
        log(f"[S{s_idx}] {name}")

        student = build_student(s_idx, n_features=n_features).to(device)
        n_par_dense = count_params(student)
        flops = count_flops(student, n_features=n_features)
        log(f"  Params (dense): {n_par_dense:,}  FLOPs/window: {flops:,}")

        # Train (dense)
        torch.manual_seed(cfg.RANDOM_SEED)
        val_auc, y_true_val, y_score_val = train_student_vanilla(
            student, device, pos_weight,
            max_epochs=args.max_epochs, patience=5, name=f"S{s_idx}"
        )

        # S5 only: magnitude-prune the trained dense model, then fine-tune
        # to recover accuracy (standard prune-then-fine-tune recipe).
        if s_idx == 5:
            log(f"  [S5] Pruning {cfg.LIGHTGRU_PRUNE_SPARSITY:.0%} of weights (global L1 unstructured) …")
            student.prune(cfg.LIGHTGRU_PRUNE_SPARSITY)
            finetune_after_prune(student, device, pos_weight,
                                  epochs=cfg.LIGHTGRU_FINETUNE_EPOCHS)
            # Re-evaluate val AUROC post prune+fine-tune (may differ from
            # the pre-prune value captured above).
            val_loader = make_dataloader("val", batch_size=cfg.KD_BATCH_SIZE,
                                         shuffle=False, num_workers=0)
            student.eval()
            vs, vl = [], []
            with torch.no_grad():
                for X_b, y_b in val_loader:
                    logit, _ = student(X_b.to(device))
                    vs.append(torch.sigmoid(logit).cpu().numpy())
                    vl.append(y_b.numpy())
            val_auc = auroc(np.concatenate(vl), np.concatenate(vs))
            log(f"  [S5] Post-prune+fine-tune val AUROC: {val_auc:.4f}")

            # Bake the pruning mask in now that no more optimizer steps will
            # touch these weights (must happen AFTER fine-tuning, else the
            # optimizer silently undoes the pruning — see students.py).
            student.finalize_prune()

        # Effective (non-zero) params — equals dense count except for S5
        n_par = student.effective_params() if hasattr(student, "effective_params") else n_par_dense
        sz_mb = (n_par / n_par_dense) * model_size_mb(student) if s_idx == 5 else model_size_mb(student)
        if s_idx == 5:
            log(f"  [S5] Effective (non-zero) params: {n_par:,} / {n_par_dense:,} dense "
                f"({n_par/n_par_dense:.1%}) — 'size_mb' below assumes sparse storage")

        # Edge constraint check (post-training/pruning, so S5 reflects its true size)
        if sz_mb > cfg.MAX_MODEL_SIZE_MB:
            log(f"  ✗ Size {sz_mb:.3f} MB exceeds limit {cfg.MAX_MODEL_SIZE_MB} MB")

        # Test evaluation
        student.eval()
        scores, labels = [], []
        with torch.no_grad():
            for X_b, y_b in test_loader:
                logit, _ = student(X_b.to(device))
                scores.append(torch.sigmoid(logit).cpu().numpy())
                labels.append(y_b.numpy())
        y_true_test  = np.concatenate(labels)
        y_score_test = np.concatenate(scores)

        dual = dual_threshold_eval(y_true_test, y_score_test, cfg.EVAL_SENSITIVITY_TARGET)
        auc_pt, auc_lo, auc_hi = bootstrap_auroc_ci(y_true_test, y_score_test, n_resamples=500)

        # CPU latency
        lat = measure_latency_cpu(student.cpu(), n_features=n_features)
        lat_ok = lat < cfg.MAX_LATENCY_MS

        log(f"\n  [S{s_idx} {name}] Test Results:")
        log(f"    AUROC  : {auc_pt:.4f} [{auc_lo:.4f}–{auc_hi:.4f}]")
        log(f"    AUPRC  : {dual['auprc']:.4f}")
        log(f"    Latency: {lat:.2f} ms  {'✓' if lat_ok else '✗ EXCEEDS LIMIT'}")

        # Save checkpoint
        ckpt_path = cfg.CHECKPOINTS_DIR / f"student_s{s_idx}_vanilla.pt"
        torch.save({
            "model_name" : f"s{s_idx}",
            "n_features" : n_features,
            "state_dict" : student.cpu().state_dict(),
            "val_auroc"  : val_auc,
            "test_auroc" : auc_pt,
        }, ckpt_path)

        rows.append({
            "student"         : f"S{s_idx} {name}",
            "n_params"        : n_par,
            "n_params_dense"  : n_par_dense,
            "flops_per_window": flops,
            "size_mb"         : round(sz_mb, 4),
            "latency_ms_cpu"  : round(lat, 2),
            "size_ok"         : sz_mb <= cfg.MAX_MODEL_SIZE_MB,
            "latency_ok"      : lat_ok,
            "val_auroc"       : round(val_auc, 4),
            "test_auroc"      : round(auc_pt, 4),
            "test_auroc_ci"   : f"[{auc_lo:.4f}–{auc_hi:.4f}]",
            "test_auprc"      : round(dual["auprc"], 4),
            "youden_sens"     : round(dual["youden_sensitivity"], 3),
            "youden_spec"     : round(dual["youden_specificity"], 3),
            "youden_f1"       : round(dual["youden_f1"], 3),
            "clin_sens"       : round(dual["clin_sensitivity"], 3),
            "clin_f1"         : round(dual["clin_f1"], 3),
        })

    # Summary — merge with any previous run's rows for students NOT rerun this
    # time (so `--students 5` doesn't discard already-valid S1-S4 rows).
    df = pd.DataFrame(rows)
    out_path = cfg.RESULTS_DIR / "screening_summary.csv"
    if out_path.exists() and set(args.students) != {1, 2, 3, 4, 5}:
        prev_df = pd.read_csv(out_path)
        rerun_prefixes = tuple(f"S{s}" for s in args.students)
        prev_df = prev_df[~prev_df["student"].str.startswith(rerun_prefixes)]
        df = pd.concat([prev_df, df], ignore_index=True)
        df["_order"] = df["student"].str.extract(r"S(\d+)").astype(int)
        df = df.sort_values("_order").drop(columns="_order").reset_index(drop=True)
        log(f"  Merged with previous {out_path} (kept rows for students not in "
            f"--students {list(args.students)})")
    df = pareto_analysis(df)
    log("\n" + "="*60)
    log("Screening Summary")
    log("="*60)
    log("\n" + df[["student","n_params","flops_per_window","size_mb","latency_ms_cpu",
                    "test_auroc","youden_sens","screening_verdict"]].to_string(index=False))

    retained = df[df["screening_verdict"] == "retained (Pareto-frontier)"]
    log(f"\n  Retained (Pareto-frontier, edge-compliant): "
        f"{', '.join(retained['student'].tolist()) if len(retained) else 'NONE — check constraints'}")
    for _, row in df[df["screening_verdict"] != "retained (Pareto-frontier)"].iterrows():
        log(f"  Excluded: {row['student']} — {row['screening_verdict']}")

    df.to_csv(out_path, index=False)
    log(f"\n  Saved → {out_path}")
    log("\n[Stage 05 COMPLETE] ✓")
    log("  → Run next: python src/06_gwo_kd.py")


if __name__ == "__main__":
    main()
