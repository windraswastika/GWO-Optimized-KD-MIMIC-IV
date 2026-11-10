"""
04_teacher.py — Stage 4: Teacher Model Training & Selection (Section 2.6)
=========================================================================
Trains two teacher candidates under identical conditions:
  Candidate A — BiLSTM + temporal self-attention
  Candidate B — TCN with dilated residual blocks

Usage:
    python src/04_teacher.py --model bilstm   # train only BiLSTM
    python src/04_teacher.py --model tcn      # train only TCN
    python src/04_teacher.py --model both     # train both (default)

Outputs (per model):
    checkpoints/teacher_<name>_best.pt        — best checkpoint
    checkpoints/teacher_<name>_history.csv    — per-epoch metrics
    results/teacher_selection.csv             — quantitative comparison
"""

import sys, os, argparse, time
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pathlib import Path
from datetime import datetime

import config as cfg
from utils.data_utils import (
    load_split_arrays, SepsisWindowDataset,
    make_dataloader, compute_class_weights,
    measure_latency_cpu, model_size_mb, count_params
)
from utils.metrics import auroc, auprc, bootstrap_auroc_ci, delong_test, full_eval_summary
from models.teacher import BiLSTMTeacher, TCNTeacher, build_teacher


def log(msg: str):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# =============================================================================
# Training loop
# =============================================================================

def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Run one training epoch, return mean loss."""
    model.train()
    total_loss = 0.0
    n_batches  = 0
    for X_b, y_b in loader:
        X_b, y_b = X_b.to(device), y_b.to(device)
        optimizer.zero_grad()
        logit, _ = model(X_b)
        loss = criterion(logit, y_b)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
        n_batches  += 1
    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    device: torch.device,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """Evaluate model, return (auroc, auprc, y_true, y_score)."""
    model.eval()
    all_scores, all_labels = [], []
    for X_b, y_b in loader:
        X_b = X_b.to(device)
        logit, _ = model(X_b)
        prob = torch.sigmoid(logit).cpu().numpy()
        all_scores.append(prob)
        all_labels.append(y_b.numpy())
    y_score = np.concatenate(all_scores)
    y_true  = np.concatenate(all_labels)
    auc = auroc(y_true, y_score)
    apr = auprc(y_true, y_score)
    return auc, apr, y_true, y_score


def train_teacher(
    model_name: str,
    n_features: int,
    device: torch.device,
    seed: int = cfg.RANDOM_SEED,
) -> dict:
    """
    Train one teacher candidate. Returns dict with metrics.
    """
    log(f"\n{'='*60}")
    log(f"Training teacher: {model_name.upper()}")
    log(f"{'='*60}")

    torch.manual_seed(seed)
    np.random.seed(seed)

    # ── Data ─────────────────────────────────────────────────────────────────
    log("Loading data …")
    # Use WeightedRandomSampler: ensures ~50% positive per batch
    # Much more effective than pos_weight alone for 3.5% imbalance
    train_loader = make_dataloader("train", batch_size=cfg.TEACHER_BATCH_SIZE,
                                   weighted_sampler=True, num_workers=0)
    val_loader   = make_dataloader("val",   batch_size=cfg.TEACHER_BATCH_SIZE,
                                   shuffle=False, num_workers=0)

    _, y_train = load_split_arrays("train")
    # With weighted sampler we don't need heavy pos_weight — use mild weighting
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    pos_weight = torch.tensor(min(n_neg / n_pos, 10.0), dtype=torch.float32).to(device)
    log(f"  pos_weight (capped at 10) = {pos_weight.item():.2f}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_teacher(model_name, n_features=n_features).to(device)
    n_par = count_params(model)
    log(f"  Parameters: {n_par:,}")

    # ── Optimiser & scheduler ─────────────────────────────────────────────────
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.TEACHER_LR,
        betas=cfg.TEACHER_ADAM_BETAS
    )
    scheduler = ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5,
        patience=5, min_lr=1e-6
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # LR warmup: ramp from lr/10 to lr over first 2 epochs
    warmup_epochs = 2
    def get_lr_factor(epoch):
        if epoch <= warmup_epochs:
            return epoch / warmup_epochs
        return 1.0
    warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=get_lr_factor)

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_auroc = -1.0
    patience_cnt   = 0
    history        = []
    best_ckpt      = cfg.CHECKPOINTS_DIR / f"teacher_{model_name}_best.pt"

    for epoch in range(1, cfg.TEACHER_MAX_EPOCHS + 1):
        t0 = time.time()

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_auc, val_apr, _, _ = evaluate(model, val_loader, device)

        # LR warmup for first epochs, then ReduceLROnPlateau
        if epoch <= warmup_epochs:
            warmup_scheduler.step()
        else:
            scheduler.step(val_auc)

        elapsed = time.time() - t0
        lr_now  = optimizer.param_groups[0]["lr"]

        history.append({
            "epoch": epoch, "train_loss": train_loss,
            "val_auroc": val_auc, "val_auprc": val_apr,
            "lr": lr_now, "elapsed_s": elapsed,
        })

        # Log every epoch for visibility
        log(f"  Epoch {epoch:3d} | loss {train_loss:.4f} | "
            f"val AUROC {val_auc:.4f} | AUPRC {val_apr:.4f} | "
            f"lr {lr_now:.2e} | {elapsed:.0f}s")

        # Early stopping on val AUROC
        if val_auc > best_val_auroc + 1e-5:
            best_val_auroc = val_auc
            patience_cnt   = 0
            torch.save({
                "epoch"         : epoch,
                "model_name"    : model_name,
                "n_features"    : n_features,
                "state_dict"    : model.state_dict(),
                "val_auroc"     : val_auc,
                "val_auprc"     : val_apr,
                "optimizer"     : optimizer.state_dict(),
            }, best_ckpt)
            log(f"  ★ New best val AUROC: {val_auc:.4f} (epoch {epoch}) → saved")
        else:
            patience_cnt += 1
            if patience_cnt >= cfg.TEACHER_PATIENCE:
                log(f"  Early stopping at epoch {epoch} (patience {cfg.TEACHER_PATIENCE})")
                break

    # ── Save history ──────────────────────────────────────────────────────────
    hist_path = cfg.CHECKPOINTS_DIR / f"teacher_{model_name}_history.csv"
    pd.DataFrame(history).to_csv(hist_path, index=False)
    log(f"  History saved → {hist_path}")

    # ── Final test evaluation ─────────────────────────────────────────────────
    log("\nLoading best checkpoint for test evaluation …")
    ckpt = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(ckpt["state_dict"])

    test_loader = make_dataloader("test", batch_size=cfg.TEACHER_BATCH_SIZE,
                                  shuffle=False, num_workers=0)
    test_auc, test_apr, y_true_test, y_score_test = evaluate(model, test_loader, device)
    auc_pt, auc_lo, auc_hi = bootstrap_auroc_ci(y_true_test, y_score_test, n_resamples=500)

    log(f"\n── Test Results: {model_name.upper()} ──")
    log(f"  Test AUROC : {auc_pt:.4f} [{auc_lo:.4f}–{auc_hi:.4f}]")
    log(f"  Test AUPRC : {test_apr:.4f}")
    log(f"  Val/Test gap: {best_val_auroc - auc_pt:+.4f}")
    log(f"  Params     : {n_par:,}")
    log(f"  Size       : {model_size_mb(model):.3f} MB")

    # CPU latency
    model_cpu = model.cpu()
    latency   = measure_latency_cpu(model_cpu, n_features=n_features)
    log(f"  CPU latency: {latency:.2f} ms/sample")

    return {
        "model_name"        : model_name,
        "n_params"          : n_par,
        "size_mb"           : model_size_mb(model.cpu()),
        "best_val_auroc"    : best_val_auroc,
        "test_auroc"        : auc_pt,
        "test_auroc_ci_lo"  : auc_lo,
        "test_auroc_ci_hi"  : auc_hi,
        "test_auprc"        : test_apr,
        "val_test_gap"      : best_val_auroc - auc_pt,
        "latency_ms_cpu"    : latency,
        "best_epoch"        : ckpt["epoch"],
        "checkpoint"        : str(best_ckpt),
        "y_true_test"       : y_true_test,
        "y_score_test"      : y_score_test,
    }


# =============================================================================
# Teacher selection analysis (Section 2.6)
# =============================================================================

def select_teacher(results: list[dict]) -> str:
    """
    Select the better teacher based on:
      1. Test AUROC (primary)
      2. Parameter economy (secondary — prefer fewer params if AUC gap < 0.005)
    Returns model_name of the selected teacher.
    """
    log("\n" + "="*60)
    log("Teacher Selection Analysis (Section 2.6)")
    log("="*60)

    df = pd.DataFrame([{k: v for k, v in r.items()
                         if k not in ("y_true_test", "y_score_test")}
                        for r in results])
    df = df.sort_values("test_auroc", ascending=False)
    log("\n" + df[["model_name","n_params","test_auroc","test_auroc_ci_lo",
                    "test_auroc_ci_hi","test_auprc","val_test_gap","latency_ms_cpu"]
                  ].to_string(index=False))

    # DeLong test if both models trained
    if len(results) == 2:
        r0, r1 = results[0], results[1]
        dt = delong_test(r0["y_true_test"], r0["y_score_test"], r1["y_score_test"])
        log(f"\nDeLong test {r0['model_name']} vs {r1['model_name']}: "
            f"z={dt['z_stat']:.3f}, p={dt['p_value']:.4f}, "
            f"significant={'YES' if dt['significant'] else 'NO'}")

    # Selection rule
    best = df.iloc[0]
    second = df.iloc[1] if len(df) > 1 else None
    if second is not None:
        auc_gap = best["test_auroc"] - second["test_auroc"]
        param_ratio = best["n_params"] / second["n_params"]
        if auc_gap < 0.005 and param_ratio > 1.5:
            # Gap is negligible but best is much larger — pick second (more economical)
            selected = second["model_name"]
            log(f"\n  AUC gap {auc_gap:.4f} < 0.005 but param ratio {param_ratio:.1f}x "
                f"→ select more economical: {selected.upper()}")
        else:
            selected = best["model_name"]
            log(f"\n  Selected: {selected.upper()} (test AUROC {best['test_auroc']:.4f})")
    else:
        selected = best["model_name"]

    # Save selection table
    sel_path = cfg.RESULTS_DIR / "teacher_selection.csv"
    df.to_csv(sel_path, index=False)
    log(f"\n  Selection table → {sel_path}")

    # Copy best to canonical checkpoint
    import shutil
    src_ckpt = cfg.CHECKPOINTS_DIR / f"teacher_{selected}_best.pt"
    dst_ckpt = cfg.CHECKPOINTS_DIR / "teacher_best.pt"
    shutil.copy(src_ckpt, dst_ckpt)
    log(f"  Canonical checkpoint → {dst_ckpt}")

    return selected


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="both",
                        choices=["bilstm", "tcn", "both"],
                        help="Which teacher to train")
    args = parser.parse_args()

    log("Stage 04 — Teacher Model Training")
    device = get_device()
    log(f"  Device: {device}")

    # Detect actual number of features from saved windows
    X_sample = np.load(cfg.PROCESSED_DIR / "windows_features.npy", mmap_mode="r")
    n_features = X_sample.shape[2]
    log(f"  n_features detected: {n_features}")

    models_to_train = ["bilstm", "tcn"] if args.model == "both" else [args.model]
    results = []

    for name in models_to_train:
        res = train_teacher(name, n_features=n_features, device=device)
        results.append(res)

    if len(results) > 1:
        selected = select_teacher(results)
        log(f"\n[Stage 04 COMPLETE] Selected teacher: {selected.upper()}")
    else:
        log(f"\n[Stage 04 COMPLETE] Trained: {results[0]['model_name'].upper()}")

    log("  → Run next: python src/05_screening.py")


if __name__ == "__main__":
    main()
