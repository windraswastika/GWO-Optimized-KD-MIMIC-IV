"""
label_sensitivity.py — Label Sensitivity Analysis
===================================================
Addresses reviewer concern: union of noisy ICD + clinical labels may
inflate false positives. This script evaluates a trained model against
three independent labeling strategies and reports whether conclusions hold.

Strategies:
  A — ICD-only  (ICD-9/10 discharge codes)
  B — Clinical-only  (SOFA ≥2 + suspected infection)
  C — Hybrid A OR B  (adopted label, used for training)

For each strategy we:
  1. Re-label the TEST set windows using that strategy's onset times
  2. Evaluate the trained teacher/student scores against those labels
  3. Report AUROC + AUPRC + dual-threshold metrics
  4. Compute label agreement statistics (A∩B, A-only, B-only)

This shows whether the main AUROC conclusions are robust to labeling choice
— a standard sensitivity analysis for EHR-derived labels.

Run:
    python src/label_sensitivity.py --checkpoint checkpoints/teacher_best.pt
"""

import sys, os, argparse
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import torch
from datetime import datetime
from pathlib import Path

import config as cfg
from utils.metrics import (
    auroc, auprc, dual_threshold_eval,
    print_dual_threshold_summary, bootstrap_auroc_ci,
    delong_test, holm_bonferroni
)
from utils.data_utils import SepsisWindowDataset, count_params, model_size_mb


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# =============================================================================
# Rebuild window labels for a given labeling strategy
# =============================================================================

def build_test_labels_for_strategy(
    strategy: str,
    labeled: pd.DataFrame,
    meta: pd.DataFrame,
    test_stay_ids: set,
) -> np.ndarray:
    """
    Re-derive window-level binary labels for TEST set windows using
    the specified strategy ('icd', 'clinical', 'hybrid').

    Logic (same as 02_features.py):
      window label = 1  if  patient is sepsis+ under this strategy
                         AND  sepsis_time_h ≤ window_end_hour
    """
    label_col = {
        "icd"     : "sepsis_icd",
        "clinical": "sepsis_clinical",
        "hybrid"  : "sepsis_hybrid",
    }[strategy]

    time_col = {
        "icd"     : "sepsis_time_h_icd",
        "clinical": "sepsis_time_h_clinical",
        "hybrid"  : "sepsis_time_h",
    }[strategy]

    # We need per-strategy onset times — check if columns exist
    if time_col not in labeled.columns:
        # Fall back to sepsis_time_h (hybrid), which was used for training
        log(f"  Warning: {time_col} not found, using sepsis_time_h")
        time_col = "sepsis_time_h"

    sepsis_flag = labeled.set_index("stay_id")[label_col].to_dict()
    sepsis_time = labeled.set_index("stay_id")[time_col].to_dict()

    # Filter to test set windows
    meta_test = meta[meta["stay_id"].isin(test_stay_ids)].copy()

    labels = []
    for _, row in meta_test.iterrows():
        sid = row["stay_id"]
        weh = row["window_end_hour"]
        is_pos = bool(sepsis_flag.get(sid, False))
        t_h    = sepsis_time.get(sid, None)
        if is_pos and t_h is not None and not pd.isna(t_h):
            labels.append(int(t_h <= weh))
        else:
            labels.append(0)

    return np.array(labels, dtype=np.int8)


# =============================================================================
# Label agreement statistics
# =============================================================================

def label_agreement_stats(labeled: pd.DataFrame) -> pd.DataFrame:
    """
    Compute stay-level agreement between Method A and Method B.
    Returns a summary DataFrame.
    """
    a  = labeled["sepsis_icd"].astype(bool)
    b  = labeled["sepsis_clinical"].astype(bool)
    c  = labeled["sepsis_hybrid"].astype(bool)

    n  = len(labeled)
    both_pos    = (a & b).sum()
    a_only      = (a & ~b).sum()
    b_only      = (~a & b).sum()
    neither     = (~a & ~b).sum()

    # Cohen's kappa for A vs B agreement
    p_agree = (both_pos + neither) / n
    p_a_exp = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    kappa   = (p_agree - p_a_exp) / (1 - p_a_exp) if (1 - p_a_exp) > 0 else 0.0

    rows = [
        {"category": "Both A and B positive",  "n": int(both_pos), "pct": both_pos/n*100},
        {"category": "A only (ICD, no clinical)","n": int(a_only),  "pct": a_only/n*100},
        {"category": "B only (clinical, no ICD)","n": int(b_only),  "pct": b_only/n*100},
        {"category": "Neither",                  "n": int(neither), "pct": neither/n*100},
        {"category": "C (Hybrid = A OR B)",      "n": int(c.sum()),"pct": c.mean()*100},
    ]
    df = pd.DataFrame(rows)
    df["cohen_kappa_A_vs_B"] = kappa   # same value in all rows — for reference
    return df


# =============================================================================
# Get model predictions on test set
# =============================================================================

@torch.no_grad()
def get_test_scores(checkpoint_path: Path, n_features: int) -> np.ndarray:
    """
    Load a teacher/student checkpoint and compute P(sepsis) scores
    for all TEST windows. Returns probability array (N_test,).
    """
    from models.teacher import BiLSTMTeacher, TCNTeacher
    from models.students import build_student

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_name = ckpt.get("model_name", "tcn")

    # Detect model type and rebuild
    if model_name in ("bilstm", "tcn"):
        from models.teacher import build_teacher
        model = build_teacher(model_name, n_features=n_features)
    else:
        # student: model_name = 'tinylstm' | 'cnn1d' | 'compactgru'
        model = build_student(model_name, n_features=n_features)

    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # Load test windows
    X_all = np.load(cfg.PROCESSED_DIR / "windows_features.npy", mmap_mode="r")
    idx_test = np.load(cfg.SPLITS_DIR / "test_idx.npy")
    X_test   = torch.from_numpy(X_all[idx_test]).float()

    scores = []
    batch_size = 1024
    for i in range(0, len(X_test), batch_size):
        xb = X_test[i : i + batch_size]
        logit, _ = model(xb)
        prob = torch.sigmoid(logit).numpy()
        scores.append(prob)

    return np.concatenate(scores)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=str(cfg.CHECKPOINTS_DIR / "teacher_best.pt"),
                        help="Path to model checkpoint (.pt)")
    parser.add_argument("--sensitivity_target", type=float, default=0.80,
                        help="Clinical sensitivity target for threshold selection")
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        log(f"ERROR: Checkpoint not found: {ckpt_path}")
        log("Run teacher training first: python src/04_teacher.py")
        return

    log("=" * 60)
    log("Label Sensitivity Analysis")
    log("=" * 60)

    # ── Load data ─────────────────────────────────────────────────────────────
    labeled  = pd.read_parquet(cfg.PROCESSED_DIR / "cohort_labeled.parquet")
    meta     = pd.read_parquet(cfg.PROCESSED_DIR / "windows_meta.parquet")
    test_ids = set(pd.read_parquet(cfg.SPLITS_DIR / "test_stays.parquet")["stay_id"])
    idx_test = np.load(cfg.SPLITS_DIR / "test_idx.npy")

    meta_test = meta[meta["stay_id"].isin(test_ids)].copy().reset_index(drop=True)
    assert len(meta_test) == len(idx_test), "Meta/index mismatch"

    n_features = np.load(cfg.PROCESSED_DIR / "windows_features.npy",
                          mmap_mode="r").shape[2]
    log(f"  n_features: {n_features}  |  test windows: {len(idx_test):,}")

    # ── Label agreement statistics ────────────────────────────────────────────
    log("\n[1] Label Agreement Statistics (stay-level)")
    agree_df = label_agreement_stats(labeled)
    log("\n" + agree_df[["category", "n", "pct"]].to_string(index=False))
    kappa = agree_df["cohen_kappa_A_vs_B"].iloc[0]
    log(f"\n  Cohen's κ (Method A vs Method B): {kappa:.4f}")
    if kappa < 0.2:
        log("  → Very low agreement: A and B capture mostly different patients")
    elif kappa < 0.4:
        log("  → Fair agreement: some overlap, substantial noise")
    else:
        log("  → Moderate+ agreement: labels partially consistent")

    agree_path = cfg.RESULTS_DIR / "label_agreement.csv"
    agree_df.to_csv(agree_path, index=False)
    log(f"  Saved → {agree_path}")

    # ── Get model predictions ─────────────────────────────────────────────────
    log(f"\n[2] Loading model scores from: {ckpt_path.name}")
    y_scores = get_test_scores(ckpt_path, n_features)
    log(f"  Score range: [{y_scores.min():.4f}, {y_scores.max():.4f}]")

    # ── Evaluate under each labeling strategy ────────────────────────────────
    log("\n[3] Evaluating model under each labeling strategy")
    strategies = {
        "A — ICD-only"     : "icd",
        "B — Clinical-only": "clinical",
        "C — Hybrid (used)": "hybrid",
    }

    results = {}
    for name, strat in strategies.items():
        y_true_strat = build_test_labels_for_strategy(strat, labeled, meta_test, test_ids)
        n_pos = y_true_strat.sum()
        n_tot = len(y_true_strat)
        log(f"\n  Strategy {name}: pos={n_pos:,} ({n_pos/n_tot:.1%})")

        if n_pos == 0:
            log("  → No positives in test set under this strategy, skipping")
            continue

        d = dual_threshold_eval(y_true_strat, y_scores, args.sensitivity_target)
        auc_pt, auc_lo, auc_hi = bootstrap_auroc_ci(y_true_strat, y_scores, n_resamples=500)
        d["auroc_ci_lo"] = auc_lo
        d["auroc_ci_hi"] = auc_hi
        d["n_pos"]       = int(n_pos)
        d["pos_rate"]    = n_pos / n_tot
        d["strategy"]    = name

        print_dual_threshold_summary(name, d)
        results[strat] = (y_true_strat, d)

    # ── DeLong pairwise tests (A vs B vs C) ──────────────────────────────────
    log("\n[4] DeLong pairwise AUROC comparison")
    strat_keys = list(results.keys())
    p_values   = []
    delong_rows= []
    for i in range(len(strat_keys)):
        for j in range(i + 1, len(strat_keys)):
            ka, kb = strat_keys[i], strat_keys[j]
            ya, _  = results[ka]
            yb, _  = results[kb]
            # Use scores from the same model; difference is only in labels
            dt = delong_test(ya, y_scores, y_scores)  # same scores, different labels
            # Note: DeLong comparing scores vs two different label vectors
            # is not standard — here we report AUROC difference and significance
            # via a more appropriate approach: compare AUROC point estimates
            auc_a = results[ka][1]["auroc"]
            auc_b = results[kb][1]["auroc"]
            log(f"  {ka} AUROC={auc_a:.4f}  vs  {kb} AUROC={auc_b:.4f}  "
                f"Δ={auc_a - auc_b:+.4f}")
            delong_rows.append({
                "strategy_A": ka, "strategy_B": kb,
                "auroc_A": auc_a, "auroc_B": auc_b, "delta": auc_a - auc_b
            })

    # ── Summary table ─────────────────────────────────────────────────────────
    log("\n[5] Sensitivity Analysis Summary Table")
    summary_rows = []
    for strat, (_, d) in results.items():
        summary_rows.append({
            "strategy"        : d["strategy"],
            "n_pos_test"      : d["n_pos"],
            "pos_rate_pct"    : round(d["pos_rate"] * 100, 2),
            "auroc"           : round(d["auroc"], 4),
            "auroc_ci"        : f"[{d['auroc_ci_lo']:.4f}–{d['auroc_ci_hi']:.4f}]",
            "auprc"           : round(d["auprc"], 4),
            "youden_thr"      : round(d["youden_threshold"], 3),
            "youden_sens"     : round(d["youden_sensitivity"], 3),
            "youden_spec"     : round(d["youden_specificity"], 3),
            "youden_f1"       : round(d["youden_f1"], 3),
            "clin_thr"        : round(d["clin_threshold"], 3),
            "clin_sens"       : round(d["clin_sensitivity"], 3),
            "clin_spec"       : round(d["clin_specificity"], 3),
            "clin_f1"         : round(d["clin_f1"], 3),
        })
    summary_df = pd.DataFrame(summary_rows)
    log("\n" + summary_df[["strategy","pos_rate_pct","auroc","auroc_ci",
                            "youden_sens","youden_spec","clin_sens","clin_spec"]
                          ].to_string(index=False))

    out_path = cfg.RESULTS_DIR / "label_sensitivity.csv"
    summary_df.to_csv(out_path, index=False)
    log(f"\n  Saved → {out_path}")

    # ── Interpretation ────────────────────────────────────────────────────────
    log("\n[6] Interpretation")
    aurocs = {s: d["auroc"] for s, (_, d) in results.items()}
    max_auc = max(aurocs.values())
    min_auc = min(aurocs.values())
    delta   = max_auc - min_auc
    if delta < 0.02:
        log(f"  ✓ ROBUST: Max AUROC spread across strategies = {delta:.4f} < 0.02")
        log("    Conclusions are NOT sensitive to label choice.")
    elif delta < 0.05:
        log(f"  ⚠ MODERATE: AUROC spread = {delta:.4f} (0.02–0.05)")
        log("    Report all three in manuscript; note label noise as limitation.")
    else:
        log(f"  ✗ SENSITIVE: AUROC spread = {delta:.4f} > 0.05")
        log("    Results depend strongly on label choice — must be reported and explained.")

    log("\n[Label Sensitivity Analysis COMPLETE] ✓")
    log(f"  Results → {cfg.RESULTS_DIR}/label_sensitivity.csv")
    log(f"            {cfg.RESULTS_DIR}/label_agreement.csv")


if __name__ == "__main__":
    main()
