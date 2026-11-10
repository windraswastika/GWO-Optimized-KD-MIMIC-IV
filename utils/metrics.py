"""
utils/metrics.py — Evaluation Metrics (Section 2.14–2.15)
===========================================================
Implements all metrics required by Methodology_Revised.md:
  - AUROC, AUPRC, F1, Precision, Sensitivity (Recall), Specificity
  - Knowledge Retention (KR)
  - Bootstrap 95% CI (1,000 resamples)
  - DeLong AUROC comparison test
  - Wilcoxon signed-rank test (cross-seed / cross-optimiser)
  - Holm–Bonferroni correction
"""

import numpy as np
import pandas as pd
from typing import Sequence
from scipy import stats as scipy_stats
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    f1_score, precision_score, recall_score,
    confusion_matrix, roc_curve
)
import warnings

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config as cfg


# =============================================================================
# Core metric functions
# =============================================================================

def auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Area Under ROC Curve (trapezoidal)."""
    return float(roc_auc_score(y_true, y_score))


def auprc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Area Under Precision-Recall Curve (trapezoidal)."""
    return float(average_precision_score(y_true, y_score))


def compute_metrics_at_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float = 0.5
) -> dict:
    """
    Compute full metric set at a given probability threshold.
    Returns dict: precision, sensitivity, specificity, f1, auroc, auprc
    """
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    precision   = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1          = (2 * precision * sensitivity / (precision + sensitivity)
                   if (precision + sensitivity) > 0 else 0.0)

    return {
        "auroc"      : auroc(y_true, y_score),
        "auprc"      : auprc(y_true, y_score),
        "precision"  : precision,
        "sensitivity": sensitivity,
        "recall"     : sensitivity,        # alias
        "specificity": specificity,
        "f1"         : f1,
        "threshold"  : threshold,
        "tp": int(tp), "fp": int(fp),
        "tn": int(tn), "fn": int(fn),
    }


def knowledge_retention(auroc_student: float, auroc_teacher: float) -> float:
    """
    KR = AUROC_student / AUROC_teacher × 100  (Section 2.14)
    """
    return (auroc_student / auroc_teacher) * 100.0


# =============================================================================
# Threshold selection  (addresses reviewer concern: §2.14 operating point)
# =============================================================================

def find_youden_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> tuple[float, float, float]:
    """
    Youden's Index: threshold that maximises (TPR + TNR - 1) = (Sensitivity + Specificity - 1).
    This is the standard statistical operating point for ROC analysis.
    Returns: (threshold, sensitivity, specificity)
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    tnr = 1.0 - fpr
    youden_j = tpr + tnr - 1.0
    best_idx  = int(np.argmax(youden_j))
    return float(thresholds[best_idx]), float(tpr[best_idx]), float(tnr[best_idx])


def find_sensitivity_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    target_sensitivity: float = 0.80,
) -> tuple[float, float, float]:
    """
    Find the lowest threshold achieving at least `target_sensitivity`.
    Clinically motivated: early warning systems should catch ≥80% of sepsis cases.
    Returns: (threshold, actual_sensitivity, specificity_at_threshold)
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    tnr = 1.0 - fpr
    # Find thresholds where sensitivity ≥ target; pick the one with highest specificity
    eligible = np.where(tpr >= target_sensitivity)[0]
    if len(eligible) == 0:
        # Model cannot reach target — return the closest
        best_idx = int(np.argmax(tpr))
    else:
        # Among eligible, maximise specificity (minimise FPR)
        best_idx = eligible[int(np.argmax(tnr[eligible]))]
    return float(thresholds[best_idx]), float(tpr[best_idx]), float(tnr[best_idx])


def dual_threshold_eval(
    y_true: np.ndarray,
    y_score: np.ndarray,
    sensitivity_target: float = 0.80,
) -> dict:
    """
    Evaluate at BOTH clinically motivated thresholds and return a unified dict.

    Thresholds:
      1. Youden's Index            — maximises statistical information
      2. Sensitivity ≥ target      — clinically motivated (early warning system)

    This dual reporting ensures conclusions are not sensitive to arbitrary
    threshold choice, addressing the reviewer concern in §2.14.
    """
    # Threshold-independent metrics
    auc  = auroc(y_true, y_score)
    apr  = auprc(y_true, y_score)

    # Youden
    thr_y, sens_y, spec_y = find_youden_threshold(y_true, y_score)
    m_y = compute_metrics_at_threshold(y_true, y_score, thr_y)

    # Clinical sensitivity target
    thr_c, sens_c, spec_c = find_sensitivity_threshold(y_true, y_score, sensitivity_target)
    m_c = compute_metrics_at_threshold(y_true, y_score, thr_c)

    return {
        # Threshold-independent
        "auroc": auc,
        "auprc": apr,
        # Youden threshold
        "youden_threshold"   : thr_y,
        "youden_sensitivity" : m_y["sensitivity"],
        "youden_specificity" : m_y["specificity"],
        "youden_precision"   : m_y["precision"],
        "youden_f1"          : m_y["f1"],
        "youden_tp"          : m_y["tp"],
        "youden_fp"          : m_y["fp"],
        "youden_tn"          : m_y["tn"],
        "youden_fn"          : m_y["fn"],
        # Clinical sensitivity threshold
        "clin_target_sensitivity" : sensitivity_target,
        "clin_threshold"     : thr_c,
        "clin_sensitivity"   : m_c["sensitivity"],
        "clin_specificity"   : m_c["specificity"],
        "clin_precision"     : m_c["precision"],
        "clin_f1"            : m_c["f1"],
        "clin_tp"            : m_c["tp"],
        "clin_fp"            : m_c["fp"],
        "clin_tn"            : m_c["tn"],
        "clin_fn"            : m_c["fn"],
    }


def print_dual_threshold_summary(model_name: str, d: dict):
    """Pretty-print dual-threshold evaluation summary."""
    print(f"\n── {model_name} ──")
    print(f"  AUROC : {d['auroc']:.4f}   AUPRC : {d['auprc']:.4f}")
    print(f"\n  [Youden's Index]  threshold={d['youden_threshold']:.3f}")
    print(f"    Sens={d['youden_sensitivity']:.3f}  Spec={d['youden_specificity']:.3f}"
          f"  Prec={d['youden_precision']:.3f}  F1={d['youden_f1']:.3f}")
    print(f"    TP={d['youden_tp']:,}  FP={d['youden_fp']:,}  "
          f"TN={d['youden_tn']:,}  FN={d['youden_fn']:,}")
    print(f"\n  [Clinical target: Sens≥{d['clin_target_sensitivity']:.0%}]"
          f"  threshold={d['clin_threshold']:.3f}")
    print(f"    Sens={d['clin_sensitivity']:.3f}  Spec={d['clin_specificity']:.3f}"
          f"  Prec={d['clin_precision']:.3f}  F1={d['clin_f1']:.3f}")
    print(f"    TP={d['clin_tp']:,}  FP={d['clin_fp']:,}  "
          f"TN={d['clin_tn']:,}  FN={d['clin_fn']:,}")

# =============================================================================
# Bootstrap confidence intervals
# =============================================================================

def bootstrap_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    metric_fn,
    n_resamples: int = cfg.BOOTSTRAP_N_RESAMPLES,
    alpha: float = 0.05,
    random_seed: int = cfg.RANDOM_SEED,
) -> tuple[float, float, float]:
    """
    Bootstrap 95% CI for any metric function.
    Args:
        metric_fn: callable(y_true, y_score) → float
    Returns:
        (point_estimate, ci_lower, ci_upper)
    """
    rng      = np.random.default_rng(random_seed)
    n        = len(y_true)
    boot_vals= []

    for _ in range(n_resamples):
        idx     = rng.integers(0, n, size=n)
        yt_b    = y_true[idx]
        ys_b    = y_score[idx]
        # Skip degenerate samples (only one class)
        if len(np.unique(yt_b)) < 2:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            boot_vals.append(metric_fn(yt_b, ys_b))

    boot_vals = np.array(boot_vals)
    point     = metric_fn(y_true, y_score)
    ci_lo     = np.percentile(boot_vals, 100 * alpha / 2)
    ci_hi     = np.percentile(boot_vals, 100 * (1 - alpha / 2))
    return float(point), float(ci_lo), float(ci_hi)


def bootstrap_auroc_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_resamples: int = cfg.BOOTSTRAP_N_RESAMPLES,
    random_seed: int = cfg.RANDOM_SEED,
) -> tuple[float, float, float]:
    return bootstrap_ci(y_true, y_score, auroc, n_resamples, random_seed=random_seed)


def bootstrap_auprc_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_resamples: int = cfg.BOOTSTRAP_N_RESAMPLES,
    random_seed: int = cfg.RANDOM_SEED,
) -> tuple[float, float, float]:
    return bootstrap_ci(y_true, y_score, auprc, n_resamples, random_seed=random_seed)


# =============================================================================
# DeLong AUROC comparison test  (Section 2.15)
# =============================================================================

def _compute_midranks(x: np.ndarray) -> np.ndarray:
    """Return mid-ranks for DeLong computation."""
    J = np.argsort(x)
    Z = np.zeros(len(x))
    N = len(x)
    i = 0
    while i < N:
        j = i
        while j < N - 1 and x[J[j]] == x[J[j+1]]:
            j += 1
        Z[J[i:j+1]] = 0.5 * (i + j + 2)   # 1-indexed mid-rank
        i = j + 1
    return Z


def delong_test(
    y_true: np.ndarray,
    y_score_A: np.ndarray,
    y_score_B: np.ndarray,
) -> dict:
    """
    DeLong's method for comparing two AUROC values.
    Returns: dict with auroc_A, auroc_B, z_stat, p_value, significant

    Fast O(n log n) implementation via placement / rank statistics,
    based on Sun & Xu (2014) "Fast Implementation of DeLong's Algorithm
    for Comparing the Areas Under Correlated Receiver Operating
    Characteristic Curves" — IEEE Signal Processing Letters 21(11).

    Replaces the earlier O(m × n_) list-comprehension version which
    stalled for n > ~50K samples.

    Reference: DeLong et al. (1988), Biometrics 44(3):837-845.
    """
    pos = y_true == 1
    neg = y_true == 0
    m   = int(pos.sum())
    n_  = int(neg.sum())

    if m == 0 or n_ == 0:
        return {"auroc_A": np.nan, "auroc_B": np.nan,
                "z_stat": np.nan, "p_value": np.nan, "significant": False}

    def _placement_values(scores_pos: np.ndarray, scores_neg: np.ndarray):
        """
        Compute V10 and V01 placement arrays in O(n log n) via searchsorted.

        V10[i] = P(score_neg < scores_pos[i]) + 0.5 * P(score_neg == scores_pos[i])
        V01[j] = P(score_pos > scores_neg[j]) + 0.5 * P(score_pos == scores_neg[j])
        """
        neg_sorted = np.sort(scores_neg)
        pos_sorted = np.sort(scores_pos)

        # V10: for every positive, fraction of negatives it beats (tie-corrected)
        left10  = np.searchsorted(neg_sorted, scores_pos, side="left")
        right10 = np.searchsorted(neg_sorted, scores_pos, side="right")
        V10 = (left10 + 0.5 * (right10 - left10)) / n_

        # V01: for every negative, fraction of positives that beat it (tie-corrected)
        left01  = np.searchsorted(pos_sorted, scores_neg, side="left")
        right01 = np.searchsorted(pos_sorted, scores_neg, side="right")
        V01 = ((m - right01) + 0.5 * (right01 - left01)) / m

        auc = float(np.mean(V10))
        s10 = float(np.var(V10, ddof=1)) / m
        s01 = float(np.var(V01, ddof=1)) / n_
        return auc, s10 + s01, V10, V01

    auc_A, var_A, V10_A, V01_A = _placement_values(y_score_A[pos], y_score_A[neg])
    auc_B, var_B, V10_B, V01_B = _placement_values(y_score_B[pos], y_score_B[neg])

    # Paired (correlated) covariance — Sun & Xu (2014) eq. 9
    covar = (float(np.cov(V10_A, V10_B)[0, 1]) / m +
             float(np.cov(V01_A, V01_B)[0, 1]) / n_)

    se = np.sqrt(max(var_A + var_B - 2 * covar, 0.0))

    if se == 0:
        z = 0.0
    else:
        z = (auc_A - auc_B) / se

    p_value = 2 * (1 - scipy_stats.norm.cdf(abs(z)))

    return {
        "auroc_A"    : float(auc_A),
        "auroc_B"    : float(auc_B),
        "z_stat"     : float(z),
        "p_value"    : float(p_value),
        "significant": bool(p_value < cfg.PVALUE_THRESHOLD),
    }


# =============================================================================
# Wilcoxon signed-rank test  (Section 2.15)
# =============================================================================

def wilcoxon_test(
    values_A: Sequence[float],
    values_B: Sequence[float],
) -> dict:
    """
    Wilcoxon signed-rank test for paired comparisons (e.g., GWO vs PSO
    best-fitness across ≥5 seeds).
    Returns: statistic, p_value, significant
    """
    result = scipy_stats.wilcoxon(values_A, values_B, alternative="two-sided")
    return {
        "statistic"  : float(result.statistic),
        "p_value"    : float(result.pvalue),
        "significant": bool(result.pvalue < cfg.PVALUE_THRESHOLD),
    }


# =============================================================================
# Holm–Bonferroni correction  (Section 2.15)
# =============================================================================

def holm_bonferroni(p_values: Sequence[float]) -> list[float]:
    """
    Holm–Bonferroni correction for multiple comparisons.
    Returns corrected p-values (same order as input).
    """
    n = len(p_values)
    order = np.argsort(p_values)
    p_arr = np.array(p_values)
    corrected = np.empty(n)

    running_max = 0.0
    for rank, idx in enumerate(order):
        adjusted = p_arr[idx] * (n - rank)
        running_max = max(running_max, adjusted)
        corrected[idx] = min(running_max, 1.0)

    return corrected.tolist()


# =============================================================================
# Full evaluation summary (convenience function)
# =============================================================================

def full_eval_summary(
    model_name: str,
    y_true: np.ndarray,
    y_score: np.ndarray,
    teacher_auroc: float | None = None,
    threshold: float = 0.5,
    n_params: int | None = None,
    model_size_mb: float | None = None,
    latency_ms: float | None = None,
) -> dict:
    """
    Compute all metrics for one model and return as dict.
    """
    m = compute_metrics_at_threshold(y_true, y_score, threshold)
    auc_pt, auc_lo, auc_hi = bootstrap_auroc_ci(y_true, y_score)
    apr_pt, apr_lo, apr_hi = bootstrap_auprc_ci(y_true, y_score)

    result = {
        "model"          : model_name,
        "auroc"          : auc_pt,
        "auroc_ci_lo"    : auc_lo,
        "auroc_ci_hi"    : auc_hi,
        "auprc"          : apr_pt,
        "auprc_ci_lo"    : apr_lo,
        "auprc_ci_hi"    : apr_hi,
        "f1"             : m["f1"],
        "precision"      : m["precision"],
        "sensitivity"    : m["sensitivity"],
        "specificity"    : m["specificity"],
    }

    if teacher_auroc is not None:
        result["knowledge_retention_pct"] = knowledge_retention(auc_pt, teacher_auroc)

    if n_params is not None:
        result["n_params"] = n_params
    if model_size_mb is not None:
        result["size_mb"] = model_size_mb
    if latency_ms is not None:
        result["latency_ms"] = latency_ms

    return result


def print_eval_summary(summary: dict):
    """Pretty-print the evaluation summary dict."""
    print(f"\n── {summary['model']} ──")
    print(f"  AUROC   : {summary['auroc']:.4f} "
          f"[{summary.get('auroc_ci_lo', '?'):.4f} – {summary.get('auroc_ci_hi', '?'):.4f}]")
    print(f"  AUPRC   : {summary['auprc']:.4f} "
          f"[{summary.get('auprc_ci_lo', '?'):.4f} – {summary.get('auprc_ci_hi', '?'):.4f}]")
    print(f"  F1      : {summary['f1']:.4f}")
    print(f"  Sens    : {summary['sensitivity']:.4f}   "
          f"Spec: {summary['specificity']:.4f}   Prec: {summary['precision']:.4f}")
    if "knowledge_retention_pct" in summary:
        print(f"  KR      : {summary['knowledge_retention_pct']:.2f}%")
    if "n_params" in summary:
        print(f"  Params  : {summary['n_params']:,}")
    if "size_mb" in summary:
        print(f"  Size    : {summary['size_mb']:.3f} MB")
    if "latency_ms" in summary:
        print(f"  Latency : {summary['latency_ms']:.2f} ms/sample (CPU)")
