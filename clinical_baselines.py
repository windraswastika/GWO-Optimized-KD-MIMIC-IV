"""
clinical_baselines.py — SOFA / MEWS / NEWS2 rule-based baselines (closes R2-1/R3-8/R3 gap)
=============================================================================================
Computes the three standard clinical early-warning/organ-dysfunction scores as rule-based
predictors and evaluates them on the SAME test split used for every other model in this study
(splits/test_idx.npy over processed/windows_features.npy / windows_labels.npy), so the
comparison in Table 7/8 of the manuscript is apples-to-apples.

No training, no GPU: this is pure rule-based scoring from already-extracted features, so it
runs entirely locally (or anywhere the processed/ and splits/ directories are available) in
under a minute.

Score definition — computed from the LAST timestep of each 12-hour window (i.e. the most
recent hour available at prediction time), using standard published thresholds:

  SOFA (6 domains, 0-24): respiratory (PaO2/FiO2), coagulation (platelets),
    hepatic (bilirubin), cardiovascular (MAP + vasopressor flag as a proxy for pressor
    dose), CNS (GCS), renal (creatinine — urine output unavailable in this feature set,
    so the renal domain relies on creatinine only, a known simplification).
  MEWS (0-14): SBP, HR, RR, temperature, consciousness (GCS used as an AVPU proxy).
  NEWS2 (0-20): RR, SpO2 (scale 1), supplemental O2 (FiO2 > 21% used as a proxy for
    "on oxygen"), SBP, HR, consciousness (GCS==15 -> Alert, else -> CVPU), temperature.

Because the underlying continuous features are Z-score normalised in windows_features.npy,
values are first denormalised using processed/train_stats.json (the same statistics fit on
the training split for every other model in this study — no leakage introduced here).

Run:
    python src/clinical_baselines.py
"""

import sys, os, json
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
from datetime import datetime

import config as cfg
from utils.metrics import dual_threshold_eval, bootstrap_auroc_ci

FEATURE_COLS = [
    "heart_rate", "sbp", "dbp", "map", "resp_rate", "spo2", "temperature",
    "wbc", "creatinine", "lactate", "bilirubin", "platelets", "inr", "pao2", "fio2",
    "gcs_total", "pf_ratio", "map_delta", "rr_trend",
    "vasopressor_flag", "antibiotic_flag",
]
IDX = {name: i for i, name in enumerate(FEATURE_COLS)}
NORMALISED_COLS = FEATURE_COLS[:19]  # all but the two binary flags


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def denormalise(last_step: np.ndarray, train_stats: dict) -> dict:
    """last_step: (N, 21) array, the final timestep of each window. Returns a dict of
    raw-scale (N,) arrays keyed by feature name."""
    raw = {}
    for i, name in enumerate(FEATURE_COLS):
        col = last_step[:, i]
        if name in train_stats:
            stats = train_stats[name]
            col = col * stats["std"] + stats["mean"]
        raw[name] = col
    return raw


# =============================================================================
# SOFA (6 domains)
# =============================================================================

def sofa_respiratory(pf_ratio):
    s = np.zeros_like(pf_ratio)
    s[pf_ratio < 400] = 1
    s[pf_ratio < 300] = 2
    s[pf_ratio < 200] = 3
    s[pf_ratio < 100] = 4
    return s

def sofa_coagulation(platelets):
    s = np.zeros_like(platelets)
    s[platelets < 150] = 1
    s[platelets < 100] = 2
    s[platelets < 50]  = 3
    s[platelets < 20]  = 4
    return s

def sofa_hepatic(bilirubin):
    s = np.zeros_like(bilirubin)
    s[bilirubin >= 1.2] = 1
    s[bilirubin >= 2.0] = 2
    s[bilirubin >= 6.0] = 3
    s[bilirubin >= 12.0] = 4
    return s

def sofa_cardiovascular(map_, vaso_flag):
    s = np.zeros_like(map_)
    s[map_ < 70] = 1
    # Vasopressor use (no dose available): treat as a proxy for a higher-severity band.
    s[vaso_flag > 0.5] = 2
    return s

def sofa_cns(gcs):
    s = np.zeros_like(gcs)
    s[gcs < 15] = 1
    s[gcs < 13] = 2
    s[gcs < 10] = 3
    s[gcs < 6]  = 4
    return s

def sofa_renal(creatinine):
    s = np.zeros_like(creatinine)
    s[creatinine >= 1.2] = 1
    s[creatinine >= 2.0] = 2
    s[creatinine >= 3.5] = 3
    s[creatinine >= 5.0] = 4
    return s

def compute_sofa(raw: dict) -> np.ndarray:
    return (
        sofa_respiratory(raw["pf_ratio"])
        + sofa_coagulation(raw["platelets"])
        + sofa_hepatic(raw["bilirubin"])
        + sofa_cardiovascular(raw["map"], raw["vasopressor_flag"])
        + sofa_cns(raw["gcs_total"])
        + sofa_renal(raw["creatinine"])
    )


# =============================================================================
# MEWS
# =============================================================================

def mews_sbp(sbp):
    s = np.zeros_like(sbp)
    s[sbp <= 70] = 3
    s[(sbp > 70) & (sbp <= 80)] = 2
    s[(sbp > 80) & (sbp <= 100)] = 1
    s[sbp >= 200] = 2
    return s

def mews_hr(hr):
    s = np.zeros_like(hr)
    s[hr <= 40] = 2
    s[(hr > 40) & (hr <= 50)] = 1
    s[(hr > 100) & (hr <= 110)] = 1
    s[(hr > 110) & (hr <= 129)] = 2
    s[hr >= 130] = 3
    return s

def mews_rr(rr):
    s = np.zeros_like(rr)
    s[rr < 9] = 2
    s[(rr >= 15) & (rr <= 20)] = 1
    s[(rr > 20) & (rr <= 29)] = 2
    s[rr >= 30] = 3
    return s

def mews_temp(temp):
    s = np.zeros_like(temp)
    s[temp < 35] = 2
    s[temp >= 38.5] = 2
    return s

def mews_avpu(gcs):
    """AVPU proxy from GCS (Alert / Voice / Pain / Unresponsive)."""
    s = np.zeros_like(gcs)
    s[(gcs >= 9) & (gcs <= 14)] = 1
    s[(gcs >= 4) & (gcs <= 8)]  = 2
    s[gcs < 4] = 3
    return s

def compute_mews(raw: dict) -> np.ndarray:
    return (
        mews_sbp(raw["sbp"]) + mews_hr(raw["heart_rate"]) + mews_rr(raw["resp_rate"])
        + mews_temp(raw["temperature"]) + mews_avpu(raw["gcs_total"])
    )


# =============================================================================
# NEWS2
# =============================================================================

def news2_rr(rr):
    s = np.zeros_like(rr)
    s[rr <= 8] = 3
    s[(rr >= 9) & (rr <= 11)] = 1
    s[(rr >= 21) & (rr <= 24)] = 2
    s[rr >= 25] = 3
    return s

def news2_spo2(spo2):
    s = np.zeros_like(spo2)
    s[(spo2 >= 94) & (spo2 <= 95)] = 1
    s[(spo2 >= 92) & (spo2 <= 93)] = 2
    s[spo2 <= 91] = 3
    return s

def news2_o2(fio2):
    """FiO2 > 21% (room air) used as a proxy for supplemental oxygen."""
    return (fio2 > 0.22).astype(np.float32) * 2

def news2_sbp(sbp):
    s = np.zeros_like(sbp)
    s[sbp <= 90] = 3
    s[(sbp > 90) & (sbp <= 100)] = 2
    s[(sbp > 100) & (sbp <= 110)] = 1
    s[sbp >= 220] = 3
    return s

def news2_hr(hr):
    s = np.zeros_like(hr)
    s[hr <= 40] = 3
    s[(hr > 40) & (hr <= 50)] = 1
    s[(hr > 90) & (hr <= 110)] = 1
    s[(hr > 110) & (hr <= 130)] = 2
    s[hr > 130] = 3
    return s

def news2_consciousness(gcs):
    return (gcs < 15).astype(np.float32) * 3

def news2_temp(temp):
    s = np.zeros_like(temp)
    s[temp <= 35.0] = 3
    s[(temp > 35.0) & (temp <= 36.0)] = 1
    s[(temp > 38.0) & (temp <= 39.0)] = 1
    s[temp > 39.0] = 2
    return s

def compute_news2(raw: dict) -> np.ndarray:
    fio2_frac = raw["fio2"] / 100.0 if np.nanmedian(raw["fio2"]) > 1.5 else raw["fio2"]
    return (
        news2_rr(raw["resp_rate"]) + news2_spo2(raw["spo2"]) + news2_o2(fio2_frac)
        + news2_sbp(raw["sbp"]) + news2_hr(raw["heart_rate"])
        + news2_consciousness(raw["gcs_total"]) + news2_temp(raw["temperature"])
    )


# =============================================================================
# Main
# =============================================================================

def main():
    log("=" * 60)
    log("Clinical rule-based baselines: SOFA / MEWS / NEWS2")
    log("=" * 60)

    train_stats = json.load(open(cfg.PROCESSED_DIR / "train_stats.json"))

    log("  Loading test split (same split_idx as every other model in this study) …")
    X_all  = np.load(cfg.PROCESSED_DIR / "windows_features.npy", mmap_mode="r")
    y_all  = np.load(cfg.PROCESSED_DIR / "windows_labels.npy")
    test_idx = np.load(cfg.SPLITS_DIR / "test_idx.npy")

    y_true = y_all[test_idx]
    last_step = np.array(X_all[test_idx, -1, :])  # (N, 21), most recent hour per window
    log(f"  Test windows: {len(test_idx):,}  positives: {y_true.sum():,} ({y_true.mean():.1%})")

    raw = denormalise(last_step, train_stats)

    scores = {
        "SOFA":  compute_sofa(raw),
        "MEWS":  compute_mews(raw),
        "NEWS2": compute_news2(raw),
    }

    rows = []
    for name, score in scores.items():
        dual = dual_threshold_eval(y_true, score, cfg.EVAL_SENSITIVITY_TARGET)
        auc_pt, auc_lo, auc_hi = bootstrap_auroc_ci(y_true, score, n_resamples=1000)
        log(f"  [{name}] range=[{score.min():.0f},{score.max():.0f}]  "
            f"AUROC={auc_pt:.4f} [{auc_lo:.4f}-{auc_hi:.4f}]  AUPRC={dual['auprc']:.4f}")
        rows.append({
            "model": name, "type": "Clinical score", "n_params": 0, "size_mb": 0.0,
            "latency_ms": 0.0,  # rule-based, negligible latency; not a learned model
            "test_auroc_ci": f"{auc_pt:.4f} [{auc_lo:.4f}–{auc_hi:.4f}]",
            "test_auprc": round(dual["auprc"], 4),
            "youden_sens": round(dual["youden_sensitivity"], 3),
            "youden_spec": round(dual["youden_specificity"], 3),
            "youden_f1": round(dual["youden_f1"], 3),
            "clin_sens": round(dual["clin_sensitivity"], 3),
        })

    df = pd.DataFrame(rows)
    out_path = cfg.RESULTS_DIR / "clinical_baselines.csv"
    df.to_csv(out_path, index=False)
    log(f"\n  Saved -> {out_path}")
    log("\n" + df.to_string(index=False))
    log("\n[Clinical baselines COMPLETE]")


if __name__ == "__main__":
    main()
