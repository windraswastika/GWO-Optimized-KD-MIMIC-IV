"""
02_features.py — Stage 2: Feature Engineering & 12-Hour Sliding Windows
=========================================================================
Implements Section 2.5 of Methodology_Revised.md:
  1. Forward-fill imputation (vitals ≤2 h, labs ≤24 h)
  2. Missingness indicators (binary, 1 if imputed/missing after ffill)
  3. Outlier clipping at 1st/99th percentile (fit on TRAIN only)
  4. Z-score normalisation (fit on TRAIN only to prevent leakage)
  5. 12-hour sliding window with 1-hour step → tensor (N_windows × 12 × 25)
  6. Window labels: 1 if hybrid sepsis present in or before window end

Outputs:
  processed/windows_features.npy   shape (N, 12, 25)
  processed/windows_labels.npy     shape (N,)
  processed/windows_meta.parquet   stay_id, window_end_hour per row

Run: python src/02_features.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import logging
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

import config as cfg


def _setup_logger(name: str, log_path: Path) -> logging.Logger:
    """Configure a logger that writes to both stdout and a timestamped log file."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    if logger.handlers:          # avoid duplicate handlers on re-import
        logger.handlers.clear()

    fmt = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%H:%M:%S")

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler — append so re-runs accumulate
    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


_RUN_TS   = datetime.now().strftime("%Y%m%d_%H%M%S")
_LOG_PATH = cfg.LOGS_DIR / f"stage02_{_RUN_TS}.log"
_logger   = _setup_logger("stage02", _LOG_PATH)


def log(msg):
    _logger.info(msg)


# ── Column names for the 25 features (in canonical order) ──────────────────
FEATURE_COLS = [
    # 7 vitals
    "heart_rate", "sbp", "dbp", "map", "resp_rate", "spo2", "temperature",
    # 8 labs
    "wbc", "creatinine", "lactate", "bilirubin", "platelets", "inr", "pao2", "fio2",
    # 4 derived
    "gcs_total", "pf_ratio", "map_delta", "rr_trend",
    # 2 binary medication flags
    "vasopressor_flag", "antibiotic_flag",
]
# NOTE: binary flags (vasopressor_flag, antibiotic_flag) are NOT normalised,
# missingness indicators are appended after normalisation.

VITAL_COLS    = ["heart_rate", "sbp", "dbp", "map", "resp_rate", "spo2", "temperature"]
LAB_COLS      = ["wbc", "creatinine", "lactate", "bilirubin", "platelets", "inr", "pao2", "fio2"]
DERIVED_COLS  = ["gcs_total", "pf_ratio", "map_delta", "rr_trend"]
BINARY_COLS   = ["vasopressor_flag", "antibiotic_flag"]
NUMERIC_COLS  = VITAL_COLS + LAB_COLS + DERIVED_COLS   # cols to normalise


# =============================================================================
# Step 1 — Forward-fill (per stay) with time limits
# =============================================================================

def forward_fill_per_stay(grid: pd.DataFrame) -> pd.DataFrame:
    """
    Forward-fill within each stay:
      - Vitals: limit 2 hours (i.e., 2 steps because step = 1 h)
      - Labs  : limit 24 hours
    """
    log("  Forward-filling vitals (≤2 h) and labs (≤24 h) …")
    grid = grid.sort_values(["stay_id", "hour_offset"]).copy()

    # Forward fill vitals (limit=2)
    grid[VITAL_COLS] = (
        grid.groupby("stay_id")[VITAL_COLS]
        .transform(lambda s: s.ffill(limit=cfg.FFILL_VITALS_HOURS))
    )

    # Forward fill labs (limit=24)
    grid[LAB_COLS] = (
        grid.groupby("stay_id")[LAB_COLS]
        .transform(lambda s: s.ffill(limit=cfg.FFILL_LABS_HOURS))
    )

    # Derived columns: same limits as their source
    grid[DERIVED_COLS] = (
        grid.groupby("stay_id")[DERIVED_COLS]
        .transform(lambda s: s.ffill(limit=cfg.FFILL_LABS_HOURS))
    )

    return grid


# =============================================================================
# Step 2 — Missingness indicators (BEFORE median imputation)
# =============================================================================

def create_missingness_indicators(grid: pd.DataFrame) -> pd.DataFrame:
    """
    Append binary columns miss_<feature>=1 for every numeric feature that is
    still NaN after forward-fill. Appended BEFORE median imputation so that
    the indicator reflects true missing status.
    """
    log("  Creating missingness indicators …")
    miss_cols = {}
    for col in NUMERIC_COLS:
        if col in grid.columns:
            miss_cols[f"miss_{col}"] = grid[col].isna().astype(np.int8)
    miss_df = pd.DataFrame(miss_cols, index=grid.index)
    return pd.concat([grid, miss_df], axis=1)


# =============================================================================
# Step 3 — Compute training-set statistics (clip + zscore) on TRAIN split
# =============================================================================

def compute_train_stats(grid_train: pd.DataFrame) -> dict:
    """
    Compute per-feature: p1, p99 (for clipping), mean, std (for z-score).
    Fit ONLY on training rows to prevent leakage (Section 2.5).
    """
    log("  Computing training-set statistics (clip + zscore) …")
    stats = {}
    for col in NUMERIC_COLS:
        if col not in grid_train.columns:
            continue
        vals = grid_train[col].dropna()
        p1   = np.percentile(vals, cfg.OUTLIER_CLIP_PERCENTILE_LOW)
        p99  = np.percentile(vals, cfg.OUTLIER_CLIP_PERCENTILE_HIGH)
        # Clip first, then compute mean/std for normalisation
        clipped = vals.clip(lower=p1, upper=p99)
        mu  = clipped.mean()
        sig = clipped.std()
        if sig == 0 or np.isnan(sig):
            sig = 1.0   # constant feature – avoid division by zero
        stats[col] = {"p1": p1, "p99": p99, "mean": mu, "std": sig}
    log(f"  Stats computed for {len(stats)} features")
    return stats


def apply_clip_zscore(grid: pd.DataFrame, stats: dict) -> pd.DataFrame:
    """
    Apply clip and z-score normalisation using pre-computed training stats.
    Binary flags are NOT normalised.
    """
    grid = grid.copy()
    for col, s in stats.items():
        if col not in grid.columns:
            continue
        grid[col] = grid[col].clip(lower=s["p1"], upper=s["p99"])
        # Impute residual NaN with training-set median (post-clip mean ≈ median for Gaussian)
        grid[col] = grid[col].fillna(s["mean"])
        grid[col] = (grid[col] - s["mean"]) / s["std"]
    return grid


# =============================================================================
# Step 4 — Build sliding windows
# =============================================================================

def build_windows(grid: pd.DataFrame, 
                  labeled_cohort: pd.DataFrame,
                  feature_cols: list[str]) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Slide a 12-hour window with 1-hour step over each stay.
    Window label = 1 if sepsis_time_h ≤ window_end_hour (label persists once positive).

    Returns:
        X_arr   : float32 array  (N_windows, 12, len(feature_cols))
        y_arr   : int8 array     (N_windows,)
        meta_df : DataFrame      stay_id, window_end_hour, window_idx
    """
    log("  Building 12-hour sliding windows …")

    sepsis_time_map = labeled_cohort.set_index("stay_id")["sepsis_time_h"].to_dict()
    sepsis_label_map = labeled_cohort.set_index("stay_id")["sepsis_hybrid"].to_dict()

    X_list, y_list, meta_rows = [], [], []

    for stay_id, stay_df in tqdm(grid.groupby("stay_id"), desc="Windows"):
        stay_df = stay_df.sort_values("hour_offset")
        hours   = stay_df["hour_offset"].values
        vals    = stay_df[feature_cols].values.astype(np.float32)

        sepsis_h = sepsis_time_map.get(stay_id)    # None if no sepsis
        is_sepsis = bool(sepsis_label_map.get(stay_id, False))

        max_end  = len(hours)   # number of rows (= number of hours indexed)
        win      = cfg.WINDOW_SIZE_HOURS
        step     = cfg.WINDOW_STEP_HOURS

        for i in range(0, max_end - win + 1, step):
            window_slice = vals[i : i + win]
            if window_slice.shape[0] < win:
                break   # incomplete window
            window_end_h = hours[i + win - 1]

            # Label: sepsis if patient is septic AND sepsis_time ≤ window_end
            if is_sepsis and sepsis_h is not None:
                label = int(sepsis_h <= window_end_h)
            else:
                label = 0

            X_list.append(window_slice)
            y_list.append(label)
            meta_rows.append({
                "stay_id"       : stay_id,
                "window_end_hour": int(window_end_h),
                "window_start_i" : i,
            })

    X_arr   = np.stack(X_list,  axis=0).astype(np.float32)   # (N, 12, F)
    y_arr   = np.array(y_list,  dtype=np.int8)                 # (N,)
    meta_df = pd.DataFrame(meta_rows)

    log(f"  Windows built: {X_arr.shape}  positives: {y_arr.sum()} ({y_arr.mean():.2%})")
    return X_arr, y_arr, meta_df


# =============================================================================
# Main
# =============================================================================

def main():
    log("=" * 60)
    log("Stage 02 — Feature Engineering & Sliding Windows")
    log("=" * 60)
    log(f"Log file → {_LOG_PATH}")

    # Load inputs
    log("Loading hourly_features.parquet …")
    grid = pd.read_parquet(cfg.PROCESSED_DIR / "hourly_features.parquet")

    log("Loading cohort_labeled.parquet …")
    labeled = pd.read_parquet(cfg.PROCESSED_DIR / "cohort_labeled.parquet")

    # Load train/val/test stay IDs if split already exists, else use all stays
    train_split_path = cfg.SPLITS_DIR / "train_stays.parquet"
    if train_split_path.exists():
        train_stays = set(pd.read_parquet(train_split_path)["stay_id"])
        log(f"  Using pre-computed train stays ({len(train_stays):,}) for stat fitting")
    else:
        log("  No split file yet — will compute stats on all stays (run 03_split.py after)")
        train_stays = set(grid["stay_id"].unique())

    # Ensure feature cols exist in grid
    available_feats = [c for c in FEATURE_COLS if c in grid.columns]
    missing_feats   = [c for c in FEATURE_COLS if c not in grid.columns]
    log(f"  Available features: {len(available_feats)}/25  |  missing: {missing_feats}")

    # ── Step 1: Forward fill ────────────────────────────────────────────────
    log("\n[Step 1] Forward filling …")
    grid = forward_fill_per_stay(grid)

    # ── Step 2: Missingness indicators ──────────────────────────────────────
    log("\n[Step 2] Missingness indicators …")
    grid = create_missingness_indicators(grid)
    miss_cols = [c for c in grid.columns if c.startswith("miss_")]
    log(f"  Missingness indicator columns: {len(miss_cols)}")

    # ── Step 3: Training-set stats → clip + z-score ─────────────────────────
    log("\n[Step 3] Clip + Z-score normalisation …")
    grid_train = grid[grid["stay_id"].isin(train_stays)]
    train_stats = compute_train_stats(grid_train)
    grid        = apply_clip_zscore(grid, train_stats)

    # Save stats for later use (needed in eICU preprocessing)
    import json
    stats_path = cfg.PROCESSED_DIR / "train_stats.json"
    with open(stats_path, "w") as f:
        json.dump(train_stats, f, indent=2)
    log(f"  Saved training stats → {stats_path}")

    # Also save as parquet (used by eicu_extract.py for normalization)
    norm_df = pd.DataFrame(train_stats).T   # features as rows
    norm_df.to_parquet(cfg.PROCESSED_DIR / "norm_stats.parquet")
    log(f"  Saved norm_stats.parquet ({len(norm_df)} features)")


    # ── Full feature list: 25 base + missingness indicators ─────────────────
    # The 25 canonical features are used for the window tensor
    window_feat_cols = [c for c in available_feats]
    # Note: missingness indicators are stored separately (not included in the
    # 12×25 tensor per methodology; they are appended as extra channel if needed)

    # ── Step 4: Sliding windows ─────────────────────────────────────────────
    log("\n[Step 4] Building sliding windows …")
    X_arr, y_arr, meta_df = build_windows(grid, labeled, window_feat_cols)

    # ── Save ─────────────────────────────────────────────────────────────────
    X_path    = cfg.PROCESSED_DIR / "windows_features.npy"
    y_path    = cfg.PROCESSED_DIR / "windows_labels.npy"
    meta_path = cfg.PROCESSED_DIR / "windows_meta.parquet"

    np.save(X_path,  X_arr)
    np.save(y_path,  y_arr)
    meta_df.to_parquet(meta_path, index=False)

    log(f"\n  Saved X → {X_path}      {X_arr.shape}")
    log(f"  Saved y → {y_path}      {y_arr.shape}")
    log(f"  Saved meta → {meta_path}  {meta_df.shape}")

    # Class balance
    log(f"\n  Class balance:")
    log(f"    Positive (sepsis): {y_arr.sum():,} ({y_arr.mean():.2%})")
    log(f"    Negative (no sep): {(y_arr==0).sum():,} ({1-y_arr.mean():.2%})")

    log("\n[Stage 02 COMPLETE] ✓")
    log("  → Run next: python src/03_split.py")


if __name__ == "__main__":
    main()
