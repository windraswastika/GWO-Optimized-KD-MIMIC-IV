"""
03_split.py — Stage 3: Patient-Level Train/Val/Test Split
==========================================================
Section 2.5: Split at the PATIENT level (no patient appears in
more than one split): 70% train / 15% val / 15% test,
stratified by hybrid sepsis label.

Outputs:
  splits/train_stays.parquet
  splits/val_stays.parquet
  splits/test_stays.parquet
  splits/train_idx.npy, val_idx.npy, test_idx.npy   (window indices)
  results/split_summary.csv

Run: python src/03_split.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split

import config as cfg


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    log("=" * 60)
    log("Stage 03 — Patient-Level Train/Val/Test Split")
    log("=" * 60)

    # Load labeled cohort (one row per stay)
    labeled = pd.read_parquet(cfg.PROCESSED_DIR / "cohort_labeled.parquet")
    meta    = pd.read_parquet(cfg.PROCESSED_DIR / "windows_meta.parquet")

    log(f"  Total stays   : {len(labeled):,}")
    log(f"  Total windows : {len(meta):,}")

    # Stratify on patient-level sepsis label (not window-level)
    # One label per stay (worst-case: any window positive → stay positive for stratification)
    stay_label = labeled[["stay_id", "sepsis_hybrid"]].drop_duplicates("stay_id").copy()
    stay_label["label"] = stay_label["sepsis_hybrid"].astype(int)

    n_pos = stay_label["label"].sum()
    n_neg = len(stay_label) - n_pos
    log(f"  Stay-level: {n_pos:,} sepsis / {n_neg:,} non-sepsis")

    # ── Step 1: train vs temp (val + test) ──────────────────────────────────
    train_stays, temp_stays = train_test_split(
        stay_label["stay_id"].values,
        test_size=(1 - cfg.SPLIT_TRAIN),
        stratify=stay_label["label"].values,
        random_state=cfg.RANDOM_SEED
    )

    temp_label = stay_label[stay_label["stay_id"].isin(temp_stays)]
    # ── Step 2: val vs test (50/50 of temp → each 15% of total) ─────────────
    val_stays, test_stays = train_test_split(
        temp_label["stay_id"].values,
        test_size=0.5,
        stratify=temp_label["label"].values,
        random_state=cfg.RANDOM_SEED
    )

    train_stays = set(train_stays)
    val_stays   = set(val_stays)
    test_stays  = set(test_stays)

    log(f"\n  Train stays: {len(train_stays):,} "
        f"(~{len(train_stays)/len(stay_label):.0%})")
    log(f"  Val   stays: {len(val_stays):,} "
        f"(~{len(val_stays)/len(stay_label):.0%})")
    log(f"  Test  stays: {len(test_stays):,} "
        f"(~{len(test_stays)/len(stay_label):.0%})")

    # Verify no overlap
    assert len(train_stays & val_stays) == 0,  "Train/Val overlap!"
    assert len(train_stays & test_stays) == 0, "Train/Test overlap!"
    assert len(val_stays   & test_stays) == 0, "Val/Test overlap!"
    log("  No patient-level overlap ✓")

    # ── Save stay ID files ───────────────────────────────────────────────────
    for name, ids in [("train", train_stays), ("val", val_stays), ("test", test_stays)]:
        df = labeled[labeled["stay_id"].isin(ids)].copy()
        out = cfg.SPLITS_DIR / f"{name}_stays.parquet"
        df.to_parquet(out, index=False)
        log(f"  Saved {name}_stays.parquet: {len(df):,} stays")

    # ── Map window meta to split indices ─────────────────────────────────────
    meta_indexed = meta.reset_index(drop=True)
    idx_train = meta_indexed.index[meta_indexed["stay_id"].isin(train_stays)].to_numpy()
    idx_val   = meta_indexed.index[meta_indexed["stay_id"].isin(val_stays)].to_numpy()
    idx_test  = meta_indexed.index[meta_indexed["stay_id"].isin(test_stays)].to_numpy()

    np.save(cfg.SPLITS_DIR / "train_idx.npy", idx_train)
    np.save(cfg.SPLITS_DIR / "val_idx.npy",   idx_val)
    np.save(cfg.SPLITS_DIR / "test_idx.npy",  idx_test)

    # Load labels for balance check
    y_all = np.load(cfg.PROCESSED_DIR / "windows_labels.npy")

    for name, idx in [("train", idx_train), ("val", idx_val), ("test", idx_test)]:
        y_split = y_all[idx]
        log(f"  {name:5s}: {len(idx):,} windows | "
            f"pos={y_split.sum():,} ({y_split.mean():.2%}) | "
            f"neg={(y_split==0).sum():,}")

    # ── Summary CSV ─────────────────────────────────────────────────────────
    summary_rows = []
    for name, stays, idx in [
        ("train", train_stays, idx_train),
        ("val",   val_stays,   idx_val),
        ("test",  test_stays,  idx_test),
    ]:
        y_s = y_all[idx]
        summary_rows.append({
            "split"       : name,
            "n_stays"     : len(stays),
            "n_windows"   : len(idx),
            "n_pos"       : int(y_s.sum()),
            "n_neg"       : int((y_s == 0).sum()),
            "pos_rate_pct": round(y_s.mean() * 100, 2),
        })
    summary = pd.DataFrame(summary_rows)
    summary_path = cfg.RESULTS_DIR / "split_summary.csv"
    summary.to_csv(summary_path, index=False)
    log(f"\n  Saved split summary → {summary_path}")
    log("\n" + summary.to_string(index=False))

    log("\n[Stage 03 COMPLETE] ✓")
    log("  → Run next: python src/04_teacher.py")


if __name__ == "__main__":
    main()
