"""
01_labeling.py — Stage 1: Sepsis-3 Hybrid Labeling
====================================================
Implements the three-method labeling scheme (Section 2.4):
  Method A — ICD-9/10 discharge codes
  Method B — Clinical Sepsis-3 criteria (SOFA ≥2 + suspected infection)
  Method C — Hybrid: A OR B  (adopted label)

Outputs:
  processed/cohort_labeled.parquet
    Columns: stay_id, subject_id, hadm_id, intime, outtime,
             sepsis_icd, sepsis_clinical, sepsis_hybrid, sepsis_time
  results/label_prevalence.csv — Table 2 content

Run:
    python src/01_labeling.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import duckdb
from tqdm import tqdm

import config as cfg


def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# =============================================================================
# Method A — ICD-based labeling
# =============================================================================

def label_icd(cohort: pd.DataFrame) -> pd.Series:
    """
    Return boolean Series (indexed by stay_id) for ICD-based sepsis.
    Matches ICD-9: 99591, 99592, 78552
    Matches ICD-10 prefixes: A40, A41, R6520, R6521
    """
    log("Method A — Loading diagnoses_icd.csv …")
    diag = pd.read_csv(
        cfg.MIMIC_FILES["diagnoses_icd"],
        usecols=["hadm_id", "icd_code", "icd_version"],
        dtype={"icd_code": str, "icd_version": int}
    )
    log(f"  Diagnoses loaded: {len(diag):,} rows")

    # ICD-9
    icd9_set = set(cfg.SEPSIS_ICD9)
    diag["icd_code_clean"] = diag["icd_code"].str.strip().str.replace(".", "", regex=False)

    mask9  = (diag["icd_version"] == 9) & diag["icd_code_clean"].isin(icd9_set)
    mask10 = (diag["icd_version"] == 10) & diag["icd_code_clean"].str[:4].isin(
        set(cfg.SEPSIS_ICD10_PREFIXES)
    )
    sepsis_hadms = set(diag.loc[mask9 | mask10, "hadm_id"])
    log(f"  ICD-sepsis hadm_ids: {len(sepsis_hadms):,}")

    cohort_icd = cohort.copy()
    cohort_icd["sepsis_icd"] = cohort_icd["hadm_id"].isin(sepsis_hadms)
    return cohort_icd.set_index("stay_id")["sepsis_icd"]


# =============================================================================
# Method B — Clinical Sepsis-3 criteria
# =============================================================================

def compute_sofa_scores(cohort: pd.DataFrame, grid: pd.DataFrame) -> pd.DataFrame:
    """
    Compute hourly SOFA scores per patient using the 6 components:
      1. Respiratory   : PaO2/FiO2 ratio (from pf_ratio column)
      2. Coagulation   : Platelets (from platelets column)
      3. Hepatic       : Bilirubin (from bilirubin column)
      4. Cardiovascular: MAP < 70 mmHg OR vasopressor in use
      5. Neurological  : GCS total (from gcs_total column)
      6. Renal         : Creatinine (from creatinine column)

    Returns DataFrame with stay_id, hour_offset, sofa_total
    """
    log("Computing SOFA scores …")
    df = grid[["stay_id", "hour_offset"]].copy()

    # 1. Respiratory — PaO2/FiO2 (P/F ratio)
    if "pf_ratio" in grid.columns:
        pf = grid["pf_ratio"]
        df["sofa_resp"] = np.select(
            [pf >= 400, pf >= 300, pf >= 200, pf >= 100, pf < 100],
            [0,          1,         2,          3,          4],
            default=np.nan
        )
    else:
        df["sofa_resp"] = np.nan

    # 2. Coagulation — Platelets (×10³/μL)
    if "platelets" in grid.columns:
        plt = grid["platelets"]
        df["sofa_coag"] = np.select(
            [plt >= 150, plt >= 100, plt >= 50, plt >= 20, plt < 20],
            [0,           1,          2,          3,          4],
            default=np.nan
        )
    else:
        df["sofa_coag"] = np.nan

    # 3. Hepatic — Bilirubin (mg/dL)
    if "bilirubin" in grid.columns:
        bili = grid["bilirubin"]
        df["sofa_hepatic"] = np.select(
            [bili < 1.2, bili < 2.0, bili < 6.0, bili < 12.0, bili >= 12.0],
            [0,           1,          2,            3,           4],
            default=np.nan
        )
    else:
        df["sofa_hepatic"] = np.nan

    # 4. Cardiovascular — MAP & vasopressors
    if "map" in grid.columns and "vasopressor_flag" in grid.columns:
        map_val  = grid["map"]
        vaso_flg = grid["vasopressor_flag"]
        df["sofa_cardio"] = np.where(
            vaso_flg == 1, 3,   # vasopressor → score ≥3 (simplified)
            np.where(map_val >= 70, 0, 1)   # MAP < 70 → score 1
        )
    else:
        df["sofa_cardio"] = np.nan

    # 5. Neurological — GCS total (range 3–15)
    if "gcs_total" in grid.columns:
        gcs = grid["gcs_total"]
        df["sofa_neuro"] = np.select(
            [gcs >= 15, gcs >= 13, gcs >= 10, gcs >= 6, gcs < 6],
            [0,          1,         2,          3,          4],
            default=np.nan
        )
    else:
        df["sofa_neuro"] = np.nan

    # 6. Renal — Creatinine (mg/dL)
    if "creatinine" in grid.columns:
        cr = grid["creatinine"]
        df["sofa_renal"] = np.select(
            [cr < 1.2, cr < 2.0, cr < 3.5, cr < 5.0, cr >= 5.0],
            [0,         1,         2,          3,          4],
            default=np.nan
        )
    else:
        df["sofa_renal"] = np.nan

    sofa_cols = ["sofa_resp", "sofa_coag", "sofa_hepatic", 
                 "sofa_cardio", "sofa_neuro", "sofa_renal"]
    df["sofa_total"] = df[sofa_cols].sum(axis=1, min_count=1)

    log(f"  SOFA computed: {df['sofa_total'].describe().to_dict()}")
    return df[["stay_id", "hour_offset", "sofa_total"] + sofa_cols]


def compute_sofa_increase(sofa_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute SOFA increase ≥2 from the 24-hour baseline (worst of first 24h).
    Returns DataFrame: stay_id, hour_offset, sofa_increase, organ_dysfunction
    """
    log("Computing SOFA baseline and delta …")

    sofa_df = sofa_df.sort_values(["stay_id", "hour_offset"])

    # Baseline SOFA: worst (max) score in the first 24 hours
    baseline = (
        sofa_df[sofa_df["hour_offset"] < 24]
        .groupby("stay_id")["sofa_total"]
        .max()
        .rename("sofa_baseline")
    )
    sofa_df = sofa_df.merge(baseline, on="stay_id", how="left")
    sofa_df["sofa_increase"]    = sofa_df["sofa_total"] - sofa_df["sofa_baseline"]
    sofa_df["organ_dysfunction"] = sofa_df["sofa_increase"] >= 2

    return sofa_df


def get_suspected_infection_times(cohort: pd.DataFrame) -> pd.DataFrame:
    """
    Suspected infection = antibiotic order + blood culture within a 72-h window.
    Returns DataFrame: stay_id, infection_start_h (earliest suspected hour)
    """
    log("Computing suspected infection times …")

    # Blood cultures from microbiologyevents
    log("  Loading microbiologyevents.csv …")
    micro = pd.read_csv(
        cfg.MIMIC_FILES["microbiologyevents"],
        usecols=["hadm_id", "charttime", "spec_type_desc"],
        parse_dates=["charttime"]
    )
    # Blood culture specimens
    blood_cx = micro[micro["spec_type_desc"].str.lower().str.contains("blood", na=False)].copy()

    # Antibiotic events already extracted → load from parquet
    abx_path = cfg.PROCESSED_DIR / "antibiotic_events.parquet"
    if abx_path.exists():
        abx_df = pd.read_parquet(abx_path)
    else:
        raise FileNotFoundError("Run 00_extract.py first to generate antibiotic_events.parquet")

    # Map hadm_id → stay_id
    hadm_stay = cohort[["hadm_id", "stay_id", "intime"]].drop_duplicates("hadm_id")
    blood_cx  = blood_cx.merge(hadm_stay, on="hadm_id", how="inner")
    # Drop rows with NaT charttime or intime
    blood_cx = blood_cx.dropna(subset=["charttime", "intime"])
    blood_cx["hour_offset"] = (
        (blood_cx["charttime"] - blood_cx["intime"]).dt.total_seconds() / 3600
    ).apply(np.floor)
    blood_cx = blood_cx.dropna(subset=["hour_offset"])
    blood_cx["hour_offset"] = blood_cx["hour_offset"].astype(int)

    # Drop rows with NaT starttime or intime in abx_df
    abx_df = abx_df.dropna(subset=["starttime", "intime"])
    abx_df["abx_start_h"] = (
        (abx_df["starttime"] - abx_df["intime"]).dt.total_seconds() / 3600
    ).apply(np.floor)
    abx_df = abx_df.dropna(subset=["abx_start_h"])
    abx_df["abx_start_h"] = abx_df["abx_start_h"].astype(int)

    # For each stay: find first antibiotic-blood-culture pair within 72 h window
    infection_times = []
    blood_by_stay = blood_cx.groupby("stay_id")["hour_offset"].apply(list).to_dict()
    abx_by_stay   = abx_df.groupby("stay_id")["abx_start_h"].apply(list).to_dict()

    for stay_id in tqdm(cohort["stay_id"], desc="Infection detection"):
        bx_hours  = blood_by_stay.get(stay_id, [])
        abx_hours = abx_by_stay.get(stay_id, [])
        if not bx_hours or not abx_hours:
            continue
        # Check all pairs for 72-h window
        earliest = None
        for bh in bx_hours:
            for ah in abx_hours:
                if abs(bh - ah) <= 72:
                    t = min(bh, ah)
                    if earliest is None or t < earliest:
                        earliest = t
        if earliest is not None:
            infection_times.append({"stay_id": stay_id, "infection_hour": earliest})

    result = pd.DataFrame(infection_times)
    log(f"  Suspected infection found in {len(result):,} stays")
    return result


def label_clinical(cohort: pd.DataFrame, grid: pd.DataFrame) -> pd.Series:
    """
    Method B: clinical sepsis = suspected infection AND organ dysfunction (SOFA ≥2 increase)
    within the stay. Returns boolean Series indexed by stay_id.
    """
    # SOFA
    sofa_df   = compute_sofa_scores(cohort, grid)
    sofa_df   = compute_sofa_increase(sofa_df)

    # Suspected infection
    infection = get_suspected_infection_times(cohort)

    # Combine: sepsis if organ dysfunction occurs within [-48h, +24h] of infection onset
    # (Sepsis-3 definition window)
    log("Combining SOFA + infection for clinical label …")
    organ_onset = (
        sofa_df[sofa_df["organ_dysfunction"]]
        .groupby("stay_id")["hour_offset"]
        .min()
        .rename("organ_hour")
        .reset_index()
    )

    combined = infection.merge(organ_onset, on="stay_id", how="inner")
    combined["time_diff"] = combined["organ_hour"] - combined["infection_hour"]
    # Sepsis-3 window: organ dysfunction within [-48, +24] h of infection
    sepsis_clinical_stays = set(
        combined[
            (combined["time_diff"] >= -48) & (combined["time_diff"] <= 24)
        ]["stay_id"]
    )

    sepsis_time_map = {}
    for _, row in combined.iterrows():
        if row["stay_id"] in sepsis_clinical_stays:
            sepsis_time_map[row["stay_id"]] = min(row["infection_hour"], row["organ_hour"])

    log(f"  Clinical sepsis stays: {len(sepsis_clinical_stays):,}")

    result = pd.Series(
        cohort["stay_id"].isin(sepsis_clinical_stays).values,
        index=cohort["stay_id"].values,
        name="sepsis_clinical"
    )
    return result, sepsis_time_map


# =============================================================================
# Main
# =============================================================================

def main():
    log("=" * 60)
    log("Stage 01 — Sepsis-3 Hybrid Labeling")
    log("=" * 60)

    # Load cohort and feature grid
    log("Loading cohort_base.parquet …")
    cohort = pd.read_parquet(cfg.PROCESSED_DIR / "cohort_base.parquet")
    log(f"  Cohort: {len(cohort):,} stays")

    log("Loading hourly_features.parquet …")
    grid = pd.read_parquet(cfg.PROCESSED_DIR / "hourly_features.parquet")
    log(f"  Grid: {grid.shape}")

    # --- Method A: ICD ---
    log("\n[Method A] ICD-based labeling …")
    sepsis_icd = label_icd(cohort)

    # --- Method B: Clinical ---
    log("\n[Method B] Clinical Sepsis-3 labeling …")
    sepsis_clin, sepsis_time_map = label_clinical(cohort, grid)

    # --- Method C: Hybrid (A OR B) ---
    log("\n[Method C] Hybrid = A OR B …")
    cohort_out = cohort[["stay_id", "subject_id", "hadm_id", "intime", "outtime"]].copy()
    cohort_out["sepsis_icd"]      = cohort_out["stay_id"].map(sepsis_icd).fillna(False).astype(bool)
    cohort_out["sepsis_clinical"] = cohort_out["stay_id"].map(sepsis_clin).fillna(False).astype(bool)
    cohort_out["sepsis_hybrid"]   = cohort_out["sepsis_icd"] | cohort_out["sepsis_clinical"]

    # ── Per-strategy onset times (needed for label sensitivity analysis) ──────
    # Method A (ICD): no clinical onset time → assign 0 (coded sepsis from admission)
    cohort_out["sepsis_time_h_icd"] = np.where(cohort_out["sepsis_icd"], 0.0, np.nan)

    # Method B (Clinical): SOFA+infection onset hour
    cohort_out["sepsis_time_h_clinical"] = cohort_out["stay_id"].map(sepsis_time_map)

    # Method C (Hybrid): clinical onset if available, else 0 for ICD-coded stays
    cohort_out["sepsis_time_h"] = cohort_out["sepsis_time_h_clinical"]
    icd_only_mask = cohort_out["sepsis_hybrid"] & cohort_out["sepsis_time_h"].isna()
    cohort_out.loc[icd_only_mask, "sepsis_time_h"] = 0
    log(f"  ICD-only sepsis stays assigned sepsis_time_h=0: {icd_only_mask.sum():,}")


    # --- Prevalence report ---
    n_total = len(cohort_out)
    n_icd   = cohort_out["sepsis_icd"].sum()
    n_clin  = cohort_out["sepsis_clinical"].sum()
    n_hyb   = cohort_out["sepsis_hybrid"].sum()

    log("\n── Prevalence Report (Table 2) ──")
    log(f"  Total stays        : {n_total:,}")
    log(f"  Method A (ICD)     : {n_icd:,}  ({n_icd/n_total:.1%})")
    log(f"  Method B (Clinical): {n_clin:,}  ({n_clin/n_total:.1%})")
    log(f"  Method C (Hybrid)  : {n_hyb:,}  ({n_hyb/n_total:.1%})")

    # Save CSV summary (Table 2)
    prev_df = pd.DataFrame({
        "method"  : ["ICD (A)", "Clinical (B)", "Hybrid (C)"],
        "n_sepsis": [n_icd, n_clin, n_hyb],
        "n_total" : [n_total, n_total, n_total],
        "prevalence_pct": [
            round(n_icd/n_total*100, 2),
            round(n_clin/n_total*100, 2),
            round(n_hyb/n_total*100, 2),
        ]
    })
    prev_out = cfg.RESULTS_DIR / "label_prevalence.csv"
    prev_df.to_csv(prev_out, index=False)
    log(f"\n  Saved prevalence table → {prev_out}")

    # Save labeled cohort
    out_path = cfg.PROCESSED_DIR / "cohort_labeled.parquet"
    cohort_out.to_parquet(out_path, index=False)
    log(f"  Saved labeled cohort → {out_path}")

    log("\n[Stage 01 COMPLETE] ✓")
    log("  → Run next: python src/02_features.py")


if __name__ == "__main__":
    main()
