"""
eicu_extract.py — eICU-CRD External Validation Extraction (Section 2.2.2)
==========================================================================
Extracts and preprocesses the eICU-CRD dataset for external (zero-shot)
validation of the trained GWO-KD model.

Hospital Selection Criteria (concrete — addresses reviewer concern):
  1. ≥ 500 ICU stays per hospital (stable feature distributions)
  2. ≥ 70% vital sign hourly coverage per hospital
  3. ≥ 10 hospitals selected (geographic/institutional diversity)

Patient Inclusion (identical to MIMIC-IV):
  - Age ≥ 18
  - ICU LOS ≥ 4 hours
  - Vital completeness ≥ 50% per stay

Feature mapping to MIMIC-IV feature space (21 features currently extracted):
  heart_rate, sbp, dbp, map, resp_rate, spo2, temperature,
  wbc, creatinine, lactate, bilirubin, platelets, inr, pao2, fio2,
  gcs_total, pf_ratio, map_delta, rr_trend,
  vasopressor_flag, antibiotic_flag

Run:
    python src/eicu_extract.py

Outputs:
    processed/eicu/cohort.parquet
    processed/eicu/hourly_features.parquet
    processed/eicu/windows_features.npy
    processed/eicu/windows_labels.npy
    processed/eicu/hospital_selection.csv   ← key table for manuscript
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

import config as cfg

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# =============================================================================
# Step 1: Hospital selection with concrete, reported criteria
# =============================================================================

def select_hospitals(patient: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Select hospitals meeting all inclusion criteria.
    Returns: (filtered_patient_df, hospital_selection_table for manuscript)
    """
    log("Selecting hospitals …")

    # Count stays per hospital
    hosp_stats = patient.groupby("hospitalid").agg(
        n_stays    = ("patientunitstayid", "count"),
        n_patients = ("uniquepid", "nunique"),
    ).reset_index()

    # Vital coverage will be computed after loading vitals — for now apply volume filter
    vol_ok = hosp_stats["n_stays"] >= cfg.EICU_MIN_STAYS_PER_HOSPITAL
    log(f"  Hospitals with ≥{cfg.EICU_MIN_STAYS_PER_HOSPITAL} stays: "
        f"{vol_ok.sum()} / {len(hosp_stats)}")

    selected = hosp_stats[vol_ok].copy()

    # Ensure minimum number of hospitals
    if len(selected) < cfg.EICU_MIN_HOSPITALS:
        # Relax to top-N by stay count if needed
        log(f"  WARNING: Only {len(selected)} hospitals meet volume criterion. "
            f"Expanding to top {cfg.EICU_MIN_HOSPITALS}.")
        selected = hosp_stats.nlargest(cfg.EICU_MIN_HOSPITALS, "n_stays")

    selected = selected.sort_values("n_stays", ascending=False).reset_index(drop=True)
    log(f"  Selected: {len(selected)} hospitals, "
        f"{selected['n_stays'].sum():,} stays, "
        f"{selected['n_patients'].sum():,} unique patients")

    patient_sel = patient[patient["hospitalid"].isin(selected["hospitalid"])].copy()
    return patient_sel, selected


# =============================================================================
# Step 2: Patient cohort (same criteria as MIMIC-IV)
# =============================================================================

def build_eicu_cohort(patient: pd.DataFrame) -> pd.DataFrame:
    """
    Apply MIMIC-IV equivalent inclusion criteria to eICU patient table.
    """
    log("Building eICU cohort …")
    n_raw = len(patient)

    # Age ≥ 18
    patient = patient[patient["age"].apply(
        lambda x: float(x) if str(x).replace('.','').isdigit() else 89.0
    ) >= cfg.EICU_MIN_AGE].copy()
    log(f"  After age ≥{cfg.EICU_MIN_AGE}: {len(patient):,} (from {n_raw:,})")

    # LOS ≥ 4 hours (unitdischargeoffset in minutes)
    patient["los_hours"] = patient["unitdischargeoffset"] / 60.0
    patient = patient[patient["los_hours"] >= cfg.EICU_MIN_LOS_HOURS].copy()
    log(f"  After LOS ≥{cfg.EICU_MIN_LOS_HOURS}h: {len(patient):,}")

    # Parse admission time (eICU uses relative offsets, not absolute timestamps)
    # We use patientunitstayid as the unit identifier
    patient = patient.reset_index(drop=True)
    log(f"  Final eICU cohort: {len(patient):,} stays")
    return patient


# =============================================================================
# Step 3: Vital sign extraction (mapping to MIMIC-IV feature names)
# =============================================================================

EICU_VITAL_MAP = {
    # eICU vitalPeriodic column → MIMIC-IV feature name
    "heartrate"         : "heart_rate",
    "systemicsystolic"  : "sbp",
    "systemicdiastolic" : "dbp",
    "systemicmean"      : "map",
    "respiration"       : "resp_rate",
    "sao2"              : "spo2",
    "temperature"       : "temperature",
}

EICU_VITAL_NORMAL_RANGES = {
    # (low, high) for outlier clipping — same logic as MIMIC-IV
    "heart_rate"  : (20,   300),
    "sbp"         : (40,   280),
    "dbp"         : (10,   200),
    "map"         : (20,   200),
    "resp_rate"   : (1,    80),
    "spo2"        : (50,   100),
    "temperature" : (25,   45),    # in °C
}
# eICU lab.csv → MIMIC feature names
EICU_LAB_MAP = {
    "WBC x 1000"     : "wbc",
    "creatinine"     : "creatinine",
    "lactate"        : "lactate",
    "total bilirubin": "bilirubin",
    "platelets x 1000": "platelets",
    "PT - INR"       : "inr",
    "paO2"           : "pao2",
    "FiO2"           : "fio2",
}
EICU_LAB_NORMAL = {
    "wbc"       : (0.5,  150),
    "creatinine": (0.1,  30),
    "lactate"   : (0.1,  30),
    "bilirubin" : (0.1,  100),
    "platelets" : (5,    2000),
    "inr"       : (0.5,  20),
    "pao2"      : (20,   700),
    "fio2"      : (18,   100),
}

# GCS from nurseCharting
EICU_GCS_NAMES = ["GCS Total", "gcs total", "glasgow coma score"]

# Vasopressor & antibiotic keywords
VASOPRESSOR_DRUGS = ["norepinephrine","vasopressin","epinephrine","dopamine","dobutamine","phenylephrine"]
ANTIBIOTIC_DRUGS  = ["vancomycin","piperacillin","meropenem","cefepime","ceftriaxone","metronidazole",
                     "ciprofloxacin","levofloxacin","ampicillin","azithromycin","clindamycin"]


def extract_labs_eicu(cohort, lab_path):
    """Extract lab features from eICU lab.csv.gz → hourly long-format."""
    import pandas as pd, numpy as np
    log("Loading eICU lab.csv.gz ...")
    lab = pd.read_csv(lab_path, compression="gzip", low_memory=False,
                      usecols=["patientunitstayid","labresultoffset","labname","labresult"])
    stay_ids = set(cohort["patientunitstayid"])
    lab = lab[lab["patientunitstayid"].isin(stay_ids)].copy()
    lab["labresult"] = pd.to_numeric(lab["labresult"], errors="coerce")
    lab = lab.dropna(subset=["labresult"])
    log(f"  Lab rows: {len(lab):,}")

    rows = []
    for eicu_name, feat_name in EICU_LAB_MAP.items():
        sub = lab[lab["labname"] == eicu_name].copy()
        if len(sub) == 0:
            continue
        lo, hi = EICU_LAB_NORMAL[feat_name]
        sub = sub[(sub["labresult"] >= lo) & (sub["labresult"] <= hi)]
        sub = sub.rename(columns={"labresult": "value"})
        sub["feature"]     = feat_name
        sub["hour_offset"] = (sub["labresultoffset"] / 60).apply(np.floor).astype(int)
        rows.append(sub[["patientunitstayid","hour_offset","feature","value"]])
    if not rows:
        log("  WARNING: no lab rows matched")
        return pd.DataFrame(columns=["patientunitstayid","hour_offset","feature","value"])
    lab_long = pd.concat(rows, ignore_index=True)
    log(f"  Lab long rows: {len(lab_long):,}")
    return lab_long


def extract_gcs_eicu(cohort, nurse_path):
    """Extract GCS from eICU nurseCharting."""
    import pandas as pd, numpy as np
    log("Loading nurseCharting for GCS ...")
    try:
        nc = pd.read_csv(nurse_path, compression="gzip", low_memory=False,
                         usecols=["patientunitstayid","nursingchartoffset","nursingchartcelltypevallabel","nursingchartvalue"])
        stay_ids = set(cohort["patientunitstayid"])
        nc = nc[nc["patientunitstayid"].isin(stay_ids)].copy()
        gcs_mask = nc["nursingchartcelltypevallabel"].str.lower().str.contains("gcs total|glasgow coma", na=False)
        nc = nc[gcs_mask].copy()
        nc["value"] = pd.to_numeric(nc["nursingchartvalue"], errors="coerce")
        nc = nc.dropna(subset=["value"])
        nc = nc[(nc["value"] >= 3) & (nc["value"] <= 15)]
        nc["feature"]     = "gcs_total"
        nc["hour_offset"] = (nc["nursingchartoffset"] / 60).apply(np.floor).astype(int)
        gcs_long = nc[["patientunitstayid","hour_offset","feature","value"]]
        log(f"  GCS rows: {len(gcs_long):,}")
        return gcs_long
    except Exception as e:
        log(f"  GCS extraction failed: {e}")
        return pd.DataFrame(columns=["patientunitstayid","hour_offset","feature","value"])


def extract_flags_eicu(cohort, med_path, inf_path):
    """Extract vasopressor and antibiotic flags (hourly binary)."""
    import pandas as pd, numpy as np
    rows_vaso, rows_abx = [], []
    stay_ids = set(cohort["patientunitstayid"])

    for path, drugs, fname in [
        (inf_path,  VASOPRESSOR_DRUGS, "vasopressor_flag"),
        (med_path,  ANTIBIOTIC_DRUGS,  "antibiotic_flag"),
    ]:
        try:
            df = pd.read_csv(path, compression="gzip", low_memory=False)
            df = df[df["patientunitstayid"].isin(stay_ids)].copy()
            offset_col = "infusionoffset" if "infusionoffset" in df.columns else "drugstartoffset"
            drug_col   = "drugname" if "drugname" in df.columns else "infusiondrugid"
            if drug_col not in df.columns:
                continue
            df[drug_col] = df[drug_col].fillna("").str.lower()
            mask = df[drug_col].str.contains("|".join(drugs), na=False)
            df = df[mask & df[offset_col].notna()].copy()
            df["hour_offset"] = (df[offset_col] / 60).apply(np.floor).astype(int)
            df["feature"] = fname; df["value"] = 1.0
            (rows_vaso if fname == "vasopressor_flag" else rows_abx).append(
                df[["patientunitstayid","hour_offset","feature","value"]])
        except Exception as e:
            log(f"  {fname} extraction warning: {e}")

    out = []
    if rows_vaso: out.append(pd.concat(rows_vaso, ignore_index=True))
    if rows_abx:  out.append(pd.concat(rows_abx,  ignore_index=True))
    if out:
        combined = pd.concat(out, ignore_index=True)
        log(f"  Flag rows: {len(combined):,}")
        return combined
    return pd.DataFrame(columns=["patientunitstayid","hour_offset","feature","value"])




def extract_vitals_eicu(
    cohort: pd.DataFrame,
    vital_path: Path,
) -> pd.DataFrame:
    """
    Load vitalPeriodic, pivot to hourly rows, map to MIMIC-IV feature names.
    Returns long-format DataFrame: (patientunitstayid, hour_offset, feature, value)
    """
    log("Loading vitalPeriodic …")
    vp = pd.read_csv(vital_path, compression="gzip", low_memory=False)
    stay_ids = set(cohort["patientunitstayid"])
    vp = vp[vp["patientunitstayid"].isin(stay_ids)].copy()
    log(f"  vitalPeriodic rows after filter: {len(vp):,}")

    rows = []
    for eicu_col, feat_name in EICU_VITAL_MAP.items():
        if eicu_col not in vp.columns:
            continue
        sub = vp[["patientunitstayid", "observationoffset", eicu_col]].dropna()
        sub = sub.rename(columns={eicu_col: "value"})
        sub["feature"] = feat_name
        # Convert offset (minutes) to hour offset
        sub["hour_offset"] = (sub["observationoffset"] / 60).apply(np.floor).astype(int)
        rows.append(sub[["patientunitstayid", "hour_offset", "feature", "value"]])

    long_df = pd.concat(rows, ignore_index=True)

    # Temperature: convert F → C if > 45 (eICU mixes units)
    temp_mask = (long_df["feature"] == "temperature") & (long_df["value"] > 45)
    long_df.loc[temp_mask, "value"] = (long_df.loc[temp_mask, "value"] - 32) * 5 / 9

    return long_df


# =============================================================================
# Step 4: Vital coverage per hospital (concrete criterion)
# =============================================================================

def compute_vital_coverage(
    cohort: pd.DataFrame,
    vital_long: pd.DataFrame,
    selected_hospitals: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute fraction of patient-hours with ≥1 vital sign recorded per hospital.
    Used to validate and report the ≥70% coverage criterion.
    """
    log("Computing vital coverage per hospital …")
    # Total patient-hours per hospital
    total_ph = cohort.groupby("hospitalid")["los_hours"].sum()

    # Patient-hours with at least one vital
    stay_hosp = cohort[["patientunitstayid", "hospitalid"]].drop_duplicates()
    vd = vital_long.merge(stay_hosp, on="patientunitstayid")
    covered_ph = (vd.groupby(["hospitalid", "patientunitstayid", "hour_offset"])
                    .size().reset_index()
                    .groupby("hospitalid")
                    .size())

    coverage = (covered_ph / total_ph.clip(lower=1)).clip(upper=1.0)
    selected_hospitals = selected_hospitals.copy()
    selected_hospitals["vital_coverage"] = selected_hospitals["hospitalid"].map(coverage)
    selected_hospitals["meets_coverage"] = selected_hospitals["vital_coverage"] >= cfg.EICU_MIN_VITAL_COVERAGE

    log(f"  Hospitals meeting vital coverage ≥{cfg.EICU_MIN_VITAL_COVERAGE:.0%}: "
        f"{selected_hospitals['meets_coverage'].sum()} / {len(selected_hospitals)}")

    # If some don't meet coverage, apply secondary filter but keep ≥EICU_MIN_HOSPITALS
    meets_both = selected_hospitals[selected_hospitals["meets_coverage"]]
    if len(meets_both) >= cfg.EICU_MIN_HOSPITALS:
        log(f"  Applying vital coverage filter → {len(meets_both)} hospitals retained")
        return meets_both
    else:
        log(f"  WARNING: Only {len(meets_both)} hospitals meet both criteria; "
            f"retaining all {len(selected_hospitals)} volume-selected hospitals")
        return selected_hospitals


# =============================================================================
# Step 5: Sepsis labeling (ICD-based, eICU diagnosis table)
# =============================================================================

EICU_SEPSIS_KEYWORDS = [
    "sepsis", "septicemia", "septic shock",
    "severe sepsis", "urosepsis", "bacteremia"
]


def label_sepsis_eicu(
    cohort: pd.DataFrame,
    diag_path: Path,
) -> pd.Series:
    """
    Label sepsis stays in eICU using diagnosis table keyword matching.
    Returns boolean Series indexed by patientunitstayid.
    Note: eICU has no ICD codes → use diagnosis string matching (same concept,
    different implementation — explicitly reported as limitation).
    """
    log("Loading eICU diagnosis table …")
    diag = pd.read_csv(diag_path, compression="gzip", low_memory=False)
    stay_ids = set(cohort["patientunitstayid"])
    diag = diag[diag["patientunitstayid"].isin(stay_ids)].copy()

    diag["diagnosisstring"] = diag["diagnosisstring"].fillna("").str.lower()
    pattern = "|".join(EICU_SEPSIS_KEYWORDS)
    sepsis_stays = diag[diag["diagnosisstring"].str.contains(pattern, na=False)][
        "patientunitstayid"
    ].unique()

    result = pd.Series(
        cohort["patientunitstayid"].isin(sepsis_stays).values,
        index=cohort["patientunitstayid"].values,
        name="sepsis_eicu"
    )
    n = result.sum()
    log(f"  Sepsis stays (keyword-based): {n:,} ({n/len(result):.1%})")
    return result


# =============================================================================
# Step 6: Hourly grid + feature windows (same logic as 02_features.py)
# =============================================================================

def build_eicu_windows(
    cohort: pd.DataFrame,
    vital_long: pd.DataFrame,
    sepsis_labels: pd.Series,
    norm_stats: dict,
    lab_long: pd.DataFrame = None,
    gcs_long: pd.DataFrame = None,
    flag_long: pd.DataFrame = None,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Build sliding windows from eICU data, normalized using MIMIC-IV train statistics.
    Critical: uses MIMIC-IV normalization — no data leakage from eICU.

    Returns: (X, y, meta_df)
    """
    log("Building hourly grid …")
    # 21 MIMIC features: 7 vitals + 8 labs + 4 derived + 2 binary
    FEATURE_COLS = [
        "heart_rate","sbp","dbp","map","resp_rate","spo2","temperature",
        "wbc","creatinine","lactate","bilirubin","platelets","inr","pao2","fio2",
        "gcs_total","pf_ratio","map_delta","rr_trend",
        "vasopressor_flag","antibiotic_flag",
    ]
    feature_names = FEATURE_COLS

    # Gabung semua sumber data
    all_parts = [vital_long]
    if lab_long  is not None and len(lab_long)  > 0: all_parts.append(lab_long)
    if gcs_long  is not None and len(gcs_long)  > 0: all_parts.append(gcs_long)
    if flag_long is not None and len(flag_long) > 0: all_parts.append(flag_long)
    combined_long = pd.concat(all_parts, ignore_index=True)

    # Pivot to hourly wide format
    agg = (combined_long.groupby(["patientunitstayid", "hour_offset", "feature"])["value"]
                     .mean().reset_index())
    wide = agg.pivot_table(index=["patientunitstayid", "hour_offset"],
                           columns="feature", values="value").reset_index()

    # Forward fill (2h for vitals)
    wide = wide.sort_values(["patientunitstayid", "hour_offset"])
    for feat in feature_names:
        if feat not in wide.columns:
            wide[feat] = np.nan
        wide[feat] = (wide.groupby("patientunitstayid")[feat]
                           .transform(lambda s: s.ffill(limit=cfg.FFILL_VITALS_HOURS)))

    # Compute derived features (setelah ffill, sebelum normalisasi)
    if "pao2" in wide.columns and "fio2" in wide.columns:
        fio2_safe = wide["fio2"].clip(lower=1)
        wide["pf_ratio"] = (wide["pao2"] / (fio2_safe / 100)).clip(0, 1000)
    if "map" in wide.columns:
        wide["map_delta"] = wide.groupby("patientunitstayid")["map"].diff().fillna(0).clip(-100, 100)
    if "resp_rate" in wide.columns:
        wide["rr_trend"] = wide.groupby("patientunitstayid")["resp_rate"].diff().fillna(0).clip(-30, 30)
    # Binary flags: fill NaN with 0
    for flag in ["vasopressor_flag","antibiotic_flag"]:
        if flag in wide.columns:
            wide[flag] = wide[flag].fillna(0)
        else:
            wide[flag] = 0.0

    # Normalize using MIMIC-IV train statistics
    for feat in feature_names:
        if feat in norm_stats:
            mu  = norm_stats[feat]["mean"]
            sig = norm_stats[feat]["std"]
            lo  = norm_stats[feat]["p1"]
            hi  = norm_stats[feat]["p99"]
            wide[feat] = wide[feat].clip(lo, hi)
            wide[feat] = (wide[feat] - mu) / (sig + 1e-8)

    # Pastikan semua 21 feature ada (tambahkan kolom 0 jika tidak ada)
    for feat in feature_names:
        if feat not in wide.columns:
            wide[feat] = 0.0

    # Final imputation: isi NaN dengan 0 (= mean di normalized space)
    # Untuk binary flags sudah di-fill 0, untuk normalized features 0 = mean
    wide[feature_names] = wide[feature_names].fillna(0.0)

    log(f"  NaN setelah imputation: {wide[feature_names].isna().sum().sum()}")

    # Sliding windows (12h, step 1h)
    W = cfg.WINDOW_SIZE_HOURS
    X_list, y_list, meta_list = [], [], []

    for sid, grp in tqdm(wide.groupby("patientunitstayid"), desc="Building windows"):
        grp   = grp.sort_values("hour_offset").reset_index(drop=True)
        hours = grp["hour_offset"].values
        feats = grp[feature_names].values.astype(np.float32)

        is_sep  = bool(sepsis_labels.get(sid, False))
        # For eICU: sepsis_time = 0 if sepsis (ICD-equivalent)
        sep_t   = 0 if is_sep else None

        los     = cohort.loc[cohort["patientunitstayid"] == sid, "los_hours"].values
        los_h   = int(los[0]) if len(los) > 0 else feats.shape[0]

        for i in range(len(grp) - W + 1):
            w_feats = feats[i : i + W]
            w_end   = int(hours[i + W - 1])
            if w_feats.shape[0] < W:
                continue
            label = int(is_sep and sep_t is not None and sep_t <= w_end)
            X_list.append(w_feats)
            y_list.append(label)
            meta_list.append({"patientunitstayid": sid, "window_end_hour": w_end})

    X = np.stack(X_list).astype(np.float32)
    y = np.array(y_list, dtype=np.int8)
    meta = pd.DataFrame(meta_list)
    log(f"  Windows: {X.shape} | positives: {y.sum():,} ({y.mean():.2%})")
    return X, y, meta


# =============================================================================
# Main
# =============================================================================

def main():
    log("=" * 60)
    log("eICU-CRD External Validation Extraction (Section 2.2.2)")
    log("=" * 60)

    # ── Check normalization stats exist ───────────────────────────────────────
    norm_path = cfg.PROCESSED_DIR / "norm_stats.parquet"
    if not norm_path.exists():
        log("ERROR: norm_stats.parquet not found.")
        log("  Run 02_features.py first — it saves train normalization statistics.")
        return
    norm_stats = pd.read_parquet(norm_path).to_dict(orient="index")
    log(f"  Loaded normalization stats for {len(norm_stats)} features")

    # ── Load patient table ────────────────────────────────────────────────────
    log("\n[Step 1] Loading patient table …")
    patient = pd.read_csv(cfg.EICU_FILES["patient"], compression="gzip",
                          low_memory=False)
    log(f"  Total stays: {len(patient):,}")

    # ── Hospital selection ────────────────────────────────────────────────────
    log("\n[Step 2] Hospital selection (concrete criteria) …")
    patient_sel, hosp_df = select_hospitals(patient)

    # ── Patient cohort ────────────────────────────────────────────────────────
    log("\n[Step 3] Patient cohort …")
    cohort = build_eicu_cohort(patient_sel)

    # ── Vital extraction ──────────────────────────────────────────────────────
    log("\n[Step 4] Vital sign extraction …")
    vital_long = extract_vitals_eicu(cohort, cfg.EICU_FILES["vitalPeriodic"])

    # ── Vital coverage per hospital (validate criterion) ──────────────────────
    log("\n[Step 4b] Vital coverage validation …")
    hosp_df = compute_vital_coverage(cohort, vital_long, hosp_df)
    hosp_path = cfg.EICU_PROCESSED_DIR / "hospital_selection.csv"
    hosp_df.to_csv(hosp_path, index=False)
    log(f"  Hospital selection table → {hosp_path}")
    log("\n" + hosp_df[["hospitalid", "n_stays", "n_patients",
                         "vital_coverage", "meets_coverage"]
                       ].head(20).to_string(index=False))

    # Filter cohort to hospitals that pass both criteria
    final_hosp = set(hosp_df["hospitalid"])
    cohort = cohort[cohort["hospitalid"].isin(final_hosp)].copy()
    vital_long = vital_long[vital_long["patientunitstayid"].isin(
        set(cohort["patientunitstayid"]))].copy()
    log(f"\n  Final cohort: {len(cohort):,} stays, "
        f"{cohort['hospitalid'].nunique()} hospitals")

    # ── Vital completeness filter ─────────────────────────────────────────────
    log("\n[Step 4c] Per-stay vital completeness filter …")
    stay_hours = cohort[["patientunitstayid", "los_hours"]].set_index("patientunitstayid")
    vital_counts = vital_long.groupby("patientunitstayid")["hour_offset"].nunique()
    completeness = (vital_counts / stay_hours["los_hours"].clip(lower=1)).clip(upper=1.0)
    low_comp = completeness[completeness < cfg.EICU_VITAL_COMPLETENESS_MIN].index
    cohort   = cohort[~cohort["patientunitstayid"].isin(low_comp)].copy()
    log(f"  Removed {len(low_comp):,} low-completeness stays → {len(cohort):,} remain")

    # ── Sepsis labeling ───────────────────────────────────────────────────────
    log("\n[Step 5] Sepsis labeling …")
    sepsis_labels = label_sepsis_eicu(cohort, cfg.EICU_FILES["diagnosis"])

    # Save cohort
    cohort_path = cfg.EICU_PROCESSED_DIR / "cohort.parquet"
    cohort.to_parquet(cohort_path, index=False)
    log(f"  Saved → {cohort_path}")

    # ── Build windows ─────────────────────────────────────────────────────────
    log("\n[Step 6] Building windows (MIMIC-IV normalization applied) …")
    # Extract labs, GCS, medication flags
    lab_long  = extract_labs_eicu(cohort, cfg.EICU_FILES["lab"])
    gcs_long  = extract_gcs_eicu(cohort, cfg.EICU_FILES["nurseCharting"])
    flag_long = extract_flags_eicu(cohort, cfg.EICU_FILES["medication"], cfg.EICU_FILES["infusionDrug"])

    X, y, meta = build_eicu_windows(cohort, vital_long, sepsis_labels, norm_stats,
                                    lab_long=lab_long, gcs_long=gcs_long, flag_long=flag_long)

    np.save(cfg.EICU_PROCESSED_DIR / "windows_features.npy", X)
    np.save(cfg.EICU_PROCESSED_DIR / "windows_labels.npy", y)
    meta.to_parquet(cfg.EICU_PROCESSED_DIR / "windows_meta.parquet", index=False)
    log(f"  Saved X: {X.shape}  y: {y.shape}")

    # ── Summary ───────────────────────────────────────────────────────────────
    log("\n── eICU Extraction Summary ──")
    log(f"  Hospitals  : {cohort['hospitalid'].nunique()} "
        f"(min stays={cfg.EICU_MIN_STAYS_PER_HOSPITAL}, "
        f"min coverage={cfg.EICU_MIN_VITAL_COVERAGE:.0%}, "
        f"min hospitals={cfg.EICU_MIN_HOSPITALS})")
    log(f"  Stays      : {len(cohort):,}")
    log(f"  Sepsis     : {sepsis_labels.sum():,} ({sepsis_labels.mean():.1%})")
    log(f"  Windows    : {len(y):,}  pos={y.sum():,} ({y.mean():.1%})")
    log(f"  Features   : {X.shape[2]} (eICU subset of MIMIC-IV feature space)")
    log(f"  Normalization: MIMIC-IV train statistics (zero-shot, no eICU leakage)")

    log("\n[eICU Extraction COMPLETE] ✓")
    log("  → Run next: python src/10_evaluate.py --dataset eicu")


if __name__ == "__main__":
    main()
