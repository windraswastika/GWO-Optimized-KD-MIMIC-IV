"""
00_extract.py — Stage 0: MIMIC-IV Data Extraction
==================================================
Extracts and joins MIMIC-IV tables to produce a clean per-patient-hour
feature table saved as processed/hourly_features.parquet.

Steps:
  1. Load icustays + patients + admissions → base cohort with age & LOS
  2. Apply inclusion/exclusion criteria (Section 2.3)
  3. Extract chartevents (vitals + GCS) via DuckDB — avoids loading 39 GB to RAM
  4. Extract labevents (8 lab values) via DuckDB — avoids loading 17 GB to RAM
  5. Extract vasopressor flags from inputevents
  6. Extract antibiotic flags from prescriptions
  7. Join everything on (stay_id, hour_offset) → hourly grid
  8. Save processed/cohort_base.parquet and processed/hourly_features.parquet

Hardware note: Apple M4, 24 GB RAM.
DuckDB memory capped at 18 GB; chunked parquet writes used for large tables.

Run:
    python src/00_extract.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import duckdb
import pandas as pd
import numpy as np
import logging
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

import config as cfg

# ─── helpers ──────────────────────────────────────────────────────────────────

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


_RUN_TS  = datetime.now().strftime("%Y%m%d_%H%M%S")
_LOG_PATH = cfg.LOGS_DIR / f"stage00_{_RUN_TS}.log"
_logger   = _setup_logger("stage00", _LOG_PATH)


def log(msg: str):
    _logger.info(msg)


def init_duckdb() -> duckdb.DuckDBPyConnection:
    """Create an in-memory DuckDB connection with memory limit."""
    con = duckdb.connect()
    con.execute(f"SET memory_limit='{cfg.DUCKDB_MEMORY_LIMIT}'")
    con.execute("SET threads=8")                # M4 has 10 cores, leave 2 for OS
    con.execute("SET enable_progress_bar=true")
    return con

# =============================================================================
# Step 1 — Base cohort: icustays + patients + admissions
# =============================================================================

def load_base_cohort() -> pd.DataFrame:
    """
    Load and filter the ICU stay cohort.
    Returns DataFrame with columns:
        stay_id, subject_id, hadm_id, intime, outtime, los_hours,
        age_years, gender, admit_type, first_careunit
    """
    log("Loading icustays.csv …")
    icu = pd.read_csv(cfg.MIMIC_FILES["icustays"], parse_dates=["intime", "outtime"])
    log(f"  icustays: {len(icu):,} rows")

    log("Loading patients.csv …")
    pts = pd.read_csv(cfg.MIMIC_FILES["patients"])
    log(f"  patients: {len(pts):,} rows")

    log("Loading admissions.csv …")
    adm = pd.read_csv(cfg.MIMIC_FILES["admissions"], parse_dates=["admittime"])
    log(f"  admissions: {len(adm):,} rows")

    # Merge
    cohort = icu.merge(pts[["subject_id", "gender", "anchor_age", "anchor_year"]], 
                       on="subject_id", how="left")
    cohort = cohort.merge(adm[["hadm_id", "admittime", "admission_type"]], 
                          on="hadm_id", how="left")

    # LOS in hours
    cohort["los_hours"] = (cohort["outtime"] - cohort["intime"]).dt.total_seconds() / 3600

    # Approximate age: anchor_age is age at anchor_year
    # admittime year - anchor_year + anchor_age
    cohort["admit_year"] = cohort["admittime"].dt.year
    cohort["age_years"]  = cohort["anchor_age"] + (cohort["admit_year"] - cohort["anchor_year"])

    return cohort


def apply_cohort_criteria(cohort: pd.DataFrame) -> pd.DataFrame:
    """
    Apply inclusion/exclusion criteria (Section 2.3):
      Include: adult ≥18 y, ICU stay ≥4 h
      Exclude: paediatric/neonatal; LOS < 4 h
      Note: ICU readmission within 24 h and vital completeness < 50%
            are applied after chartevents extraction.
    """
    log(f"  Before criteria: {len(cohort):,} stays")

    # Adult only
    cohort = cohort[cohort["age_years"] >= cfg.MIN_AGE_YEARS].copy()
    log(f"  After age ≥{cfg.MIN_AGE_YEARS}: {len(cohort):,} stays")

    # LOS ≥ 4 h
    cohort = cohort[cohort["los_hours"] >= cfg.MIN_LOS_HOURS].copy()
    log(f"  After LOS ≥{cfg.MIN_LOS_HOURS} h: {len(cohort):,} stays")

    # Remove ICU readmissions within 24 h: sort by subject + intime, diff intime from previous outtime
    cohort = cohort.sort_values(["subject_id", "intime"])
    cohort["prev_outtime"] = cohort.groupby("subject_id")["outtime"].shift(1)
    cohort["gap_hours"] = (cohort["intime"] - cohort["prev_outtime"]).dt.total_seconds() / 3600
    # Keep first admission or gap ≥ 24 h
    mask_readmit = (cohort["gap_hours"].notna()) & (cohort["gap_hours"] < 24)
    n_removed = mask_readmit.sum()
    cohort = cohort[~mask_readmit].copy()
    log(f"  After removing readmissions within 24 h: removed {n_removed}, remaining {len(cohort):,}")

    cohort = cohort.reset_index(drop=True)
    return cohort


# =============================================================================
# Step 2 — Chartevents extraction via DuckDB  (vitals + GCS)
# =============================================================================

def extract_chartevents(cohort: pd.DataFrame, con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """
    Extract chartevents for the cohort's stay_ids.
    Uses DuckDB to scan the 39 GB CSV directly — no full load to RAM.
    Returns long-format DataFrame: stay_id, charttime, itemid, valuenum
    """
    log("Extracting chartevents via DuckDB …")
    stay_ids = cohort["stay_id"].tolist()
    itemids  = cfg.CHART_ITEMIDS

    # Write stay_id and itemid lists as temp tables for DuckDB filter
    con.execute("CREATE TABLE IF NOT EXISTS tmp_stays  (stay_id INTEGER)")
    con.execute("CREATE TABLE IF NOT EXISTS tmp_citems (itemid  INTEGER)")
    con.execute("DELETE FROM tmp_stays")
    con.execute("DELETE FROM tmp_citems")
    con.executemany("INSERT INTO tmp_stays VALUES (?)",  [[s] for s in stay_ids])
    con.executemany("INSERT INTO tmp_citems VALUES (?)", [[i] for i in itemids])

    chart_csv = str(cfg.MIMIC_FILES["chartevents"])
    log(f"  Scanning {chart_csv} …  (this may take 15–40 min on M4)")

    df = con.execute(f"""
        SELECT
            c.stay_id,
            c.charttime,
            c.itemid,
            c.valuenum
        FROM read_csv_auto('{chart_csv}', header=True, parallel=True) AS c
        INNER JOIN tmp_stays  s ON c.stay_id = s.stay_id
        INNER JOIN tmp_citems i ON c.itemid   = i.itemid
        WHERE c.valuenum IS NOT NULL
          AND (c.warning IS NULL OR c.warning = 0)
    """).df()

    log(f"  Chartevents extracted: {len(df):,} rows")
    df["charttime"] = pd.to_datetime(df["charttime"])
    return df


# =============================================================================
# Step 3 — Labevents extraction via DuckDB
# =============================================================================

def extract_labevents(cohort: pd.DataFrame, con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """
    Extract labevents for the cohort. Joins on hadm_id.
    Returns long-format DataFrame: hadm_id, stay_id, charttime, itemid, valuenum
    """
    log("Extracting labevents via DuckDB …")

    hadm_ids = cohort["hadm_id"].tolist()
    itemids  = cfg.LAB_ITEMIDS_FLAT

    con.execute("CREATE TABLE IF NOT EXISTS tmp_hadms (hadm_id INTEGER)")
    con.execute("CREATE TABLE IF NOT EXISTS tmp_litems (itemid INTEGER)")
    con.execute("DELETE FROM tmp_hadms")
    con.execute("DELETE FROM tmp_litems")
    con.executemany("INSERT INTO tmp_hadms  VALUES (?)", [[h] for h in hadm_ids])
    con.executemany("INSERT INTO tmp_litems VALUES (?)", [[i] for i in itemids])

    lab_csv = str(cfg.MIMIC_FILES["labevents"])
    log(f"  Scanning {lab_csv} …  (this may take 10–20 min on M4)")

    df = con.execute(f"""
        SELECT
            l.hadm_id,
            l.charttime,
            l.itemid,
            l.valuenum
        FROM read_csv_auto('{lab_csv}', header=True, parallel=True) AS l
        INNER JOIN tmp_hadms  h ON l.hadm_id = h.hadm_id
        INNER JOIN tmp_litems i ON l.itemid   = i.itemid
        WHERE l.valuenum IS NOT NULL
    """).df()

    log(f"  Labevents extracted: {len(df):,} rows")
    df["charttime"] = pd.to_datetime(df["charttime"])

    # Join stay_id from cohort (hadm_id → stay_id mapping)
    hadm_stay = cohort[["hadm_id", "stay_id", "intime"]].drop_duplicates("hadm_id")
    df = df.merge(hadm_stay, on="hadm_id", how="inner")
    return df


# =============================================================================
# Step 4 — Vasopressor flags from inputevents
# =============================================================================

def extract_vasopressors(cohort: pd.DataFrame, con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """
    Extract vasopressor infusion events from inputevents.
    Returns: stay_id, starttime, endtime (for binary flag creation).
    """
    log("Extracting vasopressor flags from inputevents …")
    stay_ids = cohort["stay_id"].tolist()
    vaso_ids = cfg.VASOPRESSOR_ITEMIDS

    con.execute("CREATE TABLE IF NOT EXISTS tmp_vaso (itemid INTEGER)")
    con.execute("DELETE FROM tmp_vaso")
    con.executemany("INSERT INTO tmp_vaso VALUES (?)", [[i] for i in vaso_ids])

    input_csv = str(cfg.MIMIC_FILES["inputevents"])
    log(f"  Scanning {input_csv} …")

    df = con.execute(f"""
        SELECT
            ie.stay_id,
            ie.starttime,
            ie.endtime,
            ie.itemid
        FROM read_csv_auto('{input_csv}', header=True, parallel=True) AS ie
        INNER JOIN tmp_stays  s ON ie.stay_id = s.stay_id
        INNER JOIN tmp_vaso   v ON ie.itemid   = v.itemid
        WHERE ie.amount > 0
    """).df()

    log(f"  Vasopressor events: {len(df):,} rows")
    df["starttime"] = pd.to_datetime(df["starttime"])
    df["endtime"]   = pd.to_datetime(df["endtime"])
    return df


# =============================================================================
# Step 5 — Antibiotic flags from prescriptions
# =============================================================================

def extract_antibiotics(cohort: pd.DataFrame, con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """
    Extract antibiotic prescriptions (keyword match on drug name).
    Returns: hadm_id, stay_id, starttime, stoptime
    """
    log("Extracting antibiotic flags from prescriptions …")

    # Build SQL LIKE conditions
    kw_conditions = " OR ".join(
        [f"LOWER(p.drug) LIKE '%{kw.lower()}%'" for kw in cfg.ANTIBIOTIC_KEYWORDS]
    )

    rx_csv = str(cfg.MIMIC_FILES["prescriptions"])
    log(f"  Scanning {rx_csv} …")

    df = con.execute(f"""
        SELECT
            p.hadm_id,
            p.starttime,
            p.stoptime,
            p.drug
        FROM read_csv_auto('{rx_csv}', header=True, parallel=True) AS p
        INNER JOIN tmp_hadms h ON p.hadm_id = h.hadm_id
        WHERE ({kw_conditions})
    """).df()

    log(f"  Antibiotic events: {len(df):,} rows")
    df["starttime"] = pd.to_datetime(df["starttime"])
    df["stoptime"]  = pd.to_datetime(df["stoptime"])

    hadm_stay = cohort[["hadm_id", "stay_id", "intime"]].drop_duplicates("hadm_id")
    df = df.merge(hadm_stay, on="hadm_id", how="inner")
    return df


# =============================================================================
# Step 6 — Build hourly grid and pivot features
# =============================================================================

def build_hourly_grid(cohort: pd.DataFrame) -> pd.DataFrame:
    """
    Create a row for every (stay_id, hour_offset) pair up to LOS.
    hour_offset is hours since ICU admission (0, 1, 2, …).
    """
    log("Building hourly grid …")
    rows = []
    for _, row in tqdm(cohort.iterrows(), total=len(cohort), desc="Grid"):
        max_hour = int(row["los_hours"])
        for h in range(max_hour + 1):
            rows.append({
                "stay_id"    : row["stay_id"],
                "hour_offset": h,
                "hour_ts"    : row["intime"] + pd.Timedelta(hours=h),
            })
    grid = pd.DataFrame(rows)
    log(f"  Hourly grid: {len(grid):,} rows")
    return grid


def add_vital_to_grid(grid: pd.DataFrame, chart_long: pd.DataFrame, 
                      cohort: pd.DataFrame) -> pd.DataFrame:
    """
    Convert long-format chartevents into per-hour means pivoted onto the grid.
    Groups: heart_rate, sbp, dbp, map, resp_rate, spo2, temperature, gcs_*
    """
    log("Pivoting vital signs onto hourly grid …")

    # Map itemid → feature name
    itemid_to_feat: dict[int, str] = {}
    for feat, ids in {**cfg.VITAL_ITEMIDS, **cfg.GCS_ITEMIDS}.items():
        for iid in ids:
            itemid_to_feat[iid] = feat

    chart_long = chart_long.copy()
    chart_long["feature"] = chart_long["itemid"].map(itemid_to_feat)

    # Convert Fahrenheit to Celsius for temperature (itemid 223761)
    mask_f = chart_long["itemid"] == 223761
    chart_long.loc[mask_f, "valuenum"] = (chart_long.loc[mask_f, "valuenum"] - 32) * 5 / 9

    # Compute hour_offset = floor((charttime - intime) / 1h)
    intime_map = cohort.set_index("stay_id")["intime"]
    chart_long["intime"] = chart_long["stay_id"].map(intime_map)
    hour_delta = (
        (chart_long["charttime"] - chart_long["intime"])
        .dt.total_seconds() / 3600
    )
    # Drop rows where intime could not be mapped (stay_id not in cohort)
    # or where charttime is NaT — these produce NaN/inf that crash astype(int)
    n_before = len(chart_long)
    chart_long = chart_long[hour_delta.notna() & np.isfinite(hour_delta)].copy()
    n_dropped = n_before - len(chart_long)
    if n_dropped:
        log(f"  Dropped {n_dropped:,} chart rows with unmappable/null intime")
    chart_long["hour_offset"] = np.floor(hour_delta[chart_long.index]).astype(int)

    # Keep only rows within valid stay window
    chart_long = chart_long[chart_long["hour_offset"] >= 0]

    # Aggregate: mean per (stay_id, hour_offset, feature)
    agg = (
        chart_long
        .groupby(["stay_id", "hour_offset", "feature"], as_index=False)["valuenum"]
        .mean()
    )

    # Pivot: features → columns
    pivoted = agg.pivot_table(
        index=["stay_id", "hour_offset"],
        columns="feature",
        values="valuenum",
        aggfunc="mean"
    ).reset_index()
    pivoted.columns.name = None

    # Derived GCS total
    gcs_cols = ["gcs_eye", "gcs_verbal", "gcs_motor"]
    existing_gcs = [c for c in gcs_cols if c in pivoted.columns]
    if existing_gcs:
        pivoted["gcs_total"] = pivoted[existing_gcs].sum(axis=1, min_count=1)

    # Merge onto grid
    grid = grid.merge(pivoted, on=["stay_id", "hour_offset"], how="left")
    log(f"  Vitals added: {len(grid):,} rows × {len(grid.columns)} cols")
    return grid


def add_labs_to_grid(grid: pd.DataFrame, lab_long: pd.DataFrame, 
                     cohort: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot labevents onto the hourly grid (similar to vitals).
    """
    log("Pivoting lab values onto hourly grid …")

    itemid_to_feat: dict[int, str] = {}
    for feat, ids in cfg.LAB_ITEMIDS.items():
        for iid in ids:
            itemid_to_feat[iid] = feat

    lab_long = lab_long.copy()
    lab_long["feature"] = lab_long["itemid"].map(itemid_to_feat)

    intime_map = cohort.set_index("stay_id")["intime"]
    lab_long["intime"] = lab_long["stay_id"].map(intime_map)
    hour_delta = (
        (lab_long["charttime"] - lab_long["intime"])
        .dt.total_seconds() / 3600
    )
    n_before = len(lab_long)
    lab_long = lab_long[hour_delta.notna() & np.isfinite(hour_delta)].copy()
    n_dropped = n_before - len(lab_long)
    if n_dropped:
        log(f"  Dropped {n_dropped:,} lab rows with unmappable/null intime")
    lab_long["hour_offset"] = np.floor(hour_delta[lab_long.index]).astype(int)
    lab_long = lab_long[(lab_long["hour_offset"] >= 0)]

    agg = (
        lab_long
        .groupby(["stay_id", "hour_offset", "feature"], as_index=False)["valuenum"]
        .mean()
    )
    pivoted = agg.pivot_table(
        index=["stay_id", "hour_offset"],
        columns="feature",
        values="valuenum",
        aggfunc="mean"
    ).reset_index()
    pivoted.columns.name = None

    grid = grid.merge(pivoted, on=["stay_id", "hour_offset"], how="left")
    log(f"  Labs added: {len(grid):,} rows × {len(grid.columns)} cols")
    return grid


def add_med_flags_to_grid(grid: pd.DataFrame, 
                          vaso_df: pd.DataFrame, 
                          abx_df: pd.DataFrame,
                          cohort: pd.DataFrame) -> pd.DataFrame:
    """
    For each (stay_id, hour_offset), set binary flags:
      vasopressor_flag = 1 if any vasopressor infusing during that hour
      antibiotic_flag  = 1 if any antibiotic prescribed/active during that hour
    """
    log("Adding medication binary flags …")
    intime_map = cohort.set_index("stay_id")["intime"]

    # ---- Vasopressor flag ----
    vaso_df = vaso_df.copy()
    vaso_df["intime"] = vaso_df["stay_id"].map(intime_map)
    # Drop rows with NaT starttime/endtime (cannot compute hour offset)
    vaso_df = vaso_df.dropna(subset=["starttime", "endtime", "intime"])
    vaso_df["start_h"] = ((vaso_df["starttime"] - vaso_df["intime"]).dt.total_seconds() / 3600).apply(np.floor)
    vaso_df["end_h"]   = ((vaso_df["endtime"]   - vaso_df["intime"]).dt.total_seconds() / 3600).apply(np.ceil)
    vaso_df = vaso_df.dropna(subset=["start_h", "end_h"])
    vaso_df["start_h"] = vaso_df["start_h"].astype(int)
    vaso_df["end_h"]   = vaso_df["end_h"].astype(int)

    # Expand into per-hour flags
    vaso_rows = []
    for _, r in tqdm(vaso_df.iterrows(), total=len(vaso_df), desc="Vaso expand"):
        for h in range(max(0, r["start_h"]), r["end_h"] + 1):
            vaso_rows.append({"stay_id": r["stay_id"], "hour_offset": h})
    if vaso_rows:
        vaso_flag = pd.DataFrame(vaso_rows).drop_duplicates()
        vaso_flag["vasopressor_flag"] = 1
        grid = grid.merge(vaso_flag, on=["stay_id", "hour_offset"], how="left")
        grid["vasopressor_flag"] = grid["vasopressor_flag"].fillna(0).astype(int)
    else:
        grid["vasopressor_flag"] = 0

    # ---- Antibiotic flag ----
    abx_df = abx_df.copy()
    abx_df["intime"] = abx_df["stay_id"].map(intime_map)
    # Drop rows with NaT starttime/stoptime (some prescriptions have null stop)
    abx_df = abx_df.dropna(subset=["starttime", "intime"])
    # stoptime may be null — use a default of starttime + 24 h if missing
    abx_df["stoptime"] = abx_df["stoptime"].fillna(abx_df["starttime"] + pd.Timedelta(hours=24))
    abx_df["start_h"] = ((abx_df["starttime"] - abx_df["intime"]).dt.total_seconds() / 3600).apply(np.floor)
    abx_df["end_h"]   = ((abx_df["stoptime"]  - abx_df["intime"]).dt.total_seconds() / 3600).apply(np.ceil)
    abx_df = abx_df.dropna(subset=["start_h", "end_h"])
    abx_df["start_h"] = abx_df["start_h"].astype(int)
    abx_df["end_h"]   = abx_df["end_h"].astype(int)

    abx_rows = []
    for _, r in tqdm(abx_df.iterrows(), total=len(abx_df), desc="Abx expand"):
        for h in range(max(0, r["start_h"]), r["end_h"] + 1):
            abx_rows.append({"stay_id": r["stay_id"], "hour_offset": h})
    if abx_rows:
        abx_flag = pd.DataFrame(abx_rows).drop_duplicates()
        abx_flag["antibiotic_flag"] = 1
        grid = grid.merge(abx_flag, on=["stay_id", "hour_offset"], how="left")
        grid["antibiotic_flag"] = grid["antibiotic_flag"].fillna(0).astype(int)
    else:
        grid["antibiotic_flag"] = 0

    log(f"  Med flags added: vasopressor, antibiotic")
    return grid


def add_derived_features(grid: pd.DataFrame) -> pd.DataFrame:
    """
    Compute 4 derived features per row (Section 2.5):
      - pf_ratio   : PaO2 / FiO2 (= pao2 / (fio2/100) if fio2 in %)
      - map_delta  : MAP change from previous hour
      - rr_trend   : resp_rate change from previous hour
      gcs_total is already derived in vitals step.
    """
    log("Computing derived features …")
    grid = grid.sort_values(["stay_id", "hour_offset"]).copy()

    # PaO₂/FiO₂ ratio
    if "pao2" in grid.columns and "fio2" in grid.columns:
        fio2 = grid["fio2"].copy()
        # FiO2 stored as fraction (0–1) or percent (1–100) — normalise to fraction
        mask_pct = fio2 > 1.0
        fio2[mask_pct] = fio2[mask_pct] / 100.0
        fio2 = fio2.replace(0, np.nan)
        grid["pf_ratio"] = grid["pao2"] / fio2
    else:
        grid["pf_ratio"] = np.nan

    # MAP delta (hour-over-hour)
    if "map" in grid.columns:
        grid["map_delta"] = grid.groupby("stay_id")["map"].diff()
    else:
        grid["map_delta"] = np.nan

    # Resp rate trend (hour-over-hour)
    if "resp_rate" in grid.columns:
        grid["rr_trend"] = grid.groupby("stay_id")["resp_rate"].diff()
    else:
        grid["rr_trend"] = np.nan

    log("  Derived features: pf_ratio, map_delta, rr_trend")
    return grid


# =============================================================================
# Main
# =============================================================================


def main():
    log("=" * 60)
    log("Stage 00 — MIMIC-IV Data Extraction")
    log("=" * 60)
    log(f"Log file → {_LOG_PATH}")

    cohort_out  = cfg.PROCESSED_DIR / "cohort_base.parquet"
    chart_out   = cfg.PROCESSED_DIR / "chartevents_filtered.parquet"
    lab_out     = cfg.PROCESSED_DIR / "labevents_filtered.parquet"
    vaso_out    = cfg.PROCESSED_DIR / "vasopressor_events.parquet"
    abx_out     = cfg.PROCESSED_DIR / "antibiotic_events.parquet"
    grid_out    = cfg.PROCESSED_DIR / "hourly_features.parquet"

    # ── Step 1: Base cohort (fast — always rerun if needed) ──────────────────
    if cohort_out.exists():
        log(f"\n[Step 1] ✓ Loading existing cohort_base.parquet …")
        cohort = pd.read_parquet(cohort_out)
        log(f"  Cohort: {len(cohort):,} stays")
    else:
        log("\n[Step 1] Building base cohort …")
        cohort_raw = load_base_cohort()
        cohort     = apply_cohort_criteria(cohort_raw)
        cohort.to_parquet(cohort_out, index=False)
        log(f"  Saved → {cohort_out}  ({len(cohort):,} stays)")

    # ── Steps 2–5: DuckDB heavy scans (skip if already done) ─────────────────
    need_duckdb = not all(p.exists() for p in [chart_out, lab_out, vaso_out, abx_out])

    if need_duckdb:
        con = init_duckdb()
        stay_ids = cohort["stay_id"].tolist()
        hadm_ids = cohort["hadm_id"].tolist()
        con.execute("CREATE TABLE IF NOT EXISTS tmp_stays  (stay_id INTEGER)")
        con.execute("CREATE TABLE IF NOT EXISTS tmp_hadms  (hadm_id INTEGER)")
        con.execute("DELETE FROM tmp_stays")
        con.execute("DELETE FROM tmp_hadms")
        con.executemany("INSERT INTO tmp_stays VALUES (?)", [[s] for s in stay_ids])
        con.executemany("INSERT INTO tmp_hadms VALUES (?)", [[h] for h in hadm_ids])
    else:
        con = None
        log("\n[Steps 2–5] ✓ All filtered parquets already exist — skipping DuckDB scans")

    # Step 2
    if chart_out.exists():
        log(f"\n[Step 2] ✓ Loading chartevents_filtered.parquet …")
        chart_long = pd.read_parquet(chart_out)
        log(f"  {len(chart_long):,} rows")
    else:
        log("\n[Step 2] Extracting chartevents (vitals + GCS) …")
        chart_long = extract_chartevents(cohort, con)
        chart_long.to_parquet(chart_out, index=False)
        log(f"  Saved → {chart_out}")

    # Step 3
    if lab_out.exists():
        log(f"\n[Step 3] ✓ Loading labevents_filtered.parquet …")
        lab_long = pd.read_parquet(lab_out)
        log(f"  {len(lab_long):,} rows")
    else:
        log("\n[Step 3] Extracting labevents …")
        lab_long = extract_labevents(cohort, con)
        lab_long.to_parquet(lab_out, index=False)
        log(f"  Saved → {lab_out}")

    # Step 4
    if vaso_out.exists():
        log(f"\n[Step 4] ✓ Loading vasopressor_events.parquet …")
        vaso_df = pd.read_parquet(vaso_out)
        log(f"  {len(vaso_df):,} rows")
    else:
        log("\n[Step 4] Extracting vasopressor flags …")
        vaso_df = extract_vasopressors(cohort, con)
        vaso_df.to_parquet(vaso_out, index=False)
        log(f"  Saved → {vaso_out}")

    # Step 5
    if abx_out.exists():
        log(f"\n[Step 5] ✓ Loading antibiotic_events.parquet …")
        abx_df = pd.read_parquet(abx_out)
        log(f"  {len(abx_df):,} rows")
    else:
        log("\n[Step 5] Extracting antibiotic flags …")
        abx_df = extract_antibiotics(cohort, con)
        abx_df.to_parquet(abx_out, index=False)
        log(f"  Saved → {abx_out}")

    if con is not None:
        con.close()

    # ── Step 6: Build hourly grid + pivot ─────────────────────────────────────
    log("\n[Step 6] Building hourly feature grid …")
    grid = build_hourly_grid(cohort)
    grid = add_vital_to_grid(grid, chart_long, cohort)
    grid = add_labs_to_grid(grid, lab_long, cohort)
    grid = add_med_flags_to_grid(grid, vaso_df, abx_df, cohort)
    grid = add_derived_features(grid)

    # ── Summary ───────────────────────────────────────────────────────────────
    all_feat_cols = (
        list(cfg.VITAL_ITEMIDS.keys())
        + list(cfg.LAB_ITEMIDS.keys())
        + ["gcs_total", "pf_ratio", "map_delta", "rr_trend",
           "vasopressor_flag", "antibiotic_flag"]
    )
    existing_feats = [c for c in all_feat_cols if c in grid.columns]
    log(f"\n  Feature columns present ({len(existing_feats)}/25): {existing_feats}")

    miss_rate = grid[existing_feats].isnull().mean().sort_values(ascending=False)
    log("\n  Missing-value rates per feature:")
    for feat, rate in miss_rate.items():
        log(f"    {feat:20s}: {rate:.1%}")

    grid.to_parquet(grid_out, index=False)
    log(f"\n  Saved → {grid_out}")
    log(f"  Shape: {grid.shape}")

    # ── Vital completeness filter (Section 2.3) ───────────────────────────────
    log("\n[Post-filter] Vital-sign completeness < 50% …")
    vital_cols = [c for c in cfg.VITAL_ITEMIDS.keys() if c in grid.columns]
    per_stay_completeness = (
        grid.groupby("stay_id")[vital_cols]
        .apply(lambda df: df.notna().mean(axis=1).mean())
    )
    low_complete = per_stay_completeness[per_stay_completeness < cfg.VITAL_COMPLETENESS_MIN].index
    log(f"  Stays with vital completeness < {cfg.VITAL_COMPLETENESS_MIN:.0%}: {len(low_complete):,}")
    grid   = grid[~grid["stay_id"].isin(low_complete)].copy()
    cohort = cohort[~cohort["stay_id"].isin(low_complete)].copy()
    log(f"  Remaining stays: {grid['stay_id'].nunique():,}")

    grid.to_parquet(grid_out, index=False)
    cohort.to_parquet(cohort_out, index=False)
    log(f"\n  Final hourly_features.parquet: {grid.shape}")
    log(f"  Final cohort_base.parquet:      {len(cohort):,} stays")

    log("\n[Stage 00 COMPLETE] ✓")
    log("  → Run next: python src/01_labeling.py")


if __name__ == "__main__":
    main()


