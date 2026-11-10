"""
config.py — Global configuration for GWO-KD Sepsis Prediction Pipeline
Based on Methodology_Revised.md (Section 2.0 – 2.16)

Hardware: Apple M4, 24 GB RAM, MPS (Metal Performance Shaders) for PyTorch
"""

from pathlib import Path

# =============================================================================
# Project paths
# =============================================================================
BASE_DIR = Path(__file__).resolve().parent.parent   # Riset MIMIC GWO/

# Raw MIMIC-IV data
MIMIC_ICU_DIR  = BASE_DIR / "icu"
MIMIC_HOSP_DIR = BASE_DIR / "hosp"

# Raw eICU-CRD data
EICU_DIR = BASE_DIR / "eicu-crd" / "2.0"

# Output directories
PROCESSED_DIR  = BASE_DIR / "processed"
SPLITS_DIR     = BASE_DIR / "splits"
CHECKPOINTS_DIR= BASE_DIR / "checkpoints"
RESULTS_DIR    = BASE_DIR / "results"
LOGS_DIR       = BASE_DIR / "logs"

# Ensure output dirs exist
for _d in [PROCESSED_DIR, SPLITS_DIR, CHECKPOINTS_DIR, RESULTS_DIR, LOGS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# =============================================================================
# MIMIC-IV source files
# =============================================================================
MIMIC_FILES = {
    "icustays"          : MIMIC_ICU_DIR  / "icustays.csv",
    "chartevents"       : MIMIC_ICU_DIR  / "chartevents.csv",
    "inputevents"       : MIMIC_ICU_DIR  / "inputevents.csv",
    "outputevents"      : MIMIC_ICU_DIR  / "outputevents.csv",
    "admissions"        : MIMIC_HOSP_DIR / "admissions.csv",
    "patients"          : MIMIC_HOSP_DIR / "patients.csv",
    "diagnoses_icd"     : MIMIC_HOSP_DIR / "diagnoses_icd.csv",
    "labevents"         : MIMIC_HOSP_DIR / "labevents.csv",
    "prescriptions"     : MIMIC_HOSP_DIR / "prescriptions.csv",
    "microbiologyevents": MIMIC_HOSP_DIR / "microbiologyevents.csv",
}

# eICU-CRD source files
EICU_FILES = {
    "patient"             : EICU_DIR / "patient.csv.gz",
    "vitalPeriodic"       : EICU_DIR / "vitalPeriodic.csv.gz",
    "vitalAperiodic"      : EICU_DIR / "vitalAperiodic.csv.gz",
    "lab"                 : EICU_DIR / "lab.csv.gz",
    "diagnosis"           : EICU_DIR / "diagnosis.csv.gz",
    "medication"          : EICU_DIR / "medication.csv.gz",
    "infusionDrug"        : EICU_DIR / "infusionDrug.csv.gz",
    "apachePredVar"       : EICU_DIR / "apachePredVar.csv.gz",
    "microLab"            : EICU_DIR / "microLab.csv.gz",
    "nurseCharting"       : EICU_DIR / "nurseCharting.csv.gz",
    "respiratoryCharting" : EICU_DIR / "respiratoryCharting.csv.gz",
}

# =============================================================================
# eICU hospital selection criteria (Section 2.2.2)
# Concrete, measurable criteria — addresses reviewer concern about vague "k hospitals"
# =============================================================================
# Criterion 1: Minimum ICU stay volume (ensures stable feature distributions)
EICU_MIN_STAYS_PER_HOSPITAL = 500

# Criterion 2: Vital sign completeness per hospital
# (fraction of patient-hours with at least one vital sign recorded)
EICU_MIN_VITAL_COVERAGE = 0.70   # ≥70% hourly coverage

# Criterion 3: Minimum number of hospitals to include
# (ensures geographic/institutional diversity)
EICU_MIN_HOSPITALS = 10

# Criterion 4: Patient inclusion (same as MIMIC-IV)
EICU_MIN_AGE = 18
EICU_MIN_LOS_HOURS = 4
EICU_VITAL_COMPLETENESS_MIN = 0.50  # same as MIMIC-IV

# Output directory for eICU processed data
EICU_PROCESSED_DIR = PROCESSED_DIR / "eicu"
EICU_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# MIMIC-IV item IDs for the 25 features (Section 2.5)
# =============================================================================

# --- 7 Vital signs ---
VITAL_ITEMIDS = {
    "heart_rate"   : [220045],
    "sbp"          : [220179, 225309, 224167],   # systolic BP (arterial + cuff)
    "dbp"          : [220180, 225310, 224643],   # diastolic BP
    "map"          : [220052, 225312, 224322],   # mean arterial pressure
    "resp_rate"    : [220210, 224690],
    "spo2"         : [220277],
    "temperature"  : [223761, 223762],           # F and C, will convert to C
}

# --- 8 Laboratory values (from labevents) ---
LAB_ITEMIDS = {
    "wbc"          : [51300, 51301],
    "creatinine"   : [50912],
    "lactate"      : [50813],
    "bilirubin"    : [50885],
    "platelets"    : [51265],
    "inr"          : [51237],
    "pao2"         : [50821],
    "fio2"         : [50816],
}

# --- GCS components (for derived GCS total, Section 2.5) ---
GCS_ITEMIDS = {
    "gcs_eye"    : [220739],
    "gcs_verbal" : [223900],
    "gcs_motor"  : [223901],
}

# --- Medication flags (from inputevents / prescriptions) ---
# Antibiotics: broad list of antibiotic drug names (prescriptions.drug)
ANTIBIOTIC_KEYWORDS = [
    "vancomycin", "piperacillin", "meropenem", "cefepime", "ceftriaxone",
    "ciprofloxacin", "levofloxacin", "metronidazole", "ampicillin",
    "gentamicin", "tobramycin", "azithromycin", "clindamycin", "linezolid",
    "daptomycin", "fluconazole", "micafungin", "caspofungin",
    "ertapenem", "cefazolin", "nafcillin", "oxacillin", "trimethoprim",
    "sulfamethoxazole", "rifampin", "colistin", "tigecycline",
]

# Vasopressors: inputevents itemids
VASOPRESSOR_ITEMIDS = [
    221906,  # norepinephrine
    221289,  # epinephrine
    222315,  # vasopressin
    221749,  # phenylephrine
    221662,  # dopamine
    227692,  # dobutamine
]

# All chartevents itemids to extract (union of vitals + GCS)
CHART_ITEMIDS = (
    [iid for ids in VITAL_ITEMIDS.values() for iid in ids]
    + [iid for ids in GCS_ITEMIDS.values()  for iid in ids]
)

# All labevents itemids to extract
LAB_ITEMIDS_FLAT = [iid for ids in LAB_ITEMIDS.values() for iid in ids]

# =============================================================================
# Sepsis-3 ICD codes (Section 2.4)
# =============================================================================
SEPSIS_ICD9 = ["99591", "99592", "78552"]           # ICD-9 (no dot)
SEPSIS_ICD10_PREFIXES = ["A40", "A41", "R6520", "R6521"]

# =============================================================================
# Cohort inclusion/exclusion criteria (Section 2.3)
# =============================================================================
MIN_AGE_YEARS   = 18
MIN_LOS_HOURS   = 4          # minimum ICU length of stay
MIN_CHART_HOURS = 2          # chartevents present for at least first N hours
VITAL_COMPLETENESS_MIN = 0.50  # exclude if vital-sign completeness < 50 %

# =============================================================================
# Feature engineering (Section 2.5)
# =============================================================================
WINDOW_SIZE_HOURS = 12        # sliding window length
WINDOW_STEP_HOURS = 1         # step size
N_FEATURES        = 25        # total feature count (before missingness indicators)

# Forward-fill limits (in hours)
FFILL_VITALS_HOURS = 2
FFILL_LABS_HOURS   = 24

OUTLIER_CLIP_PERCENTILE_LOW  = 1
OUTLIER_CLIP_PERCENTILE_HIGH = 99

# =============================================================================
# Train/val/test split (Section 2.5)
# =============================================================================
SPLIT_TRAIN = 0.70
SPLIT_VAL   = 0.15
SPLIT_TEST  = 0.15
RANDOM_SEED = 42

# =============================================================================
# Teacher training (Section 2.6)
# =============================================================================
TEACHER_LR            = 1e-3
TEACHER_BATCH_SIZE    = 256
TEACHER_MAX_EPOCHS    = 100
TEACHER_PATIENCE      = 10     # early stopping on val AUROC
TEACHER_ADAM_BETAS    = (0.9, 0.999)

# BiLSTM architecture
BILSTM_HIDDEN1 = 256
BILSTM_HIDDEN2 = 128
BILSTM_FC_DIMS = [256, 64, 1]

# TCN architecture
TCN_N_BLOCKS     = 4
TCN_KERNEL_SIZE  = 3
TCN_FILTERS      = 64
TCN_DILATIONS    = [1, 2, 4, 8]

# =============================================================================
# Student architectures (Section 2.7.1)
# =============================================================================
STUDENT_DROPOUT = 0.2

# S1 — TinyLSTM
TINYLSTM_HIDDEN = 32

# S2 — 1D-CNN
CNN_FILTERS = [32, 16, 8]
CNN_KERNEL   = 3

# S3 — CompactGRU
COMPACT_GRU_HIDDEN1 = 32
COMPACT_GRU_HIDDEN2 = 16

# S4 — 1D-MobileNet (depthwise-separable), screened to justify exclusion (Section 2.7.2)
MOBILENET_FILTERS = [32, 16, 8]
MOBILENET_KERNEL  = 3

# S5 — Pruned LightGRU, screened to justify exclusion (Section 2.7.2)
# Trained dense at a larger width than S3, then magnitude-pruned + fine-tuned.
LIGHTGRU_DENSE_HIDDEN1 = 48
LIGHTGRU_DENSE_HIDDEN2 = 24
LIGHTGRU_PRUNE_SPARSITY = 0.6     # fraction of weights zeroed (global L1 unstructured)
LIGHTGRU_FINETUNE_EPOCHS = 10     # post-prune recovery fine-tuning

# Edge deployment constraints
MAX_MODEL_SIZE_MB  = 1.0
MAX_LATENCY_MS     = 10.0

# =============================================================================
# GWO search space (Section 2.8)
# =============================================================================
GWO_BOUNDS = {
    "T"      : (1.0,  15.0),   # distillation temperature
    "alpha_kd": (0.1,  0.90),  # BCE weight
    "beta_kd" : (0.01, 0.50),  # feature-alignment weight
    "s"       : (1.0,  3.0),   # student architecture index (rounded at eval)
}

GWO_N_WOLVES    = 10
GWO_I_MAX       = 10
GWO_EPSILON     = 1e-4    # early-stop threshold
GWO_PATIENCE    = 5       # consecutive iterations without improvement

# Metaheuristic seeds (≥5 per Section 2.16)
META_SEEDS = [0, 1, 2, 3, 4]

# =============================================================================
# Ablation programme (Section 2.13)
# =============================================================================
# (a) Univariate sweeps around the GWO-optimal point (others held fixed)
ABLATION_ALPHA_KD_GRID = [0.1, 0.3, 0.5, 0.7, 0.9]
ABLATION_BETA_KD_GRID  = [0.01, 0.05, 0.1, 0.234, 0.5]
ABLATION_T_GRID        = [1, 3, 5, 9, 12, 15]

# (c) GWO configuration ablation — reduced to a univariate design (bare-minimum
# scope, agreed 2026-09-02): fix I_max=10 and vary N_w, then fix N_w=10 and vary
# I_max, rather than the full 4x3 cross-product (which would be ~52x the main
# search's evaluation budget at 5 seeds each). The (N_w=10, I_max=10) point is
# reused from the existing main GWO run rather than rerun.
ABLATION_GWO_NW_GRID    = [5, 10, 20, 30]     # at fixed I_max=10
ABLATION_GWO_IMAX_GRID  = [10, 20, 50]        # at fixed N_w=10

# =============================================================================
# Knowledge distillation (Section 2.9)
# =============================================================================
KD_MAX_EPOCHS    = 100
KD_FINETUNE_LR   = 1e-4
KD_FINETUNE_EPOCHS = 10
KD_BATCH_SIZE    = 256

# =============================================================================
# Evaluation (Section 2.14–2.15)
# =============================================================================
BOOTSTRAP_N_RESAMPLES = 1000
PVALUE_THRESHOLD      = 0.05

# Dual-threshold evaluation (Section 2.14)
# Addresses reviewer concern: all threshold-dependent metrics reported at
# TWO operating points so conclusions are not sensitive to arbitrary choice.
#
#   1. Youden's Index  — maximises (Sensitivity + Specificity - 1)
#      Standard statistical operating point for ROC analysis.
#
#   2. Clinical Sensitivity Target — early warning system must catch ≥80% sepsis
#      Clinically motivated; equivalent to SOFA/NEWS2 early-warning convention.
EVAL_SENSITIVITY_TARGET = 0.80   # for clinical threshold selection

# Label sensitivity analysis
# Strategies evaluated: 'icd', 'clinical', 'hybrid'
LABEL_SENSITIVITY_STRATEGIES = ["icd", "clinical", "hybrid"]

# =============================================================================
# Hardware & I/O
# =============================================================================
# M4 Mac: MPS backend for PyTorch, all efficiency metrics on CPU
DEVICE_TRAINING  = "cuda"  # NVIDIA A100
DEVICE_EFFICIENCY= "cpu"   # latency measured on CPU only (Section 2.16)

# DuckDB / chunked reading
DUCKDB_MEMORY_LIMIT = "60GB"   # A100 server
CHUNK_SIZE = 500_000           # rows per chunk for large CSV reads

# Logging
LOG_INTERVAL_EPOCHS = 5

print("[config] Configuration loaded.")
