# GWO-Optimized Knowledge Distillation for Lightweight Sepsis Detection

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Dataset: MIMIC-IV v3.1](https://img.shields.io/badge/Dataset-MIMIC--IV%20v3.1-green.svg)](https://physionet.org/content/mimiciv/3.1/)
[![External Validation: eICU-CRD v2.0](https://img.shields.io/badge/External-eICU--CRD%20v2.0-teal.svg)](https://physionet.org/content/eicu-crd/2.0/)

Official implementation of the manuscript:  
**"GWO-Optimized Knowledge Distillation for Lightweight Sepsis Detection: Compressing Temporal Deep Models for Resource-Constrained Clinical Deployment"**

**Authors:** Windra Swastika¹, Heri Kristianto²  
¹ *Informatics Engineering, Faculty of Technology and Design, Universitas Ma Chung, Malang, Indonesia*  
² *Nursing Department, Faculty of Health Science, Universitas Brawijaya, Malang, Indonesia*  
*Corresponding author: [windra.swastika@machung.ac.id](mailto:windra.swastika@machung.ac.id)*

---

## Overview

Deep learning models for early sepsis prediction in the Intensive Care Unit (ICU) often suffer from high computational overhead, limiting their deployment on resource-constrained bedside and edge hardware. This repository implements a **Grey Wolf Optimizer (GWO)-guided Knowledge Distillation (GWO-KD)** framework that compresses a Temporal Convolutional Network (TCN) teacher into an ultra-lightweight 1D-CNN student for real-time sepsis screening.

### Key Highlights
* **26x Parameter Compression:** Compresses a 105,473-parameter TCN teacher (0.402 MB) into a 4,001-parameter 1D-CNN student (0.0153 MB).
* **Sub-Millisecond Edge Inference:** Achieves **0.073 ms** single-sample CPU latency (exceeding standard <1 MB and <10 ms edge AI thresholds by 65x and 137x).
* **Large-Scale Cohort & Zero-Shot External Validation:** Developed on **84,881 adult ICU stays** (6.66M sliding windows) from **MIMIC-IV v3.1** using hybrid Sepsis-3 labeling, and evaluated zero-shot on **165,524 stays** across 109 hospitals from **eICU-CRD v2.0**.
* **Comprehensive Benchmarking:** Includes Pareto backbone screening (5 lightweight architectures), comparative metaheuristics (GWO vs. PSO, GA, WOA), classical KD methods (Vanilla KD, FitNet, Attention Transfer), SOTA ML/DL baselines (XGBoost, LightGBM, LSTM, Simple TCN, RETAIN), and clinical scores (SOFA, MEWS, NEWS2).
* **Transparent Empirical Ablation:** Reports rigorous statistical analyses (DeLong's test, Wilcoxon signed-rank, Holm-Bonferroni corrections) showing that while the compact student excels in efficiency, direct hard-label (vanilla) supervision slightly outperforms distilled training across candidate backbones ($p < 0.01$).

---

## Methodology & Pipeline

The framework proceeds through six sequential stages:
1. **Data Extraction:** Cohort construction from MIMIC-IV v3.1 (`chartevents`, `labevents`, `diagnoses_icd`, `admissions`, `patients`, `icustays`, `prescriptions`).
2. **Sepsis-3 Hybrid Labeling:** Combines ICD-9/10 codes (Method A) and clinical criteria of suspected infection + SOFA $\ge 2$ increase (Method B), yielding an 8.52% positive prevalence (7,235 / 84,881 stays).
3. **Sliding-Window Feature Engineering:** 12-hour sliding windows (1-hour step) across 21 multivariate clinical features ($\mathbf{x} \in \mathbb{R}^{12 \times 21}$):
   * *7 Vital Signs:* HR, SBP, DBP, MAP, Resp Rate, $\text{SpO}_2$, Core Temp.
   * *8 Lab Values:* WBC, Creatinine, Lactate, Bilirubin, Platelets, INR, $\text{PaO}_2$, $\text{FiO}_2$.
   * *4 Derived Features:* GCS Total, $\text{PaO}_2/\text{FiO}_2$ ratio, MAP delta, Resp Rate trend.
   * *2 Medication Flags:* Antibiotic administration, Vasopressor use.
4. **Teacher Pre-Training:** Evaluates BiLSTM+Attention vs. Dilated Causal TCN (4 residual blocks, dilations $d \in \{1, 2, 4, 8\}$).
5. **GWO-Guided Knowledge Distillation:** Searches over $\mathbf{X} = (T, \alpha_{\text{KD}}, \beta_{\text{KD}}, s)$ to minimize the composite response- and feature-based loss:

$$\mathcal{L}_{\text{total}} = \alpha_{\text{KD}} \mathcal{L}_{\text{CE}}(\hat{y}_s, y) + (1 - \alpha_{\text{KD}}) T^2 \mathcal{L}_{\text{KL}}(q_s \parallel q_t) + \beta_{\text{KD}} \mathcal{L}_{\text{feat}}(r(f_s), f_t)$$

6. **Evaluation & Zero-Shot Transfer:** Internal test set benchmarking and zero-shot external validation on eICU-CRD v2.0.

---

## Key Results

### 1. Model Efficiency & Predictive Performance (MIMIC-IV Test Set)

| Model | Role / Type | Params | Size (MB) | CPU Latency (ms) | Test AUROC |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **TCN** | Frozen Teacher | 105,473 | 0.4020 | 0.604 | 0.7429 |
| **BiLSTM + Attn** | Teacher Candidate | 1,574,785 | 6.0070 | 0.912 | 0.7013 |
| **GWO-KD (S2: 1D-CNN)** | **Proposed Student** | **4,001** | **0.0153** | **0.073** | **0.7594** |
| **S2 (1D-CNN, Vanilla)** | Screened Student | 4,001 | 0.0153 | 0.073 | 0.7620 |
| **S3 (CompactGRU, Vanilla)** | Screened Student | 7,697 | 0.0294 | 0.174 | 0.7602 |
| **S1 (TinyLSTM, Vanilla)** | Screened Student | 7,073 | 0.0270 | 0.110 | 0.7585 |
| **XGBoost** | ML Baseline | N/A | N/A | N/A | 0.7764 |
| **RETAIN** | DL Baseline | 55,618 | 0.2120 | 0.264 | 0.7564 |
| **Vanilla LSTM** | DL Baseline | 22,337 | 0.0850 | 0.119 | 0.7540 |
| **Simple TCN** | DL Baseline | 5,313 | 0.0200 | 0.082 | 0.7488 |
| **LightGBM** | ML Baseline | N/A | N/A | N/A | 0.7343 |
| **SOFA / MEWS / NEWS2** | Clinical Scores | 0 | ~0 | ~0 | 0.6235 / 0.6068 / 0.6057 |

### 2. Vanilla vs. Distilled Student Ablation (DeLong's Test)
Our ablation demonstrates that while the 1D-CNN student achieves superior efficiency and beats the teacher, adding soft-label and feature distillation mildly reduces test AUROC compared to direct hard-label training across all three candidate architectures:

| Architecture | Vanilla AUROC | Distilled AUROC | $\Delta$ AUROC | DeLong $p$-value |
| :--- | :---: | :---: | :---: | :---: |
| **S1 (TinyLSTM)** | 0.7585 | 0.7556 | -0.0028 | 0.0016 |
| **S2 (1D-CNN)** | 0.7620 | 0.7594 | -0.0026 | 0.0028 |
| **S3 (CompactGRU)** | 0.7602 | 0.7575 | -0.0027 | 0.0021 |

---

## Repository Structure

```text
GWO-Optimized-KD-MIMIC-IV/
├── 00_extract.py                   # Stage 0: MIMIC-IV v3.1 cohort & hourly feature extraction (DuckDB)
├── 01_labeling.py                  # Stage 1: Sepsis-3 hybrid labeling (ICD OR SOFA ≥2 + suspected infection)
├── 02_features.py                  # Stage 2: Imputation, clipping, Z-score, 12-h sliding windows
├── 03_split.py                     # Stage 3: Patient-level 70/15/15 stratified split
├── 04_teacher.py                   # Stage 4: Teacher training & selection (BiLSTM+Attn vs TCN)
├── 05_screening.py                 # Stage 5: Student backbone screening (Pareto: AUROC, size, latency)
├── 06_gwo_kd.py                    # Stage 6: GWO search over (T, α_KD, β_KD, s) + full distillation
├── 07_baselines.py                 # Stage 7: Metaheuristic (PSO, GA, WOA) & KD baselines (Vanilla KD, FitNet, AT)
├── 07_recovery.py                  # Stage 7 resume: completes interrupted WOA seeds, Wilcoxon & DeLong tests
├── 08_sota_baselines.py            # Stage 8: XGBoost, LightGBM, Vanilla LSTM, Simple TCN, RETAIN
├── 09_ablation.py                  # Stage 9: Ablations (KD weights/temperature, loss components, GWO config)
├── 09c_gwo_remaining.py            # Stage 9(c): remaining GWO configuration grid (N_w, I_max)
├── 10_evaluate.py                  # Stage 10: Final results tables, ROC/PR curves, eICU external validation
├── eicu_extract.py                 # eICU-CRD v2.0 extraction for zero-shot external validation
├── clinical_baselines.py           # Rule-based SOFA, MEWS, NEWS2 scores on the same test split
├── delong_clinical_baselines.py    # DeLong test: GWO-KD vs SOFA / MEWS / NEWS2
├── label_sensitivity.py            # Label sensitivity: ICD-only vs clinical-only vs hybrid
├── finish_ablation_c.py            # Remaining GWO configuration ablation points (N_w=30, I_max=20/50)
├── benchmark_speedup.py            # Micro-benchmark of GWO speedup strategies (device, parallelism, DataLoader)
├── config.py                       # Global paths, feature lists, hyperparameter bounds, seeds
├── __init__.py
├── models/
│   ├── teacher.py                  # BiLSTM + temporal self-attention and dilated TCN teachers
│   └── students.py                 # S1 (TinyLSTM), S2 (1D-CNN), S3 (CompactGRU), S4, S5
├── utils/
│   ├── __init__.py
│   ├── data_utils.py               # Dataset/DataLoader helpers with class-weighted sampling
│   ├── gwo.py                      # Grey Wolf Optimizer (Algorithm 1)
│   └── metrics.py                  # AUROC, AUPRC, bootstrap CI, DeLong, Wilcoxon, Holm–Bonferroni
└── README.md
```

Scripts are numbered in execution order (`00` → `10`). Data paths are set in `config.py`; raw MIMIC-IV (`icu/`, `hosp/`) and eICU-CRD (`eicu-crd/2.0/`) tables are expected one level above this folder, and outputs are written to `processed/`, `splits/`, `checkpoints/`, `results/`, and `logs/`.

---

## Getting Started

### 1. Prerequisites & Data Access
Both **MIMIC-IV v3.1** and **eICU-CRD v2.0** are restricted-access clinical databases hosted on PhysioNet. To run the data extraction pipeline:
1. Complete the required CITI "Data or Specimens Only Research" training.
2. Sign the Data Use Agreement (DUA) on PhysioNet for [MIMIC-IV v3.1](https://physionet.org/content/mimiciv/3.1/) and [eICU-CRD v2.0](https://physionet.org/content/eicu-crd/2.0/).
3. Download the raw CSV/Parquet tables into your local `data/raw/` directory.

### 2. Installation
Clone the repository and install the dependencies:

```bash
git clone [https://github.com/windraswastika/GWO-Optimized-KD-MIMIC-IV.git](https://github.com/windraswastika/GWO-Optimized-KD-MIMIC-IV.git)
cd GWO-Optimized-KD-MIMIC-IV
pip install -r requirements.txt
```

### 3. Reproducing the Experiments

**Step 1: Cohort Extraction & Feature Engineering**
```bash
python data_pipeline/extract_mimic_iv.py --input_dir /path/to/mimic-iv-3.1 --output_dir data/processed/mimic
python data_pipeline/feature_engineering.py --data_dir data/processed/mimic --window_size 12 --step 1 --seed 42
```

**Step 2: Train Frozen Teacher Model (TCN)**
```bash
python distillation/trainer.py --phase train_teacher --teacher tcn --seed 42
```

**Step 3: Run GWO-Guided Knowledge Distillation**
```bash
python distillation/trainer.py --phase gwo_search --n_wolves 10 --max_iter 10 --seeds 5
python distillation/trainer.py --phase full_distill --arch s2_1dcnn --temp 10.12 --alpha 0.90 --beta 0.368 --seed 42
```

**Step 4: Evaluate & Run Ablations / External Validation**
```bash
python evaluation/metrics.py --model_path checkpoints/gwo_kd_s2_best.pt --test_split internal
python evaluation/metrics.py --model_path checkpoints/gwo_kd_s2_best.pt --test_split eicu_external
```

---

## Reproducibility & Ethics Statement

* **Leakage Prevention:** All splits (70% train / 15% validation / 15% test; 59,416 / 12,732 / 12,733 stays) are strictly **patient-disjoint**. Clipping bounds (1st/99th percentiles), median imputation values, and Z-score normalization parameters are fit exclusively on the MIMIC-IV training split and applied unchanged to the internal test and eICU-CRD external cohorts.
* **Hardware Environment:** Model training and metaheuristic searches were executed on an NVIDIA A100 GPU with deterministic cuDNN (`seed=42`). Edge CPU inference latencies were benchmarked on an Apple M4 CPU.
* **Ethical Approval:** Ethical approval for this study was obtained from the Health Research Ethics Committee of the Faculty of Health Sciences, Universitas Brawijaya (Approval No. `00027/DST/UN10.F1716/B/PT/2026`).

---

## Citation

If you find this repository or our empirical findings useful in your research, please cite our manuscript:

```bibtex
@article{swastika2026gwokd,
  title={GWO-Optimized Knowledge Distillation for Lightweight Sepsis Detection: Compressing Temporal Deep Models for Resource-Constrained Clinical Deployment},
  author={Swastika, Windra and Kristianto, Heri},
  journal={Manuscript submitted for publication},
  year={2026},
  url={[https://github.com/windraswastika/GWO-Optimized-KD-MIMIC-IV](https://github.com/windraswastika/GWO-Optimized-KD-MIMIC-IV)}
}
```

## License

This project is licensed under the MIT License — see the `LICENSE` file for details.
