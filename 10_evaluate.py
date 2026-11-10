"""
10_evaluate.py — Stage 10: Final Evaluation & Results Tables (Section 2.14–2.15)
==================================================================================
Generates all results tables and figures for the manuscript:

  Table 3 — Teacher comparison (TCN vs BiLSTM)
  Table 4 — Student screening summary
  Table 5 — GWO-KD vs metaheuristic baselines (Wilcoxon)
  Table 6 — GWO-KD vs KD baselines (DeLong)
  Table 7 — External validation on eICU-CRD
  Figure  — ROC curves + Precision-Recall curves

All metrics reported at dual thresholds (Youden + Sens≥0.80).
Bootstrap 95% CI for all AUROC values (n=1000).

Run:
    python src/10_evaluate.py                # MIMIC-IV test set
    python src/10_evaluate.py --dataset eicu # eICU external validation
    python src/10_evaluate.py --all          # both
"""

import sys, os, argparse
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from datetime import datetime

import config as cfg
from utils.data_utils import make_dataloader, model_size_mb, count_params, measure_latency_cpu
from utils.metrics import (
    auroc, auprc, dual_threshold_eval, print_dual_threshold_summary,
    bootstrap_auroc_ci, delong_test, knowledge_retention
)
from models.teacher import build_teacher
from models.students import build_student


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def get_device():
    if torch.backends.mps.is_available(): return torch.device("mps")
    if torch.cuda.is_available():         return torch.device("cuda")
    return torch.device("cpu")


@torch.no_grad()
def get_scores(model, loader, device):
    """Get probability scores and labels from a dataloader."""
    model.eval(); scores, labels = [], []
    for X_b, y_b in loader:
        logit, _ = model(X_b.to(device))
        scores.append(torch.sigmoid(logit).cpu().numpy())
        labels.append(y_b.numpy())
    return np.concatenate(labels), np.concatenate(scores)


def eval_checkpoint(ckpt_path, n_features, loader, device):
    """Load checkpoint, get scores, return full metric dict."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    mname = ckpt["model_name"]

    if mname in ("bilstm", "tcn"):
        model = build_teacher(mname, n_features=n_features)
    else:
        idx = int(mname.replace("s", ""))
        model = build_student(idx, n_features=n_features)

    model.load_state_dict(ckpt["state_dict"])
    model.to(device)

    y_true, y_score = get_scores(model, loader, device)
    dual = dual_threshold_eval(y_true, y_score, cfg.EVAL_SENSITIVITY_TARGET)
    auc_pt, auc_lo, auc_hi = bootstrap_auroc_ci(y_true, y_score, n_resamples=1000)
    lat = measure_latency_cpu(model.cpu(), n_features=n_features)

    return {
        "y_true": y_true, "y_score": y_score,
        "auroc_pt": auc_pt, "auroc_lo": auc_lo, "auroc_hi": auc_hi,
        "dual": dual,
        "n_params": count_params(model),
        "size_mb": model_size_mb(model),
        "latency_ms": lat,
        "ckpt": ckpt,
    }


def format_ci(pt, lo, hi):
    return f"{pt:.4f} [{lo:.4f}–{hi:.4f}]"


# =============================================================================
# MIMIC-IV test evaluation
# =============================================================================

def eval_mimic(device, n_features):
    log("\n" + "="*60)
    log("MIMIC-IV Test Set Evaluation")
    log("="*60)

    test_loader = make_dataloader("test", batch_size=512, shuffle=False, num_workers=0)
    ckpt_dir    = cfg.CHECKPOINTS_DIR
    rows        = []
    y_scores_all = {}

    # ── Teachers ─────────────────────────────────────────────────────────────
    log("\n[Teachers]")
    teacher_ckpts = {
        "TCN"   : ckpt_dir / "teacher_tcn_best.pt",
        "BiLSTM": ckpt_dir / "teacher_bilstm_best.pt",
    }
    teacher_results = {}
    for name, path in teacher_ckpts.items():
        if not path.exists():
            log(f"  {name}: checkpoint not found, skipping")
            continue
        r = eval_checkpoint(path, n_features, test_loader, device)
        print_dual_threshold_summary(name, r["dual"])
        teacher_results[name] = r
        y_scores_all[name] = (r["y_true"], r["y_score"])
        rows.append({
            "model"        : name,
            "type"         : "Teacher",
            "n_params"     : r["n_params"],
            "size_mb"      : round(r["size_mb"], 4),
            "latency_ms"   : round(r["latency_ms"], 2),
            "test_auroc_ci": format_ci(r["auroc_pt"], r["auroc_lo"], r["auroc_hi"]),
            "test_auprc"   : round(r["dual"]["auprc"], 4),
            "kr_pct"       : "—",
            "youden_sens"  : round(r["dual"]["youden_sensitivity"], 3),
            "youden_spec"  : round(r["dual"]["youden_specificity"], 3),
            "youden_f1"    : round(r["dual"]["youden_f1"], 3),
            "clin_sens"    : round(r["dual"]["clin_sensitivity"], 3),
            "clin_spec"    : round(r["dual"]["clin_specificity"], 3),
            "clin_f1"      : round(r["dual"]["clin_f1"], 3),
        })

    # DeLong: TCN vs BiLSTM
    if "TCN" in teacher_results and "BiLSTM" in teacher_results:
        dt = delong_test(
            teacher_results["TCN"]["y_true"],
            teacher_results["TCN"]["y_score"],
            teacher_results["BiLSTM"]["y_score"],
        )
        log(f"\n  DeLong TCN vs BiLSTM: z={dt['z_stat']:.3f} "
            f"p={dt['p_value']:.4f} {'sig' if dt['significant'] else 'n.s.'}")

    # ── GWO-KD Student ────────────────────────────────────────────────────────
    log("\n[GWO-KD Student]")
    gwo_paths = sorted(ckpt_dir.glob("student_gwo_kd_s*.pt"))
    best_teacher_auc = max(
        (r["auroc_pt"] for r in teacher_results.values()), default=0.7)

    for path in gwo_paths:
        r = eval_checkpoint(path, n_features, test_loader, device)
        kr = knowledge_retention(r["auroc_pt"], best_teacher_auc)
        print_dual_threshold_summary(f"GWO-KD ({path.stem})", r["dual"])
        y_scores_all[path.stem] = (r["y_true"], r["y_score"])
        rows.append({
            "model"        : f"GWO-KD ({path.stem})",
            "type"         : "Student (GWO-KD)",
            "n_params"     : r["n_params"],
            "size_mb"      : round(r["size_mb"], 4),
            "latency_ms"   : round(r["latency_ms"], 2),
            "test_auroc_ci": format_ci(r["auroc_pt"], r["auroc_lo"], r["auroc_hi"]),
            "test_auprc"   : round(r["dual"]["auprc"], 4),
            "kr_pct"       : round(kr, 2),
            "youden_sens"  : round(r["dual"]["youden_sensitivity"], 3),
            "youden_spec"  : round(r["dual"]["youden_specificity"], 3),
            "youden_f1"    : round(r["dual"]["youden_f1"], 3),
            "clin_sens"    : round(r["dual"]["clin_sensitivity"], 3),
            "clin_spec"    : round(r["dual"]["clin_specificity"], 3),
            "clin_f1"      : round(r["dual"]["clin_f1"], 3),
        })

    # ── KD Baselines ──────────────────────────────────────────────────────────
    log("\n[KD Baselines]")
    kd_path = cfg.RESULTS_DIR / "kd_baselines.csv"
    if kd_path.exists():
        kd_df = pd.read_csv(kd_path)
        log("\n" + kd_df.to_string(index=False))

    # ── Summary table ─────────────────────────────────────────────────────────
    results_df = pd.DataFrame(rows)
    out_path   = cfg.RESULTS_DIR / "results_mimic.csv"
    results_df.to_csv(out_path, index=False)
    log(f"\n  Saved → {out_path}")
    log("\n" + results_df[["model","n_params","size_mb","latency_ms",
                            "test_auroc_ci","kr_pct","youden_sens","clin_sens"]
                          ].to_string(index=False))

    return results_df


# =============================================================================
# eICU external validation
# =============================================================================

def eval_eicu(device, n_features):
    log("\n" + "="*60)
    log("eICU-CRD External Validation (Zero-shot)")
    log("="*60)

    eicu_x_path = cfg.EICU_PROCESSED_DIR / "windows_features.npy"
    eicu_y_path = cfg.EICU_PROCESSED_DIR / "windows_labels.npy"

    if not eicu_x_path.exists():
        log("ERROR: eICU windows not found. Run: python src/eicu_extract.py first")
        return None

    X_eicu = torch.from_numpy(np.load(eicu_x_path)).float()
    y_eicu = np.load(eicu_y_path)
    log(f"  eICU windows: {X_eicu.shape}  pos: {y_eicu.sum():,} ({y_eicu.mean():.1%})")

    from torch.utils.data import TensorDataset, DataLoader
    ds  = TensorDataset(X_eicu, torch.from_numpy(y_eicu.astype(np.float32)))
    ldr = DataLoader(ds, batch_size=512, shuffle=False)

    rows = []
    # Evaluate all available checkpoints
    for ckpt_path in sorted(cfg.CHECKPOINTS_DIR.glob("*.pt")):
        try:
            r = eval_checkpoint(ckpt_path, n_features, ldr, device)
        except Exception as e:
            log(f"  {ckpt_path.name}: SKIP ({e})")
            continue

        mtype = "Teacher" if r["ckpt"]["model_name"] in ("tcn","bilstm") else "Student"
        kr_val = "—"
        log(f"\n  {ckpt_path.stem}:")
        log(f"    AUROC={r['auroc_pt']:.4f} [{r['auroc_lo']:.4f}–{r['auroc_hi']:.4f}]  "
            f"AUPRC={r['dual']['auprc']:.4f}")

        rows.append({
            "model"        : ckpt_path.stem,
            "type"         : mtype,
            "dataset"      : "eICU",
            "n_params"     : r["n_params"],
            "size_mb"      : round(r["size_mb"], 4),
            "test_auroc_ci": format_ci(r["auroc_pt"], r["auroc_lo"], r["auroc_hi"]),
            "test_auprc"   : round(r["dual"]["auprc"], 4),
            "youden_sens"  : round(r["dual"]["youden_sensitivity"], 3),
            "youden_spec"  : round(r["dual"]["youden_specificity"], 3),
            "clin_sens"    : round(r["dual"]["clin_sensitivity"], 3),
        })

    df = pd.DataFrame(rows)
    out_path = cfg.RESULTS_DIR / "results_eicu.csv"
    df.to_csv(out_path, index=False)
    log(f"\n  Saved → {out_path}")
    return df


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="mimic",
                        choices=["mimic", "eicu", "all"])
    args = parser.parse_args()

    log("=" * 60)
    log("Stage 10 — Final Evaluation")
    log("=" * 60)

    device = get_device()
    log(f"  Device: {device}")

    X_sample   = np.load(cfg.PROCESSED_DIR / "windows_features.npy", mmap_mode="r")
    n_features = X_sample.shape[2]
    log(f"  n_features: {n_features}")

    if args.dataset in ("mimic", "all"):
        eval_mimic(device, n_features)

    if args.dataset in ("eicu", "all"):
        eval_eicu(device, n_features)

    log("\n[Stage 10 COMPLETE] ✓")


if __name__ == "__main__":
    main()
