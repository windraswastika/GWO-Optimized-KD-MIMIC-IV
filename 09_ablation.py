"""
09_ablation.py — Stage 9: Full Ablation Programme (Section 2.13)
===================================================================
Closes the R1-4/R2-2 reviewer gap: full ablation coverage beyond the single
GWO-optimal configuration.

(a) Distillation weights & temperature — univariate sweep around the
    GWO-optimal point (others held fixed): alpha_KD, beta_KD, T.
(b) Loss-component ablation — BCE-only / BCE+KL / BCE+feature / Full.
    BCE-only and BCE+KL are already available (screening vanilla + kd_baselines
    VANILLA row); only BCE+feature is a new run here.
(c) GWO configuration ablation (premature-convergence check) — REDUCED to a
    univariate design (agreed 2026-09-02, bare-minimum scope): fix I_max=10
    and vary N_w in {5,10,20,30}, then fix N_w=10 and vary I_max in
    {10,20,50}, instead of the full 4x3 cross-product (~52x the main search's
    evaluation budget at 5 seeds each — not tractable as a "bare minimum"
    closeout). The (N_w=10, I_max=10) point is reused from the existing main
    GWO run (results/gwo_history.csv) rather than rerun, when available.
(d) Distillation on/off — vanilla vs. GWO-distilled, per architecture
    (S1/S2/S3), with DeLong tests. The main GWO-KD run only distilled the
    winning architecture (S2); this adds two extra full_distil runs (S1, S3)
    at the same GWO-optimal (T, alpha_KD, beta_KD) so "per architecture" from
    Section 2.13(d) is genuinely covered, not just the winner.

Reuses (does not reimplement) FeatureRegressor / kd_loss / full_distil /
quick_train_student from 06_gwo_kd.py and eval_checkpoint from 10_evaluate.py,
loaded via importlib since their filenames start with a digit (same pattern
already used by 07_recovery.py in this codebase).

Run:
    python src/09_ablation.py                  # all parts, all seeds
    python src/09_ablation.py --parts a b       # only weight/temperature sweep
    python src/09_ablation.py --parts c --seeds 0 1 2 3 4
"""

import sys, os, re, time, argparse, importlib.util
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import torch
from datetime import datetime

import config as cfg
from utils.data_utils import (
    load_split_arrays, SepsisWindowDataset, make_dataloader,
    count_params, model_size_mb,
)
from utils.metrics import dual_threshold_eval, bootstrap_auroc_ci, delong_test
from models.teacher import build_teacher
from models.students import build_student


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# =============================================================================
# Reuse existing KD machinery (digit-prefixed module names -> importlib)
# =============================================================================

def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_HERE     = os.path.dirname(__file__)
_gwo_kd   = _load_module("_ablation_gwo_kd", os.path.join(_HERE, "06_gwo_kd.py"))
_evalmod  = _load_module("_ablation_evaluate", os.path.join(_HERE, "10_evaluate.py"))

FeatureRegressor = _gwo_kd.FeatureRegressor
full_distil      = _gwo_kd.full_distil
quick_train_student = _gwo_kd.quick_train_student
get_device       = _gwo_kd.get_device
eval_checkpoint  = _evalmod.eval_checkpoint

from utils.gwo import run_gwo_multiseed


# =============================================================================
# Shared setup
# =============================================================================

def load_shared(device):
    """Teacher, datasets, pos_weight — identical protocol to 06_gwo_kd.py."""
    ckpt_path = cfg.CHECKPOINTS_DIR / "teacher_best.pt"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    teacher_name = ckpt["model_name"]
    n_features   = ckpt["n_features"]
    teacher = build_teacher(teacher_name, n_features=n_features).to(device)
    teacher.load_state_dict(ckpt["state_dict"])
    teacher.eval()

    _, y_train = load_split_arrays("train")
    n_pos = y_train.sum(); n_neg = len(y_train) - n_pos
    pos_weight = torch.tensor(min(n_neg / n_pos, 10.0), dtype=torch.float32).to(device)

    X_train, y_train_arr = load_split_arrays("train")
    X_val,   y_val_arr   = load_split_arrays("val")
    train_dataset = SepsisWindowDataset(X_train, y_train_arr)
    val_dataset   = SepsisWindowDataset(X_val, y_val_arr)
    w_pos = len(y_train_arr) / (2 * n_pos) if n_pos > 0 else 1.0
    w_neg = len(y_train_arr) / (2 * n_neg) if n_neg > 0 else 1.0
    train_weights = torch.from_numpy(np.where(y_train_arr == 1, w_pos, w_neg)).float()

    return teacher, n_features, pos_weight, train_dataset, val_dataset, train_weights


def load_gwo_optimal() -> dict:
    """GWO-optimal (T, alpha_kd, beta_kd, s) from the main search's result CSV."""
    path = cfg.RESULTS_DIR / "gwo_kd_result.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — run 06_gwo_kd.py first")
    row = pd.read_csv(path).iloc[0]
    s_match = re.search(r"S(\d+)", str(row["model"]))
    return {
        "T"       : float(row["gwo_T"]),
        "alpha_kd": float(row["gwo_alpha"]),
        "beta_kd" : float(row["gwo_beta"]),
        "s"       : int(s_match.group(1)),
    }


def run_one_distil(params, teacher, n_features, pos_weight, train_dataset,
                    val_dataset, train_weights, device, test_loader, tag):
    """One full_distil run + test-set evaluation. Returns a result dict."""
    s_idx   = params["s"]
    student = build_student(s_idx, n_features=n_features).to(device)
    reg     = FeatureRegressor(student.feature_dim, teacher.feature_dim).to(device)

    log(f"  [{tag}] full_distil: s={s_idx} T={params['T']:.2f} "
        f"alpha={params['alpha_kd']:.2f} beta={params['beta_kd']:.3f}")
    val_auc, _, _ = full_distil(
        student, reg, teacher, params, device, pos_weight,
        train_dataset=train_dataset, val_dataset=val_dataset,
        train_weights=train_weights,
    )

    student.eval()
    scores, labels = [], []
    with torch.no_grad():
        for X_b, y_b in test_loader:
            z_s, _ = student(X_b.to(device))
            scores.append(torch.sigmoid(z_s).cpu().numpy())
            labels.append(y_b.numpy())
    y_true  = np.concatenate(labels)
    y_score = np.concatenate(scores)

    dual = dual_threshold_eval(y_true, y_score, cfg.EVAL_SENSITIVITY_TARGET)
    auc_pt, auc_lo, auc_hi = bootstrap_auroc_ci(y_true, y_score, n_resamples=1000)
    log(f"  [{tag}] test AUROC={auc_pt:.4f} [{auc_lo:.4f}-{auc_hi:.4f}]  AUPRC={dual['auprc']:.4f}")

    return {
        "tag": tag, "s": s_idx, "T": params["T"], "alpha_kd": params["alpha_kd"],
        "beta_kd": params["beta_kd"], "val_auroc": val_auc,
        "test_auroc": auc_pt, "test_auroc_lo": auc_lo, "test_auroc_hi": auc_hi,
        "test_auprc": dual["auprc"], "y_true": y_true, "y_score": y_score,
        "student_state_dict": {k: v.cpu().clone() for k, v in student.state_dict().items()},
    }


_NON_CSV_KEYS = ("y_true", "y_score", "student_state_dict")


def _drop_arrays(rows: list[dict]) -> pd.DataFrame:
    """Rows may carry y_true/y_score/student_state_dict for in-process reuse;
    strip them before writing to CSV."""
    return pd.DataFrame([{k: v for k, v in r.items() if k not in _NON_CSV_KEYS}
                          for r in rows])


# =============================================================================
# (a) Distillation weights & temperature sweep
# =============================================================================

def _reuse_if_baseline(params, base_params, tag, varied):
    """If this sweep point is numerically identical to the GWO-optimal base
    config, reuse the already-computed Full GWO-KD result instead of
    retraining an identical model."""
    close = (abs(params["T"] - base_params["T"]) < 1e-9
             and abs(params["alpha_kd"] - base_params["alpha_kd"]) < 1e-9
             and abs(params["beta_kd"] - base_params["beta_kd"]) < 1e-9
             and params["s"] == base_params["s"])
    if not close:
        return None
    gwo_path = cfg.RESULTS_DIR / "gwo_kd_result.csv"
    if not gwo_path.exists():
        return None
    row = pd.read_csv(gwo_path).iloc[0]
    log(f"  [{tag}] identical to GWO-optimal base config -> reusing gwo_kd_result.csv")
    return {
        "tag": tag, "s": params["s"], "T": params["T"], "alpha_kd": params["alpha_kd"],
        "beta_kd": params["beta_kd"], "val_auroc": None,
        "test_auroc": float(row["test_auroc"]), "test_auroc_lo": None, "test_auroc_hi": None,
        "test_auprc": float(row["test_auprc"]), "varied": varied,
    }


def part_a_weight_sweep(base_params, shared, device, test_loader):
    log("\n" + "=" * 60)
    log("(a) Distillation weight & temperature sweep")
    log("=" * 60)
    teacher, n_features, pos_weight, train_dataset, val_dataset, train_weights = shared

    results = []
    for alpha in cfg.ABLATION_ALPHA_KD_GRID:
        params = {**base_params, "alpha_kd": alpha}
        tag = f"alpha_kd={alpha}"
        r = _reuse_if_baseline(params, base_params, tag, "alpha_kd")
        if r is None:
            r = run_one_distil(params, teacher, n_features, pos_weight, train_dataset,
                                val_dataset, train_weights, device, test_loader, tag=tag)
            r["varied"] = "alpha_kd"
        results.append(r)

    for beta in cfg.ABLATION_BETA_KD_GRID:
        params = {**base_params, "beta_kd": beta}
        tag = f"beta_kd={beta}"
        r = _reuse_if_baseline(params, base_params, tag, "beta_kd")
        if r is None:
            r = run_one_distil(params, teacher, n_features, pos_weight, train_dataset,
                                val_dataset, train_weights, device, test_loader, tag=tag)
            r["varied"] = "beta_kd"
        results.append(r)

    for T in cfg.ABLATION_T_GRID:
        params = {**base_params, "T": float(T)}
        tag = f"T={T}"
        r = _reuse_if_baseline(params, base_params, tag, "T")
        if r is None:
            r = run_one_distil(params, teacher, n_features, pos_weight, train_dataset,
                                val_dataset, train_weights, device, test_loader, tag=tag)
            r["varied"] = "T"
        results.append(r)

    df = _drop_arrays(results)
    out = cfg.RESULTS_DIR / "ablation_weights.csv"
    df.to_csv(out, index=False)
    log(f"  Saved -> {out}")
    return df


# =============================================================================
# (b) Loss-component ablation
# =============================================================================

def part_b_loss_components(base_params, shared, device, test_loader):
    log("\n" + "=" * 60)
    log("(b) Loss-component ablation (BCE / BCE+KL / BCE+feature / Full)")
    log("=" * 60)
    teacher, n_features, pos_weight, train_dataset, val_dataset, train_weights = shared

    rows = []

    # BCE-only: reuse the vanilla screening result for the winning architecture.
    screen_path = cfg.RESULTS_DIR / "screening_summary.csv"
    screen_df = pd.read_csv(screen_path)
    s_row = screen_df[screen_df["student"].str.startswith(f"S{base_params['s']}")].iloc[0]
    rows.append({
        "component": "BCE-only (vanilla)", "test_auroc": s_row["test_auroc"],
        "test_auroc_ci": s_row["test_auroc_ci"], "test_auprc": s_row["test_auprc"],
        "source": "screening_summary.csv (reused, no new run)",
    })

    # BCE+KL: reuse the classical "VANILLA KD" (response-only) baseline.
    kd_path = cfg.RESULTS_DIR / "kd_baselines.csv"
    kd_df = pd.read_csv(kd_path)
    vanilla_kd = kd_df[kd_df["method"] == "VANILLA"].iloc[0]
    rows.append({
        "component": "BCE+KL (response-only)", "test_auroc": vanilla_kd["test_auroc"],
        "test_auroc_ci": vanilla_kd["test_auroc_ci"], "test_auprc": vanilla_kd["test_auprc"],
        "source": "kd_baselines.csv (reused, no new run)",
    })

    # BCE+feature: NEW run. alpha_kd=1.0 zeroes the KL term's weight (1-alpha);
    # T is passed through unchanged only to keep kd_loss's softmax/T scaling
    # well-defined, it has no effect on the loss value when alpha_kd=1.0.
    feat_params = {**base_params, "alpha_kd": 1.0}
    r = run_one_distil(feat_params, teacher, n_features, pos_weight, train_dataset,
                        val_dataset, train_weights, device, test_loader,
                        tag="BCE+feature")
    rows.append({
        "component": "BCE+feature", "test_auroc": r["test_auroc"],
        "test_auroc_ci": f"[{r['test_auroc_lo']:.4f}-{r['test_auroc_hi']:.4f}]",
        "test_auprc": r["test_auprc"], "source": "new run (this script)",
    })

    # Full: reuse the main GWO-KD result.
    gwo_path = cfg.RESULTS_DIR / "gwo_kd_result.csv"
    gwo_row = pd.read_csv(gwo_path).iloc[0]
    rows.append({
        "component": "Full (BCE+KL+feature)", "test_auroc": gwo_row["test_auroc"],
        "test_auroc_ci": gwo_row["test_auroc_ci"], "test_auprc": gwo_row["test_auprc"],
        "source": "gwo_kd_result.csv (reused, no new run)",
    })

    df = pd.DataFrame(rows)
    out = cfg.RESULTS_DIR / "ablation_loss_components.csv"
    df.to_csv(out, index=False)
    log("\n" + df.to_string(index=False))
    log(f"  Saved -> {out}")
    return df


# =============================================================================
# (c) GWO configuration ablation (reduced univariate grid)
# =============================================================================

def part_c_gwo_config(shared, device, seeds):
    log("\n" + "=" * 60)
    log("(c) GWO configuration ablation (N_w, I_max) -- reduced univariate grid")
    log("=" * 60)
    teacher, n_features, pos_weight, train_dataset, val_dataset, train_weights = shared

    def fitness(params: dict) -> float:
        s_idx   = params["s"]
        student = build_student(s_idx, n_features=n_features).to(device)
        reg     = FeatureRegressor(student.feature_dim, teacher.feature_dim).to(device)
        return quick_train_student(student, reg, teacher, params, device,
                                    n_epochs=5, pos_weight=pos_weight,
                                    train_dataset=train_dataset, val_dataset=val_dataset,
                                    train_weights=train_weights)

    # Reuse the (N_w=10, I_max=10) baseline from the main GWO run if its
    # per-seed history was saved; otherwise fall back to rerunning it here.
    configs = [(nw, cfg.GWO_I_MAX) for nw in cfg.ABLATION_GWO_NW_GRID] \
            + [(cfg.GWO_N_WOLVES, im) for im in cfg.ABLATION_GWO_IMAX_GRID]
    configs = list(dict.fromkeys(configs))  # de-dup the shared (10,10) point

    history_path = cfg.RESULTS_DIR / "gwo_history.csv"
    baseline_scores = None
    if history_path.exists():
        hdf = pd.read_csv(history_path)
        last_iter = hdf.groupby("seed")["iteration"].idxmax()
        baseline_scores = hdf.loc[last_iter, "alpha_score"].tolist()
        log(f"  Reusing (N_w={cfg.GWO_N_WOLVES}, I_max={cfg.GWO_I_MAX}) baseline "
            f"from {history_path} (scores={baseline_scores})")

    rows = []
    for n_w, i_max in configs:
        if (n_w, i_max) == (cfg.GWO_N_WOLVES, cfg.GWO_I_MAX) and baseline_scores is not None:
            for seed, score in zip(seeds, baseline_scores):
                rows.append({"n_wolves": n_w, "i_max": i_max, "seed": seed,
                             "best_fitness": score, "wall_clock_s": None,
                             "source": "reused from main GWO run"})
            continue

        log(f"\n  [N_w={n_w}, I_max={i_max}] running {len(seeds)} seed(s) ...")
        t0 = time.time()
        _, best_score, all_results = run_gwo_multiseed(
            fitness_fn=fitness, seeds=seeds, n_wolves=n_w, i_max=i_max,
            verbose=True, n_workers=1,
        )
        elapsed = time.time() - t0
        for seed, params, score, hist in all_results:
            rows.append({
                "n_wolves": n_w, "i_max": i_max, "seed": seed,
                "best_fitness": score, "iterations_run": len(hist),
                "wall_clock_s": round(elapsed, 1), "source": "new run (this script)",
            })
        log(f"  [N_w={n_w}, I_max={i_max}] done in {elapsed:.1f}s, best={best_score:.4f}")

    df = pd.DataFrame(rows)
    out = cfg.RESULTS_DIR / "ablation_gwo_config.csv"
    df.to_csv(out, index=False)
    log(f"  Saved -> {out}")
    return df


# =============================================================================
# (d) Distillation on/off, per architecture
# =============================================================================

def part_d_kd_onoff(base_params, shared, device, test_loader):
    log("\n" + "=" * 60)
    log("(d) Distillation on/off, per architecture (S1/S2/S3)")
    log("=" * 60)
    teacher, n_features, pos_weight, train_dataset, val_dataset, train_weights = shared

    rows = []
    for s_idx in (1, 2, 3):
        vanilla_ckpt = cfg.CHECKPOINTS_DIR / f"student_s{s_idx}_vanilla.pt"
        distil_ckpt  = cfg.CHECKPOINTS_DIR / f"student_gwo_kd_s{s_idx}.pt"

        if not vanilla_ckpt.exists():
            log(f"  S{s_idx}: vanilla checkpoint missing, skip (run 05_screening.py)")
            continue

        if s_idx == base_params["s"] and distil_ckpt.exists():
            log(f"  S{s_idx}: reusing existing GWO-KD checkpoint (winning architecture)")
        else:
            # Extra run: distil this architecture at the GWO-optimal (T, alpha, beta)
            # so "per architecture" is genuinely covered, not just the winner.
            params = {**base_params, "s": s_idx}
            r = run_one_distil(params, teacher, n_features, pos_weight, train_dataset,
                                val_dataset, train_weights, device, test_loader,
                                tag=f"S{s_idx} distilled (extra)")
            torch.save({
                "model_name": f"s{s_idx}", "n_features": n_features,
                "state_dict": r["student_state_dict"],
                "val_auroc": r["val_auroc"], "test_auroc": r["test_auroc"],
                "gwo_params": params,
            }, distil_ckpt)
            log(f"  S{s_idx}: saved extra distilled checkpoint -> {distil_ckpt}")

        r_vanilla = eval_checkpoint(vanilla_ckpt, n_features, test_loader, device)
        r_distil  = eval_checkpoint(distil_ckpt, n_features, test_loader, device)

        dt = delong_test(r_vanilla["y_true"], r_vanilla["y_score"], r_distil["y_score"])
        rows.append({
            "architecture": f"S{s_idx}",
            "vanilla_auroc": round(r_vanilla["auroc_pt"], 4),
            "distilled_auroc": round(r_distil["auroc_pt"], 4),
            "auroc_gain": round(r_distil["auroc_pt"] - r_vanilla["auroc_pt"], 4),
            "delong_z": round(dt["z_stat"], 3),
            "delong_p": round(dt["p_value"], 4),
            "significant": dt["significant"],
        })
        log(f"  S{s_idx}: vanilla={r_vanilla['auroc_pt']:.4f}  "
            f"distilled={r_distil['auroc_pt']:.4f}  "
            f"gain={r_distil['auroc_pt']-r_vanilla['auroc_pt']:+.4f}  "
            f"DeLong p={dt['p_value']:.4f} {'sig' if dt['significant'] else 'n.s.'}")

    df = pd.DataFrame(rows)
    out = cfg.RESULTS_DIR / "ablation_kd_onoff.csv"
    df.to_csv(out, index=False)
    log(f"  Saved -> {out}")
    return df


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parts", nargs="+", default=["a", "b", "c", "d"],
                         choices=["a", "b", "c", "d"])
    parser.add_argument("--seeds", type=int, nargs="+", default=cfg.META_SEEDS)
    args = parser.parse_args()

    log("=" * 60)
    log("Stage 09 - Full Ablation Programme (Section 2.13)")
    log("=" * 60)

    device = get_device()
    log(f"  Device: {device}")

    shared = load_shared(device)
    base_params = load_gwo_optimal()
    log(f"  GWO-optimal base params: {base_params}")

    test_loader = make_dataloader("test", batch_size=cfg.KD_BATCH_SIZE,
                                  shuffle=False, num_workers=0)

    if "a" in args.parts:
        part_a_weight_sweep(base_params, shared, device, test_loader)
    if "b" in args.parts:
        part_b_loss_components(base_params, shared, device, test_loader)
    if "c" in args.parts:
        part_c_gwo_config(shared, device, args.seeds)
    if "d" in args.parts:
        part_d_kd_onoff(base_params, shared, device, test_loader)

    log("\n[Stage 09 COMPLETE] (parts: %s)" % ",".join(args.parts))
    log("  -> Run next: python src/10_evaluate.py --all   (regenerate manuscript tables)")


if __name__ == "__main__":
    main()
