"""
benchmark_speedup.py — Micro-benchmark for 3 proposed GWO speedup strategies
==============================================================================
Tests (empirically, with real data & models):
  [OPT-1] MPS  vs CPU  device for student fitness evaluation
  [OPT-2] Serial vs ThreadPoolExecutor wolf parallelism (N=4 wolves)
  [OPT-3] Rebuild DataLoader each call vs Reuse (hoist) DataLoader

Uses a SUBSAMPLE of the training data so each rep finishes in ~1-2 min.
Speedup ratios remain valid (hardware throughput ratio is scale-invariant).

Usage:
    python src/benchmark_speedup.py
"""

import sys, os, time, warnings
sys.path.insert(0, os.path.dirname(__file__))
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from concurrent.futures import ThreadPoolExecutor
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset

import config as cfg
from utils.data_utils import load_split_arrays, SepsisWindowDataset
from utils.metrics import auroc
from models.teacher import build_teacher
from models.students import build_student

# ─── Benchmark settings ───────────────────────────────────────────────────────
N_REPS          = 3        # repetitions per variant
N_EPOCHS_FIT    = 3        # quick-train epochs (real = 5)
N_WOLVES_PAR    = 4        # wolves for parallelism test (real = 10)
BATCH_SIZE      = cfg.KD_BATCH_SIZE   # 256

# Subsample size — keeps each rep ~1-2 min on CPU.
# Speedup ratio is hardware-bound, not dataset-size-bound, so this is valid.
TRAIN_SUBSAMPLE = 50_000   # out of 4.6M
VAL_SUBSAMPLE   = 15_000   # out of 990K

DIVIDER = "─" * 64


def ts():
    return time.strftime("%H:%M:%S")


def log(msg):
    print(f"[{ts()}] {msg}", flush=True)


# =============================================================================
# Feature regressor
# =============================================================================

class FeatureRegressor(nn.Module):
    def __init__(self, s_dim, t_dim):
        super().__init__()
        self.proj = nn.Linear(s_dim, t_dim)

    def forward(self, x):
        return self.proj(x)


def kd_loss(z_s, z_t, f_s_proj, f_t, y, T, alpha, beta, pos_weight):
    bce_fn   = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    L_bce    = bce_fn(z_s, y)
    logits_t = torch.stack([-z_t, z_t], dim=1) / T
    logits_s = torch.stack([-z_s, z_s], dim=1) / T
    p_t      = F.softmax(logits_t, dim=1).detach()
    log_p_s  = F.log_softmax(logits_s, dim=1)
    L_kl     = F.kl_div(log_p_s, p_t, reduction="batchmean") * (T ** 2)
    L_feat   = F.mse_loss(f_s_proj, f_t.detach())
    return alpha * L_bce + (1 - alpha) * L_kl + beta * L_feat


# =============================================================================
# Core: one fitness evaluation
# =============================================================================

def one_fitness_eval(
    teacher, n_features,
    device,
    train_dataset, val_dataset, train_weights,
    pos_weight,
    n_epochs=N_EPOCHS_FIT,
    train_loader_cache=None,   # None = rebuild each call; else reuse
):
    params = {"T": 8.0, "alpha_kd": 0.5, "beta_kd": 0.1, "s": 2}

    student   = build_student(params["s"], n_features=n_features).to(device)
    regressor = FeatureRegressor(student.feature_dim, teacher.feature_dim).to(device)

    if train_loader_cache is None:
        sampler = WeightedRandomSampler(
            weights=train_weights, num_samples=len(train_weights), replacement=True
        )
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                                  sampler=sampler, num_workers=0, pin_memory=False)
    else:
        train_loader = train_loader_cache

    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=0, pin_memory=False)

    optimizer = torch.optim.Adam(
        list(student.parameters()) + list(regressor.parameters()), lr=cfg.TEACHER_LR
    )
    teacher.eval()
    student.train(); regressor.train()

    for _ in range(n_epochs):
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            with torch.no_grad():
                z_t, f_t = teacher(X_b)
            z_s, f_s = student(X_b)
            f_s_proj = regressor(f_s)
            pw = pos_weight.to(device)
            loss = kd_loss(z_s, z_t, f_s_proj, f_t, y_b,
                           params["T"], params["alpha_kd"], params["beta_kd"], pw)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(student.parameters()) + list(regressor.parameters()), 1.0)
            optimizer.step()

    student.eval()
    all_s, all_l = [], []
    with torch.no_grad():
        for X_b, y_b in val_loader:
            z_s, _ = student(X_b.to(device))
            all_s.append(torch.sigmoid(z_s).cpu().numpy())
            all_l.append(y_b.numpy())
    return auroc(np.concatenate(all_l), np.concatenate(all_s))


# =============================================================================
# Benchmark helpers
# =============================================================================

def timed_run(fn, reps=N_REPS):
    times, results = [], []
    for r in range(reps):
        t0 = time.time()
        res = fn()
        elapsed = time.time() - t0
        times.append(elapsed)
        results.append(res)
        val = f"{res:.4f}" if isinstance(res, float) else "list"
        log(f"    rep {r+1}/{reps}: {elapsed:.1f}s  (result={val})")
    return times, results


def summarize(label, times):
    mu  = np.mean(times)
    std = np.std(times)
    print(f"  {label:52s}: {mu:6.1f}s ± {std:.1f}s")
    return mu


# =============================================================================
# MAIN
# =============================================================================

def main():
    log("=" * 64)
    log("GWO Speedup Benchmark  (subsample mode)")
    log("=" * 64)

    has_mps = torch.backends.mps.is_available()
    log(f"MPS available  : {has_mps}")
    log(f"CPU count      : {os.cpu_count()}")
    log(f"N_REPS         : {N_REPS}")
    log(f"N_EPOCHS_FIT   : {N_EPOCHS_FIT}")
    log(f"N_WOLVES_PAR   : {N_WOLVES_PAR}")
    log(f"TRAIN_SUBSAMPLE: {TRAIN_SUBSAMPLE:,}")
    log(f"VAL_SUBSAMPLE  : {VAL_SUBSAMPLE:,}")
    log("")

    # ── Load data ONCE, then subsample ───────────────────────────────────────
    log("Loading data ...")
    X_train_full, y_train_full = load_split_arrays("train")
    X_val_full,   y_val_full   = load_split_arrays("val")

    rng_idx = np.random.default_rng(42)
    tr_idx  = rng_idx.choice(len(X_train_full), size=TRAIN_SUBSAMPLE, replace=False)
    vl_idx  = rng_idx.choice(len(X_val_full),   size=VAL_SUBSAMPLE,   replace=False)

    X_train = X_train_full[tr_idx]; y_train = y_train_full[tr_idx]
    X_val   = X_val_full[vl_idx];   y_val   = y_val_full[vl_idx]

    train_dataset = SepsisWindowDataset(X_train, y_train)
    val_dataset   = SepsisWindowDataset(X_val,   y_val)

    n_pos = int(y_train.sum()); n_neg = len(y_train) - n_pos
    pos_weight_cpu = torch.tensor(min(n_neg / n_pos, 10.0), dtype=torch.float32)

    w_pos = len(y_train) / (2 * n_pos) if n_pos > 0 else 1.0
    w_neg = len(y_train) / (2 * n_neg) if n_neg > 0 else 1.0
    train_weights = torch.from_numpy(
        np.where(y_train == 1, w_pos, w_neg)
    ).float()

    log(f"Full dataset : train={len(X_train_full):,}  val={len(X_val_full):,}")
    log(f"Subsample    : train={len(X_train):,}       val={len(X_val):,}")
    log(f"Class balance: {n_pos/len(y_train)*100:.1f}% positive (subsample)")
    log(f"Scale factor : {len(X_train_full)/TRAIN_SUBSAMPLE:.0f}x  "
        f"(multiply benchmark times by this to project real GWO time)")
    log("")

    # ── Load teacher ──────────────────────────────────────────────────────────
    ckpt_path = cfg.CHECKPOINTS_DIR / "teacher_best.pt"
    if not ckpt_path.exists():
        log(f"ERROR: {ckpt_path} not found — run 04_teacher.py first")
        return

    log("Loading teacher ...")
    ckpt         = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    teacher_name = ckpt["model_name"]
    n_features   = ckpt["n_features"]

    teacher_cpu = build_teacher(teacher_name, n_features=n_features)
    teacher_cpu.load_state_dict(ckpt["state_dict"])
    teacher_cpu.eval()

    if has_mps:
        teacher_mps = build_teacher(teacher_name, n_features=n_features)
        teacher_mps.load_state_dict(ckpt["state_dict"])
        teacher_mps.to("mps").eval()

    n_params_t = sum(p.numel() for p in teacher_cpu.parameters())
    log(f"Teacher: {teacher_name.upper()} ({n_params_t:,} params)")
    log("")

    results_summary = {}

    # =========================================================================
    # OPT-1: MPS vs CPU
    # =========================================================================
    print(DIVIDER)
    print("[OPT-1] Device: MPS vs CPU  (single fitness eval, hoisted loader)")
    print(DIVIDER)

    sampler_pre = WeightedRandomSampler(
        weights=train_weights, num_samples=len(train_weights), replacement=True
    )
    loader_pre = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                            sampler=sampler_pre, num_workers=0, pin_memory=False)

    log("[CPU] Running ...")
    t_cpu, _ = timed_run(lambda: one_fitness_eval(
        teacher_cpu, n_features, torch.device("cpu"),
        train_dataset, val_dataset, train_weights, pos_weight_cpu,
        train_loader_cache=loader_pre,
    ))
    mu_cpu = summarize("CPU (single eval)", t_cpu)
    results_summary["OPT1_cpu_s"] = mu_cpu

    if has_mps:
        log("[MPS] Running ...")
        pos_weight_mps = pos_weight_cpu.clone()
        t_mps, _ = timed_run(lambda: one_fitness_eval(
            teacher_mps, n_features, torch.device("mps"),
            train_dataset, val_dataset, train_weights, pos_weight_mps,
            train_loader_cache=loader_pre,
        ))
        mu_mps = summarize("MPS (single eval)", t_mps)
        speedup = mu_mps / mu_cpu
        arrow = "CPU faster ✓" if speedup > 1 else "MPS faster"
        print(f"\n  Speedup CPU/MPS = {speedup:.2f}x  ({arrow})")
        results_summary["OPT1_mps_s"]   = mu_mps
        results_summary["OPT1_speedup"] = speedup
    else:
        print("  MPS not available.")
        results_summary["OPT1_mps_s"]   = None
        results_summary["OPT1_speedup"] = None
    print()

    # =========================================================================
    # OPT-3: DataLoader rebuild vs Hoist
    # =========================================================================
    print(DIVIDER)
    print("[OPT-3] DataLoader: Rebuild each call vs Hoist (reuse)")
    print(DIVIDER)

    log("[REBUILD] Running ...")
    t_rebuild, _ = timed_run(lambda: one_fitness_eval(
        teacher_cpu, n_features, torch.device("cpu"),
        train_dataset, val_dataset, train_weights, pos_weight_cpu,
        train_loader_cache=None,
    ))
    mu_rebuild = summarize("CPU + rebuild DataLoader (baseline)", t_rebuild)

    sampler_hoist = WeightedRandomSampler(
        weights=train_weights, num_samples=len(train_weights), replacement=True
    )
    loader_hoist = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              sampler=sampler_hoist, num_workers=0, pin_memory=False)

    log("[HOIST] Running ...")
    t_hoist, _ = timed_run(lambda: one_fitness_eval(
        teacher_cpu, n_features, torch.device("cpu"),
        train_dataset, val_dataset, train_weights, pos_weight_cpu,
        train_loader_cache=loader_hoist,
    ))
    mu_hoist = summarize("CPU + hoisted DataLoader (OPT-3)", t_hoist)

    speedup_dl = mu_rebuild / mu_hoist
    saved_s    = mu_rebuild - mu_hoist
    saved_real = saved_s * (len(X_train_full) / TRAIN_SUBSAMPLE)  # scale to real data
    print(f"\n  Speedup = {speedup_dl:.2f}x  |  Saved {saved_s:.2f}s per eval (subsample)")
    print(f"  Projected saving on FULL data per eval: ~{saved_real:.0f}s")
    print(f"  Over ~500 evals on FULL data: saves ~{saved_real * 500 / 3600:.1f} hours")
    results_summary["OPT3_rebuild_s"] = mu_rebuild
    results_summary["OPT3_hoist_s"]   = mu_hoist
    results_summary["OPT3_speedup"]   = speedup_dl
    print()

    # =========================================================================
    # OPT-2: Serial vs ThreadPoolExecutor
    # =========================================================================
    print(DIVIDER)
    print(f"[OPT-2] Wolf parallelism: Serial vs ThreadPoolExecutor ({N_WOLVES_PAR} wolves)")
    print(DIVIDER)
    print("  Using CPU + hoisted loader for fair comparison.")
    print()

    sampler_par = WeightedRandomSampler(
        weights=train_weights, num_samples=len(train_weights), replacement=True
    )
    loader_par = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                            sampler=sampler_par, num_workers=0, pin_memory=False)

    def eval_one_wolf(_):
        return one_fitness_eval(
            teacher_cpu, n_features, torch.device("cpu"),
            train_dataset, val_dataset, train_weights, pos_weight_cpu,
            train_loader_cache=loader_par,
        )

    log("[SERIAL] Running ...")
    t_serial, _ = timed_run(lambda: [eval_one_wolf(i) for i in range(N_WOLVES_PAR)])
    mu_serial = summarize(f"Serial  ({N_WOLVES_PAR} wolves, CPU)", t_serial)

    log("[THREAD] Running ...")
    t_thread, _ = timed_run(lambda: list(
        ThreadPoolExecutor(max_workers=N_WOLVES_PAR).map(eval_one_wolf, range(N_WOLVES_PAR))
    ))
    mu_thread = summarize(f"Threads ({N_WOLVES_PAR} workers, CPU)", t_thread)

    speedup_par = mu_serial / mu_thread
    print(f"\n  Effective speedup = {speedup_par:.2f}x")
    if speedup_par < 1.1:
        print("  => GIL contention confirmed — threads do NOT help.")
    elif speedup_par >= 2.0:
        print("  => Meaningful parallel speedup!")
    else:
        print("  => Marginal benefit — overhead partially offsets gain.")
    results_summary["OPT2_serial_s"] = mu_serial
    results_summary["OPT2_thread_s"] = mu_thread
    results_summary["OPT2_speedup"]  = speedup_par
    print()

    # =========================================================================
    # Summary
    # =========================================================================
    print("=" * 64)
    print("FINAL SUMMARY")
    print("=" * 64)

    REAL_EVALS   = 500   # 5 seeds × 10 iter × 10 wolves
    SCALE        = len(X_train_full) / TRAIN_SUBSAMPLE
    baseline_s   = results_summary.get("OPT1_mps_s") or results_summary["OPT1_cpu_s"]
    hoist_s      = results_summary["OPT3_hoist_s"]
    cpu_s        = results_summary["OPT1_cpu_s"]
    opt1         = results_summary.get("OPT1_speedup")
    opt2         = results_summary["OPT2_speedup"]
    opt3         = results_summary["OPT3_speedup"]

    print(f"\n  [Subsample benchmark results]")
    print(f"  {'OPT-1: CPU/MPS speedup':<40} {opt1:.2f}x" if opt1 else
          f"  {'OPT-1: CPU/MPS speedup':<40} N/A")
    print(f"  {'OPT-2: Thread/Serial speedup':<40} {opt2:.2f}x")
    print(f"  {'OPT-3: Hoist/Rebuild speedup':<40} {opt3:.2f}x")

    print(f"\n  [Projected on FULL data — {REAL_EVALS} evals, serial]")
    bl_h   = baseline_s  * SCALE * REAL_EVALS / 3600
    cpu_h  = cpu_s       * SCALE * REAL_EVALS / 3600
    opt_h  = hoist_s     * SCALE * REAL_EVALS / 3600
    par_h  = opt_h / opt2 if opt2 > 1.0 else opt_h

    print(f"  {'Baseline (MPS, rebuild DL)':<40} {bl_h:5.1f} h")
    print(f"  {'CPU only (OPT-1)':<40} {cpu_h:5.1f} h")
    print(f"  {'CPU + Hoist (OPT-1 + OPT-3)':<40} {opt_h:5.1f} h")
    if opt2 > 1.1:
        print(f"  {'CPU+Hoist+Threads (all 3)':<40} {par_h:5.1f} h")

    print(f"\n  [Verdict]")
    if opt1 and opt1 > 1.2:
        print(f"  OPT-1 (MPS->CPU): CONFIRMED  {opt1:.2f}x speedup")
    elif opt1:
        print(f"  OPT-1 (MPS->CPU): MARGINAL   {opt1:.2f}x speedup")
    else:
        print(f"  OPT-1 (MPS->CPU): N/A (no MPS)")

    if opt3 > 1.05:
        saved_min = (results_summary["OPT3_rebuild_s"] - hoist_s) * SCALE * REAL_EVALS / 60
        print(f"  OPT-3 (Hoist DL): CONFIRMED  {opt3:.2f}x (~{saved_min:.0f} min saved total)")
    else:
        print(f"  OPT-3 (Hoist DL): NEGLIGIBLE {opt3:.2f}x")

    if opt2 > 1.5:
        print(f"  OPT-2 (Threads):  CONFIRMED  {opt2:.2f}x speedup")
    elif opt2 > 1.1:
        print(f"  OPT-2 (Threads):  MARGINAL   {opt2:.2f}x speedup")
    else:
        print(f"  OPT-2 (Threads):  NOT USEFUL {opt2:.2f}x (GIL contention)")

    print(f"\n[{ts()}] Benchmark complete.")


if __name__ == "__main__":
    main()
