"""
finish_ablation_c.py — Finish the remaining GWO configuration-ablation grid points
=====================================================================================
Completes Section 3.6(c) / Table (ablation_gwo_config.csv): (N_w=10,I_max=10) baseline,
N_w=5, and N_w=20 are already done (5 seeds each, results already in
results/ablation_gwo_config.csv). This script runs the THREE remaining grid points --
N_w=30 (I_max=10), I_max=20 (N_w=10), I_max=50 (N_w=10) -- 5 seeds each, 15 jobs total.

Two speedups over the original 09_ablation.py --parts c run, both specific to what's
actually slow here (verified against the current code, not applied blindly):

  1. Seed-level parallelism via ThreadPoolExecutor. The 5 seeds of a given (N_w, I_max)
     are fully independent (separate RNG streams, no shared mutable state). Unlike the
     wolf-level threading tried earlier in this project (which failed: tiny-model fitness
     evaluations are Python-overhead-dominated, so the GIL serialised everything -- see
     06_gwo_kd.py's get_device() docstring), each seed here is a *substantial* GPU-bound
     job (many epochs of real CUDA compute across up to I_max iterations). CUDA kernel
     execution releases the GIL, so concurrent seeds can make genuine use of multiple
     GPU streams / better utilise a GPU that a single 4K-parameter-model job under-uses.
     This is a materially different situation from the earlier failure, but it has NOT
     been previously benchmarked in this exact form -- watch the per-seed wall-clock
     timestamps this script logs; if they do not overlap in practice, pass --workers 1
     to fall back to serial (still correct, just not faster).

  2. Resume support: every (n_wolves, i_max, seed) result already present in
     results/ablation_gwo_config.csv is skipped, and every new result is appended to that
     file IMMEDIATELY as it completes (not batched at the end) -- so killing this script
     part-way through never loses more than the in-flight jobs.

  NOT changed: num_workers on a CPU DataLoader. quick_train_student() (the fitness
  function, imported from 06_gwo_kd.py) already uses GPUBatcher, which caches the full
  train/val tensors in GPU VRAM once and never touches a CPU DataLoader again -- there is
  no num_workers=0 bottleneck on this code path to fix.

Run:
    python src/finish_ablation_c.py 2>&1 | tee results/finish_ablation_c_log.txt
    python src-new/finish_ablation_c.py --workers 8   # try higher concurrency (tiny model, ample VRAM)
    python src/finish_ablation_c.py --workers 1   # fallback: serial, if parallel doesn't help
"""

import sys, os, csv, time, argparse
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch
from datetime import datetime
from pathlib import Path
from threading import Lock
from concurrent.futures import ThreadPoolExecutor, as_completed

import config as cfg
from utils.data_utils import load_split_arrays, SepsisWindowDataset


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# =============================================================================
# Reuse 06_gwo_kd.py (digit-prefixed filename -> importlib, same pattern used
# elsewhere in this codebase, e.g. 07_recovery.py / 09_ablation.py).
# =============================================================================

def _load_module(name, path):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_HERE = os.path.dirname(__file__)
_gwo_kd = _load_module("_finish_ablc_gwo_kd", os.path.join(_HERE, "06_gwo_kd.py"))

FeatureRegressor    = _gwo_kd.FeatureRegressor
quick_train_student = _gwo_kd.quick_train_student
get_device           = _gwo_kd.get_device

from utils.gwo import run_gwo
from models.teacher import build_teacher
from models.students import build_student


# =============================================================================
# Shared setup (identical protocol to 06_gwo_kd.py / 09_ablation.py)
# =============================================================================

def load_shared(device):
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


# =============================================================================
# Resume support
# =============================================================================

CSV_PATH = cfg.RESULTS_DIR / "ablation_gwo_config.csv"
CSV_FIELDS = ["n_wolves", "i_max", "seed", "best_fitness", "iterations_run",
              "wall_clock_s", "source"]
_csv_lock = Lock()


def load_done_set() -> set[tuple[int, int, int]]:
    done = set()
    if CSV_PATH.exists():
        with open(CSV_PATH, newline="") as f:
            for row in csv.DictReader(f):
                try:
                    done.add((int(float(row["n_wolves"])), int(float(row["i_max"])),
                              int(float(row["seed"]))))
                except (KeyError, ValueError):
                    continue
    return done


def append_result_row(row: dict):
    """Thread-safe, immediate append -- called as soon as one seed finishes."""
    with _csv_lock:
        write_header = not CSV_PATH.exists() or CSV_PATH.stat().st_size == 0
        with open(CSV_PATH, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            if write_header:
                writer.writeheader()
            writer.writerow(row)


# =============================================================================
# One (n_wolves, i_max, seed) job
# =============================================================================

def run_one_job(n_wolves, i_max, seed, shared, device):
    teacher, n_features, pos_weight, train_dataset, val_dataset, train_weights = shared

    def fitness(params: dict) -> float:
        s_idx   = params["s"]
        student = build_student(s_idx, n_features=n_features).to(device)
        reg     = FeatureRegressor(student.feature_dim, teacher.feature_dim).to(device)
        return quick_train_student(student, reg, teacher, params, device,
                                    n_epochs=5, pos_weight=pos_weight,
                                    train_dataset=train_dataset, val_dataset=val_dataset,
                                    train_weights=train_weights)

    t0 = time.time()
    log(f"  -> START  N_w={n_wolves:2d} I_max={i_max:2d} seed={seed}")
    _, best_score, history = run_gwo(
        fitness, n_wolves=n_wolves, i_max=i_max, seed=seed, verbose=False, n_workers=1,
    )
    elapsed = time.time() - t0
    log(f"  <- DONE   N_w={n_wolves:2d} I_max={i_max:2d} seed={seed}  "
        f"best={best_score:.4f}  iters={len(history)}  {elapsed:.1f}s")

    row = {
        "n_wolves": n_wolves, "i_max": i_max, "seed": seed,
        "best_fitness": best_score, "iterations_run": len(history),
        "wall_clock_s": round(elapsed, 1), "source": "finish_ablation_c.py",
    }
    append_result_row(row)
    return row


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=cfg.META_SEEDS)
    parser.add_argument("--workers", type=int, default=5,
                         help="Concurrent seed-jobs (default 5, one per seed). "
                              "Try higher if GPU VRAM allows (tiny 4K-param model); "
                              "pass 1 to force serial if parallel doesn't help.")
    args = parser.parse_args()

    log("=" * 60)
    log("Finish ablation (c) -- remaining GWO configuration grid points")
    log("=" * 60)

    device = get_device()
    log(f"  Device: {device}")
    torch.set_num_threads(max(1, (os.cpu_count() or 1) // max(1, args.workers)))
    log(f"  Concurrent seed-workers: {args.workers}  "
        f"(torch intra-op threads: {torch.get_num_threads()})")

    shared = load_shared(device)

    # The three remaining grid points from the reduced design (Methodology_Revised.md
    # 2.13c): N_w in {5,10,20,30} at I_max=10 (5,10,20 already done), and I_max in
    # {10,20,50} at N_w=10 (10 already done as the (10,10) baseline).
    remaining_configs = [(30, 10), (10, 20), (10, 50)]

    done = load_done_set()
    log(f"  Already-completed (n_wolves, i_max, seed) rows found in {CSV_PATH.name}: {len(done)}")

    jobs = []
    for n_w, i_max in remaining_configs:
        for seed in args.seeds:
            if (n_w, i_max, seed) in done:
                log(f"  SKIP (already done): N_w={n_w} I_max={i_max} seed={seed}")
                continue
            jobs.append((n_w, i_max, seed))

    if not jobs:
        log("  Nothing to do -- all remaining grid points already completed.")
        return

    log(f"  Jobs to run: {len(jobs)}  ({', '.join(f'N_w={a} I_max={b} seed={c}' for a,b,c in jobs)})")
    log("")

    t_start = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_one_job, n_w, i_max, seed, shared, device): (n_w, i_max, seed)
                   for n_w, i_max, seed in jobs}
        for fut in as_completed(futures):
            n_w, i_max, seed = futures[fut]
            try:
                fut.result()
            except Exception as e:
                log(f"  !! FAILED N_w={n_w} I_max={i_max} seed={seed}: {e}")

    log("")
    log(f"[All remaining ablation-(c) jobs finished in {time.time()-t_start:.1f}s]")
    log(f"  Results -> {CSV_PATH}")


if __name__ == "__main__":
    main()
