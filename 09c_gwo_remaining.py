import os
# os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import sys, time, argparse, importlib.util
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import torch
from datetime import datetime

import config as cfg
from utils.gwo import run_gwo_multiseed


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def _load(name, fname):
    path = os.path.join(os.path.dirname(__file__), fname)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_abl = _load("_abl_remaining", "09_ablation.py")


def shard_path(n_w, i_max):
    return cfg.RESULTS_DIR / f"ablation_gwo_config__nw{n_w}_imax{i_max}.csv"


def atomic_write(df, path):
    tmp = f"{path}.tmp.{os.getpid()}"
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def build_fitness(shared, device):
    teacher, n_features, pos_weight, train_dataset, val_dataset, train_weights = shared

    def fitness(params):
        student = _abl.build_student(params["s"], n_features=n_features).to(device)
        reg = _abl.FeatureRegressor(student.feature_dim, teacher.feature_dim).to(device)
        return _abl.quick_train_student(
            student, reg, teacher, params, device,
            n_epochs=5, pos_weight=pos_weight,
            train_dataset=train_dataset, val_dataset=val_dataset,
            train_weights=train_weights,
        )

    return fitness


def run_config(n_w, i_max, seeds, fitness, force):
    path = shard_path(n_w, i_max)

    done = {}
    if path.exists() and not force:
        prev = pd.read_csv(path)
        done = {int(s): r for s, r in zip(prev["seed"], prev.to_dict("records"))}
        log(f"  [N_w={n_w}, I_max={i_max}] resume: {sorted(done)} already done")

    rows = [done[s] for s in sorted(done)]
    for seed in seeds:
        if seed in done:
            continue
        log(f"  [N_w={n_w}, I_max={i_max}] seed={seed} ...")
        t0 = time.time()
        _, _, results = run_gwo_multiseed(
            fitness_fn=fitness, seeds=[seed], n_wolves=n_w, i_max=i_max,
            verbose=True, n_workers=1,
        )
        s, params, score, hist = results[0]
        rows.append({
            "n_wolves": n_w, "i_max": i_max, "seed": int(s),
            "best_fitness": float(score), "iterations_run": len(hist),
            "wall_clock_s": round(time.time() - t0, 1),
            "source": "09c_gwo_remaining.py",
        })
        atomic_write(pd.DataFrame(rows), path)
        torch.cuda.empty_cache()
        log(f"  [N_w={n_w}, I_max={i_max}] seed={seed} done, best={score:.4f}, "
            f"saved -> {path.name}")

    return pd.DataFrame(rows)


def do_merge():
    frames = []

    hist_path = cfg.RESULTS_DIR / "gwo_history.csv"
    if hist_path.exists():
        hdf = pd.read_csv(hist_path)
        last = hdf.loc[hdf.groupby("seed")["iteration"].idxmax()]
        frames.append(pd.DataFrame({
            "n_wolves": cfg.GWO_N_WOLVES, "i_max": cfg.GWO_I_MAX,
            "seed": last["seed"].astype(int).values,
            "best_fitness": last["alpha_score"].values, "source": "gwo_history.csv",
        }))

    for f in sorted(cfg.RESULTS_DIR.glob("ablation_gwo_config__nw*_imax*.csv")):
        frames.append(pd.read_csv(f))

    agg = cfg.RESULTS_DIR / "ablation_gwo_config.csv"
    if agg.exists():
        frames.append(pd.read_csv(agg))

    if not frames:
        log("  nothing to merge")
        return

    allrows = pd.concat(frames, ignore_index=True)
    allrows = allrows.drop_duplicates(subset=["n_wolves", "i_max", "seed"], keep="first")

    summary = (allrows.groupby(["n_wolves", "i_max"])["best_fitness"]
               .agg(["mean", "std", "count"]).reset_index()
               .sort_values(["i_max", "n_wolves"]))
    out = cfg.RESULTS_DIR / "ablation_gwo_config_final.csv"
    allrows.to_csv(out, index=False)
    log("\n" + summary.to_string(index=False))
    log(f"  Saved -> {out}")

    expected = [(nw, cfg.GWO_I_MAX) for nw in cfg.ABLATION_GWO_NW_GRID] + \
               [(cfg.GWO_N_WOLVES, im) for im in cfg.ABLATION_GWO_IMAX_GRID]
    have = set(map(tuple, summary[["n_wolves", "i_max"]].values))
    missing = [c for c in dict.fromkeys(expected) if c not in have]
    incomplete = summary[summary["count"] < len(cfg.META_SEEDS)]
    if missing:
        log(f"  MISSING configs (not yet on disk): {missing}")
    if len(incomplete):
        log(f"  INCOMPLETE configs (<{len(cfg.META_SEEDS)} seeds):\n{incomplete.to_string(index=False)}")


def parse_configs(items):
    out = []
    for it in items:
        nw, im = it.split(":")
        out.append((int(nw), int(im)))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--configs", nargs="+", default=["10:20", "10:50"])
    p.add_argument("--seeds", type=int, nargs="+", default=cfg.META_SEEDS)
    p.add_argument("--device", default=None)
    p.add_argument("--force", action="store_true")
    p.add_argument("--merge", action="store_true")
    args = p.parse_args()

    if args.merge:
        do_merge()
        return

    device = torch.device(args.device) if args.device else _abl.get_device()
    log(f"Device: {device}  PID: {os.getpid()}")

    shared = _abl.load_shared(device)
    fitness = build_fitness(shared, device)

    for n_w, i_max in parse_configs(args.configs):
        run_config(n_w, i_max, args.seeds, fitness, args.force)

    log("[09c COMPLETE] run with --merge to assemble the final table")


if __name__ == "__main__":
    main()