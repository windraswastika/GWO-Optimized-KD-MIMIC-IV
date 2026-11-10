"""
delong_clinical_baselines.py — DeLong significance: GWO-KD vs SOFA/MEWS/NEWS2
================================================================================
Completes Table 7/8 of the manuscript: computes the paired DeLong test between the
proposed GWO-KD student and each of the three rule-based clinical scores (SOFA, MEWS,
NEWS2), on the identical test split used everywhere else in this study.

Reuses (imports, does not duplicate) the scoring functions from clinical_baselines.py and
the model-building / metric utilities already used throughout this codebase.

Batched to avoid the slow single fancy-index read pattern (`X_all[test_idx]` in one shot)
that stalled an earlier ad-hoc run on a 6.7 GB memmapped array — reads the test set in
chunks instead, which is both faster and has a bounded memory footprint. Runs on GPU if
available for the (trivial) model-inference part; the DeLong computation itself is
CPU/numpy-bound regardless of device.

Run:
    python src/delong_clinical_baselines.py
"""

import sys, os, json, time, importlib.util
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch
from datetime import datetime

import config as cfg
from utils.metrics import auroc, delong_test
from models.students import build_student


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_clinical_scoring_module():
    """Import clinical_baselines.py's scoring functions without duplicating them."""
    path = os.path.join(os.path.dirname(__file__), "clinical_baselines.py")
    spec = importlib.util.spec_from_file_location("clinical_baselines_mod", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    t0 = time.time()
    log("=" * 60)
    log("DeLong test: GWO-KD vs SOFA / MEWS / NEWS2")
    log("=" * 60)

    device = get_device()
    log(f"  Device: {device}")

    cb = load_clinical_scoring_module()
    train_stats = json.load(open(cfg.PROCESSED_DIR / "train_stats.json"))

    log("  Loading GWO-KD checkpoint (student_gwo_kd_s2.pt) ...")
    ckpt_path = cfg.CHECKPOINTS_DIR / "student_gwo_kd_s2.pt"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    n_features = ckpt["n_features"]
    model = build_student(2, n_features=n_features).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    log(f"  Loaded: {ckpt_path.name}  (n_features={n_features})")

    log("  Loading test split indices and label array ...")
    test_idx = np.load(cfg.SPLITS_DIR / "test_idx.npy")
    y_all = np.load(cfg.PROCESSED_DIR / "windows_labels.npy")
    y_true = y_all[test_idx]
    log(f"  Test windows: {len(test_idx):,}  positives: {y_true.sum():,} ({y_true.mean():.1%})")

    log("  Loading feature memmap ...")
    X_all = np.load(cfg.PROCESSED_DIR / "windows_features.npy", mmap_mode="r")

    # ── Batched pass: model inference (full window) + last-timestep capture,
    #    in one read per chunk, avoiding a single giant fancy-index call. ──────
    BATCH = 8192
    n = len(test_idx)
    y_score_gwo = np.empty(n, dtype=np.float32)
    last_step = np.empty((n, n_features), dtype=np.float32)

    log(f"  Running batched inference + feature extraction (batch={BATCH}) ...")
    t_batch = time.time()
    with torch.no_grad():
        for start in range(0, n, BATCH):
            end = min(start + BATCH, n)
            idx_chunk = test_idx[start:end]
            # Single read of this chunk's full windows from the memmap.
            X_chunk = np.asarray(X_all[idx_chunk])          # (B, 12, F)
            last_step[start:end] = X_chunk[:, -1, :]

            xb = torch.from_numpy(X_chunk).float().to(device)
            logit, _ = model(xb)
            y_score_gwo[start:end] = torch.sigmoid(logit).cpu().numpy()

            if (start // BATCH) % 20 == 0:
                pct = end / n * 100
                log(f"    {end:,}/{n:,} ({pct:.1f}%) — {time.time()-t_batch:.1f}s elapsed")

    log(f"  Batched pass done in {time.time()-t_batch:.1f}s")
    gwo_auc = auroc(y_true, y_score_gwo)
    log(f"  GWO-KD recomputed test AUROC: {gwo_auc:.4f}")

    # ── Clinical scores from the captured last-timestep features ───────────────
    log("  Computing SOFA / MEWS / NEWS2 from captured last-timestep features ...")
    raw = cb.denormalise(last_step, train_stats)
    scores = {
        "SOFA":  cb.compute_sofa(raw),
        "MEWS":  cb.compute_mews(raw),
        "NEWS2": cb.compute_news2(raw),
    }

    # ── DeLong tests ─────────────────────────────────────────────────────────
    log("\n[DeLong tests: GWO-KD vs clinical scores]")
    rows = []
    for name, score in scores.items():
        t_dl = time.time()
        dt = delong_test(y_true, y_score_gwo, score)
        elapsed = time.time() - t_dl
        sig = "significant" if dt["significant"] else "n.s."
        log(f"  GWO-KD vs {name}: z={dt['z_stat']:.3f}  p={dt['p_value']:.6g}  "
            f"({sig})  [{elapsed:.1f}s]")
        rows.append({
            "comparison": f"GWO-KD vs {name}",
            "gwo_auroc": round(gwo_auc, 4),
            "other_auroc": round(auroc(y_true, score), 4),
            "delta": round(gwo_auc - auroc(y_true, score), 4),
            "z_stat": round(dt["z_stat"], 3),
            "p_value": dt["p_value"],
            "significant": dt["significant"],
        })

    import pandas as pd
    df = pd.DataFrame(rows)
    out_path = cfg.RESULTS_DIR / "delong_clinical_baselines.csv"
    df.to_csv(out_path, index=False)
    log(f"\n  Saved -> {out_path}")
    log("\n" + df.to_string(index=False))
    log(f"\n[Total runtime: {time.time()-t0:.1f}s]")
    log("[DeLong clinical baselines COMPLETE]")


if __name__ == "__main__":
    main()
