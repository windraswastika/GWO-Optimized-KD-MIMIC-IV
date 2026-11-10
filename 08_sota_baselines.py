"""
08_sota_baselines.py — Stage 8: SOTA Baseline Comparison (Section 2.13)
=========================================================================
Trains and evaluates established baselines for sepsis prediction:

  A) Classical ML (per-window feature aggregation):
     - XGBoost       (gradient boosted trees, mean/std/last features)
     - LightGBM      (fast GBDT)

  B) Deep learning:
     - Vanilla LSTM  (1-layer, 64 hidden) — no KD
     - Temporal CNN  (2-block, no dilation, no KD)

  C) SOTA reference (from recent literature):
     - RETAIN        (Choi et al. 2016 — attention RNN for EHR)

All evaluated with dual-threshold metrics and Bootstrap 95% CI.
Comparison vs GWO-KD via DeLong test + Holm–Bonferroni correction.

Run:
    python src/08_sota_baselines.py
"""

import sys, os, argparse
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datetime import datetime
from pathlib import Path

import config as cfg
from utils.data_utils import (
    make_dataloader, load_split_arrays, SepsisWindowDataset,
    measure_latency_cpu, model_size_mb, count_params
)
from utils.metrics import (
    auroc, auprc, dual_threshold_eval, bootstrap_auroc_ci,
    delong_test, holm_bonferroni, print_dual_threshold_summary
)


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def get_device():
    if torch.backends.mps.is_available(): return torch.device("mps")
    if torch.cuda.is_available():         return torch.device("cuda")
    return torch.device("cpu")


# =============================================================================
# A. Classical ML — aggregate window features (mean, std, last) → flat vector
# =============================================================================

def build_flat_features(split: str, n_features: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Aggregate 12×F window → 3F flat features: [mean, std, last_value].
    Used for XGBoost / LightGBM.
    """
    X_all = np.load(cfg.PROCESSED_DIR / "windows_features.npy", mmap_mode="r")
    y_all = np.load(cfg.PROCESSED_DIR / "windows_labels.npy")
    idx   = np.load(cfg.SPLITS_DIR / f"{split}_idx.npy")

    X = X_all[idx]   # (N, 12, F)
    y = y_all[idx].astype(np.float32)

    X_mean = X.mean(axis=1)          # (N, F)
    X_std  = X.std(axis=1)           # (N, F)
    X_last = X[:, -1, :]             # (N, F)  — last timestep

    X_flat = np.concatenate([X_mean, X_std, X_last], axis=1)   # (N, 3F)
    log(f"  [{split}] flat features: {X_flat.shape}  pos={y.sum():.0f} ({y.mean():.1%})")
    return X_flat, y


def train_xgboost(n_features: int) -> tuple[float, np.ndarray, np.ndarray]:
    try:
        import xgboost as xgb
    except ImportError:
        log("  XGBoost not installed: pip install xgboost")
        return None, None, None

    log("[XGBoost] Building features …")
    X_tr, y_tr = build_flat_features("train", n_features)
    X_va, y_va = build_flat_features("val",   n_features)
    X_te, y_te = build_flat_features("test",  n_features)

    scale_pos = int((y_tr == 0).sum() / (y_tr == 1).sum())
    log(f"  scale_pos_weight={scale_pos}")

    model = xgb.XGBClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=scale_pos,
        eval_metric="auc", early_stopping_rounds=20,
        tree_method="hist", random_state=cfg.RANDOM_SEED,
        n_jobs=-1, verbosity=0,
    )
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)

    y_score = model.predict_proba(X_te)[:, 1]
    return auroc(y_te, y_score), y_te, y_score


def train_lightgbm(n_features: int) -> tuple[float, np.ndarray, np.ndarray]:
    try:
        import lightgbm as lgb
    except ImportError:
        log("  LightGBM not installed: pip install lightgbm")
        return None, None, None

    log("[LightGBM] Building features …")
    X_tr, y_tr = build_flat_features("train", n_features)
    X_va, y_va = build_flat_features("val",   n_features)
    X_te, y_te = build_flat_features("test",  n_features)

    scale_pos = int((y_tr == 0).sum() / (y_tr == 1).sum())

    model = lgb.LGBMClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=scale_pos,
        random_state=cfg.RANDOM_SEED,
        n_jobs=-1, verbose=-1,
    )
    model.fit(X_tr, y_tr,
              eval_set=[(X_va, y_va)],
              callbacks=[lgb.early_stopping(20), lgb.log_evaluation(-1)])

    y_score = model.predict_proba(X_te)[:, 1]
    return auroc(y_te, y_score), y_te, y_score


# =============================================================================
# B. Deep learning baselines
# =============================================================================

class VanillaLSTM(nn.Module):
    """Single-layer LSTM, no KD, no attention."""
    def __init__(self, n_features: int, hidden: int = 64, dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, batch_first=True, dropout=dropout)
        self.fc   = nn.Linear(hidden, 1)
        self.feature_dim = hidden

    def forward(self, x):
        out, (h, _) = self.lstm(x)
        feat  = h[-1]           # last hidden state
        logit = self.fc(feat).squeeze(-1)
        return logit, feat


class SimpleTCN(nn.Module):
    """2-block temporal CNN without dilation (baseline, not teacher TCN)."""
    def __init__(self, n_features: int, filters: int = 32):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(n_features, filters, kernel_size=3, padding=1),
            nn.ReLU(), nn.BatchNorm1d(filters))
        self.conv2 = nn.Sequential(
            nn.Conv1d(filters, filters, kernel_size=3, padding=1),
            nn.ReLU(), nn.BatchNorm1d(filters))
        self.pool  = nn.AdaptiveAvgPool1d(1)
        self.fc    = nn.Linear(filters, 1)
        self.feature_dim = filters

    def forward(self, x):
        x = x.permute(0, 2, 1)   # (B, F, T)
        h = self.conv2(self.conv1(x))
        feat  = self.pool(h).squeeze(-1)
        logit = self.fc(feat).squeeze(-1)
        return logit, feat


class RETAIN(nn.Module):
    """
    RETAIN: Reverse Time Attention (Choi et al. 2016).
    Two GRUs: alpha (visit-level) + beta (feature-level) attention.
    """
    def __init__(self, n_features: int, hidden: int = 64):
        super().__init__()
        self.emb      = nn.Linear(n_features, hidden)
        self.alpha_rnn= nn.GRU(hidden, hidden, batch_first=True)
        self.beta_rnn = nn.GRU(hidden, hidden, batch_first=True)
        self.alpha_fc = nn.Linear(hidden, 1)
        self.beta_fc  = nn.Linear(hidden, hidden)
        self.fc       = nn.Linear(hidden, 1)
        self.feature_dim = hidden

    def forward(self, x):
        # x: (B, T, F)
        v = torch.relu(self.emb(x))                        # (B, T, H)
        # Reverse time
        v_rev = torch.flip(v, [1])
        a_out, _ = self.alpha_rnn(v_rev)                  # (B, T, H)
        b_out, _ = self.beta_rnn(v_rev)                   # (B, T, H)
        # Flip back
        a_out = torch.flip(a_out, [1])
        b_out = torch.flip(b_out, [1])
        # Attention weights
        alpha = torch.softmax(self.alpha_fc(a_out), dim=1)  # (B, T, 1)
        beta  = torch.tanh(self.beta_fc(b_out))             # (B, T, H)
        # Context vector
        c = (alpha * beta * v).sum(dim=1)                  # (B, H)
        logit = self.fc(c).squeeze(-1)
        return logit, c



# =============================================================================
# A100 GPU-resident helpers
# =============================================================================
_G8: dict = {}

def _g8_train(bs=4096):
    p=_G8["pi"]; n=_G8["ni"]; X=_G8["X"]; y=_G8["y"]
    half=bs//2; nb=len(n)//half
    pi2=p[torch.randint(0,len(p),(nb*half,))]
    ni2=n[torch.randperm(len(n))[:nb*half]]
    for i in range(nb):
        s=i*half; idx=torch.cat([pi2[s:s+half],ni2[s:s+half]]); yield X[idx],y[idx]

def _g8_val(bs=4096):
    X=_G8["Xv"]; y=_G8["yv"]
    for s in range(0,len(y),bs): yield X[s:s+bs],y[s:s+bs]

def _g8_test(bs=4096):
    X=_G8["Xt"]; y=_G8["yt"]
    for s in range(0,len(y),bs): yield X[s:s+bs],y[s:s+bs]


def train_dl_baseline(
    model: nn.Module,
    name : str,
    device: torch.device,
    pos_weight: torch.Tensor,
    max_epochs: int = 30,
    patience  : int = 10,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Train a deep learning baseline with standard BCE + weighted sampler."""
    model = model.to(device)   # pastikan model di GPU
    def _tr(): return _g8_train(4096)   # fresh GPU iter setiap epoch
    def _vl(): return _g8_val(4096)
    test_ld = _g8_test(4096)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5, min_lr=1e-6)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_auc   = -1.0
    best_state = None
    pat_cnt    = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        for X_b, y_b in _tr():
            X_b, y_b = X_b.to(device), y_b.to(device)
            logit, _ = model(X_b)
            loss = criterion(logit, y_b)
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        sc, lb = [], []
        with torch.no_grad():
            for X_b, y_b in _vl():
                logit, _ = model(X_b.to(device))
                sc.append(torch.sigmoid(logit).cpu().numpy())
                lb.append(y_b.cpu().numpy())
        val_auc = auroc(np.concatenate(lb), np.concatenate(sc))
        scheduler.step(val_auc)
        log(f"  [{name}] Epoch {epoch:2d} | val AUROC {val_auc:.4f}")

        if val_auc > best_auc + 1e-5:
            best_auc = val_auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            pat_cnt = 0
        else:
            pat_cnt += 1
            if pat_cnt >= patience:
                log(f"  [{name}] Early stop at epoch {epoch}"); break

    model.load_state_dict(best_state)
    model.eval()
    sc, lb = [], []
    with torch.no_grad():
        for X_b, y_b in test_ld:
            logit, _ = model(X_b.to(device))
            sc.append(torch.sigmoid(logit).cpu().numpy())
            lb.append(y_b.cpu().numpy())
    return best_auc, np.concatenate(lb), np.concatenate(sc)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip_classical", action="store_true")
    parser.add_argument("--skip_dl",        action="store_true")
    args = parser.parse_args()

    log("=" * 60)
    log("Stage 08 — SOTA Baseline Comparison")
    log("=" * 60)

    device = get_device()
    log(f"  Device: {device}")

    X_sample   = np.load(cfg.PROCESSED_DIR / "windows_features.npy", mmap_mode="r")
    n_features = X_sample.shape[2]

    X_train, y_train = load_split_arrays("train")
    X_val,   y_val   = load_split_arrays("val")
    X_test,  y_test  = load_split_arrays("test")
    n_pos = y_train.sum(); n_neg = len(y_train) - n_pos
    pos_weight = torch.tensor(min(n_neg/n_pos,10.0),dtype=torch.float32).to(device)
    log(f"  pos_weight: {pos_weight.item():.2f}")
    log("  Loading data to GPU VRAM...")
    _G8["X"]  = torch.from_numpy(X_train).float().to(device)
    _G8["y"]  = torch.from_numpy(y_train.astype("float32")).to(device)
    _G8["Xv"] = torch.from_numpy(X_val).float().to(device)
    _G8["yv"] = torch.from_numpy(y_val.astype("float32")).to(device)
    _G8["Xt"] = torch.from_numpy(X_test).float().to(device)
    _G8["yt"] = torch.from_numpy(y_test.astype("float32")).to(device)
    _G8["pi"] = torch.where(_G8["y"]==1)[0]
    _G8["ni"] = torch.where(_G8["y"]==0)[0]
    log(f"  GPU ready: {torch.cuda.memory_allocated()/1e9:.1f} GB")

    rows = []

    # ── Load GWO-KD result for DeLong comparison ──────────────────────────────
    gwo_y_true, gwo_y_score = None, None
    gwo_paths = sorted(cfg.CHECKPOINTS_DIR.glob("student_gwo_kd_s*.pt"))
    if gwo_paths:
        from models.students import build_student
        ckpt_g  = torch.load(gwo_paths[0], map_location="cpu", weights_only=False)
        s_idx   = int(ckpt_g["model_name"].replace("s",""))
        mdl_g   = build_student(s_idx, n_features=n_features)
        mdl_g.load_state_dict(ckpt_g["state_dict"])
        mdl_g = mdl_g.to(device)   # FIX: model ke GPU
        mdl_g.eval()
        test_ld = _g8_test(4096)
        sc, lb  = [], []
        with torch.no_grad():
            for X_b, y_b in test_ld:
                logit, _ = mdl_g(X_b)
                sc.append(torch.sigmoid(logit).cpu().numpy()); lb.append(y_b.cpu().numpy())
        gwo_y_true  = np.concatenate(lb)
        gwo_y_score = np.concatenate(sc)
        log(f"  GWO-KD loaded: AUROC={auroc(gwo_y_true, gwo_y_score):.4f}")

    # ── A. Classical ML ───────────────────────────────────────────────────────
    if not args.skip_classical:
        log("\n[A] Classical ML")
        for name, fn in [("XGBoost", train_xgboost), ("LightGBM", train_lightgbm)]:
            log(f"\n[{name}]")
            test_auc, y_true, y_score = fn(n_features)
            if y_true is None:
                continue
            dual = dual_threshold_eval(y_true, y_score, cfg.EVAL_SENSITIVITY_TARGET)
            auc_pt, auc_lo, auc_hi = bootstrap_auroc_ci(y_true, y_score, n_resamples=500)
            print_dual_threshold_summary(name, dual)
            rows.append({
                "model"        : name,
                "type"         : "Classical ML",
                "n_params"     : "N/A",
                "size_mb"      : "N/A",
                "latency_ms"   : "N/A",
                "test_auroc_ci": f"{auc_pt:.4f} [{auc_lo:.4f}–{auc_hi:.4f}]",
                "test_auprc"   : round(dual["auprc"], 4),
                "youden_sens"  : round(dual["youden_sensitivity"], 3),
                "youden_spec"  : round(dual["youden_specificity"], 3),
                "youden_f1"    : round(dual["youden_f1"], 3),
                "clin_sens"    : round(dual["clin_sensitivity"], 3),
                "y_true"       : y_true,
                "y_score"      : y_score,
            })

    # ── B. DL baselines ───────────────────────────────────────────────────────
    if not args.skip_dl:
        log("\n[B] Deep Learning Baselines")
        dl_models = {
            "VanillaLSTM": VanillaLSTM(n_features=n_features, hidden=64),
            "SimpleTCN"  : SimpleTCN(n_features=n_features, filters=32),
            "RETAIN"     : RETAIN(n_features=n_features, hidden=64),
        }
        for name, model in dl_models.items():
            log(f"\n[{name}]  params={count_params(model):,}")
            model = model.to(device)
            torch.manual_seed(cfg.RANDOM_SEED)
            _, y_true, y_score = train_dl_baseline(
                model, name, device, pos_weight)
            dual   = dual_threshold_eval(y_true, y_score, cfg.EVAL_SENSITIVITY_TARGET)
            auc_pt, auc_lo, auc_hi = bootstrap_auroc_ci(y_true, y_score, n_resamples=500)
            lat    = measure_latency_cpu(model.cpu(), n_features=n_features)
            print_dual_threshold_summary(name, dual)
            rows.append({
                "model"        : name,
                "type"         : "DL Baseline",
                "n_params"     : count_params(model),
                "size_mb"      : round(model_size_mb(model), 4),
                "latency_ms"   : round(lat, 2),
                "test_auroc_ci": f"{auc_pt:.4f} [{auc_lo:.4f}–{auc_hi:.4f}]",
                "test_auprc"   : round(dual["auprc"], 4),
                "youden_sens"  : round(dual["youden_sensitivity"], 3),
                "youden_spec"  : round(dual["youden_specificity"], 3),
                "youden_f1"    : round(dual["youden_f1"], 3),
                "clin_sens"    : round(dual["clin_sensitivity"], 3),
                "y_true"       : y_true,
                "y_score"      : y_score,
            })

    # ── DeLong vs GWO-KD ─────────────────────────────────────────────────────
    if gwo_y_true is not None and rows:
        log("\n[DeLong: GWO-KD vs each SOTA baseline]")
        p_vals, delong_rows = [], []
        for r in rows:
            if "y_true" not in r: continue
            dt = delong_test(gwo_y_true, gwo_y_score, r["y_score"])
            gwo_auc = auroc(gwo_y_true, gwo_y_score)
            oth_auc = auroc(r["y_true"], r["y_score"])
            p_vals.append(dt["p_value"])
            delong_rows.append({
                "comparison"  : f"GWO-KD vs {r['model']}",
                "gwo_auroc"   : round(gwo_auc, 4),
                "other_auroc" : round(oth_auc, 4),
                "delta"       : round(gwo_auc - oth_auc, 4),
                "z_stat"      : round(dt["z_stat"], 3),
                "p_value"     : round(dt["p_value"], 4),
                "significant" : dt["significant"],
            })
            log(f"  GWO-KD vs {r['model']}: z={dt['z_stat']:.3f} "
                f"p={dt['p_value']:.4f} {'✓' if dt['significant'] else '✗'}")

        # Holm–Bonferroni
        corrected = holm_bonferroni(p_vals)
        for i, row in enumerate(delong_rows):
            row["p_corrected"] = corrected[i]
            row["sig_corrected"] = corrected[i] < cfg.PVALUE_THRESHOLD

        pd.DataFrame(delong_rows).to_csv(
            cfg.RESULTS_DIR / "delong_sota.csv", index=False)

    # ── Save summary ──────────────────────────────────────────────────────────
    out_rows = [{k: v for k, v in r.items() if k not in ("y_true","y_score")}
                for r in rows]
    df = pd.DataFrame(out_rows)
    out_path = cfg.RESULTS_DIR / "sota_baselines.csv"
    df.to_csv(out_path, index=False)
    log(f"\n  Saved → {out_path}")
    if len(df):
        log("\n" + df[["model","n_params","test_auroc_ci",
                        "youden_sens","clin_sens"]].to_string(index=False))

    log("\n[Stage 08 COMPLETE] ✓")
    log("  → Run next: python src/09_ablation.py")


if __name__ == "__main__":
    main()
