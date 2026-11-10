"""
07_baselines.py — Stage 7: Metaheuristic & KD Baselines (Section 2.11–2.12)
=============================================================================
Two sets of baselines to compare against GWO-KD:

A) Metaheuristic baselines (same budget as GWO):
   - PSO  (Particle Swarm Optimisation)
   - GA   (Genetic Algorithm)
   - WOA  (Whale Optimisation Algorithm)
   Each run with N_w=10, I_max=10, ≥5 seeds.
   → Wilcoxon signed-rank test vs GWO (Section 2.15)

B) Classical KD baselines (same student, no metaheuristic):
   - Vanilla KD   (Hinton 2015: KL only, no feature align)
   - FitNet       (Romero 2015: feature MSE only, no KL)
   - AT            (Attention transfer: mean channel attention)
   → DeLong test vs GWO-KD (Section 2.15)

Run:
    python src/07_baselines.py
    python src/07_baselines.py --skip_meta   # KD baselines only
    python src/07_baselines.py --skip_kd     # metaheuristic only
"""

import sys, os, argparse, time
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
from pathlib import Path

import config as cfg
from utils.data_utils import (
    make_dataloader, load_split_arrays,
    measure_latency_cpu, model_size_mb, count_params
)
from utils.metrics import (
    auroc, dual_threshold_eval, bootstrap_auroc_ci,
    delong_test, wilcoxon_test, holm_bonferroni
)
from models.teacher import build_teacher
from models.students import build_student
from utils.gwo import decode_position, LOWER, UPPER, PARAM_NAMES

PARAM_BOUNDS = list(zip(LOWER, UPPER))

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def get_device():
    if torch.backends.mps.is_available(): return torch.device("mps")
    if torch.cuda.is_available():         return torch.device("cuda")
    return torch.device("cpu")


# =============================================================================
# A100 Optimisation — GPU-resident data (loaded once, shared across all evals)
# =============================================================================
_GPU: dict = {}   # set once in main(): X_train, y_train, X_val, y_val, pos_idx, neg_idx

def _gpu_train_iter(bs: int = 4096):
    """Class-balanced GPU batch iterator — no disk I/O, no torch.multinomial."""
    pos_idx = _GPU["pos_idx"]; neg_idx = _GPU["neg_idx"]
    X = _GPU["X_train"];       y = _GPU["y_train"]
    dev = X.device
    half = bs // 2
    n_batches = len(neg_idx) // half
    p_idx = pos_idx[torch.randint(0, len(pos_idx), (n_batches * half,), device=dev)]
    n_idx = neg_idx[torch.randperm(len(neg_idx), device=dev)[:n_batches * half]]
    for i in range(n_batches):
        s = i * half
        idx = torch.cat([p_idx[s:s+half], n_idx[s:s+half]])
        yield X[idx], y[idx]

def _gpu_val_iter(bs: int = 4096):
    """Sequential GPU batch iterator for validation."""
    X = _GPU["X_val"]; y = _GPU["y_val"]
    for s in range(0, len(y), bs):
        yield X[s:s+bs], y[s:s+bs]



# =============================================================================
# A. Metaheuristic optimisers  (same search space as GWO)
# =============================================================================

# ── PSO ───────────────────────────────────────────────────────────────────────
def run_pso(fitness_fn, n_wolves=10, i_max=10, seed=0,
            epsilon=1e-4, patience=5, verbose=True):
    """Particle Swarm Optimisation."""
    rng = np.random.default_rng(seed)
    n_dim = len(PARAM_NAMES)
    w, c1, c2 = 0.7, 1.5, 1.5   # inertia, cognitive, social

    pos = rng.uniform(LOWER, UPPER, (n_wolves, n_dim))
    vel = rng.uniform(-0.5, 0.5, (n_wolves, n_dim))
    pbest_pos   = pos.copy()
    pbest_score = np.full(n_wolves, -np.inf)
    gbest_pos   = pos[0].copy()
    gbest_score = -np.inf

    history = []
    no_improve = 0
    prev_best  = -np.inf

    for it in range(1, i_max + 1):
        scores = np.array([fitness_fn(decode_position(pos[i])) for i in range(n_wolves)])
        for i in range(n_wolves):
            if scores[i] > pbest_score[i]:
                pbest_score[i] = scores[i]; pbest_pos[i] = pos[i].copy()
            if scores[i] > gbest_score:
                gbest_score = scores[i]; gbest_pos = pos[i].copy()

        history.append({"iteration": it, "alpha_score": gbest_score,
                         "alpha_params": decode_position(gbest_pos)})
        if verbose:
            p = decode_position(gbest_pos)
            print(f"  [PSO it={it:2d}] best={gbest_score:.4f} | "
                  f"T={p['T']:.1f} s={p['s']}", flush=True)

        if gbest_score - prev_best < epsilon: no_improve += 1
        else: no_improve = 0
        prev_best = gbest_score
        if no_improve >= patience: break

        r1, r2 = rng.random((n_wolves, n_dim)), rng.random((n_wolves, n_dim))
        vel = (w * vel
               + c1 * r1 * (pbest_pos - pos)
               + c2 * r2 * (gbest_pos - pos))
        pos = np.clip(pos + vel, LOWER, UPPER)

    return decode_position(gbest_pos), gbest_score, history


# ── GA ────────────────────────────────────────────────────────────────────────
def run_ga(fitness_fn, n_wolves=10, i_max=10, seed=0,
           epsilon=1e-4, patience=5, verbose=True):
    """Genetic Algorithm with tournament selection, BLX-α crossover, Gaussian mutation."""
    rng = np.random.default_rng(seed)
    n_dim = len(PARAM_NAMES)
    pop = rng.uniform(LOWER, UPPER, (n_wolves, n_dim))
    scores = np.array([fitness_fn(decode_position(pop[i])) for i in range(n_wolves)])

    best_score = scores.max(); best_pos = pop[scores.argmax()].copy()
    history = []
    no_improve = 0; prev_best = -np.inf

    for it in range(1, i_max + 1):
        new_pop = []
        for _ in range(n_wolves):
            # Tournament selection (size 3)
            idx = rng.choice(n_wolves, 3, replace=False)
            p1  = pop[idx[np.argmax(scores[idx])]]
            idx = rng.choice(n_wolves, 3, replace=False)
            p2  = pop[idx[np.argmax(scores[idx])]]
            # BLX-0.5 crossover
            alpha = 0.5
            lo    = np.minimum(p1, p2) - alpha * np.abs(p1 - p2)
            hi    = np.maximum(p1, p2) + alpha * np.abs(p1 - p2)
            child = rng.uniform(lo, hi)
            # Gaussian mutation (σ=0.1 * range)
            if rng.random() < 0.2:
                child += rng.normal(0, 0.1 * (UPPER - LOWER))
            new_pop.append(np.clip(child, LOWER, UPPER))

        pop    = np.array(new_pop)
        scores = np.array([fitness_fn(decode_position(pop[i])) for i in range(n_wolves)])
        best_idx = scores.argmax()
        if scores[best_idx] > best_score:
            best_score = scores[best_idx]; best_pos = pop[best_idx].copy()

        history.append({"iteration": it, "alpha_score": best_score,
                         "alpha_params": decode_position(best_pos)})
        if verbose:
            p = decode_position(best_pos)
            print(f"  [GA  it={it:2d}] best={best_score:.4f} | "
                  f"T={p['T']:.1f} s={p['s']}", flush=True)

        if best_score - prev_best < epsilon: no_improve += 1
        else: no_improve = 0
        prev_best = best_score
        if no_improve >= patience: break

    return decode_position(best_pos), best_score, history


# ── WOA ───────────────────────────────────────────────────────────────────────
def run_woa(fitness_fn, n_wolves=10, i_max=10, seed=0,
            epsilon=1e-4, patience=5, verbose=True):
    """Whale Optimisation Algorithm."""
    rng = np.random.default_rng(seed)
    n_dim = len(PARAM_NAMES)
    pos = rng.uniform(LOWER, UPPER, (n_wolves, n_dim))
    scores = np.array([fitness_fn(decode_position(pos[i])) for i in range(n_wolves)])
    best_idx = scores.argmax(); best_pos = pos[best_idx].copy(); best_score = scores[best_idx]

    history = []
    no_improve = 0; prev_best = -np.inf

    for it in range(1, i_max + 1):
        a = 2 - 2 * it / i_max
        for i in range(n_wolves):
            r = rng.random(); A = 2 * a * r - a; C = 2 * rng.random()
            if abs(A) < 1:
                if rng.random() < 0.5:  # shrinking encircling
                    D = abs(C * best_pos - pos[i])
                    pos[i] = np.clip(best_pos - A * D, LOWER, UPPER)
                else:                   # spiral update
                    b = 1; l = rng.uniform(-1, 1)
                    D = abs(best_pos - pos[i])
                    pos[i] = np.clip(D * np.exp(b * l) * np.cos(2 * np.pi * l) + best_pos,
                                     LOWER, UPPER)
            else:                       # random whale search
                rand_pos = pos[rng.integers(n_wolves)]
                D = abs(C * rand_pos - pos[i])
                pos[i] = np.clip(rand_pos - A * D, LOWER, UPPER)

        scores = np.array([fitness_fn(decode_position(pos[i])) for i in range(n_wolves)])
        best_idx = scores.argmax()
        if scores[best_idx] > best_score:
            best_score = scores[best_idx]; best_pos = pos[best_idx].copy()

        history.append({"iteration": it, "alpha_score": best_score,
                         "alpha_params": decode_position(best_pos)})
        if verbose:
            p = decode_position(best_pos)
            print(f"  [WOA it={it:2d}] best={best_score:.4f} | "
                  f"T={p['T']:.1f} s={p['s']}", flush=True)

        if best_score - prev_best < epsilon: no_improve += 1
        else: no_improve = 0
        prev_best = best_score
        if no_improve >= patience: break

    return decode_position(best_pos), best_score, history


# =============================================================================
# B. Classical KD baselines (no metaheuristic search)
# =============================================================================

class FeatureRegressor(nn.Module):
    def __init__(self, s_dim, t_dim):
        super().__init__()
        self.proj = nn.Linear(s_dim, t_dim)
    def forward(self, x): return self.proj(x)


def train_kd_baseline(
    method   : str,          # 'vanilla', 'fitnet', 'at'
    student  : nn.Module,
    teacher  : nn.Module,
    device   : torch.device,
    pos_weight: torch.Tensor,
    params   : dict = None,  # T, alpha_kd, beta_kd (fixed defaults if None)
    max_epochs: int = 50,
    patience  : int = 10,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Train a classical KD baseline (no metaheuristic search)."""
    if params is None:
        params = {"T": 4.0, "alpha_kd": 0.5, "beta_kd": 0.1}

    reg = FeatureRegressor(student.feature_dim, teacher.feature_dim).to(device)
    optimizer = torch.optim.Adam(
        list(student.parameters()) + list(reg.parameters()), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5, min_lr=1e-6)
    bce_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # A100: GPU-resident, class-balanced, no disk I/O
    def _fresh_train(): return _gpu_train_iter(4096)
    def _fresh_val():   return _gpu_val_iter(4096)

    best_auc = -1.0; best_state = None; patience_cnt = 0
    teacher.eval()

    for epoch in range(1, max_epochs + 1):
        student.train(); reg.train()
        for X_b, y_b in _fresh_train():
            X_b, y_b = X_b.to(device), y_b.to(device)
            with torch.no_grad():
                z_t, f_t = teacher(X_b)
            z_s, f_s = student(X_b)
            f_proj = reg(f_s)

            T, alpha, beta = params["T"], params["alpha_kd"], params["beta_kd"]

            if method == "vanilla":
                # Hinton 2015: only KL soft targets + BCE
                logits_t = torch.stack([-z_t, z_t], dim=1) / T
                logits_s = torch.stack([-z_s, z_s], dim=1) / T
                L_kl = F.kl_div(F.log_softmax(logits_s, 1),
                                 F.softmax(logits_t, 1).detach(),
                                 reduction="batchmean") * T**2
                loss = alpha * bce_fn(z_s, y_b) + (1 - alpha) * L_kl

            elif method == "fitnet":
                # Romero 2015: only feature MSE (hint-based)
                loss = bce_fn(z_s, y_b) + beta * F.mse_loss(f_proj, f_t.detach())

            elif method == "at":
                # Attention transfer: mean channel attention on features
                # For 1D features: just use L2 normalised feature matching
                f_t_norm = F.normalize(f_t.detach(), dim=1)
                f_s_norm = F.normalize(f_proj, dim=1)
                loss = bce_fn(z_s, y_b) + beta * F.mse_loss(f_s_norm, f_t_norm)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(student.parameters()) + list(reg.parameters()), 1.0)
            optimizer.step()

        student.eval()
        scores, labels = [], []
        with torch.no_grad():
            for X_b, y_b in _fresh_val():
                z_s, _ = student(X_b.to(device))
                scores.append(torch.sigmoid(z_s).cpu().numpy())
                labels.append(y_b.cpu().numpy())
        y_score = np.concatenate(scores)
        y_true  = np.concatenate(labels)
        val_auc = auroc(y_true, y_score)
        scheduler.step(val_auc)

        if val_auc > best_auc + 1e-5:
            best_auc = val_auc
            best_state = {k: v.cpu().clone() for k, v in student.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                break

    student.load_state_dict(best_state)
    return best_auc, y_true, y_score


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher_ckpt", default=str(cfg.CHECKPOINTS_DIR / "teacher_best.pt"))
    parser.add_argument("--gwo_result",   default=str(cfg.RESULTS_DIR / "gwo_kd_result.csv"))
    parser.add_argument("--skip_meta",    action="store_true")
    parser.add_argument("--skip_kd",      action="store_true")
    args = parser.parse_args()

    log("=" * 60)
    log("Stage 07 — Metaheuristic + KD Baselines")
    log("=" * 60)

    device = get_device()
    log(f"  Device: {device}")

    # Load teacher
    ckpt     = torch.load(args.teacher_ckpt, map_location=device, weights_only=False)
    teacher  = build_teacher(ckpt["model_name"], n_features=ckpt["n_features"]).to(device)
    teacher.load_state_dict(ckpt["state_dict"])
    teacher.eval()
    n_features = ckpt["n_features"]
    log(f"  Teacher: {ckpt['model_name'].upper()} val AUROC={ckpt['val_auroc']:.4f}")

    X_train, y_train = load_split_arrays("train")
    X_val,   y_val   = load_split_arrays("val")
    n_pos = y_train.sum(); n_neg = len(y_train) - n_pos
    pos_weight = torch.tensor(min(n_neg / n_pos, 10.0), dtype=torch.float32).to(device)

    # A100: load data to GPU once (train ~4.7 GB + val ~1.0 GB)
    log("  Loading data to GPU VRAM...")
    _GPU["X_train"] = torch.from_numpy(X_train).float().to(device)
    _GPU["y_train"] = torch.from_numpy(y_train.astype("float32")).to(device)
    _GPU["X_val"]   = torch.from_numpy(X_val).float().to(device)
    _GPU["y_val"]   = torch.from_numpy(y_val.astype("float32")).to(device)
    _GPU["pos_idx"] = torch.where(_GPU["y_train"] == 1)[0]
    _GPU["neg_idx"] = torch.where(_GPU["y_train"] == 0)[0]
    log(f"  GPU data ready: {torch.cuda.memory_allocated()/1e9:.1f} GB used")

    test_loader = make_dataloader("test", batch_size=4096,
                                  shuffle=False, num_workers=0)

    all_results = []   # collect (method, test_AUROC, y_true, y_score)

    # ── Load GWO result for comparison ───────────────────────────────────────
    gwo_auroc = None
    if Path(args.gwo_result).exists():
        gwo_df = pd.read_csv(args.gwo_result)
        gwo_auroc = float(gwo_df["test_auroc"].iloc[0])
        log(f"  GWO-KD test AUROC: {gwo_auroc:.4f}")

    # =========================================================================
    # A. Metaheuristic baselines
    # =========================================================================
    if not args.skip_meta:
        log("\n" + "="*50)
        log("A. Metaheuristic Baselines (PSO, GA, WOA)")
        log("="*50)

        META_OPTIMISERS = {"PSO": run_pso, "GA": run_ga, "WOA": run_woa}
        meta_seed_scores = {"GWO": [], "PSO": [], "GA": [], "WOA": []}

        # Load GWO per-seed scores
        gwo_hist_path = cfg.RESULTS_DIR / "gwo_history.csv"
        if gwo_hist_path.exists():
            gwo_hist = pd.read_csv(gwo_hist_path)
            for seed in cfg.META_SEEDS:
                best = gwo_hist[gwo_hist["seed"] == seed]["alpha_score"].max()
                meta_seed_scores["GWO"].append(best)

        for meta_name, meta_fn in META_OPTIMISERS.items():
            log(f"\n[{meta_name}] N_w={cfg.GWO_N_WOLVES}  I_max={cfg.GWO_I_MAX}  seeds={cfg.META_SEEDS}")
            seed_scores = []

            for seed in cfg.META_SEEDS:
                log(f"  Seed {seed} …")
                # Quick fitness (same as GWO: 5-epoch val AUROC)
                def fitness(params):
                    s = build_student(params["s"], n_features=n_features).to(device)
                    reg = FeatureRegressor(s.feature_dim, teacher.feature_dim).to(device)
                    opt = torch.optim.Adam(list(s.parameters()) + list(reg.parameters()), lr=1e-3)
                    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
                    train_ld = _gpu_train_iter(4096)  # GPU-resident
                    T, alpha, beta = params["T"], params["alpha_kd"], params["beta_kd"]
                    s.train(); reg.train(); teacher.eval()
                    for _ in range(5):
                        for X_b, y_b in train_ld:
                            X_b, y_b = X_b.to(device), y_b.to(device)
                            with torch.no_grad(): z_t, f_t = teacher(X_b)
                            z_s, f_s = s(X_b)
                            f_proj = reg(f_s)
                            logits_t = torch.stack([-z_t, z_t], dim=1) / T
                            logits_s_t = torch.stack([-z_s, z_s], dim=1) / T
                            L_kl = F.kl_div(F.log_softmax(logits_s_t, 1),
                                            F.softmax(logits_t, 1).detach(),
                                            reduction="batchmean") * T**2
                            L_feat = F.mse_loss(f_proj, f_t.detach())
                            loss = alpha * bce(z_s, y_b) + (1-alpha)*L_kl + beta*L_feat
                            opt.zero_grad(); loss.backward(); opt.step()
                    s.eval()
                    val_ld = _gpu_val_iter(4096)  # GPU-resident
                    sc, lb = [], []
                    with torch.no_grad():
                        for X_b, y_b in val_ld:
                            z_s, _ = s(X_b.to(device))
                            sc.append(torch.sigmoid(z_s).cpu().numpy())
                            lb.append(y_b.cpu().numpy())
                    return auroc(np.concatenate(lb), np.concatenate(sc))

                _, score, _ = meta_fn(fitness, n_wolves=cfg.GWO_N_WOLVES,
                                       i_max=cfg.GWO_I_MAX, seed=seed, verbose=True)
                seed_scores.append(score)
                log(f"  → seed={seed} best score={score:.4f}")

            meta_seed_scores[meta_name] = seed_scores
            log(f"  {meta_name} scores: {[f'{s:.4f}' for s in seed_scores]}")
            log(f"  {meta_name} mean={np.mean(seed_scores):.4f} ± {np.std(seed_scores):.4f}")

        # Wilcoxon tests: GWO vs each
        log("\n[Wilcoxon signed-rank tests: GWO vs competitors]")
        wilcox_rows = []
        p_vals = []
        for meta_name in ["PSO", "GA", "WOA"]:
            if len(meta_seed_scores[meta_name]) == len(meta_seed_scores["GWO"]):
                wt = wilcoxon_test(meta_seed_scores["GWO"], meta_seed_scores[meta_name])
                p_vals.append(wt["p_value"])
                wilcox_rows.append({
                    "comparison"  : f"GWO vs {meta_name}",
                    "gwo_mean"    : np.mean(meta_seed_scores["GWO"]),
                    "other_mean"  : np.mean(meta_seed_scores[meta_name]),
                    "w_stat"      : wt.get("statistic", None),
                    "p_value"     : wt["p_value"],
                    "significant" : wt["significant"],
                })
                log(f"  GWO vs {meta_name}: p={wt['p_value']:.4f}  "
                    f"{'✓ sig' if wt['significant'] else '✗ n.s.'}")

        # Holm–Bonferroni correction
        if p_vals:
            corrected = holm_bonferroni(p_vals)
            for i, row in enumerate(wilcox_rows):
                row["p_corrected"] = corrected[i]

        pd.DataFrame(wilcox_rows).to_csv(
            cfg.RESULTS_DIR / "wilcoxon_meta.csv", index=False)
        log(f"  Wilcoxon results → {cfg.RESULTS_DIR}/wilcoxon_meta.csv")

    # =========================================================================
    # B. Classical KD baselines
    # =========================================================================
    if not args.skip_kd:
        log("\n" + "="*50)
        log("B. Classical KD Baselines (Vanilla KD, FitNet, AT)")
        log("="*50)

        # Use best student from GWO (or default S1)
        gwo_s = 1
        if Path(args.gwo_result).exists():
            gwo_s = int(pd.read_csv(args.gwo_result)["gwo_T"].iloc[0])
            # Actually read student index
            import re
            m = re.search(r"S(\d)", pd.read_csv(args.gwo_result)["model"].iloc[0])
            if m: gwo_s = int(m.group(1))
        log(f"  Student architecture: S{gwo_s}")

        kd_rows = []
        for method in ["vanilla", "fitnet", "at"]:
            log(f"\n[{method.upper()}] …")
            torch.manual_seed(cfg.RANDOM_SEED)
            student = build_student(gwo_s, n_features=n_features).to(device)
            val_auc, _, _ = train_kd_baseline(
                method, student, teacher, device, pos_weight)

            # Test evaluation
            student.eval()
            scores, labels = [], []
            with torch.no_grad():
                for X_b, y_b in test_loader:
                    z_s, _ = student(X_b.to(device))
                    scores.append(torch.sigmoid(z_s).cpu().numpy())
                    labels.append(y_b.cpu().numpy())
            y_true_t  = np.concatenate(labels)
            y_score_t = np.concatenate(scores)

            dual = dual_threshold_eval(y_true_t, y_score_t, cfg.EVAL_SENSITIVITY_TARGET)
            auc_pt, auc_lo, auc_hi = bootstrap_auroc_ci(y_true_t, y_score_t, n_resamples=500)
            lat = measure_latency_cpu(student.cpu(), n_features=n_features)

            log(f"  Test AUROC: {auc_pt:.4f} [{auc_lo:.4f}–{auc_hi:.4f}]")
            all_results.append((method.upper(), auc_pt, y_true_t, y_score_t))
            kd_rows.append({
                "method"         : method.upper(),
                "student"        : f"S{gwo_s}",
                "test_auroc"     : round(auc_pt, 4),
                "test_auroc_ci"  : f"[{auc_lo:.4f}–{auc_hi:.4f}]",
                "test_auprc"     : round(dual["auprc"], 4),
                "youden_sens"    : round(dual["youden_sensitivity"], 3),
                "youden_spec"    : round(dual["youden_specificity"], 3),
                "youden_f1"      : round(dual["youden_f1"], 3),
                "clin_sens"      : round(dual["clin_sensitivity"], 3),
                "latency_ms_cpu" : round(lat, 2),
            })

        # DeLong tests: GWO-KD vs each KD baseline
        log("\n[DeLong tests: GWO-KD vs classical KD baselines]")
        gwo_ckpt_path = cfg.CHECKPOINTS_DIR / f"student_gwo_kd_s{gwo_s}.pt"
        if gwo_ckpt_path.exists():
            gwo_ckpt   = torch.load(gwo_ckpt_path, map_location="cpu", weights_only=False)
            gwo_student = build_student(gwo_s, n_features=n_features)
            gwo_student.load_state_dict(gwo_ckpt["state_dict"])
            gwo_student.eval()
            gwo_scores, gwo_labels = [], []
            with torch.no_grad():
                for X_b, y_b in test_loader:
                    z_s, _ = gwo_student(X_b)
                    gwo_scores.append(torch.sigmoid(z_s).numpy())
                    gwo_labels.append(y_b.cpu().numpy())
            y_gwo_score = np.concatenate(gwo_scores)
            y_gwo_true  = np.concatenate(gwo_labels)

            delong_rows = []
            for method_name, test_auc, y_true_b, y_score_b in all_results:
                dt = delong_test(y_gwo_true, y_gwo_score, y_score_b)
                log(f"  GWO-KD vs {method_name}: z={dt['z_stat']:.3f} "
                    f"p={dt['p_value']:.4f} {'✓' if dt['significant'] else '✗'}")
                delong_rows.append({
                    "comparison"  : f"GWO-KD vs {method_name}",
                    "gwo_auroc"   : round(auroc(y_gwo_true, y_gwo_score), 4),
                    "other_auroc" : round(test_auc, 4),
                    "z_stat"      : round(dt["z_stat"], 3),
                    "p_value"     : round(dt["p_value"], 4),
                    "significant" : dt["significant"],
                })
            pd.DataFrame(delong_rows).to_csv(
                cfg.RESULTS_DIR / "delong_kd_baselines.csv", index=False)

        pd.DataFrame(kd_rows).to_csv(
            cfg.RESULTS_DIR / "kd_baselines.csv", index=False)
        log(f"\n  KD baselines → {cfg.RESULTS_DIR}/kd_baselines.csv")

    log("\n[Stage 07 COMPLETE] ✓")
    log("  → Run next: python src/08_sota_baselines.py")


if __name__ == "__main__":
    main()
