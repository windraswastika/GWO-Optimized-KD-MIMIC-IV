"""
utils/gwo.py — Grey Wolf Optimizer (Algorithm 1, Section 2.8)
==============================================================
Implements GWO to search the KD hyperparameter space:
  θ = (T, α_kd, β_kd, s) ∈ ℝ⁴

  T        : distillation temperature ∈ [1, 15]
  alpha_kd : BCE loss weight         ∈ [0.10, 0.90]
  beta_kd  : feature-align weight   ∈ [0.01, 0.50]
  s        : student architecture   ∈ [1.0, 3.0] → round → {1,2,3}

Hierarchy:
  α-wolf (best), β-wolf (2nd), δ-wolf (3rd) → guide ω-wolves

Budget: N_w wolves × I_max iterations × ≥5 independent seeds (Section 2.16)
"""

import os
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable


# =============================================================================
# Search space
# =============================================================================

PARAM_NAMES  = ["T", "alpha_kd", "beta_kd", "s"]
PARAM_BOUNDS = {          # (lower, upper)
    "T"       : (1.0,  15.0),
    "alpha_kd": (0.10, 0.90),
    "beta_kd" : (0.01, 0.50),
    "s"       : (1.0,  3.0),
}

LOWER = np.array([PARAM_BOUNDS[k][0] for k in PARAM_NAMES])
UPPER = np.array([PARAM_BOUNDS[k][1] for k in PARAM_NAMES])


def decode_position(pos: np.ndarray) -> dict:
    """
    Map raw position vector → named hyperparameters.
    's' is rounded to nearest integer {1, 2, 3}.
    """
    pos = np.clip(pos, LOWER, UPPER)
    return {
        "T"        : float(pos[0]),
        "alpha_kd" : float(pos[1]),
        "beta_kd"  : float(pos[2]),
        "s"        : int(round(float(pos[3]))),
    }


# =============================================================================
# GWO state
# =============================================================================

@dataclass
class GWOState:
    n_wolves : int
    n_dim    : int = len(PARAM_NAMES)
    positions: np.ndarray = field(default=None)

    # Alpha, Beta, Delta wolves (best 3)
    alpha_pos: np.ndarray = field(default=None)
    beta_pos : np.ndarray = field(default=None)
    delta_pos: np.ndarray = field(default=None)
    alpha_score: float = -np.inf
    beta_score : float = -np.inf
    delta_score: float = -np.inf

    history: list = field(default_factory=list)

    def initialise(self, rng: np.random.Generator):
        """Randomly initialise wolves within bounds."""
        self.positions = rng.uniform(
            low=LOWER, high=UPPER,
            size=(self.n_wolves, self.n_dim)
        )
        self.alpha_pos   = self.positions[0].copy()
        self.beta_pos    = self.positions[1].copy()
        self.delta_pos   = self.positions[2].copy()
        self.alpha_score = -np.inf
        self.beta_score  = -np.inf
        self.delta_score = -np.inf

    def update_hierarchy(self, scores: np.ndarray):
        """Update α, β, δ from current wolf scores."""
        order = np.argsort(scores)[::-1]   # descending fitness
        for rank, idx in enumerate(order[:3]):
            if rank == 0 and scores[idx] > self.alpha_score:
                self.alpha_score = scores[idx]
                self.alpha_pos   = self.positions[idx].copy()
            elif rank == 1 and scores[idx] > self.beta_score:
                self.beta_score  = scores[idx]
                self.beta_pos    = self.positions[idx].copy()
            elif rank == 2 and scores[idx] > self.delta_score:
                self.delta_score = scores[idx]
                self.delta_pos   = self.positions[idx].copy()

    def step(self, iteration: int, i_max: int, rng: np.random.Generator):
        """
        Update all wolf positions using GWO encircling + hunting equations.
        a decays linearly from 2 → 0 over iterations.
        """
        a = 2.0 * (1.0 - iteration / i_max)   # linearly decreasing

        new_positions = np.empty_like(self.positions)
        for i in range(self.n_wolves):
            x = self.positions[i]
            X1 = _update_towards(x, self.alpha_pos, a, rng)
            X2 = _update_towards(x, self.beta_pos,  a, rng)
            X3 = _update_towards(x, self.delta_pos, a, rng)
            new_pos = (X1 + X2 + X3) / 3.0
            new_positions[i] = np.clip(new_pos, LOWER, UPPER)

        self.positions = new_positions


def _update_towards(
    x     : np.ndarray,
    leader: np.ndarray,
    a     : float,
    rng   : np.random.Generator,
) -> np.ndarray:
    """Single GWO position update towards one leader wolf."""
    r1 = rng.random(size=x.shape)
    r2 = rng.random(size=x.shape)
    A  = 2 * a * r1 - a          # A ∈ [-a, a]
    C  = 2 * r2                   # C ∈ [0, 2]
    D  = np.abs(C * leader - x)   # distance
    return leader - A * D


# =============================================================================
# Main GWO loop
# =============================================================================

def run_gwo(
    fitness_fn: Callable[[dict], float],
    n_wolves : int = 10,
    i_max    : int = 10,
    seed     : int = 0,
    epsilon  : float = 1e-4,
    patience : int  = 5,
    verbose  : bool = True,
    n_workers: int | None = None,
) -> tuple[dict, float, list]:
    """
    Run Grey Wolf Optimizer.

    Args:
        fitness_fn : callable(params: dict) → float (higher = better, e.g. val AUROC)
        n_wolves   : swarm size (N_w)
        i_max      : max iterations
        seed       : random seed
        epsilon    : convergence threshold for α-score improvement
        patience   : iterations without improvement before early stop
        verbose    : print progress
        n_workers  : evaluate the N_w wolves of one iteration concurrently
                     (each wolf's fitness_fn call is independent). Defaults
                     to min(n_wolves, cpu_count()); pass 1 to force serial.

    Returns:
        best_params : dict of decoded hyperparameters
        best_score  : α fitness value
        history     : list of dicts per iteration
    """
    if n_workers is None:
        n_workers = max(1, min(n_wolves, os.cpu_count() or 1))

    rng   = np.random.default_rng(seed)
    state = GWOState(n_wolves=n_wolves)
    state.initialise(rng)

    no_improve = 0
    prev_alpha = -np.inf
    history    = []

    for it in range(1, i_max + 1):
        # Evaluate all wolves (independent per wolf → safe to run concurrently)
        if n_workers > 1:
            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                scores = np.array(list(pool.map(
                    lambda i: fitness_fn(decode_position(state.positions[i])),
                    range(n_wolves)
                )))
        else:
            scores = np.array([
                fitness_fn(decode_position(state.positions[i]))
                for i in range(n_wolves)
            ])

        state.update_hierarchy(scores)
        state.history.append({
            "iteration"  : it,
            "alpha_score": state.alpha_score,
            "alpha_params": decode_position(state.alpha_pos),
            "mean_score" : float(scores.mean()),
            "min_score"  : float(scores.min()),
        })
        history.append(state.history[-1])

        if verbose:
            p = decode_position(state.alpha_pos)
            print(
                f"  [GWO it={it:2d}] α={state.alpha_score:.4f} | "
                f"T={p['T']:.1f} α_kd={p['alpha_kd']:.2f} "
                f"β_kd={p['beta_kd']:.3f} s={p['s']}",
                flush=True
            )

        # Early stopping
        if state.alpha_score - prev_alpha < epsilon:
            no_improve += 1
        else:
            no_improve  = 0
        prev_alpha = state.alpha_score

        if no_improve >= patience:
            if verbose:
                print(f"  [GWO] Early stop at iteration {it} "
                      f"(no improvement for {patience} iters)", flush=True)
            break

        # Update positions (skip on last iteration)
        if it < i_max:
            state.step(it, i_max, rng)

    best_params = decode_position(state.alpha_pos)
    return best_params, state.alpha_score, history


# =============================================================================
# Multi-seed runner (Section 2.16: ≥5 seeds)
# =============================================================================

def run_gwo_multiseed(
    fitness_fn : Callable[[dict], float],
    seeds      : list[int],
    n_wolves   : int = 10,
    i_max      : int = 10,
    epsilon    : float = 1e-4,
    patience   : int   = 5,
    verbose    : bool  = True,
    n_workers  : int | None = None,
) -> tuple[dict, float, list]:
    """
    Run GWO with multiple seeds and return the best result.
    Required by Section 2.16 for statistical robustness (≥5 seeds).

    Returns:
        best_params : hyperparams from the best seed
        best_score  : best α-score across seeds
        all_results : list of (seed, params, score, history) per seed
    """
    all_results = []
    best_score  = -np.inf
    best_params = None

    for seed in seeds:
        if verbose:
            print(f"\n[GWO seed={seed}]", flush=True)
        params, score, hist = run_gwo(
            fitness_fn, n_wolves=n_wolves, i_max=i_max,
            seed=seed, epsilon=epsilon, patience=patience, verbose=verbose,
            n_workers=n_workers,
        )
        all_results.append((seed, params, score, hist))
        if score > best_score:
            best_score  = score
            best_params = params

    if verbose:
        print(f"\n[GWO Multi-seed summary]")
        for seed, params, score, _ in all_results:
            print(f"  seed={seed}: score={score:.4f}  params={params}")
        print(f"  Best: score={best_score:.4f}  params={best_params}")

    return best_params, best_score, all_results


# =============================================================================
# Quick test
# =============================================================================

if __name__ == "__main__":
    import time

    # Dummy fitness: negative distance from target params
    target = {"T": 5.0, "alpha_kd": 0.5, "beta_kd": 0.1, "s": 2}

    def dummy_fitness(params: dict) -> float:
        diff = sum((params[k] - target[k])**2 for k in target)
        return -diff

    print("Testing GWO with dummy fitness …")
    t0 = time.time()
    params, score, history = run_gwo(
        dummy_fitness, n_wolves=10, i_max=20, seed=42, verbose=True
    )
    print(f"\nBest: {params}  score={score:.6f}  ({time.time()-t0:.1f}s)")
    print(f"Target: {target}")
    assert abs(params["T"] - target["T"]) < 1.0, "GWO did not converge on T"
    print("gwo.py OK ✓")
