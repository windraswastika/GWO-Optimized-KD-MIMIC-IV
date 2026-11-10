"""
utils/data_utils.py — Data Loading Utilities
=============================================
Reusable helpers for loading processed/split data
as PyTorch Datasets/DataLoaders, with class-weighted sampler
for imbalanced sepsis data.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from pathlib import Path

import config as cfg


# =============================================================================
# Dataset
# =============================================================================

class SepsisWindowDataset(Dataset):
    """
    PyTorch Dataset for (N_windows, 12, 25) sliding-window sepsis data.

    Args:
        X: float32 array (N, W, F)
        y: int/float array (N,)
    """
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X).float()          # (N, 12, 25)
        self.y = torch.from_numpy(y.astype(np.float32))  # (N,)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_split_arrays(split: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load X and y arrays for a given split ('train', 'val', 'test').
    Returns (X_split, y_split) as numpy arrays.
    """
    X_all  = np.load(cfg.PROCESSED_DIR / "windows_features.npy", mmap_mode="r")
    y_all  = np.load(cfg.PROCESSED_DIR / "windows_labels.npy")
    idx    = np.load(cfg.SPLITS_DIR / f"{split}_idx.npy")
    return X_all[idx], y_all[idx]


def make_dataloader(
    split: str,
    batch_size: int = cfg.KD_BATCH_SIZE,
    shuffle: bool | None = None,
    weighted_sampler: bool = False,
    num_workers: int = 0,
) -> DataLoader:
    """
    Create a DataLoader for a given split.
    Args:
        split: 'train', 'val', or 'test'
        weighted_sampler: if True, use WeightedRandomSampler for class balance
                          (recommended for training to handle imbalance)
    """
    X, y = load_split_arrays(split)
    dataset = SepsisWindowDataset(X, y)

    if weighted_sampler and split == "train":
        # Class weights: inverse class frequency
        n_pos = y.sum()
        n_neg = len(y) - n_pos
        w_pos = len(y) / (2 * n_pos) if n_pos > 0 else 1.0
        w_neg = len(y) / (2 * n_neg) if n_neg > 0 else 1.0
        weights = np.where(y == 1, w_pos, w_neg)
        sampler = WeightedRandomSampler(
            weights=torch.from_numpy(weights).float(),
            num_samples=len(weights),
            replacement=True
        )
        return DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                          num_workers=num_workers, pin_memory=False)
    else:
        _shuffle = (split == "train") if shuffle is None else shuffle
        return DataLoader(dataset, batch_size=batch_size, shuffle=_shuffle,
                          num_workers=num_workers, pin_memory=False)


# =============================================================================
# Class weights for BCEWithLogitsLoss
# =============================================================================

def compute_class_weights(y: np.ndarray) -> torch.Tensor:
    """
    Compute inverse-class-frequency weight for the positive class.
    Used as `pos_weight` in BCEWithLogitsLoss.
    Returns scalar tensor: (n_neg / n_pos)
    """
    n_pos = y.sum()
    n_neg = len(y) - n_pos
    if n_pos == 0:
        return torch.tensor(1.0)
    return torch.tensor(n_neg / n_pos, dtype=torch.float32)


# =============================================================================
# Model efficiency measurement
# =============================================================================

def measure_latency_cpu(
    model: torch.nn.Module,
    n_features: int = cfg.N_FEATURES,
    window_size: int = cfg.WINDOW_SIZE_HOURS,
    n_warmup: int = 20,
    n_runs: int = 200,
) -> float:
    """
    Measure average CPU inference latency (ms/sample) for a model.
    Forces CPU evaluation regardless of training device (Section 2.16).
    Returns latency in milliseconds per single sample.
    """
    import time
    model_cpu = model.cpu().eval()
    x = torch.randn(1, window_size, n_features)  # single sample

    # Warmup
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model_cpu(x)

    # Timed runs
    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            t0 = time.perf_counter()
            _ = model_cpu(x)
            times.append((time.perf_counter() - t0) * 1000)  # ms

    return float(np.median(times))


def model_size_mb(model: torch.nn.Module) -> float:
    """Return model file size in MB (estimated from parameter bytes)."""
    total_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    return total_bytes / 1024 / 1024


def count_params(model: torch.nn.Module) -> int:
    """Return total trainable parameter count."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
