"""
models/students.py — Student Model Architectures (Section 2.7.1)
=================================================================
Three lightweight student models, all with edge constraints:
  - Model size < 1 MB
  - CPU latency < 10 ms/sample

S1 — TinyLSTM   : single LSTM (32 hidden) → 9,761 params, ~0.039 MB
S2 — 1D-CNN     : Conv1D(32→16→8, k=3) + GAP → 15,889 params, ~0.064 MB
S3 — CompactGRU : GRU(32)+GRU(16) → 29,921 params, ~0.120 MB

S4 and S5 are screened only (Section 2.7.2) to justify excluding them from
the GWO-KD candidate set — they are not used in distillation itself:
S4 — MobileNet1D    : depthwise-separable Conv1D(32→16→8)
S5 — PrunedLightGRU : GRU(48)+GRU(24), magnitude-pruned + fine-tuned

All models have the same interface as teacher:
  forward(x) → (logits, features)
  feature_dim: used by the KD regressor r(·)
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune

import config as cfg


# =============================================================================
# S1 — TinyLSTM
# =============================================================================

class TinyLSTM(nn.Module):
    """
    S1: Single-layer LSTM with 32 hidden units.
    Input  : (B, 12, F)
    Output : logits (B,), features (B, 32)
    Params : ~9,761 (varies with F)
    """
    def __init__(
        self,
        n_features: int = cfg.N_FEATURES,
        hidden: int = cfg.TINYLSTM_HIDDEN,    # 32
        dropout: float = cfg.STUDENT_DROPOUT,
    ):
        super().__init__()
        self.lstm    = nn.LSTM(n_features, hidden, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.head    = nn.Linear(hidden, 1)
        self.feature_dim = hidden   # f_s dim for KD regressor

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _, (h_n, _) = self.lstm(x)         # h_n: (1, B, hidden)
        feat  = self.dropout(h_n.squeeze(0))  # (B, hidden)
        logit = self.head(feat).squeeze(-1)   # (B,)
        return logit, feat


# =============================================================================
# S2 — 1D-CNN
# =============================================================================

class CNN1D(nn.Module):
    """
    S2: Three Conv1D layers (32→16→8 filters, kernel=3) + GAP + Linear.
    Input  : (B, 12, F)
    Output : logits (B,), features (B, 8)
    Params : ~15,889 (varies with F)
    """
    def __init__(
        self,
        n_features: int = cfg.N_FEATURES,
        filters: list[int] = None,
        kernel: int = cfg.CNN_KERNEL,        # 3
        dropout: float = cfg.STUDENT_DROPOUT,
    ):
        super().__init__()
        if filters is None:
            filters = cfg.CNN_FILTERS  # [32, 16, 8]

        layers = []
        in_ch = n_features
        for out_ch in filters:
            layers += [
                nn.Conv1d(in_ch, out_ch, kernel, padding=kernel // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            in_ch = out_ch
        self.convs   = nn.Sequential(*layers)
        self.gap     = nn.AdaptiveAvgPool1d(1)
        self.head    = nn.Linear(filters[-1], 1)
        self.feature_dim = filters[-1]   # 8

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # (B, T, F) → (B, F, T) for Conv1d
        h     = x.transpose(1, 2)                   # (B, F, T)
        h     = self.convs(h)                        # (B, 8, T)
        feat  = self.gap(h).squeeze(-1)              # (B, 8)
        logit = self.head(feat).squeeze(-1)          # (B,)
        return logit, feat


# =============================================================================
# S3 — CompactGRU
# =============================================================================

class CompactGRU(nn.Module):
    """
    S3: Two GRU layers (32, 16 hidden) + Linear.
    Input  : (B, 12, F)
    Output : logits (B,), features (B, 16)
    Params : ~29,921 (varies with F)
    """
    def __init__(
        self,
        n_features: int = cfg.N_FEATURES,
        hidden1: int = cfg.COMPACT_GRU_HIDDEN1,   # 32
        hidden2: int = cfg.COMPACT_GRU_HIDDEN2,   # 16
        dropout: float = cfg.STUDENT_DROPOUT,
    ):
        super().__init__()
        self.gru1    = nn.GRU(n_features, hidden1, batch_first=True)
        self.gru2    = nn.GRU(hidden1, hidden2, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.head    = nn.Linear(hidden2, 1)
        self.feature_dim = hidden2   # 16

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h1, _  = self.gru1(x)                      # (B, T, 32)
        h1     = self.dropout(h1)
        _, h2  = self.gru2(h1)                      # h2: (1, B, 16)
        feat   = self.dropout(h2.squeeze(0))        # (B, 16)
        logit  = self.head(feat).squeeze(-1)        # (B,)
        return logit, feat


# =============================================================================
# S4 — 1D-MobileNet (depthwise-separable convolutions)
# =============================================================================
# Screened only to justify EXCLUDING it from the candidate set (Section 2.7.2):
# not used in GWO-KD itself. Same filter progression as S2 for a fair
# comparison, but each stage is depthwise (per-channel) + pointwise (1x1),
# the standard MobileNet factorisation that reduces params vs. a plain conv.

class MobileNet1D(nn.Module):
    """
    S4: Depthwise-separable Conv1D blocks (32->16->8), GAP + Linear.
    Input  : (B, 12, F)
    Output : logits (B,), features (B, 8)
    """
    def __init__(
        self,
        n_features: int = cfg.N_FEATURES,
        filters: list[int] = None,
        kernel: int = cfg.MOBILENET_KERNEL,
        dropout: float = cfg.STUDENT_DROPOUT,
    ):
        super().__init__()
        if filters is None:
            filters = cfg.MOBILENET_FILTERS

        blocks = []
        in_ch = n_features
        for out_ch in filters:
            blocks.append(nn.Sequential(
                # Depthwise: one filter per input channel (groups=in_ch)
                nn.Conv1d(in_ch, in_ch, kernel, padding=kernel // 2, groups=in_ch),
                nn.ReLU(),
                # Pointwise: 1x1 conv mixes channels, sets the output width
                nn.Conv1d(in_ch, out_ch, 1),
                nn.ReLU(),
                nn.Dropout(dropout),
            ))
            in_ch = out_ch
        self.blocks  = nn.Sequential(*blocks)
        self.gap     = nn.AdaptiveAvgPool1d(1)
        self.head    = nn.Linear(filters[-1], 1)
        self.feature_dim = filters[-1]

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h     = x.transpose(1, 2)             # (B, F, T)
        h     = self.blocks(h)                 # (B, filters[-1], T)
        feat  = self.gap(h).squeeze(-1)        # (B, filters[-1])
        logit = self.head(feat).squeeze(-1)    # (B,)
        return logit, feat


# =============================================================================
# S5 — Pruned LightGRU
# =============================================================================
# Screened only to justify EXCLUDING it (Section 2.7.2). Trained dense at a
# larger width than S3 (48/24 vs 32/16), then globally magnitude-pruned
# (L1 unstructured, cfg.LIGHTGRU_PRUNE_SPARSITY fraction zeroed) and
# fine-tuned briefly to recover accuracy — the standard prune-then-fine-tune
# recipe. `prune_and_finetune()` performs the pruning step in-place; the
# reported "effective" parameter count assumes sparse storage of the zeroed
# weights (stated explicitly wherever this model's size is reported, since a
# naively saved dense .pt file does not itself shrink from pruning).

class PrunedLightGRU(nn.Module):
    """
    S5: Two GRU layers (48, 24 hidden) + Linear, magnitude-pruned after dense training.
    Input  : (B, 12, F)
    Output : logits (B,), features (B, 24)
    """
    def __init__(
        self,
        n_features: int = cfg.N_FEATURES,
        hidden1: int = cfg.LIGHTGRU_DENSE_HIDDEN1,   # 48
        hidden2: int = cfg.LIGHTGRU_DENSE_HIDDEN2,   # 24
        dropout: float = cfg.STUDENT_DROPOUT,
    ):
        super().__init__()
        self.gru1    = nn.GRU(n_features, hidden1, batch_first=True)
        self.gru2    = nn.GRU(hidden1, hidden2, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.head    = nn.Linear(hidden2, 1)
        self.feature_dim = hidden2
        self._pruned = False

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h1, _  = self.gru1(x)
        h1     = self.dropout(h1)
        _, h2  = self.gru2(h1)
        feat   = self.dropout(h2.squeeze(0))
        logit  = self.head(feat).squeeze(-1)
        return logit, feat

    def prune(self, sparsity: float = cfg.LIGHTGRU_PRUNE_SPARSITY):
        """
        Apply global L1-unstructured magnitude pruning. The mask stays ACTIVE
        (reparametrized `weight_orig` + `weight_mask`) rather than being baked
        in immediately: PyTorch recomputes `weight = weight_orig * weight_mask`
        on every forward call, so pruned positions stay exactly zero at
        inference throughout subsequent fine-tuning. Call `finalize_prune()`
        only after fine-tuning is done — baking the mask in before fine-tuning
        would let the optimizer drift the "pruned" weights away from zero on
        every step, silently undoing the pruning by the end of fine-tuning.
        """
        self._prune_targets = []
        for module in (self.gru1, self.gru2, self.head):
            for name, param in module.named_parameters():
                if "weight" in name and param.dim() >= 2:
                    self._prune_targets.append((module, name))
        prune.global_unstructured(
            self._prune_targets, pruning_method=prune.L1Unstructured, amount=sparsity
        )
        self._pruned = True

    def finalize_prune(self):
        """Bake the pruning mask into plain weight tensors. Call this AFTER
        fine-tuning, once no more optimizer steps will touch these weights —
        only then does `effective_params()` reflect the true post-prune model."""
        for module, name in self._prune_targets:
            prune.remove(module, name)

    def effective_params(self) -> int:
        """Non-zero parameter count (what a sparse-storage deployment would
        need). Only meaningful after `finalize_prune()` — before that, the
        reparametrized `weight_orig` is what `self.parameters()` sees, which
        is the pre-mask (non-zeroed) tensor."""
        return int(sum((p != 0).sum().item() for p in self.parameters()))


# =============================================================================
# Factory
# =============================================================================

STUDENT_REGISTRY = {
    1: TinyLSTM,
    2: CNN1D,
    3: CompactGRU,
    4: MobileNet1D,
    5: PrunedLightGRU,
    "tinylstm"  : TinyLSTM,
    "cnn1d"     : CNN1D,
    "compactgru": CompactGRU,
    "mobilenet1d"    : MobileNet1D,
    "prunedlightgru" : PrunedLightGRU,
}


def build_student(index: int | str, n_features: int = cfg.N_FEATURES, **kwargs) -> nn.Module:
    """
    Build a student by index (1/2/3) or name ('tinylstm'/'cnn1d'/'compactgru').
    Returns model on CPU.
    """
    key = int(round(index)) if isinstance(index, float) else index
    if key not in STUDENT_REGISTRY:
        raise ValueError(f"Unknown student index/name: {index}")
    return STUDENT_REGISTRY[key](n_features=n_features, **kwargs)


if __name__ == "__main__":
    # Quick sanity check + param count
    B, T, F = 4, 12, 21
    x = torch.randn(B, T, F)

    for idx, name in [(1, "TinyLSTM"), (2, "1D-CNN"), (3, "CompactGRU"),
                      (4, "MobileNet1D"), (5, "PrunedLightGRU")]:
        model = build_student(idx, n_features=F)
        logit, feat = model(x)
        n_params = sum(p.numel() for p in model.parameters())
        size_mb  = n_params * 4 / 1024 / 1024
        print(f"S{idx} {name:14s} | logits: {logit.shape} | "
              f"feat: {feat.shape} | params: {n_params:,} | size: {size_mb:.3f} MB")

    # S5 prune sanity check
    s5 = build_student(5, n_features=F)
    before = sum(p.numel() for p in s5.parameters())
    s5.prune(0.6)
    after_effective = s5.effective_params()
    print(f"S5 prune check | dense params: {before:,} | "
          f"effective (non-zero) after 60% prune: {after_effective:,}")

    print("students.py OK ✓")
