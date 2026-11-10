"""
models/teacher.py — Teacher Model Architectures (Section 2.6)
==============================================================
Two teacher candidates trained under identical conditions:
  - Candidate A: BiLSTM + temporal self-attention (~1.28 M params)
  - Candidate B: TCN with dilated residual blocks (~0.41 M params)

Both produce:
  - logits z_t (pre-sigmoid, for BCEWithLogitsLoss)
  - penultimate features f_t (for feature-based KD distillation)
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import config as cfg


# =============================================================================
# Candidate A — BiLSTM + Temporal Self-Attention
# =============================================================================

class TemporalSelfAttention(nn.Module):
    """
    Scaled dot-product temporal self-attention over sequence dimension.
    Input: (B, T, H) → Output: (B, T, H)
    """
    def __init__(self, hidden_dim: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x)
        return self.norm(x + attn_out)


class BiLSTMTeacher(nn.Module):
    """
    BiLSTM + temporal self-attention teacher (Candidate A).
    Architecture: BiLSTM(256) → BiLSTM(128) → SelfAttention → FC(256→64→1)
    ~1.28 M parameters.

    Input:  (B, 12, F)   — batch × window × features
    Output: logits (B,), features (B, 256)
    """
    def __init__(
        self,
        n_features: int = cfg.N_FEATURES,
        hidden1: int = cfg.BILSTM_HIDDEN1,
        hidden2: int = cfg.BILSTM_HIDDEN2,
        fc_dims: list[int] = None,
        dropout: float = 0.2,
    ):
        super().__init__()
        if fc_dims is None:
            fc_dims = cfg.BILSTM_FC_DIMS  # [256, 64, 1]

        self.lstm1 = nn.LSTM(
            input_size=n_features, hidden_size=hidden1,
            num_layers=1, batch_first=True, bidirectional=True, dropout=0
        )
        # BiLSTM output: hidden1*2
        self.lstm2 = nn.LSTM(
            input_size=hidden1 * 2, hidden_size=hidden2,
            num_layers=1, batch_first=True, bidirectional=True, dropout=0
        )
        # BiLSTM output: hidden2*2
        self.attention = TemporalSelfAttention(hidden_dim=hidden2 * 2, n_heads=4)
        self.dropout   = nn.Dropout(dropout)

        # FC head: hidden2*2 → fc_dims[0] → fc_dims[1] → 1
        penultimate_dim = hidden2 * 2   # 256
        self.fc1 = nn.Linear(penultimate_dim, fc_dims[0])   # 256 → 256
        self.fc2 = nn.Linear(fc_dims[0],       fc_dims[1])  # 256 → 64
        self.out  = nn.Linear(fc_dims[1],       1)           # 64  → 1

        self.feature_dim = fc_dims[1]   # dim of penultimate layer (= f_t)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            logits  : (B,) pre-sigmoid
            features: (B, feature_dim) penultimate activations f_t
        """
        # x: (B, T, F)
        h1, _ = self.lstm1(x)                            # (B, T, H1*2)
        h1    = self.dropout(h1)
        h2, _ = self.lstm2(h1)                           # (B, T, H2*2)
        h2    = self.attention(h2)                        # (B, T, H2*2)

        # Global max pooling over time
        ctx   = h2.max(dim=1).values                     # (B, H2*2)
        ctx   = self.dropout(ctx)

        h     = F.relu(self.fc1(ctx))                    # (B, 256)
        h     = self.dropout(h)
        feat  = F.relu(self.fc2(h))                      # (B, 64) = f_t
        logit = self.out(feat).squeeze(-1)               # (B,)
        return logit, feat


# =============================================================================
# Candidate B — TCN with Dilated Residual Blocks
# =============================================================================

class CausalConv1d(nn.Module):
    """Causal convolution (no future leakage) via left-padding."""
    def __init__(self, in_ch: int, out_ch: int, kernel: int, dilation: int):
        super().__init__()
        self.pad  = (kernel - 1) * dilation    # causal pad
        self.conv = nn.Conv1d(
            in_ch, out_ch, kernel,
            dilation=dilation, padding=0
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (self.pad, 0))
        return self.conv(x)


class TCNResidualBlock(nn.Module):
    """
    TCN residual block: two causal dilated conv + residual connection.
    Input/Output: (B, channels, T)
    """
    def __init__(self, n_ch: int, kernel: int, dilation: int, dropout: float = 0.2):
        super().__init__()
        self.conv1  = CausalConv1d(n_ch, n_ch, kernel, dilation)
        self.norm1  = nn.BatchNorm1d(n_ch)
        self.conv2  = CausalConv1d(n_ch, n_ch, kernel, dilation)
        self.norm2  = nn.BatchNorm1d(n_ch)
        self.drop   = nn.Dropout(dropout)
        self.relu   = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        h = self.relu(self.norm1(self.conv1(x)))
        h = self.drop(h)
        h = self.relu(self.norm2(self.conv2(h)))
        h = self.drop(h)
        return self.relu(h + residual)


class TCNTeacher(nn.Module):
    """
    TCN teacher with dilated causal convolutions (Candidate B).
    Architecture:
      Input projection → 4 ResBlocks (dil=1,2,4,8) → GlobalAvgPool → Dense
    ~0.41 M parameters.

    Input:  (B, 12, F)
    Output: logits (B,), features (B, 64)
    """
    def __init__(
        self,
        n_features: int = cfg.N_FEATURES,
        n_filters: int = cfg.TCN_FILTERS,     # 64
        kernel: int = cfg.TCN_KERNEL_SIZE,    # 3
        dilations: list[int] = None,
        dropout: float = 0.2,
    ):
        super().__init__()
        if dilations is None:
            dilations = cfg.TCN_DILATIONS  # [1, 2, 4, 8]

        # Input projection: F → n_filters
        self.input_proj = nn.Conv1d(n_features, n_filters, kernel_size=1)

        # Residual blocks
        self.blocks = nn.ModuleList([
            TCNResidualBlock(n_filters, kernel, d, dropout)
            for d in dilations
        ])

        # Global average pooling → dense
        self.gap     = nn.AdaptiveAvgPool1d(1)
        self.fc      = nn.Linear(n_filters, 64)
        self.out     = nn.Linear(64, 1)
        self.dropout = nn.Dropout(dropout)

        self.feature_dim = 64   # dim of f_t

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        x: (B, T, F)  →  logits (B,), features (B, 64)
        """
        # (B, T, F) → (B, F, T) for Conv1d
        h = x.transpose(1, 2)
        h = self.input_proj(h)           # (B, n_filters, T)
        for block in self.blocks:
            h = block(h)                 # (B, n_filters, T)

        h   = self.gap(h).squeeze(-1)   # (B, n_filters) — global average pool
        h   = self.dropout(h)
        feat = F.relu(self.fc(h))        # (B, 64) = f_t
        feat = self.dropout(feat)
        logit = self.out(feat).squeeze(-1)  # (B,)
        return logit, feat


# =============================================================================
# Factory + param count
# =============================================================================

def build_teacher(name: str = "tcn", **kwargs) -> nn.Module:
    """
    Build a teacher by name: 'bilstm' or 'tcn'.
    Returns model on CPU (caller should .to(device)).
    """
    if name.lower() == "bilstm":
        return BiLSTMTeacher(**kwargs)
    elif name.lower() == "tcn":
        return TCNTeacher(**kwargs)
    else:
        raise ValueError(f"Unknown teacher: {name}. Choose 'bilstm' or 'tcn'.")


if __name__ == "__main__":
    # Quick sanity check
    B, T, n_feat = 4, 12, 21   # current feature count (21 present)
    x = torch.randn(B, T, n_feat)

    bilstm = BiLSTMTeacher(n_features=n_feat)
    logit_b, feat_b = bilstm(x)
    n_b = sum(p.numel() for p in bilstm.parameters())
    print(f"BiLSTM  | logits: {logit_b.shape} | feat: {feat_b.shape} | params: {n_b:,}")

    tcn = TCNTeacher(n_features=n_feat)
    logit_t, feat_t = tcn(x)
    n_t = sum(p.numel() for p in tcn.parameters())
    print(f"TCN     | logits: {logit_t.shape} | feat: {feat_t.shape} | params: {n_t:,}")

    print("teacher.py OK ✓")

