"""Tiny LeWM-style encoder + AdaLN predictor for voxel_car."""

from __future__ import annotations

import math
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------


def _conv(c_in, c_out, stride=2):
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, stride=stride, padding=1, bias=False),
        nn.GroupNorm(8, c_out),
        nn.SiLU(inplace=True),
    )


class CNNEncoder(nn.Module):
    """(B, 3, H, W) -> (B, hidden_dim). Image size must be divisible by
    2**len(channels); 128x128 with 4 stages -> 8x8 feature map, pooled."""

    def __init__(self, hidden_dim: int = 192, channels=(32, 64, 128, 192)):
        super().__init__()
        layers = []
        c_prev = 3
        for c in channels:
            layers.append(_conv(c_prev, c, stride=2))
            c_prev = c
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(channels[-1], hidden_dim)
        self.hidden_dim = hidden_dim

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.proj(x)


# ---------------------------------------------------------------------------
# Projector (the BN matters -- see LeJEPA paper sec. 3.1)
# ---------------------------------------------------------------------------


class ProjMLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        # Accept (B, D) or (B, T, D); fold time into batch for BN.
        if x.dim() == 2:
            return self.net(x)
        shape = x.shape
        return self.net(x.reshape(-1, shape[-1])).reshape(*shape[:-1], -1)


# ---------------------------------------------------------------------------
# Action encoder
# ---------------------------------------------------------------------------


class ActionEncoder(nn.Module):
    def __init__(self, action_dim: int = 2, emb_dim: int = 192, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(action_dim, hidden),
            nn.SiLU(inplace=True),
            nn.Linear(hidden, emb_dim),
        )

    def forward(self, a):
        return self.net(a)


# ---------------------------------------------------------------------------
# AdaLN-conditioned causal Transformer predictor
# ---------------------------------------------------------------------------


def _modulate(x, shift, scale):
    return x * (1.0 + scale) + shift


class AdaLNBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, mlp_mult: int = 4, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(
            dim,
            n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mlp_mult, dim),
            nn.Dropout(dropout),
        )
        self.adaln = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        nn.init.zeros_(self.adaln[-1].weight)
        nn.init.zeros_(self.adaln[-1].bias)

    def forward(self, x, cond, causal_mask):
        s1, c1, g1, s2, c2, g2 = self.adaln(cond).chunk(6, dim=-1)

        h = _modulate(self.norm1(x), s1, c1)
        attn_out, _ = self.attn(
            h,
            h,
            h,
            attn_mask=causal_mask,
            need_weights=False,
        )
        x = x + g1 * attn_out

        h = _modulate(self.norm2(x), s2, c2)
        x = x + g2 * self.mlp(h)
        return x


class ARPredictor(nn.Module):
    def __init__(
        self,
        emb_dim: int = 192,
        depth: int = 3,
        n_heads: int = 4,
        mlp_mult: int = 4,
        max_len: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.pos_emb = nn.Parameter(torch.zeros(1, max_len, emb_dim))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)
        self.blocks = nn.ModuleList(
            [AdaLNBlock(emb_dim, n_heads, mlp_mult, dropout) for _ in range(depth)]
        )
        self.norm_out = nn.LayerNorm(emb_dim)

    def forward(self, z_seq, a_emb_seq):
        B, T, D = z_seq.shape
        x = z_seq + self.pos_emb[:, :T]
        mask = torch.triu(
            torch.ones(T, T, device=z_seq.device, dtype=torch.bool),
            diagonal=1,
        )
        for blk in self.blocks:
            x = blk(x, a_emb_seq, mask)
        return self.norm_out(x)


# ---------------------------------------------------------------------------
# Top-level wrapper
# ---------------------------------------------------------------------------


class VoxelCarJEPA(nn.Module):
    def __init__(
        self,
        image_hw=(128, 128),
        emb_dim: int = 192,
        enc_channels=(32, 64, 128, 192),
        pred_depth: int = 3,
        pred_heads: int = 4,
        action_dim: int = 2,
        max_len: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.image_hw = tuple(image_hw)
        self.emb_dim = int(emb_dim)
        self.action_dim = int(action_dim)

        self.encoder = CNNEncoder(hidden_dim=emb_dim, channels=enc_channels)
        self.enc_proj = ProjMLP(emb_dim, emb_dim)

        self.action_encoder = ActionEncoder(action_dim=action_dim, emb_dim=emb_dim)

        self.predictor = ARPredictor(
            emb_dim=emb_dim,
            depth=pred_depth,
            n_heads=pred_heads,
            max_len=max_len,
            dropout=dropout,
        )
        self.pred_proj = ProjMLP(emb_dim, emb_dim)

    def encode(self, imgs):
        """imgs: (B, T, 3, H, W) float or (B, 3, H, W) -> (B, T, D) or (B, D)."""
        if imgs.dim() == 4:
            return self.enc_proj(self.encoder(imgs))
        B, T = imgs.shape[:2]
        flat = imgs.reshape(B * T, *imgs.shape[2:])
        feat = self.encoder(flat)
        z = self.enc_proj(feat)
        return z.reshape(B, T, -1)

    def predict_next(self, z_seq, a_seq):
        """(B,T,D), (B,T,A) -> (B,T,D). Position t predicts state at t+1."""
        a_emb = self.action_encoder(a_seq)
        return self.pred_proj(self.predictor(z_seq, a_emb))

    @torch.no_grad()
    def rollout(self, z0, actions):
        """
        z0:      (B, D) initial latent
        actions: (B, H, A) action sequence
        Returns: (B, H, D) predicted latents z_1, ..., z_H.
        """
        B, H, A = actions.shape
        history = z0.unsqueeze(1)  # (B, 1, D)
        preds = []
        for t in range(H):
            T_hist = history.shape[1]
            a_so_far = actions[:, :T_hist]  # (B, T_hist, A)
            z_hat = self.predict_next(history, a_so_far)  # (B, T_hist, D)
            next_z = z_hat[:, -1:, :]
            preds.append(next_z)
            history = torch.cat([history, next_z], dim=1)
            # Truncate history to pos_emb capacity.
            if history.shape[1] > self.predictor.pos_emb.shape[1] - 1:
                history = history[:, -(self.predictor.pos_emb.shape[1] - 1) :]
        return torch.cat(preds, dim=1)
