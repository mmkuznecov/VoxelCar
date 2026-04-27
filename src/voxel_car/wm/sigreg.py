"""Sketched Isotropic Gaussian Regularizer (LeJEPA, 2025)."""

from __future__ import annotations
import torch
from torch import nn


class SIGReg(nn.Module):
    """
    Projects embeddings onto random unit directions and penalizes deviation of
    each univariate projection from N(0, 1) via a discretized Epps-Pulley
    characteristic-function test. By Cramer-Wold, matching all 1D marginals to
    the standard normal is equivalent to matching the joint to N(0, I).
    """

    def __init__(self, knots: int = 17, num_proj: int = 512, t_max: float = 3.0):
        super().__init__()
        self.num_proj = int(num_proj)

        t = torch.linspace(0.0, float(t_max), int(knots), dtype=torch.float32)
        dt = float(t_max) / (int(knots) - 1)
        # Trapezoidal weights.
        w = torch.full((int(knots),), 2.0 * dt, dtype=torch.float32)
        w[0] = w[-1] = dt
        # Gaussian window w(t) = exp(-t^2/2) baked in so the integrand is
        # bounded and concentrated around small t where most discriminative
        # power lives.
        window = torch.exp(-0.5 * t * t)
        self.register_buffer("t", t)
        self.register_buffer("phi_target", window)  # E[e^{itX}] for X~N(0,1)
        self.register_buffer("weights", w * window)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (N, D). Returns a scalar regularization loss."""
        if z.dim() != 2:
            raise ValueError(f"SIGReg expects (N, D), got {tuple(z.shape)}")
        N, D = z.shape

        A = torch.randn(D, self.num_proj, device=z.device, dtype=z.dtype)
        A = A / A.norm(dim=0, keepdim=True).clamp_min(1e-8)

        h = z @ A  # (N, M)
        ht = h.unsqueeze(-1) * self.t  # (N, M, K)

        cos_err = ht.cos().mean(dim=0) - self.phi_target  # (M, K)
        sin_err = ht.sin().mean(dim=0)  # (M, K)
        err = cos_err * cos_err + sin_err * sin_err  # (M, K)

        stat = (err * self.weights).sum(dim=-1) * float(N)  # (M,)
        return stat.mean()
