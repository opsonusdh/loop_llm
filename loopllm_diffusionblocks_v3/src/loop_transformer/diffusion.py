"""DiffusionBlocks utilities for continuous-time recurrent-depth training.

This implements the recurrent-depth adaptation from DiffusionBlocks (ICLR 2026):
train the entire looped network as a denoiser with one recurrent pass per
training step, sample log-normal noise levels, use EDM preconditioning and
weighting, and keep the ordinary recurrent K-loop procedure for inference.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Tuple

import torch


@dataclass(frozen=True)
class DiffusionSample:
    sigma: torch.Tensor
    c_in: torch.Tensor
    c_skip: torch.Tensor
    c_out: torch.Tensor
    c_noise: torch.Tensor
    weight: torch.Tensor


def sample_log_normal_sigma(
    batch_size: int,
    *,
    device: torch.device,
    p_mean: float = -1.2,
    p_std: float = 1.2,
    sigma_min: float = 0.002,
    sigma_max: float = 80.0,
) -> torch.Tensor:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if sigma_min <= 0 or sigma_max <= sigma_min:
        raise ValueError("require 0 < sigma_min < sigma_max")
    log_sigma = torch.randn(batch_size, device=device) * p_std + p_mean
    return log_sigma.exp().clamp_(min=sigma_min, max=sigma_max)


def edm_preconditioning(
    sigma: torch.Tensor,
    *,
    sigma_data: float = 0.5,
) -> DiffusionSample:
    if sigma_data <= 0:
        raise ValueError("sigma_data must be positive")
    sigma2 = sigma.square()
    sd2 = float(sigma_data) ** 2
    denom = sigma2 + sd2
    c_in = denom.rsqrt()
    c_skip = sd2 / denom
    c_out = sigma * float(sigma_data) / denom.sqrt()
    c_noise = torch.log(sigma) / 4.0
    weight = denom / ((sigma * float(sigma_data)).square())
    return DiffusionSample(
        sigma=sigma,
        c_in=c_in,
        c_skip=c_skip,
        c_out=c_out,
        c_noise=c_noise,
        weight=weight,
    )


def log_sigma_embedding(c_noise: torch.Tensor, dim: int) -> torch.Tensor:
    """Deterministic Fourier embedding of c_noise for conditioning blocks."""
    if dim <= 0:
        raise ValueError("dim must be positive")
    half = dim // 2
    if half == 0:
        return c_noise[:, None]
    # Fixed log-spaced frequencies keep the small conditioning module stable
    # across model sizes and require no trainable lookup table.
    freq = torch.exp(
        torch.linspace(
            math.log(1.0),
            math.log(1000.0),
            half,
            device=c_noise.device,
            dtype=c_noise.dtype,
        )
    )
    angles = c_noise[:, None] * freq[None, :]
    emb = torch.cat([angles.sin(), angles.cos()], dim=-1)
    if emb.size(1) < dim:
        emb = torch.cat([emb, c_noise[:, None]], dim=-1)
    return emb[:, :dim]
