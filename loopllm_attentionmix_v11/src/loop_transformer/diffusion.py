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


def build_diffusion_causal_mask(
    clean_length: int,
    noisy_length: int | None = None,
    *,
    noisy_time_offset: int = 0,
    device: torch.device,
) -> torch.Tensor:
    """Build the clean-context/noisy-target causal mask.

    Layout is ``[clean_0..clean_{C-1}, noisy_0..noisy_{N-1}]``. A clean
    query ``i`` sees clean positions ``<= i``. A noisy query ``j`` sees clean
    positions ``< j + noisy_time_offset`` and its own noisy latent only. It
    never sees another noisy latent.

    Training uses ``noisy_time_offset=0`` for equal-length clean/noisy streams.
    Autoregressive next-token sampling can use ``noisy_time_offset=C`` for a
    single new target token, so that target can see the complete clean prompt.

    Every noisy row has an explicit self edge. This is required because noisy
    position 0 can legitimately have no clean predecessor; without its self
    edge, CSA/HCA would receive an all-invalid candidate row and softmax would
    produce NaNs.
    """
    if clean_length <= 0:
        raise ValueError("clean_length must be positive")
    if noisy_length is None:
        noisy_length = clean_length
    if noisy_length <= 0:
        raise ValueError("noisy_length must be positive")
    if noisy_time_offset < 0:
        raise ValueError("noisy_time_offset must be >= 0")

    total = clean_length + noisy_length
    mask = torch.zeros(total, total, device=device, dtype=torch.bool)

    clean = torch.arange(clean_length, device=device)
    mask[:clean_length, :clean_length] = clean[None, :] <= clean[:, None]

    noisy = torch.arange(noisy_length, device=device)
    visible_clean = clean[None, :] < (noisy[:, None] + noisy_time_offset)
    mask[clean_length:, :clean_length] = visible_clean
    mask[clean_length:, clean_length:] = torch.eye(
        noisy_length, device=device, dtype=torch.bool
    )

    if not bool(mask.any(dim=-1).all()):
        bad = (~mask.any(dim=-1)).nonzero(as_tuple=False).flatten().tolist()
        raise RuntimeError(f"Diffusion causal mask contains all-False rows: {bad}")
    return mask
