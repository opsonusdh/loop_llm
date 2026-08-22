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


def get_block_sigmas(
    num_blocks: int,
    *,
    sigma_min: float = 0.002,
    sigma_max: float = 80.0,
    p_mean: float = -1.2,
    p_std: float = 1.2,
) -> torch.Tensor:
    """Return B+1 sigma boundaries matching SakanaAI DiffusionBlocks.

    The boundaries divide the truncated log-normal training distribution into
    equal probability mass intervals. This is adapted directly from the
    official implementation's get_block_sigmas().
    """
    if num_blocks < 1:
        raise ValueError("num_blocks must be >= 1")
    if not (0.0 < sigma_min < sigma_max):
        raise ValueError("require 0 < sigma_min < sigma_max")
    if p_std <= 0:
        raise ValueError("p_std must be > 0")
    normal = torch.distributions.Normal(0.0, 1.0)
    log_min = torch.log(torch.tensor(sigma_min, dtype=torch.float64))
    log_max = torch.log(torch.tensor(sigma_max, dtype=torch.float64))
    cdf_min = normal.cdf((log_min - p_mean) / p_std)
    cdf_max = normal.cdf((log_max - p_mean) / p_std)
    points = torch.linspace(0.0, 1.0, num_blocks + 1, dtype=torch.float64)
    cdf = cdf_min + (cdf_max - cdf_min) * points
    values = torch.exp(p_mean + p_std * normal.icdf(cdf))
    values[0] = sigma_min
    values[-1] = sigma_max
    return values.to(torch.float32)


def sample_block_sigma(
    num_blocks: int,
    *,
    sigma_min: float,
    sigma_max: float,
    p_mean: float,
    p_std: float,
    block_gamma: float = 0.0,
    device: torch.device,
) -> tuple[int, torch.Tensor, torch.Tensor]:
    """Sample one whole batch from a single block's sigma interval.

    Sakana's reference implementation samples one block per training step and
    then samples sigma inside that block's probability-mass interval. Using a
    single block for the entire batch keeps execution branch-free and ensures
    only one block receives gradients in the update.
    """
    boundaries = get_block_sigmas(
        num_blocks, sigma_min=sigma_min, sigma_max=sigma_max,
        p_mean=p_mean, p_std=p_std,
    ).to(device)
    block_idx = int(torch.randint(num_blocks, (), device=device).item())
    lo = float(boundaries[block_idx].item())
    hi = float(boundaries[block_idx + 1].item())
    if block_gamma > 0.0:
        log_lo = torch.log(torch.tensor(lo, device=device))
        log_hi = torch.log(torch.tensor(hi, device=device))
        span = log_hi - log_lo
        lo = max(sigma_min, float(torch.exp(log_lo - block_gamma * span).item()))
        hi = min(sigma_max, float(torch.exp(log_hi + block_gamma * span).item()))
    normal = torch.distributions.Normal(0.0, 1.0)
    cdf_lo = normal.cdf((torch.log(torch.tensor(lo, device=device)) - p_mean) / p_std)
    cdf_hi = normal.cdf((torch.log(torch.tensor(hi, device=device)) - p_mean) / p_std)
    u = torch.rand((), device=device) * (cdf_hi - cdf_lo) + cdf_lo
    sigma = torch.exp(torch.tensor(p_mean, device=device) + p_std * normal.icdf(u))
    return block_idx, sigma, boundaries
