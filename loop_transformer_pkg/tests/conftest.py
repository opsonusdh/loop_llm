"""Shared pytest fixtures -- small, fast configs so the full suite runs
in seconds on CPU, not minutes.
"""

from __future__ import annotations

import torch
import pytest

from loop_transformer import LoopConfig, LoopTransformer


@pytest.fixture
def tiny_config() -> LoopConfig:
    """Small enough to run instantly on CPU; big enough to exercise
    every attention type (SWA, CSA, HCA all get triggered with n_layers=4)."""
    return LoopConfig(
        vocab_size=200, dim=128, n_layers=4, n_heads=4, head_dim=64,
        ffn_hidden_dim=256, max_loops=3, min_loops=1,
        groups=4, group_dim=64, sw_window=8, csa_top_k=8, csa_m=4, hca_m_prime=8,
    )


@pytest.fixture
def tiny_model(tiny_config) -> LoopTransformer:
    torch.manual_seed(0)
    return LoopTransformer(tiny_config)


@pytest.fixture
def batch(tiny_config):
    torch.manual_seed(1)
    return torch.randint(0, tiny_config.vocab_size, (2, 32))
