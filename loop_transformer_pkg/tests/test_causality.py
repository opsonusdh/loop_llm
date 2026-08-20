"""Causality verification for CSA/HCA attention.

An earlier version of these attention mechanisms had no causal masking
at all -- any position could see compressed summaries of future tokens.
This is the definitive test: perturb tokens after a cutoff and confirm
logits before the cutoff are bit-for-bit unaffected, across every loop.
"""

from __future__ import annotations

import torch

from loop_transformer import LoopConfig, LoopTransformer


def test_zero_future_leakage_across_all_loops():
    torch.manual_seed(0)
    cfg = LoopConfig(
        vocab_size=500, dim=256, n_layers=4, n_heads=4, head_dim=64,
        ffn_hidden_dim=512, max_loops=3, groups=4, group_dim=128,
        csa_top_k=8, sw_window=6, csa_m=4, hca_m_prime=8, loop_sampling=False,
    )
    model = LoopTransformer(cfg)
    model.eval()

    torch.manual_seed(1)
    T, cutoff = 40, 15
    x1 = torch.randint(0, cfg.vocab_size, (1, T))
    x2 = x1.clone()
    x2[:, cutoff:] = torch.randint(0, cfg.vocab_size, (1, T - cutoff))

    with torch.no_grad():
        logits1, _ = model(x1)
        logits2, _ = model(x2)

    for l1, l2 in zip(logits1, logits2):
        max_diff = (l1[:, :cutoff] - l2[:, :cutoff]).abs().max().item()
        assert max_diff == 0.0, "future token perturbation leaked into earlier logits"

    # Sanity: the test isn't vacuous -- late positions must actually differ.
    late_diff = (logits1[-1][:, cutoff:] - logits2[-1][:, cutoff:]).abs().max().item()
    assert late_diff > 0.0, "test is vacuous -- late positions never differed"
