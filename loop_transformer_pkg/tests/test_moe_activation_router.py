from __future__ import annotations

import torch

from loop_transformer import LoopConfig, LoopTransformer


def _cfg() -> LoopConfig:
    return LoopConfig(
        vocab_size=300, dim=128, n_layers=2, n_heads=4, head_dim=32,
        ffn_hidden_dim=256, rope_dim=16, groups=4, group_dim=64,
        max_loops=4, loop_sampling=False,
    )


def test_moe_and_activation_router_forward_backward():
    torch.manual_seed(0)
    model = LoopTransformer(_cfg())
    x = torch.randint(0, 300, (2, 16))
    loss, step_losses = model.compute_loss(x, max_loops=4)
    assert torch.isfinite(loss)
    assert step_losses.numel() == 4
    assert model.last_activation_balance_loss is not None
    assert model.last_moe_aux_loss is not None
    assert model.last_activation_probs is not None
    assert model.last_moe_load is not None
    assert model.last_activation_probs.shape == (4,)
    assert model.last_moe_load.shape == (4,)
    assert abs(model.last_activation_probs.sum().item() - 1.0) < 1e-5
    assert abs(model.last_moe_load.sum().item() - 1.0) < 1e-5
    loss.backward()


def test_activation_balance_zero_for_uniform_batch_average():
    from loop_transformer.feedforward import FeedForward
    probs = torch.full((100, 4), 0.25)
    loss = FeedForward._activation_balance_loss(probs)
    assert loss.item() < 1e-7


def test_activation_top_k_controls_sparse_activation_routing():
    from loop_transformer.feedforward import FeedForward
    torch.manual_seed(0)
    x = torch.randn(2, 3, 32)
    ff = FeedForward(32, 64, max_loops=4, activation_top_k=2)
    probs = ff._route_activations(x, loop_idx=0)
    active = (probs > 0).sum(dim=-1)
    assert torch.all(active == 2)
    assert torch.allclose(probs.sum(dim=-1), torch.ones_like(active, dtype=probs.dtype))
