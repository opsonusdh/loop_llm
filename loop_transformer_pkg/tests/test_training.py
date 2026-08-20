"""End-to-end training convergence smoke test. Not a substitute for a
real training run, but catches any change that breaks learnability
outright (dead gradients, exploding loss, shape errors under backward).
"""

from __future__ import annotations

import torch

from loop_transformer import LoopConfig, LoopTransformer


def test_loss_decreases_on_a_fixed_overfit_batch():
    torch.manual_seed(0)
    cfg = LoopConfig(
        vocab_size=200, dim=256, n_layers=3, n_heads=4, head_dim=64,
        ffn_hidden_dim=512, max_loops=4, beta_entropy=0.05,
        groups=4, group_dim=128, sw_window=32,
    )
    model = LoopTransformer(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)

    torch.manual_seed(1)
    data = torch.randint(0, cfg.vocab_size, (4, 48))

    losses = []
    for _ in range(60):
        opt.zero_grad()
        loss, _ = model.compute_loss(data)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())

    assert all(torch.isfinite(torch.tensor(losses))), "loss became non-finite during training"
    assert losses[-1] < losses[0] * 0.5, f"loss did not drop enough: {losses[0]:.4f} -> {losses[-1]:.4f}"


def test_deeper_loops_beat_loop_one_after_training():
    """Loosely checks the 'deeper is better' property Ouro trains for:
    after a little training, later loop steps shouldn't be WORSE than
    the first (they needn't be monotonically best-to-worst every step,
    but the final step should clearly beat the first)."""
    torch.manual_seed(0)
    cfg = LoopConfig(
        vocab_size=200, dim=256, n_layers=3, n_heads=4, head_dim=64,
        ffn_hidden_dim=512, max_loops=4, groups=4, group_dim=128,
        sw_window=32, loop_sampling=False,
    )
    model = LoopTransformer(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)

    torch.manual_seed(1)
    data = torch.randint(0, cfg.vocab_size, (4, 48))

    for _ in range(60):
        opt.zero_grad()
        loss, _ = model.compute_loss(data, max_loops=4)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    _, step_losses = model.compute_loss(data, max_loops=4)
    assert step_losses[-1] < step_losses[0]
