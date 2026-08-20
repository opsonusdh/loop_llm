"""Loop-count dynamics: dynamic training-time sampling (Ohio State
arXiv:2604.07822, Parcae arXiv:2604.12946), eval-mode gating, and the
depth-extrapolation guardrail in generate() (arXiv:2604.11791).
"""

from __future__ import annotations

import warnings

import torch

from loop_transformer import LoopConfig, LoopTransformer


def test_eval_mode_never_samples(tiny_model, batch):
    tiny_model.eval()
    counts = {len(tiny_model.compute_loss(batch)[1]) for _ in range(10)}
    assert counts == {tiny_model.cfg.max_loops}


def test_train_mode_samples_with_variety(tiny_model, batch):
    tiny_model.train()
    counts = {len(tiny_model.compute_loss(batch)[1]) for _ in range(20)}
    assert len(counts) > 1, "loop count never varied across 20 training calls"
    assert counts.issubset(set(range(tiny_model.cfg.min_loops, tiny_model.cfg.max_loops + 1)))


def test_loop_sampling_disabled_always_uses_max_loops(tiny_config, batch):
    tiny_config.loop_sampling = False
    torch.manual_seed(0)
    model = LoopTransformer(tiny_config)
    model.train()
    counts = {len(model.compute_loss(batch)[1]) for _ in range(10)}
    assert counts == {tiny_config.max_loops}


def test_explicit_max_loops_overrides_sampling(tiny_model, batch):
    tiny_model.train()
    _, step_losses = tiny_model.compute_loss(batch, max_loops=2)
    assert len(step_losses) == 2


def test_max_loops_one_edge_case(tiny_model, batch):
    loss, step_losses = tiny_model.compute_loss(batch, max_loops=1)
    assert len(step_losses) == 1
    assert torch.isfinite(loss)


def test_batched_generation_fixed_depth(tiny_model):
    prompt = torch.randint(0, tiny_model.cfg.vocab_size, (3, 5))
    out = tiny_model.generate(prompt, max_new_tokens=4, n_loops=2)
    assert out.shape == (3, 9)


def test_batched_generation_adaptive(tiny_model):
    prompt = torch.randint(0, tiny_model.cfg.vocab_size, (3, 5))
    out = tiny_model.generate(prompt, max_new_tokens=4)
    assert out.shape == (3, 9)


def test_no_warning_for_in_range_n_loops(tiny_model):
    prompt = torch.randint(0, tiny_model.cfg.vocab_size, (1, 4))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        tiny_model.generate(prompt, max_new_tokens=2, n_loops=tiny_model.cfg.max_loops)
    assert len(w) == 0


def test_no_warning_in_adaptive_mode(tiny_model):
    prompt = torch.randint(0, tiny_model.cfg.vocab_size, (1, 4))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        tiny_model.generate(prompt, max_new_tokens=2)
    assert len(w) == 0


def test_warning_fires_for_depth_extrapolation(tiny_model):
    prompt = torch.randint(0, tiny_model.cfg.vocab_size, (1, 4))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        tiny_model.generate(prompt, max_new_tokens=2, n_loops=tiny_model.cfg.max_loops + 5)
    assert len(w) == 1
    assert "structural drift" in str(w[0].message)
