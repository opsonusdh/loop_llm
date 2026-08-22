import random
from pathlib import Path

import numpy as np
import torch

from loop_transformer import LoopConfig, LoopTransformer, load_checkpoint, save_checkpoint
from loop_transformer.attention_mixture import AttentionExpertMixture


def tiny_cfg(**kwargs):
    base = dict(
        vocab_size=32, dim=32, n_layers=2, n_heads=2, head_dim=16,
        ffn_hidden_dim=64, rope_dim=8, max_loops=3, min_loops=3,
        loop_sampling=False, csa_m=2, csa_top_k=3, hca_m_prime=3,
        sw_window=6, groups=2, group_dim=16,
        moe_num_shared_experts=1, moe_num_routed_experts=2, moe_top_k=1,
        activation_top_k=2, attention_mixture=True,
        attention_mixture_start_layer=1, grad_checkpointing=False,
    )
    base.update(kwargs)
    return LoopConfig(**base)


def test_adaptive_exit_actually_stops_deep_loops(monkeypatch):
    model = LoopTransformer(tiny_cfg()).eval()
    calls = {"n": 0}
    original = model._one_loop

    def counted(*args, **kwargs):
        calls["n"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(model, "_one_loop", counted)

    def fake_gate(h, loop_idx=0):
        return torch.full((h.size(0),), 0.99, device=h.device, dtype=h.dtype)

    monkeypatch.setattr(model.exit_gate, "forward", fake_gate)
    x = torch.randint(0, 32, (2, 8))
    logits, depths, _ = model.forward_adaptive_exit(x, exit_threshold=0.8)
    assert logits.shape == (2, 8, 32)
    assert torch.all(depths == 1)
    assert calls["n"] == 1, "adaptive exit must not run unused deeper loops"


def test_adaptive_exit_keeps_mixed_batch_correct(monkeypatch):
    model = LoopTransformer(tiny_cfg()).eval()

    def fake_gate(h, loop_idx=0):
        if loop_idx == 0:
            # First example exits immediately; second continues.
            vals = torch.tensor([0.95, 0.05], device=h.device, dtype=h.dtype)
            return vals[:h.size(0)]
        return torch.full((h.size(0),), 0.99, device=h.device, dtype=h.dtype)

    monkeypatch.setattr(model.exit_gate, "forward", fake_gate)
    x = torch.randint(0, 32, (2, 8))
    logits, depths, _ = model.forward_adaptive_exit(x, exit_threshold=0.8)
    assert logits.shape == (2, 8, 32)
    assert depths.tolist() == [1, 2]


def test_checkpoint_resume_restores_optimizer_and_rng(tmp_path: Path):
    cfg = tiny_cfg(attention_mixture=False)
    model = LoopTransformer(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x = torch.randint(0, 32, (2, 8))
    loss, _ = model.compute_loss(x)
    loss.backward()
    opt.step()

    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)

    ckpt = tmp_path / "resume.pt"
    save_checkpoint(ckpt, model, opt, step=7)
    expected = (random.random(), float(np.random.rand()), float(torch.rand(())))

    # Move RNG elsewhere before loading, then ensure the post-checkpoint
    # sequence resumes at exactly the same values.
    random.seed(999)
    np.random.seed(999)
    torch.manual_seed(999)
    loaded = load_checkpoint(ckpt, restore_rng=True)
    restored = (random.random(), float(np.random.rand()), float(torch.rand(())))

    assert restored == expected
    assert loaded["step"] == 7
    assert loaded["optimizer_state_dict"]


def test_attention_tolerance_is_real_dead_zone():
    mix = AttentionExpertMixture(
        [torch.nn.Identity(), torch.nn.Identity(), torch.nn.Identity()],
        dim=8, max_loops=2, top_k=1,
        balance_weight=1.0, min_probability=0.10,
        balance_tolerance=0.25,
    )
    # Manually evaluate a distribution inside the advertised tolerance band.
    # No penalty should remain when every expert is within uniform±0.25 and the
    # configured minimum floor is respected.
    probs = torch.tensor([[0.50, 0.25, 0.25]])
    assert torch.isclose(mix._balance_loss(probs), torch.tensor(0.0), atol=1e-7)

    skewed = torch.tensor([[0.90, 0.05, 0.05]])
    assert float(mix._balance_loss(skewed)) > 0.0


def test_invalid_routing_config_fails_fast():
    try:
        LoopConfig(
            vocab_size=32, dim=32, n_layers=2, n_heads=2, head_dim=16,
            ffn_hidden_dim=64, rope_dim=8, moe_num_routed_experts=2,
            moe_top_k=3,
        )
    except ValueError as exc:
        assert "moe_top_k" in str(exc)
    else:
        raise AssertionError("invalid MoE top-k configuration was accepted")


def test_attention_top_k_and_expert_count_are_validated():
    try:
        LoopConfig(attention_mixture_num_experts=1)
    except ValueError as exc:
        assert "attention_mixture_num_experts" in str(exc)
    else:
        raise AssertionError("invalid attention expert count was accepted")
