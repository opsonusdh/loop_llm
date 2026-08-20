from __future__ import annotations

import torch

from src.loop_transformer import LoopConfig, LoopTransformer
from src.loop_transformer.diffusion import edm_preconditioning, sample_log_normal_sigma


def tiny_config() -> LoopConfig:
    return LoopConfig(
        vocab_size=32,
        dim=32,
        n_layers=4,
        n_heads=4,
        head_dim=8,
        ffn_hidden_dim=64,
        rope_dim=8,
        max_loops=4,
        min_loops=1,
        loop_sampling=False,
        csa_m=2,
        csa_top_k=4,
        hca_m_prime=4,
        sw_window=8,
        groups=4,
        group_dim=8,
        moe_num_shared_experts=1,
        moe_num_routed_experts=2,
        moe_top_k=1,
        activation_top_k=2,
        activation_balance_weight=0.0,
        moe_aux_loss_weight=0.0,
        csa_aux_loss_weight=0.01,
        diffusion_blocks=True,
        diffusion_cond_dim=16,
        diffusion_sigma_min=0.1,
        diffusion_sigma_max=2.0,
        diffusion_p_mean=-0.2,
        diffusion_p_std=0.35,
    )


def test_sigma_sampler_and_edm_are_finite() -> None:
    sigma = sample_log_normal_sigma(64, device=torch.device("cpu"), sigma_min=0.1, sigma_max=2.0)
    assert float(sigma.min()) >= 0.1
    assert float(sigma.max()) <= 2.0
    edm = edm_preconditioning(sigma)
    for value in (edm.c_in, edm.c_skip, edm.c_out, edm.c_noise, edm.weight):
        assert torch.isfinite(value).all()


def test_diffusion_loss_has_single_pass_and_gradients() -> None:
    model = LoopTransformer(tiny_config())
    idx = torch.randint(0, 32, (4, 32))
    calls = {"n": 0}
    original = model._one_loop

    def wrapped(*args, **kwargs):
        calls["n"] += 1
        return original(*args, **kwargs)

    model._one_loop = wrapped  # type: ignore[method-assign]
    loss, info = model.diffusion_blocks_loss(idx)
    assert calls["n"] == 1
    assert torch.isfinite(loss)
    assert torch.isfinite(info["ce"])
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads
    assert all(torch.isfinite(g).all() for g in grads)


def test_diffusion_checkpoint_preserves_normal_inference() -> None:
    model = LoopTransformer(tiny_config()).eval()
    idx = torch.randint(0, 32, (2, 24))
    with torch.no_grad():
        logits, lambdas = model(idx, max_loops=4)
        generated = model.diffusion_euler_sample(idx, num_steps=4, top_k=5, seed=7)
    assert len(logits) == 4
    assert len(lambdas) == 4
    assert generated.shape == (2, 25)
    assert torch.isfinite(logits[-1]).all()
