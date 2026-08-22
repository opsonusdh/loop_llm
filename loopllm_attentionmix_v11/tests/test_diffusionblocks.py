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
        diffusion_num_blocks=2,
    )


def test_sigma_sampler_and_edm_are_finite() -> None:
    sigma = sample_log_normal_sigma(64, device=torch.device("cpu"), sigma_min=0.1, sigma_max=2.0)
    assert float(sigma.min()) >= 0.1
    assert float(sigma.max()) <= 2.0
    edm = edm_preconditioning(sigma)
    for value in (edm.c_in, edm.c_skip, edm.c_out, edm.c_noise, edm.weight):
        assert torch.isfinite(value).all()


def test_diffusion_loss_trains_one_physical_block_and_has_gradients() -> None:
    model = LoopTransformer(tiny_config())
    idx = torch.randint(0, 32, (4, 32))
    calls = {"n": 0}
    loss, info = model.diffusion_blocks_loss(idx)
    assert 0 <= int(info["block_idx"]) < 2
    assert int(info["num_blocks"]) == 2
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


def test_diffusion_mask_has_no_all_invalid_noisy_rows_and_attention_is_finite() -> None:
    from src.loop_transformer.diffusion import build_diffusion_causal_mask

    model = LoopTransformer(tiny_config()).train()
    clean_len = 12
    idx = torch.randint(0, 32, (2, clean_len))
    clean = model.tok_emb(idx)
    target = torch.randn_like(clean)
    combined = torch.cat([clean, target], dim=1)
    mask = build_diffusion_causal_mask(clean_len, clean_len, device=combined.device)

    # Exact safety invariant behind the reported bug: every noisy query has
    # its own diagonal key, including noisy positions with no clean predecessor.
    for i in (0, 3, 5, 11):
        row = clean_len + i
        assert bool(mask[row, row])
        assert bool(mask[row].any())

    sigma = torch.full((2,), 0.5)
    from src.loop_transformer.diffusion import edm_preconditioning, log_sigma_embedding
    edm = edm_preconditioning(sigma, sigma_data=model.cfg.diffusion_sigma_data)
    time = model.diffusion_time_mlp(log_sigma_embedding(edm.c_noise, model.cfg.diffusion_cond_dim))
    out = model._one_loop(
        [combined], loop_idx=0, diffusion_cond=time,
        attention_mask=mask,
    )
    assert torch.isfinite(out).all()
    out.square().mean().backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads
    assert all(torch.isfinite(g).all() for g in grads)


def test_diffusion_block_partition_updates_selected_block_only() -> None:
    cfg = tiny_config()
    model = LoopTransformer(cfg).train()
    idx = torch.randint(0, cfg.vocab_size, (3, 24))
    torch.manual_seed(7)
    loss, info = model.diffusion_blocks_loss(idx, block_override=1)
    loss.backward()

    selected = [p for i in (2, 3) for p in model.blocks[i].parameters() if p.grad is not None]
    other = [p for i, block in enumerate(model.blocks) if i not in (2, 3) for p in block.parameters() if p.grad is not None]
    assert selected, "selected diffusion block received no gradients"
    assert not other, "non-selected physical blocks received gradients"
    assert torch.isfinite(selected[0].grad).all()


def test_diffusion_block_sigma_boundaries_are_ordered() -> None:
    from src.loop_transformer.diffusion import get_block_sigmas
    sigmas = get_block_sigmas(3, sigma_min=0.1, sigma_max=3.0, p_mean=-0.2, p_std=0.35)
    assert sigmas.shape == (4,)
    assert torch.all(sigmas[1:] > sigmas[:-1])
    assert abs(float(sigmas[0]) - 0.1) < 1e-6
    assert abs(float(sigmas[-1]) - 3.0) < 1e-6


def test_true_blockwise_diffusion_saves_autograd_activation_memory() -> None:
    """Guard the core memory claim against accidentally reverting to full-stack training."""
    import torch.nn.functional as F

    cfg = tiny_config()
    cfg = LoopConfig(**{**cfg.__dict__, "n_layers": 4, "diffusion_num_blocks": 2})
    idx = torch.randint(0, cfg.vocab_size, (2, 20))

    def old_style_full_stack(model: LoopTransformer) -> torch.Tensor:
        context = idx[:, :-1]
        target = idx[:, 1:]
        emb = model.tok_emb(target)
        sigma = torch.full((idx.size(0),), 0.5)
        z = emb + sigma[:, None, None] * torch.randn_like(emb)
        pred = model._diffusion_denoise_once(context, z, sigma)
        return F.cross_entropy(model.lm_head(pred).reshape(-1, cfg.vocab_size), target.reshape(-1))

    def saved_bytes(fn) -> int:
        m = LoopTransformer(cfg).train()
        total = [0]

        def pack(t):
            total[0] += t.numel() * t.element_size()
            return t

        with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
            loss = fn(m)
        loss.backward()
        return total[0]

    full = saved_bytes(old_style_full_stack)
    blockwise = saved_bytes(lambda m: m.diffusion_blocks_loss(idx, block_override=1)[0])
    assert blockwise < full * 0.75, (full, blockwise)
