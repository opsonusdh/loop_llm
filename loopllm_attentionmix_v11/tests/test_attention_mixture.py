import torch

from src.loop_transformer.config import LoopConfig
from src.loop_transformer.model import LoopTransformer


def _cfg(diffusion: bool = False) -> LoopConfig:
    return LoopConfig(
        vocab_size=64,
        dim=32,
        n_layers=3,
        n_heads=4,
        head_dim=8,
        ffn_hidden_dim=64,
        rope_dim=8,
        max_loops=4,
        min_loops=4,
        loop_sampling=False,
        tie_embeddings=True,
        csa_m=2,
        csa_top_k=2,
        hca_m_prime=4,
        sw_window=4,
        groups=4,
        group_dim=16,
        moe_num_shared_experts=1,
        moe_num_routed_experts=2,
        moe_top_k=1,
        activation_top_k=1,
        attention_mixture=True,
        attention_mixture_top_k=1,
        attention_mixture_num_experts=3,
        attention_mixture_start_layer=1,
        attention_mixture_diversity_weight=0.001,
        diffusion_blocks=diffusion,
        diffusion_cond_dim=16,
        diffusion_sigma_min=0.01,
        diffusion_sigma_max=2.0,
    )


def test_attention_mixture_routes_and_backprops():
    torch.manual_seed(4)
    model = LoopTransformer(_cfg()).train()
    idx = torch.randint(0, model.cfg.vocab_size, (2, 10))
    loss, _ = model.compute_loss(idx, max_loops=4)
    loss.backward()
    assert torch.isfinite(loss)
    mixed_layers = 0
    for block in model.blocks:
        debug = block.attention_routing_debug()
        if not debug:
            continue
        mixed_layers += 1
        probs = debug["attention_probs_mean"]
        assert torch.isfinite(probs).all()
        assert probs.numel() == 3
        assert torch.isclose(probs.sum(), torch.tensor(1.0), atol=1e-5)
        if block.attn.router.loop_specific_router:
            params = [head.weight for head in block.attn.router.router_heads]
        else:
            params = [block.attn.router.weight]
        assert any(p.grad is not None for p in params)
        assert all(torch.isfinite(p.grad).all() for p in params if p.grad is not None)
    assert mixed_layers == 2


def test_diffusion_with_attention_mixture_is_finite():
    torch.manual_seed(5)
    model = LoopTransformer(_cfg(diffusion=True)).train()
    idx = torch.randint(0, model.cfg.vocab_size, (2, 10))
    loss, info = model.diffusion_blocks_loss(idx)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(info["ce"])
    for name, param in model.named_parameters():
        if param.grad is not None:
            assert torch.isfinite(param.grad).all(), name


def test_attention_mixture_diversity_loss_has_gradient():
    torch.manual_seed(17)
    model = LoopTransformer(_cfg()).train()
    idx = torch.randint(0, model.cfg.vocab_size, (2, 10))
    model.forward(idx, max_loops=4)
    losses = []
    for block in model.blocks:
        if block.attn.__class__.__name__ == "AttentionExpertMixture":
            losses.append(block.attn.collect_attention_aux_loss())
    aux = torch.stack(losses).mean()
    assert torch.isfinite(aux)
    # The auxiliary must remain connected to the routing graph.
    aux.backward()
    grads = []
    for block in model.blocks:
        if block.attn.__class__.__name__ == "AttentionExpertMixture":
            if block.attn.router.loop_specific_router:
                grads.extend([head.weight.grad for head in block.attn.router.router_heads])
            else:
                grads.append(block.attn.router.weight.grad)
    assert any(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0 for g in grads)
