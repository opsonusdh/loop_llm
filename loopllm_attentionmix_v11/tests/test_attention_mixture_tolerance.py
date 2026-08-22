import torch

from src.loop_transformer.attention_mixture import AttentionExpertMixture


class ToyExpert(torch.nn.Module):
    def forward(self, x, loop_idx=0, attention_mask=None):
        return x


def test_attention_probability_floor_and_gradient():
    torch.manual_seed(0)
    mix = AttentionExpertMixture(
        [ToyExpert(), ToyExpert(), ToyExpert()],
        dim=12,
        max_loops=4,
        top_k=1,
        min_probability=0.10,
        balance_tolerance=0.25,
    )
    x = torch.randn(2, 7, 12, requires_grad=True)
    y = mix(x, loop_idx=2)
    probs = mix._last_probs_mean
    assert probs is not None
    assert torch.all(probs >= 0.10 - 1e-6)
    assert torch.allclose(probs.sum(), torch.tensor(1.0), atol=1e-6)
    loss = y.square().mean() + mix.collect_attention_aux_loss()
    loss.backward()
    grads = [head.weight.grad for head in mix.router.router_heads] if mix.router.loop_specific_router else [mix.router.weight.grad]
    assert any(g is not None for g in grads)
    assert all(torch.isfinite(g).all() for g in grads if g is not None)


def test_floor_is_configurable_and_validated():
    try:
        AttentionExpertMixture(
            [ToyExpert(), ToyExpert(), ToyExpert()],
            dim=8,
            max_loops=2,
            min_probability=0.34,
        )
    except ValueError:
        return
    raise AssertionError('min_probability >= 1/num_experts must be rejected')
