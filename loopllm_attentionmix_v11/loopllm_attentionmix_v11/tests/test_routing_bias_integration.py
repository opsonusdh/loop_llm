import torch

from loop_transformer.config import LoopConfig
from loop_transformer.model import LoopTransformer


def test_model_updates_routing_bias_buffers():
    cfg = LoopConfig(
        vocab_size=32, dim=32, n_layers=2, n_heads=2, head_dim=16,
        ffn_hidden_dim=64, rope_dim=8, max_loops=2, min_loops=2,
        loop_sampling=False, csa_m=2, csa_top_k=3, hca_m_prime=3,
        sw_window=6, groups=2, group_dim=16, moe_num_shared_experts=1,
        moe_num_routed_experts=2, moe_top_k=1, activation_top_k=2,
        activation_bias_update_speed=0.1, moe_bias_update_speed=0.1,
        attention_mixture=False, grad_checkpointing=False,
    )
    model = LoopTransformer(cfg)
    before = [(b.ffn.activation_bias.clone(), b.ffn.expert_bias.clone()) for b in model.blocks]

    for block in model.blocks:
        block.ffn._last_activation_probs_mean = torch.tensor([0.70, 0.10, 0.10, 0.10])
        block.ffn._last_expert_load = torch.tensor([0.95, 0.05])

    model.update_routing_biases()

    for (a0, e0), block in zip(before, model.blocks):
        assert torch.any(block.ffn.activation_bias != a0)
        assert not torch.equal(block.ffn.activation_bias, a0)
        assert not torch.equal(block.ffn.expert_bias, e0)
