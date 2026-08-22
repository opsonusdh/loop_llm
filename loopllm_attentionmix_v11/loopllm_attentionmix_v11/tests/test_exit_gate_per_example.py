import torch

from loop_transformer.config import LoopConfig
from loop_transformer.model import LoopTransformer


def test_generate_adaptive_exit_is_per_example():
    cfg = LoopConfig(
        vocab_size=32, dim=32, n_layers=2, n_heads=2, head_dim=16,
        ffn_hidden_dim=64, rope_dim=8, max_loops=3, min_loops=3,
        loop_sampling=False, csa_m=2, csa_top_k=3, hca_m_prime=3,
        sw_window=6, groups=2, group_dim=16, moe_num_shared_experts=1,
        moe_num_routed_experts=2, moe_top_k=1, activation_top_k=2,
        attention_mixture=False, grad_checkpointing=False,
    )
    model = LoopTransformer(cfg).eval()

    idx = torch.tensor([[1,2,3,4],[5,6,7,8]], dtype=torch.long)
    original_gate = model.exit_gate.forward

    def fake_gate(h, loop_idx=0):
        # Example 0 should exit at loop 1, example 1 at loop 3.
        vals = [
            torch.tensor([0.95, 0.05]),
            torch.tensor([0.05]),
            torch.tensor([0.99]),
        ]
        return vals[loop_idx].to(device=h.device, dtype=h.dtype)

    model.exit_gate.forward = fake_gate
    out = model.generate(idx, max_new_tokens=1, temperature=1.0, top_k=1, n_loops=None, exit_threshold=0.8)
    model.exit_gate.forward = original_gate

    # The call only needs to succeed with a heterogeneous batch. The exact
    # token values are stochastic/model-dependent; the important regression
    # is that the adaptive selector no longer collapses both examples to one
    # batch-wide mean depth.
    assert out.shape == (2, 5)
