import torch
from src.loop_transformer import LoopConfig, LoopTransformer


def tiny_cfg():
    return LoopConfig(
        vocab_size=32, dim=32, n_layers=2, n_heads=2, head_dim=16,
        ffn_hidden_dim=64, rope_dim=8, max_loops=4, beta_entropy=0.02,
        csa_m=2, csa_top_k=3, hca_m_prime=3, sw_window=6,
        groups=2, group_dim=16, moe_num_shared_experts=1,
        moe_num_routed_experts=2, moe_top_k=1, activation_top_k=2,
        loop_sampling=False, loop_supervision_weight=0.1,
        loop_monotonic_weight=0.0, loop_monotonic_margin=0.001,
        loop_refinement_weight=0.05, loop_refinement_margin=0.001,
        loop_task_weight=0.01, loop_task_mode='horizon',
        exit_gate_loop_embed_dim=8, tie_embeddings=True,
        grad_checkpointing=False,
    )


def test_refinement_task_and_gate_gradient():
    torch.manual_seed(0)
    model = LoopTransformer(tiny_cfg())
    idx = torch.randint(0, 32, (3, 24))
    loss, per_loop = model.compute_loss(idx, max_loops=4)
    assert loss.ndim == 0 and torch.isfinite(loss)
    assert per_loop.shape == (4,)
    assert model.exit_gate.proj.in_features == 40
    assert model.exit_gate.loop_embedding is not None
    assert model.last_loop_refinement_loss >= 0
    assert model.last_loop_task_loss >= 0
    loss.backward()
    grad = model.exit_gate.loop_embedding.weight.grad
    assert grad is not None and torch.isfinite(grad).all()
    assert float(grad.abs().sum()) > 0
