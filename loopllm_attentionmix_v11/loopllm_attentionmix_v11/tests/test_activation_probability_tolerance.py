import torch

from src.loop_transformer.feedforward import FeedForward


def test_activation_dense_probability_floor_and_specialization():
    torch.manual_seed(0)
    ff = FeedForward(16, 32, activation_top_k=2, activation_min_probability=0.10, activation_balance_tolerance=0.25)
    x = torch.randn(2, 5, 16)
    ff(x, loop_idx=0)
    dense = ff._last_activation_probs_dense_mean
    assert dense is not None
    assert torch.all(dense >= 0.10 - 1e-6)
    assert torch.allclose(dense.sum(), torch.tensor(1.0), atol=1e-6)


def test_activation_balance_tolerance_has_zero_penalty_inside_band():
    ff = FeedForward(8, 16, activation_top_k=4, activation_min_probability=0.10, activation_balance_tolerance=0.25)
    # q=[0.55, .20, .15, .10] is inside [0.10, 0.5833...] for K=4.
    q = torch.tensor([0.49, 0.21, 0.20, 0.10], dtype=torch.float32)
    loss = ff._activation_balance_loss(q.repeat(8, 1))
    assert loss.item() == 0.0


def test_activation_balance_penalizes_population_floor_violation():
    ff = FeedForward(8, 16, activation_top_k=4, activation_min_probability=0.10, activation_balance_tolerance=0.25)
    q = torch.tensor([0.70, 0.20, 0.09, 0.01], dtype=torch.float32)
    loss = ff._activation_balance_loss(q.repeat(8, 1))
    assert loss.item() > 0.0


def test_activation_hard_batch_floor_penalty():
    ff = FeedForward(8, 16, activation_top_k=2, activation_min_probability=0.10, activation_balance_tolerance=0.25)
    dense = torch.full((8, 4), 0.25)
    hard = torch.tensor([[0.5, 0.5, 0.0, 0.0]] * 8)
    loss = ff._activation_balance_loss(dense, hard)
    assert loss.item() > 0.0


def test_activation_bias_update_respects_tolerance_band():
    ff = FeedForward(8, 16, activation_top_k=2, activation_min_probability=0.10, activation_balance_tolerance=0.25, activation_bias_update_speed=0.1)
    ff._last_activation_probs_mean = torch.tensor([0.49, 0.25, 0.15, 0.11])
    before = ff.activation_bias.clone()
    ff._update_routing_biases()
    assert torch.allclose(ff.activation_bias, before)
