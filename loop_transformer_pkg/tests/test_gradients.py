"""Gradient-flow verification: which parameters do/don't receive
gradient, and that gradient checkpointing produces mathematically
identical gradients to the non-checkpointed path.
"""

from __future__ import annotations

import torch

from loop_transformer import LoopConfig, LoopTransformer


def test_full_gradient_coverage_including_csa_indexer(tiny_model, batch):
    """
    The CSA indexer's own selection scores only ever feed torch.topk's
    INDICES, never its values -- index selection has no gradient from the
    main LM loss alone. That gap is fixed via CSAAttention's auxiliary
    KL-divergence loss (see attention.py's _indexer_aux_loss), which
    trains the indexer directly, gradient-isolated from everything else.
    So the expectation now is FULL coverage -- every parameter gets a
    gradient from one loss term or the other.
    """
    tiny_model.train()
    loss, _ = tiny_model.compute_loss(batch, max_loops=tiny_model.cfg.max_loops)
    loss.backward()

    no_grad = [n for n, p in tiny_model.named_parameters()
               if p.requires_grad and p.grad is None]
    assert not no_grad, f"unexpected parameters with no gradient: {no_grad}"


def test_checkpointing_gives_identical_gradients():
    def make_model(ckpt: bool) -> LoopTransformer:
        torch.manual_seed(7)
        cfg = LoopConfig(
            vocab_size=500, dim=256, n_layers=3, n_heads=4, head_dim=64,
            ffn_hidden_dim=512, max_loops=3, groups=4, group_dim=128,
            tie_embeddings=True, grad_checkpointing=ckpt, loop_sampling=False,
        )
        return LoopTransformer(cfg)

    m1, m2 = make_model(False), make_model(True)
    m2.load_state_dict(m1.state_dict())
    m1.train()
    m2.train()

    torch.manual_seed(99)
    x = torch.randint(0, 500, (2, 24))

    l1, _ = m1.compute_loss(x, max_loops=3)
    l1.backward()
    l2, _ = m2.compute_loss(x, max_loops=3)
    l2.backward()

    for (n1, p1), (n2, p2) in zip(m1.named_parameters(), m2.named_parameters()):
        assert n1 == n2
        if p1.grad is None:
            assert p2.grad is None
            continue
        max_diff = (p1.grad - p2.grad).abs().max().item()
        assert max_diff < 1e-5, f"{n1}: checkpointed gradient diverges by {max_diff}"


def test_weight_tying_shares_the_same_tensor(tiny_config):
    tiny_config.tie_embeddings = True
    torch.manual_seed(0)
    model = LoopTransformer(tiny_config)
    assert model.lm_head.weight is model.tok_emb.weight


def test_weight_tying_disabled_uses_separate_tensors(tiny_config):
    tiny_config.tie_embeddings = False
    torch.manual_seed(0)
    model = LoopTransformer(tiny_config)
    assert model.lm_head.weight is not model.tok_emb.weight
