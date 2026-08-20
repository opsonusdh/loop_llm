"""Zero-init verification (Ohio State, arXiv:2604.07822).

Every sub-layer type's output projection is zero-init so each loop
starts as a zero contribution -- see the derivation in attnres.py for
why this stabilizes our AttnRes-based recurrence even though we don't
have a single evolving hidden state like the source paper's setting.
"""

from __future__ import annotations

import torch

from loop_transformer import (
    CSAAttention,
    FeedForward,
    HCAAttention,
    SlidingWindowAttention,
)


def test_sliding_window_attention_zero_at_init():
    torch.manual_seed(0)
    mod = SlidingWindowAttention(256, 4, 64, window_size=8)
    out = mod(torch.randn(2, 16, 256))
    assert out.abs().max().item() == 0.0


def test_csa_attention_zero_at_init():
    torch.manual_seed(0)
    mod = CSAAttention(256, 4, 64, top_k=8, window_size=8, index_heads=4,
                        groups=4, group_dim=64)
    out = mod(torch.randn(2, 16, 256))
    assert out.abs().max().item() == 0.0


def test_hca_attention_zero_at_init():
    torch.manual_seed(0)
    mod = HCAAttention(256, 4, 64, m_prime=8, window_size=8, groups=4, group_dim=64)
    out = mod(torch.randn(2, 16, 256))
    assert out.abs().max().item() == 0.0


def test_feedforward_zero_at_init():
    torch.manual_seed(0)
    mod = FeedForward(256, 512)
    out = mod(torch.randn(2, 16, 256))
    assert out.abs().max().item() == 0.0


def test_full_model_first_loop_is_pure_embedding_echo(tiny_model, batch):
    """
    With every sub-layer zero at init, the model's logits at init are
    driven entirely by the tied embedding matrix dotted with itself --
    i.e. argmax(logits) == input token, everywhere. This is a bad guess
    on i.i.d. random data (which is exactly what `batch` is) but confirms
    the derivation in attnres.py rather than silently masking a bug where
    the model happens to produce reasonable-looking noise instead.
    """
    tiny_model.eval()
    with torch.no_grad():
        loop_logits, _ = tiny_model(batch, max_loops=1)
    pred = loop_logits[0].argmax(dim=-1)
    match_rate = (pred[:, :-1] == batch[:, :-1]).float().mean().item()
    assert match_rate == 1.0
