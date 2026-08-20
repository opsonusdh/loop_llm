"""Config validation should fail fast, at construction time, with a
message that says what's wrong -- not three modules deep during the
first forward pass.
"""

from __future__ import annotations

import pytest

from loop_transformer import LoopConfig, LoopConfigError


def test_valid_config_constructs_cleanly():
    LoopConfig(dim=512, n_heads=8, head_dim=64, groups=4, rope_dim=32)


def test_rope_dim_exceeding_head_dim_rejected():
    with pytest.raises(LoopConfigError, match="rope_dim"):
        LoopConfig(head_dim=32, rope_dim=64)


def test_rope_dim_odd_rejected():
    with pytest.raises(LoopConfigError, match="rope_dim"):
        LoopConfig(head_dim=64, rope_dim=33)


def test_groups_not_dividing_dim_in_rejected():
    with pytest.raises(LoopConfigError, match="groups"):
        LoopConfig(n_heads=7, head_dim=65, groups=8)


def test_min_loops_exceeding_max_loops_rejected():
    with pytest.raises(LoopConfigError, match="min_loops"):
        LoopConfig(min_loops=5, max_loops=3)


def test_negative_beta_entropy_rejected():
    with pytest.raises(LoopConfigError, match="beta_entropy"):
        LoopConfig(beta_entropy=-0.1)


def test_multiple_errors_all_reported_together():
    with pytest.raises(LoopConfigError) as exc_info:
        LoopConfig(dim=-1, n_layers=0, max_loops=0)
    msg = str(exc_info.value)
    assert "dim" in msg
    assert "n_layers" in msg
    assert "max_loops" in msg


@pytest.mark.parametrize("field,value", [
    ("vocab_size", 0),
    ("dim", 0),
    ("n_layers", 0),
    ("n_heads", 0),
    ("head_dim", 0),
    ("ffn_hidden_dim", 0),
    ("csa_m", 0),
    ("csa_top_k", 0),
    ("hca_m_prime", 0),
    ("sw_window", 0),
    ("groups", 0),
    ("group_dim", 0),
])
def test_non_positive_fields_rejected(field, value):
    with pytest.raises(LoopConfigError):
        LoopConfig(**{field: value})
