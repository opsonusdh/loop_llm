"""Tests for embedding factorization (LoopConfig.embed_dim / embed_dim_out).

Decoupled (asymmetric) by design, not forced symmetric like classic
ALBERT: Chung et al., "Rethinking Embedding Coupling in Pre-trained
Language Models" (Google, ICLR 2021) found that shrinking both input
and output embeddings together specifically hurts vocab-diverse /
multilingual models. embed_dim (input) and embed_dim_out (output) are
independently configurable; tying is only possible when they match.
"""

from __future__ import annotations

import warnings

import pytest
import torch

from loop_transformer import LoopConfig, LoopConfigError, LoopTransformer


class TestBackwardCompatibility:
    def test_default_has_no_projection_layers(self):
        torch.manual_seed(0)
        cfg = LoopConfig(vocab_size=1000, dim=256, n_layers=3, n_heads=4, head_dim=64,
                          ffn_hidden_dim=512, groups=4, group_dim=128)
        model = LoopTransformer(cfg)
        assert model.embed_proj_in is None
        assert model.embed_proj_out is None
        assert model.tok_emb.weight.shape == (1000, 256)


class TestSymmetricFactorization:
    def test_reduces_parameter_count(self):
        torch.manual_seed(0)
        base_cfg = dict(vocab_size=1000, dim=256, n_layers=3, n_heads=4, head_dim=64,
                         ffn_hidden_dim=512, groups=4, group_dim=128)
        n_unfactored = LoopTransformer(LoopConfig(**base_cfg)).num_parameters()
        n_factored = LoopTransformer(LoopConfig(**base_cfg, embed_dim=32)).num_parameters()
        assert n_factored < n_unfactored

    def test_tying_still_works_symmetrically(self):
        torch.manual_seed(0)
        cfg = LoopConfig(vocab_size=1000, dim=256, n_layers=3, n_heads=4, head_dim=64,
                          ffn_hidden_dim=512, groups=4, group_dim=128, embed_dim=32)
        model = LoopTransformer(cfg)
        assert model.tok_emb.weight is model.lm_head.weight
        assert model.tok_emb.weight.shape == (1000, 32)
        assert model.embed_proj_in is not None
        assert model.embed_proj_out is not None

    def test_forward_and_backward_work(self, ):
        torch.manual_seed(0)
        cfg = LoopConfig(vocab_size=1000, dim=256, n_layers=3, n_heads=4, head_dim=64,
                          ffn_hidden_dim=512, groups=4, group_dim=128, embed_dim=32)
        model = LoopTransformer(cfg)
        x = torch.randint(0, 1000, (2, 16))
        loss, _ = model.compute_loss(x)
        assert torch.isfinite(loss)
        loss.backward()


class TestAsymmetricFactorization:
    def test_independent_shapes(self):
        torch.manual_seed(0)
        cfg = LoopConfig(vocab_size=1000, dim=256, n_layers=3, n_heads=4, head_dim=64,
                          ffn_hidden_dim=512, groups=4, group_dim=128,
                          embed_dim=32, embed_dim_out=128, tie_embeddings=False)
        model = LoopTransformer(cfg)
        assert model.tok_emb.weight.shape == (1000, 32)
        assert model.lm_head.weight.shape == (1000, 128)

    def test_forward_and_backward_work(self):
        torch.manual_seed(0)
        cfg = LoopConfig(vocab_size=1000, dim=256, n_layers=3, n_heads=4, head_dim=64,
                          ffn_hidden_dim=512, groups=4, group_dim=128,
                          embed_dim=32, embed_dim_out=128, tie_embeddings=False)
        model = LoopTransformer(cfg)
        x = torch.randint(0, 1000, (2, 16))
        loss, _ = model.compute_loss(x)
        assert torch.isfinite(loss)
        loss.backward()


class TestValidation:
    def test_embed_dim_out_without_embed_dim_rejected(self):
        with pytest.raises(LoopConfigError, match="embed_dim_out"):
            LoopConfig(embed_dim_out=128)

    def test_asymmetric_plus_tied_rejected(self):
        with pytest.raises(LoopConfigError, match="tie_embeddings"):
            LoopConfig(dim=256, embed_dim=32, embed_dim_out=128, tie_embeddings=True)

    def test_symmetric_via_explicit_match_plus_tied_accepted(self):
        LoopConfig(dim=256, embed_dim=32, embed_dim_out=32, tie_embeddings=True)

    def test_embed_dim_at_least_dim_warns_not_errors(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            LoopConfig(dim=128, embed_dim=256)
            assert len(w) == 1
            assert "reduce parameters" in str(w[0].message)

    def test_non_positive_embed_dim_rejected(self):
        with pytest.raises(LoopConfigError):
            LoopConfig(embed_dim=0)

    def test_non_positive_embed_dim_out_rejected(self):
        with pytest.raises(LoopConfigError):
            LoopConfig(embed_dim=32, embed_dim_out=0)


class TestPreservedInvariants:
    """Factorization touches the embedding pathway that feeds everything
    downstream -- re-verify the two properties that matter most."""

    def test_causality_preserved(self):
        torch.manual_seed(0)
        cfg = LoopConfig(
            vocab_size=500, dim=256, n_layers=4, n_heads=4, head_dim=64,
            ffn_hidden_dim=512, max_loops=3, groups=4, group_dim=128,
            csa_top_k=8, sw_window=6, csa_m=4, hca_m_prime=8,
            embed_dim=32, embed_dim_out=64, tie_embeddings=False, loop_sampling=False,
        )
        model = LoopTransformer(cfg)
        model.eval()

        torch.manual_seed(1)
        T, cutoff = 40, 15
        x1 = torch.randint(0, cfg.vocab_size, (1, T))
        x2 = x1.clone()
        x2[:, cutoff:] = torch.randint(0, cfg.vocab_size, (1, T - cutoff))
        with torch.no_grad():
            logits1, _ = model(x1)
            logits2, _ = model(x2)

        for l1, l2 in zip(logits1, logits2):
            assert (l1[:, :cutoff] - l2[:, :cutoff]).abs().max().item() == 0.0
        assert (logits1[-1][:, cutoff:] - logits2[-1][:, cutoff:]).abs().max().item() > 0.0

    def test_checkpointing_consistency_preserved(self):
        def make_model(ckpt):
            torch.manual_seed(7)
            c = LoopConfig(
                vocab_size=500, dim=256, n_layers=3, n_heads=4, head_dim=64,
                ffn_hidden_dim=512, max_loops=3, groups=4, group_dim=128,
                embed_dim=32, tie_embeddings=True, grad_checkpointing=ckpt,
                loop_sampling=False,
            )
            return LoopTransformer(c)

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
            assert (p1.grad - p2.grad).abs().max().item() < 1e-5
