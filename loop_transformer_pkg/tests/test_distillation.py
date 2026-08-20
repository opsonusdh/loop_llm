"""Tests for knowledge distillation (compute_loss's teacher/distill_alpha/
distill_temperature arguments).

Classical (fixed-corpus) soft-label distillation, Hinton et al.-style:
the student's per-loop-step loss is blended between standard cross-
entropy and a temperature-scaled KL-divergence toward a frozen teacher's
output distribution. Teacher and student may be entirely different
configs (different dim, n_layers, max_loops, ...) -- only vocab_size
must match, since distillation compares output distributions over a
shared vocabulary.
"""

from __future__ import annotations

import torch

from loop_transformer import LoopConfig, LoopTransformer


def _make(cfg: LoopConfig, seed: int) -> LoopTransformer:
    torch.manual_seed(seed)
    return LoopTransformer(cfg)


class TestBasicFunctionality:
    def test_different_sized_teacher_and_student_works(self):
        teacher = _make(LoopConfig(
            vocab_size=300, dim=256, n_layers=4, n_heads=4, head_dim=64,
            ffn_hidden_dim=512, max_loops=4, groups=4, group_dim=128, loop_sampling=False,
        ), seed=0)
        student = _make(LoopConfig(
            vocab_size=300, dim=128, n_layers=2, n_heads=4, head_dim=32,
            ffn_hidden_dim=256, max_loops=2, groups=4, group_dim=64,
            rope_dim=16, loop_sampling=False,
        ), seed=1)

        x = torch.randint(0, 300, (2, 16))
        loss, step_losses = student.compute_loss(x, teacher=teacher, distill_alpha=0.5, distill_temperature=2.0)
        assert torch.isfinite(loss)
        loss.backward()


class TestTeacherIsolation:
    def test_teacher_receives_no_gradient(self):
        cfg = LoopConfig(vocab_size=300, dim=128, n_layers=2, n_heads=4, head_dim=32,
                          ffn_hidden_dim=256, rope_dim=16, groups=4, group_dim=64, loop_sampling=False)
        teacher = _make(cfg, seed=0)
        student = _make(cfg, seed=1)
        x = torch.randint(0, 300, (2, 16))

        loss, _ = student.compute_loss(x, teacher=teacher)
        loss.backward()

        assert not any(p.grad is not None and p.grad.abs().sum() > 0 for p in teacher.parameters())

    def test_student_receives_gradient(self):
        cfg = LoopConfig(vocab_size=300, dim=128, n_layers=2, n_heads=4, head_dim=32,
                          ffn_hidden_dim=256, rope_dim=16, groups=4, group_dim=64, loop_sampling=False)
        teacher = _make(cfg, seed=0)
        student = _make(cfg, seed=1)
        x = torch.randint(0, 300, (2, 16))

        loss, _ = student.compute_loss(x, teacher=teacher)
        loss.backward()

        assert all(p.grad is not None for p in student.parameters() if p.requires_grad)

    def test_teacher_training_mode_preserved(self):
        cfg = LoopConfig(vocab_size=300, dim=128, n_layers=2, n_heads=4, head_dim=32,
                          ffn_hidden_dim=256, rope_dim=16, groups=4, group_dim=64, loop_sampling=False)
        teacher = _make(cfg, seed=0)
        student = _make(cfg, seed=1)
        x = torch.randint(0, 300, (2, 16))

        teacher.train()
        student.compute_loss(x, teacher=teacher)
        assert teacher.training is True

        teacher.eval()
        student.compute_loss(x, teacher=teacher)
        assert teacher.training is False


class TestValidation:
    def test_vocab_mismatch_rejected(self):
        teacher = _make(LoopConfig(vocab_size=300, dim=128, n_layers=2, n_heads=4, head_dim=32,
                                    ffn_hidden_dim=256, rope_dim=16, groups=4, group_dim=64), seed=0)
        student = _make(LoopConfig(vocab_size=500, dim=128, n_layers=2, n_heads=4, head_dim=32,
                                    ffn_hidden_dim=256, rope_dim=16, groups=4, group_dim=64), seed=1)
        x = torch.randint(0, 300, (2, 16))
        try:
            student.compute_loss(x, teacher=teacher)
            assert False, "should have raised"
        except ValueError as e:
            assert "vocab_size" in str(e)

    def test_alpha_out_of_range_rejected(self):
        cfg = LoopConfig(vocab_size=300, dim=128, n_layers=2, n_heads=4, head_dim=32,
                          ffn_hidden_dim=256, rope_dim=16, groups=4, group_dim=64)
        teacher, student = _make(cfg, 0), _make(cfg, 1)
        x = torch.randint(0, 300, (2, 16))
        try:
            student.compute_loss(x, teacher=teacher, distill_alpha=1.5)
            assert False, "should have raised"
        except ValueError as e:
            assert "distill_alpha" in str(e)

    def test_non_positive_temperature_rejected(self):
        cfg = LoopConfig(vocab_size=300, dim=128, n_layers=2, n_heads=4, head_dim=32,
                          ffn_hidden_dim=256, rope_dim=16, groups=4, group_dim=64)
        teacher, student = _make(cfg, 0), _make(cfg, 1)
        x = torch.randint(0, 300, (2, 16))
        try:
            student.compute_loss(x, teacher=teacher, distill_temperature=0.0)
            assert False, "should have raised"
        except ValueError as e:
            assert "distill_temperature" in str(e)


class TestRegressionAgainstNoDistillation:
    def test_alpha_one_is_numerically_identical_to_no_teacher(self):
        """The critical sanity check: alpha=1.0 means 'all weight on CE,
        none on KD' -- this must produce EXACTLY the same loss as never
        passing a teacher at all, or the blending logic has a bug."""
        cfg = LoopConfig(vocab_size=200, dim=128, n_layers=3, n_heads=4, head_dim=64,
                          ffn_hidden_dim=256, max_loops=2, groups=4, group_dim=64,
                          loop_sampling=False)
        model_a = _make(cfg, seed=42)
        model_b = _make(cfg, seed=42)  # identical weights
        teacher = _make(cfg, seed=0)

        x = torch.randint(0, 200, (2, 24))
        loss_plain, _ = model_a.compute_loss(x, max_loops=2)
        loss_distill, _ = model_b.compute_loss(x, max_loops=2, teacher=teacher, distill_alpha=1.0)

        assert abs(loss_plain.item() - loss_distill.item()) < 1e-5
