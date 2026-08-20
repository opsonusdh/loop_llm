"""Tests for CSAAttention's indexer auxiliary KL-divergence loss (see
attention.py's CSAAttention docstring and _indexer_aux_loss).

The indexer's own scores only ever feed torch.topk's indices, never its
values, so without this auxiliary loss it never receives a gradient from
the main LM loss and stays at random initialization forever. These tests
verify the fix actually works: gradient reaches only the indexer, the
loss is always finite (a real NaN was found and fixed here during
development -- see _KL_MASK_VALUE's docstring in attention.py), gradient
checkpointing doesn't double-count it, and the indexer's weights
genuinely move during training.
"""

from __future__ import annotations

import torch

from loop_transformer import LoopConfig, LoopTransformer
from loop_transformer.attention import CSAAttention

INDEXER_PARAM_NAMES = {"index_down", "index_up", "index_weight", "index_key", "index_loop_scale", "index_loop_bias"}


def _first_csa_layer(model: LoopTransformer) -> CSAAttention:
    return next(m for m in model.modules() if isinstance(m, CSAAttention))


class TestGradientIsolation:
    def test_aux_loss_reaches_only_indexer_params(self):
        torch.manual_seed(0)
        csa = CSAAttention(256, 4, 64, top_k=8, window_size=8, index_heads=4,
                           groups=4, group_dim=64, m=4)
        csa.train()
        x = torch.randn(2, 16, 256)
        csa(x)
        aux_loss = csa._aux_losses[0]
        aux_loss.backward()

        for name, p in csa.named_parameters():
            is_indexer = any(name.startswith(prefix) for prefix in INDEXER_PARAM_NAMES)
            has_grad = p.grad is not None and p.grad.abs().sum().item() > 0
            if is_indexer:
                assert has_grad, f"{name} is an indexer param but got no gradient"
            else:
                assert not has_grad, f"{name} is NOT an indexer param but got gradient -- isolation broken"

    def test_full_model_indexer_gradient_coverage(self, tiny_model, batch):
        """End to end: the indexer gap that test_gradients.py used to
        document is now closed -- confirmed there at the whole-model
        level; this test confirms it's specifically the aux loss doing it."""
        tiny_model.train()
        loss, _ = tiny_model.compute_loss(batch)
        loss.backward()
        csa = _first_csa_layer(tiny_model)
        for name, p in csa.named_parameters():
            if any(name.startswith(prefix) for prefix in INDEXER_PARAM_NAMES):
                assert p.grad is not None and p.grad.abs().sum().item() > 0


class TestNumericalSafety:
    def test_aux_loss_is_always_finite(self):
        """A real NaN was found here during development: whenever a query
        has exactly one visible block, both the student's log-probability
        and the reference's probability hit an exact zero at every OTHER
        position, and 0 * (log(0) - (-inf)) = 0 * NaN with a literal -inf
        mask. Fixed via a finite sentinel (_KL_MASK_VALUE); this test
        guards against regressing that fix. Runs across several seeds and
        sequence lengths to exercise different numbers of visible blocks."""
        for seed in range(5):
            for T in [4, 8, 16, 33]:  # includes T < m (zero visible blocks) and non-multiples of m
                torch.manual_seed(seed)
                csa = CSAAttention(128, 4, 32, top_k=4, window_size=4,
                                   index_heads=2, groups=4, group_dim=32, m=4, rope_dim=16)
                csa.train()
                x = torch.randn(2, T, 128)
                csa(x)
                aux = csa._aux_losses[0]
                assert torch.isfinite(aux), f"seed={seed}, T={T}: aux loss is not finite ({aux.item()})"
                assert aux.item() >= 0, f"seed={seed}, T={T}: aux loss is negative ({aux.item()}) -- KL divergence can't be"

    def test_zero_visible_blocks_edge_case(self):
        """T < m means block 0 hasn't completed yet for any position --
        zero visible blocks everywhere. Must not crash or NaN."""
        torch.manual_seed(0)
        csa = CSAAttention(128, 4, 32, top_k=4, window_size=4, index_heads=2,
                           groups=4, group_dim=32, m=8, rope_dim=16)  # m=8 > T=4 below
        csa.train()
        x = torch.randn(2, 4, 128)
        csa(x)
        assert len(csa._aux_losses) == 1
        assert torch.isfinite(csa._aux_losses[0])


class TestCheckpointingInteraction:
    def test_no_double_counting_under_checkpointing(self):
        """torch.utils.checkpoint recomputes the forward pass during
        backward, so CSAAttention.forward() -- and the aux-loss append --
        runs twice per loop when grad_checkpointing is on: once in a
        throwaway no_grad dry-run, once with grad enabled during
        backward recomputation. Confirms the collection logic's
        requires_grad filter handles this correctly."""
        def make_model(ckpt):
            torch.manual_seed(7)
            cfg = LoopConfig(
                vocab_size=500, dim=256, n_layers=4, n_heads=4, head_dim=64,
                ffn_hidden_dim=512, max_loops=3, groups=4, group_dim=128,
                csa_top_k=8, sw_window=8, csa_m=4, hca_m_prime=8,
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
        aux1 = m1.last_csa_aux_loss.item()
        l1.backward()

        l2, _ = m2.compute_loss(x, max_loops=3)
        aux2 = m2.last_csa_aux_loss.item()
        l2.backward()

        assert abs(aux1 - aux2) < 1e-5, "aux loss value differs between checkpointed/non-checkpointed -- double-counting"

        for (n1, p1), (n2, p2) in zip(m1.named_parameters(), m2.named_parameters()):
            assert n1 == n2
            if p1.grad is None:
                assert p2.grad is None
                continue
            assert (p1.grad - p2.grad).abs().max().item() < 1e-5, f"{n1}: gradient diverges under checkpointing"


class TestLearning:
    def test_aux_loss_decreases_and_indexer_weights_move(self):
        """The real proof this is worth having: not just that gradient
        flows, but that the indexer actually learns something over
        training, closing the gap documented since early in this
        project's development."""
        torch.manual_seed(0)
        cfg = LoopConfig(
            vocab_size=200, dim=128, n_layers=3, n_heads=4, head_dim=64,
            ffn_hidden_dim=256, max_loops=2, groups=4, group_dim=64,
            csa_top_k=8, sw_window=8, csa_m=4, loop_sampling=False,
        )
        model = LoopTransformer(cfg)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

        torch.manual_seed(1)
        data = torch.randint(0, cfg.vocab_size, (4, 64))

        csa_layer = _first_csa_layer(model)
        w0 = csa_layer.index_down.weight.clone()

        aux_losses = []
        for _ in range(40):
            opt.zero_grad()
            loss, _ = model.compute_loss(data)
            aux_losses.append(model.last_csa_aux_loss.item())
            loss.backward()
            opt.step()

        weight_change = (csa_layer.index_down.weight - w0).abs().mean().item()
        assert weight_change > 1e-4, "indexer weights barely moved"
        assert aux_losses[-1] < aux_losses[0], "aux loss should decrease as the indexer learns"

    def test_csa_aux_loss_weight_zero_disables_learning_signal(self):
        """Sanity check on the weight knob itself: csa_aux_loss_weight=0
        should make the aux loss contribute nothing to total_loss (even
        though it's still computed and exposed via last_csa_aux_loss for
        inspection), so the indexer gets no gradient in that configuration."""
        torch.manual_seed(0)
        cfg = LoopConfig(
            vocab_size=200, dim=128, n_layers=3, n_heads=4, head_dim=64,
            ffn_hidden_dim=256, max_loops=2, groups=4, group_dim=64,
            csa_top_k=8, sw_window=8, csa_m=4, loop_sampling=False,
            csa_aux_loss_weight=0.0,
        )
        model = LoopTransformer(cfg)
        model.train()
        x = torch.randint(0, 200, (2, 32))
        loss, _ = model.compute_loss(x)
        loss.backward()

        csa_layer = _first_csa_layer(model)
        for name, p in csa_layer.named_parameters():
            if any(name.startswith(prefix) for prefix in INDEXER_PARAM_NAMES):
                assert p.grad is None or p.grad.abs().sum().item() == 0
