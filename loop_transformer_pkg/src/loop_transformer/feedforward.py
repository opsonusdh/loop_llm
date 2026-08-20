"""DeepSeekMoE-style sparse FFN with shared experts and activation routing.

The original dense SwiGLU FFN is replaced by a compact DeepSeekMoE-inspired
Mixture-of-Experts block: one always-on shared expert plus top-k routed
fine-grained experts.  Inside every expert, a lightweight router mixes four
activation functions.  The activation router is encouraged to use all four
activations on average, while remaining free to specialize per token.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import safe_eps


class SwiGLUExpert(nn.Module):
    """A small SwiGLU-style expert with zero-initialized output projection."""

    def __init__(self, dim: int, hidden_dim: int, num_activations: int = 4):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, dim, bias=False)
        nn.init.zeros_(self.w3.weight)
        self.num_activations = num_activations

    @staticmethod
    def _activation_stack(x: torch.Tensor) -> torch.Tensor:
        """Return four nonlinear transforms with matched sign/scale families.

        The third slot is relu(x)^2 / (1 + relu(x)) rather than plain
        relu(x)^2. Plain squared-ReLU grows quadratically and unboundedly:
        at gate=5 it's ~5x SiLU/GELU and ~25x tanh; at gate=10, ~10x and
        ~100x (checked numerically). The router that produces
        activation_probs has no visibility into this gap -- its logits
        come from `x`, not from the activation magnitudes -- so whenever a
        gate value lands in that tail, this term's contribution to
        mixed_gate can dominate regardless of how little probability mass
        it was given, and its local gradient 2*relu(x) is unbounded too.
        That's a plausible source of the large, spiky pre-clip gradient
        norms already observed in training.

        Dividing by (1 + relu(x)) leaves the small-x regime essentially
        untouched (ratio -> relu(x)^2 as x -> 0, matching the original
        term exactly where the other three activations are already
        comparable) while capping the asymptotic growth rate to linear --
        the same asymptotic rate SiLU and GELU already have -- instead of
        quadratic. This is deliberately more surgical than rescaling the
        whole 4-way stack: SiLU/GELU are untouched, and small-gate
        training dynamics (where all four terms already track each other)
        aren't perturbed -- only the unbounded tail is capped.

        The fourth slot is ELU (alpha=1), not tanh. tanh saturates
        SYMMETRICALLY: tanh'(x) = 1 - tanh(x)^2 -> 0 as |x| -> infinity in
        BOTH directions. In a looped architecture -- effectively a very
        deep, weight-tied network once unrolled across max_loops -- that
        vanishing gradient compounds across iterations, working directly
        against the model's ability to learn useful iterative computation,
        which is the entire point of the loop mechanism. tanh's output is
        also capped at magnitude 1 while SiLU/GELU/bounded-ReLU^2 are
        unbounded for positive x, so whenever the router picks tanh it
        structurally contributes the LEAST signal of the four -- the same
        scale-mismatch class fixed above for ReLU^2, just underpowered
        instead of overpowered.

        ELU(x) = x for x>0, exp(x)-1 for x<=0. For x>0: ELU'(x) = 1
        exactly, matching where SiLU'/GELU' head asymptotically -- no
        vanishing gradient and no scale mismatch on the positive side,
        which is where large gate values are more likely to land (see the
        ReLU^2 analysis above). For x<=0 it still saturates toward -1,
        keeping a genuinely different, bounded qualitative character from
        the other three (the actual reason to want a distinct 4th
        activation) without symmetric vanishing gradient. alpha=1 also
        makes ELU C1-continuous at x=0 (its slope matches the identity's
        from both sides), a nicer numerical property than the kink
        bounded_relu_sq already has from its ReLU term.
        """
        relu_sq = F.relu(x).square()
        bounded_relu_sq = relu_sq / (1.0 + F.relu(x))
        return torch.stack(
            (
                F.silu(x),
                F.gelu(x),
                bounded_relu_sq,
                F.elu(x),
            ),
            dim=-1,
        )

    def forward(
        self,
        x: torch.Tensor,
        activation_probs: torch.Tensor,
    ) -> torch.Tensor:
        gate = self.w1(x)
        value = self.w2(x)
        # [*, hidden, 4] @ [*, hidden, 4] -> [*, hidden]
        activations = self._activation_stack(gate)
        mixed_gate = (activations * activation_probs.unsqueeze(-2)).sum(dim=-1)
        return self.w3(mixed_gate * value)


class FeedForward(nn.Module):
    """DeepSeekMoE-style shared + routed FFN with 4-way activation routing.

    Compute is controlled by ``expert_hidden_dim`` and ``moe_top_k``.  The
    recommended small-model setting is one shared expert + one routed expert,
    each with half the original dense FFN width, keeping active FFN multiply
    width approximately equal to the old dense SwiGLU while increasing total
    parameter capacity through the inactive routed experts.

    ``activation_top_k`` controls how many of the four activations can be active
    for each token (1..4). ``activation_balance_loss`` is KL(mean routing ||
    uniform), so a non-uniform batch-average activation distribution is explicitly
    penalized. This does *not* force every token to use all activations: individual
    tokens remain free to choose a sparse specialist set. The same design is used
    for the expert router via
    a standard top-k load-balance auxiliary loss.
    """

    NUM_ACTIVATIONS = 4

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        *,
        num_shared_experts: int = 1,
        num_routed_experts: int = 4,
        moe_top_k: int = 1,
        expert_hidden_dim: Optional[int] = None,
        max_loops: int = 4,
        activation_balance_weight: float = 0.01,
        moe_aux_loss_weight: float = 0.01,
        activation_top_k: int = 4,
        activation_bias_update_speed: float = 0.001,
        moe_bias_update_speed: float = 0.001,
    ):
        super().__init__()
        if num_shared_experts < 1:
            raise ValueError("num_shared_experts must be >= 1")
        if num_routed_experts < 1:
            raise ValueError("num_routed_experts must be >= 1")
        if not 1 <= moe_top_k <= num_routed_experts:
            raise ValueError("moe_top_k must be in [1, num_routed_experts]")
        if max_loops < 1:
            raise ValueError("max_loops must be >= 1")
        if not 1 <= activation_top_k <= self.NUM_ACTIVATIONS:
            raise ValueError(
                f"activation_top_k must be in [1, {self.NUM_ACTIVATIONS}]"
            )

        self.dim = dim
        self.hidden_dim = hidden_dim
        self.num_shared_experts = num_shared_experts
        self.num_routed_experts = num_routed_experts
        self.moe_top_k = moe_top_k
        self.max_loops = max_loops
        self.activation_balance_weight = activation_balance_weight
        self.moe_aux_loss_weight = moe_aux_loss_weight
        self.activation_top_k = activation_top_k
        self.activation_bias_update_speed = float(activation_bias_update_speed)
        self.moe_bias_update_speed = float(moe_bias_update_speed)

        # Non-gradient loss-free-balancing biases. They are updated once per
        # optimizer step by LoopTransformer.update_routing_biases().
        self.register_buffer(
            "activation_bias",
            torch.zeros(self.NUM_ACTIVATIONS),
            persistent=True,
        )
        self.register_buffer(
            "expert_bias",
            torch.zeros(num_routed_experts),
            persistent=True,
        )

        if expert_hidden_dim is None:
            # Preserve roughly the old dense active width when one shared
            # expert and top-1 routed expert are active simultaneously.
            active_experts = num_shared_experts + moe_top_k
            expert_hidden_dim = max(1, hidden_dim // active_experts)
        if expert_hidden_dim < 1:
            raise ValueError("expert_hidden_dim must be >= 1")
        self.expert_hidden_dim = expert_hidden_dim

        self.shared_experts = nn.ModuleList(
            SwiGLUExpert(dim, expert_hidden_dim, self.NUM_ACTIVATIONS)
            for _ in range(num_shared_experts)
        )
        self.routed_experts = nn.ModuleList(
            SwiGLUExpert(dim, expert_hidden_dim, self.NUM_ACTIVATIONS)
            for _ in range(num_routed_experts)
        )

        # DeepSeek-style token router for fine-grained routed experts.
        # FiLM-style per-loop conditioning (see activation_router below for
        # the full reasoning) -- MoE previously had NO loop-awareness at
        # all, unlike activation routing's (weaker) additive-only bias.
        self.expert_router = nn.Linear(dim, num_routed_experts, bias=False)
        self.expert_loop_scale = nn.Parameter(torch.ones(max_loops, num_routed_experts))
        self.expert_loop_bias = nn.Parameter(torch.zeros(max_loops, num_routed_experts))

        # Tiny activation router. FiLM-style per-loop conditioning: the
        # shared router's raw logits are scaled AND shifted per loop_idx,
        # not just shifted. Pure additive bias (raw + loop_bias[loop_idx])
        # only lets a loop shift every token's baseline preference by the
        # same constant vector, independent of the token itself -- it
        # can't express "loop 1 prefers activation A for tokens with
        # property P, but loop 3 prefers B for those same tokens", since
        # that needs the loop index to interact WITH the token's features,
        # not just offset the result after the fact. Scaling raw_logits *
        # loop_scale[loop_idx] first lets each loop amplify or dampen how
        # strongly the (shared) weight matrix's per-activation projection
        # of x drives that logit, before the shift is added -- cheap (just
        # 2*max_loops*NUM_ACTIVATIONS extra scalars, no new weight matrix
        # or embedding-size hyperparameter to pick) and still keeps the
        # underlying feature extraction (the w_i . x projections
        # themselves) shared and sample-efficient across loops, unlike
        # giving every loop a fully separate router weight matrix would.
        # loop_scale inits to 1 and loop_bias to 0, so at step 0 this is
        # exactly activation_router(x) with no loop conditioning at all --
        # matches this codebase's existing zero-init-starts-as-identity
        # convention (see attnres.py).
        self.activation_router = nn.Linear(dim, self.NUM_ACTIVATIONS, bias=False)
        self.loop_scale = nn.Parameter(torch.ones(max_loops, self.NUM_ACTIVATIONS))
        self.loop_bias = nn.Parameter(torch.zeros(max_loops, self.NUM_ACTIVATIONS))

        # Populated during forward for the training objective / debug logging.
        self._activation_aux_losses = []
        self._moe_aux_losses = []
        self._last_activation_probs_mean: Optional[torch.Tensor] = None
        self._last_activation_probs_dense_mean: Optional[torch.Tensor] = None
        self._last_activation_entropy_mean: Optional[torch.Tensor] = None
        self._last_expert_probs_mean: Optional[torch.Tensor] = None
        self._last_expert_load: Optional[torch.Tensor] = None
        self._last_expert_entropy_mean: Optional[torch.Tensor] = None

    @property
    def last_activation_probs_mean(self) -> Optional[torch.Tensor]:
        """Post-sparsify (hard, actually-dispatched) activation usage mean."""
        return self._last_activation_probs_mean

    @property
    def last_activation_probs_dense_mean(self) -> Optional[torch.Tensor]:
        """Pre-sparsify (dense, soft) activation routing mean -- what the
        balance loss trains against; see _route_activations."""
        return self._last_activation_probs_dense_mean

    @property
    def last_activation_entropy_mean(self) -> Optional[torch.Tensor]:
        """Mean per-token entropy of the dense activation distribution."""
        return self._last_activation_entropy_mean

    @property
    def last_expert_probs_mean(self) -> Optional[torch.Tensor]:
        return self._last_expert_probs_mean

    @property
    def last_expert_load(self) -> Optional[torch.Tensor]:
        return self._last_expert_load

    @property
    def last_expert_entropy_mean(self) -> Optional[torch.Tensor]:
        """Mean per-token entropy of the dense MoE router distribution."""
        return self._last_expert_entropy_mean

    def _clear_routing_diagnostics(self) -> None:
        """Reset get_routing_debug()'s snapshot fields. Deliberately separate
        from _clear_aux_losses(): that one clears the loss-accumulation
        lists compute_loss() depends on; this one only clears the debug
        snapshot generate.py reads, so the two never interact."""
        self._last_activation_probs_mean = None
        self._last_activation_probs_dense_mean = None
        self._last_activation_entropy_mean = None
        self._last_expert_probs_mean = None
        self._last_expert_load = None
        self._last_expert_entropy_mean = None

    @staticmethod
    def _entropy(probs: torch.Tensor) -> torch.Tensor:
        """Mean per-row entropy (nats) of a [..., K] probability distribution.
        Used only for debug diagnostics (get_routing_debug), not the loss."""
        eps = safe_eps(probs.dtype)
        row_entropy = -(probs.clamp_min(eps) * probs.clamp_min(eps).log()).sum(dim=-1)
        return row_entropy.mean()

    @staticmethod
    def _activation_balance_loss(probs: torch.Tensor) -> torch.Tensor:
        """KL(q || uniform), where q is the batch/token mean routing mass."""
        q = probs.mean(dim=0)
        uniform = 1.0 / probs.size(-1)
        return torch.sum(q * torch.log(q.clamp_min(safe_eps(q.dtype)) / uniform))

    @staticmethod
    def _moe_load_balance_loss(
        probs: torch.Tensor,
        topk_indices: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Switch-style load balancing diagnostic for sparse top-k routing."""
        n_tokens, n_experts = probs.shape
        topk = topk_indices.size(-1)
        experts = torch.arange(n_experts, device=topk_indices.device)
        one_hot = (topk_indices.unsqueeze(-1) == experts).float().sum(dim=-2)
        load = one_hot.mean(dim=0) / float(topk)
        importance = probs.mean(dim=0)
        loss = n_experts * torch.sum(importance * load)
        return loss, importance, load

    def _clear_aux_losses(self) -> None:
        self._activation_aux_losses.clear()
        self._moe_aux_losses.clear()

    @torch.no_grad()
    def _update_routing_biases(self) -> None:
        """Apply one loss-free load-balancing step to this FFN.

        The update is intentionally outside autograd. Under-use receives a
        positive logit bias and over-use receives a negative bias. The
        hard-usage statistics come from the most recent forward pass.
        """
        if self._last_activation_probs_mean is not None and self.activation_bias_update_speed > 0.0:
            load = self._last_activation_probs_mean.to(
                device=self.activation_bias.device,
                dtype=self.activation_bias.dtype,
            )
            target = torch.full_like(load, 1.0 / load.numel())
            self.activation_bias.add_(
                self.activation_bias_update_speed * torch.sign(target - load)
            )

        if self._last_expert_load is not None and self.moe_bias_update_speed > 0.0:
            load = self._last_expert_load.to(
                device=self.expert_bias.device,
                dtype=self.expert_bias.dtype,
            )
            target = torch.full_like(load, 1.0 / load.numel())
            self.expert_bias.add_(
                self.moe_bias_update_speed * torch.sign(target - load)
            )

    def _route_activations(self, x: torch.Tensor, loop_idx: int) -> torch.Tensor:
        loop_idx = int(max(0, min(loop_idx, self.max_loops - 1)))
        raw_logits = self.activation_router(x)
        logits = (
            raw_logits * self.loop_scale[loop_idx]
            + self.loop_bias[loop_idx]
            + self.activation_bias
        )
        dense_probs = F.softmax(logits, dim=-1)
        dense_flat = dense_probs.reshape(-1, self.NUM_ACTIVATIONS)

        probs = dense_probs
        if self.activation_top_k < self.NUM_ACTIVATIONS:
            topk_vals, topk_idx = torch.topk(dense_probs, self.activation_top_k, dim=-1)
            sparse_probs = torch.zeros_like(dense_probs)
            sparse_probs.scatter_(
                dense_probs.ndim - 1,
                topk_idx,
                topk_vals / topk_vals.sum(dim=-1, keepdim=True).clamp_min(safe_eps(topk_vals.dtype)),
            )
            # Straight-through estimator. Forward value is exactly
            # sparse_probs (the hard, renormalized top-k weights we
            # actually want to compute with); backward gradient is as if
            # `probs = dense_probs` directly. Without this, sparse_probs'
            # nonzero entries are topk_vals/topk_vals.sum(), which at
            # activation_top_k=1 is always exactly 1 -- a constant with
            # zero local gradient. That doesn't just weaken the balance
            # loss (already fixed above): it means the MAIN task loss has
            # zero gradient to activation_router/loop_bias too, since
            # `probs` (not dense_probs) is what SwiGLUExpert.forward()
            # actually multiplies into mixed_gate. Verified empirically:
            # activation_router.weight.grad was exactly 0.0 before this.
            probs = dense_probs + (sparse_probs - dense_probs).detach()

        mean_probs = probs.reshape(-1, self.NUM_ACTIVATIONS).mean(dim=0)
        self._last_activation_probs_mean = mean_probs.detach()
        self._last_activation_probs_dense_mean = dense_flat.mean(dim=0).detach()
        self._last_activation_entropy_mean = self._entropy(dense_flat).detach()
        balance = self._activation_balance_loss(dense_flat)
        self._activation_aux_losses.append(balance)
        return probs

    def _run_sparse_routed(
        self,
        x_flat: torch.Tensor,
        activation_probs_flat: torch.Tensor,
        loop_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        loop_idx = int(max(0, min(loop_idx, self.max_loops - 1)))
        raw_router_logits = self.expert_router(x_flat)
        router_logits = (
            raw_router_logits * self.expert_loop_scale[loop_idx]
            + self.expert_loop_bias[loop_idx]
            + self.expert_bias
        )
        router_probs = F.softmax(router_logits, dim=-1)
        topk_vals, topk_idx = torch.topk(router_probs, self.moe_top_k, dim=-1)
        dense_topk = router_probs.gather(-1, topk_idx)
        topk_vals = topk_vals / topk_vals.sum(dim=-1, keepdim=True).clamp_min(safe_eps(topk_vals.dtype))
        # Same straight-through estimator as _route_activations, and for the
        # same reason: at moe_top_k=1, topk_vals/topk_vals.sum() is always
        # exactly 1 (a constant), so the dispatch weight actually multiplied
        # into each expert's output (`weights` below) carried zero local
        # gradient back to expert_router -- the main task loss could only
        # ever reach expert_router through the separate load-balance loss,
        # never directly. This keeps the forward value (dispatch weight)
        # unchanged while routing a real gradient through dense_topk.
        topk_vals = dense_topk + (topk_vals - dense_topk).detach()

        out = torch.zeros_like(x_flat)
        for expert_idx, expert in enumerate(self.routed_experts):
            # For the small study model, vectorized masked dispatch keeps all
            # inactive expert matrices from being evaluated for each token.
            mask = topk_idx.eq(expert_idx)
            if not mask.any():
                continue
            token_idx, choice_idx = mask.nonzero(as_tuple=True)
            x_sel = x_flat[token_idx]
            a_sel = activation_probs_flat[token_idx]
            expert_out = expert(x_sel, a_sel)
            weights = topk_vals[token_idx, choice_idx].unsqueeze(-1)
            out.index_add_(0, token_idx, expert_out * weights)

        moe_loss, importance, load = self._moe_load_balance_loss(router_probs, topk_idx)
        self._moe_aux_losses.append(moe_loss)
        self._last_expert_probs_mean = importance.detach()
        self._last_expert_load = load.detach()
        self._last_expert_entropy_mean = self._entropy(router_probs).detach()
        return out, importance, load

    def forward(self, x: torch.Tensor, loop_idx: int = 0) -> torch.Tensor:
        original_shape = x.shape
        x_flat = x.reshape(-1, x.size(-1))
        activation_probs = self._route_activations(x, loop_idx)
        activation_probs_flat = activation_probs.reshape(-1, self.NUM_ACTIVATIONS)

        out = torch.zeros_like(x_flat)
        for expert in self.shared_experts:
            out = out + expert(x_flat, activation_probs_flat)

        routed_out, _, _ = self._run_sparse_routed(x_flat, activation_probs_flat, loop_idx)
        out = out + routed_out

        # Keep autograd connectivity to every routed expert even when a
        # particular expert receives zero tokens in this micro-batch. This
        # contributes exactly zero to the forward value and gradient while
        # preserving optimizer/test invariants that expect all parameters to
        # participate in the graph.
        zero = out.new_zeros(())
        for expert in self.routed_experts:
            zero = zero + (
                expert.w1.weight.sum()
                + expert.w2.weight.sum()
                + expert.w3.weight.sum()
            ) * 0.0
        out = out + zero
        return out.reshape(original_shape)

    def collect_aux_loss(self) -> Tuple[torch.Tensor, torch.Tensor]:
        # Auxiliary losses are produced in both training and evaluation.
        # During eval, compute_loss() runs under torch.no_grad(), so these
        # tensors legitimately have requires_grad=False. Filtering them by
        # requires_grad can therefore produce an empty list and make
        # torch.stack([]) fail. Keep valid tensors in both modes.
        params = next(self.parameters())
        device = params.device
        dtype = params.dtype

        activation_losses = list(self._activation_aux_losses)
        if activation_losses:
            a = torch.stack(activation_losses).mean()
        else:
            a = torch.zeros((), device=device, dtype=dtype)

        moe_losses = list(self._moe_aux_losses)
        if moe_losses:
            m = torch.stack(moe_losses).mean()
        else:
            m = torch.zeros((), device=device, dtype=dtype)

        self._clear_aux_losses()
        return a, m
