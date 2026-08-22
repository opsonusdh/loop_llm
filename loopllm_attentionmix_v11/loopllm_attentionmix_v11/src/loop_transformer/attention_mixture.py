from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import safe_eps


class RoutingContextAttention(nn.Module):
    """O(T*d) attention pooling used to make token routers context-aware."""

    def __init__(self, dim: int, context_dim: int | None = None):
        super().__init__()
        context_dim = context_dim or min(64, max(16, dim // 8))
        self.q = nn.Linear(dim, context_dim, bias=False)
        self.k = nn.Linear(dim, context_dim, bias=False)
        self.v = nn.Linear(dim, context_dim, bias=False)
        self.out = nn.Linear(context_dim, context_dim, bias=False)
        nn.init.zeros_(self.out.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.q(x.mean(dim=1, keepdim=True))
        k = self.k(x)
        v = self.v(x)
        scores = torch.matmul(q, k.transpose(1, 2)) / (k.size(-1) ** 0.5)
        attn = F.softmax(scores, dim=-1)
        ctx = torch.matmul(attn, v).expand(-1, x.size(1), -1)
        return self.out(torch.tanh(ctx))


class ContextAttentionRouter(nn.Module):
    """Lightweight content router using one global attention summary per sequence.

    It is intentionally O(T*d) rather than full O(T^2): a learned query attends
    over token states to build a sequence summary, then token-level routing logits
    use both local token content and that summary. Loop embeddings are injected
    multiplicatively so the same hidden state may be routed differently at
    different recurrence depths.
    """

    def __init__(self, dim: int, num_choices: int, max_loops: int, loop_embed_dim: int = 16, loop_specific_router: bool = True):
        super().__init__()
        self.dim = dim
        self.num_choices = num_choices
        self.max_loops = max_loops
        self.loop_embed_dim = loop_embed_dim
        self.loop_specific_router = bool(loop_specific_router)
        ctx_dim = min(64, max(16, dim // 8))
        self.q = nn.Linear(dim, ctx_dim, bias=False)
        self.k = nn.Linear(dim, ctx_dim, bias=False)
        self.v = nn.Linear(dim, ctx_dim, bias=False)
        self.token_proj = nn.Linear(dim, ctx_dim, bias=False)
        self.context_proj = nn.Linear(ctx_dim, ctx_dim, bias=False)
        self.loop_embedding = nn.Embedding(max_loops, loop_embed_dim) if loop_embed_dim > 0 else None
        if self.loop_embedding is not None:
            nn.init.normal_(self.loop_embedding.weight, mean=0.0, std=0.02)
        route_in = ctx_dim * 2 + (loop_embed_dim if loop_embed_dim > 0 else 0)
        if self.loop_specific_router:
            self.router_heads = nn.ModuleList([nn.Linear(route_in, num_choices, bias=True) for _ in range(max_loops)])
            for head in self.router_heads:
                nn.init.normal_(head.weight, mean=0.0, std=0.02)
                nn.init.zeros_(head.bias)
            self.router = None
        else:
            self.router = nn.Linear(route_in, num_choices, bias=True)
            nn.init.normal_(self.router.weight, mean=0.0, std=0.02)
            nn.init.zeros_(self.router.bias)
        # Small non-zero initialization lets loop embeddings influence routing
        # from the first update. Exact-zero weights made every loop start with
        # the same router even though loop embeddings were present.

    def forward(self, x: torch.Tensor, loop_idx: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        # x [B,T,D]
        q_pool = self.q(x.mean(dim=1, keepdim=True))                 # [B,1,C]
        k = self.k(x)                                                # [B,T,C]
        v = self.v(x)                                                # [B,T,C]
        scores = torch.matmul(q_pool, k.transpose(1, 2)) / (k.size(-1) ** 0.5)
        weights = F.softmax(scores, dim=-1)                          # [B,1,T]
        context = torch.matmul(weights, v).expand(-1, x.size(1), -1) # [B,T,C]
        token = self.token_proj(x)
        context = torch.tanh(self.context_proj(context))
        parts = [token, context]
        if self.loop_embedding is not None:
            idx = max(0, min(int(loop_idx), self.max_loops - 1))
            loop = self.loop_embedding.weight[idx].view(1, 1, -1).expand(x.size(0), x.size(1), -1)
            parts.append(loop)
        inp = torch.cat(parts, dim=-1)
        if self.loop_specific_router:
            idx = max(0, min(int(loop_idx), self.max_loops - 1))
            logits = self.router_heads[idx](inp)
        else:
            logits = self.router(inp)
        return logits, weights.squeeze(1)


class AttentionExpertMixture(nn.Module):
    """Token-routed mixture of attention modules.

    All experts are evaluated in the current reference implementation and then
    sparsely mixed with a straight-through top-k router. This keeps the math
    simple and robust while making the routing pattern learnable. `top_k=1` is
    the recommended information-density experiment; experts remain semantically
    distinct while the forward path uses one expert per token.
    """

    def __init__(
        self,
        experts: Sequence[nn.Module],
        dim: int,
        max_loops: int,
        top_k: int = 1,
        balance_weight: float = 0.01,
        loop_embed_dim: int = 16,
        diversity_weight: float = 0.001,
        min_probability: float = 0.0,
        balance_tolerance: float = 0.25,
        preferred_expert: int | None = None,
    ):
        super().__init__()
        if len(experts) < 2:
            raise ValueError("AttentionExpertMixture needs at least two experts")
        if not 1 <= top_k <= len(experts):
            raise ValueError("attention top_k must be in [1, num_experts]")
        self.experts = nn.ModuleList(experts)
        self.num_experts = len(experts)
        self.top_k = top_k
        self.balance_weight = float(balance_weight)
        self.diversity_weight = float(diversity_weight)
        self.min_probability = float(min_probability)
        self.balance_tolerance = float(balance_tolerance)
        if not 0.0 <= self.min_probability < 1.0 / self.num_experts:
            raise ValueError(
                f"attention mixture min_probability must be in [0, {1.0 / self.num_experts:.6f})"
            )
        if self.balance_tolerance < 0.0:
            raise ValueError("attention mixture balance_tolerance must be >= 0")
        self.router = ContextAttentionRouter(dim, self.num_experts, max_loops, loop_embed_dim)
        self.loop_scale = nn.Parameter(torch.ones(max_loops, self.num_experts))
        self.loop_bias = nn.Parameter(torch.zeros(max_loops, self.num_experts))
        if preferred_expert is not None:
            if not 0 <= preferred_expert < self.num_experts:
                raise ValueError("preferred_expert out of range")
            # Optional warm-start prior. The model v7 experiment leaves this
            # unset so all attention families have a genuine chance to specialize.
            with torch.no_grad():
                if self.loop_specific_router:
                    for head in self.router_heads:
                        head.bias[preferred_expert] += 0.1
                else:
                    self.router.bias[preferred_expert] += 0.1
        self._loop_probs: dict[int, torch.Tensor] = {}
        self._loop_probs_train: dict[int, torch.Tensor] = {}
        self._last_probs_by_loop: dict[int, torch.Tensor] = {}
        self._balance_losses: list[torch.Tensor] = []
        self._last_probs_mean: Optional[torch.Tensor] = None
        self._last_hard_load: Optional[torch.Tensor] = None
        self._last_entropy: Optional[torch.Tensor] = None
        self._last_balance_loss: Optional[torch.Tensor] = None

    def _balance_loss(self, probs: torch.Tensor) -> torch.Tensor:
        q = probs.reshape(-1, probs.size(-1)).mean(dim=0)
        u = 1.0 / probs.size(-1)

        # Respect the configured tolerance band.  The previous implementation
        # always applied KL(q || uniform), which meant the advertised
        # ``balance_tolerance`` was not actually a dead zone: useful
        # specialization was still pulled back toward uniform routing.
        # Build a clamped target inside the tolerance band so the KL term only
        # reacts to probability mass outside that band.
        lower = max(0.0, u - self.balance_tolerance)
        upper = min(1.0, u + self.balance_tolerance)
        target = q.clamp(min=lower, max=upper)
        target = target / target.sum().clamp_min(safe_eps(target.dtype))
        outside = torch.abs(q - target)
        kl = torch.sum(
            q * torch.log(
                q.clamp_min(safe_eps(q.dtype))
                / target.clamp_min(safe_eps(target.dtype))
            )
        )
        # The explicit excess term gives a clean gradient toward the nearest
        # edge of the tolerance band, while the clamped-KL term preserves a
        # smooth probabilistic balancing signal outside the band.
        tolerance_penalty = torch.mean(outside * outside)
        return kl + tolerance_penalty

    def forward(
        self,
        x: torch.Tensor,
        loop_idx: int = 0,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        logits, _ = self.router(x, loop_idx)
        idx_loop = max(0, min(int(loop_idx), self.loop_scale.size(0) - 1))
        logits = logits * self.loop_scale[idx_loop] + self.loop_bias[idx_loop]
        raw_probs = F.softmax(logits, dim=-1)
        if getattr(self, "diagnostic_uniform_attention", False):
            raw_probs = torch.full_like(raw_probs, 1.0 / self.num_experts)
        if self.min_probability > 0.0:
            # Probability-floor parameterization: every attention family keeps
            # at least min_probability mass, while the remaining mass preserves
            # the router's learned ranking. For 3 experts and floor=0.10,
            # 30% is reserved for exploration and 70% remains learnable.
            dense = self.min_probability + (1.0 - self.num_experts * self.min_probability) * raw_probs
        else:
            dense = raw_probs
        vals, idx = torch.topk(dense, self.top_k, dim=-1)
        sparse = torch.zeros_like(dense)
        sparse.scatter_(-1, idx, vals / vals.sum(dim=-1, keepdim=True).clamp_min(safe_eps(vals.dtype)))
        probs = dense + (sparse - dense).detach()

        outputs = [expert(x, loop_idx=loop_idx, attention_mask=attention_mask) for expert in self.experts]
        stacked = torch.stack(outputs, dim=-2)  # [B,T,E,D]
        mixed = torch.sum(stacked * probs.unsqueeze(-1), dim=-2)

        self._last_probs_mean = dense.detach().mean(dim=(0, 1))
        hard = F.one_hot(idx, num_classes=self.num_experts).sum(dim=-2).float() / float(self.top_k)
        self._last_hard_load = hard.detach().mean(dim=(0, 1))
        self._last_entropy = -(dense.clamp_min(safe_eps(dense.dtype)) * dense.clamp_min(safe_eps(dense.dtype)).log()).sum(dim=-1).mean().detach()
        self._last_balance_loss = self._balance_loss(dense)
        if self.training:
            self._balance_losses.append(self._last_balance_loss)
        loop_mass = dense.mean(dim=(0, 1))
        # Keep a detached copy for diagnostics and a graph-connected copy for
        # the loop-diversity auxiliary. The previous implementation detached
        # this tensor before the diversity penalty, making that penalty
        # numerically present but gradient-dead.
        self._loop_probs[int(loop_idx)] = loop_mass.detach()
        self._loop_probs_train[int(loop_idx)] = loop_mass
        self._last_probs_by_loop[int(loop_idx)] = loop_mass.detach()
        return mixed

    def routing_debug(self) -> Dict[str, torch.Tensor]:
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        source = self._last_probs_by_loop
        per_loop = torch.stack([source[k] for k in sorted(source)], dim=0) if source else torch.empty(0, self.num_experts, device=device, dtype=dtype)
        return {
            "attention_probs_mean": self._last_probs_mean if self._last_probs_mean is not None else torch.zeros(self.num_experts, device=device, dtype=dtype),
            "attention_load": self._last_hard_load if self._last_hard_load is not None else torch.zeros(self.num_experts, device=device, dtype=dtype),
            "attention_entropy": self._last_entropy if self._last_entropy is not None else torch.zeros((), device=device, dtype=dtype),
            "attention_probs_by_loop": per_loop,
        }

    def collect_attention_aux_loss(self) -> torch.Tensor:
        if self._last_balance_loss is None:
            p = next(self.parameters())
            return torch.zeros((), device=p.device, dtype=p.dtype)
        base_balance = torch.stack(self._balance_losses).mean() if self._balance_losses else self._last_balance_loss
        loss = base_balance * self.balance_weight
        if len(self._loop_probs_train) > 1 and self.diversity_weight > 0:
            vals = [self._loop_probs_train[k] for k in sorted(self._loop_probs_train)]
            diversity = torch.zeros_like(loss)
            pairs = 0
            for i in range(len(vals)):
                for j in range(i + 1, len(vals)):
                    a = vals[i] / vals[i].norm().clamp_min(safe_eps(vals[i].dtype))
                    b = vals[j] / vals[j].norm().clamp_min(safe_eps(vals[j].dtype))
                    # Penalize near-identical routing distributions, but only
                    # weakly. We want useful specialization, not forced chaos.
                    diversity = diversity + F.relu(torch.sum(a * b) - 0.85)
                    pairs += 1
            if pairs:
                loss = loss + self.diversity_weight * diversity / pairs
        self._last_balance_loss = None
        self._balance_losses.clear()
        self._loop_probs.clear()
        self._loop_probs_train.clear()
        return loss
