"""TransformerBlock (attention + FFN, external residuals) and ExitGate
(Ouro LoopLM's per-loop halting predictor).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .attention import CSAAttention, HCAAttention, SlidingWindowAttention
from .attention_mixture import AttentionExpertMixture
from .feedforward import FeedForward
from .layers import RMSNorm


class TransformerBlock(nn.Module):
    """
    Single transformer block: attention + FFN, no internal residuals.

    Residual connections are handled EXTERNALLY by LoopedAttnRes (see
    attnres.py), so forward_attn and forward_ffn return the raw delta
    f(x), not x + f(x).

    Attention type by layer index:
        idx < 2       → SlidingWindowAttention  (warm-up layers)
        idx >= 2, even → CSAAttention            (compressed sparse)
        idx >= 2, odd  → HCAAttention            (heavily compressed)
    """

    def __init__(
        self,
        dim:            int,
        n_heads:        int,
        head_dim:       int,
        layer_idx:      int,
        ffn_hidden_dim: int,
        csa_m:          int   = 4,
        csa_top_k:      int   = 64,
        hca_m_prime:    int   = 128,
        sw_window:      int   = 128,
        groups:         int   = 8,
        group_dim:      int   = 1024,
        rope_dim:       int   = 64,
        moe_num_shared_experts: int = 1,
        moe_num_routed_experts: int = 4,
        moe_top_k: int = 1,
        moe_expert_hidden_dim: int | None = None,
        max_loops: int = 4,
        diffusion_cond_dim: int = 0,
        activation_balance_weight: float = 0.01,
        moe_aux_loss_weight: float = 0.01,
        activation_top_k: int = 4,
        activation_min_probability: float = 0.10,
        activation_balance_tolerance: float = 0.25,
        activation_bias_update_speed: float = 0.001,
        moe_bias_update_speed: float = 0.001,
        attention_mixture: bool = False,
        attention_mixture_top_k: int = 1,
        attention_mixture_balance_weight: float = 0.01,
        attention_mixture_loop_embed_dim: int = 16,
        attention_mixture_num_experts: int = 3,
        attention_mixture_start_layer: int = 2,
        attention_mixture_diversity_weight: float = 0.001,
        attention_mixture_min_probability: float = 0.10,
        attention_mixture_balance_tolerance: float = 0.25,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.attn_norm = RMSNorm(dim)
        self.ffn_norm  = RMSNorm(dim)
        # Standard Transformer residual-branch dropout. Because residuals
        # are owned by LoopedAttnRes, dropout is applied to the raw sub-layer
        # delta immediately before the caller adds that delta.
        self.dropout = nn.Dropout(dropout)

        base_swa = SlidingWindowAttention(dim, n_heads, head_dim,
                                           window_size=sw_window, rope_dim=rope_dim,
                                           max_loops=max_loops)
        base_csa = CSAAttention(
            dim=dim, n_heads=n_heads, head_dim=head_dim,
            m=csa_m, top_k=csa_top_k, window_size=sw_window,
            index_heads=min(32, n_heads), groups=groups, group_dim=group_dim,
            rope_dim=rope_dim, max_loops=max_loops,
        )
        base_hca = HCAAttention(
            dim=dim, n_heads=n_heads, head_dim=head_dim,
            m_prime=hca_m_prime, window_size=sw_window,
            groups=groups, group_dim=group_dim, rope_dim=rope_dim,
            max_loops=max_loops,
        )

        if attention_mixture and layer_idx >= attention_mixture_start_layer:
            # The deep mixture deliberately includes three genuinely different
            # information fields that already exist in the model:
            #   0: local/sliding-window attention (SWA)
            #   1: compressed-sparse attention (CSA)
            #   2: heavily-compressed attention (HCA)
            # The base modules are already constructed for every block, so this
            # adds no third copy of the parameters. The router can now choose
            # local, sparse-long-range, or highly-compressed information per
            # token and per recurrence loop.
            if layer_idx % 2 == 0:
                full_experts = [base_csa, base_hca, base_swa]
            else:
                full_experts = [base_hca, base_csa, base_swa]
            n_experts = max(2, min(attention_mixture_num_experts, len(full_experts)))
            mixture_experts = full_experts[:n_experts]
            self.attn = AttentionExpertMixture(
                mixture_experts, dim, max_loops,
                top_k=attention_mixture_top_k,
                balance_weight=attention_mixture_balance_weight,
                loop_embed_dim=attention_mixture_loop_embed_dim,
                diversity_weight=attention_mixture_diversity_weight,
                min_probability=attention_mixture_min_probability,
                balance_tolerance=attention_mixture_balance_tolerance,
                preferred_expert=None,
            )
        elif layer_idx < 2:
            self.attn = base_swa
        elif layer_idx % 2 == 0:
            self.attn = base_csa
        else:
            self.attn = base_hca

        self.ffn = FeedForward(
            dim, ffn_hidden_dim,
            num_shared_experts=moe_num_shared_experts,
            num_routed_experts=moe_num_routed_experts,
            moe_top_k=moe_top_k,
            expert_hidden_dim=moe_expert_hidden_dim,
            max_loops=max_loops,
            activation_balance_weight=activation_balance_weight,
            moe_aux_loss_weight=moe_aux_loss_weight,
            activation_top_k=activation_top_k,
            activation_min_probability=activation_min_probability,
            activation_balance_tolerance=activation_balance_tolerance,
            activation_bias_update_speed=activation_bias_update_speed,
            moe_bias_update_speed=moe_bias_update_speed,
        )

        # Per-loop FiLM conditioning on the (normed) sub-layer input, applied
        # to BOTH attention and FFN -- so the same shared weights can behave
        # differently at each loop depth, not just at each layer. Previously
        # attention had NO loop identity at all (forward_attn took no
        # loop_idx), while FFN only had it inside the activation/expert
        # ROUTERS (conditioning a routing decision, not what the FFN itself
        # receives). This is a different, complementary level: it shapes the
        # INPUT every sub-layer sees, before any routing or attention math
        # runs at all.
        #
        # Parametrized as (1 + delta) with delta and bias BOTH zero-init, not
        # scale starting at 1 directly: at delta=0 this is an exact identity
        # (safe to warm-start from an existing checkpoint with zero
        # disruption), and -- with weight decay active -- decay pulling delta
        # toward 0 REINFORCES the identity property instead of eroding it.
        # A directly-initialized-to-1 scale would instead get pulled toward
        # 0 by decay, silently shrinking the sub-layer's effective input
        # over training. (The existing activation_router/expert_router loop
        # FiLM in feedforward.py has this same erosion issue; not fixed here
        # to keep this change scoped to the attention-conditioning gap.)
        self.attn_loop_scale = nn.Parameter(torch.zeros(max_loops, dim))
        self.attn_loop_bias = nn.Parameter(torch.zeros(max_loops, dim))
        self.ffn_loop_scale = nn.Parameter(torch.zeros(max_loops, dim))
        self.ffn_loop_bias = nn.Parameter(torch.zeros(max_loops, dim))

        # Optional continuous noise-level conditioning used by DiffusionBlocks.
        # It is absent in normal LoopLM mode, so ordinary checkpoints remain
        # structurally unchanged unless diffusion_blocks=True.
        if diffusion_cond_dim > 0:
            self.diff_attn_mod = nn.Linear(diffusion_cond_dim, 2 * dim)
            self.diff_ffn_mod = nn.Linear(diffusion_cond_dim, 2 * dim)
            nn.init.zeros_(self.diff_attn_mod.weight)
            nn.init.zeros_(self.diff_attn_mod.bias)
            nn.init.zeros_(self.diff_ffn_mod.weight)
            nn.init.zeros_(self.diff_ffn_mod.bias)
        else:
            self.diff_attn_mod = None
            self.diff_ffn_mod = None

    def _film(
        self,
        h: torch.Tensor,
        loop_idx: int,
        scale: torch.Tensor,
        bias: torch.Tensor,
        diffusion_cond: torch.Tensor | None = None,
        diffusion_mod: nn.Linear | None = None,
    ) -> torch.Tensor:
        loop_idx = int(max(0, min(loop_idx, scale.size(0) - 1)))
        h = h * (1.0 + scale[loop_idx]) + bias[loop_idx]
        if diffusion_cond is not None and diffusion_mod is not None:
            gamma, beta = diffusion_mod(diffusion_cond).chunk(2, dim=-1)
            h = h * (1.0 + gamma[:, None, :]) + beta[:, None, :]
        return h

    def forward_attn(
        self, x: torch.Tensor, loop_idx: int = 0, diffusion_cond: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Attention sub-layer delta: f_attn(x).  No residual."""
        h = self._film(
            self.attn_norm(x), loop_idx, self.attn_loop_scale, self.attn_loop_bias,
            diffusion_cond, self.diff_attn_mod,
        )
        return self.dropout(self.attn(h, loop_idx=loop_idx, attention_mask=attention_mask))

    def collect_attention_aux(self) -> torch.Tensor:
        if isinstance(self.attn, AttentionExpertMixture):
            return self.attn.collect_attention_aux_loss()
        return torch.zeros((), device=next(self.parameters()).device, dtype=next(self.parameters()).dtype)

    def attention_routing_debug(self) -> dict[str, torch.Tensor]:
        if isinstance(self.attn, AttentionExpertMixture):
            return self.attn.routing_debug()
        return {}

    def forward_ffn(
        self, x: torch.Tensor, loop_idx: int = 0, diffusion_cond: torch.Tensor | None = None
    ) -> torch.Tensor:
        """FFN sub-layer delta: f_ffn(x).  No residual."""
        h = self._film(
            self.ffn_norm(x), loop_idx, self.ffn_loop_scale, self.ffn_loop_bias,
            diffusion_cond, self.diff_ffn_mod,
        )
        return self.dropout(self.ffn(h, loop_idx=loop_idx))


class ExitGate(nn.Module):
    """Per-loop instantaneous exit probability with explicit loop identity."""

    def __init__(self, dim: int, loop_embed_dim: int = 16, max_loops: int = 4):
        super().__init__()
        self.dim = dim
        self.loop_embed_dim = loop_embed_dim
        self.max_loops = max_loops
        if loop_embed_dim > 0:
            self.loop_embedding = nn.Embedding(max_loops, loop_embed_dim)
            nn.init.normal_(self.loop_embedding.weight, mean=0.0, std=0.02)
            in_dim = dim + loop_embed_dim
        else:
            self.loop_embedding = None
            in_dim = dim
        self.proj = nn.Linear(in_dim, 1, bias=True)

    def forward(self, h: torch.Tensor, loop_idx: int = 0) -> torch.Tensor:
        """h: [B,T,D] -> lambda_t [B]; loop_idx is 0-based."""
        pooled = h.mean(dim=1)
        if self.loop_embedding is not None:
            idx = max(0, min(int(loop_idx), self.max_loops - 1))
            loop_vec = self.loop_embedding.weight[idx].unsqueeze(0).expand(pooled.size(0), -1)
            pooled = torch.cat([pooled, loop_vec], dim=-1)
        return torch.sigmoid(self.proj(pooled)).squeeze(-1)

