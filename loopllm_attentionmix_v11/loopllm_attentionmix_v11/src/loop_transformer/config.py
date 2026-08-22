"""LoopConfig -- the single source of truth for model hyperparameters.

Validated in __post_init__ so misconfiguration fails immediately, at
construction time, with a message that says what's wrong and how to fix
it -- rather than surfacing as a bare AssertionError or a shape-mismatch
traceback three modules deep during the first forward pass.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Optional


class LoopConfigError(ValueError):
    """Raised by LoopConfig.__post_init__ for invalid configurations."""


@dataclass
class LoopConfig:
    """
    NOTE on csa_top_k / sw_window: CSAAttention and HCAAttention compute
    genuinely per-query attention (each query gathers its own causally-
    visible top-k blocks + local window, then attends over just those),
    costing O(T · (top_k + window)) per layer -- linear in T, and causal.

    An earlier version of this file flattened every query's picks into
    ONE shared memory pool that every query then attended over densely,
    costing O(T · (T · top_k)) -- quadratic in T, worse than plain full
    attention -- AND had no causal masking at all, letting any position
    see compressed summaries of future tokens. Both are fixed now (see
    attention.py). top_k=64 / window=128 are comfortable defaults;
    there's no longer a strong reason to keep them small purely for memory.
    """
    # ── Model dimensions ───────────────────────────────────────
    vocab_size:     int   = 50_000
    dim:            int   = 4096
    n_layers:       int   = 24       # shared transformer layers per loop
    n_heads:        int   = 32
    head_dim:       int   = 128
    ffn_hidden_dim: int   = 14_336   # ~3.5× dim (standard for 7B class)
    rope_dim:       int   = 64       # must be even and <= head_dim
    embed_dim:      Optional[int] = None  # if set, factorize tok_emb to
                                      # this size instead of dim, bridged
                                      # by a Linear(embed_dim, dim)
                                      # projection (ALBERT-style). Saves
                                      # vocab_size*(dim-embed_dim) params
                                      # minus the dim*embed_dim bridge --
                                      # a real net win once dim is large
                                      # relative to vocab_size. None (the
                                      # default) means no factorization;
                                      # existing configs are unaffected.
    embed_dim_out:  Optional[int] = None  # if set, the pre-lm_head
                                      # projection dimension (bridged by a
                                      # separate Linear(dim, embed_dim_out)),
                                      # independent of embed_dim. Requires
                                      # embed_dim to also be set. Defaults
                                      # to embed_dim (symmetric, classic
                                      # ALBERT) when left None.
                                      #
                                      # Why this is a SEPARATE knob rather
                                      # than always symmetric: Chung et al.,
                                      # "Rethinking Embedding Coupling in
                                      # Pre-trained Language Models" (Google,
                                      # ICLR 2021) found that shrinking BOTH
                                      # input and output embeddings together
                                      # specifically hurts vocab-diverse /
                                      # multilingual models, while a larger
                                      # (or fully unfactored) output
                                      # embedding recovers most of the
                                      # quality loss at modest extra cost.
                                      # If tie_embeddings=True, embed_dim_out
                                      # must equal embed_dim (or be left
                                      # None) -- tying requires matching
                                      # shapes, checked below.
    # ── LoopLM ─────────────────────────────────────────────────
    max_loops:      int   = 4        # R; Ouro trains with 4 recurrent steps
    beta_entropy:   float = 0.05     # β for KL / entropy regularisation
    # Additional direct supervision on every executed loop. 0.0 preserves the
    # original objective exactly; a positive value adds a normalized, linearly
    # increasing weighted CE/KD loss so later loops receive explicit gradient.
    loop_supervision_weight: float = 0.0
    # Weight of Ouro-style learned-exit expected loss during Stage-I. Keep 0
    # to train all loops directly, then train the gate separately in Stage-II.
    joint_exit_loss_weight: float = 0.0
    # Soft penalty discouraging LATER loops from having a higher (batch-mean)
    # loss than earlier ones: ReLU(mean_loss[t+1] - mean_loss[t] + margin),
    # summed over consecutive loop pairs. 0.0 disables it (default). This is
    # deliberately a soft hinge, not a hard constraint -- margin gives a
    # small allowed tolerance so it doesn't fight loop_supervision_weight or
    # ordinary training noise between individual steps.
    loop_monotonic_weight: float = 0.0
    loop_monotonic_margin: float = 0.01
    # New recurrent-refinement experiment knobs.
    # Small auxiliary loss that rewards useful improvement between consecutive loops.
    loop_refinement_weight: float = 0.02
    loop_refinement_margin: float = 0.001
    # Distinct sub-objective per loop: loop t predicts t+1 tokens ahead
    # (loop 0 -> next token, loop 1 -> +2, ..., loop 3 -> +4).
    loop_task_weight: float = 0.02
    loop_task_mode: str = "horizon"
    # Explicit loop identity for the exit gate. 0 disables the embedding,
    # but the refined experiment defaults to 16 dimensions.
    exit_gate_loop_embed_dim: int = 16
    # ── DiffusionBlocks recurrent-depth training ─────────────────
    # This is an optional Stage-I training objective. It follows the paper
    #'s recurrent-depth adaptation: the whole recurrent model is trained as
    # a denoiser with ONE recurrent pass per update; ordinary K-loop inference
    # remains unchanged.
    diffusion_blocks: bool = False
    diffusion_sigma_min: float = 0.002
    diffusion_sigma_max: float = 80.0
    diffusion_p_mean: float = -1.2
    diffusion_p_std: float = 1.2
    diffusion_sigma_data: float = 0.5
    diffusion_loss_weight: float = 1.0
    diffusion_normalize_embeddings: bool = True
    diffusion_cond_dim: int = 128
    # ── Mixture attention / contextual routing ─────────────────
    attention_mixture: bool = True
    attention_mixture_top_k: int = 1
    attention_mixture_balance_weight: float = 0.01
    attention_mixture_loop_embed_dim: int = 16
    attention_mixture_num_experts: int = 3
    attention_mixture_start_layer: int = 4
    attention_mixture_diversity_weight: float = 0.0015
    attention_mixture_min_probability: float = 0.10
    attention_mixture_balance_tolerance: float = 0.25
    # 0 uses the layer-specific default experts; 1 enables SWA/CSA/HCA mixture
    # where architectural candidates are selected by a content router.

    # ── DeepSeek attention ─────────────────────────────────────
    csa_m:          int   = 4
    csa_top_k:      int   = 64       # see class docstring
    csa_aux_loss_weight: float = 1.0  # weight on the CSA indexer's
                                      # auxiliary KL-divergence loss (see
                                      # attention.py's CSAAttention
                                      # docstring). Gradient-isolated by
                                      # construction (detach/no_grad), so
                                      # this doesn't compete with or need
                                      # careful balancing against the main
                                      # LM loss the way a blended multi-
                                      # task weight normally would -- it
                                      # only scales the indexer's own
                                      # effective learning rate. 1.0 is a
                                      # reasonable starting point, not an
                                      # empirically tuned value.
    hca_m_prime:    int   = 128
    sw_window:      int   = 128
    groups:         int   = 8
    group_dim:      int   = 1024     # NOTE: flat default, NOT scaled to dim.
                                      # GroupedOutputProjection cost is
                                      # group_dim * dim * (1 + groups); at
                                      # small dim, a large flat group_dim can
                                      # dominate total params. Scale this down
                                      # (e.g. ~dim//2) for smaller models.
    # ── Memory-efficiency knobs ─────────────────────────────────
    tie_embeddings:    bool = True   # share tok_emb/lm_head weights -- saves
                                      # vocab_size*dim params for free (e.g.
                                      # 38M params at vocab=50k, dim=768).
                                      # Standard practice; no known downside
                                      # for this use case.
    grad_checkpointing: bool = False # recompute each loop's activations
                                      # during backward instead of storing
                                      # them -- trades compute for ~max_loops-
                                      # fold less peak activation memory.
                                      # This is literally what the AttnRes
                                      # paper calls "activation recomputation"
                                      # and says is standard at scale (§3.1).
                                      # Only takes effect in model.train()
                                      # mode; forward() in eval mode always
                                      # runs without checkpointing.
    # ── Dynamic training-time loop count ────────────────────────
    min_loops:      int   = 1        # floor for sampled loop counts
    loop_sampling:  bool  = True     # sample R ~ clip(Poisson(λ), min_loops,
                                      # max_loops) fresh each compute_loss()
                                      # call in training mode, instead of
                                      # always training at a fixed max_loops.
                                      # Ohio State (arXiv:2604.07822) and
                                      # Parcae (arXiv:2604.12946) independently
                                      # find this increases learnable
                                      # recursion depth, improves
                                      # extrapolation beyond trained depth,
                                      # and slows the "overthinking" quality
                                      # decay from over-looping -- two
                                      # unrelated papers converging on the
                                      # same fix. Only affects compute_loss();
                                      # forward()/generate() are unaffected,
                                      # and eval-mode compute_loss() calls
                                      # always use the fixed max_loops
                                      # ceiling, so validation loss stays a
                                      # comparable, fixed-depth metric.
    # Recurrent-depth stability and depth-mixing.
    recurrent_depth_controller: bool = True
    recurrent_depth_bottleneck_dim: int = 128
    recurrent_update_init: float = 0.95
    shortcut_consistency_weight: float = 0.0
    shortcut_consistency_temperature: float = 2.0
    # Unified trainer mode: recurrent, diffusion, or memory-safe hybrid.
    training_mode: str = "recurrent"
    hybrid_diffusion_probability: float = 0.25
    loop_sampling_lambda: Optional[float] = None  # Poisson rate; defaults
                                      # to max_loops/2 if left None. Not a
                                      # value taken from either paper --
                                      # neither specifies one -- just a
                                      # reasonable default that centers the
                                      # distribution inside [min_loops,
                                      # max_loops]. Tune if you want training
                                      # to skew toward shallower/deeper loops.
    
    
    # ── DeepSeek-style MoE + activation routing ────────────────
    moe_num_shared_experts: int = 1
    moe_num_routed_experts: int = 4
    moe_top_k: int = 2
    moe_expert_hidden_dim: Optional[int] = None

    activation_balance_weight: float = 0.01
    activation_top_k: int = 2
    activation_min_probability: float = 0.10
    activation_balance_tolerance: float = 0.25
    moe_aux_loss_weight: float = 0.01
    # Loss-free-balancing-style (arXiv:2408.15664) selection bias: a small,
    # non-gradient nudge added to router logits after each optimizer step
    # (see LoopTransformer.update_routing_biases()), pushing under-used
    # experts/activations up and over-used ones down, independent of the
    # auxiliary balance losses above. 0.001 is a gentle default; only takes
    # effect if update_routing_biases() is actually called in the training
    # loop -- otherwise both bias buffers stay at their zero-init forever.
    activation_bias_update_speed: float = 0.001
    moe_bias_update_speed: float = 0.001

    def __post_init__(self) -> None:
        errors = []

        def check(cond: bool, msg: str) -> None:
            if not cond:
                errors.append(msg)

        check(self.vocab_size > 0, f"vocab_size must be > 0, got {self.vocab_size}")
        check(self.dim > 0, f"dim must be > 0, got {self.dim}")
        check(self.n_layers > 0, f"n_layers must be > 0, got {self.n_layers}")
        check(self.n_heads > 0, f"n_heads must be > 0, got {self.n_heads}")
        check(self.head_dim > 0, f"head_dim must be > 0, got {self.head_dim}")
        check(self.ffn_hidden_dim > 0, f"ffn_hidden_dim must be > 0, got {self.ffn_hidden_dim}")

        check(
            self.rope_dim % 2 == 0 and self.rope_dim <= self.head_dim,
            f"rope_dim must be even and <= head_dim; got rope_dim={self.rope_dim}, "
            f"head_dim={self.head_dim}. Either raise head_dim to >= {self.rope_dim} "
            f"or lower rope_dim (must stay even).",
        )

        if self.embed_dim is not None:
            check(self.embed_dim > 0, f"embed_dim must be > 0 if set, got {self.embed_dim}")
            if self.embed_dim >= self.dim:
                warnings.warn(
                    f"embed_dim ({self.embed_dim}) >= dim ({self.dim}) doesn't reduce "
                    f"parameters -- factorization adds a projection with no savings. "
                    f"Set embed_dim below dim, or leave it None.",
                    stacklevel=2,
                )
        check(
            self.embed_dim_out is None or self.embed_dim is not None,
            "embed_dim_out is set but embed_dim is None -- embed_dim_out only "
            "makes sense alongside a factorized input embedding. Set embed_dim too, "
            "or leave embed_dim_out unset.",
        )
        if self.embed_dim_out is not None:
            check(self.embed_dim_out > 0, f"embed_dim_out must be > 0 if set, got {self.embed_dim_out}")

        effective_embed_dim = self.embed_dim if self.embed_dim is not None else self.dim
        effective_embed_dim_out = (
            self.embed_dim_out if self.embed_dim_out is not None else effective_embed_dim
        )
        check(
            not self.tie_embeddings or effective_embed_dim_out == effective_embed_dim,
            f"tie_embeddings=True requires the input and output embedding dimensions "
            f"to match, but they're {effective_embed_dim} (input) vs "
            f"{effective_embed_dim_out} (output) -- tying two different-shaped "
            f"matrices isn't possible. Either set tie_embeddings=False, or make "
            f"embed_dim_out equal embed_dim (or leave it unset, which defaults to "
            f"matching embed_dim).",
        )

        check(self.max_loops >= 1, f"max_loops must be >= 1, got {self.max_loops}")
        check(self.min_loops >= 1, f"min_loops must be >= 1, got {self.min_loops}")
        check(
            self.min_loops <= self.max_loops,
            f"min_loops ({self.min_loops}) must be <= max_loops ({self.max_loops})",
        )
        check(self.beta_entropy >= 0, f"beta_entropy must be >= 0, got {self.beta_entropy}")
        check(self.loop_supervision_weight >= 0, f"loop_supervision_weight must be >= 0, got {self.loop_supervision_weight}")
        check(self.joint_exit_loss_weight >= 0, f"joint_exit_loss_weight must be >= 0, got {self.joint_exit_loss_weight}")
        check(self.loop_monotonic_weight >= 0, f"loop_monotonic_weight must be >= 0, got {self.loop_monotonic_weight}")
        check(self.loop_monotonic_margin >= 0, f"loop_monotonic_margin must be >= 0, got {self.loop_monotonic_margin}")
        check(self.loop_refinement_weight >= 0, f"loop_refinement_weight must be >= 0, got {self.loop_refinement_weight}")
        check(self.loop_refinement_margin >= 0, f"loop_refinement_margin must be >= 0, got {self.loop_refinement_margin}")
        check(self.loop_task_weight >= 0, f"loop_task_weight must be >= 0, got {self.loop_task_weight}")
        check(self.loop_task_mode in {"horizon", "none"}, f"loop_task_mode must be 'horizon' or 'none', got {self.loop_task_mode!r}")
        check(self.exit_gate_loop_embed_dim >= 0, f"exit_gate_loop_embed_dim must be >= 0, got {self.exit_gate_loop_embed_dim}")
        check(0.0 < self.diffusion_sigma_min < self.diffusion_sigma_max,
              f"require 0 < diffusion_sigma_min < diffusion_sigma_max, got {self.diffusion_sigma_min}, {self.diffusion_sigma_max}")
        check(self.diffusion_p_std > 0, f"diffusion_p_std must be > 0, got {self.diffusion_p_std}")
        check(self.diffusion_sigma_data > 0, f"diffusion_sigma_data must be > 0, got {self.diffusion_sigma_data}")
        check(self.diffusion_loss_weight >= 0, f"diffusion_loss_weight must be >= 0, got {self.diffusion_loss_weight}")
        check(self.diffusion_cond_dim > 0, f"diffusion_cond_dim must be > 0, got {self.diffusion_cond_dim}")
        check(not self.diffusion_blocks or self.max_loops >= 1, "diffusion_blocks requires max_loops >= 1")
        check(self.training_mode == "recurrent" or self.diffusion_blocks, "training_mode diffusion/hybrid requires diffusion_blocks=True")
        check(0.0 <= self.activation_min_probability < 0.25, f"activation_min_probability must be in [0, 0.25), got {self.activation_min_probability}")
        check(self.activation_balance_tolerance >= 0, f"activation_balance_tolerance must be >= 0, got {self.activation_balance_tolerance}")
        check(self.activation_top_k >= 1, f"activation_top_k must be >= 1, got {self.activation_top_k}")
        check(self.activation_top_k <= 4, f"activation_top_k must be <= 4, got {self.activation_top_k}")
        check(self.activation_bias_update_speed >= 0, f"activation_bias_update_speed must be >= 0, got {self.activation_bias_update_speed}")
        check(self.moe_num_shared_experts >= 0, f"moe_num_shared_experts must be >= 0, got {self.moe_num_shared_experts}")
        check(self.moe_num_routed_experts >= 1, f"moe_num_routed_experts must be >= 1, got {self.moe_num_routed_experts}")
        check(self.moe_top_k >= 1, f"moe_top_k must be >= 1, got {self.moe_top_k}")
        check(self.moe_top_k <= self.moe_num_routed_experts, f"moe_top_k ({self.moe_top_k}) must be <= moe_num_routed_experts ({self.moe_num_routed_experts})")
        check(self.moe_bias_update_speed >= 0, f"moe_bias_update_speed must be >= 0, got {self.moe_bias_update_speed}")

        check(self.attention_mixture_top_k >= 1, f"attention_mixture_top_k must be >= 1, got {self.attention_mixture_top_k}")
        check(2 <= self.attention_mixture_num_experts <= 3, f"attention_mixture_num_experts must be between 2 and 3, got {self.attention_mixture_num_experts}")
        check(self.attention_mixture_top_k <= self.attention_mixture_num_experts, f"attention_mixture_top_k ({self.attention_mixture_top_k}) must be <= attention_mixture_num_experts ({self.attention_mixture_num_experts})")
        check(self.attention_mixture_start_layer >= 0, f"attention_mixture_start_layer must be >= 0, got {self.attention_mixture_start_layer}")
        check(self.attention_mixture_balance_weight >= 0, f"attention_mixture_balance_weight must be >= 0, got {self.attention_mixture_balance_weight}")
        check(self.attention_mixture_diversity_weight >= 0, f"attention_mixture_diversity_weight must be >= 0, got {self.attention_mixture_diversity_weight}")
        check(0 <= self.attention_mixture_min_probability < 1.0 / self.attention_mixture_num_experts, f"attention_mixture_min_probability must be in [0, 1/num_experts), got {self.attention_mixture_min_probability}")
        check(self.attention_mixture_balance_tolerance >= 0, f"attention_mixture_balance_tolerance must be >= 0, got {self.attention_mixture_balance_tolerance}")
        check(
            self.loop_sampling_lambda is None or self.loop_sampling_lambda > 0,
            f"loop_sampling_lambda must be > 0 if set, got {self.loop_sampling_lambda}",
        )

        check(self.csa_m >= 1, f"csa_m must be >= 1, got {self.csa_m}")
        check(self.csa_top_k >= 1, f"csa_top_k must be >= 1, got {self.csa_top_k}")
        check(self.csa_aux_loss_weight >= 0, f"csa_aux_loss_weight must be >= 0, got {self.csa_aux_loss_weight}")
        check(self.hca_m_prime >= 1, f"hca_m_prime must be >= 1, got {self.hca_m_prime}")
        check(
            self.sw_window >= 1,
            f"sw_window must be >= 1, got {self.sw_window}. CSA/HCA's causal "
            f"local-window branch requires at least 1 (it always includes the "
            f"query's own position); 0 is only meaningful for plain "
            f"SlidingWindowAttention, which this config doesn't isolate.",
        )
        check(self.groups >= 1, f"groups must be >= 1, got {self.groups}")
        check(self.group_dim > 0, f"group_dim must be > 0, got {self.group_dim}")

        if self.groups > 0:  # avoid ZeroDivisionError below if groups<=0 already failed above
            dim_in = self.n_heads * self.head_dim
            check(
                dim_in % self.groups == 0,
                f"n_heads*head_dim ({self.n_heads}*{self.head_dim}={dim_in}) must be "
                f"divisible by groups ({self.groups}) -- this feeds "
                f"GroupedOutputProjection inside every CSA/HCA layer (i.e. every "
                f"layer past index 1). Adjust n_heads, head_dim, or groups.",
            )

        check(self.recurrent_depth_bottleneck_dim > 0, f"recurrent_depth_bottleneck_dim must be > 0, got {self.recurrent_depth_bottleneck_dim}")
        check(0.0 < self.recurrent_update_init <= 1.0, f"recurrent_update_init must be in (0,1], got {self.recurrent_update_init}")
        check(self.shortcut_consistency_weight >= 0, f"shortcut_consistency_weight must be >= 0, got {self.shortcut_consistency_weight}")
        check(self.shortcut_consistency_temperature > 0, "shortcut_consistency_temperature must be > 0")
        check(self.training_mode in {"recurrent", "diffusion", "hybrid"}, f"training_mode must be recurrent/diffusion/hybrid, got {self.training_mode!r}")
        check(0.0 <= self.hybrid_diffusion_probability <= 1.0, "hybrid_diffusion_probability must be in [0,1]")

        if errors:
            bullets = "\n".join(f"  - {e}" for e in errors)
            raise LoopConfigError(f"Invalid LoopConfig ({len(errors)} problem(s)):\n{bullets}")

