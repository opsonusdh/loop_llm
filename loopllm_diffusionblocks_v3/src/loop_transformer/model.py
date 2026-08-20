"""LoopTransformer -- the unified model.

Three-paper architecture:

Paper 1 -- Memory   (DeepSeek):
    SWA / CSA / HCA mechanisms handle long-range context within
    the token sequence at each call to forward_attn().

Paper 2 -- Amnesia  (Kimi AttnRes):
    LoopedAttnRes replaces every standard x + f(x) residual.
    Before each sub-layer runs, its input is computed as a learned
    softmax mixture over every prior sub-layer's output, across all
    loops already completed.  This gives any layer direct read
    access to any earlier representation ("depth-wise attention").

Paper 3 -- Reasoning (Ouro LoopLM):
    The N-block stack is looped R times with SHARED weights.
    A per-loop exit gate models a discrete distribution over
    halting steps; training minimises an entropy-regularised
    expected loss (Ouro Eq 4) that prevents collapse to always
    running R loops.
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint

from .attention import CSAAttention
import math
from .attnres import LoopedAttnRes
from .block import ExitGate, TransformerBlock
from .config import LoopConfig
from .layers import RMSNorm, safe_eps
from .diffusion import edm_preconditioning, log_sigma_embedding, sample_log_normal_sigma


class LoopTransformer(nn.Module):
    def __init__(self, cfg: LoopConfig):
        super().__init__()
        self.cfg = cfg

        # ── Embedding factorization (optional) ──────────────────────
        # cfg.embed_dim / cfg.embed_dim_out let the input and output
        # embeddings each be smaller than cfg.dim, with a Linear
        # projection bridging the gap -- see LoopConfig's docstring for
        # the parameter-count math and why input/output are decoupled
        # rather than forced to match (Chung et al., "Rethinking
        # Embedding Coupling in Pre-trained Language Models": naive
        # symmetric ALBERT-style shrinkage of both sides specifically
        # hurts vocab-diverse/multilingual models). Both default to
        # cfg.dim (no factorization, no projection) when left unset --
        # existing configs are completely unaffected.
        embed_dim     = cfg.embed_dim if cfg.embed_dim is not None else cfg.dim
        embed_dim_out = cfg.embed_dim_out if cfg.embed_dim_out is not None else embed_dim

        self.tok_emb = nn.Embedding(cfg.vocab_size, embed_dim)
        # nn.Embedding's PyTorch default init is std=1 -- reasonable for a
        # standalone lookup table, but wrong once this matrix is also used
        # as a Linear projection weight (tie_embeddings=True, below):
        # summing `embed_dim` terms of std~1 each gives logits std~
        # sqrt(embed_dim), not O(1). Standard GPT-2/nanoGPT-style std=0.02
        # keeps logits well-scaled regardless of tying or factorization.
        nn.init.normal_(self.tok_emb.weight, mean=0.0, std=0.02)

        self.embed_proj_in = (
            nn.Linear(embed_dim, cfg.dim, bias=False) if embed_dim != cfg.dim else None
        )

        # N transformer blocks -- same weights used each loop (LoopLM)
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=cfg.dim, n_heads=cfg.n_heads, head_dim=cfg.head_dim,
                layer_idx=i, ffn_hidden_dim=cfg.ffn_hidden_dim,
                csa_m=cfg.csa_m, csa_top_k=cfg.csa_top_k,
                hca_m_prime=cfg.hca_m_prime, sw_window=cfg.sw_window,
                groups=cfg.groups, group_dim=cfg.group_dim, rope_dim=cfg.rope_dim,
                moe_num_shared_experts=cfg.moe_num_shared_experts,
                moe_num_routed_experts=cfg.moe_num_routed_experts,
                moe_top_k=cfg.moe_top_k,
                moe_expert_hidden_dim=cfg.moe_expert_hidden_dim,
                max_loops=cfg.max_loops,
                diffusion_cond_dim=(cfg.diffusion_cond_dim if cfg.diffusion_blocks else 0),
                activation_balance_weight=cfg.activation_balance_weight,
                moe_aux_loss_weight=cfg.moe_aux_loss_weight,
                activation_top_k=cfg.activation_top_k,
            )
            for i in range(cfg.n_layers)
        ])

        # LoopedAttnRes: 2·n_layers slots, shared across loop iterations
        self.depth_attn = LoopedAttnRes(cfg.dim, n_sublayers_per_loop=2 * cfg.n_layers)

        # Exit gate + LM head -- shared weights across loop steps (LoopLM §4.1)
        self.exit_gate  = ExitGate(cfg.dim, loop_embed_dim=cfg.exit_gate_loop_embed_dim, max_loops=cfg.max_loops)
        self.final_norm = RMSNorm(cfg.dim)

        if cfg.diffusion_blocks:
            self.diffusion_time_mlp = nn.Sequential(
                nn.Linear(cfg.diffusion_cond_dim, cfg.diffusion_cond_dim),
                nn.SiLU(),
                nn.Linear(cfg.diffusion_cond_dim, cfg.diffusion_cond_dim),
            )
        else:
            self.diffusion_time_mlp = None

        self.embed_proj_out = (
            nn.Linear(cfg.dim, embed_dim_out, bias=False) if embed_dim_out != cfg.dim else None
        )
        self.lm_head = nn.Linear(embed_dim_out, cfg.vocab_size, bias=False)

        if cfg.tie_embeddings:
            # nn.Embedding.weight is [vocab, embed_dim]; nn.Linear(embed_dim_out,
            # vocab).weight is [vocab, embed_dim_out] -- only the same shape
            # (hence tie-able) when embed_dim_out == embed_dim, which
            # LoopConfig's validation already guarantees by the time we get
            # here whenever tie_embeddings=True (see config.py).
            self.lm_head.weight = self.tok_emb.weight

        # Set by compute_loss() after each call, for inspection/logging --
        # e.g. to confirm the CSA indexer is actually learning over the
        # course of training. Kept as a plain attribute rather than a third
        # return value from compute_loss() so the existing (total_loss,
        # step_losses) call signature stays exactly backward compatible.
        self.last_csa_aux_loss: Optional[torch.Tensor] = None
        self.last_activation_balance_loss: Optional[torch.Tensor] = None
        self.last_moe_aux_loss: Optional[torch.Tensor] = None
        self.last_activation_probs: Optional[torch.Tensor] = None
        self.last_moe_load: Optional[torch.Tensor] = None
        self.last_loop_supervision_loss: Optional[torch.Tensor] = None
        self.last_loop_refinement_loss: Optional[torch.Tensor] = None
        self.last_loop_task_loss: Optional[torch.Tensor] = None
        self.last_loop_monotonic_loss: Optional[torch.Tensor] = None

    def num_parameters(self, effective: bool = False) -> int:
        """
        Physical parameter count (what's actually stored) by default.

        effective=True instead reports the parameter count an equivalent
        NON-weight-shared model would need to match this model's per-token
        compute depth: physical_params + (max_loops - 1) * shared_block_params,
        since self.blocks is reused max_loops times at runtime. This is the
        comparison papers like Ouro report ("a 1.4B LoopLM with 4 loops
        matches a 4B standard transformer") -- a compute/depth comparison,
        not a storage comparison.
        """
        total = sum(p.numel() for p in self.parameters())
        if not effective:
            return total
        shared = sum(p.numel() for p in self.blocks.parameters())
        return total + (self.cfg.max_loops - 1) * shared

    # ------------------------------------------------------------------
    # Internal: one full loop pass
    # ------------------------------------------------------------------

    def _one_loop_impl(self, loop_idx: int, diffusion_cond: Optional[torch.Tensor], initial_partial: Optional[torch.Tensor], *blocks_tuple: torch.Tensor) -> torch.Tensor:
        """
        Checkpointable core of one loop pass. Identical logic to running
        through all N blocks once with LoopedAttnRes managing residuals,
        but takes `blocks` unpacked as individual tensor args since
        torch.utils.checkpoint.checkpoint needs tensor positional args
        (not an arbitrary Python list) to know what to save vs. recompute.

        For each sub-layer:
            1. Compute input via AttnRes over completed blocks + partial sum.
            2. Run the sub-layer to get the delta.
            3. Add delta to the intra-loop partial sum.

        Returns the completed partial sum b_t = Σ f_j(h_j) for this loop.
        """
        blocks = list(blocks_tuple)
        partial: Optional[torch.Tensor] = initial_partial

        for i, block in enumerate(self.blocks):
            pos_attn = 2 * i        # slot index for the attention sub-layer
            pos_ffn  = 2 * i + 1   # slot index for the FFN sub-layer

            # ── Attention sub-layer ──────────────────────────────────
            h_in     = self.depth_attn.compute_input(pos_attn, blocks, partial)
            attn_out = block.forward_attn(h_in, loop_idx=loop_idx, diffusion_cond=diffusion_cond)
            partial  = attn_out if partial is None else partial + attn_out

            # ── FFN sub-layer ────────────────────────────────────────
            # Note: `partial` now contains the attention output, so the
            # FFN input implicitly sees it through the AttnRes mixture.
            h_in    = self.depth_attn.compute_input(pos_ffn, blocks, partial)
            ffn_out = block.forward_ffn(h_in, loop_idx=loop_idx, diffusion_cond=diffusion_cond)
            partial = partial + ffn_out

        return partial  # type: ignore[return-value]

    def _one_loop(
        self,
        blocks: List[torch.Tensor],
        loop_idx: int = 0,
        *,
        diffusion_cond: Optional[torch.Tensor] = None,
        initial_partial: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run exactly one shared loop, optionally with diffusion conditioning."""
        if self.cfg.grad_checkpointing and self.training and diffusion_cond is None and initial_partial is None:
            # Preserve the original checkpointing path for ordinary LoopLM.
            return grad_checkpoint(self._one_loop_impl, loop_idx, None, None, *blocks, use_reentrant=False)
        return self._one_loop_impl(loop_idx, diffusion_cond, initial_partial, *blocks)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        idx:       torch.Tensor,
        max_loops: Optional[int] = None,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Full looped forward pass.

        Returns
        -------
        loop_logits : list of [B, T, V] tensors, one per loop step
        lambdas     : list of [B] tensors -- instantaneous exit probs λ_t
        """
        max_loops = max_loops if max_loops is not None else self.cfg.max_loops
        self._clear_ffn_aux_losses()
        embedding = self.tok_emb(idx)
        if self.embed_proj_in is not None:
            embedding = self.embed_proj_in(embedding)

        # b₀ = token embedding (Kimi paper: v₀ = h₁)
        blocks: List[torch.Tensor] = [embedding]

        loop_logits: List[torch.Tensor] = []
        lambdas:     List[torch.Tensor] = []

        for loop_idx in range(max_loops):
            # ── One loop pass through all N blocks ───────────────────
            partial = self._one_loop(blocks, loop_idx=loop_idx)

            # Completed loop → becomes new block-level source b_t
            blocks.append(partial)

            # ── Output for this loop ─────────────────────────────────
            # Aggregate across all block-level sources (AttnRes final slot)
            h_out    = self.depth_attn.compute_output(blocks)
            h_normed = self.final_norm(h_out)

            pre_logits = self.embed_proj_out(h_normed) if self.embed_proj_out is not None else h_normed
            loop_logits.append(self.lm_head(pre_logits))
            lambdas.append(self.exit_gate(h_normed, loop_idx=loop_idx))

        return loop_logits, lambdas

    def _diffusion_denoise_once(
        self,
        idx: torch.Tensor,
        z_sigma: torch.Tensor,
        sigma: torch.Tensor,
    ) -> torch.Tensor:
        """One D_theta(z_sigma, x, sigma) call with exactly one recurrent pass."""
        if not self.cfg.diffusion_blocks:
            raise RuntimeError("DiffusionBlocks conditioning is disabled")
        B = idx.size(0)
        embedding = self.tok_emb(idx)
        if self.embed_proj_in is not None:
            embedding = self.embed_proj_in(embedding)
        clean = F.normalize(embedding, dim=-1) if self.cfg.diffusion_normalize_embeddings else embedding
        edm = edm_preconditioning(sigma, sigma_data=self.cfg.diffusion_sigma_data)
        z_in = edm.c_in[:, None, None] * z_sigma
        raw_time = log_sigma_embedding(edm.c_noise, self.cfg.diffusion_cond_dim)
        assert self.diffusion_time_mlp is not None
        time_cond = self.diffusion_time_mlp(raw_time)
        self._clear_csa_aux_losses()
        self._clear_ffn_aux_losses()
        delta = self._one_loop(
            [clean],
            loop_idx=0,
            diffusion_cond=time_cond,
            initial_partial=z_in,
        )
        pred = edm.c_skip[:, None, None] * z_sigma + edm.c_out[:, None, None] * self.final_norm(delta)
        if self.cfg.diffusion_normalize_embeddings:
            pred = F.normalize(pred, dim=-1)
        return pred

    def diffusion_blocks_loss(
        self,
        idx: torch.Tensor,
        sigma_override: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Single-pass recurrent-depth DiffusionBlocks training objective.

        This is the recurrent-depth adaptation from DiffusionBlocks rather than
        the separate 4-block AR Transformer recipe.  The input sequence x is
        the clean current-token context, while the target sequence y is the
        next-token embedding sequence.  The recurrent network receives a noisy
        target latent z_sigma and clean x, plus continuous noise-level
        conditioning, then performs exactly ONE recurrent pass.
        """
        if not self.cfg.diffusion_blocks:
            raise RuntimeError("diffusion_blocks_loss requires cfg.diffusion_blocks=True")
        if idx.ndim != 2 or idx.size(1) < 2:
            raise ValueError("idx must have shape [B,T] with T >= 2")

        B = idx.size(0)
        context_ids = idx[:, :-1].contiguous()
        target_ids = idx[:, 1:].contiguous()
        target_emb = self.tok_emb(target_ids)
        if self.embed_proj_in is not None:
            target_emb = self.embed_proj_in(target_emb)
        if self.cfg.diffusion_normalize_embeddings:
            target_emb = F.normalize(target_emb, dim=-1)

        if sigma_override is None:
            sigma = sample_log_normal_sigma(
                B,
                device=idx.device,
                p_mean=self.cfg.diffusion_p_mean,
                p_std=self.cfg.diffusion_p_std,
                sigma_min=self.cfg.diffusion_sigma_min,
                sigma_max=self.cfg.diffusion_sigma_max,
            )
        else:
            if sigma_override.ndim == 0:
                sigma = sigma_override.expand(B)
            elif sigma_override.shape == (B,):
                sigma = sigma_override
            else:
                raise ValueError(f"sigma_override must be scalar or [B], got {tuple(sigma_override.shape)}")
            sigma = sigma.to(device=idx.device, dtype=torch.float32).clone().clamp_(
                min=self.cfg.diffusion_sigma_min, max=self.cfg.diffusion_sigma_max
            )

        edm = edm_preconditioning(sigma, sigma_data=self.cfg.diffusion_sigma_data)
        z_sigma = target_emb + sigma[:, None, None] * torch.randn_like(target_emb)
        pred = self._diffusion_denoise_once(context_ids, z_sigma, sigma)
        pre_logits = self.embed_proj_out(pred) if self.embed_proj_out is not None else pred
        logits = self.lm_head(pre_logits)
        ce = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            target_ids.reshape(-1),
            reduction="none",
        ).view(B, -1).mean(dim=1)
        weighted_ce = (edm.weight * ce).mean()

        csa_aux = self._collect_csa_aux_loss()
        self.last_csa_aux_loss = csa_aux.detach()
        activation_aux, moe_aux = self._collect_ffn_aux_losses()
        total = self.cfg.diffusion_loss_weight * weighted_ce
        total = total + self.cfg.csa_aux_loss_weight * csa_aux
        total = total + self.cfg.activation_balance_weight * activation_aux
        total = total + self.cfg.moe_aux_loss_weight * moe_aux

        info = {
            "loss": total.detach(),
            "ce": ce.mean().detach(),
            "weighted_ce": weighted_ce.detach(),
            "mean_sigma": sigma.mean().detach(),
            "min_sigma": sigma.min().detach(),
            "max_sigma": sigma.max().detach(),
            "csa_aux": csa_aux.detach(),
            "activation_aux": activation_aux.detach(),
            "moe_aux": moe_aux.detach(),
        }
        return total, info

    @torch.no_grad()
    def diffusion_euler_sample(
        self,
        idx: torch.Tensor,
        *,
        num_steps: int = 4,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        """Generate next tokens using the recurrent DiffusionBlocks trajectory."""
        if not self.cfg.diffusion_blocks:
            raise RuntimeError("diffusion_euler_sample requires cfg.diffusion_blocks=True")
        if num_steps < 1:
            raise ValueError("num_steps must be >= 1")
        if seed is not None:
            torch.manual_seed(seed)
        was_training = self.training
        self.eval()
        B, T = idx.shape
        embed = self.tok_emb(idx)
        if self.embed_proj_in is not None:
            embed = self.embed_proj_in(embed)
        if self.cfg.diffusion_normalize_embeddings:
            embed = F.normalize(embed, dim=-1)
        z = torch.randn_like(embed) * self.cfg.diffusion_sigma_max
        sigmas = torch.linspace(
            self.cfg.diffusion_sigma_max,
            self.cfg.diffusion_sigma_min,
            num_steps + 1,
            device=idx.device,
            dtype=z.dtype,
        )
        final_logits = None
        for i in range(num_steps):
            sigma = torch.full((B,), float(sigmas[i].item()), device=idx.device, dtype=torch.float32)
            z0 = self._diffusion_denoise_once(idx, z, sigma)
            next_sigma = sigmas[i + 1]
            step = (next_sigma - sigmas[i]) / sigmas[i].clamp_min(torch.finfo(z.dtype).tiny)
            z = z + step * (z - z0)
            final_logits = self.lm_head(z0 if i == num_steps - 1 else z)
        logits = final_logits[:, -1, :] / max(temperature, 1e-6)
        if top_k is not None:
            values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits = torch.where(
                logits < values[:, [-1]],
                torch.full_like(logits, float("-inf")),
                logits,
            )
        next_id = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
        if was_training:
            self.train()
        return torch.cat([idx, next_id], dim=1)

    @torch.no_grad()
    def diffusion_blocks_eval(
        self, idx: torch.Tensor, sigma_values: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """Deterministic validation metrics across a fixed sigma grid.

        The weighted EDM objective is intentionally retained for training, but
        its raw value is highly sigma-sensitive. Validation therefore reports
        both unweighted token CE and the weighted objective averaged over a
        fixed log-spaced sigma grid, making progress interpretable.
        """
        was_training = self.training
        self.eval()
        if sigma_values is None:
            sigma_values = torch.logspace(
                math.log10(self.cfg.diffusion_sigma_min),
                math.log10(self.cfg.diffusion_sigma_max),
                steps=6, device=idx.device, dtype=torch.float32,
            )
        ces = []
        weighted = []
        for sigma in sigma_values:
            loss, info = self.diffusion_blocks_loss(idx, sigma_override=sigma)
            ces.append(info["ce"].float())
            weighted.append(loss.float())
        result = {
            "ce": torch.stack(ces).mean().item(),
            "weighted_loss": torch.stack(weighted).mean().item(),
        }
        if was_training:
            self.train()
        return result

    # ------------------------------------------------------------------
    # Training objective  (Ouro §3.3, Eq 4)
    # ------------------------------------------------------------------

    def _sample_loop_count(self) -> int:
        """
        Sample a training-time loop count R ~ clip(Poisson(λ), min_loops,
        max_loops), used by compute_loss() when it isn't given an explicit
        max_loops. Only samples in training mode -- eval/inference always
        get the fixed cfg.max_loops ceiling (mirrors how grad_checkpointing
        similarly checks self.training elsewhere in this file), so
        validation loss stays a comparable, fixed-depth metric across runs
        rather than fluctuating with whatever depth happened to get drawn.

        Rationale: see LoopConfig.loop_sampling.
        """
        if not self.cfg.loop_sampling or not self.training:
            return self.cfg.max_loops
        lam = self.cfg.loop_sampling_lambda
        if lam is None:
            lam = self.cfg.max_loops / 2.0
        r = torch.poisson(torch.tensor(float(lam))).item()
        return int(max(self.cfg.min_loops, min(self.cfg.max_loops, round(r))))

    def _clear_csa_aux_losses(self) -> None:
        """Defensive reset before a forward pass: if something else (a
        direct model.forward() call, an interrupted previous compute_loss)
        left entries sitting in a CSAAttention instance's _aux_losses list,
        don't let them leak into this call's collection."""
        for module in self.modules():
            if isinstance(module, CSAAttention):
                module._aux_losses.clear()

    def _collect_csa_aux_loss(self) -> torch.Tensor:
        """
        Collect this call's indexer auxiliary KL-divergence loss (see
        CSAAttention._indexer_aux_loss) across every CSA layer, and clear
        each instance's list afterward.

        Filters to entries with requires_grad=True before combining: under
        gradient checkpointing, CSAAttention.forward() runs TWICE per loop
        (once in a throwaway no_grad dry-run internal to
        torch.utils.checkpoint, once again with grad enabled during
        backward recomputation) -- naively averaging both would silently
        double-count this loss whenever checkpointing is active, since the
        no_grad entry is a real (if unusable) tensor, not absent. Verified
        this interaction empirically while building it, not just reasoned
        through -- see tests/test_indexer_aux_loss.py.

        Returns a zero tensor (not None) when there's nothing to collect
        (e.g. no CSA layers at this depth, or eval mode), so callers can
        add it to the total loss unconditionally.
        """
        losses: List[torch.Tensor] = []
        for module in self.modules():
            if isinstance(module, CSAAttention):
                losses.extend(t for t in module._aux_losses if t.requires_grad)
                module._aux_losses.clear()
        if not losses:
            return torch.zeros((), device=next(self.parameters()).device)
        return torch.stack(losses).mean()

    def _clear_ffn_aux_losses(self) -> None:
        for module in self.modules():
            if hasattr(module, "_clear_aux_losses"):
                module._clear_aux_losses()

    def _collect_ffn_aux_losses(self) -> Tuple[torch.Tensor, torch.Tensor]:
        activation_losses = []
        moe_losses = []
        activation_probs = []
        moe_loads = []
        for module in self.modules():
            if hasattr(module, "collect_aux_loss"):
                a, m = module.collect_aux_loss()
                if a.requires_grad:
                    activation_losses.append(a)
                if m.requires_grad:
                    moe_losses.append(m)
            if hasattr(module, "last_activation_probs_mean"):
                p = module.last_activation_probs_mean
                if p is not None:
                    activation_probs.append(p)
            if hasattr(module, "last_expert_load"):
                l = module.last_expert_load
                if l is not None:
                    moe_loads.append(l)
        device = next(self.parameters()).device
        a = torch.stack(activation_losses).mean() if activation_losses else torch.zeros((), device=device)
        m = torch.stack(moe_losses).mean() if moe_losses else torch.zeros((), device=device)
        self.last_activation_balance_loss = a.detach()
        self.last_moe_aux_loss = m.detach()
        self.last_activation_probs = torch.stack(activation_probs).mean(dim=0) if activation_probs else None
        self.last_moe_load = torch.stack(moe_loads).mean(dim=0) if moe_loads else None
        return a, m

    def get_routing_debug(self) -> Dict[str, Dict[str, object]]:
        """
        Snapshot of routing diagnostics from the most recent forward() call,
        averaged across every FeedForward layer -- what scripts/generate.py's
        --debug flag prints per generation step.

        Deliberately independent of _collect_ffn_aux_losses()/compute_loss():
        this only READS each FeedForward's last_* properties (already
        populated as a side effect of forward() running _route_activations/
        _run_sparse_routed) and never clears anything, so it's safe to call
        after a plain forward() or generate() call -- no interaction with
        training's aux-loss bookkeeping.

        Returns {} if no FeedForward layer has run a forward pass yet
        (matches generate.py's `if not info: return`).
        """
        act_probs, act_hard, act_entropy = [], [], []
        moe_probs, moe_hard, moe_entropy = [], [], []

        for module in self.modules():
            if hasattr(module, "last_activation_probs_dense_mean"):
                p = module.last_activation_probs_dense_mean
                if p is not None:
                    act_probs.append(p)
            if hasattr(module, "last_activation_probs_mean"):
                p = module.last_activation_probs_mean
                if p is not None:
                    act_hard.append(p)
            if hasattr(module, "last_activation_entropy_mean"):
                e = module.last_activation_entropy_mean
                if e is not None:
                    act_entropy.append(e)
            if hasattr(module, "last_expert_probs_mean"):
                p = module.last_expert_probs_mean
                if p is not None:
                    moe_probs.append(p)
            if hasattr(module, "last_expert_load"):
                l = module.last_expert_load
                if l is not None:
                    moe_hard.append(l)
            if hasattr(module, "last_expert_entropy_mean"):
                e = module.last_expert_entropy_mean
                if e is not None:
                    moe_entropy.append(e)

        info: Dict[str, Dict[str, object]] = {}
        if act_probs or act_hard or act_entropy:
            info["activation"] = {
                "top_k": self.cfg.activation_top_k,
                "mean_probs": torch.stack(act_probs).mean(dim=0).tolist() if act_probs else None,
                "hard_usage": torch.stack(act_hard).mean(dim=0).tolist() if act_hard else None,
                "mean_entropy": torch.stack(act_entropy).mean().item() if act_entropy else None,
            }
        if moe_probs or moe_hard or moe_entropy:
            info["moe"] = {
                "top_k": self.cfg.moe_top_k,
                "mean_probs": torch.stack(moe_probs).mean(dim=0).tolist() if moe_probs else None,
                "hard_usage": torch.stack(moe_hard).mean(dim=0).tolist() if moe_hard else None,
                "mean_entropy": torch.stack(moe_entropy).mean().item() if moe_entropy else None,
            }
        return info

    def clear_routing_debug(self) -> None:
        """Reset get_routing_debug()'s per-layer snapshot fields so a stale
        forward pass's numbers can't leak into the next one -- e.g.
        scripts/generate.py calls this before each new generation step.
        Mirrors _clear_ffn_aux_losses()'s hasattr-dispatch pattern, but
        targets the debug-only fields, not the aux-loss lists -- keeps this
        fully independent of training's loss bookkeeping."""
        for module in self.modules():
            if hasattr(module, "_clear_routing_diagnostics"):
                module._clear_routing_diagnostics()

    def compute_loss(
        self,
        idx:       torch.Tensor,
        max_loops: Optional[int] = None,
        beta:      Optional[float] = None,
        teacher:             Optional["LoopTransformer"] = None,
        distill_alpha:       float = 0.5,
        distill_temperature: float = 2.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Entropy-regularised expected loss (Ouro Eq 4), computed PER-EXAMPLE
        and then averaged over the batch -- p_φ(t|x) is explicitly a function
        of x (Ouro Eq 3), so different sequences in a batch can prefer
        different exit points; collapsing to a single batch-wide scalar
        before combining would throw that away.

            L = Σ_t  p_φ(t|x) · L^(t)(x)  −  β · H(p_φ(·|x))

        Exit distribution via survival model (Ouro Eq 3), per example:
            p̃_t(x) = λ_t(x) · ∏_{j<t}(1 − λ_j(x))
            p_T(x)  = ∏_{j<T}(1 − λ_j(x))         (remaining mass → last step)

        The entropy term prevents the gate from collapsing to always
        exiting at step T_max (the self-reinforcement failure mode
        described in Ouro §3.3).

        If max_loops isn't given explicitly and self.training is True and
        cfg.loop_sampling is on, the loop count for THIS call is sampled
        via _sample_loop_count() rather than fixed at cfg.max_loops --
        so step_losses' length below varies call to call in that regime;
        it always reflects however many loops actually ran this time.

        Also adds cfg.csa_aux_loss_weight * (CSA indexer's KL-divergence
        auxiliary loss), gradient-isolated so it only trains the indexer
        submodules -- see CSAAttention._indexer_aux_loss. Its value is
        stashed on self.last_csa_aux_loss afterward for inspection/logging
        (e.g. to confirm the indexer is actually learning over training),
        without changing this method's return signature.

        Knowledge distillation (optional): pass `teacher`, another
        LoopTransformer instance (any size/config, same vocab_size), to
        blend the standard cross-entropy loss with a temperature-scaled
        KL-divergence toward the teacher's output distribution (Hinton
        et al.'s classic soft-label distillation, still the standard
        production approach as of 2026 -- 5-30x inference cost reduction
        typically retaining ~95-97% of teacher quality is the commonly
        reported range, not a guarantee for any specific setup).

            L^(t)_combined = alpha * L^(t)_CE  +  (1-alpha) * T^2 * KL(teacher || student_t)

        then the SAME p_exit weighting and entropy regularization above
        apply to the combined per-step loss. The teacher runs once under
        torch.no_grad() at its OWN cfg.max_loops (its trained ceiling,
        which may differ from the student's), using its final loop's
        output as the fixed target distribution; the teacher's own
        training-mode flag is saved and restored around the call, so this
        doesn't have side effects on a teacher you're still training
        elsewhere.

        Worth knowing before reaching for a much larger teacher: Distillation
        Scaling Laws (Busbridge et al., 2025) found a real capacity-gap
        effect -- too large a teacher/student ratio can hurt transfer
        rather than help it, with optimal teacher scale tracking student
        scale roughly linearly rather than "bigger is strictly better."
        This implementation is classical (fixed-corpus) distillation, not
        the newer on-policy variant (student generates its own rollouts,
        teacher supervises those) that 2026 literature reports further
        gains from -- a reasonable next step, not implemented here.

        Returns
        -------
        total_loss  : scalar tensor (differentiable)
        step_losses : [T] tensor of per-loop LM losses, batch-averaged
                      (detached, for logging only); T = loops run this call.
                      When teacher is given, these are the COMBINED
                      (CE + KD) per-step losses, not CE alone.
        """
        if teacher is not None:
            if teacher.cfg.vocab_size != self.cfg.vocab_size:
                raise ValueError(
                    f"teacher.cfg.vocab_size ({teacher.cfg.vocab_size}) != "
                    f"self.cfg.vocab_size ({self.cfg.vocab_size}) -- logit-level "
                    f"distillation requires a shared vocabulary/tokenizer."
                )
            if not (0.0 <= distill_alpha <= 1.0):
                raise ValueError(f"distill_alpha must be in [0, 1], got {distill_alpha}")
            if distill_temperature <= 0:
                raise ValueError(f"distill_temperature must be > 0, got {distill_temperature}")

        max_loops = max_loops if max_loops is not None else self._sample_loop_count()
        beta      = self.cfg.beta_entropy if beta is None else beta

        self._clear_csa_aux_losses()
        loop_logits, lambdas = self.forward(idx, max_loops)
        B = idx.size(0)

        # ── Per-example, per-loop cross-entropy L^(t)(x) ─────────────
        # reduction="none" + reshape keeps the batch dimension alive so
        # each example can be weighted by its own p_φ(t|x) below.
        step_losses_per_example = []   # will hold B-vectors, one per loop
        for logits in loop_logits:
            lgt = logits[:, :-1].contiguous()      # [B, T-1, V]
            tgt = idx[:, 1:].contiguous()          # [B, T-1]
            ce = F.cross_entropy(
                lgt.view(-1, lgt.size(-1)), tgt.view(-1), reduction="none",
            ).view(B, -1).mean(dim=1)              # [B] -- mean over positions
            step_losses_per_example.append(ce)
        step_losses_per_example = torch.stack(step_losses_per_example, dim=1)  # [B, T]
        primary_step_losses = step_losses_per_example

        # Distinct task per loop, without replacing the primary LM objective.
        # Loop t predicts t+1 tokens ahead (1-, 2-, 3-, 4-token horizons).
        # This makes the auxiliary objective genuinely depth-specific while
        # every loop still receives the normal next-token CE signal.
        loop_task_loss = torch.zeros((), device=idx.device, dtype=primary_step_losses.dtype)
        if self.cfg.loop_task_weight > 0 and self.cfg.loop_task_mode == "horizon":
            task_losses = []
            for t, logits in enumerate(loop_logits):
                horizon = t + 1
                if idx.size(1) <= horizon:
                    continue
                pred = logits[:, :-(horizon)].contiguous()
                tgt = idx[:, horizon:].contiguous()
                task_losses.append(F.cross_entropy(
                    pred.reshape(-1, pred.size(-1)), tgt.reshape(-1), reduction="mean"
                ))
            if task_losses:
                w = torch.arange(1, len(task_losses) + 1, device=idx.device, dtype=primary_step_losses.dtype)
                w = w / w.sum() * len(task_losses)
                loop_task_loss = torch.stack(task_losses).mul(w).mean()
        self.last_loop_task_loss = loop_task_loss.detach()

        if teacher is not None:
            teacher_was_training = teacher.training
            teacher.eval()
            with torch.no_grad():
                teacher_loop_logits, _ = teacher.forward(idx, max_loops=teacher.cfg.max_loops)
                teacher_logits = teacher_loop_logits[-1][:, :-1].detach()  # [B, T-1, V]
                teacher_probs = F.softmax(teacher_logits / distill_temperature, dim=-1)
            if teacher_was_training:
                teacher.train()

            kd_step_losses = []
            for logits in loop_logits:
                student_log_probs = F.log_softmax(
                    logits[:, :-1].contiguous() / distill_temperature, dim=-1,
                )  # [B, T-1, V]
                kd = F.kl_div(student_log_probs, teacher_probs, reduction="none").sum(dim=-1)
                kd_step_losses.append(kd.mean(dim=1) * (distill_temperature ** 2))  # [B]
            kd_step_losses = torch.stack(kd_step_losses, dim=1)  # [B, T]

            step_losses_per_example = (
                distill_alpha * step_losses_per_example
                + (1.0 - distill_alpha) * kd_step_losses
            )

        # ── Per-example exit distribution p_φ(t|x) via survival model ──
        p_exit = []
        survival = torch.ones_like(lambdas[0])     # [B]
        for t in range(max_loops):
            lam_t = lambdas[t]                     # [B]
            if t < max_loops - 1:
                p_t = lam_t * survival
                survival = survival * (1.0 - lam_t)
            else:
                p_t = survival                     # absorb remaining mass
            p_exit.append(p_t)
        p_exit = torch.stack(p_exit, dim=1)         # [B, T], rows sum to ~1

        # ── Objective, averaged over the batch ─────────────────────────
        expected_loss = (p_exit * step_losses_per_example).sum(dim=1).mean()
        entropy       = -(p_exit * p_exit.clamp(min=safe_eps(p_exit.dtype)).log()).sum(dim=1).mean()
        total_loss    = expected_loss - beta * entropy

        # Optional direct supervision for every executed loop. The weights
        # increase linearly with depth and are normalized to sum to one, so
        # this behaves as an auxiliary objective rather than changing the
        # loss scale merely because max_loops is larger. The default weight
        # is 0.0, preserving the original training objective exactly.
        if self.cfg.loop_supervision_weight > 0.0:
            n_loops = step_losses_per_example.size(1)
            loop_weights = torch.arange(
                1, n_loops + 1,
                device=step_losses_per_example.device,
                dtype=step_losses_per_example.dtype,
            )
            loop_weights = loop_weights / loop_weights.sum()
            loop_supervision_loss = (
                step_losses_per_example * loop_weights.unsqueeze(0)
            ).sum(dim=1).mean()
        else:
            loop_supervision_loss = torch.zeros(
                (),
                device=step_losses_per_example.device,
                dtype=step_losses_per_example.dtype,
            )
        self.last_loop_supervision_loss = loop_supervision_loss.detach()
        total_loss = total_loss + self.cfg.loop_supervision_weight * loop_supervision_loss

        csa_aux_loss = self._collect_csa_aux_loss()
        self.last_csa_aux_loss = csa_aux_loss.detach()
        total_loss = total_loss + self.cfg.csa_aux_loss_weight * csa_aux_loss

        activation_balance_loss, moe_aux_loss = self._collect_ffn_aux_losses()
        total_loss = total_loss + self.cfg.activation_balance_weight * activation_balance_loss
        total_loss = total_loss + self.cfg.moe_aux_loss_weight * moe_aux_loss

        # Per-example refinement: the next loop is rewarded only when it beats
        # the previous loop by at least a small margin. ReLU keeps this a soft
        # auxiliary signal instead of forcing impossible monotonicity on every
        # stochastic batch.
        if self.cfg.loop_refinement_weight > 0 and primary_step_losses.size(1) > 1:
            improvements = primary_step_losses[:, :-1] - primary_step_losses[:, 1:]
            refinement_loss = F.relu(self.cfg.loop_refinement_margin - improvements).mean()
        else:
            refinement_loss = torch.zeros((), device=idx.device, dtype=primary_step_losses.dtype)
        self.last_loop_refinement_loss = refinement_loss.detach()
        total_loss = total_loss + self.cfg.loop_refinement_weight * refinement_loss

        # Legacy batch-level monotonic penalty is kept separate and disabled
        # by default. It remains useful for controlled ablations.
        if self.cfg.loop_monotonic_weight > 0 and primary_step_losses.size(1) > 1:
            mean_losses = primary_step_losses.mean(dim=0)
            monotonic_loss = F.relu(
                mean_losses[1:] - mean_losses[:-1] + self.cfg.loop_monotonic_margin
            ).mean()
        else:
            monotonic_loss = torch.zeros((), device=idx.device, dtype=primary_step_losses.dtype)
        self.last_loop_monotonic_loss = monotonic_loss.detach()
        total_loss = total_loss + self.cfg.loop_monotonic_weight * monotonic_loss

        total_loss = total_loss + self.cfg.loop_task_weight * loop_task_loss

        return total_loss, primary_step_losses.mean(dim=0).detach()  # [T]

    # ------------------------------------------------------------------
    # Inference with adaptive early exit  (Ouro §3.2, Q-exit criterion)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        idx:            torch.Tensor,
        max_new_tokens: int,
        temperature:    float = 1.0,
        top_k:          Optional[int] = None,
        n_loops:        Optional[int] = None,
        exit_threshold: float = 0.8,
    ) -> torch.Tensor:
        """
        Autoregressive generation.

        n_loops=None   → adaptive exit: stop at first loop where CDF ≥ threshold
                          (inherently bounded by cfg.max_loops -- the Q-exit
                          loop below can only stop EARLIER than run_loops,
                          never later, so this path never extrapolates on
                          its own)
        n_loops=k      → always run exactly k loops (fixed depth). If
                          k > cfg.max_loops, this explicitly extrapolates
                          past the trained depth -- see warning below.

        Depth-extrapolation warning: A Mechanistic Analysis of Looped
        Reasoning Language Models (arXiv:2604.11791) specifically found
        that Ouro-style looped models show "structural drift" rather than
        clean convergence when run past their trained loop count, unlike
        some other looped architectures that stayed stable under the same
        test. Our LoopLM design follows Ouro's recipe, so this risk isn't
        ruled out for this model either. We warn rather than block --
        n_loops > max_loops may be exactly what you're deliberately
        testing -- but treat results at that depth as unverified.
        """
        if n_loops is not None and n_loops > self.cfg.max_loops:
            warnings.warn(
                f"generate(n_loops={n_loops}) exceeds cfg.max_loops="
                f"{self.cfg.max_loops} (the trained ceiling). Ouro-style "
                f"looped models have specifically been found to show "
                f"structural drift rather than clean convergence when "
                f"extrapolated past their trained loop count (arXiv:"
                f"2604.11791) -- results at this depth are unverified for "
                f"this model. Proceeding anyway since this may be "
                f"deliberate.",
                stacklevel=2,
            )
        run_loops = n_loops if n_loops is not None else self.cfg.max_loops

        for _ in range(max_new_tokens):
            loop_logits, lambdas = self.forward(idx, run_loops)

            # Select which loop's output to use
            if n_loops is not None:
                chosen = loop_logits[-1]
            else:
                # Q-exit criterion (Ouro §3.2)
                chosen   = loop_logits[-1]
                survival = 1.0
                cdf      = 0.0
                for t, (lgts, lam) in enumerate(zip(loop_logits, lambdas)):
                    p_t  = lam.mean().item() * survival
                    cdf += p_t
                    if t < run_loops - 1:
                        survival *= 1.0 - lam.mean().item()
                    if cdf >= exit_threshold:
                        chosen = lgts
                        break

            logits = chosen[:, -1, :] / max(temperature, 1e-6)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits = torch.where(logits < v[:, [-1]],
                                     torch.full_like(logits, float("-inf")), logits)
            next_id = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)

        return idx
