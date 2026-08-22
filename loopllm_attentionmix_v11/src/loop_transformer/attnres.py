"""Attention Residuals -- the "Amnesia" layer of the architecture.

Kimi Team, arXiv:2603.15031.

Standard residuals: h_l = h_{l-1} + f_{l-1}(h_{l-1})
All prior outputs are collapsed into one running sum -- no selective
access to individual earlier layers.

AttnRes replaces this with:
  h_l = Σ_i α_{i→l} · v_i   where  α = softmax over prior outputs
Each layer can now query directly from any earlier representation.

LoopedAttnRes (below) is Block AttnRes (paper §3.2) adapted so that:
  • "Blocks" = completed loop iterations
  • Slot weights are SHARED across loops (parameter-efficient,
    matches LoopLM's weight-sharing principle -- one pseudo-query
    per sub-layer POSITION, not per sub-layer instance)

Zero-init output projections (SlidingWindowAttention.out_proj,
GroupedOutputProjection.proj2, FeedForward.w3) -- why this differs
from a standard residual stream
==========================================================================
Ohio State (arXiv:2604.07822) zero-inits attention/FFN output
projections so h_{t+1} = h_t + f(h_t) starts as an EXACT identity map
(f(h_t)=0), stabilizing unbounded loop unrolling. We don't have a
single evolving h_t -- we have a growing list of block-sums combined
via attention -- so "identity map on h_t" doesn't translate literally.
What actually happens here, worked through explicitly:

  1. With every sub-layer output now zero at init, every completed
     loop's block-sum (partial in LoopTransformer._one_loop_impl) is
     EXACTLY zero.
  2. DepthAttnRes.query is ALSO zero-init (pre-existing, §5 of the
     AttnRes paper), so every AttnRes mixture is a UNIFORM average
     over its sources.
  3. A convex combination of {embedding, 0, 0, ..., 0} is always a
     non-negative SCALAR MULTIPLE of embedding, regardless of the
     mixing weights or how many zero-blocks have accumulated.
  4. Every consumer of these mixtures (attn_norm, ffn_norm, final_norm,
     DepthAttnRes.key_norm) is RMSNorm, which is scale-invariant.

So at initialization, the WHOLE network -- any depth, any loop count --
reduces to a stable, well-defined function of the embedding alone,
unaffected by how many (zero-valued) blocks have piled up. New
contributions only start influencing the output as gradients move
their output-projection weights away from zero -- which happens
immediately (d(loss)/d(w3) doesn't depend on w3's own value, only on
w3's downstream gradient and its input activations, so the very first
backward pass already updates it; gradients to whatever feeds INTO a
zero-init layer only start flowing from the second step onward, once
that layer's weight is no longer exactly zero. This is the standard,
well-tested mechanism behind ReZero-style zero-gated branches, not a
dead-init trap).

One measured consequence worth knowing: combined with tied embeddings,
this makes the model's very-first-step behavior collapse to "predict
the same token you just saw" (tok_emb dotted with itself dominates).
That's a bad guess on unstructured random data but a reasonable prior
for real text -- and it doesn't slow training; see tests/test_init.py
and tests/test_training.py.
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn

from .layers import RMSNorm


class DepthAttnRes(nn.Module):
    """
    A single sub-layer's depth-attention slot.

    Computes a learned softmax attention over a stack of source tensors:
        phi(q, k) = exp(q^T RMSNorm(k))            [Kimi Eq 2]
        h = softmax_i phi(w, k_i) · v_i             [Kimi Eq 4]

    The pseudo-query w is zero-initialised so that at step 0 all
    sources receive equal weight -- stable warm-up (paper §5).
    """

    def __init__(self, dim: int):
        super().__init__()
        self.key_norm = RMSNorm(dim)
        self.query    = nn.Parameter(torch.zeros(dim))  # zero-init per §5

    def forward(self, sources: torch.Tensor) -> torch.Tensor:
        """sources: [N, B, T, D]  →  [B, T, D]"""
        keys    = self.key_norm(sources)
        logits  = torch.einsum("d,nbtd->nbt", self.query, keys)
        weights = logits.softmax(dim=0)               # softmax over N sources
        return torch.einsum("nbt,nbtd->btd", weights, sources)


class LoopedAttnRes(nn.Module):
    """
    Block Attention Residuals adapted for LoopLM parameter sharing.

    Within each loop, 2·N sub-layers run sequentially (attn + ffn per
    transformer block).  Each position 0..2N-1 has one DepthAttnRes slot
    whose weights are REUSED every loop -- so total new parameters for
    AttnRes = (2·N + 1) DepthAttnRes modules regardless of max_loops.

    Sources available to each sub-layer call:
        blocks  = [b₀=embedding, b₁=loop1_sum, …, b_{t-1}=loop_{t-1}_sum]
        partial = running sum of sub-layer outputs so far in the current loop

    After completing a loop, `partial` is appended to `blocks` as b_t
    and then reset to None for the next loop.  This matches Kimi Eq 5:
        b_n = Σ_{j∈B_n} f_j(h_j)
    """

    def __init__(self, dim: int, n_sublayers_per_loop: int):
        super().__init__()
        # One slot per position within a loop -- shared across iterations
        self.per_sublayer = nn.ModuleList(
            [DepthAttnRes(dim) for _ in range(n_sublayers_per_loop)]
        )
        # Extra slot for the final cross-block output aggregation
        self.final_slot = DepthAttnRes(dim)

    def compute_input(
        self,
        slot_idx: int,
        blocks:   List[torch.Tensor],
        partial:  Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        AttnRes-aggregated input for the sub-layer at position `slot_idx`.
        Attends over completed block sums + current intra-loop partial sum.
        """
        sources = blocks if partial is None else [*blocks, partial]
        return self.per_sublayer[slot_idx](torch.stack(sources, dim=0))

    def compute_output(self, blocks: List[torch.Tensor]) -> torch.Tensor:
        """
        Aggregate all completed block summaries (including the loop just
        finished) for the LM head and exit gate.
        """
        return self.final_slot(torch.stack(blocks, dim=0))
