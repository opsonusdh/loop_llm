"""Attention mechanisms -- the "Memory" layer of the architecture.

SlidingWindowAttention, CSAAttention, and HCAAttention give each layer
access to the token sequence's context window, following the DeepSeek-V4
study architecture. CSA and HCA compute genuinely per-query, causal
attention -- see their docstrings for why that matters and what it
replaced.
"""

from __future__ import annotations

import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import PartialRoPE, RMSNorm, causal_mask, causal_windows, safe_eps

# Masking sentinel for CSAAttention's indexer auxiliary KL-divergence loss
# specifically -- see _indexer_aux_loss's docstring for why this needs to
# be a large finite value rather than float('-inf') here, unlike every
# other masked_fill in this file (which correctly use literal -inf).
#
# Sized dynamically to whatever dtype is actually being masked, rather
# than a single fixed constant: a Python float like -1e9 overflows
# float16's range (max magnitude ~65504) and raises "value cannot be
# converted to type c10::Half without overflow" -- reproduced directly
# with `torch.randn(4, dtype=torch.float16).masked_fill(mask, -1e9)` --
# the moment this runs under --dtype float16 (CSA's indexer aux loss is
# active whenever csa_aux_loss_weight > 0, which it is by default).
# torch.finfo(dtype).min is the same "large-but-finite" sentinel scaled
# to whatever's representable, matching the fix already applied to the
# main attention-mask overflow elsewhere in this project.
def _kl_mask_value(dtype: torch.dtype) -> float:
    return torch.finfo(dtype).min


class AttentionSink(nn.Module):
    """Learnable per-head sink logit that absorbs probability mass.

    Callers MUST apply this before masking, not after: logsumexp(-inf,
    sink_logit) = sink_logit (finite), so applying the sink after
    masked_fill would silently resurrect masked-out positions with
    nonzero attention weight. Every call site in this module does
    sink-then-mask for this reason.
    """
    def __init__(self, n_heads: int):
        super().__init__()
        self.sink_logits = nn.Parameter(torch.zeros(n_heads))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        sink = self.sink_logits.view(1, -1, 1, 1)
        return torch.logsumexp(
            torch.stack([logits, sink.expand_as(logits)], dim=0), dim=0,
        )


class AttentionOutputGate(nn.Module):
    """Per-head sigmoid gate on attention OUTPUT, applied before the output
    projection -- the fix from LT2 (arXiv:2605.20670) for compounding
    attention sinks in weight-shared LOOPED architectures specifically.

    AttentionSink (above) gives a head somewhere to dump unneeded softmax
    mass instead of distorting real tokens -- standard practice, and
    correctly implemented here (sink-before-mask, verified). But LT2 found
    that in a looped model, the sink still forms and COMPOUNDS: the same
    softmax block gets re-applied to a residual stream that already carries
    the sink pattern from the previous loop pass, getting worse each
    iteration -- a failure mode a non-looped model never encounters, since
    it only ever sees each layer's softmax once. Their ablation: a
    head-specific sigmoid gate after SDPA, before the output projection,
    eliminates it. Confirmed in exactly this looped setting, not just
    inferred from the non-looped literature.

    Weight zero-init, bias=+4 (sigmoid(4)=0.982): starts close to fully
    open -- a near-identity, safe for warm-starting from an existing
    checkpoint -- while keeping the gradient at this bias large enough
    (sigmoid'(4)~=0.018) to actually move if training finds closing it
    useful. An exact-open bias like +10 would be a closer identity but its
    gradient (~4.5e-5) would make early learning very slow.

    Per-loop bias layered on top, zero-init: lets a loop learn to close its
    gate more than others, motivated by the compounding mechanism itself
    getting worse the more loops have already run. This specific extension
    -- loop-conditioning the gate -- isn't something LT2 tested; the base
    per-head gate is the evidenced part, this is a natural but unverified
    extrapolation of it, consistent with how every other attention/FFN
    mechanism in this codebase is already loop-conditioned.
    """

    def __init__(self, dim: int, n_heads: int, max_loops: int, init_bias: float = 4.0):
        super().__init__()
        self.proj = nn.Linear(dim, n_heads, bias=True)
        nn.init.zeros_(self.proj.weight)
        nn.init.constant_(self.proj.bias, init_bias)
        self.loop_bias = nn.Parameter(torch.zeros(max_loops, n_heads))

    def forward(self, x: torch.Tensor, loop_idx: int) -> torch.Tensor:
        """x: block input [B, T, dim].  Returns gate in (0,1), [B, T, n_heads]
        -- caller reshapes to match its own attention-output layout."""
        loop_idx = int(max(0, min(loop_idx, self.loop_bias.size(0) - 1)))
        return torch.sigmoid(self.proj(x) + self.loop_bias[loop_idx])


class SlidingWindowAttention(nn.Module):
    """Full causal self-attention restricted to a local window."""

    def __init__(self, dim: int, n_heads: int, head_dim: int,
                 window_size: int = 128, rope_dim: int = 64, max_loops: int = 4):
        super().__init__()
        self.n_heads, self.head_dim = n_heads, head_dim
        self.window_size = window_size
        self.qkv      = nn.Linear(dim, 3 * n_heads * head_dim, bias=False)
        self.out_proj = nn.Linear(n_heads * head_dim, dim, bias=False)
        nn.init.zeros_(self.out_proj.weight)  # see attnres.py's zero-init note
        self.q_norm   = RMSNorm(head_dim)
        self.k_norm   = RMSNorm(head_dim)
        self.rope     = PartialRoPE(head_dim, rope_dim=rope_dim)
        self.sink     = AttentionSink(n_heads)
        self.out_gate = AttentionOutputGate(dim, n_heads, max_loops)

    def forward(self, x: torch.Tensor, loop_idx: int = 0) -> torch.Tensor:
        B, T, _ = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(2)
        q, k = self.q_norm(q), self.k_norm(k)
        q, k = self.rope.apply(q), self.rope.apply(k)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask = causal_mask(T, x.device)
        if self.window_size > 0:
            idx = torch.arange(T, device=x.device)
            mask = mask & (idx[None, :] >= idx[:, None] - self.window_size + 1)
        # Sink BEFORE mask: logsumexp(-inf, sink_logit) = sink_logit (finite),
        # so applying the sink after masked_fill would silently resurrect
        # masked-out (future/out-of-window) positions with nonzero weight --
        # masking must have the final say. Valid positions still get the
        # intended sink-boosted logit either way; only invalid ones differ.
        logits = self.sink(logits)
        logits = logits.masked_fill(~mask[None, None], float("-inf"))
        y = torch.matmul(F.softmax(logits, dim=-1), v)
        gate = self.out_gate(x, loop_idx).transpose(1, 2).unsqueeze(-1)  # [B,n_heads,T,1]
        y = y * gate
        return self.out_proj(y.transpose(1, 2).contiguous().view(B, T, -1))


class GroupedOutputProjection(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, groups: int, group_dim: int):
        super().__init__()
        if dim_in % groups != 0:
            raise ValueError(
                f"GroupedOutputProjection: dim_in ({dim_in}) must be "
                f"divisible by groups ({groups}). dim_in is n_heads*head_dim "
                f"for the attention module that owns this projection -- "
                f"adjust n_heads/head_dim/groups so the product divides evenly."
            )
        group_in   = dim_in // groups
        self.groups = groups
        self.proj1  = nn.ModuleList([nn.Linear(group_in, group_dim, bias=False)
                                     for _ in range(groups)])
        self.proj2  = nn.Linear(groups * group_dim, dim_out, bias=False)
        nn.init.zeros_(self.proj2.weight)  # see attnres.py's zero-init note;
                                            # covers both CSA and HCA, which
                                            # both route output through here.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        parts = x.chunk(self.groups, dim=-1)
        return self.proj2(torch.cat([p(c) for p, c in zip(self.proj1, parts)], dim=-1))


class CSAAttention(nn.Module):
    """
    Compressed Sparse Attention (DeepSeek-V4 §CSA) -- causal, per-query.

    Each query attends over two candidate sets, combined into one softmax:
      - up to top_k causally-visible compressed blocks (long-range, sparse)
      - a causal local window of raw tokens (recent, full detail)

    This computes attention per-query directly (einsum over each query's
    own gathered candidates), costing O(B * n_heads * T * (top_k+window)) --
    linear in T. An earlier version routed every query's picks through one
    shared memory pool attended by ALL queries densely, which cost
    O(B * n_heads * T * (T*top_k)) -- quadratic in T and WORSE than plain
    full attention -- and had no causal masking at all, so any position
    could see compressed summaries of future tokens. Both are fixed here.

    Indexer auxiliary loss
    ----------------------
    The indexer (index_down/index_up/index_weight/index_key) scores each
    (query, block) pair, but its scores only ever feed torch.topk's
    INDICES below -- index selection has no gradient, so without a
    dedicated training signal, the indexer never learns which blocks
    matter and stays at its random initialization forever.

    Fix, verified against DeepSeek-V3.2's public documentation for its
    own lightning indexer rather than guessed: train the indexer via a
    KL-divergence loss against a target built from the model's own dense
    attention -- specifically, sum PER-HEAD SOFTMAX PROBABILITIES (not
    averaged logits) across heads, then L1-normalize. Gradient isolation
    is enforced two ways: `comp` is detached before scoring, so the
    auxiliary loss can only update the indexer's own parameters, never
    the compression pathway (kv_a/kv_b/z_a/z_b); and the reference
    distribution is built under torch.no_grad() and detached, since it's
    a fixed target the indexer moves toward, not something the indexer's
    loss should reshape.

    One adaptation from DeepSeek's actual procedure: they retrofit
    sparsity onto an already-pretrained dense model, via a dedicated
    freeze-everything-but-the-indexer warm-up stage before jointly
    fine-tuning. We train this architecture from scratch -- there's no
    pretrained dense behavior to retrofit onto -- so the indexer and the
    reference it chases both develop jointly from step one, kept
    separated by the detach/no_grad boundaries above rather than by
    DeepSeek's literal staged parameter freezing, which assumes a
    different starting point than ours.
    """

    def __init__(self, dim: int, n_heads: int, head_dim: int,
                 m: int = 4, top_k: int = 64, window_size: int = 128,
                 index_heads: int = 32, index_dim: int = 64,
                 groups: int = 8, group_dim: int = 1024, rope_dim: int = 64,
                 max_loops: int = 4):
        super().__init__()
        self.n_heads, self.head_dim = n_heads, head_dim
        self.m, self.top_k, self.window_size = m, top_k, window_size
        self.index_heads, self.index_dim = index_heads, index_dim

        # Overlapping compression (produces one summary vector per m-token block)
        self.kv_a = nn.Linear(dim, head_dim, bias=False)
        self.kv_b = nn.Linear(dim, head_dim, bias=False)
        self.z_a  = nn.Linear(dim, head_dim, bias=False)
        self.z_b  = nn.Linear(dim, head_dim, bias=False)

        # Lightweight indexer: scores each (query, block) pair for top-k selection.
        self.index_down   = nn.Linear(dim, index_dim, bias=False)
        self.index_up     = nn.Linear(index_dim, index_heads * index_dim, bias=False)
        self.index_weight = nn.Linear(dim, index_heads, bias=False)
        self.index_key    = nn.Linear(head_dim, index_dim, bias=False)
        # Per-loop FiLM on the index-head weighting, same (1+delta)/zero-init
        # pattern used for activation/expert routing and attn/ffn FiLM
        # elsewhere in this codebase: lets the SAME shared indexer shift
        # which blocks look relevant at different loop depths (e.g. loop 1
        # casts a wide net, loop 4 has learned to trust a narrower, more
        # precise selection) instead of every loop scoring blocks
        # identically. Applied to index_weight's output (a fixed
        # index_heads-size vector) rather than the final per-block score,
        # since the number of blocks varies with sequence length and can't
        # be a fixed FiLM target.
        self.index_loop_scale = nn.Parameter(torch.zeros(max_loops, index_heads))
        self.index_loop_bias = nn.Parameter(torch.zeros(max_loops, index_heads))

        # Local window (recent raw detail)
        self.local_kv = nn.Linear(dim, head_dim, bias=False)

        # Query + shared (MQA-style) memory norm/rope/sink
        self.q_down   = nn.Linear(dim, dim, bias=False)
        self.q_up     = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.q_norm   = RMSNorm(head_dim)
        self.mem_norm = RMSNorm(head_dim)
        self.rope     = PartialRoPE(head_dim, rope_dim=rope_dim)
        self.sink     = AttentionSink(n_heads)
        self.out_gate = AttentionOutputGate(dim, n_heads, max_loops)

        self.grouped_proj = GroupedOutputProjection(n_heads * head_dim, dim, groups, group_dim)

        # Transient per-forward-pass state, NOT a Parameter/Buffer -- plain
        # Python attributes aren't included in state_dict(), which is what
        # we want here (nothing persistent to checkpoint). Collected and
        # cleared by LoopTransformer.compute_loss(); see model.py.
        self._aux_losses: List[torch.Tensor] = []

    def _compress(self, x: torch.Tensor) -> torch.Tensor:
        """Block j (columns [j*m, (j+1)*m)) -> one summary vector. Purely
        intra-block; causal visibility of the RESULT is enforced in forward()."""
        B, T, _ = x.shape
        m   = self.m
        pad = (m - T % m) % m
        xp  = F.pad(x, (0, 0, 0, pad)) if pad else x
        nb  = xp.size(1) // m
        Ca  = self.kv_a(xp).view(B, nb, m, self.head_dim)
        Cb  = self.kv_b(xp).view(B, nb, m, self.head_dim)
        Za  = self.z_a(xp).view(B, nb, m, self.head_dim)
        Zb  = self.z_b(xp).view(B, nb, m, self.head_dim)
        comp = []
        for i in range(nb):
            b  = torch.zeros_like(Cb[:, i]) if i == 0 else Cb[:, i - 1]
            zb = torch.full_like(Zb[:, i], float("-inf")) if i == 0 else Zb[:, i - 1]
            w  = F.softmax(torch.cat([Za[:, i], zb], dim=1), dim=1)
            comp.append((w[:, :m] * Ca[:, i]).sum(1) + (w[:, m:] * b).sum(1))
        return torch.stack(comp, dim=1)  # [B, nb, head_dim]

    def _index_scores(self, x: torch.Tensor, comp: torch.Tensor, loop_idx: int = 0) -> torch.Tensor:
        loop_idx = int(max(0, min(loop_idx, self.index_loop_scale.size(0) - 1)))
        B, T, _ = x.shape
        q = self.index_up(self.index_down(x)).view(B, T, self.index_heads, self.index_dim)
        w = self.index_weight(x)
        w = w * (1.0 + self.index_loop_scale[loop_idx]) + self.index_loop_bias[loop_idx]
        k = self.index_key(comp)
        return (F.relu(torch.einsum("bthd,bsd->bths", q, k)) * w.unsqueeze(-1)).sum(2)

    def _indexer_aux_loss(
        self,
        q:               torch.Tensor,  # [B, T, H, head_dim] -- same q used for the real attention
        comp:            torch.Tensor,  # [B, nb, head_dim] -- NOT detached (reference uses live comp)
        visible:         torch.Tensor,  # [T, nb] bool
        idx_scores_raw:  torch.Tensor,  # [B, T, nb] -- UNMASKED indexer scores (from detached comp)
    ) -> torch.Tensor:
        """KL(student=indexer || teacher=dense-attention-derived reference).
        See class docstring for the full rationale.

        Uses a large-but-finite masking sentinel (_KL_MASK_VALUE) rather
        than literal float('-inf') here specifically. Verified empirically
        while building this: whenever a query has only one visible block,
        BOTH the student's log-probability and the reference's probability
        hit an exact zero at every OTHER (masked) position. F.kl_div copes
        fine with an exact-zero target against a finite log-probability
        (0 * log(0) correctly contributes 0), but 0 * (log(0) - (-inf))
        is 0 * (-inf - (-inf)) = 0 * NaN = NaN -- literal -inf makes the
        student's log-probability at that position -inf too, not just
        very negative, and the two infinities collide. A finite sentinel
        keeps the student's log-probability very negative but not
        infinite, which resolves it without changing the result anywhere
        that isn't already this degenerate.
        """
        B, T, nb = idx_scores_raw.shape
        has_visible = visible.any(dim=-1)  # [T] -- False only for T < m (before block 0 completes)

        with torch.no_grad():
            comp_n = self.mem_norm(comp)
            ref_logits = torch.einsum("bthd,bsd->bths", q, comp_n) / math.sqrt(self.head_dim)  # [B,T,H,nb]
            mask_4d = visible[None, :, None, :] | (~has_visible)[None, :, None, None]  # unmask zero-visible rows to avoid all-masked rows; excluded from the loss below regardless
            ref_logits = ref_logits.masked_fill(~mask_4d, _kl_mask_value(ref_logits.dtype))
            ref_probs = F.softmax(ref_logits, dim=-1)              # per-head distribution, [B,T,H,nb]
            ref_probs = ref_probs.sum(dim=2)                        # sum across heads (DeepSeek's convention)
            ref_probs = ref_probs / ref_probs.sum(dim=-1, keepdim=True).clamp(min=safe_eps(ref_probs.dtype))  # L1-normalize
            ref_probs = ref_probs.detach()

        mask_3d = visible[None, :, :] | (~has_visible)[None, :, None]  # [1,T,nb]
        student_log_probs = F.log_softmax(
            idx_scores_raw.masked_fill(~mask_3d, _kl_mask_value(idx_scores_raw.dtype)), dim=-1,
        )

        kl_per_position = F.kl_div(student_log_probs, ref_probs, reduction="none").sum(dim=-1)  # [B,T]
        kl_per_position = kl_per_position.masked_fill(~has_visible[None, :], 0.0)
        n_valid = has_visible.sum().clamp(min=1)
        return kl_per_position.sum() / (n_valid * B)

    def forward(self, x: torch.Tensor, loop_idx: int = 0) -> torch.Tensor:
        B, T, _ = x.shape
        comp = self._compress(x)          # [B, nb, head_dim]
        nb   = comp.size(1)

        # ---- causal visibility: block j is visible to query t iff block
        # j's LAST covered position is <= t (fully in the past-or-present) ----
        block_last = (torch.arange(nb, device=x.device) + 1) * self.m - 1   # [nb]
        pos        = torch.arange(T, device=x.device)                        # [T]
        visible    = (block_last[None, :] <= pos[:, None])                   # [T, nb]

        # comp.detach(): forward VALUES are identical either way (topk
        # doesn't back-propagate through indices regardless), but this
        # keeps the auxiliary loss below from leaking gradient into the
        # compression pathway -- see class docstring.
        idx_scores_raw = self._index_scores(x, comp.detach(), loop_idx)
        idx_scores = idx_scores_raw.masked_fill(~visible[None], float("-inf"))

        k = min(self.top_k, nb)
        topv, topi = torch.topk(idx_scores, k, dim=-1)        # [B, T, k]
        sel_valid  = topv > float("-inf")                      # [B, T, k]

        comp_expand = comp[:, None, :, :].expand(B, T, nb, self.head_dim)
        sel = torch.gather(
            comp_expand, 2,
            topi.clamp(min=0).unsqueeze(-1).expand(B, T, k, self.head_dim),
        )  # [B, T, k, head_dim] -- clamp guards ties among all-(-inf) rows;
           # sel_valid masks those slots out of the softmax below regardless.

        # ---- causal local window (recent raw detail) ----
        w = min(self.window_size, T)
        local_x, local_valid = causal_windows(x, w)
        local_mem = self.local_kv(local_x)                     # [B, T, w, head_dim]

        # ---- combine into one per-query candidate set, one softmax ----
        mem   = torch.cat([sel, local_mem], dim=2)              # [B, T, k+w, head_dim]
        valid = torch.cat([sel_valid, local_valid], dim=2)      # [B, T, k+w]

        q = self.q_up(self.q_down(x)).view(B, T, self.n_heads, self.head_dim)
        q = self.rope.apply(self.q_norm(q))

        mem_n  = self.mem_norm(mem)
        logits = torch.einsum("bthd,btsd->bths", q, mem_n) / math.sqrt(self.head_dim)
        # Sink BEFORE mask -- see AttentionSink docstring for why order matters.
        logits = self.sink(logits.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)
        logits = logits.masked_fill(~valid[:, :, None, :], float("-inf"))
        att    = F.softmax(logits, dim=-1)

        out = torch.einsum("bths,btsd->bthd", att, mem)
        gate = self.out_gate(x, loop_idx).unsqueeze(-1)  # [B,T,n_heads,1]
        out = out * gate
        result = self.grouped_proj(out.reshape(B, T, self.n_heads * self.head_dim))

        if self.training and nb > 0:
            self._aux_losses.append(self._indexer_aux_loss(q, comp, visible, idx_scores_raw))

        return result


class HCAAttention(nn.Module):
    """
    Heavily Compressed Attention (DeepSeek-V4 §HCA) -- causal, per-query.

    Like CSAAttention but without top-k selection: each query attends over
    ALL causally-visible compressed blocks (there are few -- roughly
    T/m_prime -- so this stays cheap without needing sparsification) plus
    the same causal local window. Same fix rationale as CSAAttention:
    the original routed everything through one shared, non-causal pool.
    """

    def __init__(self, dim: int, n_heads: int, head_dim: int,
                 m_prime: int = 128, window_size: int = 128,
                 groups: int = 8, group_dim: int = 1024, rope_dim: int = 64,
                 max_loops: int = 4):
        super().__init__()
        self.n_heads, self.head_dim = n_heads, head_dim
        self.m_prime, self.window_size = m_prime, window_size

        self.kv       = nn.Linear(dim, head_dim, bias=False)
        self.z        = nn.Linear(dim, head_dim, bias=False)
        self.local_kv = nn.Linear(dim, head_dim, bias=False)

        self.q_down   = nn.Linear(dim, dim, bias=False)
        self.q_up     = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.q_norm   = RMSNorm(head_dim)
        self.mem_norm = RMSNorm(head_dim)
        self.rope     = PartialRoPE(head_dim, rope_dim=rope_dim)
        self.sink     = AttentionSink(n_heads)
        self.out_gate = AttentionOutputGate(dim, n_heads, max_loops)

        self.grouped_proj = GroupedOutputProjection(n_heads * head_dim, dim, groups, group_dim)

    def _compress(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        m   = self.m_prime
        pad = (m - T % m) % m
        xp  = F.pad(x, (0, 0, 0, pad)) if pad else x
        nb  = xp.size(1) // m
        kv  = self.kv(xp).view(B, nb, m, self.head_dim)
        z   = self.z(xp).view(B, nb, m, self.head_dim)
        return (F.softmax(z, dim=2) * kv).sum(2)   # [B, nb, head_dim]

    def forward(self, x: torch.Tensor, loop_idx: int = 0) -> torch.Tensor:
        B, T, _ = x.shape
        comp = self._compress(x)
        nb   = comp.size(1)

        block_last = (torch.arange(nb, device=x.device) + 1) * self.m_prime - 1
        pos        = torch.arange(T, device=x.device)
        visible    = (block_last[None, :] <= pos[:, None])       # [T, nb]

        w = min(self.window_size, T)
        local_x, local_valid = causal_windows(x, w)
        local_mem = self.local_kv(local_x)                        # [B, T, w, head_dim]

        comp_expand = comp[:, None, :, :].expand(B, T, nb, self.head_dim)
        mem   = torch.cat([comp_expand, local_mem], dim=2)         # [B, T, nb+w, head_dim]
        valid = torch.cat([visible[None].expand(B, T, nb), local_valid], dim=2)

        q = self.q_up(self.q_down(x)).view(B, T, self.n_heads, self.head_dim)
        q = self.rope.apply(self.q_norm(q))

        mem_n  = self.mem_norm(mem)
        logits = torch.einsum("bthd,btsd->bths", q, mem_n) / math.sqrt(self.head_dim)
        # Sink BEFORE mask -- see AttentionSink docstring for why order matters.
        logits = self.sink(logits.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)
        logits = logits.masked_fill(~valid[:, :, None, :], float("-inf"))
        att    = F.softmax(logits, dim=-1)

        out = torch.einsum("bths,btsd->bthd", att, mem)
        gate = self.out_gate(x, loop_idx).unsqueeze(-1)  # [B,T,n_heads,1]
        out = out * gate
        return self.grouped_proj(out.reshape(B, T, self.n_heads * self.head_dim))
