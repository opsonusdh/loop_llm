"""Learned recurrent-depth state control.

The controller is inspired by recent recurrent-depth work that uses a shared
core repeatedly while learning a lightweight update/retention gate conditioned
on the current hidden state, the previous recurrent state, a stable first-pass
anchor, and the recurrence index.  It is deliberately identity-biased so it
can be inserted into the existing LoopLLM without turning the first training
steps into a different model.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class RecurrentDepthController(nn.Module):
    """Token-wise GRU-like interpolation between recurrent states.

    For loop t>0:
        u_t = sigmoid(G([h_t, s_{t-1}, s_0, |h_t-s_{t-1}|, e_t]))
        m_t = MLP([h_t, s_{t-1}, s_0, |h_t-s_{t-1}|, e_t])
    s_t = h_t + a * u_t * m_t

    `h_t` remains the primary output of the shared core.  The controller is a
    small residual memory path, not an interpolation that can suppress a good
    new loop result.  It therefore adds recurrent state capacity without
    giving the optimizer an easy "turn off later loops" shortcut.
    """

    def __init__(
        self,
        dim: int,
        loop_embed_dim: int = 16,
        bottleneck_dim: int = 128,
        max_loops: int = 4,
        update_init: float = 0.95,
    ) -> None:
        super().__init__()
        if dim <= 0:
            raise ValueError("dim must be > 0")
        if loop_embed_dim <= 0:
            raise ValueError("loop_embed_dim must be > 0")
        if bottleneck_dim <= 0:
            raise ValueError("bottleneck_dim must be > 0")
        if max_loops < 1:
            raise ValueError("max_loops must be >= 1")
        if not 0.0 < update_init <= 1.0:
            raise ValueError("update_init must be in (0, 1]")

        self.dim = dim
        self.loop_embed_dim = loop_embed_dim
        self.bottleneck_dim = bottleneck_dim
        self.max_loops = max_loops

        self.loop_embedding = nn.Embedding(max_loops, loop_embed_dim)
        nn.init.normal_(self.loop_embedding.weight, mean=0.0, std=0.02)

        in_dim = dim * 4 + loop_embed_dim
        self.norm = nn.LayerNorm(in_dim)
        self.down = nn.Linear(in_dim, bottleneck_dim)
        self.up = nn.Linear(bottleneck_dim, dim)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)
        p = min(max(update_init, 1e-4), 1.0 - 1e-4)
        self.update_bias = nn.Parameter(torch.tensor(torch.logit(torch.tensor(p)).item()))
        self.memory_logit = nn.Parameter(torch.tensor(-2.1972246))  # sigmoid≈0.10

        self.last_update: Optional[torch.Tensor] = None

    def forward(
        self,
        current: torch.Tensor,
        previous: Optional[torch.Tensor],
        anchor: Optional[torch.Tensor],
        loop_idx: int,
    ) -> torch.Tensor:
        if previous is None or anchor is None:
            self.last_update = torch.ones(
                current.shape[:-1], device=current.device, dtype=current.dtype
            )
            return current

        idx = max(0, min(int(loop_idx), self.max_loops - 1))
        emb = self.loop_embedding.weight[idx].view(1, 1, -1).to(
            device=current.device, dtype=current.dtype
        )
        emb = emb.expand(current.size(0), current.size(1), -1)
        delta = current - previous
        features = torch.cat(
            [current, previous, anchor, delta.abs(), emb], dim=-1
        )
        hidden = torch.tanh(self.down(self.norm(features)))
        update = torch.sigmoid(torch.zeros(current.size(0), current.size(1), device=current.device, dtype=current.dtype) + self.update_bias)
        memory = self.up(hidden)
        memory_scale = torch.sigmoid(self.memory_logit)
        self.last_update = update.expand(current.size(0), current.size(1)).detach()
        return current + memory_scale * update.unsqueeze(-1) * memory
