"""loop_transformer -- DeepSeek CSA/HCA attention + Kimi Attention Residuals
+ Ouro LoopLM, unified.

    Memory    (DeepSeek):  SWA / CSA / HCA attend over the context window.
    Amnesia   (AttnRes):   every sub-layer's input is a learned softmax
                            mixture over ALL prior sub-layer outputs across
                            all loops, so any layer can pull from any
                            earlier one directly.
    Reasoning (LoopLM):    the block stack is applied R times with shared
                            weights; an entropy-regularised exit gate
                            allocates more computation to harder inputs.

See README.md for architecture details, configuration guidance, and
known limitations.

Quick start
-----------
    from loop_transformer import LoopConfig, LoopTransformer

    cfg = LoopConfig(vocab_size=50_000, dim=1024, n_layers=16,
                      n_heads=16, head_dim=64, ffn_hidden_dim=2752)
    model = LoopTransformer(cfg)

    loss, step_losses = model.compute_loss(token_ids)
    loss.backward()
"""

from .attention import CSAAttention, GroupedOutputProjection, HCAAttention, SlidingWindowAttention
from .attnres import DepthAttnRes, LoopedAttnRes
from .block import ExitGate, TransformerBlock
from .checkpointing import load_checkpoint, save_checkpoint
from .config import LoopConfig, LoopConfigError
from .feedforward import FeedForward
from .layers import RMSNorm, PartialRoPE
from .model import LoopTransformer

__version__ = "0.1.0"

__all__ = [
    # Primary API
    "LoopConfig",
    "LoopConfigError",
    "LoopTransformer",
    "save_checkpoint",
    "load_checkpoint",
    # Building blocks (for extension / testing / introspection)
    "TransformerBlock",
    "ExitGate",
    "DepthAttnRes",
    "LoopedAttnRes",
    "FeedForward",
    "SlidingWindowAttention",
    "CSAAttention",
    "HCAAttention",
    "GroupedOutputProjection",
    "RMSNorm",
    "PartialRoPE",
]
