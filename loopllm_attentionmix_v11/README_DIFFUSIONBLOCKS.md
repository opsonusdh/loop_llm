# DiffusionBlocks: true block-wise training

This repository uses the Sakana AI DiffusionBlocks idea in its **block-wise** form for Stage-I diffusion updates. The implementation is an adaptation to the LoopTransformer architecture.

## What changed

The physical Transformer layers are partitioned into contiguous blocks. For a 6-layer model with `--diffusion-num-blocks 3`, the partition is:

```text
Block 0: layers 0-1
Block 1: layers 2-3
Block 2: layers 4-5
```

A diffusion training step samples **one block for the entire batch**, samples sigma from that block's dedicated log-normal interval, runs only that block, and backpropagates through that block. Prefix and suffix physical layers are not executed, so their activations are not part of the diffusion autograd graph.

This is the key memory-saving property described by Sakana AI: split the network into B blocks, assign each block a noise range, condition each block on its range, and train one block per iteration. The official implementation samples one block per training step and uses block-specific sigma intervals.

## LoopTransformer-specific adaptation

The normal inference architecture remains unchanged:

```text
shared 6-layer stack -> repeated for 4 recurrent loops
```

The block-wise diffusion path is a **training-only objective**. It uses the existing attention/FFN/router parameters of the selected physical layer block plus the shared diffusion conditioning/output machinery. It does not replace the ordinary recurrent inference path.

Because LoopTransformer uses `LoopedAttnRes`, the selected layers use a local block-scoped residual-attention view during diffusion training rather than pretending that uncomputed prefix/suffix sublayers exist. This keeps the training graph self-contained while preserving the same physical `TransformerBlock` implementations, routing, and diffusion FiLM conditioning.

## Default for the compact 6-layer model

Use:

```text
--diffusion-num-blocks 3
```

so each diffusion block contains two physical Transformer layers. For architectures where `n_layers` is not divisible by the requested number of blocks, configuration validation fails early rather than silently creating uneven blocks.

## Hybrid training

`training_mode=hybrid` alternates between the normal recurrent objective and this block-wise diffusion objective. The two graphs are never accumulated together.

## Important distinction

This is different from the older recurrent-depth diffusion path that ran the entire Transformer once on every diffusion update. That older path reduced recurrent depth but did **not** provide the block-wise activation-memory reduction. The current implementation is intentionally block-wise.
