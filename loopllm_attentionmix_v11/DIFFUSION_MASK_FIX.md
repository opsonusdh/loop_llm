# DiffusionBlocks noisy-target mask fix

This revision fixes the CSA/HCA NaN failure where noisy query 0 had no valid candidate because the clean predecessor set and shifted local window were both empty.

## Correct layout

The diffusion LM now feeds the recurrent core a concatenated sequence:

`[clean context, noisy target latents]`

Training mask:

- clean query `i` -> clean keys `<= i`
- noisy query `i` -> clean keys `< i`
- noisy query `i` -> its own noisy key
- noisy queries never see other noisy keys

The noisy diagonal is explicitly enabled for every noisy position. This makes every query row non-empty, including noisy position 0.

## CSA/HCA propagation

CSA/HCA now accept the same dense diffusion visibility mask and derive candidate visibility from it:

- compressed candidates are visible only when every token covered by the compressed block is visible to that query;
- local-window candidates inherit exact per-token visibility from the diffusion mask;
- the local self candidate is therefore preserved even when a noisy query has no valid clean predecessor.

## Additional fixes

- `--diffusion-blocks` now selects `training_mode=diffusion` automatically when no explicit training mode was supplied. Explicit `--training-mode hybrid` remains respected.
- Diffusion generation now denoises one new target latent per autoregressive step. The new target is allowed to see the full clean prompt, while still only attending to itself on the noisy side.
- `generate_diffusionblocks.py` supports `--token-ids`, so synthetic/regression generation does not require tiktoken.
