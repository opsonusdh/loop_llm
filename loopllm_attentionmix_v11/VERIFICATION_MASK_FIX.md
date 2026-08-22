# Verification: DiffusionBlocks mask fix

Date: 2026-08-22

## Research basis

The requested fix follows the reported failure mode: noisy query 0 can legitimately have no clean predecessor, so it must retain a self candidate. The official SakanaAI/DiffusionBlocks repository is public and its README describes the DiffusionBlocks training framework; the recurrent-depth LM adaptation is described in the paper rather than implemented as an official LoopLM repository.

## Code changes

1. Added `build_diffusion_causal_mask()`.
2. Diffusion training uses `[clean, noisy]` concatenation with a strict autoregressive mask.
3. Added the noisy diagonal/self edge for every noisy query.
4. Threaded `attention_mask` through TransformerBlock -> SWA/CSA/HCA.
5. CSA/HCA derive compressed-block and local-window validity from the exact mask.
6. Corrected `--diffusion-blocks` so it no longer silently runs ordinary recurrent training when no training mode is supplied.
7. Diffusion generation now denoises one new-token latent with full-prompt visibility.
8. Added token-id generation fallback for synthetic tests.

## Tests

`PYTHONPATH=. pytest -q` -> **5 passed**

Specific regression test:
- checks noisy positions 0, 3, 5, 11 all have a self candidate;
- checks the full CSA/HCA path has finite outputs;
- backpropagates through the masked pass and checks all gradients are finite.

## Diffusion synthetic training

The synthetic DiffusionBlocks run now actually enters `mode=diffusion` automatically when `--diffusion-blocks` is provided.

Validation raw CE:
- step 10: 3.3941
- step 20: 3.3830
- step 30: 3.3742
- step 40: 3.3678
- step 50: 3.3649

Weighted diffusion validation:
- step 10: 93.7650
- step 20: 93.4712
- step 30: 93.2259
- step 40: 93.0583
- step 50: 92.9736

This is a genuine downward validation trend, not merely a no-NaN smoke test.

## Hybrid synthetic training

An explicit `training_mode=hybrid` run was also completed after fixing config precedence.

Observed both modes in one run:
- diffusion step(s): `mode=diffusion`
- recurrent step(s): `mode=recurrent`
- hybrid validation reported both recurrent and diffusion metrics
- checkpoint save succeeded

## Generation

The diffusion generator was tested on the synthetic checkpoint with token IDs and returned a valid extended sequence.
