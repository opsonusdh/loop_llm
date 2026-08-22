# Current ZIP Replacement Verification

This tree was built from the uploaded `loopllm_attentionmix_v11(1).zip` and then
replaced with the independently audited/refined implementation where the source
trees diverged.

## Verified fixes carried into this tree

- Real routing-bias update integration after optimizer steps.
- Checkpoint RNG capture/restore for Python, NumPy, Torch, and CUDA.
- Correct resume model/optimizer ownership.
- Attention-mixture balance tolerance as an actual dead zone.
- Per-example Q-exit semantics in evaluation/generation.
- True sequential adaptive early-exit path instead of post-hoc selection.
- Stage-II gate checkpoint overlay restricted to `exit_gate.*` tensors.
- Stage-II dedicated RNG-generator checkpoint state.
- CLI/JSON training-mode precedence and diffusion mode semantics.
- Recurrent-vs-diffusion validation metric separation.
- Fail-fast routing/config validation.
- `--token-ids` generation path without requiring tiktoken.
- Canonical generator shared by the legacy Stage-II generator entry point.
- Direct script execution imports `src/` correctly.

## Verification

- 23/23 automated regression tests pass.
- Synthetic recurrent/refinement smoke test: PASS.
- Synthetic attention-mixture specialization test: PASS.
- Synthetic DiffusionBlocks training/validation test: PASS.
- Synthetic unified training test: PASS.
- Fresh Stage-I training run: PASS.
- Stage-II exit-gate training run: PASS.
- Stage-II checkpoint integrity: non-gate parameters changed = 0.
- Fixed-loop generation: PASS.
- Adaptive gate generation: PASS, with observed earlier-loop exits.

The archive is intended to supersede the uploaded source tree for further training.
