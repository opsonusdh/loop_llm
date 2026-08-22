# LoopLM Refinement v2

This is a refined Stage-I/Stage-II experiment for the LoopTransformer project.
It keeps the existing recurrent LM, sparse/compressed attention, LoopedAttnRes,
and sparse MoE, and adds three controlled mechanisms requested from the recent
feedback:

1. Loop-to-loop refinement auxiliary loss.
2. Explicit learned loop-index embedding in the exit gate.
3. A distinct auxiliary prediction task for each loop.

## What changed

### 1. Refinement auxiliary loss

For consecutive primary next-token losses:

`improvement_t = L_t - L_(t+1)`

we add:

`ReLU(refinement_margin - improvement_t)`

with a small configurable weight. It is intentionally soft and does not force
monotonic improvement on every stochastic example.

### 2. Loop-index-aware exit gate

The gate now receives:

`concat(mean_pool(hidden), learned_loop_embedding)`

The default loop embedding size is 16. With hidden size 512 this changes the
old 513-parameter gate into a small 529-parameter weight matrix plus the
embedding table and bias, while keeping the gate tiny relative to the LM.

### 3. Distinct task per loop

The main next-token CE remains at every loop. In addition, the loop-specific
auxiliary task uses different prediction horizons:

- Loop 1 -> +1 token
- Loop 2 -> +2 tokens
- Loop 3 -> +3 tokens
- Loop 4 -> +4 tokens

This is deliberately not four unrelated classifiers. It keeps the task inside
the language-modeling family while giving later loops a progressively different
prediction objective.

## Safe defaults

The default Config values in this branch are:

- `loop_refinement_weight=0.02`
- `loop_refinement_margin=0.001`
- `loop_task_weight=0.02`
- `loop_task_mode='horizon'`
- `exit_gate_loop_embed_dim=16`
- `loop_monotonic_weight=0.0`

Direct loop supervision and the legacy batch-level monotonic penalty remain
available for ablations, but the monotonic penalty is disabled by default.

## Synthetic smoke test

The tiny synthetic experiment was run after fixing one dependency mismatch in
the initial test: the current sparse-MoE FeedForward implementation must be used
because the block passes loop-aware routing/bias parameters to it.

The smoke test then passed:

- Validation before training: `3.39904`
- Validation at step 1: `3.22246`
- Validation at step 10: `2.59219`
- Validation at step 30: `1.71152`
- Validation at step 60: `1.11349`
- Final validation sample: `1.13581`
- Gate loop embedding gradients were nonzero.
- The full train entry point was also run for 5 synthetic steps and validation
  decreased from `3.2549` at step 1 to `3.0767` at step 4.
- Stage-II gate training was run for 5 synthetic steps and the gate BCE fell
  from `0.703525` to `0.658343` before a small stochastic rebound.
- Stage-II resume from step 5 to step 6 worked.
- The refined evaluator passed and reported unchanged non-gate tensors.
- Pytest: `1 passed` in `tests/test_refinement_v2.py`.

The synthetic run is only a plumbing/learning sanity check. It is not evidence
that the architecture is better on real language data.

## Files

Replace/add these in the project:

- `src/loop_transformer/model.py`
- `src/loop_transformer/block.py`
- `src/loop_transformer/config.py`
- `src/loop_transformer/feedforward.py`
- `src/loop_transformer/attention.py`
- `src/loop_transformer/attnres.py`
- `src/loop_transformer/layers.py`
- `src/loop_transformer/checkpointing.py`
- `scripts/train_refined_v2.py`
- `scripts/train_exit_gate_refined_v2.py`
- `scripts/evaluate_exit_gate_refined_v2.py`
- `scripts/generate_with_exit_gate_refined_v2.py`
- `scripts/synthetic_refinement_smoketest.py`
- `tests/test_refinement_v2.py`

## Recommended real-data Stage-I command

Use a fresh checkpoint directory. Do not resume the old Stage-I experiment,
because the model shape now includes the loop embedding in the exit gate and the
training objective has changed.

```text
python -m scripts.train_refined_v2 `
    --train-data train.bin `
    --val-data train.val.bin `
    --data-dtype uint16 `
    --vocab-size 50281 `
    --dim 512 `
    --n-layers 6 `
    --n-heads 8 `
    --head-dim 64 `
    --ffn-hidden-dim 2048 `
    --rope-dim 64 `
    --max-loops 4 `
    --min-loops 1 `
    --no-loop-sampling `
    --loop-supervision-weight 0.05 `
    --loop-refinement-weight 0.02 `
    --loop-refinement-margin 0.001 `
    --loop-task-weight 0.02 `
    --loop-task-mode horizon `
    --exit-gate-loop-embed-dim 16 `
    --csa-m 4 `
    --csa-top-k 128 `
    --csa-aux-loss-weight 1.0 `
    --hca-m-prime 256 `
    --sw-window 256 `
    --groups 8 `
    --group-dim 64 `
    --moe-num-shared-experts 1 `
    --moe-num-routed-experts 4 `
    --moe-top-k 2 `
    --activation-top-k 2 `
    --activation-balance-weight 0.01 `
    --moe-aux-loss-weight 0.01 `
    --tie-embeddings `
    --grad-checkpointing `
    --batch-size 2 `
    --seq-len 320 `
    --max-steps 10000 `
    --dtype float32 `
    --lr 3e-4 `
    --warmup-steps 200 `
    --min-lr-ratio 1.0 `
    --checkpoint-dir "stage1_refined_v2" `
    --checkpoint-interval 100 `
    --eval-interval 100 `
    --log-interval 10 `
    --debug
```

For Colab, add the existing `--colab` flag if your copy of `train.py` uses the
Drive checkpoint/recycle-bin mode.

## Stage-II

After Stage-I has finished, run `train_exit_gate_refined_v2.py` using the new
Stage-I checkpoint. It trains **all `exit_gate.*` parameters** (projection plus
loop embedding) and freezes the rest of the language model.

The Stage-II checkpoint can be resumed with `--resume`.

## Evaluation and generation

Use `evaluate_exit_gate_refined_v2.py` to compare fixed-depth and Q-exit
inference. It supports `cpu`, `cuda`, `xpu`, `dml`, and `auto`.

Use `generate_with_exit_gate_refined_v2.py` to overlay the complete trained
`exit_gate.*` state onto a matching Stage-I model and generate with adaptive
exit. Fixed-loop generation remains available with `--loops`.

## Important

This branch is an experiment, not a claim that these auxiliary objectives are
better than the original Ouro objective. The right next real-data comparison is
A/B/C:

A. Existing Stage-I baseline.
B. Refinement + loop-specific horizon auxiliary + loop-index-aware gate.
C. Same as B but with the extra auxiliaries disabled for ablation.

Compare validation loss, per-loop loss improvements, gate depth, and generation
quality before deciding whether the new objective stays.
