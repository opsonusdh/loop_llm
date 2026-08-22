# LoopLLM Unified v5

This version unifies four ideas into one coherent model/training system:

1. **Loop-specific auxiliary objectives**: every recurrent loop keeps the normal next-token CE, while loops 1..4 also predict progressively longer horizons (+1, +2, +3, +4 tokens).
2. **Refinement auxiliary loss**: consecutive loops are rewarded when the later loop improves token CE by at least a small margin.
3. **Loop-index-aware exit gate**: the gate receives a learned embedding for its recurrence index in addition to the pooled hidden state. Stage II trains all `exit_gate.*` parameters only.
4. **Recurrent-depth controller**: a small identity-biased residual memory path conditions on the fresh loop state, previous loop state, first-loop anchor, state difference, and loop embedding. It does not replace the fresh loop result, so it cannot cheaply suppress later loops by interpolation.
5. **DiffusionBlocks recurrent-depth mode**: a single recurrent pass is used for diffusion-denoising training, while ordinary 4-loop recurrent inference remains available. Hybrid training can alternate recurrent and diffusion objectives over the exact same model parameters.

## Why the recurrent controller changed

The first controller version directly interpolated the new state with the previous state. A synthetic run showed the optimizer could reduce its update gate and avoid later-loop changes, producing worse later-loop CE. v5 replaces that with a residual memory branch added on top of the fresh loop output. This preserves the new loop result as the base path and removes that shortcut.

## Stage-I exit-gate policy

By default `joint_exit_loss_weight=0`. Stage I therefore trains all executed loops directly instead of allowing the learned gate to reduce gradient flowing into deeper loops. The Stage-II gate trainer then learns the exit policy from realized loop improvements. Set `--joint-exit-loss-weight` > 0 only for an explicit Ouro-style joint-exit ablation.

## DiffusionBlocks research basis

The official DiffusionBlocks repository is the public implementation of the ICLR 2026 paper and documents block-wise, diffusion-shaped training across ViTs and also reports recurrent-depth/LLM applications. The recurrent-depth adaptation replaces the repeated recurrent training trajectory with a single denoising-conditioned pass during training; recurrent iteration remains available at inference. This v5 keeps that mode separate and adds a hybrid mode that alternates recurrent-depth supervision with diffusion training.

## Verification performed

- `python -m compileall` passes.
- `pytest -q` passes: 4 tests.
- 80-step structured synthetic recurrent training: validation loss fell from ~3.56 to ~0.040.
- Synthetic hybrid recurrent+diffusion trainer run completed with recurrent and diffusion steps plus validation and checkpoint save.
- Stage-II exit-gate smoke training completed; all gate parameters including the new loop embedding trained.
- Stage-II evaluator confirmed non-gate tensors unchanged and 3 gate tensors changed.
- Depth evaluator was added to compare fixed depths 1..N and explicitly mark extrapolation beyond the trained depth.
- Trainer now fails early if dataset token ids exceed `vocab_size`.

## Important interpretation

The synthetic tests prove the plumbing works. They do **not** prove later recurrent loops are universally better. The current small synthetic run still shows the familiar failure mode where deeper loops can be slightly worse than the first loop. That is useful evidence: the new controller removes one optimization shortcut, but genuine depth improvement must still be established on the real corpus.

## Recommended real run

Start a fresh Stage-I run. Do not resume an older checkpoint because the recurrent-depth controller and exit-gate structure changed.

A practical Colab starting point is:

```bash
python -m scripts.train_refined_v2 \
  --train-data train.bin \
  --val-data train.val.bin \
  --data-dtype uint16 \
  --vocab-size 50281 \
  --dim 512 \
  --n-layers 6 \
  --n-heads 8 \
  --head-dim 64 \
  --ffn-hidden-dim 2048 \
  --rope-dim 64 \
  --max-loops 4 \
  --min-loops 4 \
  --no-loop-sampling \
  --loop-supervision-weight 0.05 \
  --joint-exit-loss-weight 0.0 \
  --loop-refinement-weight 0.02 \
  --loop-refinement-margin 0.001 \
  --loop-task-weight 0.02 \
  --loop-task-mode horizon \
  --exit-gate-loop-embed-dim 16 \
  --recurrent-depth-controller \
  --recurrent-depth-bottleneck-dim 128 \
  --recurrent-update-init 0.95 \
  --diffusion-blocks \
  --training-mode hybrid \
  --hybrid-diffusion-probability 0.25 \
  --diffusion-cond-dim 128 \
  --diffusion-sigma-min 0.002 \
  --diffusion-sigma-max 80 \
  --diffusion-p-mean -1.2 \
  --diffusion-p-std 1.2 \
  --diffusion-sigma-data 0.5 \
  --diffusion-loss-weight 1.0 \
  --diffusion-normalize-embeddings \
  --csa-m 4 \
  --csa-top-k 128 \
  --csa-aux-loss-weight 1.0 \
  --hca-m-prime 256 \
  --sw-window 256 \
  --groups 8 \
  --group-dim 64 \
  --moe-num-shared-experts 1 \
  --moe-num-routed-experts 4 \
  --moe-top-k 2 \
  --activation-top-k 2 \
  --activation-balance-weight 0.01 \
  --moe-aux-loss-weight 0.01 \
  --tie-embeddings \
  --grad-checkpointing \
  --batch-size 2 \
  --seq-len 320 \
  --max-steps 5000 \
  --lr 3e-4 \
  --warmup-steps 200 \
  --checkpoint-dir /content/drive/MyDrive/loop_llm_unified_v5 \
  --checkpoint-interval 100 \
  --eval-interval 100 \
  --eval-iters 20 \
  --log-interval 10 \
  --debug \
  --colab
```

After Stage I, use the Stage-II gate trainer on the fresh checkpoint, then run the gate evaluator and the new `scripts/evaluate_depth.py` before trusting adaptive inference.

## Attention mixture tolerance (v8)

The attention router now supports a minimum dense routing probability per attention family and a soft batch-level deviation tolerance. See `ATTENTION_MIXTURE_V8_TOLERANCE.md`.

Defaults: `min_probability=0.10`, `balance_tolerance=0.25`.
