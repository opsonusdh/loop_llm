# DiffusionBlocks mode for LoopLLM

This repository now contains an **optional DiffusionBlocks recurrent-depth training mode** based on Sakana AI's DiffusionBlocks paper.

The key point is that this is the **recurrent-depth adaptation**, not the paper's separate 4-block AR Transformer recipe. In the paper, recurrent-depth models are treated as one denoiser `D_theta(z_sigma, x, sigma)` and trained with **one recurrent forward pass per update**; the original multi-iteration recurrence is retained for inference. The paper reports this on Huginn and reports lower training computation from avoiding BPTT through the recurrent iterations. See Section 5.5 / Appendix E.5. 

## What the implementation adds

- EDM-style log-normal sigma sampling, with the paper defaults:
  - `P_mean=-1.2`
  - `P_std=1.2`
  - `sigma_min=0.002`
  - `sigma_max=80`
  - `sigma_data=0.5`
- EDM input/output preconditioning and loss weighting.
- Continuous sigma conditioning injected into both attention and FFN sub-layer normalization paths.
- One recurrent pass per DiffusionBlocks training step.
- Clean current-token context plus a noisy next-token latent target, so the objective remains autoregressive at the token level.
- L2-normalized embeddings in diffusion mode, matching the paper's recommendation for discrete outputs.
- Deterministic multi-sigma validation metrics that report both raw token CE and the weighted diffusion objective.
- Experimental Euler-style denoising generation through `scripts/generate_diffusionblocks.py`.

## Important separation

Normal LoopLM training and inference are unchanged when `diffusion_blocks=False`.

DiffusionBlocks training is a **Stage-I alternative**. Do not resume an ordinary LoopLM checkpoint into DiffusionBlocks mode. Train a fresh Stage-I checkpoint for this mode.

The existing Stage-II exit gate should also be treated as a separate experiment. Its current targets were learned from ordinary loop trajectories, not diffusion denoising trajectories, so it should be retrained after a successful diffusion Stage-I run.

## Real-data Colab command

Use the same model dimensions you have been using, but start a **fresh checkpoint directory**:

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
    --min-loops 1 \
    --no-loop-sampling \
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
    --diffusion-blocks \
    --diffusion-cond-dim 128 \
    --diffusion-sigma-min 0.002 \
    --diffusion-sigma-max 80 \
    --diffusion-p-mean -1.2 \
    --diffusion-p-std 1.2 \
    --diffusion-sigma-data 0.5 \
    --diffusion-loss-weight 1.0 \
    --diffusion-normalize-embeddings \
    --batch-size 2 \
    --seq-len 320 \
    --max-steps 5000 \
    --dtype float32 \
    --lr 3e-4 \
    --warmup-steps 200 \
    --checkpoint-dir /content/drive/MyDrive/loop_llm_diffusionblocks \
    --checkpoint-interval 100 \
    --eval-interval 100 \
    --eval-iters 20 \
    --log-interval 10 \
    --debug \
    --colab
```

For the first real run, keep the checkpoint directory separate from the existing Stage-I run so the experiments stay comparable.

## Experimental generation

```bash
python -m scripts.generate_diffusionblocks \
    --checkpoint /content/drive/MyDrive/loop_llm_diffusionblocks/latest.pt \
    --prompt "def fibonacci(n):" \
    --device cuda \
    --steps 4 \
    --max-new-tokens 100 \
    --temperature 0.8 \
    --top-k 50
```

This generation path is experimental. The ordinary LoopLM generator remains the reference inference path for non-diffusion checkpoints.
