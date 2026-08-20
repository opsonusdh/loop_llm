# loop-transformer

DeepSeek-V4-style long-context attention + Kimi Attention Residuals + Ouro
LoopLM, unified into one architecture and trained with a real (if
minimal) production pipeline: config validation, checkpointing with
resume, mixed precision, gradient accumulation, and a test suite that
encodes every correctness property this design depends on.

## Architecture

| Layer | Source | Role |
|---|---|---|
| `SlidingWindowAttention` / `CSAAttention` / `HCAAttention` | DeepSeek-V4 study architecture | **Memory** — attention over the context window |
| `LoopedAttnRes` | Kimi Team, [Attention Residuals](https://arxiv.org/abs/2603.15031) | **Amnesia** — replaces `x + f(x)`; every sub-layer's input is a learned softmax mixture over *every* prior sub-layer's output across all loops, so any layer can pull directly from any earlier one |
| `LoopTransformer` + `ExitGate` | ByteDance Seed, [Ouro / LoopLM](https://arxiv.org/abs/2510.25741) | **Reasoning** — the same block stack runs `R` times with shared weights; an entropy-regularized exit gate learns how much iterative refinement each input needs |
| `FeedForward` | DeepSeekMoE, [DeepSeekMoE](https://arxiv.org/abs/2401.06066) | **Specialization** — one always-on shared expert plus sparse top-k routed fine-grained experts; an auxiliary load-balance term discourages expert collapse |
| activation router | [Mixture of Activations](https://arxiv.org/abs/2605.26647) | **Nonlinearity diversity** — four activations are mixed from a lightweight state/loop-conditioned router; KL(mean routing \|\| uniform) discourages collapse while allowing token-level specialization |

Three further papers shaped specific mechanisms (see "Design decisions and their sources" below):
[Mixture-of-Recursions](https://arxiv.org/abs/2507.10524),
[Parcae](https://arxiv.org/abs/2604.12946),
[A Mechanistic Analysis of Looped Reasoning LMs](https://arxiv.org/abs/2604.11791),
[Loop, Think, & Generalize](https://arxiv.org/abs/2604.07822).

## Installation

```bash
pip install -e .            # core package
pip install -e ".[train]"   # + numpy, for scripts/train.py's data pipeline
pip install -e ".[scrape]"  # + requests, for scripts/scrape_github.py
pip install -e ".[dev]"     # + pytest, numpy, requests -- everything, for development
```

## Quick start

```python
from loop_transformer import LoopConfig, LoopTransformer

cfg = LoopConfig(
    vocab_size=50_000, dim=1024, n_layers=16,
    n_heads=16, head_dim=64, ffn_hidden_dim=2752,
)
model = LoopTransformer(cfg)

loss, step_losses = model.compute_loss(token_ids)  # token_ids: [B, T] LongTensor
loss.backward()
```

`compute_loss` returns the Ouro-style entropy-regularized expected loss
across loop steps (see its docstring), plus a `[T]` tensor of per-loop
losses for logging. `forward(idx, max_loops=...)` gives you the raw
per-loop logits and exit probabilities directly if you need them.

## Training

```bash
# 0. (Optional) Scrape a training corpus from public GitHub repos
export GITHUB_TOKEN=ghp_...   # optional but strongly recommended -- see below
python scripts/scrape_github.py --output-dir data/github_raw --num-files 5000

# 1. Tokenize (byte-level default -- zero dependencies, fine to try things out)
python scripts/prepare_data.py --input data/github_raw --output data/train.bin
# (or --input corpus.txt for a single text file from any other source)

# 2. Train
python scripts/train.py \
    --train-data data/train.bin --val-data data/train.val.bin \
    --vocab-size 256 --dim 1024 --n-layers 16 --n-heads 16 --head-dim 64 \
    --ffn-hidden-dim 2752 --max-loops 4 \
    --seq-len 1024 --batch-size 8 --max-steps 20000 \
    --dtype bfloat16 --checkpoint-dir checkpoints/run1
```

### Scraping GitHub for training data

`scripts/scrape_github.py` searches public repos by keyword/language,
lists each repo's files via the GitHub API, filters out binaries,
vendored/generated paths (`node_modules/`, `dist/`, lockfiles, ...), and
likely-minified content, deduplicates by content hash, and saves the
result as one file per blob in `--output-dir` -- ready for
`prepare_data.py`, which accepts that directory directly.

It's resumable: re-running with the same `--output-dir` picks up from a
`_state.json`/`manifest.jsonl` it maintains there, skipping repos
already visited and content already saved, rather than starting over.

**Never hardcode a GitHub token in a script or notebook cell.** Set it
as an environment variable instead:

```bash
export GITHUB_TOKEN=ghp_...   # or a fine-grained github_pat_... token
python scripts/scrape_github.py --output-dir data/github_raw --num-files 5000
```

Unauthenticated requests are capped at 60/hour (10/minute for search) --
workable for a small test run, impractical for real corpus collection.
A token raises that to 5000/hour (30/minute for search); it needs no
special scopes for public repos, just being present. `--token` exists
as a fallback but the env var is safer -- CLI arguments are visible in
shell history and in `ps` output on shared machines. If a token is ever
accidentally committed or pasted somewhere it shouldn't be, revoke it
immediately at GitHub Settings → Developer settings → Personal access
tokens, and issue a new one.

Scraped code carries a mix of open-source licenses. This pipeline
doesn't do license filtering or attribution tracking -- worth being
aware of for how you use any resulting model, particularly if you'd
ever consider distributing it.

Interrupting `train.py` with Ctrl+C (or a SIGTERM, e.g. cluster preemption) saves a
checkpoint before exiting. Resume with `--resume` pointed at the same
`--checkpoint-dir` — it restores model weights, optimizer state, and
step count, so training continues as if uninterrupted.

For real training data, bring your own tokenizer rather than the
byte-level default:

```bash
python scripts/prepare_data.py --input data/github_raw --output data/train.bin \
    --tokenizer mytokenizer:encode   # mytokenizer.py: def encode(text: str) -> list[int]
```

For architecture control beyond the flags `train.py` exposes directly,
pass `--config-json path/to/config.json` with any `LoopConfig` fields;
CLI flags override matching keys in that file.

### Sizing for a fixed memory budget

Rough guide for fitting training in a fixed GPU memory budget: mixed-
precision AdamW costs ≈16 bytes/parameter (2B params + 2B grads + 4B
master weights + 4B momentum + 4B variance). Reserve roughly a third of
your budget for that, leave the rest for activations, and turn on
`grad_checkpointing=True` — LoopLM reruns every layer `max_loops` times,
so activation memory multiplies by that factor unless checkpointed. For
example, a 16GB budget comfortably fits ≈300–400M physical parameters
(`dim=1024, n_layers=16, n_heads=16, head_dim=64` ≈ 295M) with room for
activations at seq_len≈1024, batch≈4–8.

`group_dim` defaults to a flat 1024 regardless of `dim` — at smaller
model sizes this can dominate the parameter count (`GroupedOutputProjection`
costs `group_dim * dim * (1 + groups)`). Scale it down (e.g. `dim // 2`)
for smaller configs.

### Squeezing more effective capacity into the same parameter budget

Two further knobs, both aimed at "more capability per stored parameter"
rather than raw size:

**Embedding factorization** — at large `dim`, the (tied) embedding
matrix is `vocab_size × dim` parameters, often a large fraction of the
total. Factorizing frees real parameters to spend elsewhere:

```python
cfg = LoopConfig(vocab_size=50_000, dim=4096, embed_dim=512)
```

`embed_dim=512` here costs `50_000×512 + 512×4096 ≈ 27.7M` params
instead of `50_000×4096 ≈ 204.8M` — roughly 7x less just for the
embedding. If your vocabulary spans multiple languages or scripts (e.g.
mixed English/Bengali), also set `embed_dim_out` larger than `embed_dim`
(or leave it unfactored, i.e. equal to `dim`) rather than shrinking both
symmetrically — see the design-decisions note below for why:

```python
cfg = LoopConfig(vocab_size=50_000, dim=4096, embed_dim=512,
                  embed_dim_out=2048, tie_embeddings=False)  # asymmetric requires untying
```

**Knowledge distillation** — train a smaller model against a larger
(or just previously-trained) one:

```python
teacher = LoopTransformer(teacher_cfg)
teacher.load_state_dict(torch.load("teacher.pt")["model_state_dict"])

loss, step_losses = student.compute_loss(
    batch, teacher=teacher, distill_alpha=0.5, distill_temperature=2.0,
)
```

`distill_alpha` weights cross-entropy vs. the KL-divergence-to-teacher
term (`1.0` = pure CE, no distillation; `0.0` = pure distillation).
Teacher and student can be entirely different configs — only
`vocab_size` needs to match. See the capacity-gap caveat under Known
limitations before picking a teacher/student size ratio.

## Checkpointing

```python
from loop_transformer import save_checkpoint, load_checkpoint

save_checkpoint("model.pt", model, optimizer=optimizer, step=1000)

ckpt = load_checkpoint("model.pt", optimizer=optimizer)  # optimizer arg optional
model = ckpt["model"]   # reconstructed from the checkpoint's own saved config --
                        # no need to separately remember hyperparameters
step = ckpt["step"]
```

Loads use `weights_only=True` (PyTorch's safe unpickling mode). This
format only ever stores tensors and plain Python primitives, so there's
no capability lost — but it does mean anything you stash in
`extra=` when saving must also stick to plain types.

## Testing

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

88 tests covering:
- **Zero-init** (`test_init.py`) — every sub-layer type outputs exactly
  zero at initialization; the model's first-loop behavior collapses to
  pure tied-embedding self-similarity, as derived in `attnres.py`.
- **Causality** (`test_causality.py`) — perturbing tokens after a cutoff
  produces zero change to logits before that cutoff, bit-for-bit, across
  every loop.
- **Loop dynamics** (`test_loop_dynamics.py`) — dynamic loop-count
  sampling is correctly gated by `model.training`; the depth-
  extrapolation warning in `generate()` fires exactly when it should.
- **Gradients** (`test_gradients.py`) — gradient checkpointing produces
  mathematically identical gradients to the non-checkpointed path;
  full gradient coverage (see the CSA indexer note below).
- **CSA indexer auxiliary loss** (`test_indexer_aux_loss.py`) — gradient
  reaches only the indexer's own parameters, never anything else; the
  loss is always finite (a real NaN was found and fixed during
  development); gradient checkpointing's double forward pass doesn't
  double-count it; the indexer's weights measurably move and the
  auxiliary loss decreases over a real training run.
- **Embedding factorization** (`test_embedding_factorization.py`) —
  parameter savings, symmetric and asymmetric shapes, all validation
  rules, and that causality/checkpointing consistency both still hold
  with factorization enabled.
- **Distillation** (`test_distillation.py`) — teacher receives zero
  gradient and its training-mode is preserved; vocab-mismatch and
  out-of-range argument validation; `distill_alpha=1.0` is numerically
  identical to not passing a teacher at all (the critical regression
  check for the blending logic).
- **Training** (`test_training.py`) — loss actually decreases on a fixed
  batch; later loop steps beat the first after training.
- **Checkpointing** (`test_checkpointing.py`) — save/load round-trips
  preserve weights, optimizer state, and config exactly.
- **Config validation** (`test_config.py`) — malformed configs are
  rejected at construction time with an actionable message.
- **GitHub scraper** (`test_scrape_github.py`) — filtering logic
  (extensions, vendored paths, size bounds, minified-content heuristic)
  and state save/reload, without touching the network. The scraper's
  network-facing parts (search, rate-limit handling, downloading) are
  covered by manual integration testing against the live API instead,
  same as `prepare_data.py`/`train.py`.

## Known limitations

- **Depth extrapolation beyond `max_loops` is unverified for this
  model.** [A Mechanistic Analysis of Looped Reasoning LMs](https://arxiv.org/abs/2604.11791)
  found Ouro-style looped models specifically show "structural drift"
  rather than clean convergence when run past their trained loop count.
  `generate(n_loops=k)` warns (doesn't block) when `k > max_loops`.
- **Distillation doesn't guarantee "small model matches big model."**
  [Distillation Scaling Laws](https://arxiv.org/abs/2502.08606)
  (Busbridge et al.) found a real capacity-gap effect: too large a
  teacher/student size ratio can hurt transfer rather than help it, with
  optimal teacher scale tracking student scale roughly linearly rather
  than "bigger teacher is strictly better." Treat any specific size
  ratio as something to tune, not assume. This implementation is also
  classical (fixed-corpus) distillation, not the newer on-policy variant
  (student generates its own rollouts, teacher supervises those) that
  2026 literature reports further gains from.
- **No license filtering on scraped GitHub data.** `scrape_github.py`
  filters for content quality (binaries, vendored/generated paths,
  duplicates) but not license terms. Scraped repos carry a mix of
  licenses; there's no attribution tracking or license-based inclusion/
  exclusion. Worth knowing before training on this data for anything
  beyond personal experimentation.
- **No auxiliary RL/safety alignment stage.** This is a pretraining-loss
  architecture and training loop only — no SFT, RLVR, or safety
  fine-tuning stage, unlike the full Ouro recipe.
- **CPU training works but is slow**, particularly through `CSAAttention`/
  `HCAAttention`'s Python-level per-block compression loop. Fine for
  development and the test suite; use a GPU for real training runs.

## Design decisions and their sources

- **CSA indexer auxiliary loss** — the indexer's block-selection scores
  only ever fed `torch.topk`'s *indices*, never its values, so index
  selection wasn't differentiable and the indexer never received a
  training signal from the LM loss alone — dead weight from
  initialization onward. Fixed with a KL-divergence loss verified
  against DeepSeek-V3.2's public documentation for its own lightning
  indexer (summing per-head softmax probabilities across heads, then
  L1-normalizing, as the target — not averaged logits, which was the
  first, wrong guess here) rather than reconstructed from assumption.
  Gradient-isolated via `.detach()`/`torch.no_grad()` so it only trains
  `index_down`/`index_up`/`index_weight`/`index_key`, never anything
  else — confirmed empirically, including that gradient checkpointing's
  double forward pass (a real, verified interaction, not a hypothetical
  one) doesn't double-count it. `test_indexer_aux_loss.py` covers all of
  this, including a training run where the indexer's weights measurably
  move and the auxiliary loss decreases.
- **Asymmetric embedding factorization** (`LoopConfig.embed_dim` /
  `embed_dim_out`) — ALBERT-style, but decoupled rather than forced
  symmetric. [Rethinking Embedding Coupling in Pre-trained Language
  Models](https://arxiv.org/abs/2010.12821) (Chung et al., Google, ICLR
  2021) found that shrinking *both* input and output embeddings together
  specifically hurts vocab-diverse/multilingual models, while a larger
  (or fully unfactored) output embedding recovers most of the quality at
  modest extra cost. `embed_dim` and `embed_dim_out` are independently
  configurable; tying is only possible (and only attempted) when they
  match.
- **Knowledge distillation** (`compute_loss(..., teacher=...)`) —
  classical Hinton et al.-style soft-label distillation: temperature-
  scaled KL-divergence toward a frozen teacher's output distribution,
  blended with the standard cross-entropy loss. Still the standard
  production technique as of 2026 (commonly reported: 5–30x inference
  cost reduction, ~95–97% of teacher quality retained — a typical range,
  not a promise for any specific setup). See the capacity-gap caveat
  above before picking a teacher/student size ratio.
- **Zero-init output projections** (`SlidingWindowAttention.out_proj`,
  `GroupedOutputProjection.proj2`, `FeedForward.w3`) — from
  [Loop, Think, & Generalize](https://arxiv.org/abs/2604.07822), which
  zero-inits attention/FFN output projections so recurrent blocks start
  as identity maps, stabilizing unbounded loop unrolling. Our AttnRes-
  based design doesn't have a single evolving hidden state the way that
  paper's setting does, so "identity map" doesn't translate literally —
  see the full derivation in `attnres.py` for what actually happens here
  instead (short version: the whole network reduces to a stable function
  of the token embedding alone at init, regardless of loop count).
- **Dynamic per-batch loop-count sampling** (`LoopConfig.loop_sampling`) —
  both Loop, Think, & Generalize and [Parcae](https://arxiv.org/abs/2604.12946)
  independently find that training with a *variable* loop count (sampled
  per batch from a clipped Poisson, rather than always training at a
  fixed ceiling) increases learnable recursion depth, improves
  extrapolation past trained depth, and slows the "overthinking" quality
  decay from over-looping.
- **Per-query causal attention in CSA/HCA** — an earlier version routed
  every query's top-k block selection through one shared memory pool
  that every query attended over densely: `O(T² · top_k)` (worse than
  plain full attention) *and* non-causal (any position could see
  compressed summaries of future tokens). Rewritten to genuinely
  per-query, causal attention: `O(T · (top_k + window))`, linear in `T`.
- **Weight tying + embedding re-init** — `tok_emb`/`lm_head` are tied by
  default (saves `vocab_size × dim` parameters). `nn.Embedding`'s default
  init (std=1) is fine for a standalone lookup table but wrong once that
  matrix doubles as a linear projection weight — logits blow up to
  `std ≈ √dim`. Re-initialized to std=0.02 (GPT-2/nanoGPT convention).
- **Not implemented, considered and deferred**: token-level (rather than
  sequence-level) adaptive exit routing from
  [Mixture-of-Recursions](https://arxiv.org/abs/2507.10524) — the
  biggest remaining architectural change on the table, and the most
  speculative payoff for a study-scale model. `ExitGate` currently
  mean-pools over the whole sequence, so every token in a sequence
  shares one exit decision. Also deferred: per-loop LoRA-style micro-
  adapters (letting each loop specialize slightly beyond fully shared
  weights) and multi-token prediction (denser training signal per
  parameter, DeepSeek-V3 style) — both plausible further parameter-
  efficiency gains, neither implemented here given the risk of
  destabilizing the causality/zero-init/checkpointing guarantees this
  version has been tested against.

## Project structure

```
src/loop_transformer/
    config.py        LoopConfig (validated dataclass)
    layers.py         RMSNorm, PartialRoPE, causal masking/windowing utilities
    attention.py      SlidingWindowAttention, CSAAttention, HCAAttention
    feedforward.py    DeepSeekMoE-style shared/routed FFN + 4-way activation router
    attnres.py        DepthAttnRes, LoopedAttnRes (+ zero-init derivation)
    block.py          TransformerBlock, ExitGate
    model.py          LoopTransformer (forward / compute_loss / generate)
    checkpointing.py  save_checkpoint / load_checkpoint
scripts/
    scrape_github.py  GitHub -> directory of text files (resumable, deduped)
    prepare_data.py   text file or directory -> token-id .bin file
    train.py          CLI training loop
tests/                pytest suite (see Testing, above)
```
