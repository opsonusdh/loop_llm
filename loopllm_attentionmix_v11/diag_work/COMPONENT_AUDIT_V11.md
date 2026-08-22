# LoopLLM component audit v11

## What was trained

The diagnostic corpus is not synthetic language invented for the test. It is a byte-level corpus built recursively from the repository's own `.py`, `.md`, `.txt`, `.toml`, and `.json` sources, with an 85/15 train/validation split. This exercises code, prose, configuration, comments, and documentation patterns.

Train bytes: 676,648
Validation bytes: 119,409
Tokenizer for diagnosis: UTF-8 bytes, vocab 256. This is deliberately a diagnostic tokenizer, not the production p50k tokenizer.

## Experiments

All experiments used the same seed, windows, optimizer family, small 48-dim / 2-layer model, 4 recurrent loops, recurrent-depth controller, refinement loss, loop-specific horizon task, contextual activation/MIx routing, and the same train/validation text.

- full: all components enabled
- no_attention: attention mixture router forced to a uniform mixture for diagnosis
- no_activation: activation router forced to uniform activation weights
- no_moe_specialization: MoE expert router forced to uniform expert probabilities
- depth1: single-loop baseline with recurrent-depth machinery disabled

This is an engineering ablation, not a causal production-scale claim.

## Results

| Experiment | Best validation loss | Final loop losses |
|---|---:|---|
| full | 3.5747 | 3.5773, 3.5786, 3.5790, 3.5793 |
| uniform attention | 3.5772 | 3.5839, 3.5856, 3.5865, 3.5871 |
| uniform activation | 3.6074 | 3.5775, 3.5785, 3.5790, 3.5793 |
| uniform MoE routing | 3.5806 | 3.5761, 3.5773, 3.5780, 3.5783 |
| depth 1 | 3.3301 | 3.5728 |

## Interpretation

### Overall loss

All runs decreased their training loss. The full model is numerically stable and learns the repository text.

### Depth

Depth is the current major problem. The depth-1 control is substantially better on this short diagnostic run (3.3301 vs 3.5747 best validation). Within the full model, later-loop CE is slightly worse than the first loop and the recurrent states rapidly become almost identical:

state cosine: 0.99961, 0.999998, 0.999998
loop improvements: about -0.00195, -0.00070, -0.00046

The recurrent controller is therefore not yet producing enough genuinely new computation. This is the most important next research target.

### Activation routing

The learned activation mixture is helping relative to a uniform-activation control: 3.5747 vs 3.6074. So token-conditioned activation selection appears useful even in this small run.

The hard routing is highly specialized toward one activation. The dense probabilities remain above the configured floor but hard top-k load is much more concentrated. This is not automatically bad, because the purpose of the tolerance is to avoid population death while preserving specialization. It does mean we should monitor whether that activation preference survives on real language data.

### MoE

Uniform MoE routing is slightly worse than learned routing (3.5806 vs 3.5747), so expert specialization is helping, but only weakly in this tiny run. Hard expert load is still concentrated, so the load-balance/tolerance design should remain under observation.

### Attention mixture

Learned attention routing is slightly better than the uniform-mixture control (3.5772 -> 3.5747). The router gradients are finite and non-zero.

However, hard top-1 attention dispatch still collapses onto a single expert in the final diagnostic batch even though dense probabilities are close to uniform. This reveals an important distinction: a dense probability floor does not prevent top-1 winner-take-all dispatch from picking the same expert repeatedly. The model is learning an attention mixture, but the current sparse dispatch is not yet demonstrably specializing across tokens/loops at the hard routing level.

### Recurrent attention routing improvement

A v11 change gives the attention router separate lightweight output heads per recurrence loop instead of only a shared router plus a loop embedding. This makes loop-specific routing a real parameterized mechanism. After the change, the loop-specific routing distributions became measurably different, and the full model showed positive loop-to-loop improvements on the final diagnostic batch:

+0.0122, +0.0162, +0.0250 CE improvement across loops in the v11 run.

The states still converge strongly (cosines 0.99951, 0.99993, 0.999997), so this is promising but not sufficient proof of useful recurrent depth.

## Generation

The diagnostic byte-level generator produced non-coherent text after only 20 tiny CPU steps. Example:

ntiure ane_  ro"o  ge 9aoeen ma  ^o,it  fonhnt

This is a failure of generation quality at this tiny diagnostic scale, not evidence that the architecture cannot generate language after real training. The useful result of the generation test is that the complete trained model -> logits -> sampling -> byte decode path executes.

## Code fixes in this audit

1. Added a diagnostic mode that can force attention, activation, or MoE routing to a uniform control, so the ablations do not accidentally change the learned router objective.
2. Added loop-specific attention-router heads to make recurrence-aware attention selection structurally stronger.
3. Updated attention-mixture tests for the loop-specific router representation.
4. Preserved the activation probability floor/tolerance implementation.
5. Re-ran the full test suite: 15 passed.
6. Re-ran Python compilation across source, scripts, and tests.

## Bottom line

The strongest positive findings are activation specialization and a small benefit from learned attention/expert routing. The dominant failure is recurrent depth: the model still gets very little genuinely different state from loops 2-4 in this small run, and the depth-1 control remains substantially better.

The next production experiment should therefore focus on making later loops do different useful work, while keeping the current activation/attention/MoE routing infrastructure because the component ablations do not show those systems to be harmful.
