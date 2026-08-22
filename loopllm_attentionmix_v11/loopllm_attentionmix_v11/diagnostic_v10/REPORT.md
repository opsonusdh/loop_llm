# LoopLLM component diagnosis v9

Corpus bytes: 796,057
Train bytes: 676,648
Val bytes: 119,409

| Experiment | Params | Best val loss | Final val loss | Final loop losses |
|---|---:|---:|---:|---|
| full | 179,070 | 3.5747 | 3.5747 | 3.5773, 3.5786, 3.5790, 3.5793 |
| no_attention | 137,119 | 3.5772 | 3.5772 | 3.5839, 3.5856, 3.5865, 3.5871 |
| no_activation | 179,070 | 3.6074 | 3.6084 | 3.5775, 3.5785, 3.5790, 3.5793 |
| no_moe_specialization | 123,054 | 3.5806 | 3.5806 | 3.5761, 3.5773, 3.5780, 3.5783 |
| depth1 | 173,236 | 3.3301 | 3.3301 | 3.5728 |

## Full model component observations

- Loop improvements: [-0.0019507408142089844, -0.0006988048553466797, -0.00046324729919433594]
- Loop state cosine: [0.9996107220649719, 0.9999980926513672, 0.9999979734420776]
- Recurrent update means: [1.0, 0.9513500332832336, 0.9513500332832336, 0.9513500332832336]
- Activation mean: [0.17041270434856415, 0.0, 0.751203179359436, 0.07838410884141922]
- MoE load: [0.010416666977107525, 0.4895833432674408, 0.5, 0.0]
- Attention routing: [{'layer': 1, 'attention_probs_mean': [0.3427143096923828, 0.3289344608783722, 0.3283511996269226], 'attention_load': [1.0, 0.0, 0.0], 'attention_entropy': 1.0984147787094116, 'attention_probs_by_loop': [[0.3421057164669037, 0.33020371198654175, 0.32769060134887695], [0.3416147530078888, 0.33052924275398254, 0.3278559744358063], [0.3423944413661957, 0.32949593663215637, 0.32810962200164795], [0.3427143096923828, 0.3289344608783722, 0.3283511996269226]]}]
- Auxiliary losses: {'csa': 0.0, 'activation': 0.0, 'moe': 0.0, 'attention_mix': 7.598538650199771e-05, 'refinement': 0.002037518424913287, 'loop_task': 3.337378978729248}
- Gradient norms: total=0.8028, router=0.0427, attention=0.1556

## Generation sample

```text
ntiure ane_��  ro"o	ge 9aoeen ma  ^o,it  fonhnt

```

## Interpretation

Positive contribution is inferred from lower validation loss in the full model versus the ablated control under the same data/seed/steps. This is an engineering ablation, not a causal proof at production scale.
Depth usefulness is checked from per-loop validation losses, loop improvements, recurrent state cosine, and recurrent update means.
Activation/MoE/attention usefulness is checked from ablation deltas, routing distributions, auxiliary losses, entropy/load, and non-zero gradients.