# LoopLLM component diagnosis v9

Corpus bytes: 962,116
Train bytes: 817,798
Val bytes: 144,318

| Experiment | Params | Best val loss | Final val loss | Final loop losses |
|---|---:|---:|---:|---|
| full | 179,439 | 3.3141 | 3.3141 | 3.3045, 3.3071, 3.3091, 3.3095 |
| no_attention | 137,119 | 3.2766 | 3.2766 | 3.2873, 3.2843, 3.2841, 3.2841 |
| no_activation | 179,439 | 3.2923 | 3.2923 | 3.3044, 3.3055, 3.3064, 3.3065 |
| no_moe_specialization | 123,423 | 3.2950 | 3.2950 | 3.3101, 3.2892, 3.2871, 3.2870 |
| depth1 | 173,236 | 3.0377 | 3.0377 | 3.2549 |

## Full model component observations

- Loop improvements: [0.012244701385498047, 0.016245365142822266, 0.025043964385986328]
- Loop state cosine: [0.9995126724243164, 0.9999295473098755, 0.9999966025352478]
- Recurrent update means: [1.0, 0.9506880640983582, 0.9506880640983582, 0.9506880640983582]
- Activation mean: [0.12963372468948364, 0.2217358946800232, 0.6461769938468933, 0.0024533874820917845]
- MoE load: [0.0052083334885537624, 0.0, 0.4895833432674408, 0.5052083134651184]
- Attention routing: [{'layer': 1, 'attention_probs_mean': [0.3170086443424225, 0.3340894281864166, 0.3489018976688385], 'attention_load': [0.0, 0.0, 1.0], 'attention_entropy': 1.0978463888168335, 'attention_probs_by_loop': [[0.32772600650787354, 0.307847261428833, 0.36442673206329346], [0.3176673650741577, 0.33272507786750793, 0.34960755705833435], [0.31398454308509827, 0.33793196082115173, 0.34808358550071716], [0.3170086443424225, 0.3340894281864166, 0.3489018976688385]]}]
- Auxiliary losses: {'csa': 0.0, 'activation': 0.0, 'moe': 0.0, 'attention_mix': 7.840091711841524e-05, 'refinement': 0.0014328922843560576, 'loop_task': 3.076153516769409}
- Gradient norms: total=1.8469, router=0.0230, attention=0.1585

## Generation sample

```text
eel  t 
,)hafue
iu al )e rs�   a l0;   
claeos,�
```

## Interpretation

Positive contribution is inferred from lower validation loss in the full model versus the ablated control under the same data/seed/steps. This is an engineering ablation, not a causal proof at production scale.
Depth usefulness is checked from per-loop validation losses, loop improvements, recurrent state cosine, and recurrent update means.
Activation/MoE/attention usefulness is checked from ablation deltas, routing distributions, auxiliary losses, entropy/load, and non-zero gradients.