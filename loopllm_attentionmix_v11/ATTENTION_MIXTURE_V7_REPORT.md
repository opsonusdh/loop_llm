# Attention Mixture v7 report

## Design decision

The previous attention mixture only routed between CSA and HCA in deep layers. This revision exposes three genuinely different information fields already present in the model:

1. SWA: recent/local causal context.
2. CSA: sparse long-range compressed context.
3. HCA: highly compressed long-range context.

This is preferable to inventing more activation functions because the information bottleneck under investigation is *what context is retrieved*. Mixture-of-Attention research supports token-dependent selection of attention heads/experts rather than a single fixed pattern.

## Important bug fixed

The loop-diversity auxiliary was previously computed from detached per-loop routing probabilities. It therefore existed numerically but could not change the router. v7 keeps a graph-connected copy for the auxiliary loss and a detached copy for diagnostics.

The router was also initialized with a strong single-expert bias. v7 removes that artificial preference and uses small random router weights, allowing all three attention families to specialize from the start.

## Verification

`pytest -q` -> 8 passed.

Full compileall passed.

The actual training entry point was executed on the synthetic corpus with the new mixture configuration:

- physical parameters: 187,333 on the tiny test model
- three attention experts
- top-k=1
- attention mixture enabled from layer 0
- loss: 3.0663 -> 2.1594 in 5 steps
- recurrent mode was actually entered by the trainer
- checkpoint saved successfully

## 30-step synthetic specialization test

Validation loss by recurrent depth after 30 steps:

- L1: 1.0927
- L2: 1.0433
- L3: 1.0413
- L4: 1.0535

So loops 2 and 3 improved the prediction, while loop 4 slightly regressed on this task. This is not a claim that four loops are always better; it is evidence that the recurrent path has useful but finite depth on the synthetic distribution.

Final attention routing (layer 0): mean probabilities [0.336, 0.354, 0.310], hard load [0.344, 0.417, 0.240].

Final attention routing (layer 1): mean probabilities [0.342, 0.341, 0.318], hard load [0.427, 0.385, 0.188].

Loop-conditioned routing distributions were recorded and differed across loops. The differences were modest rather than forced, which is desirable: specialization should emerge from usefulness, not from an arbitrary orthogonality constraint.

## Interpretation

The mixture is now a real information-field mixture rather than two nearly redundant compressed attention choices. Because the model already instantiated SWA/CSA/HCA modules, adding SWA as an expert changes the routing behavior without creating a fourth copy of the attention parameters.

The current top-k router evaluates all candidate experts and sparsely mixes their outputs. Therefore v7 primarily tests *representation/information specialization*. It does not yet claim end-to-end compute reduction. Conditional execution can be implemented later once routing specialization is established.
