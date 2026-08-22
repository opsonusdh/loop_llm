# LoopLLM Attention Mixture v6

## Why

The shared recurrent core previously exposed essentially the same attention field to every loop. Weight sharing is intentional in a LoopLM, but if the attention mechanism itself cannot change its receptive-field behavior with loop depth, later loops can repeatedly read almost the same information.

This revision adds a **loop- and content-routed mixture of attention families** in deeper layers while retaining the original SWA/CSA/HCA stack elsewhere.

## Architecture

- Layers before `attention_mixture_start_layer` remain unchanged.
- Mixed layers expose two compressed attention experts:
  - even layers: CSA + HCA
  - odd layers: HCA + CSA
- A lightweight O(T*d) context-attention router sees token content plus a learned loop embedding.
- A per-loop scale/bias gives the router an explicit depth-dependent degree of freedom.
- `top_k=1` uses a straight-through sparse mixture in the forward pass.
- Experts are currently all evaluated and then sparsely mixed. This is intentional for v6 because it gives a stable research baseline; it is **not yet a compute-saving sparse-expert implementation**.
- A weak diversity term discourages all recurrent loops from collapsing onto the same attention-family distribution.

## Routing of activations and FFN experts

The existing activation and MoE routers are upgraded with the same idea: an O(T*d) attention pool provides sequence context before the token-level routing logits are computed. The old token-only linear route is retained as the main path, and the context path is zero-initialized so the feature starts as an identity-preserving extension.

## Information-density hypothesis

The goal is not simply "more attention parameters." The hypothesis is that recurrent depth can change *which information scale it retrieves*:

- early computation can prefer local or relatively detailed compressed views;
- later computation can shift toward a different compressed field and refine long-range structure;
- the same hidden token can therefore be transformed by a different information-selection policy at each loop.

This is consistent with work showing that learned, content-based attention routing can specialize attention patterns (MoSA) and that explicit attention-head routing can make attention itself conditional computation (MoA). RecurrentGPT independently argues that depth sharing benefits from explicit recurrent modulation rather than a completely uniform shared transformation.

## Important trade-off

The current v6 mixture is a **capacity/information-density experiment**, not an optimized sparse-compute design. All candidate attention experts are evaluated before the top-k mixing weights are applied. Therefore compute and activation memory rise in mixed layers even when `top_k=1`.

This is deliberate for the first ablation: we want to determine whether attention specialization improves recurrent depth *before* introducing a second source of error from token dispatching/conditional kernels.

## Verification

`pytest -q` -> 7 passed.

`python -m compileall -q src scripts` -> pass.

A 9-step synthetic recurrent run with the full CLI succeeded and showed:

- training loss: 3.8401 -> 3.1205
- validation at step 6: 2.759 (via the logged run's validation)
- finite CSA auxiliary loss and MoE/activation diagnostics
- attention mixture routing active on the mixed layer
- router load and entropy reported per layer
- per-loop attention routing diagnostics printed

A 30-step direct structured synthetic run showed training/validation improving while the attention mixture router moved away from a pure preferred-expert prior.

A diffusion-mode unit smoke test with the mixture enabled also produced finite loss/gradients.

## Interpretation of the synthetic run

The short run did **not** prove that more loops are always better. In fact, on the periodic synthetic task, loop 2 was often the best depth and later loops slightly worsened the loss. That is useful: it means the mixture is not being treated as evidence of success merely because the model trains.

The important new thing is that the model now has a real mechanism to change its information field by token and by loop instead of forcing every recurrent pass through the same attention family.

## Recommended real experiment

Keep the mixture enabled only in deeper layers first:

- `--attention-mixture`
- `--attention-mixture-top-k 1`
- `--attention-mixture-start-layer 4`
- `--attention-mixture-balance-weight 0.01`
- `--attention-mixture-diversity-weight 0.001`

Do not immediately enable mixture routing in all six layers. It would multiply attention compute without first proving that the extra diversity is useful.


## v7 attention families

The attention mixture now routes among three distinct information fields: SWA (local), CSA (sparse compressed long-range), and HCA (highly compressed long-range). The router is conditioned on token content, attended sequence context, and loop identity.

The diversity auxiliary uses graph-connected routing distributions; diagnostics use detached copies.
