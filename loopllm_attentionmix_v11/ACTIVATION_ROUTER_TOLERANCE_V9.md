# Activation Router Tolerance v9

The activation router is distinct from the attention router. There are four activation families inside each SwiGLU-style expert. The previous routing loss treated the batch mean as a point that should move toward the uniform distribution `[0.25, 0.25, 0.25, 0.25]`. That is useful for preventing collapse, but it can also erase real specialization.

## New behavior

- `activation_min_probability=0.10`: every dense per-token activation probability is parameterized as
  `p = floor + (1 - 4*floor) * softmax(logits)`.
  Therefore every dense probability is at least `0.10` while preserving a fully learnable simplex for the remaining mass.
- `activation_balance_tolerance=0.25`: the batch-mean activation distribution is allowed to move freely inside a tolerance band around uniform before the balancing loss activates.
- For four activations, uniform is `0.25`, so the tolerance band is `[0.0, 0.50]`; the explicit floor tightens the lower side to `0.10`.
- Sparse top-k dispatch is unchanged. A token may therefore have zero *dispatched* probability for an activation when it is outside the selected top-k. The dense floor is the differentiable specialization constraint. A second penalty keeps the hard batch-average dispatched mass of any activation above `0.10`.
- The non-gradient loss-free balancing bias no longer nudges every batch back toward `0.25`. It only nudges an activation when its hard batch-average falls below `0.10` or above the tolerance upper boundary `0.50`.

This preserves token-level specialization while preventing population-level extinction.

## CLI

`train_refined_v2.py` now accepts:

- `--activation-min-probability 0.10`
- `--activation-balance-tolerance 0.25`

## Verification

- `pytest -q`: 15 passed.
- Dense probability floor test: passed.
- Tolerance dead-zone test: passed.
- Hard dispatched batch-floor penalty test: passed.
- Loss-free bias tolerance test: passed.
- Synthetic attention-specialization run completed with decreasing validation loss and non-uniform activation routing while every dense activation mean remained above 0.10.
