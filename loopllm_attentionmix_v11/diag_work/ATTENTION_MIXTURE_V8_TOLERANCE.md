# LoopLLM Attention Mixture v8: probability floor + balance tolerance

## Why this change

A pure KL-to-uniform objective is too rigid for a heterogeneous attention mixture. A local attention family may genuinely be more useful for some workloads, while compressed attention may be more useful for others. We therefore keep the balance objective soft, but add two explicit controls:

- `attention_mixture_min_probability` (default `0.10`): a dense routing probability floor. For `N` attention families, the parameterization is
  `p = floor + (1 - N*floor) * softmax(logits)`.
  This preserves the router's ranking while guaranteeing every family retains at least the configured probability mass.
- `attention_mixture_balance_tolerance` (default `0.25`): batch-mean routing probabilities may deviate from uniform by this amount before an additional quadratic penalty activates. The existing KL-to-uniform term remains the main stabilizer, so tolerance does not disable balance learning.

## Why a floor is better than clipping

Hard clipping followed by renormalization can distort the ranking and gradients near the floor. The affine-simplex parameterization above is smooth, sums to one exactly, and keeps the relative ordering induced by the router softmax.

For 3 experts and a 0.10 floor, 0.30 of the total mass is reserved for exploration while 0.70 remains fully learned.

## Important limitation

The floor applies to the **dense routing probabilities**. With `top_k=1`, the actual hard dispatch remains one expert per token, so an expert can still have zero hard token load even though its dense probability is >= 0.10. This is intentional: the user asked for a probability floor, not guaranteed hard routing utilization.

If hard utilization also needs a floor, the next experiment should be a batch-level/expert-choice style dispatch constraint rather than making the dense router more uniform.

## Tests

- `pytest -q` -> **10 passed** after adding two tolerance regression tests.
- Synthetic attention specialization still trains and reports all three attention families.
- A deliberately collapsed router (`bias=[20,-20,-20]`) produces dense probabilities exactly `[0.8, 0.1, 0.1]`, confirming the floor is enforced even under extreme logits.
- A 12-step run through the actual `scripts.train_refined_v2` entry point decreased validation loss from the initial state to `4.6348` at step 8.

## Recommended real-data setting

Keep:

```text
--attention-mixture-num-experts 3
--attention-mixture-top-k 1
--attention-mixture-min-probability 0.10
--attention-mixture-balance-tolerance 0.25
--attention-mixture-balance-weight 0.01
--attention-mixture-diversity-weight 0.0015
```

Do not increase the floor much above 0.10 for the first experiment. With 3 experts, a floor of 0.20 would reserve 60% of the probability mass for exploration and could seriously weaken specialization.
