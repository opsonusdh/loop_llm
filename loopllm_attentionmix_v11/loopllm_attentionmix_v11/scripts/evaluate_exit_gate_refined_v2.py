#!/usr/bin/env python3
"""Evaluate a trained LoopTransformer exit gate against fixed 4-loop inference.

The evaluator uses the model's actual Q-exit rule:
    p_t = lambda_t * survival
    CDF_t = sum_{i<=t} p_i
    exit at the first t where CDF_t >= exit_threshold

For validation batches this intentionally mirrors the current implementation's
batch-level decision, where lambda_t is averaged across the batch. It reports
both the gated loss and the fixed-depth (all 4 loops) loss, plus loop usage,
estimated compute saved, oracle diagnostics, and gate statistics.

Supported devices:
    auto, cpu, cuda, xpu, dml

DML note:
    Checkpoints are loaded on CPU first and only then moved to DirectML. This
    avoids backend-specific tensor deserialization failures during torch.load.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

SCRIPT_PATH = Path(__file__).resolve()
ROOT = SCRIPT_PATH.parents[1] if SCRIPT_PATH.parent.name == "scripts" else SCRIPT_PATH.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from loop_transformer import load_checkpoint

LOG = logging.getLogger("exit_gate_eval")


def resolve_device(spec: str) -> torch.device:
    """Resolve an explicit device or choose the best available backend."""
    if spec == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            return torch.device("xpu")
        try:
            import torch_directml  # type: ignore

            return torch_directml.device()
        except Exception:
            return torch.device("cpu")

    if spec == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested, but torch.cuda.is_available() is False.")
        return torch.device("cuda")

    if spec == "xpu":
        if not hasattr(torch, "xpu") or not torch.xpu.is_available():
            raise RuntimeError("XPU requested, but torch.xpu.is_available() is False.")
        return torch.device("xpu")

    if spec == "dml":
        try:
            import torch_directml  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "DML requested but torch-directml is not installed."
            ) from exc
        return torch_directml.device()

    return torch.device("cpu")


def load_bin(path: Path, dtype_name: str) -> np.memmap:
    dtype = np.uint16 if dtype_name == "uint16" else np.uint32
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    return np.memmap(path, dtype=dtype, mode="r")


def get_batch(
    data: np.memmap,
    batch_size: int,
    seq_len: int,
    device: torch.device,
    generator: torch.Generator,
) -> torch.Tensor:
    max_start = len(data) - seq_len
    if max_start <= 0:
        raise ValueError(
            f"seq_len ({seq_len}) is >= dataset token count ({len(data)})."
        )

    starts = torch.randint(
        0,
        max_start,
        (batch_size,),
        generator=generator,
        device="cpu",
    )
    batch = torch.stack(
        [
            torch.from_numpy(
                data[int(i) : int(i) + seq_len].astype(np.int64, copy=False)
            )
            for i in starts
        ]
    )
    return batch.to(device, non_blocking=True)


def per_example_loop_losses(
    loop_logits: list[torch.Tensor],
    idx: torch.Tensor,
) -> torch.Tensor:
    """Return [B, loops] token-mean cross entropy."""
    targets = idx[:, 1:].contiguous()
    losses: list[torch.Tensor] = []
    for logits in loop_logits:
        pred = logits[:, :-1].contiguous()
        ce = F.cross_entropy(
            pred.reshape(-1, pred.size(-1)),
            targets.reshape(-1),
            reduction="none",
        ).reshape(idx.size(0), -1).mean(dim=1)
        losses.append(ce)
    return torch.stack(losses, dim=1)


def capture_gate_hidden_and_logits(
    model: torch.nn.Module,
    idx: torch.Tensor,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Run all trained loops and capture the exact hidden states seen by ExitGate."""
    captured: list[torch.Tensor] = []

    def hook(
        _module: torch.nn.Module,
        inputs: tuple[Any, ...],
        _output: Any,
    ) -> None:
        if not inputs:
            raise RuntimeError("ExitGate hook received no hidden-state input.")
        h = inputs[0]
        if not torch.is_tensor(h) or h.ndim != 3:
            raise RuntimeError(
                "ExitGate expected [B,T,D], got "
                f"{type(h).__name__} shape={getattr(h, 'shape', None)}"
            )
        captured.append(h.detach())

    handle = model.exit_gate.register_forward_hook(hook)
    try:
        with torch.no_grad():
            logits, _lambdas = model.forward(
                idx,
                max_loops=model.cfg.max_loops,
            )
    finally:
        handle.remove()

    if len(captured) != model.cfg.max_loops:
        raise RuntimeError(
            f"Expected {model.cfg.max_loops} ExitGate inputs, got {len(captured)}."
        )
    return captured, logits


def gate_lambda_from_hidden(
    exit_gate: torch.nn.Module,
    hidden_states: list[torch.Tensor],
) -> torch.Tensor:
    """Return instantaneous exit probabilities [B,T] from the trained gate."""
    if not hasattr(exit_gate, "proj"):
        raise AttributeError("Model has no exit_gate.proj.")
    outputs = []
    for loop_idx, h in enumerate(hidden_states):
        outputs.append(exit_gate(h, loop_idx=loop_idx))
    return torch.stack(outputs, dim=1)


def q_exit_depth_from_lambdas(
    lambdas: torch.Tensor,
    threshold: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mirror model.generate(): Q-exit selection independently per example."""
    if lambdas.ndim != 2:
        raise ValueError(f"Expected lambdas [B,T], got {tuple(lambdas.shape)}")
    if not (0.0 < threshold <= 1.0):
        raise ValueError("threshold must be in (0, 1]")

    b, loops = lambdas.shape
    survival = torch.ones(b, device=lambdas.device, dtype=lambdas.dtype)
    cdf = torch.zeros_like(survival)
    chosen = torch.full((b,), loops, device=lambdas.device, dtype=torch.long)
    unresolved = torch.ones(b, device=lambdas.device, dtype=torch.bool)

    for t in range(loops):
        if t < loops - 1:
            p_t = survival * lambdas[:, t]
            survival = survival * (1.0 - lambdas[:, t])
        else:
            p_t = survival
        cdf = cdf + p_t
        hit = unresolved & (cdf >= threshold)
        chosen[hit] = t + 1
        unresolved = unresolved & ~hit

    return chosen, cdf


def per_example_expected_depth(lambdas: torch.Tensor) -> torch.Tensor:
    """Diagnostic only: expected exit depth computed independently per example."""
    b, t = lambdas.shape
    survival = torch.ones(b, device=lambdas.device, dtype=lambdas.dtype)
    probs: list[torch.Tensor] = []
    for i in range(t):
        if i < t - 1:
            p = survival * lambdas[:, i]
            survival = survival * (1.0 - lambdas[:, i])
        else:
            p = survival
        probs.append(p)
    p_exit = torch.stack(probs, dim=1)
    depth = torch.arange(1, t + 1, device=lambdas.device, dtype=lambdas.dtype)
    return (p_exit * depth.unsqueeze(0)).sum(dim=1)


def evaluate(
    model: torch.nn.Module,
    data: np.memmap,
    *,
    batch_size: int,
    seq_len: int,
    eval_batches: int,
    device: torch.device,
    rng: torch.Generator,
    threshold: float,
) -> dict[str, Any]:
    model.eval()
    loops = int(model.cfg.max_loops)

    baseline_loss_sum = 0.0
    gated_loss_sum = 0.0
    oracle_loss_sum = 0.0
    gated_tokens = 0
    total_examples = 0
    total_improvement = 0.0
    total_gate_bce_target = 0.0
    batch_depth_sum = 0.0
    per_example_expected_depth_sum = 0.0
    mean_lambda_sum = torch.zeros(loops, dtype=torch.float64)
    selected_counts = [0] * loops
    total_loop_computations = 0
    baseline_loop_computations = 0
    gated_improvement_sum = 0.0
    gated_regret_sum = 0.0
    start_time = time.time()

    for batch_index in range(eval_batches):
        idx = get_batch(data, batch_size, seq_len, device, rng)
        hidden, loop_logits = capture_gate_hidden_and_logits(model, idx)
        losses = per_example_loop_losses(loop_logits, idx)
        lambdas = gate_lambda_from_hidden(model.exit_gate, hidden)

        # Faithful baseline: always use the final trained loop.
        baseline_loss = losses[:, -1]

        # Faithful Q-exit: choose independently for every example, matching
        # model.generate() and the per-example Stage-II training target.
        chosen_depths, _cdf = q_exit_depth_from_lambdas(lambdas, threshold)
        gated_loss = losses.gather(1, (chosen_depths - 1).unsqueeze(1)).squeeze(1)

        # Oracle diagnostics: best loop chosen independently per example.
        oracle_loss = losses.min(dim=1).values

        # Depth-1..T counts under the actual batch-level decision.
        for depth_value in chosen_depths.tolist():
            selected_counts[int(depth_value) - 1] += 1
        batch_depth_sum += float(chosen_depths.float().sum().item())
        total_loop_computations += float(chosen_depths.float().sum().item())
        baseline_loop_computations += loops * batch_size

        mean_lambda_sum += lambdas.detach().double().mean(dim=0).cpu()
        per_example_expected_depth_sum += float(
            per_example_expected_depth(lambdas).detach().sum().item()
        )

        baseline_loss_sum += float(baseline_loss.sum().item())
        gated_loss_sum += float(gated_loss.sum().item())
        oracle_loss_sum += float(oracle_loss.sum().item())
        total_examples += batch_size
        gated_tokens += batch_size * (seq_len - 1)

        # How much useful loss improvement existed over each transition?
        transition_improvements = (losses[:, :-1] - losses[:, 1:]).clamp_min(0.0)
        total_improvement += float(transition_improvements.mean().item()) * batch_size

        # Same target rule used for Stage-II gate training; diagnostic only.
        continue_target = torch.sigmoid(
            50.0 * (transition_improvements - 0.005)
        )
        total_gate_bce_target += float(continue_target.mean().item()) * batch_size

        # Gate regret relative to the per-example best loop.
        gated_regret_sum += float((gated_loss - oracle_loss).sum().item())

        # If gated depth is earlier/later than final loop, sign of this quantity
        # tells us whether the chosen depth helped or hurt relative to L_final.
        gated_improvement_sum += float((baseline_loss - gated_loss).sum().item())

        if (batch_index + 1) == 1 or (batch_index + 1) % max(1, eval_batches // 5) == 0:
            LOG.info(
                "batch %d/%d | baseline=%.4f gated=%.4f mean_depth=%.3f | elapsed=%.1fs",
                batch_index + 1,
                eval_batches,
                float(baseline_loss.mean().item()),
                float(gated_loss.mean().item()),
                float(chosen_depths.float().mean().item()),
                time.time() - start_time,
            )

    baseline = baseline_loss_sum / total_examples
    gated = gated_loss_sum / total_examples
    oracle = oracle_loss_sum / total_examples
    delta = gated - baseline
    relative_delta_pct = 100.0 * delta / max(abs(baseline), 1e-12)
    avg_depth = total_loop_computations / total_examples
    compute_saved_pct = 100.0 * (1.0 - avg_depth / loops)
    oracle_gap = gated - oracle

    return {
        "num_batches": eval_batches,
        "num_examples": total_examples,
        "seq_len": seq_len,
        "max_loops": loops,
        "exit_threshold": threshold,
        "baseline_final_loop_loss": baseline,
        "gated_q_exit_loss": gated,
        "gated_minus_baseline_loss": delta,
        "gated_relative_loss_change_pct": relative_delta_pct,
        "oracle_per_example_best_loop_loss": oracle,
        "gated_minus_oracle_loss": oracle_gap,
        "mean_positive_transition_improvement": total_improvement / total_examples,
        "mean_continue_target": total_gate_bce_target / total_examples,
        "mean_gated_depth": avg_depth,
        "estimated_compute_saved_pct": compute_saved_pct,
        "mean_per_example_expected_depth_diagnostic": per_example_expected_depth_sum / total_examples,
        "mean_gate_lambda_by_loop": (mean_lambda_sum / eval_batches).tolist(),
        "exit_counts_by_loop": selected_counts,
        "exit_rates_by_loop": [c / max(total_examples, 1) for c in selected_counts],
        "loss_improvement_of_gated_vs_final_sum": gated_improvement_sum / total_examples,
        "gated_regret_vs_oracle": gated_regret_sum / total_examples,
    }


def compare_stage1_and_stage2(
    stage1_path: Path,
    gate_path: Path,
) -> dict[str, Any]:
    """Verify that only exit_gate parameters changed between Stage I and Stage II."""
    s1 = load_checkpoint(stage1_path, device="cpu")
    s2 = load_checkpoint(gate_path, device="cpu")
    a = s1["model_state_dict"]
    b = s2["model_state_dict"]

    if a.keys() != b.keys():
        return {
            "compatible_state_keys": False,
            "non_gate_changed": None,
            "reason": "state_dict keys differ",
        }

    max_non_gate_diff = 0.0
    changed_gate = 0
    changed_non_gate = 0
    for name in a:
        xa = a[name].detach().cpu()
        xb = b[name].detach().cpu()
        if xa.shape != xb.shape:
            return {
                "compatible_state_keys": False,
                "non_gate_changed": None,
                "reason": f"shape mismatch for {name}",
            }
        diff = float((xa.float() - xb.float()).abs().max().item())
        is_gate = name.startswith("exit_gate.")
        if diff > 0.0:
            if is_gate:
                changed_gate += 1
            else:
                changed_non_gate += 1
                max_non_gate_diff = max(max_non_gate_diff, diff)

    return {
        "compatible_state_keys": True,
        "gate_tensors_changed": changed_gate,
        "non_gate_tensors_changed": changed_non_gate,
        "max_non_gate_abs_diff": max_non_gate_diff,
        "non_gate_unchanged": changed_non_gate == 0,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gate-checkpoint", type=Path, required=True,
                   help="Stage-II best.pt or latest.pt containing the trained gate")
    p.add_argument("--stage1-checkpoint", type=Path, default=None,
                   help="optional Stage-I checkpoint for an integrity check")
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--data-dtype", choices=("uint16", "uint32"), default="uint16")
    p.add_argument("--batch-size", type=int, default=3)
    p.add_argument("--seq-len", type=int, default=320)
    p.add_argument("--eval-batches", type=int, default=100)
    p.add_argument("--exit-threshold", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", choices=("auto", "cpu", "cuda", "xpu", "dml"), default="auto")
    p.add_argument("--output", type=Path, default=None,
                   help="optional JSON report path")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-5s %(message)s",
    )

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.seq_len <= 1:
        raise ValueError("--seq-len must be > 1")
    if args.eval_batches <= 0:
        raise ValueError("--eval-batches must be positive")
    if not (0.0 < args.exit_threshold <= 1.0):
        raise ValueError("--exit-threshold must be in (0, 1]")
    if not args.gate_checkpoint.exists():
        raise FileNotFoundError(f"Gate checkpoint not found: {args.gate_checkpoint}")

    device = resolve_device(args.device)
    LOG.info("Device: %s", device)
    LOG.info("Gate checkpoint: %s", args.gate_checkpoint)

    data = load_bin(args.data, args.data_dtype)
    LOG.info("Validation tokens: %d", len(data))

    # Load checkpoints on CPU first. This is important for DML-produced files.
    LOG.info("Loading gate checkpoint on CPU...")
    ckpt = load_checkpoint(args.gate_checkpoint, device="cpu")
    model = ckpt["model"].to(device)
    model.eval()

    loops = int(model.cfg.max_loops)
    LOG.info("Model loops: %d", loops)
    LOG.info("Exit threshold: %.3f", args.exit_threshold)
    LOG.info("Trainable gate checkpoint step: %s", ckpt.get("step", "unknown"))

    if "extra" in ckpt:
        extra = ckpt["extra"] or {}
        LOG.info("Checkpoint stage: %s", extra.get("stage", "unknown"))

    if args.stage1_checkpoint is not None:
        if not args.stage1_checkpoint.exists():
            raise FileNotFoundError(
                f"Stage-I checkpoint not found: {args.stage1_checkpoint}"
            )
        integrity = compare_stage1_and_stage2(
            args.stage1_checkpoint,
            args.gate_checkpoint,
        )
        LOG.info(
            "Stage-I/Stage-II integrity: non_gate_unchanged=%s | changed gate tensors=%s | changed non-gate tensors=%s",
            integrity.get("non_gate_unchanged"),
            integrity.get("gate_tensors_changed"),
            integrity.get("non_gate_tensors_changed"),
        )
    else:
        integrity = None

    rng = torch.Generator(device="cpu").manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    start = time.time()
    metrics = evaluate(
        model,
        data,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        eval_batches=args.eval_batches,
        device=device,
        rng=rng,
        threshold=args.exit_threshold,
    )
    elapsed = time.time() - start

    report = {
        "device": str(device),
        "gate_checkpoint": str(args.gate_checkpoint),
        "stage1_checkpoint": str(args.stage1_checkpoint) if args.stage1_checkpoint else None,
        "checkpoint_step": ckpt.get("step"),
        "elapsed_seconds": elapsed,
        "integrity": integrity,
        **metrics,
    }

    print()
    print("=" * 72)
    print("EXIT GATE EVALUATION")
    print("=" * 72)
    print(f"Device:                       {device}")
    print(f"Validation batches:           {metrics['num_batches']}")
    print(f"Batch size:                   {metrics['num_examples'] // metrics['num_batches']}")
    print(f"Sequence length:              {metrics['seq_len']}")
    print(f"Trained max loops:            {metrics['max_loops']}")
    print(f"Q-exit threshold:             {metrics['exit_threshold']:.3f}")
    print()
    print("QUALITY")
    print(f"  Fixed {loops}-loop loss:       {metrics['baseline_final_loop_loss']:.6f}")
    print(f"  Gated loss:                   {metrics['gated_q_exit_loss']:.6f}")
    print(f"  Gated - baseline:             {metrics['gated_minus_baseline_loss']:+.6f}")
    print(f"  Relative change:              {metrics['gated_relative_loss_change_pct']:+.3f}%")
    print(f"  Oracle best-loop loss:        {metrics['oracle_per_example_best_loop_loss']:.6f}")
    print(f"  Gated - oracle:               {metrics['gated_minus_oracle_loss']:+.6f}")
    print()
    print("COMPUTE")
    print(f"  Mean gated depth:             {metrics['mean_gated_depth']:.4f} / {loops}")
    print(f"  Estimated compute saved:      {metrics['estimated_compute_saved_pct']:.2f}%")
    print(f"  Diagnostic expected depth:    {metrics['mean_per_example_expected_depth_diagnostic']:.4f}")
    print()
    print("EXIT DISTRIBUTION")
    for i, (count, rate, lam) in enumerate(
        zip(
            metrics["exit_counts_by_loop"],
            metrics["exit_rates_by_loop"],
            metrics["mean_gate_lambda_by_loop"],
        ),
        start=1,
    ):
        print(
            f"  Loop {i}: exit_count={count:6d}  exit_rate={100.0 * rate:6.2f}%  mean_lambda={lam:.4f}"
        )
    print()
    print("TRAJECTORY")
    print(f"  Mean positive improvement:   {metrics['mean_positive_transition_improvement']:.6f}")
    print(f"  Mean Stage-II continue target:{metrics['mean_continue_target']:.4f}")
    print(f"  Gated regret vs oracle:       {metrics['gated_regret_vs_oracle']:.6f}")
    print()
    if integrity is not None:
        print("CHECKPOINT INTEGRITY")
        print(f"  Non-gate tensors unchanged:  {integrity.get('non_gate_unchanged')}")
        print(f"  Gate tensors changed:        {integrity.get('gate_tensors_changed')}")
        print(f"  Non-gate tensors changed:    {integrity.get('non_gate_tensors_changed')}")
    print()
    print(f"Elapsed: {elapsed:.1f}s ({elapsed / max(metrics['num_batches'], 1):.2f}s/batch)")
    print("=" * 72)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        LOG.info("JSON report saved to %s", args.output)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
