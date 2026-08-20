#!/usr/bin/env python3
"""Stage-II adaptive exit-gate trainer for LoopTransformer.

This trainer starts from a completed Stage-I checkpoint, freezes the entire
language model, and trains only model.exit_gate.proj from realized loop-wise
loss improvements.

Target for transition t -> t+1:
    improvement = max(0, L_t - L_{t+1})
    continue_target = sigmoid(k * (improvement - gamma))
    exit_target = 1 - continue_target

The final loop is always trained toward exit.

The Stage-I checkpoint is never overwritten. A fresh checkpoint directory is
required.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import sys
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

from loop_transformer import load_checkpoint, save_checkpoint


LOG = logging.getLogger("exit_gate")


def load_bin(path: Path, dtype: np.dtype) -> np.memmap:
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
        [torch.from_numpy(data[int(i): int(i) + seq_len].astype(np.int64)) for i in starts]
    )
    return batch.to(device, non_blocking=True)


def per_example_loop_losses(
    loop_logits: list[torch.Tensor],
    idx: torch.Tensor,
) -> torch.Tensor:
    """Return [B, loops] token-mean CE, detached from the model graph."""
    losses: list[torch.Tensor] = []
    targets = idx[:, 1:].contiguous()
    for logits in loop_logits:
        pred = logits[:, :-1].contiguous()
        ce = F.cross_entropy(
            pred.reshape(-1, pred.size(-1)),
            targets.reshape(-1),
            reduction="none",
        ).reshape(idx.size(0), -1).mean(dim=1)
        losses.append(ce)
    return torch.stack(losses, dim=1)


def collect_hidden_states_and_losses(
    model: torch.nn.Module,
    idx: torch.Tensor,
) -> tuple[list[torch.Tensor], torch.Tensor]:
    """Frozen forward: capture h_normed at every loop and per-example losses."""
    captured: list[torch.Tensor] = []

    def hook(_module: torch.nn.Module, inputs: tuple[Any, ...], _output: Any) -> None:
        if not inputs:
            raise RuntimeError("ExitGate hook received no hidden-state input.")
        h = inputs[0]
        if not torch.is_tensor(h) or h.ndim != 3:
            raise RuntimeError(
                f"ExitGate expected hidden state [B,T,D], got {type(h).__name__}"
                f" with shape {getattr(h, 'shape', None)}."
            )
        captured.append(h.detach())

    handle = model.exit_gate.register_forward_hook(hook)
    try:
        with torch.no_grad():
            loop_logits, _ = model.forward(idx, max_loops=model.cfg.max_loops)
    finally:
        handle.remove()

    if len(captured) != model.cfg.max_loops:
        raise RuntimeError(
            f"Expected {model.cfg.max_loops} exit-gate hidden states, got {len(captured)}."
        )

    losses = per_example_loop_losses(loop_logits, idx).detach()
    return captured, losses


def gate_logits_from_hidden(
    exit_gate: torch.nn.Module,
    hidden_states: list[torch.Tensor],
) -> torch.Tensor:
    """Evaluate only the trainable exit gate; returns instantaneous exit logits [B,T]."""
    if not hasattr(exit_gate, "proj"):
        raise AttributeError("Expected model.exit_gate.proj for Stage-II training.")

    pooled = torch.stack([h.mean(dim=1) for h in hidden_states], dim=1)  # [B,T,D]
    return exit_gate.proj(pooled).squeeze(-1)


def make_exit_targets(
    loop_losses: torch.Tensor,
    sharpness: float,
    margin: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build [B,T] exit targets, continuation targets, and improvements."""
    if loop_losses.ndim != 2 or loop_losses.size(1) < 2:
        raise ValueError("Need at least two loop losses to train an adaptive exit gate.")

    improvements = (loop_losses[:, :-1] - loop_losses[:, 1:]).clamp_min(0.0)
    continue_targets = torch.sigmoid(sharpness * (improvements - margin))

    # λ_t is instantaneous probability of exiting at loop t.
    exit_targets = torch.empty_like(loop_losses)
    exit_targets[:, :-1] = 1.0 - continue_targets
    exit_targets[:, -1] = 1.0  # final loop must absorb all remaining mass
    return exit_targets, continue_targets, improvements


def expected_exit_depth(exit_probs: torch.Tensor) -> torch.Tensor:
    """Expected loop number under instantaneous exit probabilities λ_t."""
    b, t = exit_probs.shape
    survival = torch.ones(b, device=exit_probs.device, dtype=exit_probs.dtype)
    probs: list[torch.Tensor] = []
    for i in range(t):
        if i < t - 1:
            p = survival * exit_probs[:, i]
            survival = survival * (1.0 - exit_probs[:, i])
        else:
            p = survival
        probs.append(p)
    p_exit = torch.stack(probs, dim=1)
    depth = torch.arange(1, t + 1, device=exit_probs.device, dtype=exit_probs.dtype)
    return (p_exit * depth.unsqueeze(0)).sum(dim=1).mean()


def freeze_except_exit_gate(model: torch.nn.Module) -> list[torch.nn.Parameter]:
    trainable: list[torch.nn.Parameter] = []
    for name, param in model.named_parameters():
        is_gate = name.startswith("exit_gate.proj.")
        param.requires_grad_(is_gate)
        if is_gate:
            trainable.append(param)

    if not trainable:
        raise RuntimeError("No exit_gate.proj parameters found to train.")
    return trainable


def assert_frozen(model: torch.nn.Module) -> None:
    bad = [name for name, p in model.named_parameters() if p.requires_grad and not name.startswith("exit_gate.proj.")]
    if bad:
        raise AssertionError(f"Non-gate parameters are trainable: {bad[:10]}")


def save_gate_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    args: argparse.Namespace,
    best_val: float | None,
) -> None:
    extra = {
        "stage": "stage2_exit_gate",
        "base_checkpoint": str(args.checkpoint),
        "gate_only": True,
        "sharpness": args.sharpness,
        "margin": args.margin,
        "best_val_gate_bce": best_val,
    }
    save_checkpoint(path, model, optimizer=optimizer, step=step, extra=extra)


def evaluate(
    model: torch.nn.Module,
    data: np.memmap,
    batch_size: int,
    seq_len: int,
    device: torch.device,
    generator: torch.Generator,
    batches: int,
    sharpness: float,
    margin: float,
) -> dict[str, float]:
    was_training = model.training
    model.eval()
    total_bce = 0.0
    total_depth = 0.0
    total_improvement = 0.0
    count = 0

    try:
        for _ in range(batches):
            idx = get_batch(data, batch_size, seq_len, device, generator)
            hidden, losses = collect_hidden_states_and_losses(model, idx)
            targets, _continue, improvements = make_exit_targets(losses, sharpness, margin)

            logits = gate_logits_from_hidden(model.exit_gate, hidden)
            bce = F.binary_cross_entropy_with_logits(logits, targets)
            probs = torch.sigmoid(logits)

            total_bce += float(bce.item())
            total_depth += float(expected_exit_depth(probs).item())
            total_improvement += float(improvements.mean().item())
            count += 1
    finally:
        model.train(was_training)

    return {
        "gate_bce": total_bce / max(1, count),
        "expected_exit_depth": total_depth / max(1, count),
        "mean_positive_improvement": total_improvement / max(1, count),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--train-data", type=Path, required=True)
    p.add_argument("--val-data", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--data-dtype", choices=("uint16", "uint32"), default="uint16")
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--seq-len", type=int, default=320)
    p.add_argument("--max-steps", type=int, default=2000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--sharpness", type=float, default=50.0)
    p.add_argument("--margin", type=float, default=0.005)
    p.add_argument("--eval-interval", type=int, default=100)
    p.add_argument("--checkpoint-interval", type=int, default=100)
    p.add_argument("--eval-batches", type=int, default=25)
    p.add_argument("--log-interval", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "xpu"),
        default="auto",
        help="device to use; auto selects CUDA, then XPU, then CPU",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="resume Stage-II from output-dir/latest.pt",
    )
    p.add_argument("--colab", action="store_true")
    return p.parse_args()



def resolve_device(spec: str) -> torch.device:
    """Resolve an explicit or automatic execution device."""
    if spec == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            return torch.device("xpu")
        return torch.device("cpu")

    if spec == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA requested, but torch.cuda.is_available() is False."
            )
        return torch.device("cuda")

    if spec == "xpu":
        if not hasattr(torch, "xpu") or not torch.xpu.is_available():
            raise RuntimeError(
                "XPU requested, but torch.xpu.is_available() is False."
            )
        return torch.device("xpu")

    return torch.device("cpu")


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-5s %(message)s",
    )

    if args.max_steps <= 0:
        raise ValueError("--max-steps must be positive")
    if args.seq_len <= 1:
        raise ValueError("--seq-len must be > 1")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.lr <= 0:
        raise ValueError("--lr must be positive")
    if args.sharpness <= 0:
        raise ValueError("--sharpness must be positive")
    if args.margin < 0:
        raise ValueError("--margin must be non-negative")

    device = resolve_device(args.device)
    LOG.info("Device: %s", device)

    dtype = np.uint16 if args.data_dtype == "uint16" else np.uint32
    train_data = load_bin(args.train_data, dtype)
    val_data = load_bin(args.val_data, dtype)

    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if args.output_dir.resolve() == args.checkpoint.parent.resolve():
        raise ValueError("Stage-II output directory must differ from the Stage-I checkpoint directory.")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    stage2_path = output_dir / "latest.pt"

    # Always initialize the model from the Stage-I checkpoint first.
    LOG.info("Loading Stage-I checkpoint: %s", args.checkpoint)
    base_ckpt = load_checkpoint(args.checkpoint, device=device)
    model = base_ckpt["model"]
    model.eval()

    if not hasattr(model, "exit_gate") or not hasattr(model.exit_gate, "proj"):
        raise AttributeError("Checkpoint model does not expose exit_gate.proj.")

    trainable = freeze_except_exit_gate(model)
    assert_frozen(model)

    optimizer = torch.optim.AdamW(
        trainable,
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )

    start_step = 1
    best_val: float | None = None

    if args.resume:
        if not stage2_path.exists():
            raise FileNotFoundError(
                f"--resume was requested, but Stage-II checkpoint does not exist: "
                f"{stage2_path}"
            )

        LOG.info("Resuming Stage-II checkpoint: %s", stage2_path)

        # Load the Stage-II checkpoint WITHOUT attaching its optimizer yet.
        resume_ckpt = load_checkpoint(
            stage2_path,
            device=device,
        )

        extra = resume_ckpt.get("extra", {})
        if extra.get("stage") != "stage2_exit_gate":
            raise ValueError(
                f"{stage2_path} is not a Stage-II exit-gate checkpoint."
            )

        # Replace the model first.
        model = resume_ckpt["model"]
        model.eval()

        # Now rebuild the trainable parameter references from the NEW model.
        trainable = freeze_except_exit_gate(model)
        assert_frozen(model)

        # Now construct an optimizer that points at the NEW model parameters.
        optimizer = torch.optim.AdamW(
            trainable,
            lr=args.lr,
            betas=(0.9, 0.95),
            weight_decay=args.weight_decay,
        )

        # Finally restore the optimizer state into the correct parameter objects.
        if "optimizer_state_dict" not in resume_ckpt:
            raise ValueError(
                f"{stage2_path} has no optimizer_state_dict; "
                "exact Stage-II resume is impossible."
            )

        optimizer.load_state_dict(resume_ckpt["optimizer_state_dict"])

        start_step = int(resume_ckpt["step"]) + 1
        best_val = extra.get("best_val_gate_bce")

        LOG.info(
            "Resumed Stage-II at step %d (best_val_gate_bce=%s)",
            start_step,
            "none" if best_val is None else f"{best_val:.6f}",
        )

        LOG.info(
            "Resumed Stage-II at step %d (best_val_gate_bce=%s)",
            start_step,
            "none" if best_val is None else f"{best_val:.6f}",
        )
    else:
        LOG.info("Starting fresh Stage-II gate training from Stage-I.")

    LOG.info("Stage-I step: %s", base_ckpt.get("step", "unknown"))
    LOG.info("Loops: %d", model.cfg.max_loops)
    LOG.info("Trainable parameters: %d", sum(p.numel() for p in trainable))
    LOG.info(
        "Gate target: sigmoid(%g * (improvement - %g))",
        args.sharpness,
        args.margin,
    )
    
    (output_dir / "stage2_config.json").write_text(
        json.dumps(vars(args), indent=2, default=str),
        encoding="utf-8",
    )

    train_rng = torch.Generator(device="cpu").manual_seed(args.seed)
    val_rng = torch.Generator(device="cpu").manual_seed(args.seed + 1)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


    for step in range(start_step, args.max_steps + 1):
        model.eval()
        idx = get_batch(train_data, args.batch_size, args.seq_len, device, train_rng)

        # Frozen trajectory. No graph reaches the Transformer.
        hidden_states, loop_losses = collect_hidden_states_and_losses(model, idx)
        hidden_states = [h.detach() for h in hidden_states]
        loop_losses = loop_losses.detach()

        targets, continue_targets, improvements = make_exit_targets(
            loop_losses,
            args.sharpness,
            args.margin,
        )

        # Only exit_gate.proj is trainable here.
        if args.warmup_steps > 0:
            warm = min(1.0, step / args.warmup_steps)
            lr_now = args.lr * warm
        else:
            lr_now = args.lr
        for group in optimizer.param_groups:
            group["lr"] = lr_now

        optimizer.zero_grad(set_to_none=True)
        gate_logits = gate_logits_from_hidden(model.exit_gate, hidden_states)
        gate_bce = F.binary_cross_entropy_with_logits(gate_logits, targets)
        gate_bce.backward()

        grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable, 1.0).item())
        optimizer.step()

        with torch.no_grad():
            probs = torch.sigmoid(gate_logits)
            depth = float(expected_exit_depth(probs).item())
            mean_imp = float(improvements.mean().item())
            mean_continue = float(continue_targets.mean().item())
            mean_exit = float(probs.mean().item())

        if step == 1 or step % args.log_interval == 0:
            LOG.info(
                "step %6d  gate_bce %.6f  lr %.3e  grad %.4f  expected_exit_depth %.3f",
                step,
                float(gate_bce.item()),
                lr_now,
                grad_norm,
                depth,
            )
            LOG.info(
                "  mean loop losses: %s",
                ", ".join(f"{x:.4f}" for x in loop_losses.mean(dim=0).tolist()),
            )
            LOG.info(
                "  mean positive improvement %.6f | continue_target %.4f | exit_prob %.4f",
                mean_imp,
                mean_continue,
                mean_exit,
            )

        if step % args.eval_interval == 0 or step == args.max_steps:
            metrics = evaluate(
                model,
                val_data,
                args.batch_size,
                args.seq_len,
                device,
                val_rng,
                args.eval_batches,
                args.sharpness,
                args.margin,
            )
            LOG.info(
                "step %6d  val_gate_bce %.6f  val_expected_exit_depth %.3f  val_mean_positive_improvement %.6f",
                step,
                metrics["gate_bce"],
                metrics["expected_exit_depth"],
                metrics["mean_positive_improvement"],
            )

            if best_val is None or metrics["gate_bce"] < best_val:
                best_val = metrics["gate_bce"]
                save_gate_checkpoint(
                    output_dir / "best.pt",
                    model,
                    optimizer,
                    step,
                    args,
                    best_val,
                )
                LOG.info("Saved best gate checkpoint -> %s", output_dir / "best.pt")

        if step % args.checkpoint_interval == 0 or step == args.max_steps:
            save_gate_checkpoint(
                output_dir / "latest.pt",
                model,
                optimizer,
                step,
                args,
                best_val,
            )
            LOG.info("Checkpoint saved -> %s", output_dir / "latest.pt")

    LOG.info("Stage-II gate training complete.")
    LOG.info("Best gate checkpoint: %s", output_dir / "best.pt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
