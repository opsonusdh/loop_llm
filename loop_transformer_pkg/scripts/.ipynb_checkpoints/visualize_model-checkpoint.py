#!/usr/bin/env python3
"""Karpathy-style diagnostic visualizer for LoopTransformer.

Works from a normal terminal or from a Jupyter notebook via ``%run``.

Examples
--------
Terminal, save figures:
    python scripts/visualize_model.py \
        --checkpoint /content/drive/MyDrive/loop_llm/latest.pt \
        --data train.bin \
        --prompt "def fibonacci(n):" \
        --seed 42

Notebook / terminal, display without saving:
    python scripts/visualize_model.py \
        --checkpoint latest.pt \
        --data train.bin \
        --inline \
        --seed 42

Notes
-----
* The script never modifies the checkpoint.
* The gradient plots are a fresh diagnostic backward pass on the selected
  batch, not historical gradients from training. Historical gradient norms can
  additionally be read from a training log with --train-log.
* The vocabulary map is a PCA projection of the input token embedding matrix.
  All vocabulary points are plotted when practical; only a bounded subset is
  labeled because rendering 50k labels is not a visualization, it is punishment.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterable, Optional
import types

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import tiktoken

# Make the repository runnable from both `python scripts/...` and a notebook.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from loop_transformer.checkpointing import load_checkpoint  # noqa: E402


# ---------------------------------------------------------------------------
# Small, boring helpers. Boring is good in diagnostics.
# ---------------------------------------------------------------------------


def finite_float(x: Any, default: float = float("nan")) -> float:
    try:
        value = float(x)
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def tensor_from_any(value: Any) -> Optional[torch.Tensor]:
    """Find the first tensor in a nested module output."""
    if torch.is_tensor(value):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            found = tensor_from_any(item)
            if found is not None:
                return found
    if isinstance(value, dict):
        for item in value.values():
            found = tensor_from_any(item)
            if found is not None:
                return found
    return None


def safe_token_text(tokenizer: Any, token_id: int) -> str:
    try:
        text = tokenizer.decode([int(token_id)])
    except Exception:
        return f"<{token_id}>"
    text = text.replace("\\", "\\\\").replace("\n", "↵").replace("\r", "␍").replace("\t", "⇥")
    if not text:
        text = "∅"
    return text[:28]


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        if requested.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(f"Requested {requested}, but CUDA is not available.")
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def module_parameter_norms(model: torch.nn.Module) -> list[tuple[str, float, int]]:
    rows: list[tuple[str, float, int]] = []
    for name, param in model.named_parameters():
        if not param.is_floating_point():
            continue
        rows.append((name, float(param.detach().norm().cpu()), param.numel()))
    rows.sort(key=lambda x: x[1], reverse=True)
    return rows


def grouped_parameter_stats(model: torch.nn.Module) -> dict[str, dict[str, float]]:
    """Group parameters by first path component for a readable dashboard."""
    groups: dict[str, list[torch.Tensor]] = {}
    counts: dict[str, int] = {}
    for name, param in model.named_parameters():
        root = name.split(".", 1)[0]
        groups.setdefault(root, []).append(param.detach().float())
        counts[root] = counts.get(root, 0) + param.numel()

    result: dict[str, dict[str, float]] = {}
    for root, tensors in groups.items():
        flat = torch.cat([t.reshape(-1) for t in tensors])
        result[root] = {
            "params": float(counts[root]),
            "mean": float(flat.mean().cpu()),
            "std": float(flat.std(unbiased=False).cpu()),
            "rms": float(flat.square().mean().sqrt().cpu()),
            "abs_max": float(flat.abs().max().cpu()),
        }
    return result


def read_training_log(path: Optional[Path]) -> dict[str, list[float]]:
    result = {"step": [], "loss": [], "val_loss": [], "grad_norm": []}
    if path is None:
        return result
    if not path.exists():
        raise FileNotFoundError(f"Training log not found: {path}")

    step_re = re.compile(r"step\s+([0-9,]+)")
    loss_re = re.compile(r"\bloss\s+([0-9.eE+-]+)")
    val_re = re.compile(r"val_loss\s+([0-9.eE+-]+)")
    grad_re = re.compile(r"grad_norm\(pre-clip\)=([0-9.eE+-]+)")

    rows: dict[int, dict[str, float]] = {}
    current_step: Optional[int] = None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = step_re.search(line)
        if m:
            current_step = int(m.group(1).replace(",", ""))
            rows.setdefault(current_step, {})

        if current_step is None:
            continue

        row = rows[current_step]
        m = loss_re.search(line)
        if m and "val_loss" not in line:
            row["loss"] = finite_float(m.group(1))
        m = val_re.search(line)
        if m:
            row["val_loss"] = finite_float(m.group(1))
        m = grad_re.search(line)
        if m:
            row["grad_norm"] = finite_float(m.group(1))

    for step in sorted(rows):
        result["step"].append(step)
        result["loss"].append(rows[step].get("loss", float("nan")))
        result["val_loss"].append(rows[step].get("val_loss", float("nan")))
        result["grad_norm"].append(rows[step].get("grad_norm", float("nan")))
    return result


def read_token_data(path: Path, dtype_name: str) -> np.memmap:
    if not path.exists():
        raise FileNotFoundError(path)
    dtype = np.uint16 if dtype_name == "uint16" else np.uint32
    data = np.memmap(path, dtype=dtype, mode="r")
    if len(data) < 2:
        raise ValueError(f"Token file {path} is too small ({len(data)} tokens).")
    return data


def make_batch(
    token_data: Optional[np.memmap],
    tokenizer: Any,
    prompt: str,
    batch_size: int,
    seq_len: int,
    seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, str]:
    rng = random.Random(seed)

    if token_data is not None:
        if len(token_data) <= seq_len:
            raise ValueError(
                f"train data has {len(token_data)} tokens, but seq_len={seq_len}."
            )
        starts = [rng.randrange(0, len(token_data) - seq_len) for _ in range(batch_size)]
        batch = np.stack([np.asarray(token_data[s : s + seq_len], dtype=np.int64) for s in starts])
        idx = torch.from_numpy(batch).to(device=device, dtype=torch.long)
        description = f"random corpus batch (seed={seed}, starts={starts[:8]})"
        return idx, description

    prompt_ids = tokenizer.encode(prompt)
    if len(prompt_ids) < 2:
        raise ValueError("Prompt must encode to at least 2 tokens.")
    needed = max(seq_len, len(prompt_ids))
    ids = prompt_ids[-needed:]
    if len(ids) < seq_len:
        # Repeat the prompt cyclically only for diagnostics; never used for training.
        repeated = (ids * math.ceil(seq_len / len(ids)))[:seq_len]
        ids = repeated
    idx = torch.tensor([ids[-seq_len:]] * batch_size, dtype=torch.long, device=device)
    return idx, "prompt-derived diagnostic batch"


def pca_2d(matrix: torch.Tensor, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Return 2D PCA coordinates and explained variance ratio."""
    x = matrix.detach().float().cpu()
    x = x - x.mean(dim=0, keepdim=True)

    # Exact PCA is fine for 50k x 512 and avoids introducing sklearn as a hard
    # dependency. The covariance matrix is small (<= 512 x 512).
    cov = (x.T @ x) / max(x.shape[0] - 1, 1)
    eigvals, eigvecs = torch.linalg.eigh(cov)
    order = torch.argsort(eigvals, descending=True)
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order[:2]]
    coords = (x @ eigvecs).numpy()
    denom = eigvals.clamp_min(0).sum().item()
    variance = (eigvals[:2].clamp_min(0) / denom).numpy() if denom > 0 else np.zeros(2)
    return coords, variance


def get_embedding_matrix(model: torch.nn.Module) -> torch.Tensor:
    emb = getattr(model, "tok_emb", None)
    if emb is None or not hasattr(emb, "weight"):
        raise AttributeError("Model has no tok_emb.weight embedding matrix.")
    return emb.weight.detach()


def run_forward_probe(
    model: torch.nn.Module,
    idx: torch.Tensor,
    selected_layer: int,
) -> dict[str, Any]:
    """Capture one selected block's activations and per-loop outputs."""
    blocks = getattr(model, "blocks", None)

    if blocks is None or len(blocks) == 0:
        raise AttributeError(
            "Model does not expose model.blocks; cannot select a Transformer layer."
        )

    if not (0 <= selected_layer < len(blocks)):
        raise IndexError(
            f"Layer {selected_layer} outside [0, {len(blocks) - 1}]."
        )

    block = blocks[selected_layer]

    capture: dict[str, Any] = {
        "activation_inputs": [],
        "activation_outputs": [],
        "attention_outputs": [],
    }

    # Save the original bound methods.
    original_forward_attn = block.forward_attn
    original_forward_ffn = block.forward_ffn

    def wrapped_forward_attn(
        self: torch.nn.Module,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        # Call the REAL implementation with exactly the arguments
        # that the model itself supplied.
        output = original_forward_attn(*args, **kwargs)

        value = tensor_from_any(output)
        if value is not None:
            capture["attention_outputs"].append(
                value.detach().float().cpu()
            )

        return output

    def wrapped_forward_ffn(
        self: torch.nn.Module,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        # Capture the actual FFN input.
        if args:
            input_value = tensor_from_any(args[0])
        else:
            input_value = tensor_from_any(kwargs.get("x"))

        if input_value is not None:
            capture["activation_inputs"].append(
                input_value.detach().float().cpu()
            )

        # Call the REAL implementation.
        output = original_forward_ffn(*args, **kwargs)

        value = tensor_from_any(output)
        if value is not None:
            capture["activation_outputs"].append(
                value.detach().float().cpu()
            )

        return output

    # Replace the methods temporarily.
    block.forward_attn = types.MethodType(
        wrapped_forward_attn,
        block,
    )
    block.forward_ffn = types.MethodType(
        wrapped_forward_ffn,
        block,
    )

    try:
        with torch.no_grad():
            loop_logits, lambdas = model.forward(
                idx,
                max_loops=model.cfg.max_loops,
            )
    finally:
        # Absolutely restore the original methods, even if forward crashes.
        block.forward_attn = original_forward_attn
        block.forward_ffn = original_forward_ffn

    if not capture["activation_inputs"]:
        raise RuntimeError(
            "Selected layer produced no FFN input activation. "
            "The model did not call block.forward_ffn() during the probe."
        )

    # One FFN input is captured for every loop that actually executes.
    activation = capture["activation_inputs"][-1]

    logits_cpu = [
        x.detach().float().cpu()
        for x in loop_logits
    ]

    lambda_cpu = [
        x.detach().float().cpu()
        for x in lambdas
    ]

    # Compute per-loop CE loss on the same diagnostic batch.
    step_losses: list[float] = []

    targets = idx[:, 1:].cpu()

    for logits in logits_cpu:
        ce = F.cross_entropy(
            logits[:, :-1, :].reshape(-1, logits.size(-1)),
            targets.reshape(-1),
        )
        step_losses.append(float(ce.item()))

    return {
        "activation": activation,
        "activation_inputs_by_loop": capture["activation_inputs"],
        "activation_outputs_by_loop": capture["activation_outputs"],
        "attention_outputs_by_loop": capture["attention_outputs"],
        "loop_logits": logits_cpu,
        "lambdas": lambda_cpu,
        "loop_losses": step_losses,
    }


def compute_probe_gradients(
    model: torch.nn.Module,
    idx: torch.Tensor,
) -> dict[str, Any]:
    """One deterministic diagnostic backward pass; model weights are restored."""
    was_training = model.training
    model.train()
    model.zero_grad(set_to_none=True)
    try:
        # Explicit max_loops gives a stable comparison across all layers.
        loss, _ = model.compute_loss(idx, max_loops=model.cfg.max_loops)
        loss.backward()
        layer_grads: list[tuple[str, float]] = []
        total_sq = 0.0
        for name, param in model.named_parameters():
            if param.grad is None:
                continue
            norm = float(param.grad.detach().float().norm().cpu())
            layer_grads.append((name, norm))
            total_sq += norm * norm
        total = math.sqrt(total_sq)
    finally:
        model.zero_grad(set_to_none=True)
        model.train(was_training)

    return {"loss": float(loss.detach().cpu()), "total": total, "params": layer_grads}



# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def new_fig(title: str, figsize: tuple[float, float] = (10, 6)):
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    ax.set_title(title, loc="left", fontsize=13, fontweight="bold")
    ax.grid(alpha=0.18)
    return fig, ax


def plot_train_history(history: dict[str, list[float]], save: bool, out_dir: Path) -> Optional[Path]:
    if not history["step"]:
        return None
    fig, ax = new_fig("Training dynamics: loss + gradient norm")
    steps = np.asarray(history["step"])
    loss = np.asarray(history["loss"], dtype=float)
    val = np.asarray(history["val_loss"], dtype=float)
    grad = np.asarray(history["grad_norm"], dtype=float)

    ax.plot(steps, loss, label="train loss", linewidth=1.3)
    if np.isfinite(val).any():
        ax.plot(steps[np.isfinite(val)], val[np.isfinite(val)], label="val loss", linewidth=1.6)
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax2 = ax.twinx()
    if np.isfinite(grad).any():
        ax2.plot(steps[np.isfinite(grad)], grad[np.isfinite(grad)], label="grad norm", alpha=0.55, linewidth=1.0)
        ax2.set_ylabel("pre-clip grad norm")
    ax.legend(loc="upper right")
    return save_fig(fig, "01_training_dynamics", save, out_dir)


def plot_parameter_groups(stats: dict[str, dict[str, float]], save: bool, out_dir: Path) -> Optional[Path]:
    names = list(stats)
    rms = [stats[n]["rms"] for n in names]
    params = [stats[n]["params"] for n in names]
    fig, ax = new_fig("Parameter health by top-level module", (10, 6))
    y = np.arange(len(names))
    ax.barh(y, rms)
    ax.set_yticks(y, names)
    ax.set_xlabel("parameter RMS")
    ax2 = ax.twiny()
    ax2.plot(params, y, "o", alpha=0.65)
    ax2.set_xlabel("parameter count")
    return save_fig(fig, "02_parameter_groups", save, out_dir)


def plot_parameter_hist(model: torch.nn.Module, save: bool, out_dir: Path) -> Optional[Path]:
    pieces = []
    for param in model.parameters():
        if param.is_floating_point():
            flat = param.detach().float().reshape(-1).cpu()
            if flat.numel() > 300_000:
                step = max(1, flat.numel() // 300_000)
                flat = flat[::step]
            pieces.append(flat)
    values = torch.cat(pieces).numpy() if pieces else np.array([])
    fig, ax = new_fig("Global parameter distribution")
    if values.size:
        ax.hist(values, bins=200, density=True)
        ax.set_yscale("log")
    ax.set_xlabel("weight value")
    ax.set_ylabel("density (log scale)")
    return save_fig(fig, "03_parameter_distribution", save, out_dir)


def plot_gradients(probe: dict[str, Any], save: bool, out_dir: Path) -> Optional[Path]:
    rows = sorted(probe["params"], key=lambda x: x[1], reverse=True)[:40]
    names = [n for n, _ in rows][::-1]
    vals = [v for _, v in rows][::-1]
    fig, ax = new_fig("Fresh diagnostic gradient norms: top 40 tensors", (11, 9))
    ax.barh(np.arange(len(names)), vals)
    ax.set_yticks(np.arange(len(names)), names, fontsize=7)
    ax.set_xlabel("L2 gradient norm")
    ax.set_xscale("log")
    return save_fig(fig, "04_gradient_norms", save, out_dir)


def plot_activation(activation: torch.Tensor, layer: int, save: bool, out_dir: Path) -> list[Path]:
    # [B, T, D] is the expected TransformerBlock activation layout. Be tolerant
    # of extra singleton dimensions and use the last dimension as features.
    x = activation
    while x.ndim > 3:
        x = x.reshape(-1, x.shape[-2], x.shape[-1])
    if x.ndim == 2:
        x = x.unsqueeze(0)
    flat = x.reshape(-1, x.shape[-1])

    figs: list[Path] = []
    fig, ax = new_fig(f"Layer {layer} activation distribution")
    vals = flat.numpy().ravel()
    if vals.size > 500_000:
        vals = vals[:: max(1, vals.size // 500_000)]
    ax.hist(vals, bins=200, density=True)
    ax.set_xlabel("activation")
    ax.set_ylabel("density")
    p = save_fig(fig, f"05_layer_{layer}_activation_distribution", save, out_dir)
    if p:
        figs.append(p)

    mean = flat.mean(dim=0).numpy()
    std = flat.std(dim=0, unbiased=False).numpy()
    dead = float(np.mean(std < 1e-6))
    order = np.argsort(std)
    show = order[-128:]
    fig, ax = new_fig(f"Layer {layer}: feature-wise activation mean ± std | near-constant={dead:.2%}")
    ax.plot(show, mean[show], linewidth=1.0, label="mean")
    ax.fill_between(show, mean[show] - std[show], mean[show] + std[show], alpha=0.25, label="± std")
    ax.set_xlabel("feature index (highest-variance 128 shown)")
    ax.set_ylabel("activation")
    ax.legend()
    p = save_fig(fig, f"06_layer_{layer}_feature_stats", save, out_dir)
    if p:
        figs.append(p)
    return figs


def plot_loops(probe_result: dict[str, Any], save: bool, out_dir: Path) -> list[Path]:
    paths: list[Path] = []
    loop_losses = probe_result["loop_losses"]
    lambdas = probe_result["lambdas"]
    loops = np.arange(1, len(loop_losses) + 1)

    fig, ax = new_fig("Loop behavior on the diagnostic batch")
    ax.plot(loops, loop_losses, marker="o", label="LM loss")
    ax.set_xlabel("loop")
    ax.set_ylabel("cross-entropy")
    ax.set_xticks(loops)
    ax2 = ax.twinx()
    lambda_means = np.array([float(x.mean()) for x in lambdas])
    ax2.plot(loops, lambda_means, marker="s", alpha=0.7, label="mean λ")
    ax2.set_ylabel("exit probability λ")
    p = save_fig(fig, "07_loop_losses_and_exit", save, out_dir)
    if p:
        paths.append(p)

    if len(loop_losses) > 1:
        deltas = np.diff(np.asarray(loop_losses, dtype=float))
        fig, ax = new_fig("Loop-to-loop loss change")
        ax.axhline(0.0, linewidth=1.0)
        ax.bar(np.arange(2, len(loop_losses) + 1), deltas)
        ax.set_xlabel("current loop")
        ax.set_ylabel("L_current − L_previous")
        p = save_fig(fig, "08_loop_loss_deltas", save, out_dir)
        if p:
            paths.append(p)
    return paths


def plot_vocab_map(
    model: torch.nn.Module,
    tokenizer: Any,
    seed: int,
    labels: int,
    max_points: int,
    save: bool,
    out_dir: Path,
) -> Optional[Path]:
    emb = get_embedding_matrix(model)
    vocab_size = emb.shape[0]
    coords, variance = pca_2d(emb, seed)

    # Plot every token unless the vocabulary is extremely large; otherwise keep
    # the map responsive while maintaining deterministic sampling.
    ids = np.arange(vocab_size)
    if vocab_size > max_points:
        rng = np.random.default_rng(seed)
        ids = np.sort(rng.choice(vocab_size, size=max_points, replace=False))

    fig, ax = new_fig(
        f"Vocabulary geometry: PCA(embedding) | PC1={variance[0]:.1%}, PC2={variance[1]:.1%}",
        (12, 10),
    )
    ax.scatter(coords[ids, 0], coords[ids, 1], s=4, alpha=0.22)

    # Deterministic labels: beginning/end plus evenly spaced IDs and tokens with
    # distinctive punctuation/whitespace are useful landmarks.
    label_ids = set(np.linspace(0, vocab_size - 1, max(1, labels), dtype=int).tolist())
    special_candidates = [0, 1, 2, 3, 10, 13, 32, 46, 58, 91, 93, 123, 125, 256, 50256, 50257, 50258, 50259, 50260, 50261, 50262, 50263, 50264, 50265, 50266, 50267, 50268, 50269, 50270, 50271, 50272, 50273, 50274, 50275, 50276, 50277, 50278, 50279, 50280]
    label_ids.update(i for i in special_candidates if 0 <= i < vocab_size)
    for token_id in sorted(label_ids):
        x, y = coords[token_id]
        ax.annotate(safe_token_text(tokenizer, token_id), (x, y), fontsize=7, alpha=0.85)

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    return save_fig(fig, "09_vocab_pca_2d", save, out_dir)


def plot_token_frequency(
    token_data: Optional[np.memmap],
    vocab_size: int,
    tokenizer: Any,
    seed: int,
    labels: int,
    save: bool,
    out_dir: Path,
) -> Optional[Path]:
    if token_data is None:
        return None
    arr = np.asarray(token_data, dtype=np.int64)
    counts = np.bincount(arr, minlength=vocab_size)
    order = np.argsort(counts)[::-1]
    top = order[: min(labels, vocab_size)]

    fig, ax = new_fig("Token frequency: the long tail")
    ranks = np.arange(1, vocab_size + 1)
    sorted_counts = counts[order]
    ax.loglog(ranks, np.maximum(sorted_counts, 1))
    ax.set_xlabel("token rank")
    ax.set_ylabel("count")

    fig2, ax2 = new_fig("Most frequent tokens")
    display_n = min(30, len(top))
    ids = top[:display_n][::-1]
    vals = counts[ids][::-1]
    ax2.barh(np.arange(display_n), vals)
    ax2.set_yticks(np.arange(display_n), [safe_token_text(tokenizer, int(i)) for i in ids], fontsize=8)
    ax2.set_xlabel("count")
    paths = []
    p = save_fig(fig, "10_token_frequency_rank", save, out_dir)
    if p:
        paths.append(p)
    p = save_fig(fig2, "11_top_tokens", save, out_dir)
    if p:
        paths.append(p)
    return paths[-1] if paths else None


def plot_top_predictions(
    probe_result: dict[str, Any],
    tokenizer: Any,
    save: bool,
    out_dir: Path,
) -> Optional[Path]:
    logits = probe_result["loop_logits"][-1][0, -1]
    probs = F.softmax(logits.float(), dim=-1)
    k = min(20, probs.numel())
    values, ids = torch.topk(probs, k)
    values = values.numpy()[::-1]
    ids = ids.numpy()[::-1]

    fig, ax = new_fig("Next-token distribution at the final loop")
    y = np.arange(k)
    ax.barh(y, values)
    ax.set_yticks(y, [safe_token_text(tokenizer, int(i)) for i in ids], fontsize=8)
    ax.set_xlabel("probability")
    return save_fig(fig, "12_top_next_tokens", save, out_dir)


def plot_routing_debug(model: torch.nn.Module, save: bool, out_dir: Path) -> Optional[Path]:
    if not hasattr(model, "get_routing_debug"):
        return None
    try:
        info = model.get_routing_debug()
    except Exception:
        return None
    if not info:
        return None

    rows: list[tuple[str, float]] = []
    for family in ("activation", "moe"):
        block = info.get(family)
        if not block:
            continue
        probs = block.get("mean_probs")
        if probs is None:
            continue
        for i, value in enumerate(probs):
            rows.append((f"{family[0].upper()}{i}", float(value)))
    if not rows:
        return None

    fig, ax = new_fig("Router mean probabilities from the diagnostic forward")
    names = [x[0] for x in rows]
    vals = [x[1] for x in rows]
    ax.bar(np.arange(len(rows)), vals)
    ax.set_xticks(np.arange(len(rows)), names)
    ax.set_ylabel("mean probability")
    return save_fig(fig, "13_routing_probabilities", save, out_dir)


def save_fig(fig: Any, stem: str, save: bool, out_dir: Path) -> Optional[Path]:
    if save:
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{stem}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path
    return None


def show_open_figures() -> None:
    """Render open figures in notebooks; open a GUI window in normal terminals."""
    figure_numbers = plt.get_fignums()

    if not figure_numbers:
        return

    # Jupyter / IPython / VS Code notebook.
    try:
        from IPython import get_ipython
        from IPython.display import display

        shell = get_ipython()

        if shell is not None:
            for number in figure_numbers:
                fig = plt.figure(number)
                display(fig)
                plt.close(fig)
            return

    except ImportError:
        pass

    # Normal desktop terminal with an interactive Matplotlib backend.
    plt.show(block=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Visualize and diagnose a LoopTransformer checkpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--data", type=Path, default=None, help="Optional uint16/uint32 token file for a random diagnostic batch.")
    p.add_argument("--data-dtype", choices=["uint16", "uint32"], default="uint16")
    p.add_argument("--prompt", default="The future of artificial intelligence is")
    p.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--layer", type=int, default=None, help="Transformer block index. Omit to choose one deterministically from --seed.")
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--seq-len", type=int, default=320)
    p.add_argument("--vocab-labels", type=int, default=60)
    p.add_argument("--max-vocab-points", type=int, default=50000)
    p.add_argument("--train-log", type=Path, default=None, help="Optional training log for historical loss/val-loss/grad-norm plots.")
    p.add_argument("--output-dir", type=Path, default=ROOT / "model_viz")
    p.add_argument("--inline", action="store_true", help="Display plots instead of saving them.")
    p.add_argument("--no-backward", action="store_true", help="Skip the diagnostic backward pass (faster, but no fresh gradient plot).")
    return p


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    if args.batch_size < 1 or args.seq_len < 2:
        raise ValueError("batch-size must be >= 1 and seq-len must be >= 2")
    if args.vocab_labels < 1 or args.max_vocab_points < 100:
        raise ValueError("vocab-labels must be >= 1 and max-vocab-points must be >= 100")

    seed_everything(args.seed)
    device = choose_device(args.device)
    print(f"[viz] device: {device}")
    print(f"[viz] checkpoint: {args.checkpoint}")

    ckpt = load_checkpoint(args.checkpoint, device=device)
    model = ckpt["model"]
    model.eval()

    tokenizer = tiktoken.get_encoding("p50k_base")
    vocab_size = int(model.cfg.vocab_size)
    if tokenizer.n_vocab != vocab_size:
        print(f"[viz] WARNING: checkpoint vocab_size={vocab_size}, p50k_base n_vocab={tokenizer.n_vocab}")

    token_data = read_token_data(args.data, args.data_dtype) if args.data else None
    idx, batch_desc = make_batch(
        token_data,
        tokenizer,
        args.prompt,
        args.batch_size,
        args.seq_len,
        args.seed,
        device,
    )
    print(f"[viz] diagnostic batch: {batch_desc}")
    print(f"[viz] shape: {tuple(idx.shape)} | token range: [{int(idx.min())}, {int(idx.max())}]")

    blocks = getattr(model, "blocks", None)
    if blocks is None or len(blocks) == 0:
        raise RuntimeError("Checkpoint model exposes no transformer blocks.")
    n_layers = len(blocks)
    selected_layer = args.layer if args.layer is not None else random.Random(args.seed).randrange(n_layers)
    if not (0 <= selected_layer < n_layers):
        raise ValueError(f"--layer must be in [0, {n_layers - 1}]")
    print(f"[viz] selected layer: {selected_layer}/{n_layers - 1}")

    probe_result = run_forward_probe(model, idx, selected_layer)
    print(f"[viz] loop losses: {[round(x, 5) for x in probe_result['loop_losses']]}")
    print(f"[viz] mean exit lambdas: {[round(float(x.mean()), 5) for x in probe_result['lambdas']]}")

    grad_probe = None
    if not args.no_backward:
        grad_probe = compute_probe_gradients(model, idx)
        print(f"[viz] fresh diagnostic loss: {grad_probe['loss']:.5f}")
        print(f"[viz] fresh diagnostic total grad norm: {grad_probe['total']:.5f}")

    save = not args.inline
    out_dir = args.output_dir
    generated: list[str] = []

    if save:
        out_dir.mkdir(parents=True, exist_ok=True)
        meta = {
            "checkpoint": str(args.checkpoint),
            "step": ckpt.get("step"),
            "seed": args.seed,
            "selected_layer": selected_layer,
            "device": str(device),
            "vocab_size": vocab_size,
            "model_parameters": int(sum(p.numel() for p in model.parameters())),
            "batch_shape": list(idx.shape),
            "batch_description": batch_desc,
            "loop_losses": probe_result["loop_losses"],
            "mean_exit_lambdas": [float(x.mean()) for x in probe_result["lambdas"]],
            "fresh_grad_norm": None if grad_probe is None else grad_probe["total"],
            "config": asdict(model.cfg) if is_dataclass(model.cfg) else repr(model.cfg),
        }
        (out_dir / "00_report.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        generated.append("00_report.json")

    path = plot_train_history(read_training_log(args.train_log), save, out_dir)
    if path: generated.append(path.name)
    path = plot_parameter_groups(grouped_parameter_stats(model), save, out_dir)
    if path: generated.append(path.name)
    path = plot_parameter_hist(model, save, out_dir)
    if path: generated.append(path.name)
    if grad_probe is not None:
        path = plot_gradients(grad_probe, save, out_dir)
        if path: generated.append(path.name)
    for path in plot_activation(probe_result["activation"], selected_layer, save, out_dir):
        generated.append(path.name)
    for path in plot_loops(probe_result, save, out_dir):
        generated.append(path.name)
    path = plot_vocab_map(model, tokenizer, args.seed, args.vocab_labels, args.max_vocab_points, save, out_dir)
    if path: generated.append(path.name)
    path = plot_token_frequency(token_data, vocab_size, tokenizer, args.seed, args.vocab_labels, save, out_dir)
    if path: generated.append(path.name)
    path = plot_top_predictions(probe_result, tokenizer, save, out_dir)
    if path: generated.append(path.name)
    path = plot_routing_debug(model, save, out_dir)
    if path: generated.append(path.name)

    if save:
        print(f"[viz] saved {len(generated)} artifacts to {out_dir}")
        for name in generated:
            print(f"  - {name}")
    else:
        print("[viz] inline mode: displaying figures, nothing written to disk")
        show_open_figures()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
