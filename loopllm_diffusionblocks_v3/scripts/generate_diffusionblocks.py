#!/usr/bin/env python3
"""Generate one autoregressive next token using a DiffusionBlocks-trained LoopLM.

This is an experimental inference utility for the recurrent-depth DiffusionBlocks
mode. It denoises a latent next-token state through a small Euler schedule and
then samples the final token. Standard LoopLM generation remains available via
the existing generator for non-diffusion checkpoints.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.loop_transformer import load_checkpoint


def resolve_device(spec: str) -> torch.device:
    if spec == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            return torch.device("xpu")
        try:
            import torch_directml
            return torch_directml.device()
        except (ImportError, RuntimeError):
            return torch.device("cpu")
    if spec == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
        return torch.device("cuda")
    if spec == "xpu":
        if not hasattr(torch, "xpu") or not torch.xpu.is_available():
            raise RuntimeError("XPU requested but torch.xpu.is_available() is False")
        return torch.device("xpu")
    if spec == "dml":
        try:
            import torch_directml
        except ImportError as exc:
            raise RuntimeError("DML requested but torch-directml is not installed") from exc
        return torch_directml.device()
    return torch.device("cpu")


def load_encoder():
    try:
        import tiktoken
        return tiktoken.get_encoding("p50k_base")
    except ImportError as exc:
        raise RuntimeError("Install tiktoken to use text prompts: pip install tiktoken") from exc


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--prompt", required=True)
    p.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "xpu", "dml"))
    p.add_argument("--steps", type=int, default=4)
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if args.steps < 1:
        raise ValueError("--steps must be >= 1")
    if args.max_new_tokens < 1:
        raise ValueError("--max-new-tokens must be >= 1")

    device = resolve_device(args.device)
    ckpt = load_checkpoint(args.checkpoint, device="cpu")
    model = ckpt["model"]
    if not model.cfg.diffusion_blocks:
        raise RuntimeError(
            "Checkpoint was not trained with --diffusion-blocks. "
            "Use the ordinary generator for a standard LoopLM checkpoint."
        )
    model = model.to(device).eval()

    enc = load_encoder()
    ids = torch.tensor([enc.encode(args.prompt)], dtype=torch.long, device=device)
    print(f"device={device} steps={args.steps} prompt_tokens={ids.size(1)}")

    for _ in range(args.max_new_tokens):
        ids = model.diffusion_euler_sample(
            ids,
            num_steps=args.steps,
            temperature=args.temperature,
            top_k=args.top_k,
            seed=args.seed,
        )
        args.seed += 1

    text = enc.decode(ids[0].detach().cpu().tolist())
    print("\n--- GENERATED ---")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
