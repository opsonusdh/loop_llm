from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np


def make_data(path: Path, n: int, seq_len: int, vocab: int) -> None:
    base = np.arange(16, dtype=np.uint16)
    seq = np.concatenate([base, (base + 1) % 16]).astype(np.uint16)
    rows = []
    for i in range(n):
        rows.append(np.roll(seq, i % len(seq)))
    arr = np.stack(rows).astype(np.uint16).reshape(-1)
    arr.tofile(path)


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    data_dir = root / "synthetic_diffusionblocks"
    data_dir.mkdir(exist_ok=True)
    train_bin = data_dir / "train.bin"
    val_bin = data_dir / "val.bin"
    cfg = data_dir / "config.json"
    out = data_dir / "ckpt"
    make_data(train_bin, 256, 32, 32)
    make_data(val_bin, 32, 32, 32)
    cfg.write_text(json.dumps({
        "vocab_size": 32,
        "dim": 32,
        "n_layers": 4,
        "n_heads": 4,
        "head_dim": 8,
        "ffn_hidden_dim": 64,
        "rope_dim": 8,
        "max_loops": 4,
        "min_loops": 1,
        "loop_sampling": False,
        "csa_m": 2,
        "csa_top_k": 4,
        "hca_m_prime": 4,
        "sw_window": 8,
        "groups": 4,
        "group_dim": 8,
        "moe_num_shared_experts": 1,
        "moe_num_routed_experts": 2,
        "moe_top_k": 1,
        "activation_top_k": 2,
        "activation_balance_weight": 0.0,
        "moe_aux_loss_weight": 0.0,
        "csa_aux_loss_weight": 0.01,
        "tie_embeddings": True,
        "grad_checkpointing": False,
        "diffusion_blocks": True,
        "diffusion_sigma_min": 0.1,
        "diffusion_sigma_max": 2.0,
        "diffusion_p_mean": -0.2,
        "diffusion_p_std": 0.35,
        "diffusion_cond_dim": 16,
        "diffusion_normalize_embeddings": True,
    }, indent=2), encoding="utf-8")

    cmd = [
        sys.executable, "-m", "scripts.train_refined_v2",
        "--train-data", str(train_bin),
        "--val-data", str(val_bin),
        "--data-dtype", "uint16",
        "--config-json", str(cfg),
        "--diffusion-blocks",
        "--device", "cpu",
        "--dtype", "float32",
        "--batch-size", "8",
        "--seq-len", "32",
        "--max-steps", "60",
        "--warmup-steps", "5",
        "--lr", "0.0003",
        "--weight-decay", "0.0",
        "--checkpoint-dir", str(out),
        "--checkpoint-interval", "15",
        "--eval-interval", "10",
        "--eval-iters", "3",
        "--log-interval", "10",
        "--debug",
        "--seed", "7",
    ]
    print("RUNNING:", " ".join(cmd))
    return subprocess.call(cmd, cwd=root)


if __name__ == "__main__":
    raise SystemExit(main())
