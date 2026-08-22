#!/usr/bin/env python3
"""Training entry point for LoopTransformer.

Reads token-id binary files produced by prepare_data.py, trains with
AdamW + warmup/cosine LR + optional mixed precision + optional gradient
accumulation, checkpoints periodically, resumes cleanly if interrupted,
and evaluates on a held-out split if provided.

Quick start
-----------
    # 1. Tokenize (byte-level default -- no dependencies, just to try things out)
    python scripts/prepare_data.py --input corpus.txt --output data/train.bin

    # 2. Train
    python scripts/train.py --train-data data/train.bin --val-data data/train.val.bin \\
        --vocab-size 256 --dim 512 --n-layers 8 --n-heads 8 --head-dim 64 \\
        --seq-len 512 --batch-size 8 --max-steps 2000 \\
        --checkpoint-dir checkpoints/run1

Resuming after an interruption (Ctrl+C, preemption, crash) is automatic:
point --resume at the same --checkpoint-dir and it picks up from the
latest checkpoint, including optimizer state and step count.

For full architecture control beyond what's exposed as CLI flags, pass
--config-json pointing at a JSON file with any LoopConfig fields; CLI
flags below override matching keys in that file.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import signal
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from src.loop_transformer import LoopConfig, LoopTransformer, load_checkpoint, save_checkpoint

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger("train")


# ======================================================================
# Data
# ======================================================================

def load_bin(path: Path, dtype: np.dtype) -> np.memmap:
    """Memory-mapped read of a token-id binary file (see prepare_data.py).
    memmap means the whole file is NOT loaded into RAM at once -- only the
    pages actually touched by sampled batches are -- so this scales to
    corpora far larger than available memory."""
    return np.memmap(path, dtype=dtype, mode="r")


def get_batch(data: np.memmap, batch_size: int, seq_len: int, device: torch.device) -> torch.Tensor:
    """Sample `batch_size` random contiguous windows of length `seq_len`.
    Returns raw token ids -- LoopTransformer.compute_loss does its own
    input/target shift internally (idx[:-1] vs idx[1:])."""
    max_start = len(data) - seq_len
    if max_start <= 0:
        raise ValueError(
            f"seq_len ({seq_len}) is >= the dataset's token count ({len(data)}). "
            f"Use a shorter --seq-len or a bigger dataset."
        )
    ix = torch.randint(max_start, (batch_size,))
    batch = torch.stack([
        torch.from_numpy(data[i : i + seq_len].astype(np.int64)) for i in ix
    ])
    return batch.to(device, non_blocking=True)


def validate_vocab_range(data: np.memmap, vocab_size: int, name: str) -> None:
    """Fail early when a binary corpus contains ids outside the model vocabulary."""
    if len(data) == 0:
        raise ValueError(f"{name} dataset is empty")
    max_id = int(np.max(data))
    min_id = int(np.min(data))
    if min_id < 0 or max_id >= vocab_size:
        raise ValueError(
            f"{name} token id range [{min_id}, {max_id}] exceeds vocab_size={vocab_size}. "
            "Check the tokenizer/vocab-size pair and data-dtype."
        )


# ======================================================================
# LR schedule
# ======================================================================

def get_lr(step: int, warmup_steps: int, max_steps: int, peak_lr: float, min_lr_ratio: float = 0.1) -> float:
    """Linear warmup, then cosine decay to `min_lr_ratio * peak_lr`.

    Always returns a plain Python float, never a numpy scalar: this value
    gets assigned into optimizer.param_groups[...]['lr'], and a leaked
    numpy.float64 there breaks safe (weights_only=True) checkpoint loading
    later, since numpy scalar types aren't in that loader's default
    allowlist. np.cos below returns numpy.float64, so the explicit
    float(...) cast at the end is load-bearing, not decorative.
    """
    if step < warmup_steps:
        return float(peak_lr * (step + 1) / warmup_steps)
    if step >= max_steps:
        return float(peak_lr * min_lr_ratio)
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    cosine = 0.5 * (1.0 + np.cos(np.pi * progress))
    return float(peak_lr * min_lr_ratio + (peak_lr - peak_lr * min_lr_ratio) * cosine)


# ======================================================================
# Evaluation
# ======================================================================

@torch.no_grad()
def estimate_loss(model: LoopTransformer, data: np.memmap, batch_size: int,
                   seq_len: int, device: torch.device, eval_iters: int) -> float:
    """Average loss over eval_iters random batches. model.eval() means
    compute_loss() uses the fixed cfg.max_loops ceiling (no dynamic
    sampling), so this number is comparable across calls -- see
    LoopConfig.loop_sampling."""
    was_training = model.training
    model.eval()
    losses = []
    for _ in range(eval_iters):
        batch = get_batch(data, batch_size, seq_len, device)
        if model.cfg.diffusion_blocks:
            loss, _ = model.diffusion_blocks_loss(batch)
        else:
            loss, _ = model.compute_loss(batch)
        losses.append(loss.item())
    if was_training:
        model.train()
    return float(np.mean(losses))


# ======================================================================
# Config assembly: JSON file + CLI overrides
# ======================================================================

def build_config(args: argparse.Namespace) -> LoopConfig:
    cfg_dict = {}
    if args.config_json:
        with open(args.config_json) as f:
            cfg_dict.update(json.load(f))

    # CLI flags override the JSON file wherever explicitly provided.
    cli_fields = {
        # Core dimensions
        "vocab_size": args.vocab_size,
        "dim": args.dim,
        "n_layers": args.n_layers,
        "n_heads": args.n_heads,
        "head_dim": args.head_dim,
        "ffn_hidden_dim": args.ffn_hidden_dim,
        "rope_dim": args.rope_dim,
        "embed_dim": args.embed_dim,
        "embed_dim_out": args.embed_dim_out,

        # LoopLM / recurrence
        "max_loops": args.max_loops,
        "min_loops": args.min_loops,
        "beta_entropy": args.beta_entropy,
        "loop_supervision_weight": args.loop_supervision_weight,
        "joint_exit_loss_weight": args.joint_exit_loss_weight,
        "loop_sampling": args.loop_sampling,
        "loop_monotonic_weight": args.loop_monotonic_weight,
        "loop_monotonic_margin": args.loop_monotonic_margin,
        "loop_refinement_weight": args.loop_refinement_weight,
        "loop_refinement_margin": args.loop_refinement_margin,
        "loop_task_weight": args.loop_task_weight,
        "loop_task_mode": args.loop_task_mode,
        "exit_gate_loop_embed_dim": args.exit_gate_loop_embed_dim,
        "loop_sampling_lambda": args.loop_sampling_lambda,
        "recurrent_depth_controller": args.recurrent_depth_controller,
        "recurrent_depth_bottleneck_dim": args.recurrent_depth_bottleneck_dim,
        "recurrent_update_init": args.recurrent_update_init,
        "shortcut_consistency_weight": args.shortcut_consistency_weight,
        "shortcut_consistency_temperature": args.shortcut_consistency_temperature,
        "training_mode": args.training_mode,
        "hybrid_diffusion_probability": args.hybrid_diffusion_probability,

        # DiffusionBlocks recurrent-depth training
        "diffusion_blocks": args.diffusion_blocks,
        "diffusion_sigma_min": args.diffusion_sigma_min,
        "diffusion_sigma_max": args.diffusion_sigma_max,
        "diffusion_p_mean": args.diffusion_p_mean,
        "diffusion_p_std": args.diffusion_p_std,
        "diffusion_sigma_data": args.diffusion_sigma_data,
        "diffusion_loss_weight": args.diffusion_loss_weight,
        "diffusion_normalize_embeddings": args.diffusion_normalize_embeddings,
        "diffusion_cond_dim": args.diffusion_cond_dim,

        # Mixture of attention experts / contextual routing
        "attention_mixture": args.attention_mixture,
        "attention_mixture_top_k": args.attention_mixture_top_k,
        "attention_mixture_balance_weight": args.attention_mixture_balance_weight,
        "attention_mixture_loop_embed_dim": args.attention_mixture_loop_embed_dim,
        "attention_mixture_num_experts": args.attention_mixture_num_experts,
        "attention_mixture_start_layer": args.attention_mixture_start_layer,
        "attention_mixture_diversity_weight": args.attention_mixture_diversity_weight,
        "attention_mixture_min_probability": args.attention_mixture_min_probability,
        "attention_mixture_balance_tolerance": args.attention_mixture_balance_tolerance,

        # DeepSeek-style attention
        "csa_m": args.csa_m,
        "csa_top_k": args.csa_top_k,
        "csa_aux_loss_weight": args.csa_aux_loss_weight,
        "hca_m_prime": args.hca_m_prime,
        "sw_window": args.sw_window,
        "groups": args.groups,
        "group_dim": args.group_dim,

        # DeepSeekMoE + activation routing
        "moe_num_shared_experts": args.moe_num_shared_experts,
        "moe_num_routed_experts": args.moe_num_routed_experts,
        "moe_top_k": args.moe_top_k,
        "moe_expert_hidden_dim": args.moe_expert_hidden_dim,
        "activation_balance_weight": args.activation_balance_weight,
        "activation_top_k": args.activation_top_k,
        "activation_min_probability": args.activation_min_probability,
        "activation_balance_tolerance": args.activation_balance_tolerance,
        "moe_aux_loss_weight": args.moe_aux_loss_weight,

        # Parameter sharing / memory
        "tie_embeddings": args.tie_embeddings,
        "grad_checkpointing": args.grad_checkpointing,
    }
    for k, v in cli_fields.items():
        if v is not None:
            cfg_dict[k] = v

    return LoopConfig(**cfg_dict)


# ======================================================================
# Main
# ======================================================================

_DRIVE_SERVICE = None


def _get_drive_service():
    """Authenticate once and return a Google Drive API service."""
    global _DRIVE_SERVICE
    if _DRIVE_SERVICE is not None:
        return _DRIVE_SERVICE

    try:
        from google.colab import auth
        import google.auth
        from googleapiclient.discovery import build
    except ImportError as exc:
        raise RuntimeError(
            "Google Drive recycle-bin management requires "
            "google-api-python-client. Install it with: "
            "pip install -q google-api-python-client"
        ) from exc

    auth.authenticate_user()
    credentials, _ = google.auth.default(
        scopes=["https://www.googleapis.com/auth/drive"]
    )
    _DRIVE_SERVICE = build(
        "drive",
        "v3",
        credentials=credentials,
        cache_discovery=False,
    )
    return _DRIVE_SERVICE


def _resolve_drive_folder_id(service, checkpoint_dir: Path) -> str:
    """Resolve /content/drive/MyDrive/... to a Drive folder ID."""
    parts = checkpoint_dir.resolve().parts

    try:
        mydrive_idx = parts.index("MyDrive")
    except ValueError as exc:
        raise RuntimeError(
            f"Checkpoint directory is not under /content/drive/MyDrive: {checkpoint_dir}"
        ) from exc

    current_parent = "root"

    for name in parts[mydrive_idx + 1:]:
        safe_name = name.replace("'", "''")
        response = (
            service.files()
            .list(
                q=(
                    f"name = '{safe_name}' "
                    f"and '{current_parent}' in parents "
                    "and mimeType = 'application/vnd.google-apps.folder' "
                    "and trashed = false"
                ),
                spaces="drive",
                fields="files(id,name)",
                pageSize=10,
            )
            .execute()
        )

        matches = response.get("files", [])
        if not matches:
            raise RuntimeError(
                f"Could not resolve Google Drive folder '{name}' "
                f"while processing {checkpoint_dir}"
            )

        current_parent = matches[0]["id"]

    return current_parent


def _trash_old_checkpoints(service, checkpoint_dir: Path, keep_name: str) -> int:
    """Move older generated checkpoints in this Drive folder to Trash."""
    folder_id = _resolve_drive_folder_id(service, checkpoint_dir)

    response = (
        service.files()
        .list(
            q=f"'{folder_id}' in parents and trashed = false",
            spaces="drive",
            fields="files(id,name)",
            pageSize=1000,
        )
        .execute()
    )

    moved = 0
    for item in response.get("files", []):
        name = item.get("name", "")
        generated = (
            name == "latest.pt"
            or name.startswith("checkpoint_step_")
        )

        if generated and name != keep_name:
            service.files().update(
                fileId=item["id"],
                body={"trashed": True},
            ).execute()
            moved += 1

    return moved


def _flush_google_drive_trash(service) -> int:
    """Permanently delete everything currently in the Drive Trash."""
    deleted = 0
    page_token = None

    while True:
        response = (
            service.files()
            .list(
                q="trashed = true",
                spaces="drive",
                fields="nextPageToken, files(id)",
                pageSize=1000,
                pageToken=page_token,
            )
            .execute()
        )

        for item in response.get("files", []):
            service.files().delete(fileId=item["id"]).execute()
            deleted += 1

        page_token = response.get("nextPageToken")
        if not page_token:
            break

    return deleted


def _save_checkpoint(
    checkpoint_dir: Path,
    model: LoopTransformer,
    optimizer: torch.optim.Optimizer,
    step: int,
    *,
    colab: bool,
    flush_recycle_bin: bool,
) -> Path:
    """Save a checkpoint, optionally using Drive Trash rotation."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if colab and flush_recycle_bin:
        # IMPORTANT: write a new file first. We do not rely on filesystem
        # overwrite semantics to move the previous checkpoint to Trash.
        checkpoint_path = checkpoint_dir / f"checkpoint_step_{step}.pt"

        save_checkpoint(
            checkpoint_path,
            model,
            optimizer=optimizer,
            step=step,
        )

        service = _get_drive_service()

        old_count = _trash_old_checkpoints(
            service,
            checkpoint_dir,
            checkpoint_path.name,
        )
        if old_count:
            log.info(
                f"Moved {old_count} previous checkpoint file(s) to "
                "Google Drive Trash."
            )

        deleted_count = _flush_google_drive_trash(service)
        log.info(
            f"Flushed Google Drive Trash: permanently deleted "
            f"{deleted_count} item(s)."
        )

        return checkpoint_path

    # Standard mode keeps the original single latest.pt behavior.
    checkpoint_path = checkpoint_dir / "latest.pt"
    save_checkpoint(
        checkpoint_path,
        model,
        optimizer=optimizer,
        step=step,
    )
    return checkpoint_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Data
    p.add_argument("--train-data", required=True, type=Path)
    p.add_argument("--val-data", type=Path, default=None)
    p.add_argument("--data-dtype", default="uint32", choices=["uint16", "uint32"],
                    help="dtype used by prepare_data.py when writing token ids")
    # Architecture (all optional -- None means "use LoopConfig's default,
    # or whatever --config-json specifies")
    p.add_argument("--config-json", type=Path, default=None,
                    help="JSON file with any LoopConfig fields; CLI flags below override it")
    p.add_argument("--vocab-size", type=int, default=None)
    p.add_argument("--dim", type=int, default=None)
    p.add_argument("--n-layers", type=int, default=None)
    p.add_argument("--n-heads", type=int, default=None)
    p.add_argument("--head-dim", type=int, default=None)
    p.add_argument("--rope-dim", type=int, default=None,
                    help="must be even and <= head-dim; LoopConfig default is 64")
    p.add_argument("--ffn-hidden-dim", type=int, default=None)
    p.add_argument("--embed-dim", type=int, default=None)
    p.add_argument("--embed-dim-out", type=int, default=None)

    # LoopLM / recurrence
    p.add_argument("--max-loops", type=int, default=None)
    p.add_argument("--min-loops", type=int, default=None)
    p.add_argument("--beta-entropy", type=float, default=None)
    p.add_argument("--joint-exit-loss-weight", type=float, default=None,
                    help="blend weight for Ouro learned-exit expected loss; 0 trains every loop directly (recommended Stage-I)")
    p.add_argument("--loop-supervision-weight", type=float, default=None,
                    help="extra direct supervision over every executed loop; 0 preserves the original objective")
    p.add_argument("--loop-monotonic-weight", type=float, default=None,
                    help="soft batch-mean monotonicity penalty; 0 disables")
    p.add_argument("--loop-monotonic-margin", type=float, default=None,
                    help="tolerance for the monotonicity penalty")
    p.add_argument("--loop-refinement-weight", type=float, default=None,
                    help="per-example auxiliary loss rewarding loop-to-loop CE improvement")
    p.add_argument("--loop-refinement-margin", type=float, default=None,
                    help="required CE improvement before the refinement penalty becomes zero")
    p.add_argument("--loop-task-weight", type=float, default=None,
                    help="auxiliary weight for the distinct per-loop horizon-prediction task")
    p.add_argument("--loop-task-mode", choices=("horizon", "none"), default=None,
                    help="auxiliary task family used by recurrent depth")
    p.add_argument("--exit-gate-loop-embed-dim", type=int, default=None,
                    help="learned loop-index embedding width for exit gate")
    p.add_argument("--loop-sampling", dest="loop_sampling",
                    action="store_true", default=None)
    p.add_argument("--no-loop-sampling", dest="loop_sampling",
                    action="store_false")
    p.add_argument("--loop-sampling-lambda", type=float, default=None)
    p.add_argument("--recurrent-depth-controller", "--recurrent-depth-mixer", dest="recurrent_depth_controller", action="store_true", default=None,
                    help="enable learned recurrent-depth state controller (legacy --recurrent-depth-mixer alias retained)")
    p.add_argument("--no-recurrent-depth-controller", "--no-recurrent-depth-mixer", dest="recurrent_depth_controller", action="store_false")
    p.add_argument("--recurrent-depth-bottleneck-dim", "--recurrent-depth-mixer-dim", dest="recurrent_depth_bottleneck_dim", type=int, default=None)
    p.add_argument("--recurrent-update-init", type=float, default=None)
    p.add_argument("--shortcut-consistency-weight", type=float, default=None)
    p.add_argument("--shortcut-consistency-temperature", type=float, default=None)
    p.add_argument("--training-mode", choices=("recurrent", "diffusion", "hybrid"), default=None)
    p.add_argument("--hybrid-diffusion-probability", type=float, default=None)

    # DiffusionBlocks recurrent-depth training.
    p.add_argument("--diffusion-blocks", action="store_true",
                   help="train with the DiffusionBlocks recurrent-depth single-pass denoising objective")
    p.add_argument("--diffusion-sigma-min", type=float, default=None)
    p.add_argument("--diffusion-sigma-max", type=float, default=None)
    p.add_argument("--diffusion-p-mean", type=float, default=None)
    p.add_argument("--diffusion-p-std", type=float, default=None)
    p.add_argument("--diffusion-sigma-data", type=float, default=None)
    p.add_argument("--diffusion-loss-weight", type=float, default=None)
    p.add_argument("--diffusion-normalize-embeddings", dest="diffusion_normalize_embeddings", action="store_true", default=None)
    p.add_argument("--no-diffusion-normalize-embeddings", dest="diffusion_normalize_embeddings", action="store_false")
    p.add_argument("--diffusion-cond-dim", type=int, default=None)

    # Mixture of attention experts / contextual routing
    p.add_argument("--attention-mixture", dest="attention_mixture", action="store_true", default=None,
                   help="enable content+loop-routed mixture of SWA/CSA/HCA experts")
    p.add_argument("--no-attention-mixture", dest="attention_mixture", action="store_false")
    p.add_argument("--attention-mixture-top-k", type=int, default=None)
    p.add_argument("--attention-mixture-balance-weight", type=float, default=None)
    p.add_argument("--attention-mixture-loop-embed-dim", type=int, default=None)
    p.add_argument("--attention-mixture-num-experts", type=int, default=None,
                   help="number of attention families in the mixture (2 or 3; default 3)")
    p.add_argument("--attention-mixture-start-layer", type=int, default=None)
    p.add_argument("--attention-mixture-diversity-weight", type=float, default=None)
    p.add_argument("--attention-mixture-min-probability", type=float, default=None,
                   help="minimum dense routing probability per attention family (default 0.10)")
    p.add_argument("--attention-mixture-balance-tolerance", type=float, default=None,
                   help="batch-mean deviation tolerance before extra balance penalty (default 0.25)")

    # DeepSeek-style attention
    p.add_argument("--csa-m", type=int, default=None)
    p.add_argument("--csa-top-k", type=int, default=None)
    p.add_argument("--csa-aux-loss-weight", type=float, default=None)
    p.add_argument("--hca-m-prime", type=int, default=None)
    p.add_argument("--sw-window", type=int, default=None)
    p.add_argument("--groups", type=int, default=None)
    p.add_argument("--group-dim", type=int, default=None,
                    help="scale this down (e.g. dim//2) for smaller models -- see README")

    # DeepSeekMoE + activation routing
    p.add_argument("--moe-num-shared-experts", type=int, default=None)
    p.add_argument("--moe-num-routed-experts", type=int, default=None)
    p.add_argument("--moe-top-k", type=int, default=None)
    p.add_argument("--moe-expert-hidden-dim", type=int, default=None,
                    help="per-expert hidden width; default keeps active FFN width near the dense baseline")
    p.add_argument("--activation-balance-weight", type=float, default=None)
    p.add_argument("--activation-top-k", type=int, default=None,
                    help="number of activation functions active per token (1-4)")
    p.add_argument("--activation-min-probability", type=float, default=None,
                    help="minimum dense per-token activation probability before sparse top-k (default 0.10)")
    p.add_argument("--activation-balance-tolerance", type=float, default=None,
                    help="dead-zone around uniform batch mean before activation balance penalty/nudge (default 0.25)")
    p.add_argument("--moe-aux-loss-weight", type=float, default=None)
    p.add_argument("--tie-embeddings", dest="tie_embeddings", action="store_true", default=None)
    p.add_argument("--no-tie-embeddings", dest="tie_embeddings", action="store_false")
    p.add_argument("--grad-checkpointing", dest="grad_checkpointing", action="store_true", default=None,
                    help="trade compute for ~max_loops-fold less activation memory")
    p.add_argument("--no-grad-checkpointing", dest="grad_checkpointing", action="store_false",
                    help="force off, e.g. to override a --config-json file that set it on")
    # Optimization
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--min-lr-ratio", type=float, default=0.1)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.95)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--warmup-steps", type=int, default=200)
    p.add_argument("--max-steps", type=int, default=10_000)
    p.add_argument("--grad-accum-steps", type=int, default=1)
    # Batch shape
    p.add_argument("--batch-size", type=int, default=8, help="micro-batch size (per grad-accum step)")
    p.add_argument("--seq-len", type=int, default=1024)
    # Runtime
    p.add_argument("--device", default=None, help="default: cuda if available, else cpu")
    p.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16", "float16"])
    p.add_argument("--seed", type=int, default=1337)
    # Logging / checkpointing
    p.add_argument("--checkpoint-dir", type=Path, required=True)
    p.add_argument("--checkpoint-interval", type=int, default=500)
    p.add_argument("--eval-interval", type=int, default=200)
    p.add_argument("--eval-iters", type=int, default=20)
    p.add_argument("--log-interval", type=int, default=10)
    p.add_argument("--debug", action="store_true",
                    help="at each --log-interval step, also print per-loop losses, "
                         "actual loop count used, gradient norm, CSA/FFN router diagnostics, "
                         "and the batch's token id range")
    p.add_argument(
        "--colab",
        action="store_true",
        help="Colab mode: require Google Drive at /content/drive and require "
             "--checkpoint-dir to be on Drive. Periodic saves replace latest.pt "
             "in place so the local Colab runtime does not accumulate checkpoints.",
    )
    p.add_argument(
        "--flush-recycle-bin",
        action="store_true",
        help="After each checkpoint save in --colab mode, permanently empty "
             "the authenticated Google Drive trash. Destructive: all trashed "
             "Drive items are permanently deleted.",
    )
    p.add_argument("--init-from", type=Path, default=None,
                    help=("initialize model weights from an existing checkpoint without restoring its optimizer; "
                          "preserves the new training objective and continues the step counter from that checkpoint"))
    p.add_argument("--resume", action="store_true",
                    help="resume from the latest checkpoint in --checkpoint-dir if one exists")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    if args.resume and args.init_from:
        raise ValueError("--resume and --init-from are mutually exclusive; use --resume for an exact continuation or --init-from for a new objective/optimizer.")

    if args.flush_recycle_bin and not args.colab:
        raise ValueError("--flush-recycle-bin requires --colab.")

    if args.colab:
        drive_root = Path("/content/drive/MyDrive")
        if not drive_root.exists():
            raise RuntimeError(
                "Colab mode requested, but Google Drive is not mounted at "
                "/content/drive. Run in a Colab cell first:\n"
                "    from google.colab import drive\n"
                "    drive.mount('/content/drive')"
            )
        checkpoint_dir_str = str(args.checkpoint_dir.resolve())
        drive_root_str = str(drive_root.resolve())
        if not checkpoint_dir_str.startswith(drive_root_str + "/") and checkpoint_dir_str != drive_root_str:
            raise ValueError(
                "--colab requires --checkpoint-dir to be inside "
                "/content/drive/MyDrive/ so checkpoints persist."
            )
        args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        log.info(f"Colab persistent checkpoint mode: {args.checkpoint_dir}")

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype_map = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
    amp_dtype = dtype_map[args.dtype]

    if amp_dtype == torch.float32:
        use_amp = False
    elif amp_dtype == torch.bfloat16:
        use_amp = True  # autocast supports bf16 on both cuda and cpu
    else:  # float16
        if device.type != "cuda":
            log.warning("--dtype float16 is only well-supported on CUDA "
                        f"(device is {device.type}); falling back to float32.")
            amp_dtype = torch.float32
            use_amp = False
        else:
            use_amp = True
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and amp_dtype == torch.float16))

    log.info(f"Device: {device}   dtype: {amp_dtype} (amp={'on' if use_amp else 'off'})")

    data_dtype = np.uint16 if args.data_dtype == "uint16" else np.uint32
    train_data = load_bin(args.train_data, data_dtype)
    val_data = load_bin(args.val_data, data_dtype) if args.val_data else None
    log.info(f"Train tokens: {len(train_data):,}"
             + (f"   Val tokens: {len(val_data):,}" if val_data is not None else "   (no val set)"))

    cfg = build_config(args)
    # ``--diffusion-blocks`` selects the DiffusionBlocks recurrent-depth
    # objective unless the caller explicitly asks for another composed mode.
    # The previous implementation left training_mode at its dataclass default
    # (recurrent), which silently ignored the diffusion objective in the main
    # training loop even though diffusion_blocks=True was set.
    if cfg.diffusion_blocks and args.training_mode is None and cfg.training_mode == "recurrent":
        cfg.training_mode = "diffusion"
    validate_vocab_range(train_data, cfg.vocab_size, "train")
    if val_data is not None:
        validate_vocab_range(val_data, cfg.vocab_size, "validation")
    if cfg.diffusion_blocks:
        if cfg.loop_sampling:
            log.info("DiffusionBlocks mode: forcing loop_sampling=False because training is single-pass.")
            cfg.loop_sampling = False
        log.info(
            "DiffusionBlocks mode: one recurrent pass per training step; normal %d-loop inference is preserved.",
            cfg.max_loops,
        )
    log.info(f"Config: {cfg}")

    model = LoopTransformer(cfg).to(device)
    log.info(f"Physical params: {model.num_parameters():,}   "
             f"Effective (unrolled) params: {model.num_parameters(effective=True):,}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay,
    )

    start_step = 0

    if args.init_from is not None:
        if not args.init_from.exists():
            raise FileNotFoundError(f"--init-from checkpoint not found: {args.init_from}")
        log.info(f"Initializing weights from {args.init_from} (optimizer is NOT restored)")
        init_ckpt = torch.load(args.init_from, map_location=device, weights_only=True)
        if "model_state_dict" not in init_ckpt:
            raise ValueError(f"Checkpoint {args.init_from} does not contain model_state_dict")
        try:
            model.load_state_dict(init_ckpt["model_state_dict"], strict=True)
        except RuntimeError as exc:
            raise RuntimeError(
                "--init-from checkpoint architecture does not match the current configuration. "
                "Use the same parameter-shape configuration when warm-starting."
            ) from exc
        saved_step = init_ckpt.get("step")
        if not isinstance(saved_step, int):
            raise ValueError(f"Checkpoint {args.init_from} has no valid integer step")
        start_step = saved_step + 1
        log.info(f"Warm-start complete; continuing at step {start_step} with a fresh optimizer")
    ckpt_path = args.checkpoint_dir / "latest.pt"

    # In Colab + recycle-bin mode, use the newest numbered checkpoint.
    if args.colab and args.flush_recycle_bin:
        candidates = []
        for p in args.checkpoint_dir.glob("checkpoint_step_*.pt"):
            suffix = p.stem[len("checkpoint_step_"):]
            if suffix.isdigit():
                candidates.append((int(suffix), p))

        if candidates:
            candidates.sort(key=lambda item: item[0])
            ckpt_path = candidates[-1][1]

    if args.resume and ckpt_path.exists():
        log.info(f"Resuming from {ckpt_path}")
        ckpt = load_checkpoint(
            ckpt_path,
            device=device,
            optimizer=optimizer,
        )
        model = ckpt["model"]
        start_step = ckpt["step"] + 1
        log.info(f"Resumed at step {start_step}")
    elif args.resume:
        log.warning(
            f"--resume given but no checkpoint found at {ckpt_path}; "
            "starting fresh"
        )

    # Ctrl+C / SIGTERM (e.g. preemption on a shared cluster) triggers a
    # final checkpoint save before exit, instead of losing the run.
    interrupted = {"flag": False}

    def _handle_interrupt(signum, frame):
        log.warning("Interrupt received -- will checkpoint and exit after this step")
        interrupted["flag"] = True

    signal.signal(signal.SIGINT, _handle_interrupt)
    signal.signal(signal.SIGTERM, _handle_interrupt)

    model.train()
    t0 = time.time()
    running_loss = 0.0

    for step in range(start_step, args.max_steps):
        lr = get_lr(step, args.warmup_steps, args.max_steps, args.lr, args.min_lr_ratio)
        for g in optimizer.param_groups:
            g["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0
        last_step_losses = None
        last_batch = None
        for _ in range(args.grad_accum_steps):
            batch = get_batch(train_data, args.batch_size, args.seq_len, device)
            last_batch = batch
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                if model.cfg.training_mode == "hybrid":
                    loss, mode_info = model.hybrid_loss(batch)
                    training_mode_used = mode_info["mode"]
                    if training_mode_used == "diffusion":
                        diffusion_info = mode_info
                        step_losses = torch.stack([diffusion_info["ce"]])
                    else:
                        diffusion_info = None
                        step_losses = mode_info["step_losses"]
                elif model.cfg.training_mode == "diffusion":
                    loss, diffusion_info = model.diffusion_blocks_loss(batch)
                    step_losses = torch.stack([diffusion_info["ce"]])
                    training_mode_used = "diffusion"
                else:
                    diffusion_info = None
                    loss, step_losses = model.compute_loss(batch)
                    training_mode_used = "recurrent"
                loss = loss / args.grad_accum_steps
            last_step_losses = step_losses
            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()
            step_loss += loss.item()

        if scaler.is_enabled():
            scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        if scaler.is_enabled():
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        running_loss += step_loss

        if step % args.log_interval == 0:
            elapsed = time.time() - t0
            avg_loss = running_loss / max(1, min(step - start_step + 1, args.log_interval))
            log.info(f"step {step:>7,}  loss {step_loss:.4f}  lr {lr:.2e}  "
                     f"mode={training_mode_used}  {elapsed:.1f}s elapsed")
            running_loss = 0.0

            if args.debug:
                per_loop = "  ".join(f"L{i}={v:.4f}" for i, v in enumerate(last_step_losses.tolist()))
                log.info(f"  [debug] per-loop losses: {per_loop}  "
                         f"(n_loops_this_step={len(last_step_losses)}, "
                         f"cfg.max_loops={model.cfg.max_loops})")
                if training_mode_used == "diffusion":
                    log.info(
                        f"  [debug] grad_norm(pre-clip)={grad_norm:.4f}  "
                        f"csa_aux_loss={model.last_csa_aux_loss.item():.4f}  "
                        f"activation_balance={model.last_activation_balance_loss.item():.4f}  "
                        f"moe_aux={model.last_moe_aux_loss.item():.4f}"
                    )
                else:
                    log.info(f"  [debug] grad_norm(pre-clip)={grad_norm:.4f}  "
                             f"csa_aux_loss={model.last_csa_aux_loss.item():.4f}  "
                             f"activation_balance={model.last_activation_balance_loss.item():.4f}  "
                             f"moe_aux={model.last_moe_aux_loss.item():.4f}  "
                             f"refinement_aux={model.last_loop_refinement_loss.item():.4f}  "
                             f"loop_task_aux={model.last_loop_task_loss.item():.4f}  "
                             f"monotonic_aux={model.last_loop_monotonic_loss.item():.4f}  "
                             f"joint_exit={model.last_joint_exit_loss.item():.4f}")
                if model.last_activation_probs is not None:
                    log.info("  [debug] activation mean probs: " + " ".join(
                        f"A{i}={p:.3f}" for i, p in enumerate(model.last_activation_probs.tolist())
                    ))
                if model.last_moe_load is not None:
                    log.info("  [debug] MoE routed load: " + " ".join(
                        f"E{i}={p:.3f}" for i, p in enumerate(model.last_moe_load.tolist())
                    ))
                if model.last_attention_routing:
                    for item in model.last_attention_routing:
                        probs = item["attention_probs_mean"].tolist()
                        load = item["attention_load"].tolist()
                        log.info(
                            "  [debug] attention mixture layer=%s probs=%s load=%s entropy=%.3f",
                            int(item["layer"].item()),
                            " ".join(f"A{i}={p:.3f}" for i, p in enumerate(probs)),
                            " ".join(f"E{i}={p:.3f}" for i, p in enumerate(load)),
                            item["attention_entropy"].item(),
                        )
                        by_loop = item.get("attention_probs_by_loop")
                        if by_loop is not None and by_loop.numel() > 0:
                            for li, row in enumerate(by_loop.tolist()):
                                log.info(
                                    "  [debug] attention-by-loop layer=%s loop=%s %s",
                                    int(item["layer"].item()), li + 1,
                                    " ".join(f"A{i}={p:.3f}" for i, p in enumerate(row)),
                                )
                log.info(f"  [debug] batch token id range: "
                         f"[{last_batch.min().item()}, {last_batch.max().item()}]  "
                         f"(vocab_size={model.cfg.vocab_size})")
                if training_mode_used == "diffusion":
                    log.info(
                        "  [debug] diffusion sigma mean=%.4g range=[%.4g, %.4g] weighted_ce=%.4f",
                        diffusion_info["mean_sigma"].item(),
                        diffusion_info["min_sigma"].item(),
                        diffusion_info["max_sigma"].item(),
                        diffusion_info["weighted_ce"].item(),
                    )

        if val_data is not None and step % args.eval_interval == 0 and step > start_step:
            if model.cfg.training_mode == "diffusion":
                model.eval()
                val_ce = []
                val_weighted = []
                for _ in range(args.eval_iters):
                    vb = get_batch(val_data, args.batch_size, args.seq_len, device)
                    metrics = model.diffusion_blocks_eval(vb)
                    val_ce.append(metrics["ce"])
                    val_weighted.append(metrics["weighted_loss"])
                model.train()
                log.info(
                    f"step {step:>7,}  diffusion_val_ce {float(np.mean(val_ce)):.4f} "
                    f"diffusion_val_weighted {float(np.mean(val_weighted)):.4f}"
                )
            elif model.cfg.training_mode == "hybrid":
                val_loss = estimate_loss(model, val_data, args.batch_size, args.seq_len, device, args.eval_iters)
                model.eval()
                val_ce = []
                val_weighted = []
                for _ in range(max(1, args.eval_iters // 2)):
                    vb = get_batch(val_data, args.batch_size, args.seq_len, device)
                    metrics = model.diffusion_blocks_eval(vb)
                    val_ce.append(metrics["ce"])
                    val_weighted.append(metrics["weighted_loss"])
                model.train()
                log.info(f"step {step:>7,}  hybrid_recurrent_val_loss {val_loss:.4f} "
                         f"hybrid_diffusion_val_ce {float(np.mean(val_ce)):.4f} "
                         f"hybrid_diffusion_weighted {float(np.mean(val_weighted)):.4f}")
            else:
                val_loss = estimate_loss(model, val_data, args.batch_size, args.seq_len, device, args.eval_iters)
                log.info(f"step {step:>7,}  val_loss {val_loss:.4f}")

        should_checkpoint = (step % args.checkpoint_interval == 0 and step > start_step) or interrupted["flag"]
        if should_checkpoint:
            saved_path = _save_checkpoint(
                args.checkpoint_dir,
                model,
                optimizer,
                step,
                colab=args.colab,
                flush_recycle_bin=args.flush_recycle_bin,
            )
            log.info(
                f"Checkpoint saved at step {step} -> {saved_path}"
            )

        if interrupted["flag"]:
            log.warning("Exiting after checkpoint due to interrupt.")
            sys.exit(0)

    saved_path = _save_checkpoint(
        args.checkpoint_dir,
        model,
        optimizer,
        args.max_steps - 1,
        colab=args.colab,
        flush_recycle_bin=args.flush_recycle_bin,
    )
    log.info(
        f"Training complete. Final checkpoint: {saved_path}"
    )

if __name__ == "__main__":
    main()
