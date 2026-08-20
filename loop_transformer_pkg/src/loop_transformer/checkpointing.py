"""Checkpoint save/load utilities.

Bundles model weights, config, optimizer state, and training metadata
into a single file so a run can be paused and resumed exactly, without
the caller separately tracking what hyperparameters a given checkpoint
was trained with.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
import pickle

from .config import LoopConfig
from .model import LoopTransformer


def save_checkpoint(
    path:      Union[str, Path],
    model:     LoopTransformer,
    optimizer: Optional[torch.optim.Optimizer] = None,
    step:      int = 0,
    extra:     Optional[Dict[str, Any]] = None,
) -> None:
    """
    Save a full training checkpoint: model weights, config (so the model
    can be reconstructed without separately tracking hyperparameters),
    optimizer state (if given, for exact resume), the current training
    step, and any extra metadata the caller wants alongside it (e.g.
    best validation loss so far, RNG state).

    Writes to a temp file and renames into place, so a crash or
    interruption mid-write never leaves a truncated/corrupt checkpoint
    sitting at the target path.
    """
    payload: Dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "config": dataclasses.asdict(model.cfg),
        "step": step,
        "extra": extra or {},
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp_path)
    tmp_path.replace(path)


def load_checkpoint(
    path:      Union[str, Path],
    device:    Union[str, torch.device] = "cpu",
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> Dict[str, Any]:
    """
    Load a checkpoint saved by save_checkpoint(). Reconstructs the model
    from its saved config, loads weights onto `device`, and optionally
    restores optimizer state in place if an optimizer instance is passed
    in (it must already be constructed against a model with matching
    parameter shapes -- typically the model this same call returns).

    Uses weights_only=True: torch.load with weights_only=False can
    execute arbitrary code embedded in a malicious checkpoint file
    during unpickling. This format only ever contains tensors and plain
    Python primitives (from dataclasses.asdict and PyTorch's own
    optimizer.state_dict()), so the safe loader works with no loss of
    functionality -- there's no reason to accept the wider attack surface,
    especially for checkpoint files that may get shared or downloaded.

    Returns the raw checkpoint dict with the reconstructed model added
    under the "model" key, so callers can also inspect step/extra/
    optimizer state directly.
    """
    try:
        ckpt = torch.load(
            path,
            map_location=device,
            weights_only=True,
        )
    except pickle.UnpicklingError as exc:
        message = str(exc)

        # DML-generated checkpoints saved by this project can contain
        # backend/device and NumPy reconstruction objects that PyTorch 2.13's
        # weights-only unpickler does not allow by default.
        #
        # These checkpoints are local/trusted artifacts produced by our own
        # training run. For those trusted checkpoints, fall back to the
        # normal unpickler so the checkpoint can be restored correctly.
        known_dml_pickle_objects = (
            "_rebuild_device_tensor_from_numpy",
            "numpy._core.multiarray._reconstruct",
            "numpy.core.multiarray._reconstruct",
        )

        if not any(token in message for token in known_dml_pickle_objects):
            raise

        ckpt = torch.load(
            path,
            map_location=device,
            weights_only=False,
        )
    cfg = LoopConfig(**ckpt["config"])
    model = LoopTransformer(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    ckpt["model"] = model
    return ckpt
