"""Checkpoint save/load for the refined LoopLM experiment."""
from __future__ import annotations

import dataclasses, pickle, random
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import torch

from .config import LoopConfig
from .model import LoopTransformer


def _capture_rng_state() -> Dict[str, Any]:
    """Capture all RNG streams used by training for exact continuation."""
    np_name, np_keys, np_pos, np_has_gauss, np_cached = np.random.get_state()
    state: Dict[str, Any] = {
        "torch": torch.get_rng_state(),
        "python": random.getstate(),
        # Store NumPy's uint32 key array as a tensor so weights_only=True can
        # safely deserialize the checkpoint without importing NumPy objects.
        "numpy": (
            np_name,
            torch.from_numpy(np_keys.copy()),
            int(np_pos),
            int(np_has_gauss),
            float(np_cached),
        ),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: Dict[str, Any]) -> None:
    """Restore RNG streams captured by _capture_rng_state()."""
    if not state:
        return
    if "torch" in state:
        torch.set_rng_state(state["torch"])
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np_name, np_keys, np_pos, np_has_gauss, np_cached = state["numpy"]
        if torch.is_tensor(np_keys):
            np_keys = np_keys.cpu().numpy()
        np.random.set_state((
            np_name, np.asarray(np_keys, dtype=np.uint32),
            int(np_pos), int(np_has_gauss), float(np_cached),
        ))
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(
    path: Union[str, Path],
    model: LoopTransformer,
    optimizer: Optional[torch.optim.Optimizer] = None,
    step: int = 0,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    payload = {
        "model_state_dict": model.state_dict(),
        "config": dataclasses.asdict(model.cfg),
        "step": step,
        "extra": extra or {},
        "rng_state": _capture_rng_state(),
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def load_checkpoint(
    path: Union[str, Path],
    device: Union[str, torch.device] = "cpu",
    optimizer: Optional[torch.optim.Optimizer] = None,
    *,
    restore_rng: bool = False,
) -> Dict[str, Any]:
    try:
        ckpt = torch.load(path, map_location=device, weights_only=True)
    except (pickle.UnpicklingError, RuntimeError) as exc:
        msg = str(exc)
        trusted_dml_markers = (
            "_rebuild_device_tensor_from_numpy",
            "numpy._core.multiarray._reconstruct",
            "numpy.core.multiarray._reconstruct",
        )
        if not any(m in msg for m in trusted_dml_markers):
            raise
        ckpt = torch.load(path, map_location=device, weights_only=False)

    cfg = LoopConfig(**ckpt["config"])
    model = LoopTransformer(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)

    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    if restore_rng:
        _restore_rng_state(ckpt.get("rng_state", {}))

    ckpt["model"] = model
    return ckpt
