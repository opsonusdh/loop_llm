"""Checkpoint save/load for the refined LoopLM experiment."""
from __future__ import annotations
import dataclasses, pickle
from pathlib import Path
from typing import Any, Dict, Optional, Union
import torch
from .config import LoopConfig
from .model import LoopTransformer

def save_checkpoint(path: Union[str,Path], model: LoopTransformer, optimizer: Optional[torch.optim.Optimizer]=None, step:int=0, extra:Optional[Dict[str,Any]]=None)->None:
    payload={"model_state_dict": model.state_dict(), "config": dataclasses.asdict(model.cfg), "step": step, "extra": extra or {}}
    if optimizer is not None: payload["optimizer_state_dict"] = optimizer.state_dict()
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix(path.suffix+'.tmp'); torch.save(payload,tmp); tmp.replace(path)

def load_checkpoint(path: Union[str,Path], device: Union[str,torch.device]="cpu", optimizer:Optional[torch.optim.Optimizer]=None)->Dict[str,Any]:
    try:
        ckpt=torch.load(path,map_location=device,weights_only=True)
    except pickle.UnpicklingError as exc:
        msg=str(exc)
        trusted_dml_markers=("_rebuild_device_tensor_from_numpy","numpy._core.multiarray._reconstruct","numpy.core.multiarray._reconstruct")
        if not any(m in msg for m in trusted_dml_markers): raise
        ckpt=torch.load(path,map_location=device,weights_only=False)
    cfg=LoopConfig(**ckpt["config"]); model=LoopTransformer(cfg).to(device); model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt: optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    ckpt["model"]=model; return ckpt
