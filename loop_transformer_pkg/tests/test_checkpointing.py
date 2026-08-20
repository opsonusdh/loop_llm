"""Checkpoint save/load: round-trip correctness, optimizer state
restoration, and safety (weights_only loading)."""

from __future__ import annotations

import torch

from loop_transformer import LoopConfig, LoopTransformer, load_checkpoint, save_checkpoint


def test_save_load_roundtrip_preserves_weights_and_config(tmp_path, tiny_config):
    torch.manual_seed(0)
    model = LoopTransformer(tiny_config)
    ckpt_path = tmp_path / "model.pt"

    save_checkpoint(ckpt_path, model, step=123)
    ckpt = load_checkpoint(ckpt_path)
    loaded = ckpt["model"]

    assert ckpt["step"] == 123
    assert loaded.cfg == tiny_config

    for (n1, p1), (n2, p2) in zip(model.named_parameters(), loaded.named_parameters()):
        assert n1 == n2
        assert torch.equal(p1, p2)


def test_save_load_roundtrip_preserves_optimizer_state(tmp_path, tiny_config, batch):
    torch.manual_seed(0)
    model = LoopTransformer(tiny_config)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Take a step so the optimizer actually has state (exp_avg etc.) to save.
    loss, _ = model.compute_loss(batch)
    loss.backward()
    opt.step()

    ckpt_path = tmp_path / "model.pt"
    save_checkpoint(ckpt_path, model, optimizer=opt, step=5)

    torch.manual_seed(0)
    new_model = LoopTransformer(tiny_config)
    new_opt = torch.optim.AdamW(new_model.parameters(), lr=1e-3)
    ckpt = load_checkpoint(ckpt_path, optimizer=new_opt)

    assert ckpt["step"] == 5
    orig_state = list(opt.state_dict()["state"].values())
    new_state = list(new_opt.state_dict()["state"].values())
    assert len(orig_state) == len(new_state)
    for a, b in zip(orig_state, new_state):
        assert torch.equal(a["exp_avg"], b["exp_avg"])


def test_load_checkpoint_reconstructs_model_without_separate_config(tmp_path, tiny_config):
    """The whole point of bundling config into the checkpoint: you
    shouldn't need to remember/pass hyperparameters separately to load."""
    torch.manual_seed(0)
    model = LoopTransformer(tiny_config)
    ckpt_path = tmp_path / "model.pt"
    save_checkpoint(ckpt_path, model)

    ckpt = load_checkpoint(ckpt_path)  # note: no config passed in at all
    assert isinstance(ckpt["model"], LoopTransformer)
    assert ckpt["model"].cfg.dim == tiny_config.dim


def test_save_creates_parent_directories(tmp_path, tiny_config):
    torch.manual_seed(0)
    model = LoopTransformer(tiny_config)
    nested_path = tmp_path / "a" / "b" / "c" / "model.pt"
    save_checkpoint(nested_path, model)
    assert nested_path.exists()
