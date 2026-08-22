from __future__ import annotations
import json
from pathlib import Path
import torch
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.loop_transformer import LoopConfig, LoopTransformer


def make_pattern(n: int, t: int, vocab: int) -> torch.Tensor:
    pos = torch.arange(t)
    starts = torch.arange(n) % vocab
    # Mix local, periodic, and segment-level structure so different fields matter.
    local = 3 * pos
    periodic = 5 * (pos // 4)
    segment = 7 * (pos // 12)
    return ((starts[:, None] + local[None, :] + periodic[None, :] + segment[None, :]) % vocab).long()


def routing_snapshot(model: LoopTransformer) -> list[dict]:
    out = []
    for li, block in enumerate(model.blocks):
        dbg = block.attention_routing_debug()
        if not dbg:
            continue
        fdbg = {
            "activation_probs_dense_mean": block.ffn.last_activation_probs_dense_mean,
            "activation_probs_mean": block.ffn.last_activation_probs_mean,
        }
        out.append({
            "layer": li,
            "attention_probs_mean": [float(x) for x in dbg["attention_probs_mean"].cpu()],
            "attention_load": [float(x) for x in dbg["attention_load"].cpu()],
            "attention_entropy": float(dbg["attention_entropy"].cpu()),
            "attention_by_loop": [[float(x) for x in row.cpu()] for row in dbg["attention_probs_by_loop"]],
            "activation_probs_dense_mean": ([float(x) for x in fdbg["activation_probs_dense_mean"].cpu()] if fdbg.get("activation_probs_dense_mean") is not None else None),
            "activation_probs_mean": ([float(x) for x in fdbg["activation_probs_mean"].cpu()] if fdbg.get("activation_probs_mean") is not None else None),
        })
    return out


def evaluate(model, data):
    model.eval()
    with torch.no_grad():
        _, losses = model.compute_loss(data, max_loops=4)
        routing = routing_snapshot(model)
        val = [float(x) for x in losses.cpu()]
        imp = [float(x) for x in (losses[:-1] - losses[1:]).cpu()]
    model.train()
    return val, imp, routing


def main() -> None:
    torch.manual_seed(123)
    cfg = LoopConfig(
        vocab_size=48, dim=48, n_layers=2, n_heads=3, head_dim=16,
        ffn_hidden_dim=96, rope_dim=8, max_loops=4, min_loops=4,
        loop_sampling=False, loop_supervision_weight=0.05,
        loop_refinement_weight=0.02, loop_refinement_margin=0.001,
        loop_task_weight=0.02, loop_task_mode="horizon",
        exit_gate_loop_embed_dim=8,
        recurrent_depth_controller=True, recurrent_depth_bottleneck_dim=32,
        recurrent_update_init=0.95,
        csa_m=2, csa_top_k=4, hca_m_prime=4, sw_window=8,
        groups=3, group_dim=16, moe_num_shared_experts=1,
        moe_num_routed_experts=2, moe_top_k=1, activation_top_k=2,
        activation_min_probability=0.10, activation_balance_tolerance=0.25,
        activation_balance_weight=0.02, moe_aux_loss_weight=0.0,
        csa_aux_loss_weight=0.0, tie_embeddings=True, grad_checkpointing=False,
        attention_mixture=True, attention_mixture_top_k=1,
        attention_mixture_num_experts=3, attention_mixture_start_layer=0,
        attention_mixture_balance_weight=0.01,
        attention_mixture_loop_embed_dim=8,
        attention_mixture_diversity_weight=0.04,
    )
    model = LoopTransformer(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3)
    train = make_pattern(8, 24, 48)
    val = make_pattern(4, 24, 48)
    before, imp_before, _ = evaluate(model, val)
    history = []
    for step in range(1, 31):
        opt.zero_grad(set_to_none=True)
        loss, loop_losses = model.compute_loss(train, max_loops=4)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 5 == 0:
            vl, imp, routing = evaluate(model, val)
            row = {"step": step, "train": float(loss.detach()), "val": vl, "improvement": imp, "routing": routing}
            history.append(row)
            print(json.dumps(row))
    after, imp_after, routing_after = evaluate(model, val)
    out = {
        "val_before": before,
        "improvement_before": imp_before,
        "val_after": after,
        "improvement_after": imp_after,
        "routing_after": routing_after,
        "history": history,
        "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }
    p = Path("tmp/attention_specialization.json")
    p.parent.mkdir(exist_ok=True)
    p.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
