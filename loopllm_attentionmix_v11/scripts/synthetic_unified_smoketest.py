from __future__ import annotations
import json
from pathlib import Path
import torch
from src.loop_transformer import LoopConfig, LoopTransformer


def make_pattern(n: int, t: int, vocab: int) -> torch.Tensor:
    # Structured sequence: x_t = (x_0 + 3*t + 2*(t//4)) mod vocab.
    starts = torch.arange(n) % vocab
    pos = torch.arange(t).unsqueeze(0)
    return ((starts.unsqueeze(1) + 3*pos + 2*(pos//4)) % vocab).long()


def main() -> None:
    torch.manual_seed(7)
    cfg = LoopConfig(
        vocab_size=32, dim=48, n_layers=2, n_heads=3, head_dim=16,
        ffn_hidden_dim=96, rope_dim=8, max_loops=4, min_loops=4,
        loop_sampling=False, beta_entropy=0.02,
        loop_supervision_weight=0.05, loop_refinement_weight=0.02,
        loop_refinement_margin=0.001, loop_task_weight=0.02,
        loop_task_mode="horizon", exit_gate_loop_embed_dim=8,
        recurrent_depth_controller=True, recurrent_depth_bottleneck_dim=32, recurrent_update_init=0.95,
        shortcut_consistency_weight=0.01,
        csa_m=2, csa_top_k=4, hca_m_prime=4, sw_window=8,
        groups=3, group_dim=16, moe_num_shared_experts=1,
        moe_num_routed_experts=2, moe_top_k=1, activation_top_k=2,
        activation_balance_weight=0.0, moe_aux_loss_weight=0.0,
        csa_aux_loss_weight=0.0, tie_embeddings=True, grad_checkpointing=False,
    )
    model = LoopTransformer(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    train = make_pattern(24, 48, 32)
    val = make_pattern(8, 48, 32)

    def eval_batch(x: torch.Tensor):
        model.eval()
        with torch.no_grad():
            _, losses = model.compute_loss(x, max_loops=4)
            improvements = losses[:-1] - losses[1:]
            cos = model.last_loop_state_cosines
            upd = model.last_recurrent_update_means
        model.train()
        return losses, improvements, cos, upd

    before, _, _, _ = eval_batch(val)
    history = []
    for step in range(1, 81):
        opt.zero_grad(set_to_none=True)
        loss, losses = model.compute_loss(train, max_loops=4)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 20 == 0:
            vl, imp, cos, upd = eval_batch(val)
            row = dict(step=step, train=float(loss.detach()), val=[float(x) for x in vl.detach()], improvement=[float(x) for x in imp.detach()],
                       state_cosine=[float(x) for x in cos.detach()], update_mean=[float(x) for x in upd.detach()])
            history.append(row)
            print(json.dumps(row))
    after, imp, cos, upd = eval_batch(val)
    summary = {
        "val_before": [float(x) for x in before],
        "val_after": [float(x) for x in after],
        "improvement_after": [float(x) for x in imp],
        "state_cosine_after": [float(x) for x in cos],
        "update_mean_after": [float(x) for x in upd],
        "history": history,
    }
    out = Path("tmp/unified_smoketest.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
