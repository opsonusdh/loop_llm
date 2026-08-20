#!/usr/bin/env python3
from __future__ import annotations
import sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
from loop_transformer import LoopConfig, LoopTransformer

def make_stream(vocab:int,n:int,seed:int):
    # Deterministic low-entropy sequence with a long-range 8-token motif plus a rare distractor.
    g=torch.Generator().manual_seed(seed)
    motif=torch.tensor([1,2,3,4,5,6,7,8],dtype=torch.long)
    x=motif.repeat(n//8+1)[:n].clone()
    noise=torch.randint(0,2,(n,),generator=g)
    x=torch.where(noise==0,x,(x+1).remainder(vocab))
    return x

def batch(data,bs,T,g):
    starts=torch.randint(0,len(data)-T,(bs,),generator=g)
    return torch.stack([data[int(i):int(i)+T] for i in starts])

def evaluate(model,data,g,batches=8):
    model.eval(); vals=[]; loops=[]
    with torch.no_grad():
        for _ in range(batches):
            x=batch(data,4,24,g); loss,per_loop=model.compute_loss(x,max_loops=model.cfg.max_loops); vals.append(float(loss)); loops.append(per_loop)
    return sum(vals)/len(vals), torch.stack(loops).mean(0)

def main():
    torch.manual_seed(123)
    cfg=LoopConfig(
        vocab_size=32,dim=32,n_layers=2,n_heads=2,head_dim=16,ffn_hidden_dim=64,rope_dim=8,
        max_loops=4,beta_entropy=.02,csa_m=2,csa_top_k=3,hca_m_prime=3,sw_window=6,groups=2,group_dim=16,
        moe_num_shared_experts=1,moe_num_routed_experts=2,moe_top_k=1,activation_top_k=2,loop_sampling=False,
        loop_supervision_weight=0.0,loop_monotonic_weight=0.0,loop_refinement_weight=.05,loop_refinement_margin=.001,
        loop_task_weight=.01,loop_task_mode='horizon',exit_gate_loop_embed_dim=8,tie_embeddings=True,grad_checkpointing=False,
    )
    model=LoopTransformer(cfg)
    opt=torch.optim.AdamW(model.parameters(),lr=2e-3,betas=(.9,.95))
    train=make_stream(32,1024,7); val=make_stream(32,512,17); g=torch.Generator().manual_seed(99)
    before,_=evaluate(model,val,g,batches=4); print(f'VAL_BEFORE={before:.5f}')
    for step in range(1,61):
        model.train(); x=batch(train,4,24,g); opt.zero_grad(set_to_none=True); loss,per_loop=model.compute_loss(x,max_loops=4); loss.backward(); grad=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.0).item()); opt.step()
        if step in (1,10,30,60):
            val_loss,val_loops=evaluate(model,val,g,batches=4)
            print(f'STEP={step} TRAIN={loss.item():.5f} VAL={val_loss:.5f} GRAD={grad:.4f} LOOP_CE=' + ','.join(f'{x:.4f}' for x in per_loop.tolist()) + f' REFINE={float(model.last_loop_refinement_loss):.6f} TASK={float(model.last_loop_task_loss):.6f} MONO={float(model.last_loop_monotonic_loss):.6f}')
            print('VAL_LOOP_CE=' + ','.join(f'{x:.4f}' for x in val_loops.tolist()))
    after,val_loops=evaluate(model,val,g,batches=8); print(f'VAL_AFTER={after:.5f}'); print('VAL_LOOP_CE_FINAL=' + ','.join(f'{x:.4f}' for x in val_loops.tolist()))
    assert torch.isfinite(loss)
    assert after < before, f'validation failed to improve: {before:.5f} -> {after:.5f}'
    assert model.exit_gate.loop_embedding is not None and model.exit_gate.proj.in_features==40
    assert model.last_loop_refinement_loss >= 0 and model.last_loop_task_loss >= 0
    print('SMOKE_TEST=PASS')
if __name__=='__main__': main()
