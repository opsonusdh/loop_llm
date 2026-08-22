#!/usr/bin/env python3
"""Measure whether recurrent depth actually changes predictive quality.

Runs the same checkpoint at fixed depths 1..N and reports per-depth CE,
relative improvement, state cosine, and recurrent update utilization. Depths
above the configured max_loops are explicitly marked extrapolation.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/'src'))
from loop_transformer import load_checkpoint

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--data-dtype',choices=('uint16','uint32'),default='uint16')
    p.add_argument('--batch-size',type=int,default=2)
    p.add_argument('--seq-len',type=int,default=320)
    p.add_argument('--eval-batches',type=int,default=50)
    p.add_argument('--max-depth',type=int,default=9)
    p.add_argument('--device',choices=('cpu','cuda','xpu','dml','auto'),default='auto')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--output',type=Path,default=None)
    a=p.parse_args()
    if a.device=='auto':
        if torch.cuda.is_available(): dev=torch.device('cuda')
        elif hasattr(torch,'xpu') and torch.xpu.is_available(): dev=torch.device('xpu')
        else:
            try:
                import torch_directml; dev=torch_directml.device()
            except Exception: dev=torch.device('cpu')
    elif a.device=='dml':
        import torch_directml; dev=torch_directml.device()
    else: dev=torch.device(a.device)
    data=np.memmap(a.data,dtype=(np.uint16 if a.data_dtype=='uint16' else np.uint32),mode='r')
    ck=load_checkpoint(a.checkpoint,device='cpu'); model=ck['model'].to(dev).eval()
    g=torch.Generator(device='cpu').manual_seed(a.seed)
    rows=[]
    for depth in range(1,a.max_depth+1):
        losses=[]; cos_terms=[]; update_terms=[]
        for _ in range(a.eval_batches):
            starts=torch.randint(0,len(data)-a.seq_len,(a.batch_size,),generator=g)
            idx=torch.stack([torch.from_numpy(data[int(i):int(i)+a.seq_len].astype(np.int64,copy=False)) for i in starts]).to(dev)
            with torch.no_grad():
                logits,_=model(idx,max_loops=depth)
                ce=F.cross_entropy(logits[-1][:,:-1].reshape(-1,logits[-1].size(-1)),idx[:,1:].reshape(-1)).item()
                losses.append(ce)
                states=getattr(model,'_last_loop_states',[])
                if len(states)>1:
                    a=states[-2].mean(dim=1); b=states[-1].mean(dim=1)
                    cos_terms.append(float(torch.nn.functional.cosine_similarity(a,b,dim=-1).mean()))
                upd=getattr(model,'last_recurrent_update_means',None)
                if upd is not None and len(upd):
                    update_terms.append(float(upd[-1]))
        row={'depth':depth,'loss':float(np.mean(losses)),'extrapolation':depth>model.cfg.max_loops,
             'state_cosine_last_transition':float(np.mean(cos_terms)) if cos_terms else None,
             'update_mean_last_transition':float(np.mean(update_terms)) if update_terms else None}
        rows.append(row); print(json.dumps(row))
    base=rows[0]['loss']
    for r in rows: r['relative_to_depth1']=(r['loss']-base)/max(base,1e-12)
    print(json.dumps({'checkpoint_step':ck.get('step'),'trained_max_loops':model.cfg.max_loops,'rows':rows},indent=2))
    if a.output:
        a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps({'rows':rows},indent=2))
if __name__=='__main__': main()
