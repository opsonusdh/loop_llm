#!/usr/bin/env python3
from __future__ import annotations
import argparse, logging, random, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/'src'))
from loop_transformer.checkpointing import load_checkpoint, save_checkpoint
LOG=logging.getLogger('exit_gate')

def resolve_device(spec:str)->torch.device:
    if spec=='auto':
        if torch.cuda.is_available(): return torch.device('cuda')
        if hasattr(torch,'xpu') and torch.xpu.is_available(): return torch.device('xpu')
        try:
            import torch_directml
            return torch_directml.device()
        except (ImportError,RuntimeError): return torch.device('cpu')
    if spec=='cuda':
        if not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
        return torch.device('cuda')
    if spec=='xpu':
        if not hasattr(torch,'xpu') or not torch.xpu.is_available(): raise RuntimeError('XPU unavailable')
        return torch.device('xpu')
    if spec=='dml':
        import torch_directml
        return torch_directml.device()
    return torch.device('cpu')

def get_batch(data,batch_size,seq_len,device,g):
    max_start=len(data)-seq_len
    if max_start<=0: raise ValueError('sequence length exceeds data')
    starts=torch.randint(0,max_start,(batch_size,),generator=g)
    x=torch.stack([torch.from_numpy(data[int(i):int(i)+seq_len].astype(np.int64)) for i in starts])
    return x.to(device)

def collect(model,idx):
    hidden=[]
    def hook(_m,inputs,_o):
        if not inputs or not torch.is_tensor(inputs[0]): raise RuntimeError('exit gate hook did not receive hidden state')
        hidden.append(inputs[0].detach())
    h=model.exit_gate.register_forward_hook(hook)
    try:
        with torch.no_grad(): logits,_=model.forward(idx,max_loops=model.cfg.max_loops)
    finally:
        h.remove()
    losses=[]; target=idx[:,1:]
    for lg in logits:
        ce=F.cross_entropy(lg[:,:-1].reshape(-1,lg.size(-1)),target.reshape(-1),reduction='none')
        losses.append(ce.reshape(idx.size(0),-1).mean(1))
    return hidden,torch.stack(losses,1)

def gate_probs(model,hidden):
    return torch.stack([model.exit_gate(h,loop_idx=t) for t,h in enumerate(hidden)],dim=1)

def target_probs(losses,sharpness,margin):
    improvements=(losses[:,:-1]-losses[:,1:]).clamp_min(0)
    cont=torch.sigmoid(sharpness*(improvements-margin))
    out=torch.empty_like(losses)
    out[:,:-1]=1.0-cont
    out[:,-1]=1.0
    return out,improvements


def evaluate_gate(model, data, batch_size, seq_len, device, g, batches, sharpness, margin):
    model.eval(); total=0.0; count=0; depth_sum=0.0; imp_sum=0.0
    with torch.no_grad():
        for _ in range(batches):
            idx=get_batch(data,batch_size,seq_len,device,g)
            hidden,losses=collect(model,idx)
            targets,imps=target_probs(losses,sharpness,margin)
            probs=gate_probs(model,hidden)
            bce=F.binary_cross_entropy(probs,targets)
            survival=torch.ones(probs.size(0),device=probs.device)
            pexit=[]
            for t in range(probs.size(1)):
                if t < probs.size(1)-1:
                    p=survival*probs[:,t]; survival=survival*(1-probs[:,t])
                else:
                    p=survival
                pexit.append(p)
            pexit=torch.stack(pexit,1); depths=torch.arange(1,probs.size(1)+1,device=probs.device,dtype=probs.dtype)
            depth=(pexit*depths).sum(1).mean()
            total+=float(bce.item()); depth_sum+=float(depth.item()); imp_sum+=float(imps.mean().item()); count+=1
    return {
        'gate_bce': total/max(1,count),
        'expected_depth': depth_sum/max(1,count),
        'mean_positive_improvement': imp_sum/max(1,count),
    }

def main():
    p=argparse.ArgumentParser(description='Stage-II gate trainer for refined LoopLM with loop-index embedding')
    p.add_argument('--checkpoint',type=Path,required=True); p.add_argument('--train-data',type=Path,required=True); p.add_argument('--val-data',type=Path,required=True); p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--data-dtype',choices=('uint16','uint32'),default='uint16'); p.add_argument('--batch-size',type=int,default=2); p.add_argument('--seq-len',type=int,default=320); p.add_argument('--max-steps',type=int,default=2000)
    p.add_argument('--lr',type=float,default=1e-3); p.add_argument('--warmup-steps',type=int,default=100); p.add_argument('--sharpness',type=float,default=50.0); p.add_argument('--margin',type=float,default=0.005)
    p.add_argument('--device',choices=('auto','cpu','cuda','xpu','dml'),default='auto'); p.add_argument('--resume',action='store_true'); p.add_argument('--eval-interval',type=int,default=100); p.add_argument('--checkpoint-interval',type=int,default=100); p.add_argument('--eval-batches',type=int,default=25); p.add_argument('--seed',type=int,default=42); p.add_argument('--log-interval',type=int,default=10)
    a=p.parse_args(); logging.basicConfig(level=logging.INFO,format='%(asctime)s  %(levelname)-5s %(message)s'); dev=resolve_device(a.device); LOG.info('Device: %s',dev)
    dt=np.uint16 if a.data_dtype=='uint16' else np.uint32; train=np.memmap(a.train_data,dtype=dt,mode='r'); val=np.memmap(a.val_data,dtype=dt,mode='r')
    base=load_checkpoint(a.checkpoint,device='cpu'); model=base['model'].to(dev).eval()
    def freeze_gate(m):
        for n,q in m.named_parameters(): q.requires_grad_(n.startswith('exit_gate.'))
        bad=[n for n,q in m.named_parameters() if q.requires_grad and not n.startswith('exit_gate.')]
        if bad: raise AssertionError(bad[:10])
        return [q for n,q in m.named_parameters() if n.startswith('exit_gate.')]
    trainable=freeze_gate(model); opt=torch.optim.AdamW(trainable,lr=a.lr,betas=(.9,.95))
    out=a.output_dir; out.mkdir(parents=True,exist_ok=True); latest=out/'latest.pt'; start=1; best=None
    if a.resume:
        ck=load_checkpoint(latest,device='cpu'); extra=ck.get('extra',{})
        if extra.get('stage')!='stage2_exit_gate': raise ValueError('resume checkpoint is not refined Stage-II')
        model=ck['model'].to(dev).eval(); trainable=freeze_gate(model); opt=torch.optim.AdamW(trainable,lr=a.lr,betas=(.9,.95)); opt.load_state_dict(ck['optimizer_state_dict']); start=int(ck['step'])+1; best=extra.get('best_val_bce')
    g=torch.Generator(device='cpu').manual_seed(a.seed)
    vg=torch.Generator(device='cpu').manual_seed(a.seed+1)
    for step in range(start,a.max_steps+1):
        idx=get_batch(train,a.batch_size,a.seq_len,dev,g); hidden,losses=collect(model,idx); targets,_=target_probs(losses,a.sharpness,a.margin)
        warm=min(1.0,step/max(1,a.warmup_steps));
        for group in opt.param_groups: group['lr']=a.lr*warm
        opt.zero_grad(set_to_none=True); probs=gate_probs(model,hidden); bce=F.binary_cross_entropy(probs,targets); bce.backward(); grad=float(torch.nn.utils.clip_grad_norm_(trainable,1.0).item()); opt.step()
        if step==1 or step%a.log_interval==0:
            LOG.info('step %d gate_bce %.6f lr %.3e grad %.4f train_mean_exit %.3f',step,bce.item(),a.lr*warm,grad,float(probs.mean().item()))
        if step%a.eval_interval==0 or step==a.max_steps:
            metrics=evaluate_gate(model,val,a.batch_size,a.seq_len,dev,vg,a.eval_batches,a.sharpness,a.margin)
            LOG.info('step %d val_gate_bce %.6f expected_depth %.3f mean_positive_improvement %.6f',step,metrics['gate_bce'],metrics['expected_depth'],metrics['mean_positive_improvement'])
            if best is None or metrics['gate_bce'] < best:
                best=metrics['gate_bce']
                save_checkpoint(out/'best.pt',model,opt,step,extra={'stage':'stage2_exit_gate','base_checkpoint':str(a.checkpoint),'best_val_bce':best,'sharpness':a.sharpness,'margin':a.margin})
        if step%a.checkpoint_interval==0 or step==a.max_steps:
            save_checkpoint(latest,model,opt,step,extra={'stage':'stage2_exit_gate','base_checkpoint':str(a.checkpoint),'best_val_bce':best,'sharpness':a.sharpness,'margin':a.margin})
    LOG.info('Stage-II refined gate training complete: %s (best=%s)',latest,'none' if best is None else f'{best:.6f}')
if __name__=='__main__': main()
