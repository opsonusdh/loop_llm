from __future__ import annotations

import argparse, json, math, random, time, sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
from loop_transformer import LoopConfig, LoopTransformer

SEED = 42


def seed_all(seed: int = SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def collect_repo_text(root: Path) -> str:
    files = []
    for p in root.rglob('*'):
        if not p.is_file():
            continue
        if any(part.startswith('.') for part in p.parts):
            continue
        if p.suffix.lower() in {'.py','.md','.txt','.toml','.json'} and p.stat().st_size < 400_000:
            files.append(p)
    files.sort()
    chunks=[]
    for p in files:
        try:
            t=p.read_text('utf-8', errors='ignore')
        except Exception:
            continue
        if t.strip():
            chunks.append(f"\n\n# FILE: {p.relative_to(root)}\n\n{t}")
    return ''.join(chunks)


def write_byte_corpus(text: str, out_dir: Path, split: float=0.85):
    data = np.frombuffer(text.encode('utf-8'), dtype=np.uint8)
    cut = int(len(data)*split)
    train = data[:cut]; val=data[cut:]
    out_dir.mkdir(parents=True, exist_ok=True)
    train.tofile(out_dir/'train.bin'); val.tofile(out_dir/'val.bin')
    (out_dir/'train.txt').write_text(train.tobytes().decode('utf-8', errors='ignore'), 'utf-8')
    (out_dir/'val.txt').write_text(val.tobytes().decode('utf-8', errors='ignore'), 'utf-8')
    return train, val


def sample_batch(arr: np.ndarray, batch_size: int, seq_len: int, rng: np.random.Generator):
    starts = rng.integers(0, len(arr)-seq_len, size=batch_size)
    return torch.tensor(np.stack([arr[s:s+seq_len] for s in starts]).astype(np.int64))


def make_cfg(kind: str) -> LoopConfig:
    common = dict(
        vocab_size=256, dim=48, n_layers=2, n_heads=3, head_dim=16, ffn_hidden_dim=96,
        rope_dim=16, max_loops=4, min_loops=4, beta_entropy=0.05,
        loop_supervision_weight=0.05, joint_exit_loss_weight=0.0,
        loop_monotonic_weight=0.0, loop_refinement_weight=0.02, loop_refinement_margin=0.001,
        loop_task_weight=0.02, loop_task_mode='horizon', exit_gate_loop_embed_dim=8,
        recurrent_depth_controller=True, recurrent_depth_bottleneck_dim=16, recurrent_update_init=0.95,
        shortcut_consistency_weight=0.0, shortcut_consistency_temperature=2.0,
        diffusion_blocks=False, training_mode='recurrent',
        csa_m=2, csa_top_k=6, csa_aux_loss_weight=0.1, hca_m_prime=6, sw_window=12,
        groups=3, group_dim=16, tie_embeddings=True, grad_checkpointing=False,
        moe_num_shared_experts=1, moe_num_routed_experts=4, moe_top_k=1, moe_expert_hidden_dim=64,
        activation_balance_weight=0.01, activation_top_k=2, activation_min_probability=0.10,
        activation_balance_tolerance=0.25, activation_bias_update_speed=0.001,
        moe_aux_loss_weight=0.01, moe_bias_update_speed=0.001,
        attention_mixture=True, attention_mixture_top_k=1, attention_mixture_balance_weight=0.01,
        attention_mixture_loop_embed_dim=8, attention_mixture_num_experts=3,
        attention_mixture_start_layer=1, attention_mixture_diversity_weight=0.001,
        attention_mixture_min_probability=0.10, attention_mixture_balance_tolerance=0.25,
    )
    if kind == 'no_attention':
        common['attention_mixture'] = False
    elif kind == 'no_activation':
        common['activation_top_k'] = 4
    elif kind == 'no_moe_specialization':
        common['moe_num_routed_experts'] = 1
        common['moe_top_k'] = 1
    elif kind == 'depth1':
        common['max_loops'] = 1; common['min_loops'] = 1
        common['recurrent_depth_controller'] = False
        common['loop_supervision_weight'] = 0.0
        common['loop_refinement_weight'] = 0.0
        common['loop_task_weight'] = 0.0
    elif kind != 'full':
        raise ValueError(kind)
    return LoopConfig(**common)


def flatten_grad_norm(model: torch.nn.Module, predicate):
    s=0.0; n=0
    for name,p in model.named_parameters():
        if predicate(name) and p.grad is not None:
            g=float(p.grad.detach().float().norm().item()); s += g*g; n += 1
    return math.sqrt(s), n


def summarize_model(model):
    d={}
    d['params_total']=sum(p.numel() for p in model.parameters())
    d['params_trainable']=sum(p.numel() for p in model.parameters() if p.requires_grad)
    d['loop_state_cosines']=(model.last_loop_state_cosines.detach().cpu().tolist() if model.last_loop_state_cosines is not None else [])
    d['loop_improvements']=(model.last_loop_improvements.detach().cpu().tolist() if model.last_loop_improvements is not None else [])
    d['recurrent_update_means']=(model.last_recurrent_update_means.detach().cpu().tolist() if model.last_recurrent_update_means is not None else [])
    d['activation_mean']=(model.last_activation_probs.detach().cpu().tolist() if model.last_activation_probs is not None else None)
    d['moe_load']=(model.last_moe_load.detach().cpu().tolist() if model.last_moe_load is not None else None)
    d['attention'] = []
    for r in model.last_attention_routing:
        d['attention'].append({k: v.detach().cpu().tolist() for k,v in r.items()})
    d['aux']={
        'csa': float(model.last_csa_aux_loss or 0),
        'activation': float(model.last_activation_balance_loss or 0),
        'moe': float(model.last_moe_aux_loss or 0),
        'attention_mix': float(model.last_attention_mixture_loss or 0),
        'refinement': float(model.last_loop_refinement_loss or 0),
        'loop_task': float(model.last_loop_task_loss or 0),
    }
    return d


def train_one(kind: str, train: np.ndarray, val: np.ndarray, out: Path, steps=50):
    seed_all(SEED)
    device=torch.device('cpu')
    cfg=make_cfg(kind)
    model=LoopTransformer(cfg).to(device)
    if kind == 'no_activation':
        for b in model.blocks:
            b.ffn.diagnostic_uniform_activations = True
    if kind == 'no_moe_specialization':
        for b in model.blocks:
            b.ffn.diagnostic_uniform_experts = True
    if kind == 'no_attention':
        for b in model.blocks:
            if hasattr(b.attn, 'diagnostic_uniform_attention'):
                b.attn.diagnostic_uniform_attention = True
    opt=torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.01)
    rng=np.random.default_rng(SEED)
    history=[]
    best_val=float('inf')
    for step in range(1, steps+1):
        model.train(); opt.zero_grad(set_to_none=True)
        batch=sample_batch(train, 2, 48, rng).to(device)
        loss, step_losses=model.compute_loss(batch)
        loss.backward()
        grad_total=float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item())
        grad_router,_=flatten_grad_norm(model, lambda n: 'router' in n or 'attention_mixture' in n)
        grad_attn,_=flatten_grad_norm(model, lambda n: '.attn.' in n or 'attention_mixture' in n)
        opt.step(); model.update_routing_biases()
        with torch.no_grad():
            model.eval(); vb=sample_batch(val, 2, 48, rng).to(device); vloss, vsteps=model.compute_loss(vb)
        best_val=min(best_val,float(vloss.item()))
        if step==1 or step%10==0 or step==steps:
            s=summarize_model(model)
            history.append({
                'step':step,'loss':float(loss.item()),'val_loss':float(vloss.item()),
                'loop_losses':step_losses.cpu().tolist(),'grad_total':grad_total,
                'grad_router':grad_router,'grad_attn':grad_attn,'diag':s,
            })
    out.mkdir(parents=True, exist_ok=True)
    torch.save({'model_state_dict':model.state_dict(),'config':cfg.__dict__,'history':history}, out/'checkpoint.pt')
    return {'kind':kind,'best_val':best_val,'final':history[-1],'history':history,'params':sum(p.numel() for p in model.parameters())}


def generate(model, prompt: str, steps=160, temperature=0.8):
    model.eval(); ids=torch.tensor([[b for b in prompt.encode('utf-8')]],dtype=torch.long)
    ids=ids[:, -48:]
    out=ids.clone()
    for _ in range(steps):
        logits,_=model.forward(out, max_loops=model.cfg.max_loops)
        lg=logits[-1][:,-1]/temperature
        probs=F.softmax(lg,dim=-1)
        nxt=torch.multinomial(probs,1)
        out=torch.cat([out,nxt],dim=1)
        out=out[:,-48:]
    return bytes(int(x) for x in out[0].tolist()).decode('utf-8',errors='replace')


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--repo',type=Path,default=Path('.'))
    ap.add_argument('--out',type=Path,default=Path('diagnostic_v9'))
    ap.add_argument('--steps',type=int,default=50)
    args=ap.parse_args()
    seed_all()
    text=collect_repo_text(args.repo)
    train,val=write_byte_corpus(text,args.out/'corpus')
    results={}
    for kind in ['full','no_attention','no_activation','no_moe_specialization','depth1']:
        print(f'=== {kind} ===',flush=True)
        results[kind]=train_one(kind,train,val,args.out/kind,args.steps)
        print('best_val',results[kind]['best_val'],flush=True)
    # full generation from in-memory reload
    full=results['full']
    cfg=make_cfg('full'); model=LoopTransformer(cfg)
    ck=torch.load(args.out/'full'/'checkpoint.pt',weights_only=True)
    model.load_state_dict(ck['model_state_dict'])
    sample=generate(model,'# FILE: synthetic\nclass LoopTransformer:',steps=160)
    (args.out/'full'/'sample.txt').write_text(sample,'utf-8')
    # analysis report
    lines=['# LoopLLM component diagnosis v9','',f'Corpus bytes: {len(train)+len(val):,}',f'Train bytes: {len(train):,}',f'Val bytes: {len(val):,}','']
    lines.append('| Experiment | Params | Best val loss | Final val loss | Final loop losses |')
    lines.append('|---|---:|---:|---:|---|')
    for k,r in results.items():
        fl=r['final']; ll=', '.join(f'{x:.4f}' for x in fl['loop_losses'])
        lines.append(f"| {k} | {r['params']:,} | {r['best_val']:.4f} | {fl['val_loss']:.4f} | {ll} |")
    full_hist=results['full']['history']
    lines += ['', '## Full model component observations','']
    last=full_hist[-1]['diag']
    lines.append(f"- Loop improvements: {last['loop_improvements']}")
    lines.append(f"- Loop state cosine: {last['loop_state_cosines']}")
    lines.append(f"- Recurrent update means: {last['recurrent_update_means']}")
    lines.append(f"- Activation mean: {last['activation_mean']}")
    lines.append(f"- MoE load: {last['moe_load']}")
    lines.append(f"- Attention routing: {last['attention']}")
    lines.append(f"- Auxiliary losses: {last['aux']}")
    lines.append(f"- Gradient norms: total={full_hist[-1]['grad_total']:.4f}, router={full_hist[-1]['grad_router']:.4f}, attention={full_hist[-1]['grad_attn']:.4f}")
    lines += ['', '## Generation sample','', '```text', sample, '```']
    lines += ['', '## Interpretation','',
              'Positive contribution is inferred from lower validation loss in the full model versus the ablated control under the same data/seed/steps. This is an engineering ablation, not a causal proof at production scale.',
              'Depth usefulness is checked from per-loop validation losses, loop improvements, recurrent state cosine, and recurrent update means.',
              'Activation/MoE/attention usefulness is checked from ablation deltas, routing distributions, auxiliary losses, entropy/load, and non-zero gradients.']
    (args.out/'REPORT.md').write_text('\n'.join(lines), 'utf-8')
    (args.out/'results.json').write_text(json.dumps(results,indent=2), 'utf-8')
    print('REPORT',args.out/'REPORT.md')

if __name__=='__main__': main()
