#!/usr/bin/env python3
"""S11: S6 full-gallery retrieval followed by Qwen2.5-VL top-K reranking."""
from __future__ import annotations
import argparse, gc, hashlib, json, os, re, sys
from pathlib import Path
from typing import Any, Sequence
import numpy as np
import torch
import yaml
from PIL import Image
ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from benchmark_progress import PhaseTracker, progress_bar  # noqa: E402
DEFAULT_CONFIG=Path(__file__).resolve().parent/'config.yaml'
METHOD_ID='retrieve_qwen25vl_rerank'; ADAPTER_VERSION='2026-08-13-v1-rank-safe-topk'; CACHE_SCHEMA=1; SELECT_SCHEMA=1

def load_yaml(p):
    with p.open('r',encoding='utf-8') as f: x=yaml.safe_load(f)
    if not isinstance(x,dict): raise TypeError(f'Expected YAML mapping: {p}')
    return x

def load_jsonl(p):
    out=[]
    with p.open('r',encoding='utf-8') as f:
        for n,l in enumerate(f,1):
            if l.strip():
                x=json.loads(l)
                if not isinstance(x,dict): raise TypeError(f'{p}:{n}: row must be object')
                out.append(x)
    return out

def resolve(v):
    p=Path(v); return (p if p.is_absolute() else ROOT/p).resolve()
def rel(p):
    try:return str(p.resolve().relative_to(ROOT.resolve()))
    except ValueError:return str(p.resolve())
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        while True:
            b=f.read(8*1024*1024)
            if not b: break
            h.update(b)
    return h.hexdigest()
def read_json(p):
    try:x=json.loads(p.read_text(encoding='utf-8'))
    except Exception:return None
    return x if isinstance(x,dict) else None
def write_json(p,x):
    p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n',encoding='utf-8')
def gallery_index(gallery):
    d={}
    for i,r in enumerate(gallery):
        k=r.get('image_id')
        if k in d: raise ValueError(f'Duplicate gallery image_id {k!r}')
        d[k]=i
    return d
def image_path(row,i):
    v=row.get('path')
    if not isinstance(v,str) or not v: raise KeyError(f'Gallery row {i} has no path')
    p=resolve(v)
    if not p.is_file(): raise FileNotFoundError(p)
    return p
def validate_scores(x,shape,label):
    if x.shape!=shape: raise ValueError(f'{label} shape={x.shape}, expected={shape}')
    if not np.issubdtype(x.dtype,np.floating): raise TypeError(f'{label} must be floating')
    for s in range(0,shape[0],256):
        if not np.isfinite(np.asarray(x[s:s+256])).all(): raise ValueError(f'{label} contains NaN/Inf')
def ensure_val(main_g,main_q,val_g,val_q):
    for p in (val_g,val_q):
        if not p.is_file(): raise FileNotFoundError(f'Missing validation manifest: {rel(p)}')
    if sha(main_g)==sha(val_g) and sha(main_q)==sha(val_q): raise RuntimeError('Validation manifests equal evaluation manifests; refusing leakage.')
def validate_qwen(cfg):
    v=cfg['verifier']; d=resolve(v['checkpoint_dir']); m=resolve(v['prepared_marker'])
    data=read_json(m)
    if data is None or data.get('repo_id')!=v['repo_id'] or data.get('revision')!=v['revision']:
        raise RuntimeError('Pinned Qwen prepared marker missing/stale; run checkpoint preparation.')
    for item in data.get('files',[]):
        p=d/str(item.get('path',''))
        if not p.is_file() or p.stat().st_size!=item.get('size'): raise RuntimeError(f'Incomplete Qwen file: {rel(p)}')
    return d,m
def qdtype(n):
    d={'bfloat16':torch.bfloat16,'bf16':torch.bfloat16,'float16':torch.float16,'fp16':torch.float16,'float32':torch.float32}
    if n not in d: raise ValueError(f'Unsupported qwen dtype {n}')
    return d[n]
def prompt(cfg,instruction):
    t=str(cfg['verifier']['prompt_template'])
    if t.count('{instruction}')!=1: raise ValueError('Verifier prompt must contain exactly one {instruction}')
    x=str(instruction or '').strip()
    if not x: raise ValueError('Empty query instruction')
    return t.replace('{instruction}',x)
def parse_score(text):
    m=re.search(r'(?im)^\s*SCORE\s*=\s*(\d{1,3})\s*$',text.strip())
    if not m: raise RuntimeError(f'Qwen verifier output is not SCORE=N: {text!r}')
    v=int(m.group(1))
    if not 0<=v<=100: raise RuntimeError(f'Qwen verifier score out of range: {v}')
    return v/100.0

def s6_sources(cfg,main_g,main_q,val_g,val_q):
    f=cfg['first_stage']; main_scores=resolve(f['scores']); run=read_json(resolve(f['run'])); alpha=read_json(resolve(f['alpha_selection']))
    if not main_scores.is_file() or run is None or run.get('method')!=f['method']:
        raise FileNotFoundError('S11 requires a completed S6 run. Run `python run_baseline.py groundingdino_clipreid_set_text` first.')
    if alpha is None or not isinstance(alpha.get('selected_alpha'),(int,float)): raise RuntimeError('Missing/stale S6 alpha_selection.json')
    afp=alpha.get('fingerprint')
    if not isinstance(afp,dict) or afp.get('validation_gallery_sha256')!=sha(val_g) or afp.get('validation_queries_sha256')!=sha(val_q):
        raise RuntimeError('S6 alpha selection was not produced from the configured validation manifests.')
    run_alpha=(run.get('fusion') or {}).get('selected_alpha')
    if not isinstance(run_alpha,(int,float)) or abs(float(run_alpha)-float(alpha['selected_alpha']))>1e-12:
        raise RuntimeError('S6 run.json and alpha_selection.json disagree on selected alpha.')
    mg=load_jsonl(main_g); mq=load_jsonl(main_q); x=np.load(main_scores,mmap_mode='r',allow_pickle=False); validate_scores(x,(len(mq),len(mg)),'S6 main scores')
    vr=np.load(resolve(f['validation']['reid_scores']),mmap_mode='r',allow_pickle=False); vt=np.load(resolve(f['validation']['clip_text_scores']),mmap_mode='r',allow_pickle=False)
    vg=load_jsonl(val_g); vq=load_jsonl(val_q); shape=(len(vq),len(vg)); validate_scores(vr,shape,'S6 val ReID'); validate_scores(vt,shape,'S6 val CLIP')
    a=float(alpha['selected_alpha']); return x,vr,vt,a,mg,mq,vg,vq

def prepare_val_s6(cfg,vr,vt,a,shape):
    p=resolve(cfg['cache']['validation_first_stage_scores']); meta=p.with_suffix(p.suffix+'.meta.json')
    expected={'schema':CACHE_SCHEMA,'adapter':ADAPTER_VERSION,'reid_sha':sha(resolve(cfg['first_stage']['validation']['reid_scores'])),'clip_sha':sha(resolve(cfg['first_stage']['validation']['clip_text_scores'])),'alpha':a}
    if p.is_file() and read_json(meta)==expected:
        x=np.load(p,mmap_mode='r',allow_pickle=False)
        if x.shape==shape:return x
    p.parent.mkdir(parents=True,exist_ok=True); out=np.lib.format.open_memmap(p,'w+',dtype=np.float32,shape=shape)
    for s in range(0,shape[0],128): out[s:s+128]=a*np.asarray(vr[s:s+128],np.float32)+(1-a)*np.asarray(vt[s:s+128],np.float32)
    out.flush(); write_json(meta,expected); return np.load(p,mmap_mode='r',allow_pickle=False)

def verifier_fingerprint(cfg,manifest_g,manifest_q,first_stage_path,marker,max_k):
    v=cfg['verifier']; return {'schema':CACHE_SCHEMA,'adapter':ADAPTER_VERSION,'gallery_sha':sha(manifest_g),'query_sha':sha(manifest_q),'first_stage_sha':sha(first_stage_path),'qwen_marker_sha':sha(marker),'repo_id':v['repo_id'],'revision':v['revision'],'processor':v['processor'],'generation':v['generation'],'prompt':v['prompt_template'],'max_k':int(max_k)}

@torch.inference_mode()
def qwen_verify_cache(cfg,gallery,queries,first_scores,first_stage_path,manifest_g,manifest_q,checkpoint,marker,cache_path,max_k,device):
    fp=verifier_fingerprint(cfg,manifest_g,manifest_q,first_stage_path,marker,max_k); meta=cache_path.with_suffix(cache_path.suffix+'.meta.json')
    if cache_path.is_file() and read_json(meta)==fp:
        z=np.load(cache_path,allow_pickle=False); idx=np.asarray(z['indices'],np.int64); sc=np.asarray(z['scores'],np.float32)
        if idx.shape==(len(queries),max_k) and sc.shape==idx.shape and np.isfinite(sc).all():
            print(f'Using Qwen verifier cache: {rel(cache_path)}',flush=True); return idx,sc
    if device.type!='cuda': raise RuntimeError('S11 Qwen verifier requires CUDA')
    os.environ['HF_HUB_OFFLINE']='1'; os.environ['TRANSFORMERS_OFFLINE']='1'; os.environ['HF_DATASETS_OFFLINE']='1'
    from transformers import AutoProcessor,Qwen2_5_VLForConditionalGeneration
    from qwen_vl_utils import process_vision_info
    v=cfg['verifier']; processor=AutoProcessor.from_pretrained(str(checkpoint),min_pixels=int(v['processor']['min_pixels']),max_pixels=int(v['processor']['max_pixels']),local_files_only=True)
    model=Qwen2_5_VLForConditionalGeneration.from_pretrained(str(checkpoint),torch_dtype=qdtype(str(cfg['runtime']['qwen_dtype'])),attn_implementation=str(cfg['runtime']['qwen_attn_implementation']),local_files_only=True).to(device); model.eval()
    gidx=gallery_index(gallery); indices=np.empty((len(queries),max_k),np.int64); scores=np.empty((len(queries),max_k),np.float32)
    for qi in progress_bar(range(len(queries)),desc=f'Qwen verify top-{max_k}',total=len(queries),unit='query'):
        order=np.argsort(-np.asarray(first_scores[qi]),kind='stable'); top=order[:max_k]; indices[qi]=top
        ref_i=gidx[queries[qi]['image_id']]; ref=image_path(gallery[ref_i],ref_i); inst=prompt(cfg,queries[qi].get('text'))
        for j,gi in enumerate(top):
            cand=image_path(gallery[int(gi)],int(gi)); messages=[{'role':'user','content':[{'type':'image','image':ref.as_uri()},{'type':'image','image':cand.as_uri()},{'type':'text','text':inst}]}]
            rendered=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True); images,videos=process_vision_info(messages)
            inputs=processor(text=[rendered],images=images,videos=videos,padding=True,return_tensors='pt').to(device)
            gen=model.generate(**inputs,max_new_tokens=int(v['generation']['max_new_tokens']),do_sample=False,num_beams=1,repetition_penalty=float(v['generation']['repetition_penalty']),use_cache=True)
            out=processor.batch_decode(gen[:,inputs.input_ids.shape[1]:],skip_special_tokens=True,clean_up_tokenization_spaces=False)[0]
            scores[qi,j]=parse_score(out)
    cache_path.parent.mkdir(parents=True,exist_ok=True); np.savez(cache_path,indices=indices,scores=scores); write_json(meta,fp)
    del model,processor; gc.collect(); torch.cuda.empty_cache(); return indices,scores

def rank_prior(k):
    if k==1:return np.ones(1,np.float32)
    return (1.0-np.arange(k,dtype=np.float32)/(k-1)).astype(np.float32)
def reranked_order(first_row,top_indices,verifier,k,w):
    base=np.argsort(-np.asarray(first_row),kind='stable'); expected=base[:k]
    if not np.array_equal(expected,np.asarray(top_indices[:k])): raise RuntimeError('Verifier cache top-K does not match first-stage ranking')
    fused=w*rank_prior(k)+(1-w)*np.asarray(verifier[:k],np.float32); local=np.argsort(-fused,kind='stable')
    return np.concatenate([expected[local],base[k:]])
def ap_from_order(order,query,gidx):
    self_i=gidx[query['image_id']]; order=order[order!=self_i]; pos={gidx[x] for x in query['full_positive_ids'] if x in gidx}; pos.discard(self_i)
    if not pos: raise ValueError(f"{query.get('query_id')}: no validation Full positives")
    ranks=np.flatnonzero(np.isin(order,list(pos)))+1
    if not len(ranks): raise ValueError('No positive after ranking')
    return float(np.mean(np.arange(1,len(ranks)+1)/ranks))
def select_params(cfg,val_scores,indices,verifier,gallery,queries):
    p=resolve(cfg['rerank']['result']); expected={'schema':SELECT_SCHEMA,'adapter':ADAPTER_VERSION,'val_scores_sha':sha(resolve(cfg['cache']['validation_first_stage_scores'])),'verifier_sha':sha(resolve(cfg['cache']['validation_verifier'])),'top_k_grid':[int(x) for x in cfg['rerank']['top_k_grid']],'weight_grid':[float(x) for x in cfg['rerank']['first_stage_weight_grid']]}
    cur=read_json(p)
    if cur and all(cur.get(k)==v for k,v in expected.items()): return int(cur['selected_top_k']),float(cur['selected_first_stage_weight']),cur
    gidx=gallery_index(gallery); best=None; trials=[]
    for k in sorted(expected['top_k_grid']):
        for w in sorted(expected['weight_grid'],reverse=True):
            aps=[]
            for qi,q in enumerate(queries): aps.append(ap_from_order(reranked_order(val_scores[qi],indices[qi],verifier[qi],k,w),q,gidx))
            m=float(np.mean(aps)); trials.append({'top_k':k,'first_stage_weight':w,'Full-mAP':m})
            key=(m,-k,w)
            if best is None or key>best[0]: best=(key,k,w,m)
    payload={**expected,'selected_top_k':best[1],'selected_first_stage_weight':best[2],'selected_full_map':best[3],'trials':trials,'tie_break':cfg['rerank']['tie_break']}; write_json(p,payload); return best[1],best[2],payload

def write_final(first_scores,indices,verifier,k,w,out_path):
    q,n=first_scores.shape; out=np.lib.format.open_memmap(out_path,'w+',dtype=np.float32,shape=(q,n)); rank_scores=np.arange(n,0,-1,dtype=np.float32)
    for qi in progress_bar(range(q),desc='Write reranked full rankings',total=q,unit='query'):
        order=reranked_order(first_scores[qi],indices[qi],verifier[qi],k,w); row=np.empty(n,np.float32); row[order]=rank_scores; out[qi]=row
    out.flush(); return out

def main():
    tracker=PhaseTracker(METHOD_ID,total=7)
    with tracker.phase('Load config and validate S6/validation artifacts'):
        p=argparse.ArgumentParser(); p.add_argument('--config',default=str(DEFAULT_CONFIG)); args=p.parse_args(); cfg=load_yaml(resolve(args.config)); config_path=resolve(args.config)
        mg=resolve(cfg['data']['gallery_manifest']); mq=resolve(cfg['data']['query_manifest']); vg=resolve(cfg['rerank']['validation']['gallery_manifest']); vq=resolve(cfg['rerank']['validation']['query_manifest']); ensure_val(mg,mq,vg,vq)
        main,vr,vt,a,gallery,queries,val_gallery,val_queries=s6_sources(cfg,mg,mq,vg,vq); val=prepare_val_s6(cfg,vr,vt,a,(len(val_queries),len(val_gallery)))
        device=torch.device(str(cfg['runtime']['device'])); checkpoint,marker=validate_qwen(cfg); maxk=max(int(x) for x in cfg['rerank']['top_k_grid'])
        if maxk>len(val_gallery) or maxk>len(gallery): raise ValueError('top-K exceeds gallery size')
        tracker.log(f'S6_alpha={a:.2f} max_K={maxk} main={main.shape} val={val.shape}')
    with tracker.phase('Verify validation top-K candidates with Qwen'):
        vi,vs=qwen_verify_cache(cfg,val_gallery,val_queries,val,resolve(cfg['cache']['validation_first_stage_scores']),vg,vq,checkpoint,marker,resolve(cfg['cache']['validation_verifier']),maxk,device)
    with tracker.phase('Select top-K and fusion weight on validation Full-mAP'):
        k,w,sel=select_params(cfg,val,vi,vs,val_gallery,val_queries); tracker.log(f'selected_K={k} first_stage_weight={w:.2f} val_Full-mAP={sel["selected_full_map"]:.6f}')
    with tracker.phase('Verify main top-K candidates with Qwen'):
        mi,ms=qwen_verify_cache(cfg,gallery,queries,main,resolve(cfg['first_stage']['scores']),mg,mq,checkpoint,marker,resolve(cfg['cache']['main_verifier']),k,device)
    with tracker.phase('Construct rank-safe complete full-gallery scores'):
        od=resolve(cfg['output']['dir']); od.mkdir(parents=True,exist_ok=True); scores=write_final(main,mi,ms,k,w,od/'scores.npy'); validate_scores(scores,main.shape,'final scores')
    with tracker.phase('Validate ranking invariants'):
        for qi in range(min(32,len(queries))):
            base=np.argsort(-np.asarray(main[qi]),kind='stable'); final=np.argsort(-np.asarray(scores[qi]),kind='stable')
            if set(base[:k])!=set(final[:k]): raise RuntimeError('Top-K membership invariant violated')
            if not np.array_equal(base[k:],final[k:]): raise RuntimeError('Outside-top-K order invariant violated')
        tracker.log('top-K membership preserved; outside-top-K order preserved')
    with tracker.phase('Write run metadata'):
        od=resolve(cfg['output']['dir']); payload={'method':cfg['method'],'display_name':cfg['display_name'],'group':cfg['group'],'cpr_supervision':cfg['cpr_supervision'],'adapter_version':ADAPTER_VERSION,'first_stage_method':cfg['first_stage']['method'],'verifier':{'repo_id':cfg['verifier']['repo_id'],'revision':cfg['verifier']['revision']},'selected_top_k':k,'selected_first_stage_weight':w,'selection':rel(resolve(cfg['rerank']['result'])),'score_semantics':'strictly_descending_rank_surrogate','outside_top_k_order_preserved':True,'query_image_removed_inside_method':False,'config':rel(config_path),'num_queries':len(queries),'num_gallery':len(gallery),'scores':rel(od/'scores.npy'),'higher_is_better':True}; write_json(od/'run.json',payload); tracker.log(f'scores={rel(od/"scores.npy")}')
    tracker.finish()
if __name__=='__main__': main()
