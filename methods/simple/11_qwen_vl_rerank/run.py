#!/usr/bin/env python3
"""S11: S6 full-gallery retrieval followed by Qwen2.5-VL top-K reranking."""
from __future__ import annotations
import argparse, gc, hashlib, json, os, re, sys
from pathlib import Path
from statistics import fmean
from tempfile import TemporaryDirectory
import numpy as np
import torch
import yaml
ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from benchmark_data import manifest_fingerprint
from cpr_person.cache import atomic_json
from cpr_person.protocol import require_aligned_run
from cpr_person.validation import verify_tuning_manifests
from evaluate import average_precision, full_relevance, rank_scores
from benchmark_progress import PhaseTracker, progress_bar  # noqa: E402
DEFAULT_CONFIG=Path(__file__).resolve().parent/'config.yaml'
METHOD_ID = "retrieve_qwen25vl_rerank"
ADAPTER_VERSION = "rcr-val-test-nonself-topk-v2"
CACHE_SCHEMA = 2
SELECT_SCHEMA = 2

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
    from cpr_person.adapter import require_person_branch
    require_aligned_run(run, f["method"])
    require_person_branch(run, main_g, main_q)
    if alpha is None or not isinstance(alpha.get('selected_alpha'),(int,float)): raise RuntimeError('Missing/stale S6 alpha_selection.json')
    afp=alpha.get('fingerprint')
    if not isinstance(afp,dict) or afp.get('validation_gallery_sha256')!=sha(val_g) or afp.get('validation_queries_sha256')!=sha(val_q):
        raise RuntimeError('S6 alpha selection was not produced from the configured validation manifests.')
    if alpha.get("selection_split") != "val" or run.get("validation_fingerprint") != manifest_fingerprint(ROOT, "val"):
        raise RuntimeError("Run S6 with selection on the current RCR val split first")
    for kind, key in (("reid_scores", "validation_reid_scores_sha256"),
                      ("clip_text_scores", "validation_clip_scores_sha256")):
        if afp.get(key) != sha(resolve(f["validation"][kind])):
            raise RuntimeError(f"S6 validation {kind} differ from the frozen selection")
    run_alpha=(run.get('fusion') or {}).get('selected_alpha')
    if not isinstance(run_alpha,(int,float)) or abs(float(run_alpha)-float(alpha['selected_alpha']))>1e-12:
        raise RuntimeError('S6 run.json and alpha_selection.json disagree on selected alpha.')
    mg=load_jsonl(main_g); mq=load_jsonl(main_q); x=np.load(main_scores,mmap_mode='r',allow_pickle=False); validate_scores(x,(len(mq),len(mg)),'S6 main scores')
    vr=np.load(resolve(f['validation']['reid_scores']),mmap_mode='r',allow_pickle=False); vt=np.load(resolve(f['validation']['clip_text_scores']),mmap_mode='r',allow_pickle=False)
    vg=load_jsonl(val_g); vq=load_jsonl(val_q); shape=(len(vq),len(vg)); validate_scores(vr,shape,'S6 val ReID'); validate_scores(vt,shape,'S6 val CLIP')
    a=float(alpha['selected_alpha']); return x,vr,vt,a,mg,mq,vg,vq,run

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
        ref_i = gidx[queries[qi]["image_id"]]
        top = rank_scores(first_scores[qi], ref_i)[:max_k]
        indices[qi] = top
        ref = image_path(gallery[ref_i], ref_i)
        inst = prompt(cfg, queries[qi].get("text"))
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
    return np.linspace(1.0, 0.0, k, dtype=np.float32)


def reranked_order(base, top_indices, verifier, k, weight):
    top = base[:k]
    if not np.array_equal(top, top_indices[:k]):
        raise RuntimeError("Verifier cache top-K does not match the eligible first-stage ranking")
    fused = weight * rank_prior(k) + (1.0 - weight) * verifier[:k]
    local = np.argsort(-fused, kind="stable")
    return np.concatenate([top[local], base[k:]])


def select_params(cfg, val_scores, indices, verifier, gallery, queries, validation_fingerprint):
    rerank = cfg["rerank"]
    path = resolve(rerank["result"])
    expected = {
        "schema": SELECT_SCHEMA,
        "adapter": ADAPTER_VERSION,
        "selection_split": "val",
        "validation_fingerprint": validation_fingerprint,
        "val_scores_sha": sha(resolve(cfg["cache"]["validation_first_stage_scores"])),
        "verifier_sha": sha(resolve(cfg["cache"]["validation_verifier"])),
        "top_k_grid": sorted({int(x) for x in rerank["top_k_grid"]}),
        "weight_grid": sorted({float(x) for x in rerank["first_stage_weight_grid"]}, reverse=True),
    }
    if not expected["weight_grid"] or any(not 0.0 <= w <= 1.0 for w in expected["weight_grid"]):
        raise ValueError("Rerank weights must lie in [0, 1]")
    current = read_json(path)
    if current and all(current.get(key) == value for key, value in expected.items()):
        k, weight = current.get("selected_top_k"), current.get("selected_first_stage_weight")
        if k in expected["top_k_grid"] and weight in expected["weight_grid"]:
            return k, weight, current

    gallery_ids = gallery_index(gallery)
    relevance = [full_relevance(query, gallery_ids) for query in queries]
    bases = [rank_scores(row, self_index) for row, (self_index, _) in zip(val_scores, relevance, strict=True)]
    trials = []
    for k in expected["top_k_grid"]:
        for weight in expected["weight_grid"]:
            score = fmean(
                average_precision(reranked_order(base, top, values, k, weight), positives)
                for base, top, values, (_, positives) in zip(bases, indices, verifier, relevance, strict=True)
            )
            trials.append({"top_k": k, "first_stage_weight": weight, "Full-mAP": score})
    best = max(trials, key=lambda row: (row["Full-mAP"], -row["top_k"], row["first_stage_weight"]))
    result = {
        **expected,
        "selected_top_k": best["top_k"],
        "selected_first_stage_weight": best["first_stage_weight"],
        "selected_full_map": best["Full-mAP"],
        "trials": trials,
        "tie_break": rerank["tie_break"],
    }
    atomic_json(path, result)
    return best["top_k"], best["first_stage_weight"], result


def write_final(first_scores, indices, verifier, self_indices, k, weight, path):
    num_queries, num_gallery = first_scores.shape
    scores = np.arange(num_gallery - 1, 0, -1, dtype=np.float32)
    with TemporaryDirectory(prefix=".rerank-", dir=path.parent) as directory:
        temporary = Path(directory) / path.name
        output = np.lib.format.open_memmap(
            temporary, mode="w+", dtype=np.float32, shape=(num_queries, num_gallery)
        )
        try:
            for qi in progress_bar(
                range(num_queries), desc="Write reranked full rankings", total=num_queries, unit="query"
            ):
                base = rank_scores(first_scores[qi], self_indices[qi])
                order = reranked_order(base, indices[qi], verifier[qi], k, weight)
                output[qi, order] = scores
                output[qi, self_indices[qi]] = 0.0
            output.flush()
        finally:
            del output
        temporary.replace(path)
    return np.load(path, mmap_mode="r", allow_pickle=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    config_path = resolve(parser.parse_args().config)
    cfg = load_yaml(config_path)
    tracker = PhaseTracker(METHOD_ID, total=6)
    with tracker.phase("Load test and validate S6 sources"):
        test_gallery_path = resolve(cfg["data"]["gallery_manifest"])
        test_query_path = resolve(cfg["data"]["query_manifest"])
        val_gallery_path = resolve(cfg["rerank"]["validation"]["gallery_manifest"])
        val_query_path = resolve(cfg["rerank"]["validation"]["query_manifest"])
        validation_fingerprint = verify_tuning_manifests(
            ROOT, test_gallery_path, test_query_path, val_gallery_path, val_query_path
        )
        test_scores, reid, text, alpha, gallery, queries, val_gallery, val_queries, first_stage = s6_sources(
            cfg, test_gallery_path, test_query_path, val_gallery_path, val_query_path
        )
        val_scores = prepare_val_s6(cfg, reid, text, alpha, (len(val_queries), len(val_gallery)))
        device = torch.device(cfg["runtime"]["device"])
        checkpoint, marker = validate_qwen(cfg)
        ks = [int(k) for k in cfg["rerank"]["top_k_grid"]]
        if not ks or min(ks) < 1 or max(ks) >= min(len(val_gallery), len(gallery)):
            raise ValueError("Rerank top-K must fit both non-self galleries")

    with tracker.phase("Verify validation top-K candidates"):
        val_indices, val_verifier = qwen_verify_cache(
            cfg, val_gallery, val_queries, val_scores,
            resolve(cfg["cache"]["validation_first_stage_scores"]), val_gallery_path, val_query_path,
            checkpoint, marker, resolve(cfg["cache"]["validation_verifier"]), max(ks), device,
        )
    with tracker.phase("Select top-K and fusion weight on validation Full-mAP"):
        k, weight, selection = select_params(
            cfg, val_scores, val_indices, val_verifier, val_gallery, val_queries, validation_fingerprint
        )
        tracker.log(f"selected_K={k} weight={weight:.2f} val_Full-mAP={selection['selected_full_map']:.6f}")

    with tracker.phase("Verify test top-K candidates with frozen settings"):
        indices, verifier = qwen_verify_cache(
            cfg, gallery, queries, test_scores,
            resolve(cfg["first_stage"]["scores"]), test_gallery_path, test_query_path,
            checkpoint, marker, resolve(cfg["cache"]["main_verifier"]), k, device,
        )
    with tracker.phase("Write complete test rankings"):
        output = resolve(cfg["output"]["dir"])
        output.mkdir(parents=True, exist_ok=True)
        (output / "run.json").unlink(missing_ok=True)
        gallery_ids = gallery_index(gallery)
        self_indices = [gallery_ids[query["image_id"]] for query in queries]
        scores = write_final(test_scores, indices, verifier, self_indices, k, weight, output / "scores.npy")
        validate_scores(scores, test_scores.shape, "final test scores")

    with tracker.phase("Write run metadata"):
        atomic_json(output / "run.json", {
            "method": cfg["method"],
            "display_name": cfg["display_name"],
            "group": cfg["group"],
            "cpr_supervision": cfg["cpr_supervision"],
            "adapter_version": ADAPTER_VERSION,
            "first_stage_method": cfg["first_stage"]["method"],
            "scene_clip_sha256": first_stage["scene_clip_sha256"],
            "shared_person_protocol": first_stage["shared_person_protocol"],
            "verifier": {key: cfg["verifier"][key] for key in ("repo_id", "revision")},
            "selected_top_k": k,
            "selected_first_stage_weight": weight,
            "selection": rel(resolve(cfg["rerank"]["result"])),
            "validation_fingerprint": validation_fingerprint,
            "data_fingerprint": manifest_fingerprint(ROOT),
            "score_semantics": "strictly_descending_rank_surrogate",
            "outside_top_k_order_preserved": True,
            "query_image_excluded_from_shortlist": True,
            "config": rel(config_path),
            "num_queries": len(queries),
            "num_gallery": len(gallery),
            "scores": rel(output / "scores.npy"),
            "higher_is_better": True,
        })
    tracker.finish()


if __name__ == "__main__":
    main()
