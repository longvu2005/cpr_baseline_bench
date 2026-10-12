#!/usr/bin/env python3
"""Optional native-model smoke test on a small, explicitly non-reportable subset.

Install each selected method's requirements and checkpoints first. This script
does not install models, download weights, tune parameters, or run evaluation.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from run_baseline import discover_methods, resolve_method


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods",nargs="+",default=["fafa_setmatch"])
    parser.add_argument("--gallery-limit",type=int,default=40)
    parser.add_argument("--query-limit",type=int,default=4)
    args=parser.parse_args()
    if args.gallery_limit<30 or args.query_limit<1:
        parser.error("Use at least 30 gallery scenes (BASIC needs >=25 persons) and one query")
    load=lambda path:[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    full_gallery=load(ROOT/"data/gallery.jsonl")
    queries=load(ROOT/"data/queries.jsonl")[:args.query_limit]
    reference_ids={q["image_id"] for q in queries}
    refs=[g for g in full_gallery if g["image_id"] in reference_ids]
    gallery=refs+[g for g in full_gallery if g["image_id"] not in reference_ids][:max(0,args.gallery_limit-len(refs))]
    specs=discover_methods()
    with tempfile.TemporaryDirectory(prefix="cpr-native-smoke-") as temporary:
        directory=Path(temporary);gm,qm=directory/"gallery.jsonl",directory/"queries.jsonl"
        gm.write_text("".join(json.dumps(g)+"\n" for g in gallery))
        qm.write_text("".join(json.dumps(q)+"\n" for q in queries))
        for name in args.methods:
            spec=resolve_method(name,specs)
            cfg=yaml.safe_load(spec.config_path.read_text())
            cfg["data"]={"gallery_manifest":str(gm),"query_manifest":str(qm)}
            cfg["output"]={"dir":str(directory/spec.method_id)}
            # A smoke run uses fixed defaults only. No subset labels select alpha.
            for branch in ("composition","fusion"):
                selection=cfg.get(branch,{}).get("alpha_selection")
                if isinstance(selection,dict) and selection.get("mode")!="fixed":
                    raise ValueError(f"{name}: smoke test requires a separately prepared fixed-alpha config")
            cp=directory/(spec.method_id+".yaml");cp.write_text(yaml.safe_dump(cfg,sort_keys=False))
            subprocess.run([sys.executable,str(spec.run_path),"--config",str(cp)],cwd=ROOT,check=True)
            scores=np.load(directory/spec.method_id/"scores.npy",mmap_mode="r",allow_pickle=False)
            if scores.shape!=(len(queries),len(gallery)) or not np.isfinite(scores).all():
                raise ValueError(f"Invalid native smoke scores: {name}")
            print(f"NATIVE SMOKE PASS {name}: {scores.shape}; subset scores are not benchmark results",flush=True)


if __name__=="__main__":
    main()
