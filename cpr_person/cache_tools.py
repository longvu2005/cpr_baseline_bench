"""Inspect mounted assets or copy only fully verified shared-stage caches."""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

from .cache import StageCache, atomic_file, read_json, resolve
from .detection import detection_fingerprint, load_detections
from .localization import localization_fingerprint, validate_localization


ASSET_NAMES = {"groundingdino_swint_ogc.pth", "ViT-B-32.pt", "tuned_recall_at1_step.pt"}
CACHE_NAMES = {"persons.npz", "members.json", "person_candidates.jsonl", "person_detections.npz",
               "query_localization.json", "meta.json", "metadata.json", "run.json"}


def inspect_roots(roots):
    records = []
    for root in roots:
        root = Path(root)
        if not root.is_dir():
            records.append({"root": str(root), "error": "not a directory"})
            continue
        for directory, dirs, files in os.walk(root):
            dirs[:] = [d for d in dirs if not d.startswith((".git", ".venv", "__pycache__"))
                       and d not in {"site-packages", "node_modules"}]
            for name in files:
                if name not in ASSET_NAMES | CACHE_NAMES:
                    continue
                path = Path(directory) / name
                row = {"path": str(path), "bytes": path.stat().st_size}
                if name in ASSET_NAMES:
                    row["kind"] = "checkpoint_candidate_verify_checksum_before_use"
                elif name in {"persons.npz", "members.json"}:
                    meta = read_json(path.with_name(name + ".meta.json"))
                    row["kind"] = "shared_cache_candidate" if isinstance(meta,dict) and "fingerprint" in meta else "unverified"
                    row["version"] = (meta or {}).get("fingerprint",{}).get("adapter_version")
                else:
                    row["kind"] = "legacy_or_other_metadata_no_automatic_reuse"
                records.append(row)
    return records


def import_shared(source_root, gallery_manifest, query_manifest=None, *, cfg=None):
    """No model inference. Incompatible/legacy cache formats fail closed."""
    gallery_manifest = resolve(gallery_manifest)
    gallery = [json.loads(line) for line in gallery_manifest.read_text().splitlines() if line.strip()]
    fp = detection_fingerprint(gallery_manifest, cfg)
    cache = StageCache("detections", fp, read_roots=[source_root])
    d = load_detections(cache,len(gallery))
    if d is None:
        raise ValueError("No equivalent verified shared detections; legacy/DINOv3 tensors are not converted by shape")
    destination = cache.directory / "persons.npz"
    if d.path != destination:
        with atomic_file(destination) as output, d.path.open("rb") as source:
            shutil.copyfileobj(source, output)
        cache.seal(destination)
    result = {"detections": str(destination)}
    if query_manifest is not None:
        query_manifest = resolve(query_manifest)
        queries = [json.loads(line) for line in query_manifest.read_text().splitlines() if line.strip()]
        loc_cache = StageCache("query_localization", localization_fingerprint(query_manifest,d,cfg), read_roots=[source_root])
        path = loc_cache.find("members.json")
        if path is None:
            raise ValueError("No equivalent verified shared Subject memberships")
        payload = read_json(path)
        validate_localization(payload["members"], queries, gallery, d)
        destination = loc_cache.directory / "members.json"
        if path != destination:
            with atomic_file(destination) as output, path.open("rb") as source:
                shutil.copyfileobj(source, output)
            loc_cache.seal(destination)
        result["query_localization"] = str(destination)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="command",required=True)
    inspect=sub.add_parser("inspect");inspect.add_argument("roots",nargs="+")
    migrate=sub.add_parser("import-shared")
    migrate.add_argument("source_root");migrate.add_argument("--gallery-manifest",default="data/gallery.jsonl")
    migrate.add_argument("--query-manifest")
    args=parser.parse_args()
    if args.command=="inspect":
        result=inspect_roots(args.roots)
    else:
        result=import_shared(args.source_root,args.gallery_manifest,args.query_manifest)
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()
