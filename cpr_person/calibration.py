"""Freeze an unmatched score from separate, explicitly noncorresponding pairs."""
from __future__ import annotations

import argparse
import json

import numpy as np

from .cache import atomic_json, read_json, resolve, sha256_file
from .validation import ensure_separate_queries, public_query_signature


def load_rows(path):
    return [json.loads(line) for line in resolve(path).read_text().splitlines() if line.strip()]


def freeze(scores_path, query_manifest, gallery_manifest, checkpoint, source_commit, dataset_id,
           output, *, benchmark_queries="data/queries.jsonl", benchmark_gallery="data/gallery.jsonl", quantile=.01):
    ensure_separate_queries(benchmark_queries, query_manifest)
    if not dataset_id.strip() or not 0<=quantile<=1:
        raise ValueError("Need an external dataset ID and a quantile in [0,1]")
    source_gallery=load_rows(gallery_manifest)
    # Portable public references + real paths reject a renamed manifest over
    # the same evaluation images. No benchmark positive/identity labels enter.
    refs={str(resolve(row["path"])) for row in source_gallery}
    if refs & {str(resolve(row["path"])) for row in load_rows(benchmark_gallery)}:
        raise ValueError("Calibration gallery contains benchmark evaluation images")
    values=np.load(resolve(scores_path),allow_pickle=False)
    if not values.size or not np.isfinite(values).all() or np.any(values>1e-5):
        raise ValueError("Expected finite native negative-squared-Euclidean scores of noncorresponding pairs")
    artifact={"schema":"external-instruct-unmatched-v1","dataset_id":dataset_id,
        "pairwise_metric":"negative_squared_euclidean", "unmatched_score":float(np.quantile(values,quantile)),
        "quantile":float(quantile),"noncorresponding_pairs_attested":True,
        "source_commit":source_commit,"checkpoint_sha256":sha256_file(resolve(checkpoint)),
        "pair_scores_sha256":sha256_file(resolve(scores_path)),
        "query_manifest_sha256":sha256_file(resolve(query_manifest)),
        "gallery_manifest_sha256":sha256_file(resolve(gallery_manifest)),
        "public_query_signatures":sorted(public_query_signature(q) for q in load_rows(query_manifest)),
        "gallery_real_paths":sorted(refs),"evaluation_labels_used":False}
    atomic_json(resolve(output),artifact)
    return artifact


def apply_instruct_calibration(cfg, checkpoint, query_manifest, gallery_manifest):
    path=cfg["setmatch"].get("unmatched_calibration")
    if not path:
        return
    artifact=read_json(resolve(path))
    if (not isinstance(artifact,dict) or artifact.get("schema")!="external-instruct-unmatched-v1"
            or artifact.get("checkpoint_sha256")!=sha256_file(checkpoint)
            or artifact.get("source_commit")!=cfg["source"]["commit"]
            or artifact.get("pairwise_metric")!="negative_squared_euclidean"
            or artifact.get("evaluation_labels_used") is not False
            or artifact.get("noncorresponding_pairs_attested") is not True):
        raise ValueError("Incompatible external Instruct-ReID unmatched calibration")
    if ({public_query_signature(q) for q in load_rows(query_manifest)} & set(artifact["public_query_signatures"])
            or {str(resolve(g["path"])) for g in load_rows(gallery_manifest)} & set(artifact["gallery_real_paths"])):
        raise ValueError("Calibration inputs overlap the current evaluation inputs")
    value=float(artifact["unmatched_score"])
    if not np.isfinite(value) or value>0:
        raise ValueError("Invalid native distance-space unmatched score")
    cfg["setmatch"]["unmatched_score"]=value
    cfg["setmatch"]["unmatched_score_source"]="frozen_quantile_of_separate_noncorresponding_Instruct_ReID_pairs"
    cfg["setmatch"]["calibration_provenance"]={"artifact_sha256":sha256_file(resolve(path)),
        "dataset_id":artifact["dataset_id"],"quantile":artifact["quantile"],
        "checkpoint_sha256":artifact["checkpoint_sha256"],"pair_scores_sha256":artifact["pair_scores_sha256"]}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ["scores","queries","gallery","checkpoint","source-commit","dataset-id","output"]:
        p.add_argument("--"+name,required=True)
    p.add_argument("--quantile",type=float,default=.01)
    p.add_argument("--noncorresponding-pairs",action="store_true",required=True,
                   help="Attest that the external score file contains only noncorresponding person pairs")
    a=p.parse_args()
    artifact=freeze(a.scores,a.queries,a.gallery,a.checkpoint,a.source_commit,a.dataset_id,a.output,quantile=a.quantile)
    print(json.dumps({"unmatched_score":artifact["unmatched_score"],"dataset_id":artifact["dataset_id"]}))


if __name__=="__main__":
    main()
