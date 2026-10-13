"""Full-gallery inference for the seven additions; evaluation stays external."""
from __future__ import annotations

import argparse
import os
import platform
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from time import perf_counter

import numpy as np

from benchmark_data import manifest_fingerprint
from benchmark_progress import progress_bar
from cpr_person.adapter import load_person_context, matching_metadata
from cpr_person.cache import ROOT, atomic_json, canonical_hash, resolve
from cpr_person.matching import score_feature_sets
from . import VERSION
from .backends import QwenGenerator, SentenceEncoder, build_backend
from .captions import CaptionStore, person_captions, scene_captions, structure_text
from .common import (cached_features, feature_cache, image_digest, load_config, load_jsonl,
                     public_inputs, query_text, seed_runtime, source_images, write_scene_scores)
from .prepare import validate_prepared
from .person_cache import require_shared_context


def environment_metadata(device):
    import torch
    packages = {}
    for package in ("torch", "torchvision", "transformers", "sentence-transformers", "numpy", "ftfy", "qwen-vl-utils"):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            pass
    return {"python": platform.python_version(), "packages": packages, "device": str(device),
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None}


def run_scene(cfg, gallery, queries, gallery_index, fingerprint, device, output, backend_factory=build_backend):
    runtime = cfg["runtime"]
    gallery_cache = feature_cache(cfg["method"] + "/gallery", fingerprint, runtime)
    query_fp = {**fingerprint, "public_queries": canonical_hash(queries), "composition": cfg["composition"]}
    query_cache = feature_cache(cfg["method"] + "/queries", query_fp, runtime)
    backend = None
    def get_backend():
        nonlocal backend
        if backend is None:
            backend = backend_factory(cfg, device)
        return backend
    gallery_features = cached_features(gallery_cache, "gallery.npy", len(gallery),
        lambda start, end: get_backend().images(source_images(gallery, start, end)),
        int(runtime["image_batch_size"]))
    references = [gallery_index[row["image_id"]] for row in queries]
    texts = [query_text(row, cfg["composition"]["text_field"]) for row in queries]
    def query_batch(start, end):
        images = source_images([gallery[index] for index in references[start:end]], 0, end - start)
        return get_backend().queries(images, texts[start:end],
                                     reference_features=gallery_features[references[start:end]])
    query_features = cached_features(query_cache, "queries.npy", len(queries), query_batch,
                                     int(runtime["query_batch_size"]))
    write_scene_scores(query_features, gallery_features, output / "scores.npy",
                       int(runtime["score_query_batch_size"]), int(runtime["score_gallery_batch_size"]))
    return {"composition": cfg["composition"], "retrieval": "full-gallery cosine",
            "reference_image_masked_by_inference": False}


def write_person_scores(query_features, gallery_features, query_offsets, gallery_offsets, path, cfg):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".partial")
    score = np.lib.format.open_memmap(temp, mode="w+", dtype=np.float32,
                                     shape=(len(query_offsets) - 1, len(gallery_offsets) - 1))
    try:
        for qi in progress_bar(range(len(score)), desc="Shared Partial SetMatch", unit="query"):
            score[qi] = score_feature_sets(query_features[query_offsets[qi]:query_offsets[qi + 1]],
                gallery_features, gallery_offsets, float(cfg["setmatch"]["unmatched_score"]),
                int(cfg["runtime"]["score_gallery_batch_size"]), metric="cosine")
        score.flush()
    finally:
        del score
    os.replace(temp, path)


def run_captions(cfg, gallery, queries, gallery_index, person, fingerprint, device, output,
                 generator_factory=None, text_encoder_factory=SentenceEncoder):
    store = CaptionStore(cfg, fingerprint["assets"],
                         generator_factory or (lambda: QwenGenerator(cfg, device)))
    if person is None:
        gallery_records, query_records = scene_captions(cfg, gallery, queries, gallery_index, store)
        gallery_text, query_texts = gallery_records, query_records
        query_offsets = None
    else:
        gallery_records, query_records, query_offsets = person_captions(cfg, gallery, queries, gallery_index, person, store)
        gallery_text = [structure_text(row) for row in gallery_records]
        query_texts = [structure_text(row) for row in query_records]
    atomic_json(output / "captions.json", {"gallery": gallery_records, "queries": query_records,
        "query_offsets": None if query_offsets is None else query_offsets.tolist(),
        "gallery_offsets": None if person is None else person.detections.offsets.tolist(),
        "gallery_image_ids": [row["image_id"] for row in gallery],
        "query_ids": [row["query_id"] for row in queries]})
    generation_metadata = store.metadata()
    store.release()
    encoder = None
    def get_encoder():
        nonlocal encoder
        if encoder is None:
            encoder = text_encoder_factory(cfg, device)
        return encoder
    def embed_records(stage, texts):
        cache = feature_cache(cfg["method"] + "/" + stage,
            {**fingerprint, "text_sha256": canonical_hash(texts), "text_encoding": cfg["text_encoding"]}, cfg["runtime"])
        return cached_features(cache, "features.npy", len(texts),
            lambda start, end: get_encoder().texts(texts[start:end]), int(cfg["runtime"]["text_batch_size"]))
    gallery_features = embed_records("gallery_text", gallery_text)
    query_features = embed_records("query_text", query_texts)
    if person is None:
        write_scene_scores(query_features, gallery_features, output / "scores.npy",
                           int(cfg["runtime"]["score_query_batch_size"]), int(cfg["runtime"]["score_gallery_batch_size"]))
    else:
        write_person_scores(query_features, gallery_features, query_offsets, person.detections.offsets,
                            output / "scores.npy", cfg)
    return {"generation": generation_metadata, "prompts": cfg["prompts"],
            "captioner_config": cfg["generation"], "text_encoding": cfg["text_encoding"],
            "reference_image_masked_by_inference": False}


def run(cfg):
    started = perf_counter()
    # The canonical evaluator reads these exact paths. Reject alternate manifests
    # instead of writing scores it would evaluate against different row ordering.
    gm, qm = resolve(cfg["data"]["gallery_manifest"]), resolve(cfg["data"]["query_manifest"])
    if gm != ROOT / "data/gallery.jsonl" or qm != ROOT / "data/queries.jsonl":
        raise ValueError("Use the canonical manifests required by evaluate.py")
    if resolve(cfg["output"]["dir"]) != ROOT / "runs" / cfg["method"]:
        raise ValueError("Output must be runs/<method>, as required by the evaluator")
    data_fingerprint = manifest_fingerprint(ROOT)
    raw_gallery, raw_queries = load_jsonl(gm), load_jsonl(qm)
    gallery, queries, index = public_inputs(raw_gallery, raw_queries)
    marker = validate_prepared(cfg)
    device = seed_runtime(cfg["runtime"])
    # Shared detector/selector are separate workers. Complete them before loading
    # a large Qwen model to avoid competing GPU allocations.
    person = require_shared_context(cfg, str(device), loader=load_person_context) if cfg.get("person_protocol") else None
    cfg = {**cfg, "runtime": {**cfg["runtime"], "resolved_device": str(device),
                              "resolved_dtype": "fp32" if device.type == "cpu" else cfg["runtime"]["dtype"]}}
    digest = (person.detections.fingerprint["gallery_image_content_sha256"] if person is not None
              else image_digest(gallery))
    fingerprint = {"version": VERSION, "assets": canonical_hash(marker),
                   "gallery": canonical_hash(gallery), "gallery_image_content_sha256": digest,
                   "backend": cfg["backend"], "environment": environment_metadata(device)}
    if person is not None:
        fingerprint["person_protocol"] = {key: value for key, value in person.metadata().items()
                                          if key.endswith("fingerprint") or key.endswith("sha256") or key == "version"}
    output = resolve(cfg["output"]["dir"])
    output.mkdir(parents=True, exist_ok=True)
    # On failure, old successful metadata must not describe newly partial scores.
    # Existing completed scores are replaced atomically after successful scoring.
    (output / "run.json").unlink(missing_ok=True)
    extra = (run_captions(cfg, gallery, queries, index, person, fingerprint, device, output)
             if cfg["backend"] in {"caption_to_caption", "structured_caption"}
             else run_scene(cfg, gallery, queries, index, fingerprint, device, output))
    if manifest_fingerprint(ROOT) != data_fingerprint:
        raise RuntimeError("Manifests changed during inference")
    metadata = {"method": cfg["method"], "display_name": cfg["display_name"], "group": cfg["group"],
        "cpr_supervision": cfg["cpr_supervision"], "higher_is_better": True,
        "num_queries": len(queries), "num_gallery": len(gallery),
        "gallery_image_ids": [row["image_id"] for row in gallery],
        "query_ids": [row["query_id"] for row in queries], "data_fingerprint": data_fingerprint,
        "image_provenance": {"gallery_image_content_sha256": digest}, "assets": marker,
        "adapter_version": VERSION, "input_fields": "public allowlist; no GT IDs, positives, case labels or target images",
        "runtime": cfg["runtime"], "environment": environment_metadata(device),
        "elapsed_seconds": perf_counter() - started, **extra}
    if person is not None:
        metadata["shared_person_protocol"] = person.metadata()
        metadata["setmatch"] = matching_metadata(cfg, score_space="cosine")
    atomic_json(output / "run.json", metadata)
    print(f"Saved complete scores: {output / 'scores.npy'}", flush=True)


def main(config_path):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(config_path))
    args = parser.parse_args()
    run(load_config(resolve(args.config)))
