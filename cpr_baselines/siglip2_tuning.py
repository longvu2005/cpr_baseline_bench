"""Select the frozen SigLIP2 image/text fusion weight using complete RCR val."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from benchmark_data import split_paths, verify_manifest
from cpr_person.cache import ROOT, atomic_json, canonical_hash, read_json
from .backends import build_backend
from .clip_fusion import select_image_weight
from .common import (cached_features, feature_cache, image_digest, load_jsonl,
                     public_inputs, query_text, source_images)
from .provenance import execution_identity


def tune_siglip2(cfg, marker, device, output, backend_factory=build_backend):
    """Tune once on val; never expose relevance labels to the encoder."""
    if cfg["method"] != "siglip2_compose" or cfg["cpr_supervision"] != "Val only":
        raise ValueError("SigLIP2 validation selection requires its Val only protocol")
    if cfg["composition"]["alpha_selection"] != "validation_full_map":
        raise ValueError("SigLIP2 alpha selection must use validation Full-mAP")
    grid = sorted({float(value) for value in cfg["tuning"]["image_weights"]})
    if grid != [i / 20 for i in range(21)]:
        raise ValueError("SigLIP2 alpha grid must span 0..1 in steps of 0.05")

    validation = verify_manifest(ROOT, "val")
    gallery_path, query_path = split_paths(ROOT, "val")
    raw_gallery, raw_queries = load_jsonl(gallery_path), load_jsonl(query_path)
    gallery, queries, index = public_inputs(raw_gallery, raw_queries)
    val_images = image_digest(gallery)
    fingerprint = {
        "schema": "siglip2-compose-val-alpha-v1",
        "method": cfg["method"],
        "validation_fingerprint": validation["fingerprint"],
        "validation_image_sha256": val_images,
        "assets_sha256": canonical_hash(marker),
        "runtime": cfg["runtime"],
        "composition": {"text_field": cfg["composition"]["text_field"],
                        "max_text_tokens": cfg["composition"]["max_text_tokens"],
                        "formula": cfg["composition"]["formula"]},
        "implementation_sha256": canonical_hash(execution_identity(__file__)["implementation"]),
        "image_weights": grid,
    }
    path = Path(output) / "tuning.json"
    cached = read_json(path)
    if (isinstance(cached, dict) and cached.get("fingerprint") == fingerprint
            and cached.get("image_weight") in grid
            and isinstance(cached.get("trials"), list)
            and len(cached["trials"]) == len(grid)
            and isinstance(cached.get("validation_full_map"), (int, float))
            and np.isfinite(cached["validation_full_map"])):
        return cached

    # Both embedding caches contain native frozen features, independent of alpha.
    runtime = cfg["runtime"]
    feature_fp = {**fingerprint, "gallery": canonical_hash(gallery)}
    encoder = None

    def get_encoder():
        nonlocal encoder
        if encoder is None:
            encoder = backend_factory(cfg, device)
        return encoder

    gallery_features = cached_features(
        feature_cache(cfg["method"] + "/val_gallery", feature_fp, runtime),
        "gallery.npy", len(gallery),
        lambda start, end: get_encoder().images(source_images(gallery, start, end)),
        int(runtime["image_batch_size"]),
    )
    texts = [query_text(row, cfg["composition"]["text_field"]) for row in queries]
    text_features = cached_features(
        feature_cache(cfg["method"] + "/val_text",
                      {**feature_fp, "queries": canonical_hash(queries)}, runtime),
        "text.npy", len(queries),
        lambda start, end: get_encoder().texts(texts[start:end]),
        int(runtime["text_batch_size"]),
    )

    # Score each native branch once, then use the exact CLIP early-fusion
    # selector, including normalized query composition and its tie rule.
    import torch
    references = np.asarray([index[row["image_id"]] for row in queries], np.int64)
    image_scores = np.empty((len(queries), len(gallery)), dtype=np.float32)
    text_scores = np.empty_like(image_scores)
    g = torch.from_numpy(np.asarray(gallery_features)).to(device)
    batch = int(runtime["score_query_batch_size"])
    with torch.inference_mode():
        for start in range(0, len(queries), batch):
            stop = min(start + batch, len(queries))
            r = torch.from_numpy(np.asarray(gallery_features[references[start:stop]])).to(device)
            t = torch.from_numpy(np.asarray(text_features[start:stop])).to(device)
            image_scores[start:stop] = (r @ g.T).cpu().numpy()
            text_scores[start:stop] = (t @ g.T).cpu().numpy()

    weight, score, trials = select_image_weight(
        image_scores, text_scores, raw_gallery, raw_queries, grid,
        early_features=(np.asarray(gallery_features[references]), np.asarray(text_features)),
    )
    if (verify_manifest(ROOT, "val") != validation or image_digest(gallery) != val_images):
        raise RuntimeError("Validation manifests or image bytes changed during tuning")
    receipt = {
        "split": "val", "metric": "Full-mAP", "image_weight": weight,
        "text_weight": 1.0 - weight, "validation_full_map": score,
        "fingerprint": fingerprint, "trials": trials,
    }
    atomic_json(path, receipt)
    return receipt
