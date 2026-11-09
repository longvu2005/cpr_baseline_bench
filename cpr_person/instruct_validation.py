"""Reuse the unchanged native Instruct encoder on complete RCR val."""

from __future__ import annotations

import numpy as np

from benchmark_data import split_paths, load_jsonl, manifest_image_digest
from cpr_baselines.provenance import execution_identity
from .adapter import load_person_context
from .cache import ROOT, StageCache, read_json, atomic_json, sha256_file, canonical_hash
from .validation import verify_tuning_manifests
from .unmatched import native_distance_grid, select_unmatched


def select_for_instruct(
    native, cfg, model, device, checkpoint, bert_weight, config_bert
):
    test_g, test_q = split_paths(ROOT, "test")
    val_g, val_q = split_paths(ROOT, "val")
    validation = verify_tuning_manifests(ROOT, test_g, test_q, val_g, val_q)
    if cfg["setmatch"].get("unmatched_calibration"):
        raise ValueError(
            "Do not mix validation selection with external unmatched calibration"
        )
    gallery, queries = load_jsonl(val_g), load_jsonl(val_q)
    person = load_person_context(gallery, queries, val_g, val_q, device)
    query_indices = native.query_gallery_indices(
        queries, native.build_gallery_index(gallery)
    )
    targets = native.query_targets(person, queries, query_indices)
    provenance = {
        "validation_fingerprint": validation,
        "image_sha256": manifest_image_digest(ROOT, val_g),
        "person": person.metadata()["localization_fingerprint"],
        "model": cfg["model"],
        "instruction": cfg["instruction_adapter"],
        "runtime": cfg["runtime"],
        "source": cfg["source"],
        "checkpoint_sha256": sha256_file(checkpoint),
        "bert_sha256": sha256_file(bert_weight),
        "bert_config_sha256": sha256_file(config_bert),
        "execution": execution_identity(native.__file__),
    }
    cache = StageCache("instruct_validation", provenance)
    stage = cache.directory
    stage.mkdir(parents=True, exist_ok=True)
    gf = native.gallery_feature_fingerprint(
        cfg=cfg,
        detection_cache=person.detections.path,
        checkpoint=checkpoint,
        gallery_manifest=val_g,
        bert_weight=bert_weight,
        config_bert=config_bert,
    )
    gallery_path = stage / "gallery.npy"
    features = native.load_array_cache(
        gallery_path,
        gf,
        (int(person.detections.offsets[-1]), cfg["model"]["feature_dim"]),
        "val gallery",
    )
    if features is None:
        features = native.compute_gallery_instruct_features(
            cfg=cfg,
            gallery=gallery,
            offsets=person.detections.offsets,
            boxes=person.detections.boxes,
            model=model,
            device=device,
            cache_path=gallery_path,
            cache_meta=gf,
        )
    qf = native.query_feature_fingerprint(
        cfg=cfg,
        query_manifest=val_q,
        detection_cache=person.detections.path,
        selector_cache=person.localization.path,
        checkpoint=checkpoint,
        bert_weight=bert_weight,
        config_bert=config_bert,
        localization=targets,
    )
    query_path = stage / "queries.npz"
    loaded = native.load_query_feature_cache(
        query_path, qf, len(queries), cfg["model"]["feature_dim"]
    )
    if loaded is None:
        loaded = native.compute_query_instruct_features(
            cfg=cfg,
            gallery=gallery,
            boxes=person.detections.boxes,
            localization=targets,
            model=model,
            device=device,
            cache_path=query_path,
            cache_meta=qf,
        )
    offsets, query_features = loaded
    fingerprint = {
        **provenance,
        "gallery_features_sha256": sha256_file(gallery_path),
        "query_features_sha256": sha256_file(query_path),
    }
    selection_path = stage / "selection.json"
    verified_selection = cache.find("selection.json")
    existing = read_json(verified_selection) if verified_selection else None
    if isinstance(existing, dict) and existing.get("fingerprint") == fingerprint:
        result = existing
    else:
        grid, policy = native_distance_grid(query_features, features)

        def pair_scores(qi):
            q = np.asarray(query_features[offsets[qi] : offsets[qi + 1]], np.float32)
            scores = np.empty((len(q), len(features)), np.float32)
            chunk = int(cfg["runtime"]["score_person_chunk_size"])
            for start in range(0, len(features), chunk):
                scores[:, start : start + chunk] = native.negative_squared_euclidean(
                    q, features[start : start + chunk]
                )
            return scores

        result = select_unmatched(
            grid,
            gallery,
            queries,
            person.detections.offsets,
            pair_scores,
            selection_path,
            validation,
        )
        result.update(fingerprint=fingerprint, candidate_policy=policy)
        atomic_json(selection_path, result)
        cache.seal(selection_path)
    value = result.get("selected_unmatched_score")
    if (
        not isinstance(value, (int, float))
        or not np.isfinite(value)
        or value > 0
        or result.get("selection_split") != "val"
        or result.get("validation_fingerprint") != validation
    ):
        raise ValueError("Invalid frozen native unmatched selection")
    if validation != verify_tuning_manifests(ROOT, test_g, test_q, val_g, val_q):
        raise RuntimeError("Validation manifests changed during selection")
    cfg["setmatch"].update(
        unmatched_score=float(value),
        unmatched_score_source="validation_Full_mAP_native_distance_quantiles",
        calibration_provenance={
            "selection_split": "val",
            "artifact": str(selection_path),
            "artifact_sha256": sha256_file(selection_path),
            "fingerprint_sha256": canonical_hash(fingerprint),
            "checkpoint_sha256": fingerprint["checkpoint_sha256"],
        },
    )
    cfg["cpr_supervision"] = "Val only"
    return validation
