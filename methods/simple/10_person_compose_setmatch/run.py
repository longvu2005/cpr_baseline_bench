#!/usr/bin/env python3
"""S10: Per-Person CLIP Compose + SetMatch."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Sequence

import clip
import numpy as np
import torch
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark_progress import PhaseTracker, progress_bar  # noqa: E402

from cpr_person.adapter import load_person_context, matching_metadata, manifest_identity
from cpr_person.cache import feature_config_hash
from cpr_person.detection import protocol_config
from cpr_person.matching import score_feature_sets
from cpr_person.detection import read_rgb
DEFAULT_CONFIG = Path(__file__).resolve().parent / "config.yaml"
METHOD_ID = "per_person_clip_compose_setmatch"
ADAPTER_VERSION = "2026-10-12-shared-person-v1"
PERSON_FEATURE_SCHEMA = 1
ALPHA_SCHEMA = 1


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise TypeError(f"Expected YAML mapping: {path}")
    return data


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise TypeError(f"{path}:{lineno}: JSONL row must be an object")
            rows.append(row)
    return rows


def resolve_path(value: str) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else ROOT / path).resolve()


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def meta_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".meta.json")


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def device_from(name: str) -> torch.device:
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return device


def load_module(method_dir: Path, module_name: str):
    run_path = method_dir / "run.py"
    if not run_path.is_file():
        raise FileNotFoundError(f"Missing source adapter: {rel(run_path)}")
    spec = importlib.util.spec_from_file_location(module_name, run_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import adapter from {rel(run_path)}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def build_gallery_index(gallery: Sequence[dict[str, Any]]) -> dict[Any, int]:
    index: dict[Any, int] = {}
    for gi, row in enumerate(gallery):
        image_id = row.get("image_id")
        if image_id is None:
            raise KeyError(f"Gallery row {gi} missing image_id")
        if image_id in index:
            raise ValueError(f"Duplicate gallery image_id: {image_id!r}")
        index[image_id] = gi
    return index


def query_gallery_indices(queries: Sequence[dict[str, Any]], gallery_index: dict[Any, int]) -> np.ndarray:
    indices: list[int] = []
    for qi, query in enumerate(queries):
        image_id = query.get("image_id")
        if image_id not in gallery_index:
            raise ValueError(f"Query row {qi}: image_id {image_id!r} missing from gallery")
        indices.append(gallery_index[image_id])
    return np.asarray(indices, dtype=np.int64)


def person_feature_fingerprint(*, config_path: Path, detection_cache: Path, clip_checkpoint: Path, clip_name: str) -> dict[str, Any]:
    return {
        "schema": PERSON_FEATURE_SCHEMA,
        "adapter_version": ADAPTER_VERSION,
        "feature_config_sha256": feature_config_hash(config_path),
        "detection_cache_sha256": sha256_file(detection_cache),
        "shared_detection_identity": detection_cache.parent.name,
        "clip_checkpoint_sha256": sha256_file(clip_checkpoint),
        "clip_name": clip_name,
    }


def load_cached_person_features(cache_path: Path, expected_meta: dict[str, Any], expected_shape: tuple[int, int]) -> np.ndarray | None:
    if not cache_path.is_file() or not meta_path(cache_path).is_file():
        return None
    if read_json(meta_path(cache_path)) != expected_meta:
        print(f"Ignoring stale person-feature cache: {rel(cache_path)}", flush=True)
        return None
    try:
        features = np.load(cache_path, mmap_mode="r", allow_pickle=False)
    except Exception as error:
        print(f"Ignoring invalid person-feature cache {rel(cache_path)}: {error}", flush=True)
        return None
    if features.shape != expected_shape:
        print(f"Ignoring incompatible person-feature cache {rel(cache_path)}: shape={features.shape}, expected={expected_shape}", flush=True)
        return None
    if features.dtype.kind != "f":
        print(f"Ignoring non-floating person-feature cache: {rel(cache_path)}", flush=True)
        return None
    print(f"Using person-feature cache: {rel(cache_path)}", flush=True)
    return features


def image_path(row: dict[str, Any], index: int) -> Path:
    value = row.get("path")
    if not isinstance(value, str) or not value:
        raise KeyError(f"Gallery row {index} has no usable path")
    path = (ROOT / value).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


@torch.no_grad()
def compute_person_features(*, gallery: Sequence[dict[str, Any]], offsets: np.ndarray, boxes: np.ndarray, model, preprocess, cache_path: Path, cache_meta: dict[str, Any], batch_size: int, device: torch.device, np_dtype: np.dtype) -> np.ndarray:
    total_persons = int(offsets[-1])
    feature_dim = int(model.text_projection.shape[1])
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temp = cache_path.with_name(cache_path.name + ".part")
    temp.unlink(missing_ok=True)

    if total_persons == 0:
        with temp.open("wb") as handle:
            np.save(handle, np.empty((0, feature_dim), dtype=np_dtype))
        os.replace(temp, cache_path)
        write_json(meta_path(cache_path), cache_meta)
        return np.load(cache_path, mmap_mode="r", allow_pickle=False)

    mmap = np.lib.format.open_memmap(temp, mode="w+", dtype=np_dtype, shape=(total_persons, feature_dim))
    pending: list[torch.Tensor] = []
    write_cursor = 0

    def flush_batch() -> None:
        nonlocal pending, write_cursor
        if not pending:
            return
        batch = torch.stack(pending, dim=0).to(device, non_blocking=True)
        feat = model.encode_image(batch).float()
        feat /= feat.norm(dim=1, keepdim=True).clamp_min(1e-12)
        count = feat.shape[0]
        mmap[write_cursor : write_cursor + count] = feat.cpu().numpy().astype(np_dtype, copy=False)
        write_cursor += count
        pending = []

    try:
        for gi, row in enumerate(progress_bar(gallery, desc="Encode detected persons with CLIP", total=len(gallery), unit="image")):
            start, end = int(offsets[gi]), int(offsets[gi + 1])
            if end <= start:
                continue
            path = image_path(row, gi)
            with Image.open(path) as im:
                image = im.convert("RGB")
                for box in boxes[start:end]:
                    left = max(0, int(math.floor(float(box[0]))))
                    top = max(0, int(math.floor(float(box[1]))))
                    right = min(image.width, int(math.ceil(float(box[2]))))
                    bottom = min(image.height, int(math.ceil(float(box[3]))))
                    if right <= left or bottom <= top:
                        raise RuntimeError(f"Invalid cached crop for gallery row {gi}: {box.tolist()}")
                    pending.append(preprocess(image.crop((left, top, right, bottom))))
                    if len(pending) >= batch_size:
                        flush_batch()
        flush_batch()
        if write_cursor != total_persons:
            raise RuntimeError(f"Encoded {write_cursor} persons, expected {total_persons}")
        mmap.flush()
        mmap = None
        os.replace(temp, cache_path)
        write_json(meta_path(cache_path), cache_meta)
    except Exception:
        try:
            mmap = None
        except Exception:
            pass
        temp.unlink(missing_ok=True)
        raise
    return np.load(cache_path, mmap_mode="r", allow_pickle=False)


@torch.no_grad()
def clip_text_features(model, texts: list[str], batch_size: int, device: torch.device) -> np.ndarray:
    chunks: list[np.ndarray] = []
    for start in range(0, len(texts), batch_size):
        tokens = clip.tokenize(texts[start : start + batch_size], truncate=True).to(device)
        feat = model.encode_text(tokens).float()
        feat /= feat.norm(dim=1, keepdim=True).clamp_min(1e-12)
        chunks.append(feat.cpu().numpy())
    return np.concatenate(chunks, axis=0).astype(np.float32, copy=False)


def query_compose_text(query: dict[str, Any], subject: dict[str, Any]) -> str:
    if str(query.get("relation_text") or "").strip():
        text = str(query.get("text") or "").strip()
        if text:
            return text
    modify = str(subject.get("modify_text") or "").strip()
    if modify:
        return modify
    relation = str(query.get("relation_text") or "").strip()
    if relation:
        return relation
    text = str(query.get("text") or "").strip()
    if text:
        return text
    raise ValueError(f"Query {query.get('query_id')} has no usable composition text")


def prepare_query_data(*, queries, query_indices, person_features, offsets, model, device,
                       text_batch_size, person, preprocess):
    prepared, full_features = [], {}
    with torch.no_grad():
        for qi, query in enumerate(queries):
            gi = int(query_indices[qi])
            selected, texts = [], []
            for si, member in person.localization.flat(qi):
                ref = member["global_person_index"]
                if ref is None:
                    if gi not in full_features:
                        image = read_rgb(image_path(person.gallery[gi], gi))
                        feature = model.encode_image(preprocess(image).unsqueeze(0).to(device)).float()
                        feature /= feature.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                        full_features[gi] = feature[0].cpu().numpy()
                    selected.append(full_features[gi])
                else:
                    selected.append(np.asarray(person_features[ref], dtype=np.float32))
                texts.append(query_compose_text(query, query["subjects"][si]))
            composed_text = clip_text_features(model, texts, text_batch_size, device)
            prepared.append({"valid": True, "count": len(selected),
                             "selected_query_features": np.stack(selected).astype(np.float32),
                             "compose_text_features": composed_text.astype(np.float32)})
    return prepared


def compose_query_set(selected_query_features: np.ndarray, compose_text_features: np.ndarray, alpha: float) -> np.ndarray:
    composed = alpha * selected_query_features + (1.0 - alpha) * compose_text_features
    norms = np.linalg.norm(composed, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return (composed / norms).astype(np.float32, copy=False)


def score_one_query(*, query_features, all_features, offsets, counts,
                    person_image_index, unmatched_score, feature_chunk_size):
    return score_feature_sets(query_features, all_features, offsets, unmatched_score, feature_chunk_size)


def compute_scores_for_alpha(*, prepared_queries: list[dict[str, Any]], all_features: np.ndarray, offsets: np.ndarray, unmatched_score: float, feature_chunk_size: int, alpha: float) -> np.ndarray:
    counts = np.diff(offsets).astype(np.int64, copy=False)
    total_persons = int(offsets[-1])
    person_image_index = np.repeat(np.arange(len(counts), dtype=np.int64), counts.astype(np.int64))
    if person_image_index.shape != (total_persons,):
        raise RuntimeError("Internal person-to-image index shape mismatch")
    scores = np.empty((len(prepared_queries), len(counts)), dtype=np.float32)
    for qi, prepared in enumerate(progress_bar(prepared_queries, desc=f"SetMatch alpha={alpha:.2f}", total=len(prepared_queries), unit="query")):
        if not prepared.get("valid", False):
            scores[qi].fill(unmatched_score)
            continue
        query_features = compose_query_set(prepared["selected_query_features"], prepared["compose_text_features"], alpha)
        scores[qi] = score_one_query(query_features=query_features, all_features=all_features, offsets=offsets, counts=counts, person_image_index=person_image_index, unmatched_score=unmatched_score, feature_chunk_size=feature_chunk_size)
    return scores


def validate_score_matrix(scores: np.ndarray, shape: tuple[int, int], label: str) -> None:
    if scores.shape != shape:
        raise ValueError(f"{label}: shape={scores.shape}, expected={shape}")
    if not np.issubdtype(scores.dtype, np.floating):
        raise TypeError(f"{label} must be floating point")
    for start in range(0, shape[0], 256):
        if not np.isfinite(np.asarray(scores[start : start + 256])).all():
            raise ValueError(f"{label} contains NaN/Inf")


def ensure_validation_split(main_gallery_manifest: Path, main_query_manifest: Path, val_gallery_manifest: Path, val_query_manifest: Path) -> None:
    from cpr_person.validation import ensure_separate_queries
    ensure_separate_queries(main_query_manifest, val_query_manifest)
    for path in (val_gallery_manifest, val_query_manifest):
        if not path.is_file():
            raise FileNotFoundError(f"Missing validation manifest: {rel(path)}. S10 requires a separate validation split for alpha selection.")
    if sha256_file(main_query_manifest) == sha256_file(val_query_manifest):
        raise RuntimeError(
            "Validation query manifest is byte-identical to the main evaluation "
            "query manifest. Refusing to tune alpha on evaluation labels."
        )


def measure(scores: np.ndarray, positives: set[int]) -> float:
    order = np.argsort(-scores)
    rel_mask = np.isin(order, list(positives))
    ranks = np.flatnonzero(rel_mask) + 1
    if len(ranks) == 0:
        raise ValueError("No positive found for a validation query.")
    return float(np.mean(np.arange(1, len(ranks) + 1) / ranks))


def full_map(scores: np.ndarray, gallery: Sequence[dict[str, Any]], queries: Sequence[dict[str, Any]]) -> float:
    gallery_index = build_gallery_index(gallery)
    aps: list[float] = []
    for qi, query in enumerate(queries):
        image_id = query.get("image_id")
        if image_id not in gallery_index:
            raise ValueError(f"Validation query {query.get('query_id', qi)} image missing from gallery")
        positives = query.get("full_positive_ids")
        if not isinstance(positives, list) or not positives:
            raise ValueError(f"Validation query {query.get('query_id', qi)} has no full_positive_ids")
        s = np.array(scores[qi], copy=True)
        self_idx = gallery_index[image_id]
        s[self_idx] = -np.inf
        pos = {gallery_index[x] for x in positives if x in gallery_index}
        pos.discard(self_idx)
        if not pos:
            raise ValueError(f"Validation query {query.get('query_id', qi)} has no usable Full positives")
        aps.append(measure(s, pos))
    return float(np.mean(aps))


def select_alpha(*, cfg: dict[str, Any], config_path: Path, main_gallery_manifest: Path, main_query_manifest: Path, val_gallery_manifest: Path, val_query_manifest: Path, val_gallery: Sequence[dict[str, Any]], val_queries: Sequence[dict[str, Any]], prepared_val_queries: list[dict[str, Any]], val_person_features: np.ndarray, val_offsets: np.ndarray) -> tuple[float, dict[str, Any]]:
    selection = cfg["composition"]["alpha_selection"]
    result_path = resolve_path(str(selection["result"]))
    grid = [float(x) for x in selection["grid"]]
    expected = {
        "schema": ALPHA_SCHEMA,
        "adapter_version": ADAPTER_VERSION,
        "config_sha256": sha256_file(config_path),
        "main_gallery_manifest_sha256": sha256_file(main_gallery_manifest),
        "main_query_manifest_sha256": sha256_file(main_query_manifest),
        "val_gallery_manifest_sha256": sha256_file(val_gallery_manifest),
        "val_query_manifest_sha256": sha256_file(val_query_manifest),
        "grid": grid,
        "metric": str(selection["metric"]),
        "tie_break": str(selection["tie_break"]),
    }
    current = read_json(result_path)
    if current is not None:
        core = {k: current.get(k) for k in expected}
        if core == expected and isinstance(current.get("best_alpha"), (int, float)):
            return float(current["best_alpha"]), current

    unmatched_score = float(cfg["setmatch"]["unmatched_score"])
    chunk = int(cfg["runtime"]["score_feature_chunk_size"])
    rows: list[dict[str, Any]] = []
    best_alpha = None
    best_map = None
    for alpha in grid:
        scores = compute_scores_for_alpha(prepared_queries=prepared_val_queries, all_features=val_person_features, offsets=val_offsets, unmatched_score=unmatched_score, feature_chunk_size=chunk, alpha=float(alpha))
        score = full_map(scores, val_gallery, val_queries)
        rows.append({"alpha": float(alpha), "Full-mAP": float(score)})
        if best_map is None or score > best_map + 1e-12 or (abs(score - best_map) <= 1e-12 and float(alpha) < float(best_alpha)):
            best_map = float(score)
            best_alpha = float(alpha)
    payload = {**expected, "best_alpha": float(best_alpha), "best_full_map": float(best_map), "trials": rows}
    write_json(result_path, payload)
    return float(best_alpha), payload


def main() -> None:
    tracker = PhaseTracker(METHOD_ID, total=8)

    with tracker.phase("Load config, shared protocol, and manifests"):
        parser = argparse.ArgumentParser()
        parser.add_argument("--config", default=str(DEFAULT_CONFIG))
        args = parser.parse_args()
        config_path = resolve_path(args.config)
        cfg = load_yaml(config_path)

        main_gallery_manifest = resolve_path(str(cfg["data"]["gallery_manifest"]))
        main_query_manifest = resolve_path(str(cfg["data"]["query_manifest"]))
        main_gallery = load_jsonl(main_gallery_manifest)
        main_queries = load_jsonl(main_query_manifest)

        selection_cfg = cfg["composition"]["alpha_selection"]
        alpha_mode = str(selection_cfg.get("mode", "fixed")).strip().lower()
        if alpha_mode not in {"fixed", "validation"}:
            raise ValueError("composition.alpha_selection.mode must be 'fixed' or 'validation'")

        alpha = None
        val_gallery_manifest = None
        val_query_manifest = None
        val_gallery = None
        val_queries = None
        if alpha_mode == "fixed":
            try:
                alpha = float(selection_cfg["fixed_alpha"])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError("composition.alpha_selection.fixed_alpha must be numeric") from error
            if not math.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
                raise ValueError("composition.alpha_selection.fixed_alpha must be in [0, 1]")
            cpr_supervision = "No"
        else:
            val_gallery_manifest = resolve_path(str(selection_cfg["validation"]["gallery_manifest"]))
            val_query_manifest = resolve_path(str(selection_cfg["validation"]["query_manifest"]))
            ensure_validation_split(main_gallery_manifest, main_query_manifest, val_gallery_manifest, val_query_manifest)
            val_gallery = load_jsonl(val_gallery_manifest)
            val_queries = load_jsonl(val_query_manifest)
            cpr_supervision = "Val only"

        main_shape = (len(main_queries), len(main_gallery))
        if alpha_mode == "validation":
            tracker.log(f"main_gallery={len(main_gallery):,} main_queries={len(main_queries):,} val_gallery={len(val_gallery):,} val_queries={len(val_queries):,} alpha_mode=validation")
        else:
            tracker.log(f"main_gallery={len(main_gallery):,} main_queries={len(main_queries):,} alpha_mode=fixed alpha={alpha:.2f}")

    with tracker.phase("Consume shared candidates and Subject memberships"):
        device = device_from(str(cfg["runtime"]["device"]))
        main_person = load_person_context(main_gallery, main_queries, main_gallery_manifest, main_query_manifest, device)
        main_det_cache, main_offsets, main_boxes = main_person.detections.path, main_person.detections.offsets, main_person.detections.boxes
        gdino_attention_backend = protocol_config()["detector"]["attention_backend"]
        val_person = val_det_cache = val_offsets = val_boxes = None
        if alpha_mode == "validation":
            val_person = load_person_context(val_gallery, val_queries, val_gallery_manifest, val_query_manifest, device)
            val_det_cache, val_offsets, val_boxes = val_person.detections.path, val_person.detections.offsets, val_person.detections.boxes

    with tracker.phase("Load CLIP ViT-L/14"):
        device = device_from(str(cfg["runtime"]["device"]))
        checkpoint = resolve_path(str(cfg["clip"]["checkpoint"]))
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Missing CLIP checkpoint: {rel(checkpoint)}")
        if sha256_file(checkpoint) != str(cfg["clip"]["checkpoint_sha256"]):
            raise RuntimeError("CLIP checkpoint checksum mismatch")
        model, preprocess = clip.load(str(checkpoint), device=device, jit=False)
        model.eval()
        if device.type != "cuda":
            model.float()
        feature_dim = int(model.text_projection.shape[1])
        dtype_name = str(cfg["runtime"]["feature_cache_dtype"])
        if dtype_name not in {"float16", "float32"}:
            raise ValueError("feature_cache_dtype must be float16 or float32")
        np_dtype = np.float16 if dtype_name == "float16" else np.float32

    with tracker.phase("Prepare CLIP person-feature caches"):
        main_feature_cache = resolve_path(str(cfg["cache"]["main"]["person_features"]))
        main_feature_meta = person_feature_fingerprint(config_path=config_path, detection_cache=main_det_cache, clip_checkpoint=checkpoint, clip_name=str(cfg["clip"]["name"]))
        main_expected_shape = (int(main_offsets[-1]), feature_dim)
        main_person_features = load_cached_person_features(main_feature_cache, main_feature_meta, main_expected_shape)
        if main_person_features is None:
            main_person_features = compute_person_features(gallery=main_gallery, offsets=main_offsets, boxes=main_boxes, model=model, preprocess=preprocess, cache_path=main_feature_cache, cache_meta=main_feature_meta, batch_size=int(cfg["runtime"]["clip_image_batch_size"]), device=device, np_dtype=np_dtype)

        val_person_features = None
        if alpha_mode == "validation":
            val_feature_cache = resolve_path(str(cfg["cache"]["validation"]["person_features"]))
            val_feature_meta = person_feature_fingerprint(config_path=config_path, detection_cache=val_det_cache, clip_checkpoint=checkpoint, clip_name=str(cfg["clip"]["name"]))
            val_expected_shape = (int(val_offsets[-1]), feature_dim)
            val_person_features = load_cached_person_features(val_feature_cache, val_feature_meta, val_expected_shape)
            if val_person_features is None:
                val_person_features = compute_person_features(gallery=val_gallery, offsets=val_offsets, boxes=val_boxes, model=model, preprocess=preprocess, cache_path=val_feature_cache, cache_meta=val_feature_meta, batch_size=int(cfg["runtime"]["clip_image_batch_size"]), device=device, np_dtype=np_dtype)

    with tracker.phase("Prepare query-localized person compositions"):
        main_query_indices = query_gallery_indices(main_queries, build_gallery_index(main_gallery))
        main_prepared_queries = prepare_query_data(queries=main_queries, query_indices=main_query_indices, person_features=main_person_features, offsets=main_offsets, model=model, device=device, text_batch_size=int(cfg["runtime"]["clip_text_batch_size"]), person=main_person, preprocess=preprocess)

        val_prepared_queries = None
        if alpha_mode == "validation":
            val_query_indices = query_gallery_indices(val_queries, build_gallery_index(val_gallery))
            val_prepared_queries = prepare_query_data(queries=val_queries, query_indices=val_query_indices, person_features=val_person_features, offsets=val_offsets, model=model, device=device, text_batch_size=int(cfg["runtime"]["clip_text_batch_size"]), person=val_person, preprocess=preprocess)

    with tracker.phase("Resolve composition alpha"):
        if alpha_mode == "validation":
            alpha, alpha_payload = select_alpha(cfg=cfg, config_path=config_path, main_gallery_manifest=main_gallery_manifest, main_query_manifest=main_query_manifest, val_gallery_manifest=val_gallery_manifest, val_query_manifest=val_query_manifest, val_gallery=val_gallery, val_queries=val_queries, prepared_val_queries=val_prepared_queries, val_person_features=val_person_features, val_offsets=val_offsets)
            tracker.log(f"selected_alpha={alpha:.2f} best_full_map={alpha_payload.get('best_full_map'):.6f}")
        else:
            alpha_payload = {"selection_mode": "fixed", "selected_alpha": float(alpha), "selected_full_map": None}
            tracker.log(f"alpha={alpha:.2f} fixed; no CPR labels used for alpha selection")

    with tracker.phase("Compute final benchmark scores"):
        output_dir = resolve_path(str(cfg["output"]["dir"]))
        output_dir.mkdir(parents=True, exist_ok=True)
        scores_path = output_dir / "scores.npy"
        scores = compute_scores_for_alpha(prepared_queries=main_prepared_queries, all_features=main_person_features, offsets=main_offsets, unmatched_score=float(cfg["setmatch"]["unmatched_score"]), feature_chunk_size=int(cfg["runtime"]["score_feature_chunk_size"]), alpha=float(alpha))
        validate_score_matrix(scores, main_shape, "final scores")
        np.save(scores_path, scores.astype(np.float32, copy=False))

    with tracker.phase("Write run metadata"):
        output_dir = resolve_path(str(cfg["output"]["dir"]))
        run_path = output_dir / "run.json"
        payload = {
            "shared_person_protocol": main_person.metadata(),
            "setmatch": matching_metadata(cfg, score_space="cosine"),
            "method": cfg["method"],
            "display_name": cfg["display_name"],
            "group": cfg["group"],
            "cpr_supervision": cpr_supervision,
            "adapter_version": ADAPTER_VERSION,
            "alpha": float(alpha),
            "alpha_selection_mode": alpha_mode,
            "composition_formula": cfg["composition"]["formula"],
            "alpha_selection": rel(resolve_path(str(selection_cfg["result"]))) if alpha_mode == "validation" else None,
            "alpha_selection_details": alpha_payload,
            "shared_protocol": str(cfg["shared_protocol"]["method"]),
            "groundingdino_attention_backend": gdino_attention_backend,
            "clip": cfg["clip"],
            "config": rel(config_path),
            "num_queries": len(main_queries),
            "num_gallery": len(main_gallery),
            "scores": rel(output_dir / "scores.npy"),
            "higher_is_better": True,
        }
        payload["data_fingerprint"] = manifest_identity(payload["shared_person_protocol"])
        write_json(run_path, payload)
        tracker.log(f"scores={rel(output_dir / 'scores.npy')} run={rel(run_path)}")

    tracker.finish()


if __name__ == "__main__":
    main()
