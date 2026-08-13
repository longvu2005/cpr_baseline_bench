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
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark_progress import PhaseTracker, progress_bar  # noqa: E402

DEFAULT_CONFIG = Path(__file__).resolve().parent / "config.yaml"
METHOD_ID = "per_person_clip_compose_setmatch"
ADAPTER_VERSION = "2026-08-14-v2-py312-s5-sync-fixed-alpha"
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
        "config_sha256": sha256_file(config_path),
        "detection_cache_sha256": sha256_file(detection_cache),
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
        del mmap
        os.replace(temp, cache_path)
        write_json(meta_path(cache_path), cache_meta)
    except Exception:
        try:
            del mmap
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


def hungarian_maximize(similarity: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    try:
        return linear_sum_assignment(similarity, maximize=True)
    except TypeError:
        return linear_sum_assignment(-similarity)


def query_compose_text(query: dict[str, Any], subject: dict[str, Any]) -> str:
    case = str(query.get("case", "")).strip()
    if case == "RELATIONAL":
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


def prepare_query_data(*, queries: Sequence[dict[str, Any]], query_indices: np.ndarray, person_features: np.ndarray, offsets: np.ndarray, model, device: torch.device, text_batch_size: int) -> list[dict[str, Any]]:
    prepared: list[dict[str, Any]] = []
    for qi, query in enumerate(progress_bar(queries, desc="Prepare query person compositions", total=len(queries), unit="query")):
        subjects = query.get("subjects")
        if not isinstance(subjects, list) or not subjects:
            prepared.append({"valid": False, "reason": "no_subjects", "count": 0})
            continue
        source_gi = int(query_indices[qi])
        q_start, q_end = int(offsets[source_gi]), int(offsets[source_gi + 1])
        query_persons = np.asarray(person_features[q_start:q_end], dtype=np.float32)
        m = len(subjects)
        if query_persons.shape[0] < m or m == 0:
            prepared.append({"valid": False, "reason": "too_few_query_persons", "count": m})
            continue
        select_texts = []
        compose_texts = []
        try:
            for si, subject in enumerate(subjects):
                if not isinstance(subject, dict):
                    raise TypeError(f"subject {si} is not an object")
                select_text = str(subject.get("select_text") or "").strip()
                if not select_text:
                    raise ValueError(f"subject {si} has empty select_text")
                select_texts.append(select_text)
                compose_texts.append(query_compose_text(query, subject))
        except Exception as error:
            prepared.append({"valid": False, "reason": f"bad_subject_text:{error}", "count": m})
            continue
        selector_feat = clip_text_features(model, select_texts, text_batch_size, device)
        similarity = selector_feat @ query_persons.T
        if m == 1:
            assignment = np.asarray([int(np.argmax(similarity[0]))], dtype=np.int64)
        else:
            rows, cols = hungarian_maximize(similarity)
            if len(rows) != m:
                prepared.append({"valid": False, "reason": "localization_assignment_failed", "count": m})
                continue
            order = np.argsort(rows)
            assignment = cols[order].astype(np.int64, copy=False)
        selected_query_features = query_persons[assignment]
        compose_text_feat = clip_text_features(model, compose_texts, text_batch_size, device)
        prepared.append({
            "valid": True,
            "count": m,
            "selected_query_features": selected_query_features.astype(np.float32, copy=False),
            "compose_text_features": compose_text_feat.astype(np.float32, copy=False),
        })
    return prepared


def compose_query_set(selected_query_features: np.ndarray, compose_text_features: np.ndarray, alpha: float) -> np.ndarray:
    composed = alpha * selected_query_features + (1.0 - alpha) * compose_text_features
    norms = np.linalg.norm(composed, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return (composed / norms).astype(np.float32, copy=False)


def score_one_query(*, query_features: np.ndarray, all_features: np.ndarray, offsets: np.ndarray, counts: np.ndarray, person_image_index: np.ndarray, unmatched_score: float, feature_chunk_size: int) -> np.ndarray:
    num_gallery = len(counts)
    m = int(query_features.shape[0])
    result = np.full(num_gallery, unmatched_score, dtype=np.float32)
    if m == 0:
        return result
    q = np.asarray(query_features, dtype=np.float32)
    if m == 1:
        total_persons = int(offsets[-1])
        if total_persons == 0:
            return result
        person_scores = np.empty(total_persons, dtype=np.float32)
        query_vector = q[0]
        for start in range(0, total_persons, feature_chunk_size):
            end = min(start + feature_chunk_size, total_persons)
            chunk = np.asarray(all_features[start:end], dtype=np.float32)
            person_scores[start:end] = chunk @ query_vector
        image_scores = np.full(num_gallery, -np.inf, dtype=np.float32)
        np.maximum.at(image_scores, person_image_index, person_scores)
        nonempty = counts > 0
        result[nonempty] = image_scores[nonempty]
        return result
    eligible = np.flatnonzero(counts >= m)
    for gi in eligible:
        start, end = int(offsets[gi]), int(offsets[gi + 1])
        gallery_features = np.asarray(all_features[start:end], dtype=np.float32)
        similarity = q @ gallery_features.T
        rows, cols = hungarian_maximize(similarity)
        if len(rows) != m:
            raise RuntimeError(f"Hungarian assignment returned {len(rows)} pairs for query set size {m}")
        assigned = similarity[rows, cols]
        result[gi] = float(np.min(assigned))
    return result


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
    for path in (val_gallery_manifest, val_query_manifest):
        if not path.is_file():
            raise FileNotFoundError(f"Missing validation manifest: {rel(path)}. S10 requires a separate validation split for alpha selection.")
    if sha256_file(main_gallery_manifest) == sha256_file(val_gallery_manifest) and sha256_file(main_query_manifest) == sha256_file(val_query_manifest):
        raise RuntimeError("Validation manifests are byte-identical to the main evaluation manifests. Refusing to tune alpha on the evaluation set.")


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
    tracker = PhaseTracker(METHOD_ID, total=7)

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

        s5_method_dir = resolve_path(str(cfg["shared_protocol"]["method_dir"]))
        s5_config_path = resolve_path(str(cfg["shared_protocol"]["config"]))
        s5_cfg = load_yaml(s5_config_path)
        s5 = load_module(s5_method_dir, "cpr_s10_source_s5")

        main_shape = (len(main_queries), len(main_gallery))
        if alpha_mode == "validation":
            tracker.log(f"main_gallery={len(main_gallery):,} main_queries={len(main_queries):,} val_gallery={len(val_gallery):,} val_queries={len(val_queries):,} alpha_mode=validation")
        else:
            tracker.log(f"main_gallery={len(main_gallery):,} main_queries={len(main_queries):,} alpha_mode=fixed alpha={alpha:.2f}")

    with tracker.phase("Prepare shared/person detections"):
        device = s5.device_from(str(s5_cfg["runtime"]["device"]))
        gd_checkout = s5.ensure_clean_pinned_source(s5_cfg["source"]["groundingdino"], "Grounding DINO")
        detector_config = s5.require_file(gd_checkout / str(s5_cfg["detector"]["config"]), "Grounding DINO config")
        detector_checkpoint = s5.require_file(resolve_path(str(s5_cfg["detector"]["checkpoint"])), "Grounding DINO checkpoint")
        s5.configure_groundingdino_offline(s5_cfg)
        gdino_attention_backend = s5.configure_groundingdino_source(gd_checkout)

        main_det_cache = resolve_path(str(cfg["cache"]["main"]["detections"]))
        main_det_meta = s5.detection_fingerprint(cfg=s5_cfg, config_path=s5_config_path, gallery_manifest=main_gallery_manifest, detector_config=detector_config, detector_checkpoint=detector_checkpoint, attention_backend=gdino_attention_backend)
        loaded = s5.load_detection_cache(main_det_cache, main_det_meta, len(main_gallery))
        if loaded is None:
            main_offsets, main_boxes, _main_conf = s5.compute_detections(cfg=s5_cfg, gallery=main_gallery, detector_config=detector_config, detector_checkpoint=detector_checkpoint, device=device)
            s5.save_detection_cache(main_det_cache, main_det_meta, main_offsets, main_boxes, _main_conf)
        else:
            main_offsets, main_boxes, _main_conf = loaded

        val_det_cache = None
        val_offsets = None
        val_boxes = None
        if alpha_mode == "validation":
            val_det_cache = resolve_path(str(cfg["cache"]["validation"]["detections"]))
            val_det_meta = s5.detection_fingerprint(cfg=s5_cfg, config_path=s5_config_path, gallery_manifest=val_gallery_manifest, detector_config=detector_config, detector_checkpoint=detector_checkpoint, attention_backend=gdino_attention_backend)
            loaded = s5.load_detection_cache(val_det_cache, val_det_meta, len(val_gallery))
            if loaded is None:
                val_offsets, val_boxes, _val_conf = s5.compute_detections(cfg=s5_cfg, gallery=val_gallery, detector_config=detector_config, detector_checkpoint=detector_checkpoint, device=device)
                s5.save_detection_cache(val_det_cache, val_det_meta, val_offsets, val_boxes, _val_conf)
            else:
                val_offsets, val_boxes, _val_conf = loaded

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
        main_prepared_queries = prepare_query_data(queries=main_queries, query_indices=main_query_indices, person_features=main_person_features, offsets=main_offsets, model=model, device=device, text_batch_size=int(cfg["runtime"]["clip_text_batch_size"]))

        val_prepared_queries = None
        if alpha_mode == "validation":
            val_query_indices = query_gallery_indices(val_queries, build_gallery_index(val_gallery))
            val_prepared_queries = prepare_query_data(queries=val_queries, query_indices=val_query_indices, person_features=val_person_features, offsets=val_offsets, model=model, device=device, text_batch_size=int(cfg["runtime"]["clip_text_batch_size"]))

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
        write_json(run_path, payload)
        tracker.log(f"scores={rel(output_dir / 'scores.npy')} run={rel(run_path)}")

    tracker.finish()


if __name__ == "__main__":
    main()
