#!/usr/bin/env python3
"""S6: Grounding DINO + CLIP-ReID - Set + Text.

Final score:
    alpha * ReID-Set + (1 - alpha) * CLIP-text

The ReID-Set branch is exactly the S5 adapter (same Grounding DINO detections,
CLIP-ReID MSMT17 embeddings, Hungarian assignment and strict-min aggregation).
The semantic branch is global OpenAI CLIP ViT-L/14 text-to-gallery similarity.
By default alpha is fixed, so the baseline is label-free and runnable from a
clean benchmark checkout. An optional validation mode can select alpha by
Full-mAP on a genuinely separate validation manifest. Test/evaluation positive
labels are never used for alpha selection.
"""

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
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark_progress import PhaseTracker, progress_bar  # noqa: E402

DEFAULT_CONFIG = Path(__file__).resolve().parent / "config.yaml"
METHOD_ID = "groundingdino_clipreid_set_text"
ADAPTER_VERSION = "2026-08-14-v2-fixed-or-val-fusion"
CLIP_CACHE_SCHEMA = 2
SCORE_CACHE_SCHEMA = 1
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


def load_s5_module(method_dir: Path):
    run_path = method_dir / "run.py"
    if not run_path.is_file():
        raise FileNotFoundError(
            f"S6 extends S5 but {rel(run_path)} is missing. Apply/install S5 first."
        )
    spec = importlib.util.spec_from_file_location("cpr_s5_groundingdino_clipreid_set", run_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import S5 adapter from {rel(run_path)}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def device_from(name: str) -> torch.device:
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return device


class GalleryDataset(Dataset):
    def __init__(self, rows: Sequence[dict[str, Any]], preprocess):
        self.rows = rows
        self.preprocess = preprocess

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        value = self.rows[index].get("path")
        if not isinstance(value, str) or not value:
            raise KeyError(f"Gallery row {index} has no usable path")
        path = (ROOT / value).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        with Image.open(path) as image:
            return self.preprocess(image.convert("RGB"))


def validate_score_matrix(scores: np.ndarray, shape: tuple[int, int], label: str) -> None:
    if scores.shape != shape:
        raise ValueError(f"{label} shape={scores.shape}, expected {shape}")
    if not np.issubdtype(scores.dtype, np.floating):
        raise TypeError(f"{label} must be floating point")
    for start in range(0, shape[0], 256):
        if not np.isfinite(np.asarray(scores[start : start + 256])).all():
            raise ValueError(f"{label} contains NaN/Inf")


def clip_gallery_fingerprint(
    checkpoint: Path, gallery_manifest: Path, model_name: str
) -> dict[str, Any]:
    # Match methods/simple/02_clip_text exactly so the ViT-L/14 gallery cache
    # is genuinely shared rather than each method invalidating the other's metadata.
    return {
        "schema": CLIP_CACHE_SCHEMA,
        "model_name": model_name,
        "checkpoint_sha256": sha256_file(checkpoint),
        "gallery_manifest_sha256": sha256_file(gallery_manifest),
    }


@torch.no_grad()
def prepare_clip_gallery_features(
    *,
    model,
    preprocess,
    gallery: Sequence[dict[str, Any]],
    gallery_manifest: Path,
    checkpoint: Path,
    cache_path: Path,
    cfg: dict[str, Any],
    device: torch.device,
) -> tuple[np.ndarray, bool]:
    feature_dim = int(model.text_projection.shape[1])
    expected_shape = (len(gallery), feature_dim)
    expected_meta = clip_gallery_fingerprint(
        checkpoint, gallery_manifest, str(cfg["clip_text"]["name"])
    )
    current_meta = read_json(meta_path(cache_path))
    if cache_path.is_file() and current_meta == expected_meta:
        cached = np.load(cache_path, mmap_mode="r", allow_pickle=False)
        if cached.shape == expected_shape and np.isfinite(np.asarray(cached[: min(8, len(cached))])).all():
            print(f"Using CLIP gallery cache: {rel(cache_path)}", flush=True)
            return cached, True
        print(f"Ignoring incompatible CLIP gallery cache: {rel(cache_path)}", flush=True)
    elif cache_path.is_file():
        print(f"Ignoring stale CLIP gallery cache: {rel(cache_path)}", flush=True)

    loader = DataLoader(
        GalleryDataset(gallery, preprocess),
        batch_size=int(cfg["runtime"]["clip_image_batch_size"]),
        shuffle=False,
        num_workers=int(cfg["runtime"]["num_workers"]),
        pin_memory=(device.type == "cuda"),
    )
    chunks: list[np.ndarray] = []
    for images in progress_bar(loader, desc="CLIP encode gallery", total=len(loader), unit="batch"):
        feat = model.encode_image(images.to(device, non_blocking=True)).float()
        feat /= feat.norm(dim=1, keepdim=True).clamp_min(1e-12)
        chunks.append(feat.cpu().numpy())
    features = np.concatenate(chunks, axis=0).astype(np.float32, copy=False)
    if features.shape != expected_shape:
        raise RuntimeError(f"CLIP gallery features {features.shape}, expected {expected_shape}")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, features)
    write_json(meta_path(cache_path), expected_meta)
    return np.load(cache_path, mmap_mode="r", allow_pickle=False), False


@torch.no_grad()
def encode_text(model, texts: Sequence[str], batch_size: int, device: torch.device) -> np.ndarray:
    chunks: list[np.ndarray] = []
    for start in progress_bar(
        range(0, len(texts), batch_size),
        desc="CLIP encode query text",
        total=(len(texts) + batch_size - 1) // batch_size,
        unit="batch",
    ):
        tokens = clip.tokenize(list(texts[start : start + batch_size]), truncate=True).to(device)
        feat = model.encode_text(tokens).float()
        feat /= feat.norm(dim=1, keepdim=True).clamp_min(1e-12)
        chunks.append(feat.cpu().numpy())
    return np.concatenate(chunks, axis=0).astype(np.float32, copy=False)


def clip_score_fingerprint(
    *,
    cfg: dict[str, Any],
    checkpoint: Path,
    gallery_manifest: Path,
    query_manifest: Path,
) -> dict[str, Any]:
    return {
        "schema": SCORE_CACHE_SCHEMA,
        "kind": "clip_text_scores",
        "adapter_version": ADAPTER_VERSION,
        "model": str(cfg["clip_text"]["name"]),
        "text_field": str(cfg["clip_text"]["text_field"]),
        "checkpoint_sha256": sha256_file(checkpoint),
        "gallery_manifest_sha256": sha256_file(gallery_manifest),
        "query_manifest_sha256": sha256_file(query_manifest),
    }


@torch.no_grad()
def compute_clip_text_scores(
    *,
    cfg: dict[str, Any],
    model,
    gallery_features: np.ndarray,
    queries: Sequence[dict[str, Any]],
    gallery_manifest: Path,
    query_manifest: Path,
    checkpoint: Path,
    output_path: Path,
    device: torch.device,
) -> tuple[np.ndarray, bool]:
    expected_shape = (len(queries), gallery_features.shape[0])
    fingerprint = clip_score_fingerprint(
        cfg=cfg,
        checkpoint=checkpoint,
        gallery_manifest=gallery_manifest,
        query_manifest=query_manifest,
    )
    if output_path.is_file() and read_json(meta_path(output_path)) == fingerprint:
        cached = np.load(output_path, mmap_mode="r", allow_pickle=False)
        try:
            validate_score_matrix(cached, expected_shape, "cached CLIP text scores")
            print(f"Using CLIP text-score cache: {rel(output_path)}", flush=True)
            return cached, True
        except (TypeError, ValueError):
            print(f"Ignoring incompatible CLIP text-score cache: {rel(output_path)}", flush=True)

    text_field = str(cfg["clip_text"]["text_field"])
    texts: list[str] = []
    for qi, query in enumerate(queries):
        text = query.get(text_field)
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Query {qi} has no usable {text_field!r}")
        texts.append(text)
    text_features = encode_text(
        model, texts, int(cfg["runtime"]["clip_text_batch_size"]), device
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scores = np.lib.format.open_memmap(
        output_path, mode="w+", dtype=np.float32, shape=expected_shape
    )
    gallery_tensor = torch.from_numpy(np.asarray(gallery_features, dtype=np.float32)).to(device)
    batch_size = int(cfg["runtime"]["clip_score_batch_size"])
    for start in progress_bar(
        range(0, len(queries), batch_size),
        desc="CLIP text-to-gallery scores",
        total=(len(queries) + batch_size - 1) // batch_size,
        unit="batch",
    ):
        end = min(start + batch_size, len(queries))
        query_tensor = torch.from_numpy(text_features[start:end]).to(device)
        scores[start:end] = (query_tensor @ gallery_tensor.T).cpu().numpy()
    scores.flush()
    write_json(meta_path(output_path), fingerprint)
    return scores, False


def s5_score_fingerprint(
    *,
    s5,
    s5_cfg_path: Path,
    gallery_manifest: Path,
    query_manifest: Path,
    detection_cache: Path,
    feature_cache: Path,
) -> dict[str, Any]:
    return {
        "schema": SCORE_CACHE_SCHEMA,
        "kind": "s5_reid_set_scores",
        "adapter_version": ADAPTER_VERSION,
        "s5_adapter_version": str(s5.ADAPTER_VERSION),
        "s5_config_sha256": sha256_file(s5_cfg_path),
        "gallery_manifest_sha256": sha256_file(gallery_manifest),
        "query_manifest_sha256": sha256_file(query_manifest),
        "detection_meta": read_json(s5.meta_path(detection_cache)),
        "feature_meta": read_json(s5.meta_path(feature_cache)),
    }


def prepare_s5_scores(
    *,
    s5,
    s5_cfg: dict[str, Any],
    s5_cfg_path: Path,
    gallery_manifest: Path,
    query_manifest: Path,
    detection_cache: Path,
    feature_cache: Path,
    score_cache: Path,
    device: torch.device,
) -> tuple[np.ndarray, dict[str, Any]]:
    gallery = s5.load_jsonl(gallery_manifest)
    queries = s5.load_jsonl(query_manifest)
    gallery_index = s5.build_gallery_index(gallery)
    query_indices = s5.query_gallery_indices(queries, gallery_index)

    gdino_source = s5.ensure_clean_pinned_source(
        s5_cfg["source"]["groundingdino"], "Grounding DINO"
    )
    clipreid_source = s5.ensure_clean_pinned_source(
        s5_cfg["source"]["clip_reid"], "CLIP-ReID"
    )
    detector_config = s5.require_file(
        gdino_source / str(s5_cfg["detector"]["config"]), "Grounding DINO config"
    )
    detector_checkpoint = s5.require_file(
        s5.resolve_path(str(s5_cfg["detector"]["checkpoint"])),
        "Grounding DINO checkpoint",
    )
    reid_checkpoint = s5.require_file(
        s5.resolve_path(str(s5_cfg["reid"]["checkpoint"])), "CLIP-ReID checkpoint"
    )
    clip_backbone = s5.require_file(
        s5.resolve_path(str(s5_cfg["reid"]["openai_clip_checkpoint"])),
        "OpenAI CLIP ViT-B/16 checkpoint",
    )
    s5.configure_groundingdino_offline(s5_cfg)

    detect_meta = s5.detection_fingerprint(
        cfg=s5_cfg,
        config_path=s5_cfg_path,
        gallery_manifest=gallery_manifest,
        detector_config=detector_config,
        detector_checkpoint=detector_checkpoint,
    )
    cached_det = s5.load_detection_cache(detection_cache, detect_meta, len(gallery))
    detection_hit = cached_det is not None
    if cached_det is None:
        offsets, boxes, confidences = s5.compute_detections(
            cfg=s5_cfg,
            gallery=gallery,
            detector_config=detector_config,
            detector_checkpoint=detector_checkpoint,
            device=device,
        )
        s5.save_detection_cache(detection_cache, detect_meta, offsets, boxes, confidences)
    else:
        offsets, boxes, confidences = cached_det

    feat_meta = s5.feature_fingerprint(
        cfg=s5_cfg,
        config_path=s5_cfg_path,
        detection_cache=detection_cache,
        reid_checkpoint=reid_checkpoint,
        clip_backbone=clip_backbone,
    )
    expected_feature_shape = (int(offsets[-1]), int(s5_cfg["reid"]["feature_dim"]))
    features = s5.load_feature_cache(feature_cache, feat_meta, expected_feature_shape)
    feature_hit = features is not None
    if features is None:
        features = s5.compute_reid_features(
            cfg=s5_cfg,
            gallery=gallery,
            offsets=offsets,
            boxes=boxes,
            source_root=clipreid_source,
            checkpoint=reid_checkpoint,
            clip_backbone=clip_backbone,
            cache_path=feature_cache,
            cache_meta=feat_meta,
            device=device,
        )

    fingerprint = s5_score_fingerprint(
        s5=s5,
        s5_cfg_path=s5_cfg_path,
        gallery_manifest=gallery_manifest,
        query_manifest=query_manifest,
        detection_cache=detection_cache,
        feature_cache=feature_cache,
    )
    expected_shape = (len(queries), len(gallery))
    score_hit = False
    if score_cache.is_file() and read_json(meta_path(score_cache)) == fingerprint:
        scores = np.load(score_cache, mmap_mode="r", allow_pickle=False)
        try:
            validate_score_matrix(scores, expected_shape, "cached S5 scores")
            score_hit = True
        except (TypeError, ValueError):
            score_hit = False
    if not score_hit:
        scores = s5.compute_scores(
            cfg=s5_cfg,
            queries=queries,
            query_indices=query_indices,
            features=features,
            offsets=offsets,
            output_path=score_cache,
        )
        write_json(meta_path(score_cache), fingerprint)
    else:
        print(f"Using S5 ReID-Set score cache: {rel(score_cache)}", flush=True)

    return scores, {
        "detections_hit": detection_hit,
        "features_hit": feature_hit,
        "scores_hit": score_hit,
        "num_detected_persons": int(offsets[-1]),
    }


def full_map(scores: np.ndarray, gallery: Sequence[dict[str, Any]], queries: Sequence[dict[str, Any]]) -> float:
    gallery_index = {row["image_id"]: i for i, row in enumerate(gallery)}
    if len(gallery_index) != len(gallery):
        raise ValueError("Validation gallery has duplicate image_id")
    aps: list[float] = []
    for qi, query in enumerate(queries):
        image_id = query.get("image_id")
        if image_id not in gallery_index:
            raise ValueError(f"Validation query {qi}: query image missing from gallery")
        positives_raw = query.get("full_positive_ids")
        if not isinstance(positives_raw, list) or not positives_raw:
            raise ValueError(
                f"Validation query {qi}: full_positive_ids is required for alpha selection"
            )
        positives = {gallery_index[x] for x in positives_raw if x in gallery_index}
        self_idx = gallery_index[image_id]
        positives.discard(self_idx)
        if not positives:
            raise ValueError(f"Validation query {qi}: no Full positives")
        row = np.asarray(scores[qi], dtype=np.float32).copy()
        row[self_idx] = -np.inf
        order = np.argsort(-row)
        relevant = np.isin(order, list(positives))
        ranks = np.flatnonzero(relevant) + 1
        if len(ranks) == 0:
            raise ValueError(f"Validation query {qi}: no positive found after ranking")
        aps.append(float(np.mean(np.arange(1, len(ranks) + 1) / ranks)))
    return float(np.mean(aps))


def alpha_fingerprint(
    *,
    cfg: dict[str, Any],
    config_path: Path,
    s5_cfg_path: Path,
    val_gallery: Path,
    val_queries: Path,
    clip_checkpoint: Path,
) -> dict[str, Any]:
    selection = cfg["fusion"]["alpha_selection"]
    return {
        "schema": ALPHA_SCHEMA,
        "adapter_version": ADAPTER_VERSION,
        "metric": str(selection["metric"]),
        "grid": [float(x) for x in selection["grid"]],
        "tie_break": str(selection["tie_break"]),
        "config_sha256": sha256_file(config_path),
        "s5_config_sha256": sha256_file(s5_cfg_path),
        "validation_gallery_sha256": sha256_file(val_gallery),
        "validation_queries_sha256": sha256_file(val_queries),
        "clip_checkpoint_sha256": sha256_file(clip_checkpoint),
    }


def select_alpha(
    *,
    cfg: dict[str, Any],
    config_path: Path,
    s5_cfg_path: Path,
    val_gallery_path: Path,
    val_query_path: Path,
    clip_checkpoint: Path,
    reid_scores: np.ndarray,
    clip_scores: np.ndarray,
) -> tuple[float, dict[str, Any], bool]:
    selection = cfg["fusion"]["alpha_selection"]
    result_path = resolve_path(str(selection["result"]))
    fingerprint = alpha_fingerprint(
        cfg=cfg,
        config_path=config_path,
        s5_cfg_path=s5_cfg_path,
        val_gallery=val_gallery_path,
        val_queries=val_query_path,
        clip_checkpoint=clip_checkpoint,
    )
    current = read_json(result_path)
    if current is not None and current.get("fingerprint") == fingerprint:
        alpha = current.get("selected_alpha")
        if isinstance(alpha, (int, float)) and 0.0 <= float(alpha) <= 1.0:
            print(
                f"Using validation-selected alpha={float(alpha):.2f} from {rel(result_path)}",
                flush=True,
            )
            return float(alpha), current, True

    gallery = load_jsonl(val_gallery_path)
    queries = load_jsonl(val_query_path)
    expected_shape = (len(queries), len(gallery))
    validate_score_matrix(reid_scores, expected_shape, "validation ReID-Set scores")
    validate_score_matrix(clip_scores, expected_shape, "validation CLIP scores")

    results: list[dict[str, float]] = []
    best_alpha: float | None = None
    best_map = -math.inf
    for value in [float(x) for x in selection["grid"]]:
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"alpha grid contains invalid value {value}")
        fused = value * np.asarray(reid_scores) + (1.0 - value) * np.asarray(clip_scores)
        metric = full_map(fused, gallery, queries)
        results.append({"alpha": value, "Full-mAP": metric})
        if metric > best_map + 1e-12:
            best_map = metric
            best_alpha = value
        elif abs(metric - best_map) <= 1e-12 and best_alpha is not None and value < best_alpha:
            best_alpha = value
    if best_alpha is None:
        raise RuntimeError("Alpha selection produced no candidate")
    payload = {
        "method": METHOD_ID,
        "selection_split": "validation",
        "selection_metric": "Full-mAP",
        "selected_alpha": float(best_alpha),
        "selected_full_map": float(best_map),
        "tie_break": str(selection["tie_break"]),
        "candidates": results,
        "fingerprint": fingerprint,
    }
    write_json(result_path, payload)
    print(
        f"Selected alpha={best_alpha:.2f} on validation Full-mAP={best_map:.6f}",
        flush=True,
    )
    return float(best_alpha), payload, False


def fuse_scores(
    reid_scores: np.ndarray,
    clip_scores: np.ndarray,
    alpha: float,
    output_path: Path,
) -> np.ndarray:
    if reid_scores.shape != clip_scores.shape:
        raise ValueError(
            f"Branch score-shape mismatch: ReID={reid_scores.shape}, CLIP={clip_scores.shape}"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out = np.lib.format.open_memmap(
        output_path, mode="w+", dtype=np.float32, shape=reid_scores.shape
    )
    for start in progress_bar(
        range(0, reid_scores.shape[0], 128),
        desc="Fuse ReID-Set + CLIP text",
        total=(reid_scores.shape[0] + 127) // 128,
        unit="batch",
    ):
        end = min(start + 128, reid_scores.shape[0])
        out[start:end] = (
            alpha * np.asarray(reid_scores[start:end], dtype=np.float32)
            + (1.0 - alpha) * np.asarray(clip_scores[start:end], dtype=np.float32)
        )
    out.flush()
    return out


def main() -> None:
    tracker = PhaseTracker(METHOD_ID, total=8)

    with tracker.phase("Load configs, manifests, and pinned S5 adapter"):
        parser = argparse.ArgumentParser()
        parser.add_argument("--config", default=str(DEFAULT_CONFIG))
        args = parser.parse_args()
        config_path = resolve_path(args.config)
        cfg = load_yaml(config_path)
        if str(cfg.get("method")) != METHOD_ID:
            raise ValueError(f"Config method must be {METHOD_ID!r}")

        s5_dir = resolve_path(str(cfg["base_set"]["method_dir"]))
        s5 = load_s5_module(s5_dir)
        s5_cfg_path = resolve_path(str(cfg["base_set"]["config"]))
        s5_cfg = s5.load_yaml(s5_cfg_path)
        if str(s5_cfg.get("method")) != str(cfg["base_set"]["method"]):
            raise ValueError("S5 method/config mismatch")

        gallery_manifest = resolve_path(str(cfg["data"]["gallery_manifest"]))
        query_manifest = resolve_path(str(cfg["data"]["query_manifest"]))
        gallery = load_jsonl(gallery_manifest)
        queries = load_jsonl(query_manifest)
        if not gallery or not queries:
            raise ValueError("Canonical gallery/query manifests must be non-empty")

        selection_cfg = cfg["fusion"]["alpha_selection"]
        alpha_mode = str(selection_cfg.get("mode", "fixed")).strip().lower()
        if alpha_mode not in {"fixed", "validation"}:
            raise ValueError(
                "fusion.alpha_selection.mode must be 'fixed' or 'validation', "
                f"got {alpha_mode!r}"
            )

        alpha = None
        val_gallery_manifest = None
        val_query_manifest = None

        if alpha_mode == "fixed":
            try:
                alpha = float(selection_cfg["fixed_alpha"])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    "fusion.alpha_selection.fixed_alpha must be a numeric value in [0, 1]"
                ) from error
            if not math.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
                raise ValueError(
                    f"fusion.alpha_selection.fixed_alpha must be in [0, 1], got {alpha}"
                )
        else:
            val_cfg = selection_cfg["validation"]
            val_gallery_manifest = resolve_path(str(val_cfg["gallery_manifest"]))
            val_query_manifest = resolve_path(str(val_cfg["query_manifest"]))
            if not val_gallery_manifest.is_file() or not val_query_manifest.is_file():
                raise FileNotFoundError(
                    "S6 alpha_selection.mode='validation' requires separate validation "
                    "manifests to avoid test leakage.\n"
                    f"Expected: {rel(val_gallery_manifest)} and {rel(val_query_manifest)}\n"
                    "Do not point these paths at the canonical evaluation manifests."
                )
            if (
                sha256_file(val_gallery_manifest) == sha256_file(gallery_manifest)
                and sha256_file(val_query_manifest) == sha256_file(query_manifest)
            ):
                raise RuntimeError(
                    "Validation manifests are identical to canonical evaluation manifests; "
                    "refusing to tune alpha on the evaluation set."
                )

        cpr_supervision = "Val only" if alpha_mode == "validation" else "No"

        device = device_from(str(cfg["runtime"].get("device", "cuda")))
        if device.type != "cuda":
            raise RuntimeError("S6 requires CUDA because it reuses the pinned S5 CLIP-ReID adapter")
        clip_checkpoint = resolve_path(str(cfg["clip_text"]["checkpoint"]))
        if not clip_checkpoint.is_file():
            raise FileNotFoundError(
                f"Missing CLIP ViT-L/14 checkpoint: {rel(clip_checkpoint)}. "
                "Run this baseline through run_baseline.py."
            )
        expected_sha = str(cfg["clip_text"]["checkpoint_sha256"])
        actual_sha = sha256_file(clip_checkpoint)
        if actual_sha != expected_sha:
            raise RuntimeError(
                f"CLIP ViT-L/14 checksum mismatch: expected {expected_sha}, got {actual_sha}"
            )
        output_dir = resolve_path(str(cfg["output"]["dir"]))
        output_dir.mkdir(parents=True, exist_ok=True)
        validation_status = "separate" if alpha_mode == "validation" else "not-required"
        tracker.log(
            f"main={len(queries):,}x{len(gallery):,} alpha_mode={alpha_mode} "
            f"validation_manifests={validation_status} device={device}"
        )

    with tracker.phase("Load OpenAI CLIP ViT-L/14"):
        model, preprocess = clip.load(str(clip_checkpoint), device=device, jit=False)
        model.eval()

    val_cache = None
    val_reid_scores = None
    val_clip_scores = None
    val_s5_stats = None
    val_clip_gallery_hit = None
    val_clip_scores_hit = None
    alpha_meta = None
    alpha_cache_hit = False

    with tracker.phase("Prepare validation ReID-Set branch"):
        if alpha_mode == "validation":
            if val_gallery_manifest is None or val_query_manifest is None:
                raise RuntimeError("Internal error: validation manifests were not resolved")
            val_cache = cfg["cache"]["validation"]
            val_reid_scores, val_s5_stats = prepare_s5_scores(
                s5=s5,
                s5_cfg=s5_cfg,
                s5_cfg_path=s5_cfg_path,
                gallery_manifest=val_gallery_manifest,
                query_manifest=val_query_manifest,
                detection_cache=resolve_path(str(val_cache["detections"])),
                feature_cache=resolve_path(str(val_cache["reid_features"])),
                score_cache=resolve_path(str(val_cache["reid_set_scores"])),
                device=device,
            )
            tracker.log(f"validation_s5_cache={val_s5_stats}")
        else:
            tracker.log("skipped: alpha_selection.mode=fixed")

    with tracker.phase("Prepare validation CLIP text branch"):
        if alpha_mode == "validation":
            if (
                val_gallery_manifest is None
                or val_query_manifest is None
                or val_cache is None
            ):
                raise RuntimeError("Internal error: validation state is incomplete")
            val_gallery = load_jsonl(val_gallery_manifest)
            val_queries = load_jsonl(val_query_manifest)
            val_clip_gallery, val_clip_gallery_hit = prepare_clip_gallery_features(
                model=model,
                preprocess=preprocess,
                gallery=val_gallery,
                gallery_manifest=val_gallery_manifest,
                checkpoint=clip_checkpoint,
                cache_path=resolve_path(str(val_cache["clip_gallery_features"])),
                cfg=cfg,
                device=device,
            )
            val_clip_scores, val_clip_scores_hit = compute_clip_text_scores(
                cfg=cfg,
                model=model,
                gallery_features=val_clip_gallery,
                queries=val_queries,
                gallery_manifest=val_gallery_manifest,
                query_manifest=val_query_manifest,
                checkpoint=clip_checkpoint,
                output_path=resolve_path(str(val_cache["clip_text_scores"])),
                device=device,
            )
            tracker.log(
                f"validation_clip_gallery_hit={val_clip_gallery_hit} "
                f"validation_clip_scores_hit={val_clip_scores_hit}"
            )
        else:
            tracker.log("skipped: alpha_selection.mode=fixed")

    with tracker.phase("Select alpha on validation Full-mAP"):
        if alpha_mode == "validation":
            if (
                val_gallery_manifest is None
                or val_query_manifest is None
                or val_reid_scores is None
                or val_clip_scores is None
            ):
                raise RuntimeError("Internal error: validation scores are incomplete")
            alpha, alpha_meta, alpha_cache_hit = select_alpha(
                cfg=cfg,
                config_path=config_path,
                s5_cfg_path=s5_cfg_path,
                val_gallery_path=val_gallery_manifest,
                val_query_path=val_query_manifest,
                clip_checkpoint=clip_checkpoint,
                reid_scores=val_reid_scores,
                clip_scores=val_clip_scores,
            )
            tracker.log(
                f"alpha={alpha:.2f} "
                f"validation_Full-mAP={alpha_meta['selected_full_map']:.6f} "
                f"cache_hit={alpha_cache_hit}"
            )
        else:
            if alpha is None:
                raise RuntimeError("Internal error: fixed alpha was not resolved")
            alpha_meta = {
                "method": METHOD_ID,
                "selection_mode": "fixed",
                "selection_split": None,
                "selection_metric": None,
                "selected_alpha": float(alpha),
                "selected_full_map": None,
            }
            tracker.log(
                f"alpha={alpha:.2f} fixed; no CPR labels used for fusion selection"
            )

    if alpha is None or alpha_meta is None:
        raise RuntimeError("Internal error: alpha selection did not produce a usable value")

    with tracker.phase("Prepare canonical ReID-Set branch"):
        main_cache = cfg["cache"]["main"]
        s5_main_cache = s5_cfg["cache"]
        main_reid_scores, main_s5_stats = prepare_s5_scores(
            s5=s5,
            s5_cfg=s5_cfg,
            s5_cfg_path=s5_cfg_path,
            gallery_manifest=gallery_manifest,
            query_manifest=query_manifest,
            detection_cache=s5.resolve_path(str(s5_main_cache["detections"])),
            feature_cache=s5.resolve_path(str(s5_main_cache["reid_features"])),
            score_cache=resolve_path(str(main_cache["reid_set_scores"])),
            device=device,
        )
        tracker.log(f"main_s5_cache={main_s5_stats}")

    with tracker.phase("Prepare canonical CLIP text branch and fuse scores"):
        main_clip_gallery, main_clip_gallery_hit = prepare_clip_gallery_features(
            model=model,
            preprocess=preprocess,
            gallery=gallery,
            gallery_manifest=gallery_manifest,
            checkpoint=clip_checkpoint,
            cache_path=resolve_path(str(main_cache["clip_gallery_features"])),
            cfg=cfg,
            device=device,
        )
        main_clip_scores, main_clip_scores_hit = compute_clip_text_scores(
            cfg=cfg,
            model=model,
            gallery_features=main_clip_gallery,
            queries=queries,
            gallery_manifest=gallery_manifest,
            query_manifest=query_manifest,
            checkpoint=clip_checkpoint,
            output_path=resolve_path(str(main_cache["clip_text_scores"])),
            device=device,
        )
        scores_path = output_dir / "scores.npy"
        scores = fuse_scores(main_reid_scores, main_clip_scores, alpha, scores_path)
        validate_score_matrix(scores, (len(queries), len(gallery)), "final scores")
        tracker.log(
            f"clip_gallery_hit={main_clip_gallery_hit} clip_scores_hit={main_clip_scores_hit} "
            f"scores={scores.shape}"
        )

    with tracker.phase("Write reproducibility metadata"):
        run = {
            "method": METHOD_ID,
            "display_name": str(cfg.get("display_name")),
            "group": str(cfg.get("group")),
            "cpr_supervision": cpr_supervision,
            "adapter_version": ADAPTER_VERSION,
            "config": rel(config_path),
            "components": {
                "reid_set": {
                    "method": str(cfg["base_set"]["method"]),
                    "config": rel(s5_cfg_path),
                    "detector": "Grounding DINO Swin-T",
                    "reid": "CLIP-ReID ViT-B/16 (MSMT17)",
                    "assignment": "maximum-weight Hungarian",
                    "aggregation": "strict minimum",
                },
                "clip_text": {
                    "model": str(cfg["clip_text"]["name"]),
                    "checkpoint": rel(clip_checkpoint),
                    "checkpoint_sha256": sha256_file(clip_checkpoint),
                    "text_field": str(cfg["clip_text"]["text_field"]),
                    "gallery_representation": "global full-scene CLIP image embedding",
                },
            },
            "fusion": {
                "formula": "alpha * ReID-Set + (1-alpha) * CLIP-text",
                "score_normalization": "none",
                "selected_alpha": alpha,
                "selection_mode": alpha_mode,
                "fixed_alpha": float(alpha) if alpha_mode == "fixed" else None,
                "selection_split": "validation" if alpha_mode == "validation" else None,
                "selection_metric": "Full-mAP" if alpha_mode == "validation" else None,
                "selection_result": (
                    rel(resolve_path(str(selection_cfg["result"])))
                    if alpha_mode == "validation"
                    else None
                ),
                "validation_full_map": (
                    float(alpha_meta["selected_full_map"])
                    if alpha_mode == "validation"
                    else None
                ),
                "grid": (
                    [float(x) for x in selection_cfg["grid"]]
                    if alpha_mode == "validation"
                    else None
                ),
            },
            "benchmark_adaptation": {
                "purpose": (
                    "Strong conventional baseline: CLIP-ReID answers WHO at person-set level; "
                    "global CLIP text similarity answers WHAT at scene level."
                ),
                "validation_labels_used_only_for_alpha_selection": alpha_mode == "validation",
                "evaluation_labels_used_for_alpha_selection": False,
                "query_image_removed_inside_method": False,
                "score_direction": "higher_is_better",
            },
            "runtime": cfg["runtime"],
            "cache": {
                "main_s5": main_s5_stats,
                "validation_s5": (
                    val_s5_stats if alpha_mode == "validation" else None
                ),
                "alpha_selection_hit": (
                    alpha_cache_hit if alpha_mode == "validation" else False
                ),
                "main_clip_gallery_hit": main_clip_gallery_hit,
                "main_clip_scores_hit": main_clip_scores_hit,
                "validation_clip_gallery_hit": (
                    val_clip_gallery_hit if alpha_mode == "validation" else None
                ),
                "validation_clip_scores_hit": (
                    val_clip_scores_hit if alpha_mode == "validation" else None
                ),
            },
            "num_queries": len(queries),
            "num_gallery": len(gallery),
            "scores": rel(scores_path),
            "higher_is_better": True,
        }
        run_path = output_dir / "run.json"
        write_json(run_path, run)
        tracker.log(f"scores={rel(scores_path)} run={rel(run_path)} alpha={alpha:.2f}")

    tracker.finish()


if __name__ == "__main__":
    main()
