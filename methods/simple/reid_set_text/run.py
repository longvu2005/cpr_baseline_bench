#!/usr/bin/env python3
"""S6: Grounding DINO + CLIP-ReID - Set + Text.

Final score:
    alpha * ReID-Set + (1 - alpha) * CLIP-text

The ReID-Set branch is exactly the S5 adapter (same Grounding DINO detections,
CLIP-ReID MSMT17 embeddings, shared Subject localization and Partial SetMatch).
The semantic branch is global OpenAI CLIP ViT-B/32 text-to-gallery similarity.
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
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluate import full_map
from cpr_baselines.provenance import execution_identity
from cpr_person.validation import verify_tuning_manifests
from benchmark_data import manifest_image_digest
from benchmark_progress import PhaseTracker, progress_bar  # noqa: E402

from cpr_baselines.scene_clip import gallery_features, gallery_fingerprint as clip_gallery_fingerprint
from cpr_baselines.scene_protocol import check_config
from cpr_baselines.scene_clip import load_scene_clip, verify_scene_clip, scene_identity, text_features as hf_text_features
from cpr_person.adapter import matching_metadata, manifest_identity
DEFAULT_CONFIG = Path(__file__).resolve().parent / "config.yaml"
METHOD_ID = "groundingdino_clipreid_set_text"
ADAPTER_VERSION = "2026-08-14-v2-fixed-or-val-fusion"
SCORE_CACHE_SCHEMA = 1
ALPHA_SCHEMA = 2


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


def validate_score_matrix(scores: np.ndarray, shape: tuple[int, int], label: str) -> None:
    if scores.shape != shape:
        raise ValueError(f"{label} shape={scores.shape}, expected {shape}")
    if not np.issubdtype(scores.dtype, np.floating):
        raise TypeError(f"{label} must be floating point")
    for start in range(0, shape[0], 256):
        if not np.isfinite(np.asarray(scores[start : start + 256])).all():
            raise ValueError(f"{label} contains NaN/Inf")


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
    fingerprint = clip_gallery_fingerprint(checkpoint, gallery_manifest, cfg["clip_text"]["name"])
    features, _, hit = gallery_features(
        model, preprocess, gallery, cache_path, cfg["runtime"], device, fingerprint,
    )
    return features, hit


@torch.no_grad()
def encode_text(model, texts: Sequence[str], batch_size: int, device: torch.device) -> np.ndarray:
    return hf_text_features(model, list(texts), batch_size)


def clip_score_fingerprint(
    *,
    cfg: dict[str, Any],
    checkpoint: Path,
    gallery_manifest: Path,
    query_manifest: Path,
) -> dict[str, Any]:
    return {
        "execution": execution_identity(__file__),
        "schema": SCORE_CACHE_SCHEMA,
        "kind": "clip_text_scores",
        "encoding_runtime": cfg["runtime"],
        "adapter_version": ADAPTER_VERSION,
        "model": str(cfg["clip_text"]["name"]),
        "text_field": str(cfg["clip_text"]["text_field"]),
        "scene_clip": scene_identity(checkpoint),
        "gallery_fingerprint": clip_gallery_fingerprint(checkpoint, gallery_manifest, cfg["clip_text"]["name"]),
        "gallery_manifest_sha256": sha256_file(gallery_manifest),
        "gallery_image_content_sha256": manifest_image_digest(ROOT, gallery_manifest),
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


def prepare_s5_scores(*, s5, s5_cfg, s5_cfg_path, gallery_manifest, query_manifest,
                      detection_cache, feature_cache, score_cache, device):
    person, features, query_sets, stats = s5.prepare_person_branch(cfg=s5_cfg, config_path=s5_cfg_path,
        gallery_manifest=gallery_manifest, query_manifest=query_manifest, device=device, feature_cache=feature_cache)
    fingerprint = {"execution": execution_identity(__file__), "schema": "shared-reid-branch-v1", "native_adapter": s5.ADAPTER_VERSION,
                   "shared": person.metadata()["localization_fingerprint"],
                   "features_meta": read_json(s5.meta_path(feature_cache)), "setmatch": s5_cfg["setmatch"]}
    shape = (len(person.queries), len(person.gallery))
    score_hit = False
    if score_cache.is_file() and read_json(meta_path(score_cache)) == fingerprint:
        try:
            scores = np.load(score_cache, mmap_mode="r", allow_pickle=False)
            validate_score_matrix(scores, shape, "shared ReID branch")
            score_hit = True
        except (OSError, ValueError, TypeError):
            pass
    if not score_hit:
        scores = s5.compute_scores(cfg=s5_cfg, queries=person.queries,
            query_indices=s5.query_gallery_indices(person.queries, s5.build_gallery_index(person.gallery)),
            features=features, offsets=person.detections.offsets, output_path=score_cache,
            query_features_list=query_sets)
        write_json(meta_path(score_cache), fingerprint)
    return scores, {**stats, "scores_hit": score_hit}


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
        "execution": execution_identity(__file__),
        "schema": ALPHA_SCHEMA,
        "adapter_version": ADAPTER_VERSION,
        "metric": str(selection["metric"]),
        "grid": [float(x) for x in selection["grid"]],
        "tie_break": str(selection["tie_break"]),
        "config_sha256": sha256_file(config_path),
        "s5_config_sha256": sha256_file(s5_cfg_path),
        "validation_gallery_sha256": sha256_file(val_gallery),
        "validation_queries_sha256": sha256_file(val_queries),
        "scene_clip": scene_identity(clip_checkpoint),
        "validation_reid_scores_sha256": sha256_file(
            resolve_path(str(cfg["cache"]["validation"]["reid_set_scores"]))
        ),
        "validation_clip_scores_sha256": sha256_file(
            resolve_path(str(cfg["cache"]["validation"]["clip_text_scores"]))
        ),
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
        "selection_split": "val",
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
        alpha_mode = str(selection_cfg.get("mode", "validation")).strip().lower()
        if alpha_mode not in {"fixed", "validation"}:
            raise ValueError(
                "fusion.alpha_selection.mode must be 'fixed' or 'validation', "
                f"got {alpha_mode!r}"
            )

        alpha = None
        validation_fingerprint = None
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
            validation_fingerprint = verify_tuning_manifests(
                ROOT, gallery_manifest, query_manifest, val_gallery_manifest, val_query_manifest
            )

        cpr_supervision = "Val only" if alpha_mode == "validation" else "No"

        device = device_from(str(cfg["runtime"].get("device", "cuda")))
        if device.type != "cuda":
            raise RuntimeError("S6 requires CUDA because it reuses the pinned S5 CLIP-ReID adapter")
        clip_checkpoint = check_config(cfg["clip_text"])
        if not clip_checkpoint.is_dir():
            raise FileNotFoundError(
                f"Missing CLIP ViT-B/32 checkpoint: {rel(clip_checkpoint)}. "
                "Run this baseline through run_baseline.py."
            )
        actual_sha = verify_scene_clip(clip_checkpoint, cfg["clip_text"]["name"])
        output_dir = resolve_path(str(cfg["output"]["dir"]))
        output_dir.mkdir(parents=True, exist_ok=True)
        validation_status = "separate" if alpha_mode == "validation" else "not-required"
        tracker.log(
            f"main={len(queries):,}x{len(gallery):,} alpha_mode={alpha_mode} "
            f"validation_manifests={validation_status} device={device}"
        )

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

    with tracker.phase("Load frozen HF CLIP ViT-B/32"):
        model, preprocess = load_scene_clip(clip_checkpoint, device)

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

    with tracker.phase("Resolve fusion alpha"):
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
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    with tracker.phase("Write reproducibility metadata"):
        run = {
            "execution": execution_identity(__file__),
            "scene_clip_sha256": actual_sha,
            "scene_clip": scene_identity(clip_checkpoint),
            "shared_person_protocol": main_s5_stats["shared_person_protocol"],
            "setmatch": matching_metadata(s5_cfg, score_space="cosine"),
            "method": METHOD_ID,
            "display_name": str(cfg.get("display_name")),
            "group": str(cfg.get("group")),
            "cpr_supervision": cpr_supervision,
            "validation_fingerprint": validation_fingerprint,
            "adapter_version": ADAPTER_VERSION,
            "config": rel(config_path),
            "components": {
                "reid_set": {
                    "method": str(cfg["base_set"]["method"]),
                    "config": rel(s5_cfg_path),
                    "detector": "Shared RCR Grounding DINO Tiny boxes",
                    "reid": "CLIP-ReID ViT-B/16 (MSMT17)",
                    "assignment": "maximum-weight Hungarian",
                    "aggregation": "mean over all predicted query components",
                },
                "clip_text": {
                    "model": str(cfg["clip_text"]["name"]),
                    "checkpoint": rel(clip_checkpoint),
                    "scene_clip": scene_identity(clip_checkpoint),
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
                "selection_split": "val" if alpha_mode == "validation" else None,
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
        run["data_fingerprint"] = manifest_identity(run["shared_person_protocol"])
        write_json(run_path, run)
        tracker.log(f"scores={rel(scores_path)} run={rel(run_path)} alpha={alpha:.2f}")

    tracker.finish()


if __name__ == "__main__":
    main()
