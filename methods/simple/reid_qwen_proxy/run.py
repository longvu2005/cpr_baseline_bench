#!/usr/bin/env python3
"""S9: ReID-Set + Qwen Edit Proxy.

Final score:
    alpha * ReID-Set + (1 - alpha) * generated-proxy similarity

Main/evaluation branch reuse:
- ReID-Set branch: reuses S5 scores.npy
- Proxy branch: reuses S8 scores.npy

Validation branch construction:
- reuses S5 adapter logic to build validation ReID-Set scores
- reuses S8 adapter logic to build validation generated-proxy scores

Alpha is selected strictly on a separate validation split by Full-mAP only.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import os
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

from cpr_person.adapter import matching_metadata, manifest_identity
from cpr_person.adapter import require_person_branch
from cpr_person.adapter import require_manifest_branch
from cpr_baselines.scene_protocol import check_config
from cpr_baselines.scene_clip import load_scene_clip, verify_scene_clip, scene_identity
from cpr_person.protocol import require_aligned_run
DEFAULT_CONFIG = Path(__file__).resolve().parent / "config.yaml"
METHOD_ID = "reid_set_qwen_edit_proxy"
ADAPTER_VERSION = "2026-08-13-v1-hybrid-val-fullmap"
BRANCH_SCORE_SCHEMA = 1
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


def validate_score_matrix(scores: np.ndarray, shape: tuple[int, int], label: str) -> None:
    if scores.shape != shape:
        raise ValueError(f"{label}: shape {scores.shape}, expected {shape}")
    if not np.issubdtype(scores.dtype, np.floating):
        raise TypeError(f"{label} must be floating point")
    for start in range(0, shape[0], 256):
        block = np.asarray(scores[start : start + 256])
        if not np.isfinite(block).all():
            raise ValueError(f"{label} contains NaN/Inf")


def load_source_scores(*, scores_path: Path, run_path: Path, expected_method: str, shape: tuple[int, int], label: str) -> np.ndarray:
    from cpr_person.protocol import require_score_artifact
    if not scores_path.is_file():
        raise FileNotFoundError(
            f"Missing {label} source scores: {rel(scores_path)}. "
            f"Run `python run_baseline.py {expected_method}` first."
        )
    if not run_path.is_file():
        raise FileNotFoundError(
            f"Missing {label} source run.json: {rel(run_path)}. "
            f"Run `python run_baseline.py {expected_method}` first."
        )
    run = read_json(run_path)
    if run is None or run.get("method") != expected_method:
        raise ValueError(
            f"{label} run.json mismatch: expected method {expected_method!r}, got {None if run is None else run.get('method')!r}"
        )
    require_score_artifact(run, scores_path)
    scores = np.load(scores_path, mmap_mode="r", allow_pickle=False)
    validate_score_matrix(scores, shape, label)
    return scores


def branch_score_fingerprint(*, kind: str, config_path: Path, gallery_manifest: Path, query_manifest: Path, extras: dict[str, Any]) -> dict[str, Any]:
    return {
        "execution": execution_identity(__file__),
        "schema": BRANCH_SCORE_SCHEMA,
        "adapter_version": ADAPTER_VERSION,
        "kind": kind,
        "config_sha256": sha256_file(config_path),
        "gallery_manifest_sha256": sha256_file(gallery_manifest),
        "gallery_image_content_sha256": manifest_image_digest(ROOT, gallery_manifest),
        "query_manifest_sha256": sha256_file(query_manifest),
        "extras": extras,
    }



def load_cached_scores(path: Path, expected_meta: dict[str, Any], shape: tuple[int, int], label: str) -> np.ndarray | None:
    current = read_json(meta_path(path))
    if not path.is_file() or current != expected_meta:
        return None
    try:
        scores = np.load(path, mmap_mode="r", allow_pickle=False)
    except Exception:
        return None
    try:
        validate_score_matrix(scores, shape, label)
    except Exception:
        return None
    print(f"Using cached {label}: {rel(path)}", flush=True)
    return scores


def save_scores(path: Path, scores: np.ndarray, meta: dict[str, Any]) -> np.ndarray:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".part")
    temp.unlink(missing_ok=True)
    try:
        with temp.open("wb") as handle:
            np.save(handle, np.asarray(scores, dtype=np.float32))
        os.replace(temp, path)
    except Exception:
        temp.unlink(missing_ok=True)
        raise
    write_json(meta_path(path), meta)
    return np.load(path, mmap_mode="r", allow_pickle=False)


def build_gallery_index(gallery: Sequence[dict[str, Any]]) -> dict[Any, int]:
    index: dict[Any, int] = {}
    for gi, row in enumerate(gallery):
        if "image_id" not in row:
            raise KeyError(f"Gallery row {gi} missing image_id")
        image_id = row["image_id"]
        if image_id in index:
            raise ValueError(f"Duplicate gallery image_id: {image_id!r}")
        index[image_id] = gi
    return index


def select_alpha(*, cfg: dict[str, Any], config_path: Path, val_gallery_manifest: Path, val_query_manifest: Path, val_reid_scores: np.ndarray, val_proxy_scores: np.ndarray, gallery: Sequence[dict[str, Any]], queries: Sequence[dict[str, Any]]) -> tuple[float, dict[str, Any]]:
    selection = cfg["fusion"]["alpha_selection"]
    result_path = resolve_path(str(selection["result"]))
    grid = sorted({float(x) for x in selection["grid"]})
    if not grid or any(not 0.0 <= value <= 1.0 for value in grid):
        raise ValueError("alpha_selection.grid must be nonempty and in [0, 1]")
    tie_break = str(selection.get("tie_break", "smallest_alpha"))
    if tie_break != "smallest_alpha":
        raise ValueError("Only tie_break=smallest_alpha is supported")
    expected = {
        "execution": execution_identity(__file__),
        "schema": ALPHA_SCHEMA,
        "selection_split": "val",
        "adapter_version": ADAPTER_VERSION,
        "config_sha256": sha256_file(config_path),
        "val_gallery_manifest_sha256": sha256_file(val_gallery_manifest),
        "val_query_manifest_sha256": sha256_file(val_query_manifest),
        "val_reid_scores_sha256": sha256_file(resolve_path(str(cfg["cache"]["validation"]["reid_scores"]))),
        "val_proxy_scores_sha256": sha256_file(resolve_path(str(cfg["cache"]["validation"]["proxy_scores"]))),
        "grid": grid,
        "metric": str(selection["metric"]),
        "tie_break": tie_break,
    }
    current = read_json(result_path)
    if current is not None:
        core = {k: current.get(k) for k in expected}
        if core == expected and current.get("best_alpha") in grid:
            return float(current["best_alpha"]), current

    rows: list[dict[str, Any]] = []
    best_alpha = None
    best_map = None
    for alpha in progress_bar(grid, desc="Select alpha", total=len(grid), unit="alpha"):
        fused = alpha * np.asarray(val_reid_scores, dtype=np.float32) + (1.0 - alpha) * np.asarray(val_proxy_scores, dtype=np.float32)
        score = full_map(fused, gallery, queries)
        rows.append({"alpha": float(alpha), "Full-mAP": float(score)})
        if best_map is None or score > best_map + 1e-12 or (abs(score - best_map) <= 1e-12 and float(alpha) < float(best_alpha)):
            best_map = float(score)
            best_alpha = float(alpha)
    payload = {
        **expected,
        "best_alpha": float(best_alpha),
        "best_full_map": float(best_map),
        "trials": rows,
    }
    write_json(result_path, payload)
    return float(best_alpha), payload


def build_validation_reid_scores(*, cfg, config_path, val_gallery_manifest, val_query_manifest,
                                 val_gallery, val_queries, s5, tracker):
    cache_cfg = cfg["cache"]["validation"]
    base_cfg_path = resolve_path(cfg["base_set"]["config"])
    base_cfg = load_yaml(base_cfg_path)
    person, features, query_sets, stats = s5.prepare_person_branch(cfg=base_cfg, config_path=base_cfg_path,
        gallery_manifest=val_gallery_manifest, query_manifest=val_query_manifest,
        device=s5.device_from(cfg["runtime"]["device"]), feature_cache=resolve_path(cache_cfg["reid_features"]))
    output_path = resolve_path(cache_cfg["reid_scores"])
    fp = {"execution": execution_identity(__file__), "schema": "shared-reid-branch-v1", "localization": person.localization.fingerprint,
          "feature_meta": read_json(s5.meta_path(resolve_path(cache_cfg["reid_features"]))),
          "setmatch": base_cfg["setmatch"]}
    cached = load_cached_scores(output_path, fp, (len(val_queries),len(val_gallery)), "shared validation ReID")
    if cached is not None:
        return cached
    scores = s5.compute_scores(cfg=base_cfg, queries=val_queries,
        query_indices=s5.query_gallery_indices(val_queries, s5.build_gallery_index(val_gallery)),
        features=features, offsets=person.detections.offsets, output_path=output_path,
        query_features_list=query_sets)
    return save_scores(output_path, scores, fp)


def build_validation_proxy_scores(*, cfg: dict[str, Any], config_path: Path, val_gallery_manifest: Path, val_query_manifest: Path, val_gallery: Sequence[dict[str, Any]], val_queries: Sequence[dict[str, Any]], s8, tracker: PhaseTracker) -> np.ndarray:
    cache_cfg = cfg["cache"]["validation"]
    output_path = resolve_path(str(cache_cfg["proxy_scores"]))
    expected_shape = (len(val_queries), len(val_gallery))
    proxy_cfg_path = resolve_path(str(cfg["proxy"]["config"]))
    proxy_cfg = load_yaml(proxy_cfg_path)
    checkpoint = check_config(proxy_cfg["retriever"])
    clip_sha = verify_scene_clip(checkpoint, proxy_cfg["retriever"]["name"])
    device = s8.device_from(str(cfg["runtime"]["device"]))
    generator_dir, generator_marker, _marker_data = s8.validate_generator_artifacts(proxy_cfg)
    val_proxy_cfg = json.loads(json.dumps(proxy_cfg))
    val_proxy_cfg["runtime"]["generator_dtype"] = str(cfg["runtime"]["proxy_generator_dtype"])
    val_proxy_cfg["runtime"]["clip_image_batch_size"] = int(cfg["runtime"]["proxy_clip_image_batch_size"])
    val_proxy_cfg["runtime"]["clip_score_batch_size"] = int(cfg["runtime"]["proxy_clip_score_batch_size"])
    val_proxy_cfg["runtime"]["num_workers"] = int(cfg["runtime"]["num_workers"])
    val_proxy_cfg["cache"]["edited_queries_dir"] = str(cache_cfg["proxy_edited_queries_dir"])
    val_proxy_cfg["cache"]["edited_queries_manifest"] = str(cache_cfg["proxy_edited_queries_manifest"])
    val_proxy_cfg["cache"]["edited_query_features"] = str(cache_cfg["proxy_query_features"])
    val_proxy_cfg["cache"]["gallery_features"] = str(cache_cfg["proxy_gallery_features"])
    edit_fp = s8.edit_fingerprint(cfg=val_proxy_cfg, config_path=proxy_cfg_path,
        gallery_manifest=val_gallery_manifest, query_manifest=val_query_manifest,
        generator_marker=generator_marker)
    edited_manifest_path = resolve_path(str(cache_cfg["proxy_edited_queries_manifest"]))
    edited_paths = s8.load_edit_cache(edited_manifest_path,
        resolve_path(str(cache_cfg["proxy_edited_queries_dir"])), edit_fp, val_queries)
    if edited_paths is None:
        with tracker.phase("Generate validation proxy images"):
            edited_paths, edited_manifest_path, _ = s8.generate_or_load_edited_queries(
                cfg=val_proxy_cfg, config_path=proxy_cfg_path, gallery_manifest=val_gallery_manifest,
                query_manifest=val_query_manifest, generator_marker=generator_marker,
                generator_dir=generator_dir, gallery=val_gallery, queries=val_queries, device=device)
    gallery_cache_fp = s8.clip_gallery_cache_fingerprint(checkpoint, val_gallery_manifest, str(proxy_cfg["retriever"]["name"]))
    extras = {
        "scene_clip_sha256": clip_sha,
        "scene_clip": scene_identity(checkpoint),
        "encoding_runtime": cfg["runtime"],
        "proxy_config_sha256": sha256_file(proxy_cfg_path),
        "generator_fingerprint": edit_fp,
        "gallery_fingerprint": gallery_cache_fp,
        "edited_manifest_sha256": sha256_file(edited_manifest_path),
        "edited_image_sha256s": [sha256_file(path) for path in edited_paths],
    }
    expected_meta = branch_score_fingerprint(kind="validation_proxy", config_path=config_path, gallery_manifest=val_gallery_manifest, query_manifest=val_query_manifest, extras=extras)
    cached = load_cached_scores(output_path, expected_meta, expected_shape, "validation proxy scores")
    if cached is not None:
        return cached

    with tracker.phase("Build validation generated-proxy branch"):
        model, preprocess = load_scene_clip(checkpoint, device)
        gallery_features, _ = s8.clip_gallery_features(model, preprocess, val_gallery, resolve_path(str(cache_cfg["proxy_gallery_features"])), val_proxy_cfg["runtime"], device, gallery_cache_fp)
        query_cache_fp = s8.query_feature_cache_fingerprint(proxy_cfg_path, edited_manifest_path, checkpoint, val_query_manifest)
        query_features, _ = s8.clip_query_image_features(model, preprocess, edited_paths, resolve_path(str(cache_cfg["proxy_query_features"])), val_proxy_cfg["runtime"], device, query_cache_fp)
        gallery_tensor = torch.from_numpy(np.asarray(gallery_features)).to(device)
        scores = np.lib.format.open_memmap(output_path, mode="w+", dtype=np.float32, shape=expected_shape)
        batch = int(cfg["runtime"]["proxy_clip_score_batch_size"])
        for start in progress_bar(range(0, len(val_queries), batch), desc="Proxy score validation queries", total=(len(val_queries) + batch - 1) // batch, unit="batch"):
            end = min(start + batch, len(val_queries))
            query = torch.from_numpy(np.asarray(query_features[start:end])).to(device)
            scores[start:end] = (query @ gallery_tensor.T).cpu().numpy()
        scores.flush()
        del model
        del gallery_tensor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        validate_score_matrix(scores, expected_shape, "validation proxy scores")
        return save_scores(output_path, np.asarray(scores, dtype=np.float32), expected_meta)


def main() -> None:
    tracker = PhaseTracker(METHOD_ID, total=6)

    with tracker.phase("Load config, source methods, and manifests"):
        parser = argparse.ArgumentParser()
        parser.add_argument("--config", default=str(DEFAULT_CONFIG))
        args = parser.parse_args()
        config_path = resolve_path(args.config)
        cfg = load_yaml(config_path)

        main_gallery_manifest = resolve_path(str(cfg["data"]["gallery_manifest"]))
        main_query_manifest = resolve_path(str(cfg["data"]["query_manifest"]))
        main_gallery = load_jsonl(main_gallery_manifest)
        main_queries = load_jsonl(main_query_manifest)
        expected_shape = (len(main_queries), len(main_gallery))

        selection_cfg = cfg["fusion"]["alpha_selection"]
        val_gallery_manifest = resolve_path(str(selection_cfg["validation"]["gallery_manifest"]))
        val_query_manifest = resolve_path(str(selection_cfg["validation"]["query_manifest"]))
        validation_fingerprint = verify_tuning_manifests(
            ROOT, main_gallery_manifest, main_query_manifest, val_gallery_manifest, val_query_manifest
        )
        val_gallery = load_jsonl(val_gallery_manifest)
        val_queries = load_jsonl(val_query_manifest)

        s5 = load_module(resolve_path(str(cfg["base_set"]["method_dir"])), "cpr_s9_source_s5")
        s8 = load_module(resolve_path(str(cfg["proxy"]["method_dir"])), "cpr_s9_source_s8")
        tracker.log(f"main_gallery={len(main_gallery):,} main_queries={len(main_queries):,} val_gallery={len(val_gallery):,} val_queries={len(val_queries):,}")

    with tracker.phase("Load main branch scores from S5 and S8"):
        main_reid_scores = load_source_scores(scores_path=resolve_path(str(cfg["base_set"]["main_scores"])), run_path=resolve_path(str(cfg["base_set"]["main_run"])), expected_method=str(cfg["base_set"]["method"]), shape=expected_shape, label="main ReID-Set")
        main_proxy_scores = load_source_scores(scores_path=resolve_path(str(cfg["proxy"]["main_scores"])), run_path=resolve_path(str(cfg["proxy"]["main_run"])), expected_method=str(cfg["proxy"]["method"]), shape=expected_shape, label="main generated-proxy")

    main_person_protocol = require_person_branch(
        read_json(resolve_path(cfg["base_set"]["main_run"])), main_gallery_manifest, main_query_manifest)
    proxy_run = read_json(resolve_path(cfg["proxy"]["main_run"]))
    require_manifest_branch(proxy_run, main_gallery_manifest, main_query_manifest)
    from cpr_baselines.scene_clip import require_current_scene_run
    require_current_scene_run(proxy_run)
    require_aligned_run(proxy_run, cfg["proxy"]["method"])
    val_reid_scores = build_validation_reid_scores(cfg=cfg, config_path=config_path, val_gallery_manifest=val_gallery_manifest, val_query_manifest=val_query_manifest, val_gallery=val_gallery, val_queries=val_queries, s5=s5, tracker=tracker)
    val_proxy_scores = build_validation_proxy_scores(cfg=cfg, config_path=config_path, val_gallery_manifest=val_gallery_manifest, val_query_manifest=val_query_manifest, val_gallery=val_gallery, val_queries=val_queries, s8=s8, tracker=tracker)

    with tracker.phase("Select alpha on validation Full-mAP"):
        alpha, alpha_payload = select_alpha(cfg=cfg, config_path=config_path, val_gallery_manifest=val_gallery_manifest, val_query_manifest=val_query_manifest, val_reid_scores=val_reid_scores, val_proxy_scores=val_proxy_scores, gallery=val_gallery, queries=val_queries)
        tracker.log(f"selected_alpha={alpha:.2f} best_full_map={alpha_payload.get('best_full_map'):.6f}")

    with tracker.phase("Fuse main branch scores"):
        output_dir = resolve_path(str(cfg["output"]["dir"]))
        output_dir.mkdir(parents=True, exist_ok=True)
        scores_path = output_dir / "scores.npy"
        fused = np.lib.format.open_memmap(scores_path, mode="w+", dtype=np.float32, shape=expected_shape)
        batch = 256
        for start in progress_bar(range(0, len(main_queries), batch), desc="Fuse main queries", total=(len(main_queries) + batch - 1) // batch, unit="batch"):
            end = min(start + batch, len(main_queries))
            fused[start:end] = alpha * np.asarray(main_reid_scores[start:end], dtype=np.float32) + (1.0 - alpha) * np.asarray(main_proxy_scores[start:end], dtype=np.float32)
        fused.flush()
        validate_score_matrix(fused, expected_shape, "final scores")

    with tracker.phase("Write run metadata"):
        output_dir = resolve_path(str(cfg["output"]["dir"]))
        run_path = output_dir / "run.json"
        payload = {
            "execution": execution_identity(__file__),
            "scene_clip_sha256": proxy_run["scene_clip_sha256"],
            "scene_clip": proxy_run["scene_clip"],
            "shared_person_protocol": main_person_protocol,
            "setmatch": matching_metadata(load_yaml(resolve_path(cfg["base_set"]["config"])), score_space="cosine"),
            "method": cfg["method"],
            "display_name": cfg["display_name"],
            "group": cfg["group"],
            "validation_fingerprint": validation_fingerprint,
            "cpr_supervision": cfg["cpr_supervision"],
            "formula": cfg["fusion"]["formula"],
            "alpha": alpha,
            "alpha_selection": rel(resolve_path(str(cfg["fusion"]["alpha_selection"]["result"]))),
            "source_methods": {
                "reid_set": str(cfg["base_set"]["method"]),
                "generated_proxy": str(cfg["proxy"]["method"]),
            },
            "main_source_scores": {
                "reid_set": rel(resolve_path(str(cfg["base_set"]["main_scores"]))),
                "generated_proxy": rel(resolve_path(str(cfg["proxy"]["main_scores"]))),
            },
            "validation_branch_scores": {
                "reid_set": rel(resolve_path(str(cfg["cache"]["validation"]["reid_scores"]))),
                "generated_proxy": rel(resolve_path(str(cfg["cache"]["validation"]["proxy_scores"]))),
            },
            "config": rel(config_path),
            "num_queries": len(main_queries),
            "num_gallery": len(main_gallery),
            "scores": rel(resolve_path(str(cfg["output"]["dir"])) / "scores.npy"),
            "higher_is_better": True,
        }
        payload["data_fingerprint"] = manifest_identity(payload["shared_person_protocol"])
        write_json(run_path, payload)
        tracker.log(f"scores={rel(resolve_path(str(cfg['output']['dir'])) / 'scores.npy')} run={rel(run_path)}")

    tracker.finish()


if __name__ == "__main__":
    main()
