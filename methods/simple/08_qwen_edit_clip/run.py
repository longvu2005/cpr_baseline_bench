#!/usr/bin/env python3
"""S8: frozen Qwen image editing followed by OpenAI CLIP ViT-L/14 retrieval.

For each canonical query, the full reference scene and canonical modification text
are sent to a frozen image editor, producing one synthetic target image. CLIP
then embeds that edited image and every full gallery scene. No CPR labels,
 target_ids, positives, boxes, or case annotations are consumed here.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
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
METHOD_ID = "qwen_image_edit_clip"
ADAPTER_VERSION = "2026-08-13-v1-fixed-prompt-offline"
EDIT_CACHE_SCHEMA = 1
QUERY_FEATURE_CACHE_SCHEMA = 1
CLIP_GALLERY_CACHE_SCHEMA = 2  # intentionally identical to existing CLIP baselines


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise TypeError(f"{path}:{line_no}: JSONL row must be an object")
            rows.append(row)
    return rows


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"Expected YAML mapping: {path}")
    return value


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


def device_from(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but unavailable")
    return device


def gallery_image_path(row: dict[str, Any], index: int) -> Path:
    value = row.get("path")
    if not isinstance(value, str) or not value.strip():
        raise KeyError(f"Gallery row {index} has no usable path")
    path = resolve_path(value)
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


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


def query_gallery_indices(
    queries: Sequence[dict[str, Any]], gallery_index: dict[Any, int]
) -> np.ndarray:
    indices: list[int] = []
    for qi, query in enumerate(queries):
        image_id = query.get("image_id")
        if image_id not in gallery_index:
            raise ValueError(f"Query row {qi}: image_id {image_id!r} missing from gallery")
        indices.append(gallery_index[image_id])
    return np.asarray(indices, dtype=np.int64)


def validate_generator_artifacts(cfg: dict[str, Any]) -> tuple[Path, Path, dict[str, Any]]:
    gen = cfg["generator"]
    checkpoint_dir = resolve_path(str(gen["checkpoint_dir"]))
    marker = resolve_path(str(gen["prepared_marker"]))
    if not marker.is_file():
        raise FileNotFoundError(
            f"Missing generator prepared marker: {rel(marker)}. Run download_checkpoint.py first."
        )
    try:
        data = json.loads(marker.read_text(encoding="utf-8"))
    except Exception as error:
        raise RuntimeError(f"Invalid generator prepared marker: {rel(marker)}") from error
    if data.get("repo_id") != str(gen["repo_id"]):
        raise RuntimeError("Generator marker repo_id does not match config")
    if data.get("revision") != str(gen["revision"]):
        raise RuntimeError("Generator marker revision does not match config")
    if data.get("pipeline_class") != str(gen["pipeline_class"]):
        raise RuntimeError("Generator marker pipeline_class does not match config")
    files = data.get("files")
    if not isinstance(files, list) or not files:
        raise RuntimeError("Generator marker has no file inventory")
    for item in files:
        if not isinstance(item, dict):
            raise RuntimeError("Malformed generator marker file entry")
        path = checkpoint_dir / str(item.get("path", ""))
        size = item.get("size")
        if not path.is_file() or not isinstance(size, int) or path.stat().st_size != size:
            raise RuntimeError(
                f"Generator snapshot is incomplete/stale at {rel(path)}; rerun checkpoint preparation"
            )
    return checkpoint_dir, marker, data


def fixed_prompt(cfg: dict[str, Any], modification: str) -> str:
    template = str(cfg["generator"]["prompt_template"])
    if template.count("{modification}") != 1:
        raise ValueError(
            "generator.prompt_template must contain exactly one {modification} placeholder"
        )
    text = str(modification or "").strip()
    if not text:
        raise ValueError("Canonical query text is empty")
    return template.replace("{modification}", text)


def edit_fingerprint(
    *,
    cfg: dict[str, Any],
    config_path: Path,
    gallery_manifest: Path,
    query_manifest: Path,
    generator_marker: Path,
) -> dict[str, Any]:
    gen = cfg["generator"]
    return {
        "schema": EDIT_CACHE_SCHEMA,
        "adapter_version": ADAPTER_VERSION,
        "config_sha256": sha256_file(config_path),
        "gallery_manifest_sha256": sha256_file(gallery_manifest),
        "query_manifest_sha256": sha256_file(query_manifest),
        "generator_marker_sha256": sha256_file(generator_marker),
        "repo_id": str(gen["repo_id"]),
        "revision": str(gen["revision"]),
        "pipeline_class": str(gen["pipeline_class"]),
        "input_text_field": str(gen["input_text_field"]),
        "prompt_template": str(gen["prompt_template"]),
        "negative_prompt": str(gen.get("negative_prompt", "")),
        "generation": gen["generation"],
        "output": gen.get("output", {}),
    }


def edit_meta_path(manifest_path: Path) -> Path:
    return manifest_path.with_suffix(manifest_path.suffix + ".meta.json")


def load_edit_cache(
    manifest_path: Path,
    cache_dir: Path,
    expected_fingerprint: dict[str, Any],
    queries: Sequence[dict[str, Any]],
) -> list[Path] | None:
    meta_path = edit_meta_path(manifest_path)
    if not manifest_path.is_file() or not meta_path.is_file() or not cache_dir.is_dir():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if meta != expected_fingerprint:
        print(f"Ignoring stale edited-image cache: {rel(manifest_path)}", flush=True)
        return None

    rows = load_jsonl(manifest_path)
    if len(rows) != len(queries):
        return None
    paths: list[Path] = []
    for qi, (row, query) in enumerate(zip(rows, queries)):
        if row.get("query_id") != query.get("query_id"):
            return None
        if row.get("image_id") != query.get("image_id"):
            return None
        edited_path = row.get("edited_image")
        if not isinstance(edited_path, str) or not edited_path.strip():
            return None
        path = resolve_path(edited_path)
        if not path.is_file():
            return None
        paths.append(path)
    print(f"Using edited-image cache: {rel(manifest_path)}", flush=True)
    return paths


def generator_dtype(name: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    key = str(name).lower()
    if key not in mapping:
        raise ValueError(f"Unsupported generator_dtype: {name!r}")
    return mapping[key]


def generate_one_image(pipe, image: Image.Image, prompt: str, negative_prompt: str, cfg: dict[str, Any], device: torch.device) -> Image.Image:
    generation = cfg["generator"]["generation"]
    seed = int(generation["seed"])
    kwargs = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "num_inference_steps": int(generation["num_inference_steps"]),
        "guidance_scale": float(generation["guidance_scale"]),
    }
    if device.type == "cuda":
        kwargs["generator"] = torch.Generator(device="cuda").manual_seed(seed)
    else:
        kwargs["generator"] = torch.Generator().manual_seed(seed)

    last_error: Exception | None = None
    for image_value in (image, [image]):
        try:
            result = pipe(image=image_value, **kwargs)
            break
        except TypeError as error:
            last_error = error
            continue
    else:
        raise RuntimeError(
            "Failed to call QwenImageEditPlusPipeline with supported image argument forms"
        ) from last_error

    images = getattr(result, "images", None)
    if not isinstance(images, list) or not images:
        raise RuntimeError("Qwen image editor returned no output images")
    output = images[0]
    if not isinstance(output, Image.Image):
        raise RuntimeError("Qwen image editor output is not a PIL image")
    return output.convert("RGB")


class PathDataset(Dataset):
    def __init__(self, paths: Sequence[Path], preprocess):
        self.paths = list(paths)
        self.preprocess = preprocess

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        with Image.open(self.paths[index]) as image:
            return self.preprocess(image.convert("RGB"))


class GalleryDataset(Dataset):
    def __init__(self, rows: Sequence[dict[str, Any]], preprocess):
        self.rows = list(rows)
        self.preprocess = preprocess

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        with Image.open(gallery_image_path(self.rows[i], i)) as image:
            return self.preprocess(image.convert("RGB"))


@torch.no_grad()
def clip_gallery_features(model, preprocess, rows, cache, runtime, device, cache_fingerprint):
    feature_dim = int(model.text_projection.shape[1])
    expected_shape = (len(rows), feature_dim)
    meta_path = cache.with_suffix(cache.suffix + ".meta.json")

    if cache.is_file() and meta_path.is_file():
        x = np.load(cache, mmap_mode="r")
        try:
            cached_fingerprint = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            cached_fingerprint = None
        if x.shape == expected_shape and cached_fingerprint == cache_fingerprint:
            print(f"Using gallery cache: {rel(cache)}", flush=True)
            return x, cache
        print(f"Ignoring stale/incompatible gallery cache: {rel(cache)}", flush=True)
    elif cache.is_file():
        print(f"Ignoring legacy gallery cache without fingerprint: {rel(cache)}", flush=True)

    loader = DataLoader(
        GalleryDataset(rows, preprocess),
        batch_size=runtime["clip_image_batch_size"],
        shuffle=False,
        num_workers=runtime["num_workers"],
        pin_memory=(device.type == "cuda"),
    )
    chunks = []
    for images in progress_bar(loader, desc="Encode gallery", total=len(loader), unit="batch"):
        x = model.encode_image(images.to(device)).float()
        x /= x.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        chunks.append(x.cpu().numpy())
    x = np.concatenate(chunks).astype(np.float32)
    if x.shape != expected_shape:
        raise ValueError(f"Encoded gallery has shape {x.shape}, expected {expected_shape}")
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, x)
    meta_path.write_text(json.dumps(cache_fingerprint, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return np.load(cache, mmap_mode="r"), cache


def clip_gallery_cache_fingerprint(checkpoint: Path, gallery_manifest: Path, model_name: str) -> dict[str, Any]:
    return {
        "schema": CLIP_GALLERY_CACHE_SCHEMA,
        "model_name": model_name,
        "checkpoint_sha256": sha256_file(checkpoint),
        "gallery_manifest_sha256": sha256_file(gallery_manifest),
    }


def query_feature_cache_fingerprint(config_path: Path, edited_manifest_path: Path, clip_checkpoint: Path, query_manifest: Path) -> dict[str, Any]:
    return {
        "schema": QUERY_FEATURE_CACHE_SCHEMA,
        "adapter_version": ADAPTER_VERSION,
        "config_sha256": sha256_file(config_path),
        "edited_manifest_sha256": sha256_file(edited_manifest_path),
        "clip_checkpoint_sha256": sha256_file(clip_checkpoint),
        "query_manifest_sha256": sha256_file(query_manifest),
    }


@torch.no_grad()
def clip_query_image_features(model, preprocess, edited_paths, cache, runtime, device, cache_fingerprint):
    feature_dim = int(model.text_projection.shape[1])
    expected_shape = (len(edited_paths), feature_dim)
    meta_path = cache.with_suffix(cache.suffix + ".meta.json")

    if cache.is_file() and meta_path.is_file():
        x = np.load(cache, mmap_mode="r")
        try:
            cached_fingerprint = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            cached_fingerprint = None
        if x.shape == expected_shape and cached_fingerprint == cache_fingerprint:
            print(f"Using edited-query feature cache: {rel(cache)}", flush=True)
            return x, cache
        print(f"Ignoring stale/incompatible edited-query feature cache: {rel(cache)}", flush=True)
    elif cache.is_file():
        print(f"Ignoring legacy edited-query feature cache without fingerprint: {rel(cache)}", flush=True)

    loader = DataLoader(
        PathDataset(edited_paths, preprocess),
        batch_size=runtime["clip_image_batch_size"],
        shuffle=False,
        num_workers=runtime["num_workers"],
        pin_memory=(device.type == "cuda"),
    )
    chunks = []
    for images in progress_bar(loader, desc="Encode edited queries", total=len(loader), unit="batch"):
        x = model.encode_image(images.to(device)).float()
        x /= x.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        chunks.append(x.cpu().numpy())
    x = np.concatenate(chunks).astype(np.float32)
    if x.shape != expected_shape:
        raise ValueError(f"Encoded edited queries have shape {x.shape}, expected {expected_shape}")
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, x)
    meta_path.write_text(json.dumps(cache_fingerprint, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return np.load(cache, mmap_mode="r"), cache


def generate_or_load_edited_queries(
    *,
    cfg: dict[str, Any],
    config_path: Path,
    gallery_manifest: Path,
    query_manifest: Path,
    generator_marker: Path,
    generator_dir: Path,
    gallery: Sequence[dict[str, Any]],
    queries: Sequence[dict[str, Any]],
    device: torch.device,
) -> tuple[list[Path], Path, Path]:
    cache_dir = resolve_path(str(cfg["cache"]["edited_queries_dir"]))
    manifest_path = resolve_path(str(cfg["cache"]["edited_queries_manifest"]))
    fingerprint = edit_fingerprint(
        cfg=cfg,
        config_path=config_path,
        gallery_manifest=gallery_manifest,
        query_manifest=query_manifest,
        generator_marker=generator_marker,
    )
    cached = load_edit_cache(manifest_path, cache_dir, fingerprint, queries)
    if cached is not None:
        return cached, manifest_path, edit_meta_path(manifest_path)

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HOME"] = str((ROOT / ".cache" / "huggingface").resolve())

    from diffusers import QwenImageEditPlusPipeline

    dtype = generator_dtype(str(cfg["runtime"]["generator_dtype"]))
    pipe = QwenImageEditPlusPipeline.from_pretrained(
        str(generator_dir),
        torch_dtype=dtype,
        local_files_only=True,
    )
    if hasattr(pipe, "set_progress_bar_config"):
        pipe.set_progress_bar_config(disable=True)
    if device.type == "cuda":
        pipe = pipe.to("cuda")
    elif device.type == "mps":
        pipe = pipe.to("mps")
    else:
        pipe = pipe.to("cpu")

    gallery_index = build_gallery_index(gallery)
    negative_prompt = str(cfg["generator"].get("negative_prompt", "")).strip()
    image_suffix = str(cfg["generator"].get("output", {}).get("image_format", "PNG")).lower()
    if image_suffix == "jpeg":
        image_suffix = "jpg"

    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    edited_paths: list[Path] = []
    with manifest_path.open("w", encoding="utf-8") as handle:
        for qi, query in enumerate(progress_bar(queries, desc="Generate edited queries", total=len(queries), unit="query")):
            gallery_row = gallery[gallery_index[query["image_id"]]]
            source_path = gallery_image_path(gallery_row, gallery_index[query["image_id"]])
            with Image.open(source_path) as image:
                source = image.convert("RGB")
            prompt = fixed_prompt(cfg, str(query[cfg["generator"]["input_text_field"]]))
            edited = generate_one_image(pipe, source, prompt, negative_prompt, cfg, device)
            out_path = cache_dir / f"{qi:05d}_{query['query_id']}.{image_suffix}"
            save_format = str(cfg["generator"].get("output", {}).get("image_format", "PNG")).upper()
            edited.save(out_path, format=save_format)
            row = {
                "query_id": query["query_id"],
                "image_id": query["image_id"],
                "edited_image": rel(out_path),
                "prompt": prompt,
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            rows.append(row)
            edited_paths.append(out_path)

    meta_path = edit_meta_path(manifest_path)
    meta_path.write_text(json.dumps(fingerprint, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    del pipe
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return edited_paths, manifest_path, meta_path


def main() -> None:
    tracker = PhaseTracker(METHOD_ID, total=6)

    with tracker.phase("Load config and manifests"):
        parser = argparse.ArgumentParser()
        parser.add_argument("--config", default=str(DEFAULT_CONFIG))
        args = parser.parse_args()

        config_path = resolve_path(args.config)
        cfg = load_yaml(config_path)
        gallery_manifest = resolve_path(str(cfg["data"]["gallery_manifest"]))
        query_manifest = resolve_path(str(cfg["data"]["query_manifest"]))
        gallery = load_jsonl(gallery_manifest)
        queries = load_jsonl(query_manifest)
        output_dir = resolve_path(str(cfg["output"]["dir"]))
        output_dir.mkdir(parents=True, exist_ok=True)
        device = device_from(str(cfg["runtime"].get("device", "auto")))
        tracker.log(
            f"gallery={len(gallery):,} queries={len(queries):,} device={device} "
            f"clip_image_batch={cfg['runtime']['clip_image_batch_size']}"
        )

    with tracker.phase("Validate prepared generator artifacts"):
        generator_dir, generator_marker, generator_marker_data = validate_generator_artifacts(cfg)
        tracker.log(f"generator_dir={rel(generator_dir)} files={len(generator_marker_data['files'])}")

    with tracker.phase("Generate or reuse synthetic edited query images"):
        edited_paths, edited_manifest_path, _ = generate_or_load_edited_queries(
            cfg=cfg,
            config_path=config_path,
            gallery_manifest=gallery_manifest,
            query_manifest=query_manifest,
            generator_marker=generator_marker,
            generator_dir=generator_dir,
            gallery=gallery,
            queries=queries,
            device=device,
        )
        tracker.log(f"edited_queries={len(edited_paths):,} manifest={rel(edited_manifest_path)}")

    with tracker.phase("Load CLIP retriever", f"{cfg['retriever']['name']} on {device}"):
        clip_checkpoint = resolve_path(str(cfg["retriever"]["checkpoint"]))
        if not clip_checkpoint.is_file():
            raise FileNotFoundError(
                f"Missing CLIP checkpoint: {rel(clip_checkpoint)}. Run download_checkpoint.py first."
            )
        expected_sha = str(cfg["retriever"]["checkpoint_sha256"])
        actual_sha = sha256_file(clip_checkpoint)
        if actual_sha != expected_sha:
            raise RuntimeError(
                f"CLIP checkpoint checksum mismatch: expected {expected_sha}, got {actual_sha}"
            )
        model, preprocess = clip.load(str(clip_checkpoint), device=device, jit=False)
        model.eval()
        if device.type != "cuda":
            model.float()

    with tracker.phase("Prepare CLIP features"):
        gallery_cache = resolve_path(str(cfg["cache"]["gallery_features"]))
        gallery_fingerprint = clip_gallery_cache_fingerprint(
            clip_checkpoint, gallery_manifest, str(cfg["retriever"]["name"])
        )
        gallery_features, gallery_cache_used = clip_gallery_features(
            model, preprocess, gallery, gallery_cache, cfg["runtime"], device, gallery_fingerprint
        )
        query_feature_cache = resolve_path(str(cfg["cache"]["edited_query_features"]))
        query_fingerprint = query_feature_cache_fingerprint(
            config_path, edited_manifest_path, clip_checkpoint, query_manifest
        )
        query_features, query_feature_cache_used = clip_query_image_features(
            model, preprocess, edited_paths, query_feature_cache, cfg["runtime"], device, query_fingerprint
        )
        tracker.log(
            f"gallery_features={gallery_features.shape} query_features={query_features.shape} "
            f"gallery_cache={rel(gallery_cache_used)} query_cache={rel(query_feature_cache_used)}"
        )

    with tracker.phase("Compute query-gallery score matrix"):
        scores_path = output_dir / "scores.npy"
        scores = np.lib.format.open_memmap(
            scores_path, "w+", dtype=np.float32, shape=(len(queries), len(gallery))
        )
        gallery_tensor = torch.from_numpy(np.asarray(gallery_features)).to(device)
        batch = int(cfg["runtime"]["clip_score_batch_size"])
        score_steps = (len(queries) + batch - 1) // batch
        for start in progress_bar(range(0, len(queries), batch), desc="Score queries", total=score_steps, unit="batch"):
            end = min(start + batch, len(queries))
            query = torch.from_numpy(np.asarray(query_features[start:end])).to(device)
            scores[start:end] = (query @ gallery_tensor.T).cpu().numpy()
        scores.flush()

    with tracker.phase("Write run metadata and outputs"):
        run = {
            "method": cfg["method"],
            "display_name": cfg["display_name"],
            "group": cfg["group"],
            "cpr_supervision": cfg["cpr_supervision"],
            "generator": cfg["generator"],
            "retriever": cfg["retriever"],
            "runtime": cfg["runtime"],
            "config": rel(config_path),
            "edited_queries_manifest": rel(edited_manifest_path),
            "edited_query_features": rel(query_feature_cache_used),
            "gallery_features": rel(gallery_cache_used),
            "num_queries": len(queries),
            "num_gallery": len(gallery),
            "scores": rel(scores_path),
            "higher_is_better": True,
        }
        run_path = output_dir / "run.json"
        run_path.write_text(json.dumps(run, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        tracker.log(f"scores={rel(scores_path)} run={rel(run_path)}")

    tracker.finish()


if __name__ == "__main__":
    main()
