#!/usr/bin/env python3
"""S7: frozen Qwen2.5-VL query rewrite followed by OpenAI CLIP ViT-L/14 retrieval.

For each canonical query, Qwen sees the complete query/reference scene plus the
canonical query text and emits one target-image caption using one fixed prompt.
CLIP then embeds that rewritten caption and every full gallery scene. No CPR
labels, target_ids, positives, boxes, or case annotations are consumed here.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import re
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
METHOD_ID = "qwen25vl_rewrite_clip"
ADAPTER_VERSION = "2026-08-13-v1-fixed-prompt-offline"
REWRITE_CACHE_SCHEMA = 1
TEXT_FEATURE_CACHE_SCHEMA = 1
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


def canonical_json_hash(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


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


def validate_qwen_artifacts(cfg: dict[str, Any]) -> tuple[Path, Path, dict[str, Any]]:
    mllm = cfg["mllm"]
    checkpoint_dir = resolve_path(str(mllm["checkpoint_dir"]))
    marker = resolve_path(str(mllm["prepared_marker"]))
    if not marker.is_file():
        raise FileNotFoundError(
            f"Missing Qwen prepared marker: {rel(marker)}. Run download_checkpoint.py first."
        )
    try:
        data = json.loads(marker.read_text(encoding="utf-8"))
    except Exception as error:
        raise RuntimeError(f"Invalid Qwen prepared marker: {rel(marker)}") from error
    if data.get("repo_id") != str(mllm["repo_id"]):
        raise RuntimeError("Qwen marker repo_id does not match config")
    if data.get("revision") != str(mllm["revision"]):
        raise RuntimeError("Qwen marker revision does not match config")
    files = data.get("files")
    if not isinstance(files, list) or not files:
        raise RuntimeError("Qwen marker has no file inventory")
    for item in files:
        if not isinstance(item, dict):
            raise RuntimeError("Malformed Qwen marker file entry")
        path = checkpoint_dir / str(item.get("path", ""))
        size = item.get("size")
        if not path.is_file() or not isinstance(size, int) or path.stat().st_size != size:
            raise RuntimeError(
                f"Qwen snapshot is incomplete/stale at {rel(path)}; rerun checkpoint preparation"
            )
    return checkpoint_dir, marker, data


def fixed_prompt(cfg: dict[str, Any], modification: str) -> str:
    template = str(cfg["mllm"]["prompt_template"])
    if template.count("{modification}") != 1:
        raise ValueError("mllm.prompt_template must contain exactly one {modification} placeholder")
    text = str(modification or "").strip()
    if not text:
        raise ValueError("Canonical query text is empty")
    return template.replace("{modification}", text)


def rewrite_fingerprint(
    *,
    cfg: dict[str, Any],
    config_path: Path,
    gallery_manifest: Path,
    query_manifest: Path,
    qwen_marker: Path,
) -> dict[str, Any]:
    mllm = cfg["mllm"]
    return {
        "schema": REWRITE_CACHE_SCHEMA,
        "adapter_version": ADAPTER_VERSION,
        "config_sha256": sha256_file(config_path),
        "gallery_manifest_sha256": sha256_file(gallery_manifest),
        "query_manifest_sha256": sha256_file(query_manifest),
        "qwen_marker_sha256": sha256_file(qwen_marker),
        "repo_id": str(mllm["repo_id"]),
        "revision": str(mllm["revision"]),
        "input_text_field": str(mllm["input_text_field"]),
        "prompt_template": str(mllm["prompt_template"]),
        "processor": mllm["processor"],
        "generation": mllm["generation"],
    }


def rewrite_meta_path(cache_path: Path) -> Path:
    return cache_path.with_suffix(cache_path.suffix + ".meta.json")


def load_rewrite_cache(
    cache_path: Path,
    expected_fingerprint: dict[str, Any],
    queries: Sequence[dict[str, Any]],
) -> list[str] | None:
    meta_path = rewrite_meta_path(cache_path)
    if not cache_path.is_file() or not meta_path.is_file():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if meta != expected_fingerprint:
        print(f"Ignoring stale rewrite cache: {rel(cache_path)}", flush=True)
        return None

    rows = load_jsonl(cache_path)
    if len(rows) != len(queries):
        return None
    captions: list[str] = []
    for qi, (row, query) in enumerate(zip(rows, queries)):
        if row.get("query_id") != query.get("query_id"):
            return None
        if row.get("image_id") != query.get("image_id"):
            return None
        caption = row.get("rewritten_text")
        if not isinstance(caption, str) or not caption.strip():
            return None
        captions.append(caption.strip())
    print(f"Using rewrite cache: {rel(cache_path)}", flush=True)
    return captions


def normalize_caption(text: str) -> str:
    value = re.sub(r"\s+", " ", text).strip()
    if not value:
        raise RuntimeError("Qwen produced an empty target description")
    return value


def qwen_dtype(name: str) -> torch.dtype:
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
        raise ValueError(f"Unsupported qwen_dtype: {name!r}")
    return mapping[key]


@torch.inference_mode()
def generate_rewritten_queries(
    *,
    cfg: dict[str, Any],
    queries: Sequence[dict[str, Any]],
    gallery: Sequence[dict[str, Any]],
    query_indices: np.ndarray,
    checkpoint_dir: Path,
    device: torch.device,
    cache_path: Path,
    fingerprint: dict[str, Any],
) -> list[str]:
    if device.type != "cuda":
        raise RuntimeError(
            "Qwen2.5-VL-7B S7 is configured as a GPU baseline. Use CUDA for reproducible inference."
        )

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"

    try:
        from qwen_vl_utils import process_vision_info
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    except ImportError as error:
        raise RuntimeError(
            "Missing Qwen runtime dependency. Run S7 through run_baseline.py so requirements are installed."
        ) from error

    mllm = cfg["mllm"]
    processor_cfg = mllm["processor"]
    generation_cfg = mllm["generation"]
    processor = AutoProcessor.from_pretrained(
        str(checkpoint_dir),
        min_pixels=int(processor_cfg["min_pixels"]),
        max_pixels=int(processor_cfg["max_pixels"]),
        local_files_only=True,
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(checkpoint_dir),
        torch_dtype=qwen_dtype(cfg["runtime"]["qwen_dtype"]),
        attn_implementation=str(cfg["runtime"].get("qwen_attn_implementation", "sdpa")),
        local_files_only=True,
    ).to(device)
    model.eval()

    text_field = str(mllm["input_text_field"])
    captions: list[str] = []
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = cache_path.with_suffix(cache_path.suffix + ".part")
    temp_path.unlink(missing_ok=True)

    try:
        with temp_path.open("w", encoding="utf-8") as out:
            for qi in progress_bar(
                range(len(queries)),
                desc="Rewrite queries with Qwen2.5-VL",
                total=len(queries),
                unit="query",
            ):
                query = queries[qi]
                gi = int(query_indices[qi])
                image_path = gallery_image_path(gallery[gi], gi)
                prompt = fixed_prompt(cfg, str(query.get(text_field) or ""))
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image_path.as_uri()},
                            {"type": "text", "text": prompt},
                        ],
                    }
                ]
                rendered = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                image_inputs, video_inputs = process_vision_info(messages)
                inputs = processor(
                    text=[rendered],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                ).to(device)
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=int(generation_cfg["max_new_tokens"]),
                    do_sample=bool(generation_cfg["do_sample"]),
                    num_beams=int(generation_cfg["num_beams"]),
                    repetition_penalty=float(generation_cfg["repetition_penalty"]),
                    use_cache=True,
                )
                trimmed = generated_ids[:, inputs.input_ids.shape[1] :]
                decoded = processor.batch_decode(
                    trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )[0]
                caption = normalize_caption(decoded)
                captions.append(caption)
                out.write(
                    json.dumps(
                        {
                            "query_id": query.get("query_id"),
                            "image_id": query.get("image_id"),
                            "rewritten_text": caption,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        os.replace(temp_path, cache_path)
        rewrite_meta_path(cache_path).write_text(
            json.dumps(fingerprint, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise
    finally:
        del model
        del processor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return captions


class GalleryDataset(Dataset):
    def __init__(self, rows: Sequence[dict[str, Any]], preprocess) -> None:
        self.rows = rows
        self.preprocess = preprocess

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> torch.Tensor:
        path = gallery_image_path(self.rows[index], index)
        with Image.open(path) as image:
            return self.preprocess(image.convert("RGB"))


def build_clip_gallery_cache_fingerprint(
    checkpoint: Path, gallery_manifest: Path, model_name: str
) -> dict[str, Any]:
    # Keep byte-for-byte schema compatibility with methods/simple/01_clip_image
    # and 02_clip_text so all ViT-L/14 simple baselines can share this cache.
    return {
        "schema": CLIP_GALLERY_CACHE_SCHEMA,
        "model_name": model_name,
        "checkpoint_sha256": sha256_file(checkpoint),
        "gallery_manifest_sha256": sha256_file(gallery_manifest),
    }


@torch.inference_mode()
def gallery_features(
    *,
    model,
    preprocess,
    rows: Sequence[dict[str, Any]],
    cache: Path,
    runtime: dict[str, Any],
    device: torch.device,
    fingerprint: dict[str, Any],
) -> np.ndarray:
    feature_dim = int(model.text_projection.shape[1])
    expected_shape = (len(rows), feature_dim)
    meta_path = cache.with_suffix(cache.suffix + ".meta.json")
    if cache.is_file() and meta_path.is_file():
        data = np.load(cache, mmap_mode="r", allow_pickle=False)
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            meta = None
        if data.shape == expected_shape and meta == fingerprint:
            print(f"Using CLIP gallery cache: {rel(cache)}", flush=True)
            return data
        print(f"Ignoring stale/incompatible CLIP gallery cache: {rel(cache)}", flush=True)
    elif cache.is_file():
        print(f"Ignoring CLIP gallery cache without fingerprint: {rel(cache)}", flush=True)

    loader = DataLoader(
        GalleryDataset(rows, preprocess),
        batch_size=int(runtime["clip_image_batch_size"]),
        shuffle=False,
        num_workers=int(runtime["num_workers"]),
        pin_memory=(device.type == "cuda"),
    )
    chunks: list[np.ndarray] = []
    for images in progress_bar(
        loader, desc="Encode CLIP gallery", total=len(loader), unit="batch"
    ):
        features = model.encode_image(images.to(device)).float()
        features /= features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        chunks.append(features.cpu().numpy())
    output = np.concatenate(chunks).astype(np.float32)
    if output.shape != expected_shape or not np.isfinite(output).all():
        raise ValueError(f"Invalid CLIP gallery features: {output.shape}")
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, output)
    meta_path.write_text(
        json.dumps(fingerprint, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return np.load(cache, mmap_mode="r", allow_pickle=False)


def text_feature_fingerprint(
    *, cache_path: Path, rewrite_fingerprint_value: dict[str, Any], checkpoint: Path, model_name: str
) -> dict[str, Any]:
    return {
        "schema": TEXT_FEATURE_CACHE_SCHEMA,
        "adapter_version": ADAPTER_VERSION,
        "rewrite_cache_sha256": sha256_file(cache_path),
        "rewrite_fingerprint_sha256": canonical_json_hash(rewrite_fingerprint_value),
        "clip_model_name": model_name,
        "clip_checkpoint_sha256": sha256_file(checkpoint),
    }


@torch.inference_mode()
def rewritten_text_features(
    *,
    model,
    captions: Sequence[str],
    cache: Path,
    fingerprint: dict[str, Any],
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    feature_dim = int(model.text_projection.shape[1])
    expected_shape = (len(captions), feature_dim)
    meta_path = cache.with_suffix(cache.suffix + ".meta.json")
    if cache.is_file() and meta_path.is_file():
        data = np.load(cache, mmap_mode="r", allow_pickle=False)
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            meta = None
        if data.shape == expected_shape and meta == fingerprint:
            print(f"Using rewritten-text CLIP cache: {rel(cache)}", flush=True)
            return data
        print(f"Ignoring stale rewritten-text CLIP cache: {rel(cache)}", flush=True)

    chunks: list[np.ndarray] = []
    for start in progress_bar(
        range(0, len(captions), batch_size),
        desc="Encode rewritten text with CLIP",
        total=(len(captions) + batch_size - 1) // batch_size,
        unit="batch",
    ):
        tokens = clip.tokenize(captions[start : start + batch_size], truncate=True).to(device)
        features = model.encode_text(tokens).float()
        features /= features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        chunks.append(features.cpu().numpy())
    output = np.concatenate(chunks).astype(np.float32)
    if output.shape != expected_shape or not np.isfinite(output).all():
        raise ValueError(f"Invalid rewritten text features: {output.shape}")
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, output)
    meta_path.write_text(
        json.dumps(fingerprint, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return np.load(cache, mmap_mode="r", allow_pickle=False)


@torch.inference_mode()
def main() -> None:
    tracker = PhaseTracker(METHOD_ID, total=8)

    with tracker.phase("Load config and canonical manifests"):
        parser = argparse.ArgumentParser()
        parser.add_argument("--config", default=str(DEFAULT_CONFIG))
        args = parser.parse_args()
        config_path = resolve_path(args.config)
        cfg = load_yaml(config_path)
        if cfg.get("method") != METHOD_ID:
            raise ValueError(f"config method must be {METHOD_ID!r}")
        gallery_manifest = resolve_path(str(cfg["data"]["gallery_manifest"]))
        query_manifest = resolve_path(str(cfg["data"]["query_manifest"]))
        gallery = load_jsonl(gallery_manifest)
        queries = load_jsonl(query_manifest)
        gallery_index = build_gallery_index(gallery)
        q_indices = query_gallery_indices(queries, gallery_index)
        output_dir = resolve_path(str(cfg["output"]["dir"]))
        output_dir.mkdir(parents=True, exist_ok=True)
        device = device_from(str(cfg["runtime"].get("device", "cuda")))
        tracker.log(
            f"gallery={len(gallery):,} queries={len(queries):,} device={device}"
        )

    with tracker.phase("Validate prepared model artifacts"):
        qwen_dir, qwen_marker, qwen_marker_data = validate_qwen_artifacts(cfg)
        clip_checkpoint = resolve_path(str(cfg["retriever"]["checkpoint"]))
        if not clip_checkpoint.is_file():
            raise FileNotFoundError(
                f"Missing CLIP checkpoint: {rel(clip_checkpoint)}. Run download_checkpoint.py first."
            )
        expected_clip_hash = str(cfg["retriever"]["checkpoint_sha256"])
        actual_clip_hash = sha256_file(clip_checkpoint)
        if actual_clip_hash != expected_clip_hash:
            raise RuntimeError(
                f"CLIP checkpoint checksum mismatch: expected {expected_clip_hash}, got {actual_clip_hash}"
            )
        tracker.log(
            f"qwen={cfg['mllm']['repo_id']}@{str(cfg['mllm']['revision'])[:12]} "
            f"clip={cfg['retriever']['name']}"
        )

    with tracker.phase("Rewrite every query with frozen Qwen2.5-VL"):
        rewrite_cache = resolve_path(str(cfg["cache"]["rewritten_queries"]))
        rewrite_fp = rewrite_fingerprint(
            cfg=cfg,
            config_path=config_path,
            gallery_manifest=gallery_manifest,
            query_manifest=query_manifest,
            qwen_marker=qwen_marker,
        )
        captions = load_rewrite_cache(rewrite_cache, rewrite_fp, queries)
        if captions is None:
            captions = generate_rewritten_queries(
                cfg=cfg,
                queries=queries,
                gallery=gallery,
                query_indices=q_indices,
                checkpoint_dir=qwen_dir,
                device=device,
                cache_path=rewrite_cache,
                fingerprint=rewrite_fp,
            )
        tracker.log(f"rewritten_queries={len(captions):,} cache={rel(rewrite_cache)}")

    with tracker.phase("Load OpenAI CLIP retriever", str(cfg["retriever"]["name"])):
        clip_model, preprocess = clip.load(str(clip_checkpoint), device=device, jit=False)
        clip_model.eval()
        if device.type != "cuda":
            clip_model.float()

    with tracker.phase("Prepare global gallery CLIP features"):
        gallery_cache = resolve_path(str(cfg["cache"]["gallery_features"]))
        gallery_fp = build_clip_gallery_cache_fingerprint(
            clip_checkpoint, gallery_manifest, str(cfg["retriever"]["name"])
        )
        gfeat = gallery_features(
            model=clip_model,
            preprocess=preprocess,
            rows=gallery,
            cache=gallery_cache,
            runtime=cfg["runtime"],
            device=device,
            fingerprint=gallery_fp,
        )
        tracker.log(f"gallery_features={gfeat.shape} cache={rel(gallery_cache)}")

    with tracker.phase("Encode rewritten target descriptions with CLIP"):
        text_cache = resolve_path(str(cfg["cache"]["rewritten_text_features"]))
        text_fp = text_feature_fingerprint(
            cache_path=rewrite_cache,
            rewrite_fingerprint_value=rewrite_fp,
            checkpoint=clip_checkpoint,
            model_name=str(cfg["retriever"]["name"]),
        )
        tfeat = rewritten_text_features(
            model=clip_model,
            captions=captions,
            cache=text_cache,
            fingerprint=text_fp,
            batch_size=int(cfg["runtime"]["clip_text_batch_size"]),
            device=device,
        )
        tracker.log(f"query_text_features={tfeat.shape} cache={rel(text_cache)}")

    with tracker.phase("Compute complete query-gallery score matrix"):
        scores_path = output_dir / "scores.npy"
        scores = np.lib.format.open_memmap(
            scores_path,
            mode="w+",
            dtype=np.float32,
            shape=(len(queries), len(gallery)),
        )
        gallery_tensor = torch.from_numpy(np.asarray(gfeat)).to(device)
        score_batch = int(cfg["runtime"]["clip_score_batch_size"])
        for start in progress_bar(
            range(0, len(queries), score_batch),
            desc="Score rewritten queries",
            total=(len(queries) + score_batch - 1) // score_batch,
            unit="batch",
        ):
            end = min(start + score_batch, len(queries))
            query_tensor = torch.from_numpy(np.asarray(tfeat[start:end])).to(device)
            block = (query_tensor @ gallery_tensor.T).float().cpu().numpy()
            if not np.isfinite(block).all():
                raise ValueError(f"Non-finite score block for query rows {start}:{end}")
            scores[start:end] = block
        scores.flush()

    with tracker.phase("Write reproducibility metadata"):
        run = {
            "method": METHOD_ID,
            "display_name": str(cfg["display_name"]),
            "group": str(cfg["group"]),
            "cpr_supervision": str(cfg["cpr_supervision"]),
            "adapter_version": ADAPTER_VERSION,
            "pipeline": "full query scene + canonical query text -> frozen Qwen rewrite -> CLIP global retrieval",
            "mllm": {
                "repo_id": cfg["mllm"]["repo_id"],
                "revision": cfg["mllm"]["revision"],
                "license": cfg["mllm"].get("license"),
                "checkpoint_dir": rel(qwen_dir),
                "prepared_marker": rel(qwen_marker),
                "prepared_marker_sha256": sha256_file(qwen_marker),
                "input_text_field": cfg["mllm"]["input_text_field"],
                "prompt_template": cfg["mllm"]["prompt_template"],
                "processor": cfg["mllm"]["processor"],
                "generation": cfg["mllm"]["generation"],
            },
            "retriever": cfg["retriever"],
            "runtime": cfg["runtime"],
            "data": {
                "gallery_manifest": rel(gallery_manifest),
                "query_manifest": rel(query_manifest),
            },
            "caches": {
                "rewritten_queries": rel(rewrite_cache),
                "rewritten_text_features": rel(text_cache),
                "gallery_features": rel(gallery_cache),
            },
            "config": rel(config_path),
            "num_queries": len(queries),
            "num_gallery": len(gallery),
            "scores": rel(scores_path),
            "higher_is_better": True,
            "benchmark_adaptation": {
                "single_prompt_for_all_cases": True,
                "query_image": "complete canonical scene",
                "query_text": str(cfg["mllm"]["input_text_field"]),
                "gallery_image": "complete canonical scene",
                "uses_target_ids": False,
                "uses_full_positive_ids": False,
                "uses_gt_boxes": False,
                "case_specific_logic": False,
            },
        }
        run_path = output_dir / "run.json"
        run_path.write_text(
            json.dumps(run, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        tracker.log(f"scores={rel(scores_path)} run={rel(run_path)}")

    tracker.finish()


if __name__ == "__main__":
    main()
