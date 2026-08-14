#!/usr/bin/env python3
"""P7 BGE-VL-MLLM-S1 direct full-scene CPR inference adapter."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Sequence

# Inference must consume only artifacts prepared by download_checkpoint.py.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.dont_write_bytecode = True

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from transformers import AutoModel

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark_data import ensure_gallery_layout  # noqa: E402
from benchmark_progress import PhaseTracker, progress_bar  # noqa: E402

DEFAULT_CONFIG = Path(__file__).resolve().parent / "config.yaml"
METHOD_ID = "bge_vl_mllm_s1"
ADAPTER_VERSION = "2026-08-14-v2-auto-dispatch-offload"
GALLERY_CACHE_SCHEMA = 1
QUERY_CACHE_SCHEMA = 1


def resolve_path(value: str) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else ROOT / path).resolve()


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


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
                raise TypeError(f"{path}:{lineno}: expected a JSON object")
            rows.append(row)
    return rows


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def meta_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".meta.json")


def gallery_image_path(row: dict[str, Any], index: int) -> Path:
    value = row.get("path")
    if not isinstance(value, str) or not value.strip():
        raise KeyError(f"Gallery row {index} has no usable path")
    path = resolve_path(value)
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def build_gallery_index(gallery: Sequence[dict[str, Any]]) -> dict[Any, int]:
    result: dict[Any, int] = {}
    for index, row in enumerate(gallery):
        if "image_id" not in row:
            raise KeyError(f"Gallery row {index} missing image_id")
        image_id = row["image_id"]
        if image_id in result:
            raise ValueError(f"Duplicate gallery image_id: {image_id!r}")
        result[image_id] = index
    return result


def query_gallery_indices(queries: Sequence[dict[str, Any]], index: dict[Any, int]) -> list[int]:
    result: list[int] = []
    for qi, row in enumerate(queries):
        image_id = row.get("image_id")
        if image_id not in index:
            raise ValueError(f"Query row {qi}: image_id {image_id!r} missing from gallery")
        result.append(index[image_id])
    return result


def query_texts(queries: Sequence[dict[str, Any]], field: str) -> list[str]:
    result: list[str] = []
    for qi, row in enumerate(queries):
        value = row.get(field)
        if not isinstance(value, str) or not value.strip():
            raise KeyError(f"Query row {qi} has no usable {field!r}")
        result.append(value.strip())
    return result


def torch_dtype(name: str, device: torch.device) -> torch.dtype:
    if name == "float16":
        if device.type != "cuda":
            raise RuntimeError("BGE-VL-MLLM-S1 float16 inference requires CUDA")
        return torch.float16
    if name == "bfloat16":
        if device.type != "cuda" or not torch.cuda.is_bf16_supported():
            raise RuntimeError("Configured bfloat16 is not supported by this CUDA device")
        return torch.bfloat16
    raise ValueError(f"Unsupported torch_dtype: {name!r}")


def build_max_memory(runtime: dict[str, Any]) -> tuple[dict[int | str, int], dict[str, Any]]:
    """Reserve activation headroom while keeping the exact official FP16 weights."""
    placement = str(runtime.get("model_placement", "accelerate_auto"))
    if placement != "accelerate_auto":
        raise ValueError("runtime.model_placement must be 'accelerate_auto' when specified")

    headroom_gib = float(runtime.get("cuda_headroom_gib", 2.0))
    cpu_offload_gib = float(runtime.get("cpu_offload_max_gib", 8.0))
    if headroom_gib <= 0:
        raise ValueError("runtime.cuda_headroom_gib must be > 0 when specified")
    if cpu_offload_gib <= 0:
        raise ValueError("runtime.cpu_offload_max_gib must be > 0 when specified")

    headroom_bytes = int(headroom_gib * (1024**3))
    max_memory: dict[int | str, int] = {}
    gpu_info: list[dict[str, Any]] = []
    for index in range(torch.cuda.device_count()):
        with torch.cuda.device(index):
            free_bytes, total_bytes = torch.cuda.mem_get_info()
        usable_bytes = int(free_bytes) - headroom_bytes
        if usable_bytes <= 0:
            raise RuntimeError(
                f"CUDA:{index} has only {free_bytes / (1024**3):.2f} GiB free; "
                f"cannot reserve {headroom_gib:.2f} GiB inference headroom"
            )
        max_memory[index] = usable_bytes
        gpu_info.append(
            {
                "device": f"cuda:{index}",
                "free_gib_before_load": round(free_bytes / (1024**3), 3),
                "total_gib": round(total_bytes / (1024**3), 3),
                "weight_budget_gib": round(usable_bytes / (1024**3), 3),
            }
        )

    max_memory["cpu"] = int(cpu_offload_gib * (1024**3))
    return max_memory, {
        "policy": placement,
        "cuda_headroom_gib": headroom_gib,
        "cpu_offload_max_gib": cpu_offload_gib,
        "gpus": gpu_info,
    }


def summarize_device_map(model) -> dict[str, int]:
    mapping = getattr(model, "hf_device_map", None)
    if not isinstance(mapping, dict) or not mapping:
        raise RuntimeError("Accelerate did not attach the expected hf_device_map")

    summary: dict[str, int] = {}
    for target in mapping.values():
        label = f"cuda:{target}" if isinstance(target, int) else str(target)
        if label == "disk":
            raise RuntimeError(
                "BGE-VL placement fell back to disk offload; "
                "increase CPU/GPU memory instead of changing the model weights."
            )
        summary[label] = summary.get(label, 0) + 1
    return summary


def release_cuda_cache() -> None:
    gc.collect()
    for index in range(torch.cuda.device_count()):
        with torch.cuda.device(index):
            torch.cuda.empty_cache()


def validate_prepared(cfg: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    c = cfg["checkpoint"]
    snapshot = resolve_path(str(c["local_snapshot"]))
    marker_path = resolve_path(str(c["prepared_marker"]))
    marker = read_json(marker_path)
    if marker is None:
        raise FileNotFoundError(
            f"Missing prepared marker {rel(marker_path)}; run download_checkpoint.py first"
        )
    expected = {
        "schema": 1,
        "hf_repo_id": str(c["hf_repo_id"]),
        "hf_revision": str(c["hf_revision"]),
        "tensor_bytes": int(c["tensor_bytes"]),
        "source_commit": str(cfg["source"]["commit"]),
    }
    for key, value in expected.items():
        if marker.get(key) != value:
            raise RuntimeError(f"Prepared marker mismatch for {key}: {marker.get(key)!r} != {value!r}")
    checks = {
        "config.json": str(c["config_sha256"]),
        str(c["weight_index"]): str(c["weight_index_sha256"]),
        str(c["remote_code"]): str(c["remote_code_sha256"]),
    }
    for name, expected_hash in checks.items():
        path = snapshot / name
        if not path.is_file() or sha256_file(path) != expected_hash:
            raise RuntimeError(f"Missing or changed pinned snapshot file: {rel(path)}")
    shards = marker.get("weight_shards")
    if not isinstance(shards, list) or len(shards) != int(c["num_shards"]):
        raise RuntimeError("Prepared marker has an invalid BGE-VL weight-shard list")
    for name in shards:
        path = snapshot / str(name)
        if not path.is_file() or path.stat().st_size <= 0:
            raise FileNotFoundError(f"Missing or empty BGE-VL weight shard: {rel(path)}")
    return snapshot, marker


def cache_fingerprint(
    *, cfg: dict[str, Any], config_path: Path, gallery_manifest: Path,
    query_manifest: Path | None, role: str,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": GALLERY_CACHE_SCHEMA if role == "gallery" else QUERY_CACHE_SCHEMA,
        "adapter_version": ADAPTER_VERSION,
        "config_sha256": sha256_file(config_path),
        "gallery_manifest_sha256": sha256_file(gallery_manifest),
        "hf_revision": str(cfg["checkpoint"]["hf_revision"]),
        "role": role,
        "embedding_dim": int(cfg["model"]["embedding_dim"]),
        "normalization": True,
    }
    if query_manifest is not None:
        payload["query_manifest_sha256"] = sha256_file(query_manifest)
        payload["text_field"] = str(cfg["composition"]["text_field"])
        payload["task_instruction"] = str(cfg["composition"]["task_instruction"])
    return payload


def load_feature_cache(
    path: Path, expected_meta: dict[str, Any], expected_shape: tuple[int, int], label: str
) -> np.ndarray | None:
    if not path.is_file() or read_json(meta_path(path)) != expected_meta:
        return None
    try:
        array = np.load(path, mmap_mode="r", allow_pickle=False)
    except Exception as error:
        print(f"Ignoring invalid {label} cache: {error}", flush=True)
        return None
    if array.shape != expected_shape or array.dtype != np.float32:
        print(f"Ignoring incompatible {label} cache: {rel(path)}", flush=True)
        return None
    for start in range(0, len(array), 256):
        if not np.isfinite(np.asarray(array[start : start + 256])).all():
            print(f"Ignoring non-finite {label} cache: {rel(path)}", flush=True)
            return None
    print(f"Using {label} cache: {rel(path)}", flush=True)
    return array


def last_token_embeddings(model, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
    # Official remote code patches MistralForCausalLM.forward to return the final
    # hidden-state tensor directly. Disabling hidden-state history and KV cache is
    # mathematically equivalent to the official example and materially lowers VRAM.
    outputs = model(**inputs, output_hidden_states=False, use_cache=False)
    if not isinstance(outputs, torch.Tensor) or outputs.ndim != 3:
        raise TypeError(f"Unexpected BGE-VL output type/shape: {type(outputs)!r}")
    # Preserve the official operation order: normalize in model dtype, then cast
    # the compact cached representation to float32 for benchmark scoring.
    return F.normalize(outputs[:, -1, :], dim=-1).float()


@torch.inference_mode()
def encode_gallery(
    model,
    paths: Sequence[str],
    cache_path: Path,
    cache_meta: dict[str, Any],
    cfg: dict[str, Any],
) -> np.ndarray:
    dim = int(cfg["model"]["embedding_dim"])
    shape = (len(paths), dim)
    cached = load_feature_cache(cache_path, cache_meta, shape, "BGE-VL gallery-feature")
    if cached is not None:
        return cached
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    features = np.lib.format.open_memmap(cache_path, mode="w+", dtype=np.float32, shape=shape)
    batch_size = int(cfg["runtime"]["gallery_batch_size"])
    steps = (len(paths) + batch_size - 1) // batch_size
    for start in progress_bar(
        range(0, len(paths), batch_size),
        desc="BGE-VL encode gallery",
        total=steps,
        unit="batch",
    ):
        end = min(start + batch_size, len(paths))
        inputs = model.data_process(images=list(paths[start:end]), q_or_c=str(cfg["composition"]["candidate_role"]))
        embeddings = last_token_embeddings(model, inputs)
        if embeddings.shape != (end - start, dim):
            raise RuntimeError(f"Unexpected gallery embedding shape: {tuple(embeddings.shape)}")
        if not torch.isfinite(embeddings).all():
            raise RuntimeError(f"Non-finite gallery embedding near rows {start}:{end}")
        features[start:end] = embeddings.cpu().numpy()
    features.flush()
    write_json(meta_path(cache_path), cache_meta)
    del features
    return np.load(cache_path, mmap_mode="r", allow_pickle=False)


@torch.inference_mode()
def encode_queries(
    model, paths: Sequence[str], texts: Sequence[str], cache_path: Path,
    cache_meta: dict[str, Any], cfg: dict[str, Any],
) -> np.ndarray:
    dim = int(cfg["model"]["embedding_dim"])
    shape = (len(paths), dim)
    cached = load_feature_cache(cache_path, cache_meta, shape, "BGE-VL query-feature")
    if cached is not None:
        return cached
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    features = np.lib.format.open_memmap(cache_path, mode="w+", dtype=np.float32, shape=shape)
    batch_size = int(cfg["runtime"]["query_batch_size"])
    steps = (len(paths) + batch_size - 1) // batch_size
    for start in progress_bar(
        range(0, len(paths), batch_size),
        desc="BGE-VL encode queries",
        total=steps,
        unit="batch",
    ):
        end = min(start + batch_size, len(paths))
        inputs = model.data_process(
            images=list(paths[start:end]),
            text=list(texts[start:end]),
            q_or_c=str(cfg["composition"]["query_role"]),
            task_instruction=str(cfg["composition"]["task_instruction"]),
        )
        embeddings = last_token_embeddings(model, inputs)
        if embeddings.shape != (end - start, dim):
            raise RuntimeError(f"Unexpected query embedding shape: {tuple(embeddings.shape)}")
        if not torch.isfinite(embeddings).all():
            raise RuntimeError(f"Non-finite query embedding near rows {start}:{end}")
        features[start:end] = embeddings.cpu().numpy()
    features.flush()
    write_json(meta_path(cache_path), cache_meta)
    del features
    return np.load(cache_path, mmap_mode="r", allow_pickle=False)


def validate_scores(scores: np.ndarray, shape: tuple[int, int]) -> None:
    if scores.shape != shape or scores.dtype != np.float32:
        raise ValueError(f"Invalid scores.npy: shape={scores.shape}, dtype={scores.dtype}, expected={shape}/float32")
    for start in range(0, shape[0], 256):
        if not np.isfinite(np.asarray(scores[start : start + 256])).all():
            raise ValueError("scores.npy contains NaN/Inf")


def main() -> None:
    tracker = PhaseTracker(METHOD_ID, total=6)
    with tracker.phase("Load config, canonical manifests, and prepared snapshot"):
        parser = argparse.ArgumentParser()
        parser.add_argument("--config", default=str(DEFAULT_CONFIG))
        args = parser.parse_args()
        config_path = resolve_path(args.config)
        cfg = load_yaml(config_path)
        if cfg.get("method") != METHOD_ID:
            raise ValueError(f"config method must be {METHOD_ID!r}")
        if cfg["composition"].get("normalize_query") is not True:
            raise ValueError("Official BGE-VL inference requires normalized query embeddings")
        if cfg["composition"].get("normalize_gallery") is not True:
            raise ValueError("Official BGE-VL inference requires normalized gallery embeddings")
        if cfg["runtime"].get("feature_cache_dtype") != "float32":
            raise ValueError("BGE-VL feature_cache_dtype must be float32")
        for key in ("gallery_batch_size", "query_batch_size", "score_batch_size"):
            if int(cfg["runtime"].get(key, 0)) <= 0:
                raise ValueError(f"runtime.{key} must be a positive integer")
        snapshot, marker = validate_prepared(cfg)
        gallery_manifest = resolve_path(str(cfg["data"]["gallery_manifest"]))
        query_manifest = resolve_path(str(cfg["data"]["query_manifest"]))
        gallery = load_jsonl(gallery_manifest)
        queries = load_jsonl(query_manifest)
        gallery_root = ensure_gallery_layout(ROOT, gallery_rows=gallery, repair=True)
        gallery_index = build_gallery_index(gallery)
        query_indices = query_gallery_indices(queries, gallery_index)
        gallery_paths = [str(gallery_image_path(row, i)) for i, row in enumerate(gallery)]
        query_paths = [gallery_paths[i] for i in query_indices]
        texts = query_texts(queries, str(cfg["composition"]["text_field"]))
        device = torch.device(str(cfg["runtime"]["device"]))
        if device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("BGE-VL-MLLM-S1 is configured for CUDA, but CUDA is unavailable")
        dtype = torch_dtype(str(cfg["model"]["torch_dtype"]), device)
        max_memory, placement_info = build_max_memory(cfg["runtime"])
        tracker.log(f"gallery={len(gallery):,} queries={len(queries):,} device={device} dtype={dtype}")
        tracker.log(f"gallery_root={gallery_root}")

    with tracker.phase("Load pinned official BGE-VL-MLLM-S1 offline"):
        model = AutoModel.from_pretrained(
            str(snapshot),
            trust_remote_code=True,
            local_files_only=True,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            attn_implementation=str(cfg["model"]["attention_implementation"]),
            device_map="auto",
            max_memory=max_memory,
        )
        # Keep Accelerate's dispatch. Calling .to(cuda) here would collapse the
        # 7B model back onto GPU 0 and reproduce the original OOM.
        model = model.eval().requires_grad_(False)
        model.set_processor(str(snapshot))
        if model.__class__.__name__ != "LLaVANextForEmbedding":
            raise RuntimeError(f"Unexpected remote model class: {model.__class__.__name__}")
        if int(model.config.text_config.hidden_size) != int(cfg["model"]["embedding_dim"]):
            raise RuntimeError("Pinned BGE-VL text hidden size does not match embedding_dim")
        if getattr(model.processor.tokenizer, "padding_side", None) != "left":
            raise RuntimeError("Pinned BGE-VL tokenizer must use left padding for last-token pooling")
        placement_info["resolved_modules_per_device"] = summarize_device_map(model)
        placement_info["input_device"] = str(model.device)
        tracker.log(f"model={cfg['checkpoint']['hf_repo_id']} revision={cfg['checkpoint']['hf_revision'][:12]}")
        tracker.log(f"placement={json.dumps(placement_info, sort_keys=True)}")
        release_cuda_cache()

    with tracker.phase("Encode normalized full-scene gallery candidates"):
        gallery_cache = resolve_path(str(cfg["cache"]["gallery_features"]))
        gallery_meta = cache_fingerprint(
            cfg=cfg, config_path=config_path, gallery_manifest=gallery_manifest,
            query_manifest=None, role="gallery",
        )
        gallery_features = encode_gallery(model, gallery_paths, gallery_cache, gallery_meta, cfg)
        tracker.log(f"gallery_features={gallery_features.shape} cache={rel(gallery_cache)}")

    with tracker.phase("Encode full-scene image + instruction queries"):
        query_cache = resolve_path(str(cfg["cache"]["query_features"]))
        query_meta = cache_fingerprint(
            cfg=cfg, config_path=config_path, gallery_manifest=gallery_manifest,
            query_manifest=query_manifest, role="query",
        )
        query_features = encode_queries(model, query_paths, texts, query_cache, query_meta, cfg)
        tracker.log(f"query_features={query_features.shape} cache={rel(query_cache)}")

    # The encoder is no longer needed once both normalized feature caches exist.
    # Free all dispatched shards before allocating the score matrix on CUDA.
    del model
    release_cuda_cache()

    with tracker.phase("Compute complete query-gallery score matrix"):
        output_dir = resolve_path(str(cfg["output"]["dir"]))
        output_dir.mkdir(parents=True, exist_ok=True)
        scores_path = output_dir / "scores.npy"
        scores = np.lib.format.open_memmap(
            scores_path, mode="w+", dtype=np.float32, shape=(len(queries), len(gallery))
        )
        gallery_tensor = torch.from_numpy(np.asarray(gallery_features)).to(device)
        batch_size = int(cfg["runtime"]["score_batch_size"])
        steps = (len(queries) + batch_size - 1) // batch_size
        for start in progress_bar(
            range(0, len(queries), batch_size),
            desc="BGE-VL score queries",
            total=steps,
            unit="batch",
        ):
            end = min(start + batch_size, len(queries))
            query_tensor = torch.from_numpy(np.asarray(query_features[start:end])).to(device)
            scores[start:end] = (query_tensor @ gallery_tensor.T).cpu().numpy()
        scores.flush()
        validate_scores(scores, (len(queries), len(gallery)))

    with tracker.phase("Write reproducibility metadata"):
        run_path = output_dir / "run.json"
        payload = {
            "method": cfg["method"],
            "display_name": cfg["display_name"],
            "group": cfg["group"],
            "cpr_supervision": cfg["cpr_supervision"],
            "paper": cfg["paper"],
            "source": cfg["source"],
            "checkpoint": {
                "hf_repo_id": cfg["checkpoint"]["hf_repo_id"],
                "hf_revision": cfg["checkpoint"]["hf_revision"],
                "status": cfg["checkpoint"]["status"],
                "snapshot": rel(snapshot),
                "tensor_bytes": cfg["checkpoint"]["tensor_bytes"],
            },
            "model": cfg["model"],
            "composition": cfg["composition"],
            "runtime": cfg["runtime"],
            "model_placement": placement_info,
            "config": rel(config_path),
            "gallery_features": rel(gallery_cache),
            "query_features": rel(query_cache),
            "num_queries": len(queries),
            "num_gallery": len(gallery),
            "scores": rel(scores_path),
            "higher_is_better": True,
            "prepared_marker_schema": marker.get("schema"),
            "adapter_version": ADAPTER_VERSION,
            "uses_cpr_labels": False,
            "removes_query_image": False,
        }
        write_json(run_path, payload)
        tracker.log(f"scores={rel(scores_path)} run={rel(run_path)}")

    tracker.finish()


if __name__ == "__main__":
    main()
