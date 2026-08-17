#!/usr/bin/env python3
"""P10 Imagine and Seek: paper-guided, training-free CPR reproduction."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.dont_write_bytecode = True

import clip
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import CLIPImageProcessor, CLIPTextModelWithProjection, CLIPVisionModelWithProjection

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark_progress import PhaseTracker, progress_bar  # noqa: E402

DEFAULT_CONFIG = Path(__file__).resolve().parent / "config.yaml"
METHOD_ID = "imagine_seek"
ADAPTER_VERSION = "2026-08-18-v1-paper-guided-fullscene-lincir"
PROXY_FEATURE_SCHEMA = 1
QUERY_COMPONENT_SCHEMA = 1


def load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
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
                raise TypeError(f"{path}:{lineno}: expected JSON object")
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


def stable_hash(payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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
    values: list[int] = []
    for qi, query in enumerate(queries):
        image_id = query.get("image_id")
        if image_id not in gallery_index:
            raise ValueError(f"Query row {qi}: image_id {image_id!r} missing from gallery")
        values.append(gallery_index[image_id])
    return np.asarray(values, dtype=np.int64)


def torch_dtype(name: str, device: torch.device) -> torch.dtype:
    if device.type != "cuda":
        return torch.float32
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    value = name.lower()
    if value not in mapping:
        raise ValueError(f"Unsupported dtype: {name}")
    return mapping[value]


def device_from(cfg: dict[str, Any]) -> torch.device:
    device = torch.device(str(cfg["runtime"]["device"]))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Imagine-and-Seek config requests CUDA, but CUDA is unavailable")
    return device


def run_lincir_if_needed(cfg: dict[str, Any]) -> tuple[Path, Path]:
    base = cfg["base_retriever"]
    scores = resolve_path(str(base["scores"]))
    gallery_features = resolve_path(str(base["gallery_features"]))
    if scores.is_file() and gallery_features.is_file():
        print(f"Using LinCIR dependency cache: {rel(scores)}", flush=True)
        return scores, gallery_features
    if not bool(base.get("auto_run", True)):
        raise FileNotFoundError(
            f"Missing LinCIR outputs: {rel(scores)} and/or {rel(gallery_features)}"
        )
    method_dir = resolve_path(str(base["method_dir"]))
    script = method_dir / "run.py"
    if not script.is_file():
        raise FileNotFoundError(f"Missing LinCIR runner: {rel(script)}")
    print("LinCIR outputs missing; running the repository P5 full-scene adapter first.", flush=True)
    subprocess.run([sys.executable, str(script)], cwd=str(ROOT), check=True)
    if not scores.is_file() or not gallery_features.is_file():
        raise FileNotFoundError("LinCIR run finished without required scores/gallery features")
    return scores, gallery_features


def ensure_proxy_manifest(cfg: dict[str, Any]) -> Path:
    manifest = resolve_path(str(cfg["proxy"]["manifest"]))
    script = Path(__file__).resolve().parent / "prepare_proxies.py"
    # Always invoke the validator/preparer. Cache fingerprints make this cheap when valid.
    subprocess.run([sys.executable, str(script), "--stage", "all"], cwd=str(ROOT), check=True)
    if not manifest.is_file():
        raise FileNotFoundError(f"Proxy preparation did not produce {rel(manifest)}")
    return manifest


def validate_proxy_manifest(
    cfg: dict[str, Any], queries: Sequence[dict[str, Any]], rows: Sequence[dict[str, Any]]
) -> None:
    if len(rows) != len(queries):
        raise ValueError(f"Proxy manifest has {len(rows)} rows, expected {len(queries)}")
    count = int(cfg["proxy"]["count_per_query"])
    for qi, row in enumerate(rows):
        if row.get("query_index") != qi or row.get("image_id") != queries[qi].get("image_id"):
            raise ValueError(f"Proxy manifest row {qi} is not aligned to canonical queries")
        for key in ("original_captions", "target_captions"):
            value = row.get(key)
            if not isinstance(value, list) or not value:
                raise ValueError(f"Proxy manifest row {qi} missing {key}")
        paths = row.get("proxy_paths")
        if not isinstance(paths, list) or len(paths) != count:
            raise ValueError(f"Proxy manifest row {qi}: expected {count} proxies")
        for value in paths:
            if not isinstance(value, str) or not resolve_path(value).is_file():
                raise FileNotFoundError(f"Proxy manifest row {qi}: missing proxy {value!r}")


def build_processor(cfg: dict[str, Any]) -> CLIPImageProcessor:
    # Match the existing P5 LinCIR CLIP ViT-L/14 image preprocessing exactly.
    size = 224
    return CLIPImageProcessor(
        crop_size={"height": size, "width": size},
        do_center_crop=True,
        do_convert_rgb=True,
        do_normalize=True,
        do_rescale=True,
        do_resize=True,
        image_mean=[0.48145466, 0.4578275, 0.40821073],
        image_std=[0.26862954, 0.26130258, 0.27577711],
        resample=3,
        size={"shortest_edge": size},
    )


class ProxyDataset(Dataset):
    def __init__(self, paths: Sequence[Path], processor: CLIPImageProcessor):
        self.paths = list(paths)
        self.processor = processor

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        with Image.open(self.paths[index]) as image:
            return self.processor(images=image.convert("RGB"), return_tensors="pt").pixel_values[0]


def load_array_cache(path: Path, meta_path: Path, expected_meta: dict[str, Any], expected_shape: tuple[int, ...]) -> np.ndarray | None:
    if not path.is_file() or read_json(meta_path) != expected_meta:
        return None
    try:
        array = np.load(path, mmap_mode="r", allow_pickle=False)
    except Exception:
        return None
    if array.shape != expected_shape or array.dtype.kind != "f":
        return None
    sample = np.asarray(array.reshape(-1, expected_shape[-1])[: min(16, int(np.prod(expected_shape[:-1])) or 1)])
    if not np.isfinite(sample).all():
        return None
    print(f"Using feature cache: {rel(path)}", flush=True)
    return array


@torch.no_grad()
def encode_proxy_images(
    *,
    cfg: dict[str, Any],
    proxy_rows: Sequence[dict[str, Any]],
    image_encoder: CLIPVisionModelWithProjection,
    processor: CLIPImageProcessor,
    device: torch.device,
    dtype: torch.dtype,
    clip_model_sha: str,
    proxy_manifest: Path,
) -> np.ndarray:
    count = int(cfg["proxy"]["count_per_query"])
    dim = int(cfg["base_retriever"]["projection_dim"])
    output = resolve_path(str(cfg["cache"]["proxy_features"]))
    meta_path = resolve_path(str(cfg["cache"]["proxy_features_meta"]))
    meta = {
        "schema": PROXY_FEATURE_SCHEMA,
        "adapter_version": ADAPTER_VERSION,
        "proxy_manifest_sha256": sha256_file(proxy_manifest),
        "clip_model_sha256": clip_model_sha,
        "count_per_query": count,
    }
    expected = (len(proxy_rows), count, dim)
    cached = load_array_cache(output, meta_path, meta, expected)
    if cached is not None:
        return cached

    flat_paths = [resolve_path(value) for row in proxy_rows for value in row["proxy_paths"]]
    loader = DataLoader(
        ProxyDataset(flat_paths, processor),
        batch_size=int(cfg["runtime"]["clip_image_batch_size"]),
        shuffle=False,
        num_workers=int(cfg["runtime"]["num_workers"]),
        pin_memory=(device.type == "cuda"),
    )
    chunks: list[np.ndarray] = []
    for images in progress_bar(loader, desc="IP-CIR encode proxies", total=len(loader), unit="batch"):
        features = image_encoder(pixel_values=images.to(device, dtype=dtype, non_blocking=True)).image_embeds.float()
        features = F.normalize(features, dim=-1)
        chunks.append(features.cpu().numpy())
    array = np.concatenate(chunks, axis=0).reshape(expected).astype(np.float32, copy=False)
    if not np.isfinite(array).all():
        raise RuntimeError("Non-finite proxy features")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, array)
    write_json(meta_path, meta)
    return np.load(output, mmap_mode="r", allow_pickle=False)


@torch.no_grad()
def encode_caption_sets(
    *,
    rows: Sequence[dict[str, Any]],
    key: str,
    text_encoder: CLIPTextModelWithProjection,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    dim = int(text_encoder.config.projection_dim)
    reduced: list[np.ndarray] = []
    for row in progress_bar(rows, desc=f"IP-CIR encode {key}", total=len(rows), unit="query"):
        captions = [str(x).strip() for x in row[key] if str(x).strip()]
        if not captions:
            raise ValueError(f"No usable {key}")
        chunks: list[torch.Tensor] = []
        for start in range(0, len(captions), batch_size):
            batch = captions[start : start + batch_size]
            tokens = clip.tokenize(batch, context_length=77, truncate=True).to(device)
            features = text_encoder(input_ids=tokens).text_embeds.float()
            chunks.append(F.normalize(features, dim=-1))
        all_features = torch.cat(chunks, dim=0)
        mean = F.normalize(all_features.mean(dim=0, keepdim=True), dim=-1)[0]
        reduced.append(mean.cpu().numpy())
    array = np.stack(reduced, axis=0).astype(np.float32, copy=False)
    if array.shape != (len(rows), dim) or not np.isfinite(array).all():
        raise RuntimeError(f"Invalid reduced text features for {key}: {array.shape}")
    return array


def query_component_meta(
    *,
    cfg: dict[str, Any],
    query_manifest: Path,
    proxy_manifest: Path,
    gallery_features_path: Path,
    clip_model_sha: str,
) -> dict[str, Any]:
    return {
        "schema": QUERY_COMPONENT_SCHEMA,
        "adapter_version": ADAPTER_VERSION,
        "query_manifest_sha256": sha256_file(query_manifest),
        "proxy_manifest_sha256": sha256_file(proxy_manifest),
        "gallery_features_sha256": sha256_file(gallery_features_path),
        "clip_model_sha256": clip_model_sha,
        "original_caption_reduce": cfg["representation"]["original_caption_reduce"],
        "target_caption_reduce": cfg["representation"]["target_caption_reduce"],
    }


@torch.no_grad()
def build_query_components(
    *,
    cfg: dict[str, Any],
    queries: Sequence[dict[str, Any]],
    query_indices: np.ndarray,
    proxy_rows: Sequence[dict[str, Any]],
    gallery_features: np.ndarray,
    text_encoder: CLIPTextModelWithProjection,
    device: torch.device,
    query_manifest: Path,
    proxy_manifest: Path,
    gallery_features_path: Path,
    clip_model_sha: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    path = resolve_path(str(cfg["cache"]["query_components"]))
    meta_path = resolve_path(str(cfg["cache"]["query_components_meta"]))
    meta = query_component_meta(
        cfg=cfg,
        query_manifest=query_manifest,
        proxy_manifest=proxy_manifest,
        gallery_features_path=gallery_features_path,
        clip_model_sha=clip_model_sha,
    )
    old = read_json(meta_path)
    if path.is_file() and old == meta:
        try:
            data = np.load(path, allow_pickle=False)
            fq, fo, ft, fs = data["fq"], data["fo"], data["ft"], data["fs"]
            expected = (len(queries), int(cfg["base_retriever"]["projection_dim"]))
            if all(x.shape == expected and np.isfinite(x).all() for x in (fq, fo, ft, fs)):
                print(f"Using query-component cache: {rel(path)}", flush=True)
                return fq, fo, ft, fs
        except Exception:
            pass

    fq = np.asarray(gallery_features[query_indices], dtype=np.float32)
    # P5 gallery features are already L2 normalized, but normalize again defensively.
    fq = fq / np.maximum(np.linalg.norm(fq, axis=1, keepdims=True), 1e-12)
    batch_size = int(cfg["runtime"]["clip_text_batch_size"])
    fo = encode_caption_sets(
        rows=proxy_rows,
        key="original_captions",
        text_encoder=text_encoder,
        device=device,
        batch_size=batch_size,
    )
    ft = encode_caption_sets(
        rows=proxy_rows,
        key="target_captions",
        text_encoder=text_encoder,
        device=device,
        batch_size=batch_size,
    )
    fs = (ft - fo).astype(np.float32, copy=False)
    if not np.isfinite(fs).all():
        raise RuntimeError("Non-finite semantic perturbation features")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, fq=fq, fo=fo, ft=ft, fs=fs)
    write_json(meta_path, meta)
    return fq, fo, ft, fs


def safe_denominator(values: np.ndarray, epsilon: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    sign = np.where(values < 0.0, -1.0, 1.0).astype(np.float32)
    return np.where(np.abs(values) < epsilon, sign * epsilon, values).astype(np.float32)


def construct_proxy_representation(
    *,
    cfg: dict[str, Any],
    fp: np.ndarray,
    fq: np.ndarray,
    fs: np.ndarray,
) -> np.ndarray:
    # Eq. (1), with only a numerical epsilon guard and optional final L2 normalization.
    epsilon = float(cfg["representation"]["denominator_epsilon"])
    fp_max = fp.max(axis=-1, keepdims=True)
    fq_max = safe_denominator(fq.max(axis=-1, keepdims=True), epsilon)[:, None, :]
    fs_max = safe_denominator(fs.max(axis=-1, keepdims=True), epsilon)[:, None, :]
    query_scale = fp_max / fq_max
    semantic_scale = fp_max / fs_max
    frp = (
        fp
        + float(cfg["representation"]["query_residual_weight"]) * query_scale * fq[:, None, :]
        + float(cfg["representation"]["semantic_residual_weight"]) * semantic_scale * fs[:, None, :]
    )
    if bool(cfg["representation"]["normalize_proxy_representation"]):
        norm = np.linalg.norm(frp, axis=-1, keepdims=True)
        frp = frp / np.maximum(norm, 1e-12)
    frp = frp.astype(np.float32, copy=False)
    if not np.isfinite(frp).all():
        raise RuntimeError("Eq. (1) produced non-finite proxy representations")
    return frp


def validate_scores(scores: np.ndarray, expected: tuple[int, int]) -> None:
    if scores.shape != expected:
        raise ValueError(f"scores shape={scores.shape}, expected={expected}")
    if not np.issubdtype(scores.dtype, np.floating):
        raise TypeError("scores.npy must be floating point")
    for start in range(0, expected[0], 128):
        if not np.isfinite(np.asarray(scores[start : start + 128])).all():
            raise ValueError("scores.npy contains NaN/Inf")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run P10 Imagine-and-Seek CPR reproduction")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()
    config_path = resolve_path(args.config)
    cfg = load_yaml(config_path)
    tracker = PhaseTracker(METHOD_ID, total=7)

    with tracker.phase("Load canonical manifests and P10 configuration"):
        gallery_manifest = resolve_path(str(cfg["data"]["gallery_manifest"]))
        query_manifest = resolve_path(str(cfg["data"]["query_manifest"]))
        gallery = load_jsonl(gallery_manifest)
        queries = load_jsonl(query_manifest)
        gallery_index = build_gallery_index(gallery)
        query_indices = query_gallery_indices(queries, gallery_index)
        device = device_from(cfg)
        dtype = torch_dtype(str(cfg["runtime"]["clip_dtype"]), device)
        tracker.log(f"gallery={len(gallery):,} queries={len(queries):,} device={device}")

    with tracker.phase("Prepare/load LinCIR baseline retrieval outputs"):
        base_scores_path, gallery_features_path = run_lincir_if_needed(cfg)
        base_scores = np.load(base_scores_path, mmap_mode="r", allow_pickle=False)
        gallery_features = np.load(gallery_features_path, mmap_mode="r", allow_pickle=False)
        dim = int(cfg["base_retriever"]["projection_dim"])
        if base_scores.shape != (len(queries), len(gallery)):
            raise ValueError(f"LinCIR scores shape mismatch: {base_scores.shape}")
        if gallery_features.shape != (len(gallery), dim):
            raise ValueError(f"LinCIR gallery feature shape mismatch: {gallery_features.shape}")

    with tracker.phase("Prepare/validate five imagined proxies per query"):
        proxy_manifest = ensure_proxy_manifest(cfg)
        proxy_rows = load_jsonl(proxy_manifest)
        validate_proxy_manifest(cfg, queries, proxy_rows)
        tracker.log(f"proxy_manifest={rel(proxy_manifest)}")

    with tracker.phase("Load the exact LinCIR CLIP ViT-L/14 feature space"):
        snapshot = resolve_path(str(cfg["base_retriever"]["clip_snapshot"]))
        model_file = snapshot / "model.safetensors"
        if not model_file.is_file() or not (snapshot / "config.json").is_file():
            raise FileNotFoundError(f"Incomplete LinCIR CLIP snapshot: {rel(snapshot)}")
        clip_model_sha = sha256_file(model_file)
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            str(snapshot), local_files_only=True, torch_dtype=dtype
        ).to(device).eval().requires_grad_(False)
        text_encoder = CLIPTextModelWithProjection.from_pretrained(
            str(snapshot), local_files_only=True, torch_dtype=dtype
        ).to(device).eval().requires_grad_(False)
        processor = build_processor(cfg)
        if int(image_encoder.config.projection_dim) != dim or int(text_encoder.config.projection_dim) != dim:
            raise RuntimeError("Unexpected CLIP projection dimension")

    with tracker.phase("Build IP-CIR proxy and semantic representations"):
        proxy_features = encode_proxy_images(
            cfg=cfg,
            proxy_rows=proxy_rows,
            image_encoder=image_encoder,
            processor=processor,
            device=device,
            dtype=dtype,
            clip_model_sha=clip_model_sha,
            proxy_manifest=proxy_manifest,
        )
        fq, fo, ft, fs = build_query_components(
            cfg=cfg,
            queries=queries,
            query_indices=query_indices,
            proxy_rows=proxy_rows,
            gallery_features=gallery_features,
            text_encoder=text_encoder,
            device=device,
            query_manifest=query_manifest,
            proxy_manifest=proxy_manifest,
            gallery_features_path=gallery_features_path,
            clip_model_sha=clip_model_sha,
        )
        frp = construct_proxy_representation(cfg=cfg, fp=np.asarray(proxy_features), fq=fq, fs=fs)
        del image_encoder, text_encoder
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    with tracker.phase("Compute Eq. (2) complete query-gallery scores"):
        output_dir = resolve_path(str(cfg["output"]["dir"]))
        output_dir.mkdir(parents=True, exist_ok=True)
        scores_path = output_dir / "scores.npy"
        scores = np.lib.format.open_memmap(
            scores_path,
            mode="w+",
            dtype=np.float32,
            shape=(len(queries), len(gallery)),
        )
        gallery_tensor = torch.from_numpy(np.asarray(gallery_features)).to(device=device, dtype=torch.float32)
        batch = int(cfg["runtime"]["score_batch_size"])
        lam = float(cfg["fusion"]["lambda_text"])
        if not (0.0 <= lam <= 1.0):
            raise ValueError("fusion.lambda_text must be in [0,1]")
        aggregation = str(cfg["proxy"]["aggregation"])
        if aggregation != "mean_similarity":
            raise ValueError(f"Unsupported proxy aggregation: {aggregation}")
        steps = (len(queries) + batch - 1) // batch
        for start in progress_bar(
            range(0, len(queries), batch),
            desc="IP-CIR fuse scores",
            total=steps,
            unit="batch",
        ):
            end = min(start + batch, len(queries))
            # [B, P, D] @ [D, G] -> [B, P, G], then average over five proxies.
            proxy_tensor = torch.from_numpy(frp[start:end]).to(device=device, dtype=torch.float32)
            sp = torch.matmul(proxy_tensor, gallery_tensor.T).mean(dim=1).cpu().numpy()
            st = np.asarray(base_scores[start:end], dtype=np.float32)
            sb = st * sp
            scores[start:end] = lam * st + (1.0 - lam) * sb
        scores.flush()
        validate_scores(scores, (len(queries), len(gallery)))

    with tracker.phase("Write reproducibility metadata"):
        marker_path = resolve_path(str(cfg["migc"]["prepared_marker"]))
        marker = read_json(marker_path) or {}
        payload = {
            "method": cfg["method"],
            "display_name": cfg["display_name"],
            "group": cfg["group"],
            "cpr_supervision": cfg["cpr_supervision"],
            "paper": cfg["paper"],
            "implementation_status": "REPRODUCED",
            "adapter_version": ADAPTER_VERSION,
            "cpr_adaptation": "direct_full_scene",
            "base_retriever": {
                "method": cfg["base_retriever"]["method"],
                "scores": rel(base_scores_path),
                "gallery_features": rel(gallery_features_path),
            },
            "proxy": {
                **cfg["proxy"],
                "manifest": rel(proxy_manifest),
                "reference_conditioning": cfg["migc"]["reference_conditioning"],
                "aggregation_note": "Paper specifies five proxies but not an explicit released multi-proxy aggregation rule; this reproduction averages proxy similarities.",
            },
            "representation": cfg["representation"],
            "fusion": cfg["fusion"],
            "paper_equations": {
                "eq1": "f_RP = f_p + max(f_p)/max(f_q)*f_q + max(f_p)/max(f_s)*f_s; f_s=f_t-f_o",
                "eq2": "S_b=S_t*S_p; S_f=lambda*S_t+(1-lambda)*S_b",
            },
            "generator": {
                "captioner": cfg["captioner"],
                "layout_llm": cfg["layout_llm"],
                "migc": {
                    "repository": cfg["migc"]["repository"],
                    "commit": cfg["migc"]["commit"],
                    "checkpoint_status": cfg["migc"]["checkpoint_status"],
                    "checkpoint_sha256": marker.get("migc_checkpoint", {}).get("sha256"),
                },
                "stable_diffusion": cfg["stable_diffusion"],
                "limitation": "Public reproduction uses MIGC text/layout control; exact IP-CIR ELITE reference-image conditioning is not claimed.",
            },
            "runtime": cfg["runtime"],
            "config": rel(config_path),
            "num_queries": len(queries),
            "num_gallery": len(gallery),
            "scores": rel(output_dir / "scores.npy"),
            "higher_is_better": True,
            "uses_cpr_labels_for_training_or_tuning": False,
            "query_image_removed_inside_method": False,
        }
        run_path = output_dir / "run.json"
        write_json(run_path, payload)
        tracker.log(f"scores={rel(scores_path)} run={rel(run_path)}")

    tracker.finish()


if __name__ == "__main__":
    main()
