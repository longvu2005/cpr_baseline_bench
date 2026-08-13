#!/usr/bin/env python3
"""P5 LinCIR: direct full-scene CPR adapter around official LinCIR inference."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

# Keep the pinned checkout immutable when imported.
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
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
from benchmark_data import ensure_gallery_layout  # noqa: E402

DEFAULT_CONFIG = Path(__file__).resolve().parent / "config.yaml"
METHOD_ID = "lincir"
ADAPTER_VERSION = "2026-08-13-v1-fullscene-official-large"
GALLERY_CACHE_SCHEMA = 1
QUERY_CACHE_SCHEMA = 1


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


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def meta_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".meta.json")


def is_generated_python_artifact(path: str) -> bool:
    normalized = path.strip().strip('"').replace("\\", "/")
    return normalized.lower().endswith((".pyc", ".pyo"))


def tracked_dirty(checkout: Path) -> str:
    output = subprocess.check_output(
        ["git", "-C", str(checkout), "status", "--porcelain", "--untracked-files=no"],
        text=True,
    )
    dirty: list[str] = []
    for line in output.splitlines():
        if not line.strip():
            continue
        if len(line) < 4:
            dirty.append(line)
            continue
        paths = line[3:].strip().split(" -> ")
        if paths and all(is_generated_python_artifact(x) for x in paths):
            continue
        dirty.append(line)
    return "\n".join(dirty).strip()


def validate_source(cfg: dict[str, Any]) -> Path:
    checkout = resolve_path(str(cfg["source"]["local_checkout"]))
    if not checkout.is_dir():
        raise FileNotFoundError(
            f"Missing pinned LinCIR source: {rel(checkout)}. Run checkpoint preparation first."
        )
    expected = str(cfg["source"]["commit"])
    actual = subprocess.check_output(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual != expected:
        raise RuntimeError(f"LinCIR source commit mismatch: expected {expected}, got {actual}")
    dirty = tracked_dirty(checkout)
    if dirty:
        raise RuntimeError(f"Pinned LinCIR source has tracked modifications:\n{dirty}")
    return checkout


def load_source_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import official source module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def torch_load_full(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def device_from(name: str) -> torch.device:
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("LinCIR config requests CUDA, but CUDA is unavailable")
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


def query_gallery_indices(queries: Sequence[dict[str, Any]], gallery_index: dict[Any, int]) -> np.ndarray:
    indices: list[int] = []
    for qi, query in enumerate(queries):
        image_id = query.get("image_id")
        if image_id not in gallery_index:
            raise ValueError(f"Query row {qi}: image_id {image_id!r} missing from gallery")
        indices.append(gallery_index[image_id])
    return np.asarray(indices, dtype=np.int64)


def validate_prepared_artifacts(cfg: dict[str, Any]) -> tuple[Path, Path, dict[str, Any]]:
    checkpoint = resolve_path(str(cfg["checkpoint"]["path"]))
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing LinCIR checkpoint: {rel(checkpoint)}")
    actual_ckpt = sha256_file(checkpoint)
    if actual_ckpt != str(cfg["checkpoint"]["sha256"]):
        raise RuntimeError("LinCIR checkpoint checksum mismatch")

    snapshot = resolve_path(str(cfg["backbone"]["local_snapshot"]))
    model_file = snapshot / str(cfg["backbone"]["model_file"])
    config_file = snapshot / "config.json"
    if not model_file.is_file() or not config_file.is_file():
        raise FileNotFoundError(f"Incomplete local CLIP snapshot: {rel(snapshot)}")
    actual_model = sha256_file(model_file)
    if actual_model != str(cfg["backbone"]["model_sha256"]):
        raise RuntimeError("Pinned CLIP model.safetensors checksum mismatch")

    marker_path = resolve_path(str(cfg["checkpoint"]["prepared_marker"]))
    marker = read_json(marker_path)
    if marker is None:
        raise FileNotFoundError(f"Missing LinCIR prepared marker: {rel(marker_path)}")
    if marker.get("source_commit") != str(cfg["source"]["commit"]):
        raise RuntimeError("LinCIR prepared marker source commit mismatch")
    return checkpoint, snapshot, marker


def build_processor(cfg: dict[str, Any]) -> CLIPImageProcessor:
    # Exact processor arguments from official LinCIR models.py build_text_encoder().
    size = int(cfg["backbone"]["image_size"])
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


class ImageRowsDataset(Dataset):
    def __init__(self, gallery: Sequence[dict[str, Any]], indices: np.ndarray | None, processor):
        self.gallery = gallery
        self.indices = indices
        self.processor = processor

    def __len__(self) -> int:
        return len(self.gallery) if self.indices is None else len(self.indices)

    def __getitem__(self, index: int) -> torch.Tensor:
        gi = index if self.indices is None else int(self.indices[index])
        with Image.open(gallery_image_path(self.gallery[gi], gi)) as image:
            pixel_values = self.processor(images=image.convert("RGB"), return_tensors="pt").pixel_values[0]
        return pixel_values


def load_feature_cache(path: Path, expected_meta: dict[str, Any], expected_shape: tuple[int, int], label: str) -> np.ndarray | None:
    if not path.is_file() or not meta_path(path).is_file():
        return None
    if read_json(meta_path(path)) != expected_meta:
        print(f"Ignoring stale {label} cache: {rel(path)}", flush=True)
        return None
    try:
        x = np.load(path, mmap_mode="r", allow_pickle=False)
    except Exception as error:
        print(f"Ignoring invalid {label} cache {rel(path)}: {error}", flush=True)
        return None
    if x.shape != expected_shape or x.dtype.kind != "f":
        print(f"Ignoring incompatible {label} cache: {rel(path)}", flush=True)
        return None
    sample = np.asarray(x[: min(8, len(x))])
    if not np.isfinite(sample).all():
        print(f"Ignoring non-finite {label} cache: {rel(path)}", flush=True)
        return None
    print(f"Using {label} cache: {rel(path)}", flush=True)
    return x


def gallery_fingerprint(cfg: dict[str, Any], config_path: Path, gallery_manifest: Path, checkpoint: Path, snapshot: Path) -> dict[str, Any]:
    return {
        "schema": GALLERY_CACHE_SCHEMA,
        "adapter_version": ADAPTER_VERSION,
        "config_sha256": sha256_file(config_path),
        "gallery_manifest_sha256": sha256_file(gallery_manifest),
        "lincir_checkpoint_sha256": sha256_file(checkpoint),
        "clip_model_sha256": sha256_file(snapshot / str(cfg["backbone"]["model_file"])),
        "source_commit": str(cfg["source"]["commit"]),
        "backbone_revision": str(cfg["backbone"]["hf_revision"]),
        "normalization": bool(cfg["composition"]["normalize_gallery"]),
    }


def query_fingerprint(cfg: dict[str, Any], config_path: Path, gallery_manifest: Path, query_manifest: Path, checkpoint: Path, snapshot: Path) -> dict[str, Any]:
    return {
        "schema": QUERY_CACHE_SCHEMA,
        "adapter_version": ADAPTER_VERSION,
        "config_sha256": sha256_file(config_path),
        "gallery_manifest_sha256": sha256_file(gallery_manifest),
        "query_manifest_sha256": sha256_file(query_manifest),
        "lincir_checkpoint_sha256": sha256_file(checkpoint),
        "clip_model_sha256": sha256_file(snapshot / str(cfg["backbone"]["model_file"])),
        "source_commit": str(cfg["source"]["commit"]),
        "prompt_template": str(cfg["composition"]["prompt_template"]),
        "text_field": str(cfg["composition"]["text_field"]),
        "phi_l2_normalize_input": bool(cfg["phi"]["l2_normalize_input"]),
        "normalize_query": bool(cfg["composition"]["normalize_query"]),
    }


def build_prompts(cfg: dict[str, Any], queries: Sequence[dict[str, Any]]) -> list[str]:
    template = str(cfg["composition"]["prompt_template"])
    placeholder = str(cfg["composition"]["placeholder"])
    field = str(cfg["composition"]["text_field"])
    if template.count("{modification}") != 1:
        raise ValueError("LinCIR prompt_template must contain exactly one {modification}")
    if template.count(placeholder) != 1:
        raise ValueError("LinCIR prompt_template must contain exactly one pseudo-token placeholder")
    prompts: list[str] = []
    for qi, query in enumerate(queries):
        value = query.get(field)
        if not isinstance(value, str) or not value.strip():
            raise KeyError(f"Query row {qi} has no usable {field!r}")
        modification = value.strip()
        if placeholder in modification:
            raise ValueError(f"Query row {qi} contains reserved LinCIR placeholder {placeholder!r}")
        prompts.append(template.format(modification=modification))
    return prompts


@torch.no_grad()
def encode_gallery(*, gallery, image_encoder, processor, cache_path: Path, cache_meta: dict[str, Any], cfg: dict[str, Any], device: torch.device, dtype: torch.dtype) -> np.ndarray:
    dim = int(cfg["backbone"]["projection_dim"])
    expected_shape = (len(gallery), dim)
    cached = load_feature_cache(cache_path, cache_meta, expected_shape, "LinCIR gallery-feature")
    if cached is not None:
        return cached
    loader = DataLoader(
        ImageRowsDataset(gallery, None, processor),
        batch_size=int(cfg["runtime"]["gallery_batch_size"]),
        shuffle=False,
        num_workers=int(cfg["runtime"]["num_workers"]),
        pin_memory=(device.type == "cuda"),
    )
    chunks: list[np.ndarray] = []
    for images in progress_bar(loader, desc="LinCIR encode gallery", total=len(loader), unit="batch"):
        feat = image_encoder(pixel_values=images.to(device, dtype=dtype, non_blocking=True)).image_embeds.float()
        if bool(cfg["composition"]["normalize_gallery"]):
            feat = F.normalize(feat, dim=-1)
        chunks.append(feat.cpu().numpy())
    array = np.concatenate(chunks, axis=0).astype(np.float32, copy=False)
    if array.shape != expected_shape or not np.isfinite(array).all():
        raise RuntimeError(f"Invalid LinCIR gallery features: {array.shape}")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, array)
    write_json(meta_path(cache_path), cache_meta)
    return np.load(cache_path, mmap_mode="r", allow_pickle=False)


@torch.no_grad()
def encode_queries(*, gallery, queries, query_indices, image_encoder, text_encoder, phi, pseudo_encoder, processor, cache_path: Path, cache_meta: dict[str, Any], cfg: dict[str, Any], device: torch.device, dtype: torch.dtype) -> np.ndarray:
    dim = int(cfg["backbone"]["projection_dim"])
    expected_shape = (len(queries), dim)
    cached = load_feature_cache(cache_path, cache_meta, expected_shape, "LinCIR query-feature")
    if cached is not None:
        return cached

    placeholder = str(cfg["composition"]["placeholder"])
    expected_placeholder_id = int(cfg["composition"]["placeholder_token_id"])
    token_test = clip.tokenize([placeholder], context_length=77)
    actual_placeholder_id = int(token_test[0, 1].item())
    if actual_placeholder_id != expected_placeholder_id:
        raise RuntimeError(
            f"OpenAI CLIP placeholder token mismatch: {placeholder!r} -> {actual_placeholder_id}, "
            f"expected {expected_placeholder_id}"
        )

    prompts = build_prompts(cfg, queries)
    loader = DataLoader(
        ImageRowsDataset(gallery, query_indices, processor),
        batch_size=int(cfg["runtime"]["query_batch_size"]),
        shuffle=False,
        num_workers=int(cfg["runtime"]["num_workers"]),
        pin_memory=(device.type == "cuda"),
    )
    chunks: list[np.ndarray] = []
    offset = 0
    for images in progress_bar(loader, desc="LinCIR compose queries", total=len(loader), unit="batch"):
        n = int(images.shape[0])
        batch_prompts = prompts[offset : offset + n]
        try:
            tokens = clip.tokenize(batch_prompts, context_length=77, truncate=False).to(device)
        except RuntimeError as error:
            raise RuntimeError(
                f"LinCIR prompt exceeds CLIP context length near query rows {offset}:{offset+n}; "
                "the adapter preserves the official no-truncation evaluation behavior."
            ) from error
        placeholder_counts = (tokens == expected_placeholder_id).sum(dim=1)
        if not torch.equal(placeholder_counts.cpu(), torch.ones(n, dtype=placeholder_counts.dtype)):
            raise RuntimeError("Every LinCIR prompt must contain exactly one pseudo-token placeholder")

        image_features = image_encoder(
            pixel_values=images.to(device, dtype=dtype, non_blocking=True)
        ).image_embeds
        if bool(cfg["phi"]["l2_normalize_input"]):
            image_features = F.normalize(image_features, dim=-1)
        pseudo_tokens = phi(image_features)
        text_features = pseudo_encoder.encode_with_pseudo_tokens_HF(
            text_encoder, tokens, pseudo_tokens
        ).float()
        if bool(cfg["composition"]["normalize_query"]):
            text_features = F.normalize(text_features, dim=-1)
        chunks.append(text_features.cpu().numpy())
        offset += n

    array = np.concatenate(chunks, axis=0).astype(np.float32, copy=False)
    if array.shape != expected_shape or not np.isfinite(array).all():
        raise RuntimeError(f"Invalid LinCIR query features: {array.shape}")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, array)
    write_json(meta_path(cache_path), cache_meta)
    return np.load(cache_path, mmap_mode="r", allow_pickle=False)


def validate_scores(scores: np.ndarray, shape: tuple[int, int]) -> None:
    if scores.shape != shape:
        raise ValueError(f"scores shape {scores.shape}, expected {shape}")
    if not np.issubdtype(scores.dtype, np.floating):
        raise TypeError("scores.npy must be floating point")
    for start in range(0, shape[0], 256):
        if not np.isfinite(np.asarray(scores[start : start + 256])).all():
            raise ValueError("scores.npy contains NaN/Inf")


def main() -> None:
    tracker = PhaseTracker(METHOD_ID, total=6)

    with tracker.phase("Load config, manifests, and prepared artifacts"):
        parser = argparse.ArgumentParser()
        parser.add_argument("--config", default=str(DEFAULT_CONFIG))
        args = parser.parse_args()
        config_path = resolve_path(args.config)
        cfg = load_yaml(config_path)
        source = validate_source(cfg)
        checkpoint, snapshot, marker = validate_prepared_artifacts(cfg)

        gallery_manifest = resolve_path(str(cfg["data"]["gallery_manifest"]))
        query_manifest = resolve_path(str(cfg["data"]["query_manifest"]))
        gallery = load_jsonl(gallery_manifest)
        queries = load_jsonl(query_manifest)
        gallery_root = ensure_gallery_layout(ROOT, gallery_rows=gallery, repair=True)
        gallery_index = build_gallery_index(gallery)
        query_indices = query_gallery_indices(queries, gallery_index)
        device = device_from(str(cfg["runtime"]["device"]))
        dtype = torch.float16 if str(cfg["runtime"]["mixed_precision"]) == "fp16" and device.type == "cuda" else torch.float32
        tracker.log(
            f"gallery={len(gallery):,} queries={len(queries):,} device={device} dtype={dtype}"
        )
        tracker.log(f"gallery_root={gallery_root}")

    with tracker.phase("Load official LinCIR Phi and pinned CLIP ViT-L/14"):
        official_models = load_source_module(source / "models.py", "cpr_lincir_official_models")
        pseudo_encoder = load_source_module(
            source / "encode_with_pseudo_tokens.py", "cpr_lincir_official_pseudo_encoder"
        )

        image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            str(snapshot), local_files_only=True, torch_dtype=dtype
        ).to(device).eval().requires_grad_(False)
        text_encoder = CLIPTextModelWithProjection.from_pretrained(
            str(snapshot), local_files_only=True, torch_dtype=dtype
        ).to(device).eval().requires_grad_(False)
        processor = build_processor(cfg)

        phi_cfg = cfg["phi"]
        phi = official_models.Phi(
            input_dim=int(phi_cfg["input_dim"]),
            hidden_dim=int(phi_cfg["hidden_dim"]),
            output_dim=int(phi_cfg["output_dim"]),
            dropout=float(phi_cfg["dropout"]),
        )
        raw = torch_load_full(checkpoint)
        if not isinstance(raw, dict) or "Phi" not in raw:
            raise KeyError("Official LinCIR checkpoint missing 'Phi'")
        phi.load_state_dict(raw["Phi"], strict=True)
        phi = phi.to(device, dtype=dtype).eval().requires_grad_(False)
        del raw
        gc.collect()

        if int(image_encoder.config.projection_dim) != int(cfg["backbone"]["projection_dim"]):
            raise RuntimeError("Unexpected CLIP vision projection dimension")
        if int(text_encoder.config.projection_dim) != int(cfg["backbone"]["projection_dim"]):
            raise RuntimeError("Unexpected CLIP text projection dimension")
        if int(text_encoder.config.hidden_size) != int(cfg["backbone"]["text_hidden_size"]):
            raise RuntimeError("Unexpected CLIP text hidden size")

    with tracker.phase("Prepare normalized full-scene gallery features"):
        gallery_cache = resolve_path(str(cfg["cache"]["gallery_features"]))
        gallery_meta = gallery_fingerprint(
            cfg, config_path, gallery_manifest, checkpoint, snapshot
        )
        gallery_features = encode_gallery(
            gallery=gallery,
            image_encoder=image_encoder,
            processor=processor,
            cache_path=gallery_cache,
            cache_meta=gallery_meta,
            cfg=cfg,
            device=device,
            dtype=dtype,
        )
        tracker.log(f"gallery_features={gallery_features.shape} cache={rel(gallery_cache)}")

    with tracker.phase("Compose full-scene query features with LinCIR"):
        query_cache = resolve_path(str(cfg["cache"]["query_features"]))
        query_meta = query_fingerprint(
            cfg, config_path, gallery_manifest, query_manifest, checkpoint, snapshot
        )
        query_features = encode_queries(
            gallery=gallery,
            queries=queries,
            query_indices=query_indices,
            image_encoder=image_encoder,
            text_encoder=text_encoder,
            phi=phi,
            pseudo_encoder=pseudo_encoder,
            processor=processor,
            cache_path=query_cache,
            cache_meta=query_meta,
            cfg=cfg,
            device=device,
            dtype=dtype,
        )
        tracker.log(f"query_features={query_features.shape} cache={rel(query_cache)}")

    with tracker.phase("Compute complete query-gallery score matrix"):
        output_dir = resolve_path(str(cfg["output"]["dir"]))
        output_dir.mkdir(parents=True, exist_ok=True)
        scores_path = output_dir / "scores.npy"
        scores = np.lib.format.open_memmap(
            scores_path,
            mode="w+",
            dtype=np.float32,
            shape=(len(queries), len(gallery)),
        )
        gallery_tensor = torch.from_numpy(np.asarray(gallery_features)).to(device)
        batch = int(cfg["runtime"]["score_batch_size"])
        steps = (len(queries) + batch - 1) // batch
        for start in progress_bar(
            range(0, len(queries), batch),
            desc="LinCIR score queries",
            total=steps,
            unit="batch",
        ):
            end = min(start + batch, len(queries))
            query_tensor = torch.from_numpy(np.asarray(query_features[start:end])).to(device)
            scores[start:end] = (query_tensor @ gallery_tensor.T).cpu().numpy()
        scores.flush()
        validate_scores(scores, (len(queries), len(gallery)))

    with tracker.phase("Write run metadata"):
        output_dir = resolve_path(str(cfg["output"]["dir"]))
        run_path = output_dir / "run.json"
        payload = {
            "method": cfg["method"],
            "display_name": cfg["display_name"],
            "group": cfg["group"],
            "cpr_supervision": cfg["cpr_supervision"],
            "paper": cfg["paper"],
            "source": {
                "repository": cfg["source"]["repository"],
                "commit": cfg["source"]["commit"],
                "checkout": rel(source),
            },
            "checkpoint": {
                "path": rel(checkpoint),
                "sha256": sha256_file(checkpoint),
                "status": cfg["checkpoint"]["status"],
                "hf_repo_id": cfg["checkpoint"]["hf_repo_id"],
                "hf_revision": cfg["checkpoint"]["hf_revision"],
            },
            "backbone": cfg["backbone"],
            "phi": cfg["phi"],
            "composition": cfg["composition"],
            "runtime": cfg["runtime"],
            "gallery_features": rel(resolve_path(str(cfg["cache"]["gallery_features"]))),
            "query_features": rel(resolve_path(str(cfg["cache"]["query_features"]))),
            "config": rel(config_path),
            "num_queries": len(queries),
            "num_gallery": len(gallery),
            "scores": rel(output_dir / "scores.npy"),
            "higher_is_better": True,
            "prepared_marker_schema": marker.get("schema"),
            "adapter_version": ADAPTER_VERSION,
        }
        write_json(run_path, payload)
        tracker.log(f"scores={rel(output_dir / 'scores.npy')} run={rel(run_path)}")

    tracker.finish()


if __name__ == "__main__":
    main()
