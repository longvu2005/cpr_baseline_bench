#!/usr/bin/env python3
"""P4: official Pic2Word adapted to full-scene CPR queries without SetMatch."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import shutil
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset

os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark_progress import PhaseTracker, progress_bar  # noqa: E402

DEFAULT_CONFIG = Path(__file__).resolve().parent / "config.yaml"
METHOD_ID = "pic2word"
ADAPTER_VERSION = "2026-08-13-v1-full-scene-full-text"
GALLERY_CACHE_SCHEMA = 1
QUERY_CACHE_SCHEMA = 2


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
                raise TypeError(f"{path}:{lineno}: row must be a JSON object")
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
    return device


def is_generated_python_artifact(path: str) -> bool:
    normalized = path.strip().strip('"').replace("\\", "/")
    return normalized.lower().endswith((".pyc", ".pyo"))


def validate_pinned_source(cfg: dict[str, Any]) -> Path:
    if shutil.which("git") is None:
        raise RuntimeError("System tool 'git' is required to validate the pinned Pic2Word source checkout")
    source = cfg["source"]
    checkout = resolve_path(str(source["local_checkout"]))
    if not checkout.is_dir():
        raise FileNotFoundError(
            f"Missing pinned Pic2Word source: {rel(checkout)}. Run download_checkpoint.py first."
        )
    expected = str(source["commit"])
    actual = subprocess.check_output(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual != expected:
        raise RuntimeError(f"Pic2Word source commit mismatch: expected {expected}, got {actual}")
    status = subprocess.check_output(
        ["git", "-C", str(checkout), "status", "--porcelain", "--untracked-files=no"],
        text=True,
    )
    dirty: list[str] = []
    for line in status.splitlines():
        if not line.strip():
            continue
        if len(line) < 4:
            dirty.append(line)
            continue
        paths = line[3:].strip().split(" -> ")
        if paths and all(is_generated_python_artifact(x) for x in paths):
            continue
        dirty.append(line)
    if dirty:
        raise RuntimeError(
            f"Pinned Pic2Word source has tracked modifications: {rel(checkout)}\n"
            + "\n".join(dirty)
        )
    return checkout


def validate_prepared_artifacts(cfg: dict[str, Any]) -> tuple[Path, Path, dict[str, Any]]:
    marker = resolve_path(str(cfg["checkpoint"]["prepared_marker"]))
    checkpoint = resolve_path(str(cfg["checkpoint"]["path"]))
    clip_checkpoint = resolve_path(str(cfg["model"]["openai_clip_checkpoint"]))
    if not marker.is_file():
        raise FileNotFoundError(
            f"Missing prepared marker: {rel(marker)}. Run download_checkpoint.py first."
        )
    data = read_json(marker)
    if data is None:
        raise ValueError(f"Invalid prepared marker: {rel(marker)}")
    if not checkpoint.is_file() or not clip_checkpoint.is_file():
        raise FileNotFoundError("Prepared Pic2Word/CLIP artifact is missing")
    checkpoint_sha = sha256_file(checkpoint)
    marker_sha = data.get("checkpoint", {}).get("sha256")
    if checkpoint_sha != marker_sha:
        raise RuntimeError("Pic2Word checkpoint changed after preparation")
    clip_sha = sha256_file(clip_checkpoint)
    if clip_sha != str(cfg["model"]["openai_clip_sha256"]):
        raise RuntimeError("OpenAI CLIP ViT-L/14 checksum mismatch")
    return checkpoint, clip_checkpoint, data


def strip_module_prefix(state: dict[str, Any]) -> dict[str, Any]:
    if state and next(iter(state)).startswith("module."):
        return {key[len("module.") :]: value for key, value in state.items()}
    return state


def torch_load_full(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def ensure_pic2word_tokenizer_compat(tokenizer_module, model, placeholder: str) -> int:
    """Repair the pinned official tokenizer's special-token naming mismatch.

    The pinned Pic2Word clip.py expects <start_of_text>/<end_of_text>, but its
    bundled SimpleTokenizer exposes <|startoftext|>/<|endoftext|>. Add runtime
    aliases only; do not modify the pinned official checkout on disk.
    """
    tokenizer = getattr(tokenizer_module, "_tokenizer", None)
    encoder = getattr(tokenizer, "encoder", None)
    encode = getattr(tokenizer, "encode", None)
    if tokenizer is None or not isinstance(encoder, dict) or not callable(encode):
        raise RuntimeError(
            "Pinned Pic2Word tokenizer API is incompatible with this adapter"
        )

    aliases = {
        "<start_of_text>": "<|startoftext|>",
        "<end_of_text>": "<|endoftext|>",
    }
    for expected_name, canonical_name in aliases.items():
        if expected_name in encoder:
            continue
        if canonical_name not in encoder:
            raise RuntimeError(
                "Pinned Pic2Word tokenizer is missing both special-token names: "
                f"{expected_name!r} and {canonical_name!r}"
            )
        encoder[expected_name] = int(encoder[canonical_name])

    eot_token = int(encoder["<end_of_text>"])
    if eot_token != int(model.end_id):
        raise RuntimeError(
            "Pic2Word tokenizer/model EOT mismatch: "
            f"tokenizer={eot_token}, model.end_id={int(model.end_id)}"
        )

    placeholder_ids = tokenizer.encode(placeholder)
    if len(placeholder_ids) != 1:
        raise RuntimeError(
            f"Pic2Word placeholder {placeholder!r} must tokenize to exactly one token, "
            f"got {placeholder_ids}"
        )

    # Exercise the official tokenize() now, before the expensive gallery pass.
    probe = tokenizer_module.tokenize([placeholder])
    split_ind = int(placeholder_ids[0])
    if probe.shape[0] != 1 or probe.shape[1] < 3:
        raise RuntimeError(f"Unexpected Pic2Word token tensor shape: {tuple(probe.shape)}")
    if int(probe[0, 0]) != int(encoder["<start_of_text>"]):
        raise RuntimeError("Pic2Word tokenizer preflight produced an invalid SOT token")
    if int(probe[0, 1]) != split_ind or int(probe[0, 2]) != eot_token:
        raise RuntimeError("Pic2Word tokenizer preflight produced unexpected token ids")
    return split_ind


def load_official_model(cfg: dict[str, Any], source_root: Path, checkpoint: Path, clip_checkpoint: Path, device: torch.device):
    source_str = str(source_root)
    if source_str not in sys.path:
        sys.path.insert(0, source_str)

    from model import clip as pic2word_clip  # type: ignore
    from model.model import IM2TEXT  # type: ignore

    model, _preprocess_train, preprocess_val = pic2word_clip.load(
        str(clip_checkpoint), device=device, jit=False, is_train=False
    )
    model.float()
    mapper_cfg = cfg["model"]["im2text"]
    img2text = IM2TEXT(
        embed_dim=int(model.embed_dim),
        middle_dim=int(mapper_cfg["middle_dim"]),
        output_dim=int(model.token_embedding.weight.shape[1]),
        n_layer=int(mapper_cfg["n_layer"]),
        dropout=float(mapper_cfg["dropout"]),
    ).to(device)

    raw = torch_load_full(checkpoint)
    if not isinstance(raw, dict):
        raise TypeError("Pic2Word checkpoint must be a dict")
    if "state_dict_img2text" not in raw:
        raise KeyError("Pic2Word checkpoint is missing state_dict_img2text")

    mapper_raw = raw["state_dict_img2text"]
    if not isinstance(mapper_raw, dict):
        raise TypeError("Invalid Pic2Word state_dict_img2text")
    mapper = strip_module_prefix(mapper_raw)

    state_raw = raw.get("state_dict")
    if state_raw is not None:
        if not isinstance(state_raw, dict):
            raise TypeError("Invalid Pic2Word state_dict")
        state = strip_module_prefix(state_raw)
        model.load_state_dict(state, strict=True)
    else:
        print(
            "[info] mapper-only Pic2Word checkpoint: using verified OpenAI CLIP "
            "ViT-L/14 backbone weights",
            flush=True,
        )

    img2text.load_state_dict(mapper, strict=True)
    model.to(device).float().eval()
    img2text.to(device).float().eval()
    split_ind = ensure_pic2word_tokenizer_compat(
        pic2word_clip,
        model,
        str(cfg["composition"]["placeholder"]),
    )
    print(f"[ok] Pic2Word tokenizer preflight split_ind={split_ind}", flush=True)
    return model, img2text, preprocess_val, pic2word_clip


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


class GalleryDataset(Dataset):
    def __init__(self, rows: Sequence[dict[str, Any]], preprocess):
        self.rows = rows
        self.preprocess = preprocess

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        with Image.open(gallery_image_path(self.rows[index], index)) as image:
            return self.preprocess(image.convert("RGB"))


class QueryImageDataset(Dataset):
    def __init__(self, gallery: Sequence[dict[str, Any]], query_indices: np.ndarray, preprocess):
        self.gallery = gallery
        self.query_indices = query_indices
        self.preprocess = preprocess

    def __len__(self) -> int:
        return len(self.query_indices)

    def __getitem__(self, index: int):
        gi = int(self.query_indices[index])
        with Image.open(gallery_image_path(self.gallery[gi], gi)) as image:
            return self.preprocess(image.convert("RGB"))


def gallery_fingerprint(cfg: dict[str, Any], config_path: Path, gallery_manifest: Path, checkpoint: Path, clip_checkpoint: Path) -> dict[str, Any]:
    return {
        "schema": GALLERY_CACHE_SCHEMA,
        "adapter_version": ADAPTER_VERSION,
        "config_sha256": sha256_file(config_path),
        "gallery_manifest_sha256": sha256_file(gallery_manifest),
        "pic2word_checkpoint_sha256": sha256_file(checkpoint),
        "clip_checkpoint_sha256": sha256_file(clip_checkpoint),
        "source_commit": str(cfg["source"]["commit"]),
        "backbone": str(cfg["model"]["backbone"]),
    }


def query_fingerprint(cfg: dict[str, Any], config_path: Path, gallery_manifest: Path, query_manifest: Path, checkpoint: Path, clip_checkpoint: Path) -> dict[str, Any]:
    return {
        "schema": QUERY_CACHE_SCHEMA,
        "adapter_version": ADAPTER_VERSION,
        "config_sha256": sha256_file(config_path),
        "gallery_manifest_sha256": sha256_file(gallery_manifest),
        "query_manifest_sha256": sha256_file(query_manifest),
        "pic2word_checkpoint_sha256": sha256_file(checkpoint),
        "clip_checkpoint_sha256": sha256_file(clip_checkpoint),
        "source_commit": str(cfg["source"]["commit"]),
        "prompt_template": str(cfg["composition"]["prompt_template"]),
        "text_field": str(cfg["composition"]["text_field"]),
    }


def load_feature_cache(path: Path, expected_meta: dict[str, Any], expected_shape: tuple[int, int], label: str) -> np.ndarray | None:
    if not path.is_file() or not meta_path(path).is_file():
        return None
    if read_json(meta_path(path)) != expected_meta:
        print(f"Ignoring stale {label} cache: {rel(path)}", flush=True)
        return None
    try:
        features = np.load(path, mmap_mode="r", allow_pickle=False)
    except Exception as error:
        print(f"Ignoring invalid {label} cache {rel(path)}: {error}", flush=True)
        return None
    if features.shape != expected_shape or features.dtype.kind != "f":
        print(f"Ignoring incompatible {label} cache: {rel(path)}", flush=True)
        return None
    if not np.isfinite(np.asarray(features[: min(8, len(features))])).all():
        print(f"Ignoring non-finite {label} cache: {rel(path)}", flush=True)
        return None
    print(f"Using {label} cache: {rel(path)}", flush=True)
    return features


@torch.no_grad()
def encode_gallery(model, preprocess, gallery, cache_path: Path, cache_meta: dict[str, Any], cfg: dict[str, Any], device: torch.device) -> np.ndarray:
    feature_dim = int(model.text_projection.shape[1])
    expected_shape = (len(gallery), feature_dim)
    cached = load_feature_cache(cache_path, cache_meta, expected_shape, "gallery-feature")
    if cached is not None:
        return cached

    loader = DataLoader(
        GalleryDataset(gallery, preprocess),
        batch_size=int(cfg["runtime"]["gallery_batch_size"]),
        shuffle=False,
        num_workers=int(cfg["runtime"]["num_workers"]),
        pin_memory=(device.type == "cuda"),
    )
    chunks: list[np.ndarray] = []
    for images in progress_bar(loader, desc="Pic2Word encode gallery", total=len(loader), unit="batch"):
        features = model.encode_image(images.to(device, non_blocking=True)).float()
        features /= features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        chunks.append(features.cpu().numpy())
    array = np.concatenate(chunks, axis=0).astype(np.float32, copy=False)
    if array.shape != expected_shape or not np.isfinite(array).all():
        raise RuntimeError(f"Invalid gallery features: shape={array.shape}")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, array)
    write_json(meta_path(cache_path), cache_meta)
    return np.load(cache_path, mmap_mode="r", allow_pickle=False)


def build_prompts(cfg: dict[str, Any], queries: Sequence[dict[str, Any]]) -> list[str]:
    template = str(cfg["composition"]["prompt_template"])
    placeholder = str(cfg["composition"]["placeholder"])
    text_field = str(cfg["composition"]["text_field"])
    if template.count("{modification}") != 1:
        raise ValueError("prompt_template must contain exactly one {modification} slot")
    if template.count(placeholder) != 1:
        raise ValueError("prompt_template must contain exactly one Pic2Word placeholder")
    prompts: list[str] = []
    for qi, query in enumerate(queries):
        value = query.get(text_field)
        if not isinstance(value, str) or not value.strip():
            raise KeyError(f"Query row {qi} has no usable {text_field!r}")
        modification = value.strip()
        if placeholder in modification:
            raise ValueError(
                f"Query row {qi} contains reserved Pic2Word placeholder {placeholder!r}"
            )
        prompts.append(template.format(modification=modification))
    return prompts


@torch.no_grad()
def encode_queries(model, img2text, tokenizer_module, preprocess, gallery, queries, query_indices, cache_path: Path, cache_meta: dict[str, Any], cfg: dict[str, Any], device: torch.device) -> np.ndarray:
    feature_dim = int(model.text_projection.shape[1])
    expected_shape = (len(queries), feature_dim)
    cached = load_feature_cache(cache_path, cache_meta, expected_shape, "query-feature")
    if cached is not None:
        return cached

    prompts = build_prompts(cfg, queries)
    placeholder = str(cfg["composition"]["placeholder"])
    split_ind = ensure_pic2word_tokenizer_compat(tokenizer_module, model, placeholder)

    loader = DataLoader(
        QueryImageDataset(gallery, query_indices, preprocess),
        batch_size=int(cfg["runtime"]["query_batch_size"]),
        shuffle=False,
        num_workers=int(cfg["runtime"]["num_workers"]),
        pin_memory=(device.type == "cuda"),
    )
    chunks: list[np.ndarray] = []
    offset = 0
    for images in progress_bar(loader, desc="Pic2Word compose queries", total=len(loader), unit="batch"):
        batch_size = int(images.shape[0])
        batch_prompts = prompts[offset : offset + batch_size]
        text_tokens = tokenizer_module.tokenize(batch_prompts).to(device, non_blocking=True)
        # The placeholder is intentionally early in the fixed template; truncation therefore cannot remove it.
        if not (text_tokens == split_ind).sum(dim=1).eq(1).all():
            raise RuntimeError("Pic2Word placeholder must occur exactly once after tokenization")
        image_features = model.encode_image(images.to(device, non_blocking=True)).float()
        pseudo_words = img2text(image_features)
        composed = model.encode_text_img_retrieval(
            text_tokens, pseudo_words, split_ind=split_ind, repeat=False
        ).float()
        composed /= composed.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        chunks.append(composed.cpu().numpy())
        offset += batch_size
    array = np.concatenate(chunks, axis=0).astype(np.float32, copy=False)
    if array.shape != expected_shape or not np.isfinite(array).all():
        raise RuntimeError(f"Invalid Pic2Word query features: shape={array.shape}")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, array)
    write_json(meta_path(cache_path), cache_meta)
    return np.load(cache_path, mmap_mode="r", allow_pickle=False)


def validate_scores(scores: np.ndarray, shape: tuple[int, int]) -> None:
    if scores.shape != shape:
        raise ValueError(f"scores.npy shape={scores.shape}, expected={shape}")
    if scores.dtype.kind != "f":
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
        if str(cfg.get("method")) != METHOD_ID:
            raise ValueError(f"config method must be {METHOD_ID!r}")
        source_root = validate_pinned_source(cfg)
        checkpoint, clip_checkpoint, prepared = validate_prepared_artifacts(cfg)
        gallery_manifest = resolve_path(str(cfg["data"]["gallery_manifest"]))
        query_manifest = resolve_path(str(cfg["data"]["query_manifest"]))
        gallery = load_jsonl(gallery_manifest)
        queries = load_jsonl(query_manifest)
        gallery_index = build_gallery_index(gallery)
        query_indices: list[int] = []
        for qi, query in enumerate(queries):
            image_id = query.get("image_id")
            if image_id not in gallery_index:
                raise ValueError(f"Query row {qi}: image_id {image_id!r} missing from gallery")
            query_indices.append(gallery_index[image_id])
        query_indices_np = np.asarray(query_indices, dtype=np.int64)
        device = device_from(str(cfg["runtime"]["device"]))
        tracker.log(f"gallery={len(gallery):,} queries={len(queries):,} device={device}")

    with tracker.phase("Load official Pic2Word ViT-L/14 model"):
        model, img2text, preprocess, tokenizer_module = load_official_model(
            cfg, source_root, checkpoint, clip_checkpoint, device
        )
        tracker.log(
            f"checkpoint_sha256={prepared['checkpoint']['sha256']} "
            f"backbone={cfg['model']['backbone']}"
        )

    with tracker.phase("Prepare full-gallery CLIP image features"):
        gallery_cache = resolve_path(str(cfg["cache"]["gallery_features"]))
        gmeta = gallery_fingerprint(
            cfg, config_path, gallery_manifest, checkpoint, clip_checkpoint
        )
        gallery_features = encode_gallery(
            model, preprocess, gallery, gallery_cache, gmeta, cfg, device
        )
        tracker.log(f"gallery_features={gallery_features.shape} cache={rel(gallery_cache)}")

    with tracker.phase("Compose full-scene Pic2Word query features"):
        query_cache = resolve_path(str(cfg["cache"]["query_features"]))
        qmeta = query_fingerprint(
            cfg,
            config_path,
            gallery_manifest,
            query_manifest,
            checkpoint,
            clip_checkpoint,
        )
        query_features = encode_queries(
            model,
            img2text,
            tokenizer_module,
            preprocess,
            gallery,
            queries,
            query_indices_np,
            query_cache,
            qmeta,
            cfg,
            device,
        )
        tracker.log(f"query_features={query_features.shape} cache={rel(query_cache)}")

    with tracker.phase("Score every query against full gallery"):
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
        for start in progress_bar(
            range(0, len(queries), batch),
            desc="Pic2Word score queries",
            total=(len(queries) + batch - 1) // batch,
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
        run = {
            "method": cfg["method"],
            "display_name": cfg["display_name"],
            "group": cfg["group"],
            "cpr_supervision": cfg["cpr_supervision"],
            "paper": cfg["paper"],
            "source": {
                "repository": cfg["source"]["repository"],
                "commit": cfg["source"]["commit"],
                "checkout": rel(source_root),
            },
            "checkpoint": {
                "path": rel(checkpoint),
                "sha256": sha256_file(checkpoint),
                "status": cfg["checkpoint"]["status"],
                "source_url": cfg["checkpoint"]["source_url"],
            },
            "model": cfg["model"],
            "adaptation": {
                "query_image": "full canonical query/reference scene",
                "query_text": f"full canonical queries.jsonl[{cfg['composition']['text_field']!r}]",
                "prompt_template": cfg["composition"]["prompt_template"],
                "setmatch": False,
                "person_detection": False,
                "target_box_used": False,
                "target_identity_label_used": False,
                "evaluation_labels_used": False,
                "query_image_removed_inside_method": False,
            },
            "cache": {
                "gallery_features": rel(gallery_cache),
                "query_features": rel(query_cache),
            },
            "config": rel(config_path),
            "num_queries": len(queries),
            "num_gallery": len(gallery),
            "scores": rel(output_dir / "scores.npy"),
            "higher_is_better": True,
        }
        write_json(run_path, run)
        tracker.log(f"scores={rel(output_dir / 'scores.npy')} run={rel(run_path)}")

    tracker.finish()


if __name__ == "__main__":
    main()
