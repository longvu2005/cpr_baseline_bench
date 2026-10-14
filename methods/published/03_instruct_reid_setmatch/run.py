#!/usr/bin/env python3
"""P3: official Instruct-ReID language-instruction inference + CPR SetMatch.

This is a thin benchmark adapter around the pinned official Instruct-ReID model.
Target people are localized from predicted person detections using a CLIP text
selector, never GT boxes or identities. The official language-instructed ReID
fusion feature is then used for person-level matching.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import yaml
from PIL import Image
from torchvision import transforms as TVT

os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark_progress import PhaseTracker, progress_bar
from cpr_person.adapter import load_person_context, manifest_identity, matching_metadata
from cpr_person.calibration import apply_instruct_calibration
from cpr_person.detection import crop_person as crop_box
from cpr_person.matching import score_feature_sets

DEFAULT_CONFIG = Path(__file__).resolve().parent / "config.yaml"
METHOD_ID = "instruct_reid_setmatch"
ADAPTER_VERSION = "2026-10-12-shared-person-v1"
DETECTION_CACHE_SCHEMA = 1
SELECTOR_CACHE_SCHEMA = 1
GALLERY_FEATURE_CACHE_SCHEMA = 1
QUERY_FEATURE_CACHE_SCHEMA = 1


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"Expected YAML mapping: {path}")
    return value


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


def require_file(path: Path, label: str) -> Path:
    if not path.is_file() or path.stat().st_size <= 0:
        raise FileNotFoundError(f"Missing {label}: {rel(path)}. Run download_checkpoint.py first.")
    return path


def generated_python_artifact(path: str) -> bool:
    normalized = path.strip().strip('"').replace("\\", "/")
    return normalized.lower().endswith((".pyc", ".pyo"))


def ensure_clean_pinned_source(cfg: dict[str, Any]) -> Path:
    source = cfg["source"]
    checkout = resolve_path(str(source["local_checkout"]))
    if not checkout.is_dir():
        raise FileNotFoundError(f"Missing official source checkout: {rel(checkout)}")
    actual = subprocess.check_output(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True
    ).strip()
    expected = str(source["commit"])
    if actual != expected:
        raise RuntimeError(f"Official source commit mismatch: expected {expected}, got {actual}")
    status = subprocess.check_output(
        ["git", "-C", str(checkout), "status", "--porcelain", "--untracked-files=no"],
        text=True,
    )
    dirty: list[str] = []
    for line in status.splitlines():
        if not line.strip():
            continue
        path = line[3:].strip() if len(line) >= 4 else line
        parts = path.split(" -> ")
        if parts and all(generated_python_artifact(x) for x in parts):
            continue
        dirty.append(line)
    if dirty:
        raise RuntimeError(
            f"Pinned official source has tracked local modifications: {rel(checkout)}\n"
            + "\n".join(dirty)
        )
    return checkout


def device_from(name: str) -> torch.device:
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if device.type != "cuda":
        raise RuntimeError(
            "Pinned official Instruct-ReID language inference moves tokenized instructions to CUDA. "
            "P3 therefore requires runtime.device=cuda."
        )
    return device


def image_path(row: dict[str, Any], index: int) -> Path:
    value = row.get("path")
    if not isinstance(value, str) or not value.strip():
        raise KeyError(f"Gallery row {index} has no usable path")
    path = resolve_path(value)
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def build_gallery_index(gallery: Sequence[dict[str, Any]]) -> dict[Any, int]:
    result: dict[Any, int] = {}
    for gi, row in enumerate(gallery):
        if "image_id" not in row:
            raise KeyError(f"Gallery row {gi} missing image_id")
        image_id = row["image_id"]
        if image_id in result:
            raise ValueError(f"Duplicate gallery image_id: {image_id!r}")
        result[image_id] = gi
    return result


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


def load_array_cache(
    path: Path, expected_meta: dict[str, Any], expected_shape: tuple[int, ...], label: str
) -> np.ndarray | None:
    if not path.is_file() or read_json(meta_path(path)) != expected_meta:
        return None
    try:
        array = np.load(path, mmap_mode="r", allow_pickle=False)
    except Exception as error:
        print(f"Ignoring invalid {label} cache {rel(path)}: {error}", flush=True)
        return None
    if array.shape != expected_shape or array.dtype.kind != "f":
        print(f"Ignoring incompatible {label} cache {rel(path)}", flush=True)
        return None
    print(f"Using {label} cache: {rel(path)}", flush=True)
    return array


def modification_instruction(query: dict[str, Any], subject: dict[str, Any]) -> str:
    # RELATIONAL adaptation without consuming the `case` annotation: the presence
    # of relation_text tells us to preserve the full multi-person instruction.
    if str(query.get("relation_text") or "").strip():
        full = str(query.get("text") or "").strip()
        if full:
            return full
    modify = str(subject.get("modify_text") or "").strip()
    if modify:
        return modify
    full = str(query.get("text") or "").strip()
    if full:
        return full
    raise ValueError(f"Query {query.get('query_id')} has no usable modification instruction")


def pre_caption(text: str, max_words: int) -> str:
    # Exact normalization semantics used by official preprocessor_attr.py.
    caption = re.sub(r"([,.'!?\"()*#:;~])", "", text.lower())
    caption = caption.replace("-", " ").replace("/", " ").replace("<person>", "person")
    caption = re.sub(r"\s{2,}", " ", caption).rstrip("\n").strip(" ")
    words = caption.split(" ") if caption else []
    if len(words) > max_words:
        caption = " ".join(words[:max_words])
    return caption


def make_instruct_args(cfg: dict[str, Any]) -> SimpleNamespace:
    values = dict(cfg["model"]["args"])
    values["test_task_type"] = str(cfg["model"]["test_task_type"])
    return SimpleNamespace(**values)


def checkpoint_num_classes(checkpoint: dict[str, Any]) -> int:
    state = checkpoint.get("state_dict")
    if not isinstance(state, dict):
        raise RuntimeError("Final Instruct-ReID checkpoint missing state_dict")
    for key, value in state.items():
        normalized = str(key).removeprefix("module.")
        if normalized == "classifier.weight" and hasattr(value, "shape") and len(value.shape) == 2:
            return int(value.shape[0])
    # Classifier shape has no effect in eval; official copy_state_dict can skip it.
    return 1


def load_official_instruct_model(
    *,
    cfg: dict[str, Any],
    checkout: Path,
    checkpoint_path: Path,
    bert_dir: Path,
    config_bert: Path,
    device: torch.device,
    move_to_device: bool = True,
):
    checkout_text = str(checkout)
    if checkout_text not in sys.path:
        sys.path.insert(0, checkout_text)

    import reid.models.pass_transformer_joint as joint  # type: ignore
    import reid.models.tokenization_bert as official_tokenization  # type: ignore
    from reid import models  # type: ignore
    from reid.utils.serialization import copy_state_dict  # type: ignore

    # The released source contains literal '<your project root> ...' placeholders.
    # Keep the checkout immutable and redirect only those constructor calls.
    #
    # Instruct-ReID's pinned slow BertTokenizer was written against Transformers
    # 4.31: it initializes ``self.vocab`` *after* ``PreTrainedTokenizer.__init__``.
    # Transformers 4.39 calls ``get_vocab()`` from the parent constructor, so on
    # Python 3.12 (where we use 4.39 to obtain binary tokenizers wheels) the
    # original ordering raises ``AttributeError: ... has no attribute 'vocab'``.
    # Seed the exact official vocabulary before entering the original constructor;
    # the original initializer then continues unchanged and re-loads the same file.
    original_tokenizer_init = joint.BertTokenizer.__init__
    original_tokenizer = joint.BertTokenizer.from_pretrained
    original_bert_model = joint.BertForMaskedLM.from_pretrained
    original_bert_config = joint.BertConfig.from_json_file

    def tokenizer_init_compat(self, vocab_file, *args, **kwargs):
        self.vocab = official_tokenization.load_vocab(vocab_file)
        return original_tokenizer_init(self, vocab_file, *args, **kwargs)

    def tokenizer_local(cls, _upstream_placeholder, *args, **kwargs):
        return original_tokenizer(str(bert_dir), *args, **kwargs)

    def bert_model_local(cls, _upstream_placeholder, *args, **kwargs):
        return original_bert_model(str(bert_dir), *args, **kwargs)

    def bert_config_local(cls, _upstream_placeholder, *args, **kwargs):
        return original_bert_config(str(config_bert), *args, **kwargs)

    joint.BertTokenizer.__init__ = tokenizer_init_compat
    joint.BertTokenizer.from_pretrained = classmethod(tokenizer_local)
    joint.BertForMaskedLM.from_pretrained = classmethod(bert_model_local)
    joint.BertConfig.from_json_file = classmethod(bert_config_local)

    try:
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        except TypeError:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("state_dict"), dict):
            raise RuntimeError("Final checkpoint is incompatible with official test_joint.py format")
        model = models.create(
            str(cfg["model"]["architecture"]),
            num_classes=checkpoint_num_classes(checkpoint),
            net_config=make_instruct_args(cfg),
        )
        copy_state_dict(checkpoint["state_dict"], model, strip="module.")
    finally:
        joint.BertTokenizer.__init__ = original_tokenizer_init
        joint.BertTokenizer.from_pretrained = original_tokenizer
        joint.BertForMaskedLM.from_pretrained = original_bert_model
        joint.BertConfig.from_json_file = original_bert_config

    model.eval()
    if move_to_device:
        model.to(device)
    return model


def instruct_transform(cfg: dict[str, Any]):
    size = [int(x) for x in cfg["model"]["image_size"]]
    mean = [float(x) for x in cfg["model"]["normalization_mean"]]
    std = [float(x) for x in cfg["model"]["normalization_std"]]
    return TVT.Compose([TVT.Resize(size), TVT.ToTensor(), TVT.Normalize(mean=mean, std=std)])


@torch.no_grad()
def run_instruct_batch(model, images: list[torch.Tensor], instructions: list[str], device: torch.device) -> np.ndarray:
    if not images:
        return np.empty((0, 0), dtype=np.float32)
    batch = torch.stack(images).to(device, non_blocking=True)
    output = model(batch, instructions)
    if not isinstance(output, (tuple, list)) or len(output) < 3:
        raise RuntimeError("Official Instruct-ReID model returned an unexpected inference output")
    feature = output[2]
    if not isinstance(feature, torch.Tensor) or feature.ndim != 2:
        raise RuntimeError("Official Instruct-ReID fusion feature is not a 2-D tensor")
    return feature.float().cpu().numpy().astype(np.float32, copy=False)


def gallery_feature_fingerprint(
    *,
    cfg: dict[str, Any],
    detection_cache: Path,
    checkpoint: Path,
    gallery_manifest: Path,
    bert_weight: Path,
    config_bert: Path,
) -> dict[str, Any]:
    return {
        "schema": GALLERY_FEATURE_CACHE_SCHEMA,
        "adapter_version": ADAPTER_VERSION,
        "source_commit": str(cfg["source"]["commit"]),
        "detection_cache_sha256": sha256_file(detection_cache),
        "shared_detection_identity": detection_cache.parent.name,
        "checkpoint_sha256": sha256_file(checkpoint),
        "bert_weight_sha256": sha256_file(bert_weight),
        "config_bert_sha256": sha256_file(config_bert),
        "gallery_manifest_sha256": sha256_file(gallery_manifest),
        "gallery_instruction": str(cfg["instruction_adapter"]["gallery_instruction"]),
        "model": cfg["model"],
    }


@torch.no_grad()
def compute_gallery_instruct_features(
    *,
    cfg: dict[str, Any],
    gallery: Sequence[dict[str, Any]],
    offsets: np.ndarray,
    boxes: np.ndarray,
    model,
    device: torch.device,
    cache_path: Path,
    cache_meta: dict[str, Any],
) -> np.ndarray:
    total = int(offsets[-1])
    feature_dim = int(cfg["model"]["feature_dim"])
    dtype_name = str(cfg["runtime"]["gallery_feature_dtype"])
    if dtype_name not in {"float16", "float32"}:
        raise ValueError("gallery_feature_dtype must be float16 or float32")
    np_dtype = np.float16 if dtype_name == "float16" else np.float32
    batch_size = int(cfg["runtime"]["instruct_batch_size"])
    transform = instruct_transform(cfg)
    neutral = pre_caption(
        str(cfg["instruction_adapter"]["gallery_instruction"]),
        int(cfg["model"]["official_text_max_words"]),
    )

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temp = cache_path.with_name(cache_path.name + ".part")
    temp.unlink(missing_ok=True)
    mmap = np.lib.format.open_memmap(temp, mode="w+", dtype=np_dtype, shape=(total, feature_dim))
    pending_images: list[torch.Tensor] = []
    pending_texts: list[str] = []
    cursor = 0

    def flush() -> None:
        nonlocal pending_images, pending_texts, cursor
        if not pending_images:
            return
        feature = run_instruct_batch(model, pending_images, pending_texts, device)
        if feature.shape[1] != feature_dim:
            raise RuntimeError(f"Official fusion feature dim={feature.shape[1]}, expected {feature_dim}")
        count = len(feature)
        mmap[cursor : cursor + count] = feature.astype(np_dtype, copy=False)
        cursor += count
        pending_images = []
        pending_texts = []

    try:
        for gi, row in enumerate(progress_bar(gallery, desc="Encode gallery persons with Instruct-ReID", total=len(gallery), unit="image")):
            start, end = int(offsets[gi]), int(offsets[gi + 1])
            if end <= start:
                continue
            with Image.open(image_path(row, gi)) as handle:
                image = handle.convert("RGB")
                for box in boxes[start:end]:
                    pending_images.append(transform(crop_box(image, box)))
                    pending_texts.append(neutral)
                    if len(pending_images) >= batch_size:
                        flush()
        flush()
        if cursor != total:
            raise RuntimeError(f"Encoded {cursor} gallery persons, expected {total}")
        mmap.flush()
        mmap = None
        os.replace(temp, cache_path)
        write_json(meta_path(cache_path), cache_meta)
    except Exception:
        try:
            mmap = None
        except Exception:
            pass
        temp.unlink(missing_ok=True)
        raise
    return np.load(cache_path, mmap_mode="r", allow_pickle=False)


def query_feature_fingerprint(
    *,
    cfg: dict[str, Any],
    query_manifest: Path,
    detection_cache: Path,
    selector_cache: Path,
    checkpoint: Path,
    bert_weight: Path,
    config_bert: Path,
    localization: list[dict[str, Any]],
) -> dict[str, Any]:
    compact_localization = [
        {
            "valid": bool(item.get("valid")),
            "predicted_boxes": [t["box"] for t in item.get("targets", [])],
            "instructions": [str(t["instruction"]) for t in item.get("targets", [])],
        }
        for item in localization
    ]
    return {
        "schema": QUERY_FEATURE_CACHE_SCHEMA,
        "adapter_version": ADAPTER_VERSION,
        "source_commit": str(cfg["source"]["commit"]),
        "query_manifest_sha256": sha256_file(query_manifest),
        "detection_cache_sha256": sha256_file(detection_cache),
        "shared_detection_identity": detection_cache.parent.name,
        "selector_cache_sha256": sha256_file(selector_cache),
        "checkpoint_sha256": sha256_file(checkpoint),
        "bert_weight_sha256": sha256_file(bert_weight),
        "config_bert_sha256": sha256_file(config_bert),
        "instruction_adapter": cfg["instruction_adapter"],
        "localization_hash": canonical_hash(compact_localization),
        "model": cfg["model"],
    }


def load_query_feature_cache(
    path: Path, expected_meta: dict[str, Any], num_queries: int, feature_dim: int
) -> tuple[np.ndarray, np.ndarray] | None:
    if not path.is_file() or read_json(meta_path(path)) != expected_meta:
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            offsets = np.asarray(data["offsets"], dtype=np.int64)
            features = np.asarray(data["features"])
    except Exception as error:
        print(f"Ignoring invalid query feature cache {rel(path)}: {error}", flush=True)
        return None
    if offsets.shape != (num_queries + 1,) or int(offsets[0]) != 0 or np.any(np.diff(offsets) < 0):
        return None
    if features.shape != (int(offsets[-1]), feature_dim) or features.dtype.kind != "f":
        return None
    if not np.isfinite(features).all():
        return None
    print(f"Using query-target Instruct-ReID feature cache: {rel(path)}", flush=True)
    return offsets, features


@torch.no_grad()
def compute_query_instruct_features(
    *,
    cfg: dict[str, Any],
    gallery: Sequence[dict[str, Any]],
    boxes: np.ndarray,
    localization: list[dict[str, Any]],
    model,
    device: torch.device,
    cache_path: Path,
    cache_meta: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    feature_dim = int(cfg["model"]["feature_dim"])
    batch_size = int(cfg["runtime"]["instruct_batch_size"])
    transform = instruct_transform(cfg)
    max_words = int(cfg["model"]["official_text_max_words"])

    tasks: list[tuple[int, int, str]] = []
    offsets = np.zeros(len(localization) + 1, dtype=np.int64)
    for qi, item in enumerate(localization):
        if bool(item.get("valid")):
            for target in item.get("targets", []):
                tasks.append(
                    (
                        int(target["source_gallery_index"]),
                        np.asarray(target["box"], dtype=np.float32),
                        pre_caption(str(target["instruction"]), max_words),
                    )
                )
        offsets[qi + 1] = len(tasks)

    features = np.empty((len(tasks), feature_dim), dtype=np.float32)
    pending_images: list[torch.Tensor] = []
    pending_texts: list[str] = []
    pending_indices: list[int] = []

    def flush() -> None:
        nonlocal pending_images, pending_texts, pending_indices
        if not pending_images:
            return
        batch_feat = run_instruct_batch(model, pending_images, pending_texts, device)
        if batch_feat.shape != (len(pending_indices), feature_dim):
            raise RuntimeError(f"Unexpected query fusion feature shape: {batch_feat.shape}")
        features[np.asarray(pending_indices, dtype=np.int64)] = batch_feat
        pending_images = []
        pending_texts = []
        pending_indices = []

    # Grouping by task rather than query keeps exact query-target order while batching model calls.
    for ti, (source_gi, predicted_box, instruction) in enumerate(
        progress_bar(tasks, desc="Encode query targets with Instruct-ReID", total=len(tasks), unit="target")
    ):
        with Image.open(image_path(gallery[source_gi], source_gi)) as handle:
            image = handle.convert("RGB")
            pending_images.append(transform(crop_box(image, predicted_box)))
        pending_texts.append(instruction)
        pending_indices.append(ti)
        if len(pending_images) >= batch_size:
            flush()
    flush()

    if not np.isfinite(features).all():
        raise RuntimeError("Query Instruct-ReID features contain NaN/Inf")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temp = cache_path.with_name(cache_path.name + ".part")
    temp.unlink(missing_ok=True)
    try:
        with temp.open("wb") as handle:
            np.savez(handle, offsets=offsets, features=features.astype(np.float16))
        os.replace(temp, cache_path)
        write_json(meta_path(cache_path), cache_meta)
    except Exception:
        temp.unlink(missing_ok=True)
        raise
    with np.load(cache_path, allow_pickle=False) as data:
        return np.asarray(data["offsets"], dtype=np.int64), np.asarray(data["features"])


def negative_squared_euclidean(query: np.ndarray, gallery: np.ndarray) -> np.ndarray:
    q = np.asarray(query, dtype=np.float32)
    g = np.asarray(gallery, dtype=np.float32)
    return -(
        np.sum(q * q, axis=1, keepdims=True)
        + np.sum(g * g, axis=1, keepdims=True).T
        - 2.0 * (q @ g.T)
    )


def score_one_query(*, query_features, gallery_features, offsets, counts,
                    person_image_index, unmatched_score, person_chunk_size):
    return score_feature_sets(query_features, gallery_features, offsets, unmatched_score,
                              person_chunk_size, metric="negative_squared_euclidean")


def compute_scores(
    *,
    queries: Sequence[dict[str, Any]],
    query_offsets: np.ndarray,
    query_features: np.ndarray,
    gallery_offsets: np.ndarray,
    gallery_features: np.ndarray,
    cfg: dict[str, Any],
    output_path: Path,
) -> np.ndarray:
    counts = np.diff(gallery_offsets).astype(np.int64, copy=False)
    total_persons = int(gallery_offsets[-1])
    person_image_index = np.repeat(np.arange(len(counts), dtype=np.int64), counts)
    if person_image_index.shape != (total_persons,):
        raise RuntimeError("Internal person-image index mismatch")

    unmatched = float(cfg["setmatch"]["unmatched_score"])
    if not math.isfinite(unmatched):
        raise ValueError("setmatch.unmatched_score must be finite")
    chunk = int(cfg["runtime"]["score_person_chunk_size"])
    if chunk <= 0:
        raise ValueError("score_person_chunk_size must be positive")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    scores = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=np.float32,
        shape=(len(queries), len(counts)),
    )
    for qi in progress_bar(range(len(queries)), desc="Instruct-ReID SetMatch", total=len(queries), unit="query"):
        start, end = int(query_offsets[qi]), int(query_offsets[qi + 1])
        scores[qi] = score_one_query(
            query_features=query_features[start:end],
            gallery_features=gallery_features,
            offsets=gallery_offsets,
            counts=counts,
            person_image_index=person_image_index,
            unmatched_score=unmatched,
            person_chunk_size=chunk,
        )
    scores.flush()
    return scores


def validate_scores(scores: np.ndarray, shape: tuple[int, int]) -> None:
    if scores.shape != shape:
        raise ValueError(f"scores.npy shape={scores.shape}, expected={shape}")
    if scores.dtype.kind != "f":
        raise TypeError("scores.npy must be floating point")
    for start in range(0, shape[0], 256):
        block = np.asarray(scores[start : start + 256])
        if not np.isfinite(block).all():
            raise ValueError(f"scores.npy contains NaN/Inf in rows {start}:{start + len(block)}")


def main() -> None:
    tracker = PhaseTracker(METHOD_ID, total=8)

    with tracker.phase("Load config, manifests, and prepared artifacts"):
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
        query_indices = query_gallery_indices(queries, gallery_index)

        checkout = ensure_clean_pinned_source(cfg)
        checkpoint_path = require_file(resolve_path(str(cfg["checkpoint"]["path"])), "final Instruct-ReID checkpoint")
        apply_instruct_calibration(cfg, checkpoint_path, query_manifest, gallery_manifest)
        checkpoint_marker_path = require_file(
            resolve_path(str(cfg["checkpoint"]["prepared_marker"])), "final checkpoint prepared marker"
        )
        checkpoint_marker = read_json(checkpoint_marker_path)
        if checkpoint_marker is None or checkpoint_marker.get("checkpoint_sha256") != sha256_file(checkpoint_path):
            raise RuntimeError("Final Instruct-ReID prepared marker/checkpoint mismatch")

        bert_dir = resolve_path(str(cfg["runtime_assets"]["bert_dir"]))
        bert_weight = require_file(bert_dir / "pytorch_model.bin", "BERT runtime weight")
        config_bert = require_file(resolve_path(str(cfg["runtime_assets"]["config_bert"])), "BERT config")
        device = device_from(str(cfg["runtime"]["device"]))
        tracker.log(f"gallery={len(gallery):,} queries={len(queries):,} device={device}")

    # Construct and load the complete official model on CPU before any expensive
    # detector/selector work. This makes dependency/checkpoint incompatibilities
    # fail in seconds rather than after the full gallery has consumed GPU time.
    with tracker.phase("Preflight pinned official Instruct-ReID model on CPU"):
        model = load_official_instruct_model(
            cfg=cfg,
            checkout=checkout,
            checkpoint_path=checkpoint_path,
            bert_dir=bert_dir,
            config_bert=config_bert,
            device=device,
            move_to_device=False,
        )
        tracker.log(
            f"architecture={cfg['model']['architecture']} task={cfg['model']['test_task_type']} "
            f"feature_dim={cfg['model']['feature_dim']} constructor=ok"
        )

    with tracker.phase("Consume shared person candidates and Subject memberships"):
        person = load_person_context(gallery, queries, gallery_manifest, query_manifest, device)
        detection_cache = person.detections.path
        offsets, boxes = person.detections.offsets, person.detections.boxes
        selector_cache = person.localization.path
        localization = []
        for qi, query in enumerate(queries):
            targets = [{"source_gallery_index": int(query_indices[qi]),
                        "global_person_index": member["global_person_index"],
                        "box": member["box"],
                        "instruction": modification_instruction(query, query["subjects"][si])}
                       for si, member in person.localization.flat(qi)]
            localization.append({"valid": True, "reason": None, "targets": targets})
        valid_queries = len(localization)

    with tracker.phase("Move pinned official Instruct-ReID model to CUDA"):
        model.to(device).eval()
        tracker.log(f"model_device={device}")

    with tracker.phase("Encode gallery person fusion features"):
        gallery_cache = resolve_path(str(cfg["cache"]["gallery_features"]))
        gallery_meta = gallery_feature_fingerprint(
            cfg=cfg,
            detection_cache=detection_cache,
            checkpoint=checkpoint_path,
            gallery_manifest=gallery_manifest,
            bert_weight=bert_weight,
            config_bert=config_bert,
        )
        gallery_features = load_array_cache(
            gallery_cache,
            gallery_meta,
            (int(offsets[-1]), int(cfg["model"]["feature_dim"])),
            "gallery Instruct-ReID fusion-feature",
        )
        if gallery_features is None:
            gallery_features = compute_gallery_instruct_features(
                cfg=cfg,
                gallery=gallery,
                offsets=offsets,
                boxes=boxes,
                model=model,
                device=device,
                cache_path=gallery_cache,
                cache_meta=gallery_meta,
            )

    with tracker.phase("Encode localized query targets with instructions"):
        query_cache = resolve_path(str(cfg["cache"]["query_features"]))
        query_meta = query_feature_fingerprint(
            cfg=cfg,
            query_manifest=query_manifest,
            detection_cache=detection_cache,
            selector_cache=selector_cache,
            checkpoint=checkpoint_path,
            bert_weight=bert_weight,
            config_bert=config_bert,
            localization=localization,
        )
        loaded_query = load_query_feature_cache(
            query_cache,
            query_meta,
            len(queries),
            int(cfg["model"]["feature_dim"]),
        )
        if loaded_query is None:
            query_offsets, query_features = compute_query_instruct_features(
                cfg=cfg,
                gallery=gallery,
                boxes=boxes,
                localization=localization,
                model=model,
                device=device,
                cache_path=query_cache,
                cache_meta=query_meta,
            )
        else:
            query_offsets, query_features = loaded_query
        del model
        torch.cuda.empty_cache()

    with tracker.phase("Compute Partial SetMatch scores"):
        output_dir = resolve_path(str(cfg["output"]["dir"]))
        output_dir.mkdir(parents=True, exist_ok=True)
        scores_path = output_dir / "scores.npy"
        scores = compute_scores(
            queries=queries,
            query_offsets=query_offsets,
            query_features=query_features,
            gallery_offsets=offsets,
            gallery_features=gallery_features,
            cfg=cfg,
            output_path=scores_path,
        )
        validate_scores(scores, (len(queries), len(gallery)))

    with tracker.phase("Write reproducibility metadata"):
        invalid_reasons: dict[str, int] = {}
        for item in localization:
            if not bool(item.get("valid")):
                reason = str(item.get("reason") or "unknown")
                invalid_reasons[reason] = invalid_reasons.get(reason, 0) + 1
        run = {
            "method": cfg["method"],
            "display_name": cfg["display_name"],
            "group": cfg["group"],
            "cpr_supervision": cfg["cpr_supervision"],
            "paper": cfg["paper"],
            "source": {
                "repository": cfg["source"]["repository"],
                "commit": cfg["source"]["commit"],
                "checkout": rel(checkout),
            },
            "checkpoint": {
                "path": rel(checkpoint_path),
                "sha256": sha256_file(checkpoint_path),
                "status": cfg["checkpoint"]["status"],
                "task": cfg["checkpoint"]["task"],
                "official_source_url": checkpoint_marker.get("source_url"),
                "official_source_path": checkpoint_marker.get("source_path"),
            },
            "model": cfg["model"],
            "runtime_assets": {
                "bert_weight": rel(bert_weight),
                "bert_weight_sha256": sha256_file(bert_weight),
                "config_bert": rel(config_bert),
                "config_bert_sha256": sha256_file(config_bert),
            },
            "localization": cfg["localization"],
            "instruction_adapter": cfg["instruction_adapter"],
            "setmatch": matching_metadata(cfg),
            "shared_person_protocol": person.metadata(),
            "adapter_semantics": {
                "predicted_person_instances": True,
                "gt_target_boxes_used": False,
                "target_ids_used": False,
                "positive_labels_used": False,
                "query_image_removed_inside_method": False,
                "gallery_instruction_is_fixed_native_neutral_instruction": True,
                "pairwise_geometry_preserves_official_squared_euclidean": True,
            },
            "cache": {
                "detections": rel(detection_cache),
                "selector_features": rel(selector_cache),
                "gallery_features": rel(gallery_cache),
                "query_features": rel(query_cache),
            },
            "statistics": {
                "detected_persons": int(offsets[-1]),
                "images_without_person": int(np.sum(np.diff(offsets) == 0)),
                "localized_queries": int(valid_queries),
                "failed_query_localization": int(len(queries) - valid_queries),
                "failed_localization_reasons": invalid_reasons,
            },
            "config": rel(config_path),
            "num_queries": len(queries),
            "num_gallery": len(gallery),
            "scores": rel(scores_path),
            "higher_is_better": True,
        }
        run["data_fingerprint"] = manifest_identity(run["shared_person_protocol"])
        run_path = output_dir / "run.json"
        write_json(run_path, run)
        tracker.log(f"scores={rel(scores_path)} run={rel(run_path)}")

    tracker.finish()


if __name__ == "__main__":
    main()
