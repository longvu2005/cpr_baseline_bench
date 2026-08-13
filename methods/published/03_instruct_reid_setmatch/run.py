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
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

import clip
import numpy as np
import torch
import yaml
from PIL import Image
from scipy.optimize import linear_sum_assignment
from torchvision import transforms as TVT
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.transforms import functional as TVF

os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark_progress import PhaseTracker, progress_bar  # noqa: E402

DEFAULT_CONFIG = Path(__file__).resolve().parent / "config.yaml"
METHOD_ID = "instruct_reid_setmatch"
ADAPTER_VERSION = "2026-08-13-v1-official-language-reid-setmatch"
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


def validate_detection_arrays(
    offsets: np.ndarray, boxes: np.ndarray, confidences: np.ndarray, num_images: int
) -> None:
    if offsets.shape != (num_images + 1,):
        raise ValueError(f"Detection offsets shape={offsets.shape}, expected {(num_images + 1,)}")
    if offsets.dtype.kind not in "iu" or int(offsets[0]) != 0 or np.any(np.diff(offsets) < 0):
        raise ValueError("Detection offsets must be monotonic integers starting at zero")
    if boxes.shape != (int(offsets[-1]), 4):
        raise ValueError("Detection box count does not match offsets")
    if confidences.shape != (int(offsets[-1]),):
        raise ValueError("Detection confidence count does not match offsets")
    if not np.isfinite(boxes).all() or not np.isfinite(confidences).all():
        raise ValueError("Detection cache contains NaN/Inf")
    if len(boxes) and np.any((boxes[:, 2] <= boxes[:, 0]) | (boxes[:, 3] <= boxes[:, 1])):
        raise ValueError("Detection cache contains invalid xyxy boxes")


def detection_fingerprint(
    cfg: dict[str, Any], gallery_manifest: Path, detector_checkpoint: Path
) -> dict[str, Any]:
    detector = cfg["localization"]["detector"]
    return {
        "schema": DETECTION_CACHE_SCHEMA,
        "adapter_version": ADAPTER_VERSION,
        "gallery_manifest_sha256": sha256_file(gallery_manifest),
        "detector_checkpoint_sha256": sha256_file(detector_checkpoint),
        "backend": str(detector["backend"]),
        "score_threshold": float(detector["score_threshold"]),
        "max_persons_per_image": int(detector["max_persons_per_image"]),
        "person_label": int(detector["person_label"]),
    }


def load_detection_cache(
    path: Path, expected_meta: dict[str, Any], num_images: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    if not path.is_file() or read_json(meta_path(path)) != expected_meta:
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            offsets = np.asarray(data["offsets"], dtype=np.int64)
            boxes = np.asarray(data["boxes"], dtype=np.float32)
            confidences = np.asarray(data["confidences"], dtype=np.float32)
        validate_detection_arrays(offsets, boxes, confidences, num_images)
    except Exception as error:
        print(f"Ignoring invalid detection cache {rel(path)}: {error}", flush=True)
        return None
    print(f"Using person-detection cache: {rel(path)}", flush=True)
    return offsets, boxes, confidences


def save_detection_cache(
    path: Path,
    cache_meta: dict[str, Any],
    offsets: np.ndarray,
    boxes: np.ndarray,
    confidences: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".part")
    temp.unlink(missing_ok=True)
    try:
        with temp.open("wb") as handle:
            np.savez(handle, offsets=offsets, boxes=boxes, confidences=confidences)
        os.replace(temp, path)
        write_json(meta_path(path), cache_meta)
    except Exception:
        temp.unlink(missing_ok=True)
        raise


@torch.no_grad()
def compute_detections(
    cfg: dict[str, Any], gallery: Sequence[dict[str, Any]], checkpoint: Path, device: torch.device
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    detector_cfg = cfg["localization"]["detector"]
    model = fasterrcnn_resnet50_fpn_v2(weights=None, weights_backbone=None)
    try:
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.to(device).eval()

    threshold = float(detector_cfg["score_threshold"])
    person_label = int(detector_cfg["person_label"])
    max_persons = int(detector_cfg["max_persons_per_image"])
    offsets = np.zeros(len(gallery) + 1, dtype=np.int64)
    boxes_all: list[np.ndarray] = []
    conf_all: list[np.ndarray] = []
    total = 0

    for gi, row in enumerate(progress_bar(gallery, desc="Detect persons", total=len(gallery), unit="image")):
        with Image.open(image_path(row, gi)) as handle:
            tensor = TVF.to_tensor(handle.convert("RGB")).to(device)
        output = model([tensor])[0]
        labels = output["labels"]
        scores = output["scores"]
        boxes = output["boxes"]
        keep = (labels == person_label) & (scores >= threshold)
        boxes = boxes[keep]
        scores = scores[keep]
        if len(scores):
            order = torch.argsort(scores, descending=True)[:max_persons]
            boxes = boxes[order]
            scores = scores[order]
            box_np = boxes.detach().cpu().numpy().astype(np.float32, copy=False)
            conf_np = scores.detach().cpu().numpy().astype(np.float32, copy=False)
        else:
            box_np = np.empty((0, 4), dtype=np.float32)
            conf_np = np.empty((0,), dtype=np.float32)
        boxes_all.append(box_np)
        conf_all.append(conf_np)
        total += len(box_np)
        offsets[gi + 1] = total

    del model
    torch.cuda.empty_cache()
    boxes_np = np.concatenate(boxes_all, axis=0) if total else np.empty((0, 4), np.float32)
    conf_np = np.concatenate(conf_all, axis=0) if total else np.empty((0,), np.float32)
    validate_detection_arrays(offsets, boxes_np, conf_np, len(gallery))
    return offsets, boxes_np, conf_np


def crop_box(image: Image.Image, box: np.ndarray) -> Image.Image:
    left = max(0, int(math.floor(float(box[0]))))
    top = max(0, int(math.floor(float(box[1]))))
    right = min(image.width, int(math.ceil(float(box[2]))))
    bottom = min(image.height, int(math.ceil(float(box[3]))))
    if right <= left or bottom <= top:
        raise RuntimeError(f"Invalid crop box: {box.tolist()}")
    return image.crop((left, top, right, bottom))


def selector_fingerprint(
    detection_cache: Path, selector_checkpoint: Path, gallery_manifest: Path
) -> dict[str, Any]:
    return {
        "schema": SELECTOR_CACHE_SCHEMA,
        "adapter_version": ADAPTER_VERSION,
        "detection_cache_sha256": sha256_file(detection_cache),
        "selector_checkpoint_sha256": sha256_file(selector_checkpoint),
        "gallery_manifest_sha256": sha256_file(gallery_manifest),
        "selector_model": "ViT-B/32",
    }


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


@torch.no_grad()
def compute_selector_person_features(
    *,
    gallery: Sequence[dict[str, Any]],
    offsets: np.ndarray,
    boxes: np.ndarray,
    model,
    preprocess,
    device: torch.device,
    cache_path: Path,
    cache_meta: dict[str, Any],
    batch_size: int,
) -> np.ndarray:
    total = int(offsets[-1])
    feature_dim = int(model.text_projection.shape[1])
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temp = cache_path.with_name(cache_path.name + ".part")
    temp.unlink(missing_ok=True)
    mmap = np.lib.format.open_memmap(temp, mode="w+", dtype=np.float16, shape=(total, feature_dim))
    pending: list[torch.Tensor] = []
    cursor = 0

    def flush() -> None:
        nonlocal pending, cursor
        if not pending:
            return
        batch = torch.stack(pending).to(device, non_blocking=True)
        feat = model.encode_image(batch).float()
        feat /= feat.norm(dim=1, keepdim=True).clamp_min(1e-12)
        count = len(feat)
        mmap[cursor : cursor + count] = feat.cpu().numpy().astype(np.float16)
        cursor += count
        pending = []

    try:
        for gi, row in enumerate(progress_bar(gallery, desc="Encode selector person crops", total=len(gallery), unit="image")):
            start, end = int(offsets[gi]), int(offsets[gi + 1])
            if end <= start:
                continue
            with Image.open(image_path(row, gi)) as handle:
                image = handle.convert("RGB")
                for box in boxes[start:end]:
                    pending.append(preprocess(crop_box(image, box)))
                    if len(pending) >= batch_size:
                        flush()
        flush()
        if cursor != total:
            raise RuntimeError(f"Selector encoded {cursor} persons, expected {total}")
        mmap.flush()
        del mmap
        os.replace(temp, cache_path)
        write_json(meta_path(cache_path), cache_meta)
    except Exception:
        try:
            del mmap
        except Exception:
            pass
        temp.unlink(missing_ok=True)
        raise
    return np.load(cache_path, mmap_mode="r", allow_pickle=False)


def selector_text(subject: dict[str, Any], query: dict[str, Any], fallback_order: Sequence[str]) -> str:
    for field in fallback_order:
        if field == "text":
            value = query.get("text")
        else:
            value = subject.get(field)
        text = str(value or "").strip()
        if text:
            return text
    raise ValueError(f"Query {query.get('query_id')} subject has no usable selector text")


@torch.no_grad()
def encode_clip_texts(model, texts: list[str], device: torch.device) -> np.ndarray:
    tokens = clip.tokenize(texts, truncate=True).to(device)
    feat = model.encode_text(tokens).float()
    feat /= feat.norm(dim=1, keepdim=True).clamp_min(1e-12)
    return feat.cpu().numpy().astype(np.float32, copy=False)


def hungarian_maximize(similarity: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    try:
        return linear_sum_assignment(similarity, maximize=True)
    except TypeError:
        return linear_sum_assignment(-similarity)


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


def localize_query_targets(
    *,
    queries: Sequence[dict[str, Any]],
    query_indices: np.ndarray,
    offsets: np.ndarray,
    selector_features: np.ndarray,
    selector_model,
    device: torch.device,
    fallback_order: Sequence[str],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for qi, query in enumerate(progress_bar(queries, desc="Localize query targets", total=len(queries), unit="query")):
        subjects = query.get("subjects")
        if not isinstance(subjects, list) or not subjects:
            result.append({"valid": False, "reason": "no_subjects", "targets": []})
            continue
        source_gi = int(query_indices[qi])
        start, end = int(offsets[source_gi]), int(offsets[source_gi + 1])
        num_detected = end - start
        if num_detected < len(subjects):
            result.append({"valid": False, "reason": "too_few_predicted_persons", "targets": []})
            continue
        try:
            texts = [selector_text(subject, query, fallback_order) for subject in subjects]
            instructions = [modification_instruction(query, subject) for subject in subjects]
        except Exception as error:
            result.append({"valid": False, "reason": f"text_error:{error}", "targets": []})
            continue
        text_feat = encode_clip_texts(selector_model, texts, device)
        person_feat = np.asarray(selector_features[start:end], dtype=np.float32)
        similarity = text_feat @ person_feat.T
        if len(subjects) == 1:
            cols = np.asarray([int(np.argmax(similarity[0]))], dtype=np.int64)
        else:
            rows, raw_cols = hungarian_maximize(similarity)
            if len(rows) != len(subjects):
                result.append({"valid": False, "reason": "hungarian_localization_failed", "targets": []})
                continue
            order = np.argsort(rows)
            cols = raw_cols[order].astype(np.int64, copy=False)
        targets = [
            {
                "source_gallery_index": source_gi,
                "global_person_index": start + int(col),
                "instruction": instructions[si],
            }
            for si, col in enumerate(cols)
        ]
        result.append({"valid": True, "reason": None, "targets": targets})
    return result


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
):
    checkout_text = str(checkout)
    if checkout_text not in sys.path:
        sys.path.insert(0, checkout_text)

    from reid import models  # type: ignore
    import reid.models.pass_transformer_joint as joint  # type: ignore
    from reid.utils.serialization import copy_state_dict  # type: ignore

    # The released source contains literal '<your project root> ...' placeholders.
    # Keep the checkout immutable and redirect only those three constructor calls.
    original_tokenizer = joint.BertTokenizer.from_pretrained
    original_bert_model = joint.BertForMaskedLM.from_pretrained
    original_bert_config = joint.BertConfig.from_json_file

    def tokenizer_local(cls, _upstream_placeholder, *args, **kwargs):
        return original_tokenizer(str(bert_dir), *args, **kwargs)

    def bert_model_local(cls, _upstream_placeholder, *args, **kwargs):
        return original_bert_model(str(bert_dir), *args, **kwargs)

    def bert_config_local(cls, _upstream_placeholder, *args, **kwargs):
        return original_bert_config(str(config_bert), *args, **kwargs)

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
        joint.BertTokenizer.from_pretrained = original_tokenizer
        joint.BertForMaskedLM.from_pretrained = original_bert_model
        joint.BertConfig.from_json_file = original_bert_config

    model.to(device).eval()
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
        del mmap
        os.replace(temp, cache_path)
        write_json(meta_path(cache_path), cache_meta)
    except Exception:
        try:
            del mmap
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
            "global_person_indices": [int(t["global_person_index"]) for t in item.get("targets", [])],
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
                        int(target["global_person_index"]),
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
    for ti, (source_gi, global_person_index, instruction) in enumerate(
        progress_bar(tasks, desc="Encode query targets with Instruct-ReID", total=len(tasks), unit="target")
    ):
        with Image.open(image_path(gallery[source_gi], source_gi)) as handle:
            image = handle.convert("RGB")
            pending_images.append(transform(crop_box(image, boxes[global_person_index])))
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


def score_one_query(
    *,
    query_features: np.ndarray,
    gallery_features: np.ndarray,
    offsets: np.ndarray,
    counts: np.ndarray,
    person_image_index: np.ndarray,
    unmatched_score: float,
    person_chunk_size: int,
) -> np.ndarray:
    num_gallery = len(counts)
    m = int(query_features.shape[0])
    result = np.full(num_gallery, unmatched_score, dtype=np.float32)
    if m == 0:
        return result

    q = np.asarray(query_features, dtype=np.float32)
    if m == 1:
        total_persons = int(offsets[-1])
        if total_persons == 0:
            return result
        person_scores = np.empty(total_persons, dtype=np.float32)
        q_norm = float(np.dot(q[0], q[0]))
        for start in range(0, total_persons, person_chunk_size):
            end = min(start + person_chunk_size, total_persons)
            g = np.asarray(gallery_features[start:end], dtype=np.float32)
            person_scores[start:end] = -(q_norm + np.sum(g * g, axis=1) - 2.0 * (g @ q[0]))
        image_scores = np.full(num_gallery, -np.inf, dtype=np.float32)
        np.maximum.at(image_scores, person_image_index, person_scores)
        nonempty = counts > 0
        result[nonempty] = image_scores[nonempty]
        return result

    eligible = np.flatnonzero(counts >= m)
    for gi in eligible:
        start, end = int(offsets[gi]), int(offsets[gi + 1])
        gallery = np.asarray(gallery_features[start:end], dtype=np.float32)
        similarity = negative_squared_euclidean(q, gallery)
        rows, cols = hungarian_maximize(similarity)
        if len(rows) != m:
            raise RuntimeError(f"Hungarian returned {len(rows)} pairs for query set size {m}")
        result[gi] = float(np.min(similarity[rows, cols]))
    return result


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
        checkpoint_marker_path = require_file(
            resolve_path(str(cfg["checkpoint"]["prepared_marker"])), "final checkpoint prepared marker"
        )
        checkpoint_marker = read_json(checkpoint_marker_path)
        if checkpoint_marker is None or checkpoint_marker.get("checkpoint_sha256") != sha256_file(checkpoint_path):
            raise RuntimeError("Final Instruct-ReID prepared marker/checkpoint mismatch")

        bert_dir = resolve_path(str(cfg["runtime_assets"]["bert_dir"]))
        bert_weight = require_file(bert_dir / "pytorch_model.bin", "BERT runtime weight")
        config_bert = require_file(resolve_path(str(cfg["runtime_assets"]["config_bert"])), "BERT config")
        detector_checkpoint = require_file(
            resolve_path(str(cfg["localization"]["detector"]["checkpoint"])), "person detector checkpoint"
        )
        selector_checkpoint = require_file(
            resolve_path(str(cfg["localization"]["query_selector"]["checkpoint"])), "CLIP selector checkpoint"
        )
        device = device_from(str(cfg["runtime"]["device"]))
        tracker.log(f"gallery={len(gallery):,} queries={len(queries):,} device={device}")

    with tracker.phase("Detect predicted person instances"):
        detection_cache = resolve_path(str(cfg["cache"]["detections"]))
        det_meta = detection_fingerprint(cfg, gallery_manifest, detector_checkpoint)
        loaded = load_detection_cache(detection_cache, det_meta, len(gallery))
        if loaded is None:
            offsets, boxes, confidences = compute_detections(cfg, gallery, detector_checkpoint, device)
            save_detection_cache(detection_cache, det_meta, offsets, boxes, confidences)
        else:
            offsets, boxes, confidences = loaded
        counts = np.diff(offsets)
        tracker.log(
            f"detected_persons={int(offsets[-1]):,} images_without_person={int(np.sum(counts == 0)):,}"
        )

    with tracker.phase("Encode person crops for target selection"):
        selector_model, selector_preprocess = clip.load(str(selector_checkpoint), device=device, jit=False)
        selector_model.eval()
        selector_cache = resolve_path(str(cfg["cache"]["selector_features"]))
        selector_meta = selector_fingerprint(detection_cache, selector_checkpoint, gallery_manifest)
        selector_dim = int(selector_model.text_projection.shape[1])
        selector_features = load_array_cache(
            selector_cache,
            selector_meta,
            (int(offsets[-1]), selector_dim),
            "CLIP selector person-feature",
        )
        if selector_features is None:
            selector_features = compute_selector_person_features(
                gallery=gallery,
                offsets=offsets,
                boxes=boxes,
                model=selector_model,
                preprocess=selector_preprocess,
                device=device,
                cache_path=selector_cache,
                cache_meta=selector_meta,
                batch_size=int(cfg["runtime"]["selector_batch_size"]),
            )

        localization = localize_query_targets(
            queries=queries,
            query_indices=query_indices,
            offsets=offsets,
            selector_features=selector_features,
            selector_model=selector_model,
            device=device,
            fallback_order=list(cfg["localization"]["query_selector"]["fallback_order"]),
        )
        valid_queries = sum(bool(item.get("valid")) for item in localization)
        tracker.log(f"localized_queries={valid_queries:,}/{len(queries):,}")
        del selector_model
        torch.cuda.empty_cache()

    with tracker.phase("Load pinned official Instruct-ReID inference model"):
        model = load_official_instruct_model(
            cfg=cfg,
            checkout=checkout,
            checkpoint_path=checkpoint_path,
            bert_dir=bert_dir,
            config_bert=config_bert,
            device=device,
        )
        tracker.log(
            f"architecture={cfg['model']['architecture']} task={cfg['model']['test_task_type']} "
            f"feature_dim={cfg['model']['feature_dim']}"
        )

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

    with tracker.phase("Compute Hungarian + strict-min SetMatch scores"):
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
            "setmatch": cfg["setmatch"],
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
        run_path = output_dir / "run.json"
        write_json(run_path, run)
        tracker.log(f"scores={rel(scores_path)} run={rel(run_path)}")

    tracker.finish()


if __name__ == "__main__":
    main()
