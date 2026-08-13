#!/usr/bin/env python3
"""Grounding DINO + CLIP-ReID - Set baseline for the CPR benchmark.

This deliberately text-free S5 baseline tests the reviewer hypothesis that CPR
can be solved by detecting people and re-identifying them. Every predicted
person in the query/reference scene and every predicted person in each gallery
scene is embedded with the official MSMT17 CLIP-ReID ViT-B/16 model. Pairwise
cosine similarity is matched one-to-one with maximum-weight Hungarian
assignment, then aggregated with a strict minimum over the assigned pairs.

No CPR target_ids, positives, GT boxes, subject-to-box mapping, select_text,
modify_text, relation_text, or query text is used by the method.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from scipy.optimize import linear_sum_assignment
from torchvision import transforms as TVT
from torchvision.ops import box_convert, nms

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark_progress import PhaseTracker, progress_bar  # noqa: E402

DEFAULT_CONFIG = Path(__file__).resolve().parent / "config.yaml"
METHOD_ID = "groundingdino_clipreid_set"
ADAPTER_VERSION = "2026-08-13-v4-clipreid-preflight-oomsafe"
# Keep the detector-stage cache identity stable: this patch changes only the
# CLIP-ReID adapter/config loading path, not Grounding DINO detections.
DETECTION_ADAPTER_VERSION = "2026-08-13-v2-py312-source-setmatch"
DETECTION_CACHE_SCHEMA = 1
FEATURE_CACHE_SCHEMA = 1


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


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise TypeError(f"Expected YAML mapping: {path}")
    return data


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


def meta_path(cache_path: Path) -> Path:
    return cache_path.with_suffix(cache_path.suffix + ".meta.json")


def read_meta(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def write_meta(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def device_from(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return device


def image_path(row: dict[str, Any], index: int) -> Path:
    value = row.get("path")
    if not isinstance(value, str) or not value:
        raise KeyError(f"Gallery row {index} has no usable 'path'")
    path = (ROOT / value).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def build_gallery_index(gallery: Sequence[dict[str, Any]]) -> dict[Any, int]:
    result: dict[Any, int] = {}
    for index, row in enumerate(gallery):
        if "image_id" not in row:
            raise KeyError(f"Gallery row {index} has no image_id")
        image_id = row["image_id"]
        if image_id in result:
            raise ValueError(f"Duplicate gallery image_id: {image_id!r}")
        result[image_id] = index
    return result


def query_gallery_indices(
    queries: Sequence[dict[str, Any]], gallery_index: dict[Any, int]
) -> np.ndarray:
    indices: list[int] = []
    for qi, query in enumerate(queries):
        if "image_id" not in query:
            raise KeyError(f"Query row {qi} has no image_id")
        image_id = query["image_id"]
        if image_id not in gallery_index:
            raise ValueError(f"Query image missing from gallery: {image_id!r}")
        indices.append(gallery_index[image_id])
    return np.asarray(indices, dtype=np.int64)


def git_head(checkout: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True
    ).strip()


def ensure_clean_pinned_source(source: dict[str, Any], label: str) -> Path:
    checkout = resolve_path(str(source["local_checkout"]))
    expected = str(source["commit"])
    if not checkout.is_dir():
        raise FileNotFoundError(
            f"Missing pinned {label} source: {rel(checkout)}. "
            "Run this baseline through run_baseline.py first."
        )
    actual = git_head(checkout)
    if actual != expected:
        raise RuntimeError(
            f"{label} source commit mismatch: expected {expected}, got {actual}. "
            "Re-run download_checkpoint.py."
        )
    dirty = subprocess.check_output(
        ["git", "-C", str(checkout), "status", "--porcelain", "--untracked-files=no"],
        text=True,
    ).strip()
    if dirty:
        raise RuntimeError(
            f"Pinned {label} source has tracked local modifications: {rel(checkout)}\n{dirty}"
        )
    return checkout


def configure_groundingdino_source(source_root: Path) -> str:
    """Import Grounding DINO from the pinned checkout without building a wheel.

    The upstream package tries to compile its optional CUDA/C++ extension during
    pip installation.  That wheel build is fragile on hosted Python 3.12
    environments.  The pinned source already contains an equivalent PyTorch
    implementation of multi-scale deformable attention, so use it when the
    custom extension is unavailable.

    Returns the attention backend name for cache/reproducibility metadata.
    """

    source_text = str(source_root)
    if source_text in sys.path:
        sys.path.remove(source_text)
    sys.path.insert(0, source_text)

    # Import the exact module from the pinned checkout.  It may emit the
    # upstream "custom C++ ops" warning before we install the explicit fallback.
    import groundingdino  # type: ignore
    from groundingdino.models.GroundingDINO import ms_deform_attn as msda  # type: ignore

    package_root = Path(groundingdino.__file__).resolve().parent
    expected_root = (source_root / "groundingdino").resolve()
    if package_root != expected_root:
        raise RuntimeError(
            "Grounding DINO import did not resolve to the pinned checkout: "
            f"expected {rel(expected_root)}, got {rel(package_root)}"
        )

    try:
        from groundingdino import _C as _groundingdino_c  # type: ignore
        getattr(_groundingdino_c, "ms_deform_attn_forward")
    except Exception as error:
        class _PytorchMSDeformAttnFunction:
            @staticmethod
            def apply(
                value: torch.Tensor,
                value_spatial_shapes: torch.Tensor,
                value_level_start_index: torch.Tensor,
                sampling_locations: torch.Tensor,
                attention_weights: torch.Tensor,
                im2col_step: int,
            ) -> torch.Tensor:
                del value_level_start_index, im2col_step
                return msda.multi_scale_deformable_attn_pytorch(
                    value,
                    value_spatial_shapes,
                    sampling_locations,
                    attention_weights,
                )

        # MultiScaleDeformableAttention.forward resolves this module global at
        # call time, so replacing it here affects model inference without
        # modifying the pinned official checkout.
        msda.MultiScaleDeformableAttnFunction = _PytorchMSDeformAttnFunction
        print(
            "[warn] Grounding DINO custom CUDA/C++ op is unavailable; "
            "using the official pure-PyTorch deformable-attention fallback. "
            f"Original import error: {type(error).__name__}: {error}",
            flush=True,
        )
        return "official_pytorch_fallback"

    return "official_custom_cuda_op"


def require_file(path: Path, label: str) -> Path:
    if not path.is_file() or path.stat().st_size <= 0:
        raise FileNotFoundError(
            f"Missing {label}: {rel(path)}. Run this baseline through run_baseline.py first."
        )
    return path


def _module_is_from_checkout(module: Any, source_root: Path) -> bool:
    module_file = getattr(module, "__file__", None)
    if not module_file:
        return False
    try:
        return Path(module_file).resolve().is_relative_to(source_root.resolve())
    except (OSError, ValueError):
        return False


def _purge_foreign_top_level_package(name: str, source_root: Path) -> None:
    """Remove a same-named top-level package imported from outside CLIP-ReID.

    CLIP-ReID uses generic top-level package names such as ``config`` and
    ``model``.  In a benchmark process that already imported other projects,
    Python may otherwise reuse an unrelated module from ``sys.modules`` even
    after CLIP-ReID is put first on ``sys.path``.
    """

    current = sys.modules.get(name)
    if current is None or _module_is_from_checkout(current, source_root):
        return
    for module_name in list(sys.modules):
        if module_name == name or module_name.startswith(name + "."):
            del sys.modules[module_name]


def merge_clipreid_official_config(model_cfg: Any, config_path: Path) -> None:
    """Merge the pinned config while handling its one known empty DATASETS node."""

    from yacs.config import CfgNode as CN

    raw = load_yaml(config_path)
    if "DATASETS" not in raw or raw["DATASETS"] is not None:
        raise RuntimeError(
            "Pinned CLIP-ReID config no longer has the expected empty DATASETS "
            "placeholder; refusing to apply the compatibility workaround silently"
        )
    sanitized = dict(raw)
    sanitized.pop("DATASETS")
    model_cfg.merge_from_other_cfg(CN(sanitized))
    print(
        "[compat] ignored exact pinned CLIP-ReID YAML placeholder: DATASETS",
        flush=True,
    )

def configure_groundingdino_offline(cfg: dict[str, Any]) -> Path:
    cache_root = resolve_path(str(cfg["detector"]["runtime_cache"]))
    marker = resolve_path(str(cfg["detector"]["runtime_assets_marker"]))
    if not marker.is_file():
        raise FileNotFoundError(
            f"Missing Grounding DINO runtime-assets marker: {rel(marker)}. "
            "Run download_checkpoint.py first."
        )
    marker_data = read_meta(marker)
    expected = {
        "source_commit": str(cfg["source"]["groundingdino"]["commit"]),
        "text_encoder": "bert-base-uncased",
        "hf_home": rel(cache_root / "huggingface"),
    }
    if marker_data is None or any(marker_data.get(k) != v for k, v in expected.items()):
        raise RuntimeError(
            f"Stale Grounding DINO runtime-assets marker: {rel(marker)}. "
            "Re-run download_checkpoint.py."
        )
    files = marker_data.get("files")
    if not isinstance(files, list) or not files:
        raise RuntimeError(f"Runtime-assets marker has no cache inventory: {rel(marker)}")
    for item in files:
        if not isinstance(item, dict):
            raise RuntimeError(f"Invalid runtime cache inventory in {rel(marker)}")
        name = item.get("path")
        size = item.get("size")
        if not isinstance(name, str) or not isinstance(size, int):
            raise RuntimeError(f"Invalid runtime cache inventory in {rel(marker)}")
        path = cache_root / name
        if not path.is_file() or path.stat().st_size != size:
            raise RuntimeError(
                f"Grounding DINO runtime cache is incomplete: {rel(path)}. "
                "Re-run download_checkpoint.py."
            )

    os.environ["HF_HOME"] = str(cache_root / "huggingface")
    os.environ["XDG_CACHE_HOME"] = str(cache_root / "xdg")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    return cache_root


def detection_fingerprint(
    *,
    cfg: dict[str, Any],
    config_path: Path,
    gallery_manifest: Path,
    detector_config: Path,
    detector_checkpoint: Path,
    attention_backend: str,
) -> dict[str, Any]:
    detector_cfg = cfg["detector"]
    payload = {
        "schema": DETECTION_CACHE_SCHEMA,
        "adapter_version": DETECTION_ADAPTER_VERSION,
        "config_sha256": sha256_file(config_path),
        "gallery_manifest_sha256": sha256_file(gallery_manifest),
        "groundingdino_source_commit": str(cfg["source"]["groundingdino"]["commit"]),
        "detector_config_sha256": sha256_file(detector_config),
        "detector_checkpoint_sha256": sha256_file(detector_checkpoint),
        "attention_backend": attention_backend,
        "detector": {
            "text_prompt": str(detector_cfg["text_prompt"]),
            "box_threshold": float(detector_cfg["box_threshold"]),
            "text_threshold": float(detector_cfg["text_threshold"]),
            "nms_iou_threshold": float(detector_cfg["nms_iou_threshold"]),
            "min_box_size_px": float(detector_cfg["min_box_size_px"]),
            "max_persons_per_image": detector_cfg.get("max_persons_per_image"),
        },
    }
    payload["fingerprint"] = canonical_hash(payload)
    return payload


def feature_fingerprint(
    *,
    cfg: dict[str, Any],
    config_path: Path,
    detection_cache: Path,
    reid_checkpoint: Path,
    clip_backbone: Path,
) -> dict[str, Any]:
    reid_cfg = cfg["reid"]
    payload = {
        "schema": FEATURE_CACHE_SCHEMA,
        "adapter_version": ADAPTER_VERSION,
        "config_sha256": sha256_file(config_path),
        "detection_cache_sha256": sha256_file(detection_cache),
        "clip_reid_source_commit": str(cfg["source"]["clip_reid"]["commit"]),
        "reid_checkpoint_sha256": sha256_file(reid_checkpoint),
        "openai_clip_checkpoint_sha256": sha256_file(clip_backbone),
        "reid": {
            "name": str(reid_cfg["name"]),
            "training_dataset": str(reid_cfg["training_dataset"]),
            "official_config": str(reid_cfg["official_config"]),
            "num_classes": int(reid_cfg["num_classes"]),
            "camera_num": int(reid_cfg["camera_num"]),
            "view_num": int(reid_cfg["view_num"]),
            "feature_dim": int(reid_cfg["feature_dim"]),
            "input_size": list(reid_cfg["input_size"]),
            "pixel_mean": list(reid_cfg["pixel_mean"]),
            "pixel_std": list(reid_cfg["pixel_std"]),
        },
    }
    payload["fingerprint"] = canonical_hash(payload)
    return payload


def validate_detection_arrays(
    offsets: np.ndarray,
    boxes: np.ndarray,
    confidences: np.ndarray,
    num_images: int,
) -> None:
    if offsets.shape != (num_images + 1,):
        raise ValueError(f"Detection offsets shape {offsets.shape}, expected {(num_images + 1,)}")
    if offsets.dtype.kind not in "iu":
        raise TypeError("Detection offsets must be integer")
    if int(offsets[0]) != 0:
        raise ValueError("Detection offsets must start at zero")
    if np.any(offsets[1:] < offsets[:-1]):
        raise ValueError("Detection offsets must be monotonic")
    total = int(offsets[-1])
    if boxes.shape != (total, 4):
        raise ValueError(f"Detection boxes shape {boxes.shape}, expected {(total, 4)}")
    if confidences.shape != (total,):
        raise ValueError(
            f"Detection confidences shape {confidences.shape}, expected {(total,)}"
        )
    if not np.isfinite(boxes).all() or not np.isfinite(confidences).all():
        raise ValueError("Detection cache contains non-finite values")
    if total and np.any(boxes[:, 2:] <= boxes[:, :2]):
        raise ValueError("Detection cache contains invalid xyxy boxes")


def load_detection_cache(
    cache_path: Path,
    expected_meta: dict[str, Any],
    num_images: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    sidecar = meta_path(cache_path)
    if not cache_path.is_file() or not sidecar.is_file():
        return None
    current_meta = read_meta(sidecar)
    if current_meta != expected_meta:
        print(f"Ignoring stale detection cache: {rel(cache_path)}", flush=True)
        return None
    try:
        with np.load(cache_path, allow_pickle=False) as data:
            offsets = np.asarray(data["offsets"], dtype=np.int64)
            boxes = np.asarray(data["boxes"], dtype=np.float32)
            confidences = np.asarray(data["confidences"], dtype=np.float32)
        validate_detection_arrays(offsets, boxes, confidences, num_images)
    except Exception as error:
        print(f"Ignoring invalid detection cache {rel(cache_path)}: {error}", flush=True)
        return None
    print(f"Using person-detection cache: {rel(cache_path)}", flush=True)
    return offsets, boxes, confidences


def save_detection_cache(
    cache_path: Path,
    cache_meta: dict[str, Any],
    offsets: np.ndarray,
    boxes: np.ndarray,
    confidences: np.ndarray,
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temp = cache_path.with_name(cache_path.name + ".part")
    temp.unlink(missing_ok=True)
    try:
        with temp.open("wb") as handle:
            np.savez(handle, offsets=offsets, boxes=boxes, confidences=confidences)
        os.replace(temp, cache_path)
    except Exception:
        temp.unlink(missing_ok=True)
        raise
    write_meta(meta_path(cache_path), cache_meta)


def filter_person_boxes(
    *,
    boxes_cxcywh: torch.Tensor,
    confidences: torch.Tensor,
    width: int,
    height: int,
    cfg: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    if boxes_cxcywh.numel() == 0:
        return np.empty((0, 4), np.float32), np.empty((0,), np.float32)

    scale = torch.tensor([width, height, width, height], dtype=boxes_cxcywh.dtype)
    xyxy = box_convert(boxes_cxcywh * scale, in_fmt="cxcywh", out_fmt="xyxy").float()
    scores = confidences.float()

    xyxy[:, 0::2].clamp_(0.0, float(width))
    xyxy[:, 1::2].clamp_(0.0, float(height))
    min_size = float(cfg["detector"]["min_box_size_px"])
    valid = (xyxy[:, 2] - xyxy[:, 0] >= min_size) & (
        xyxy[:, 3] - xyxy[:, 1] >= min_size
    )
    xyxy = xyxy[valid]
    scores = scores[valid]
    if xyxy.numel() == 0:
        return np.empty((0, 4), np.float32), np.empty((0,), np.float32)

    keep = nms(xyxy, scores, float(cfg["detector"]["nms_iou_threshold"]))
    xyxy = xyxy[keep]
    scores = scores[keep]

    order = torch.argsort(scores, descending=True)
    xyxy = xyxy[order]
    scores = scores[order]
    max_persons = cfg["detector"].get("max_persons_per_image")
    if max_persons is not None:
        limit = int(max_persons)
        if limit <= 0:
            raise ValueError("max_persons_per_image must be null or a positive integer")
        xyxy = xyxy[:limit]
        scores = scores[:limit]

    return xyxy.cpu().numpy().astype(np.float32), scores.cpu().numpy().astype(np.float32)


@torch.no_grad()
def compute_detections(
    *,
    cfg: dict[str, Any],
    gallery: Sequence[dict[str, Any]],
    detector_config: Path,
    detector_checkpoint: Path,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Environment has already been switched to repository-local HF offline mode.
    from groundingdino.util.inference import load_image, load_model, predict

    detector = load_model(
        str(detector_config), str(detector_checkpoint), device=str(device)
    ).to(device)
    detector.eval()

    all_boxes: list[np.ndarray] = []
    all_confidences: list[np.ndarray] = []
    offsets = np.zeros(len(gallery) + 1, dtype=np.int64)
    total = 0

    for gi, row in enumerate(
        progress_bar(gallery, desc="Detect persons", total=len(gallery), unit="image")
    ):
        path = image_path(row, gi)
        image_source, image_tensor = load_image(str(path))
        height, width = image_source.shape[:2]
        boxes, logits, _phrases = predict(
            model=detector,
            image=image_tensor,
            caption=str(cfg["detector"]["text_prompt"]),
            box_threshold=float(cfg["detector"]["box_threshold"]),
            text_threshold=float(cfg["detector"]["text_threshold"]),
            device=str(device),
        )
        boxes_np, conf_np = filter_person_boxes(
            boxes_cxcywh=boxes,
            confidences=logits,
            width=width,
            height=height,
            cfg=cfg,
        )
        all_boxes.append(boxes_np)
        all_confidences.append(conf_np)
        total += len(boxes_np)
        offsets[gi + 1] = total

    del detector
    if device.type == "cuda":
        torch.cuda.empty_cache()

    boxes = (
        np.concatenate(all_boxes, axis=0)
        if total
        else np.empty((0, 4), dtype=np.float32)
    )
    confidences = (
        np.concatenate(all_confidences, axis=0)
        if total
        else np.empty((0,), dtype=np.float32)
    )
    validate_detection_arrays(offsets, boxes, confidences, len(gallery))
    return offsets, boxes, confidences


def validate_clipreid_adapter_contract(model_cfg: Any, cfg: dict[str, Any]) -> None:
    """Reject silent drift between the pinned official recipe and this adapter."""

    expected_size = [int(x) for x in cfg["reid"]["input_size"]]
    expected_mean = [float(x) for x in cfg["reid"]["pixel_mean"]]
    expected_std = [float(x) for x in cfg["reid"]["pixel_std"]]
    checks = {
        "MODEL.NAME": (str(model_cfg.MODEL.NAME), "ViT-B-16"),
        "MODEL.STRIDE_SIZE": (list(model_cfg.MODEL.STRIDE_SIZE), [16, 16]),
        "INPUT.SIZE_TRAIN": (list(model_cfg.INPUT.SIZE_TRAIN), expected_size),
        "INPUT.SIZE_TEST": (list(model_cfg.INPUT.SIZE_TEST), expected_size),
        "INPUT.PIXEL_MEAN": (list(model_cfg.INPUT.PIXEL_MEAN), expected_mean),
        "INPUT.PIXEL_STD": (list(model_cfg.INPUT.PIXEL_STD), expected_std),
        "TEST.NECK_FEAT": (str(model_cfg.TEST.NECK_FEAT), "before"),
        "MODEL.SIE_CAMERA": (bool(model_cfg.MODEL.SIE_CAMERA), False),
        "MODEL.SIE_VIEW": (bool(model_cfg.MODEL.SIE_VIEW), False),
    }
    mismatches = [
        f"{name}: official={actual!r}, adapter_expected={expected!r}"
        for name, (actual, expected) in checks.items()
        if actual != expected
    ]
    if int(cfg["reid"]["feature_dim"]) != 1280:
        mismatches.append(
            "reid.feature_dim: "
            f"configured={int(cfg['reid']['feature_dim'])!r}, adapter_expected=1280"
        )
    if int(cfg["reid"]["num_classes"]) != 1041:
        mismatches.append(
            "reid.num_classes: "
            f"configured={int(cfg['reid']['num_classes'])!r}, MSMT17_expected=1041"
        )
    if int(cfg["reid"]["camera_num"]) != 15:
        mismatches.append(
            "reid.camera_num: "
            f"configured={int(cfg['reid']['camera_num'])!r}, MSMT17_expected=15"
        )
    if int(cfg["reid"]["view_num"]) != 1:
        mismatches.append(
            "reid.view_num: "
            f"configured={int(cfg['reid']['view_num'])!r}, MSMT17_expected=1"
        )
    if mismatches:
        raise RuntimeError(
            "CLIP-ReID adapter contract mismatch; refusing expensive inference:\n  - "
            + "\n  - ".join(mismatches)
        )


def load_clipreid_model(
    *,
    cfg: dict[str, Any],
    source_root: Path,
    checkpoint: Path,
    clip_backbone: Path,
    device: torch.device,
):
    if device.type != "cuda":
        raise RuntimeError(
            "The pinned official CLIP-ReID ViT implementation hard-codes CUDA during "
            "model construction. Run S5 with runtime.device=cuda in a CUDA environment."
        )
    if device.index is not None:
        torch.cuda.set_device(device)

    source_text = str(source_root)
    if source_text in sys.path:
        sys.path.remove(source_text)
    sys.path.insert(0, source_text)

    # CLIP-ReID deliberately uses generic top-level package names (``config``
    # and ``model``). Make their origin deterministic instead of trusting the
    # process-wide import cache.
    _purge_foreign_top_level_package("config", source_root)
    _purge_foreign_top_level_package("model", source_root)
    config_module = importlib.import_module("config")
    make_model_module = importlib.import_module("model.make_model_clipreid")
    if not _module_is_from_checkout(config_module, source_root):
        raise RuntimeError(
            "CLIP-ReID config import did not resolve to the pinned checkout: "
            f"{getattr(config_module, '__file__', None)}"
        )
    if not _module_is_from_checkout(make_model_module, source_root):
        raise RuntimeError(
            "CLIP-ReID model import did not resolve to the pinned checkout: "
            f"{getattr(make_model_module, '__file__', None)}"
        )
    official_cfg = config_module.cfg

    model_cfg = official_cfg.clone()
    merge_clipreid_official_config(
        model_cfg, source_root / str(cfg["reid"]["official_config"])
    )
    validate_clipreid_adapter_contract(model_cfg, cfg)
    model_cfg.DATASETS.NAMES = "msmt17"
    model_cfg.TEST.WEIGHT = str(checkpoint)
    model_cfg.freeze()

    # The official helper normally downloads OpenAI CLIP in model construction.
    # Replace only that download step with the already-prepared exact local file;
    # the official CLIP build_model and CLIP-ReID architecture remain unchanged.
    def load_local_clip_to_cpu(
        backbone_name: str,
        h_resolution: int,
        w_resolution: int,
        vision_stride_size: int,
    ):
        if backbone_name != "ViT-B-16":
            raise ValueError(f"S5 expects ViT-B-16, got {backbone_name!r}")
        try:
            jit_model = torch.jit.load(str(clip_backbone), map_location="cpu").eval()
            state_dict = jit_model.state_dict()
        except RuntimeError:
            try:
                state_dict = torch.load(
                    clip_backbone, map_location="cpu", weights_only=True
                )
            except TypeError:
                state_dict = torch.load(clip_backbone, map_location="cpu")
        return make_model_module.clip.build_model(
            state_dict, h_resolution, w_resolution, vision_stride_size
        )

    make_model_module.load_clip_to_cpu = load_local_clip_to_cpu
    model = make_model_module.make_model(
        model_cfg,
        num_class=int(cfg["reid"]["num_classes"]),
        camera_num=int(cfg["reid"]["camera_num"]),
        view_num=int(cfg["reid"]["view_num"]),
    )
    model.load_param(str(checkpoint))
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def validate_clipreid_forward(
    *,
    model: Any,
    cfg: dict[str, Any],
    device: torch.device,
) -> tuple[int, ...]:
    """Run one real CUDA forward pass and validate the adapter feature contract."""

    height, width = [int(x) for x in cfg["reid"]["input_size"]]
    feature_dim = int(cfg["reid"]["feature_dim"])
    sample = torch.zeros((1, 3, height, width), device=device, dtype=torch.float32)
    feature = model(sample, cam_label=None, view_label=None)
    if feature.ndim != 2 or tuple(feature.shape) != (1, feature_dim):
        raise RuntimeError(
            "CLIP-ReID preflight returned an unexpected feature shape: "
            f"{tuple(feature.shape)}, expected {(1, feature_dim)}"
        )
    if not torch.isfinite(feature).all():
        raise RuntimeError("CLIP-ReID preflight produced non-finite features")
    # CUDA kernels are asynchronous; force synchronization so latent device-side
    # failures surface during the cheap preflight instead of during full encoding.
    torch.cuda.synchronize(device)
    shape = tuple(int(x) for x in feature.shape)
    del feature, sample
    return shape


def _is_cuda_oom(error: RuntimeError) -> bool:
    oom_type = getattr(torch.cuda, "OutOfMemoryError", ())
    return (isinstance(error, oom_type) if oom_type else False) or (
        "out of memory" in str(error).lower() and "cuda" in str(error).lower()
    )


def reid_transform(cfg: dict[str, Any]):
    size = [int(x) for x in cfg["reid"]["input_size"]]
    mean = [float(x) for x in cfg["reid"]["pixel_mean"]]
    std = [float(x) for x in cfg["reid"]["pixel_std"]]
    return TVT.Compose(
        [
            TVT.Resize(size),
            TVT.ToTensor(),
            TVT.Normalize(mean=mean, std=std),
        ]
    )


def load_feature_cache(
    cache_path: Path,
    expected_meta: dict[str, Any],
    expected_shape: tuple[int, int],
) -> np.ndarray | None:
    sidecar = meta_path(cache_path)
    if not cache_path.is_file() or not sidecar.is_file():
        return None
    if read_meta(sidecar) != expected_meta:
        print(f"Ignoring stale ReID feature cache: {rel(cache_path)}", flush=True)
        return None
    try:
        features = np.load(cache_path, mmap_mode="r")
    except Exception as error:
        print(f"Ignoring invalid ReID feature cache {rel(cache_path)}: {error}", flush=True)
        return None
    if features.shape != expected_shape:
        print(
            f"Ignoring incompatible ReID feature cache {rel(cache_path)}: "
            f"shape={features.shape}, expected={expected_shape}",
            flush=True,
        )
        return None
    if features.dtype.kind != "f":
        print(f"Ignoring non-floating ReID feature cache: {rel(cache_path)}", flush=True)
        return None
    print(f"Using CLIP-ReID person-feature cache: {rel(cache_path)}", flush=True)
    return features


@torch.no_grad()
def compute_reid_features(
    *,
    cfg: dict[str, Any],
    gallery: Sequence[dict[str, Any]],
    offsets: np.ndarray,
    boxes: np.ndarray,
    source_root: Path,
    checkpoint: Path,
    clip_backbone: Path,
    cache_path: Path,
    cache_meta: dict[str, Any],
    device: torch.device,
) -> np.ndarray:
    total_persons = int(offsets[-1])
    feature_dim = int(cfg["reid"]["feature_dim"])
    dtype_name = str(cfg["runtime"]["feature_cache_dtype"])
    if dtype_name not in {"float16", "float32"}:
        raise ValueError("feature_cache_dtype must be float16 or float32")
    np_dtype = np.float16 if dtype_name == "float16" else np.float32

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temp = cache_path.with_name(cache_path.name + ".part")
    temp.unlink(missing_ok=True)

    if total_persons == 0:
        with temp.open("wb") as handle:
            np.save(handle, np.empty((0, feature_dim), dtype=np_dtype))
        os.replace(temp, cache_path)
        write_meta(meta_path(cache_path), cache_meta)
        return np.load(cache_path, mmap_mode="r")

    model = load_clipreid_model(
        cfg=cfg,
        source_root=source_root,
        checkpoint=checkpoint,
        clip_backbone=clip_backbone,
        device=device,
    )
    transform = reid_transform(cfg)
    configured_batch_size = int(cfg["runtime"]["reid_batch_size"])
    if configured_batch_size <= 0:
        raise ValueError("reid_batch_size must be > 0")
    active_batch_size = configured_batch_size

    mmap = np.lib.format.open_memmap(
        temp, mode="w+", dtype=np_dtype, shape=(total_persons, feature_dim)
    )
    pending: list[torch.Tensor] = []
    write_cursor = 0

    def flush_batch() -> None:
        nonlocal write_cursor, pending, active_batch_size
        if not pending:
            return

        cpu_batch = torch.stack(pending, dim=0)
        cursor = 0
        while cursor < cpu_batch.shape[0]:
            take = min(active_batch_size, int(cpu_batch.shape[0] - cursor))
            device_batch = None
            feature = None
            try:
                device_batch = cpu_batch[cursor : cursor + take].to(
                    device, non_blocking=True
                )
                feature = model(device_batch, cam_label=None, view_label=None)
                feature = F.normalize(feature.float(), dim=1, eps=1e-12)
                if feature.ndim != 2 or feature.shape != (take, feature_dim):
                    raise RuntimeError(
                        f"CLIP-ReID returned shape {tuple(feature.shape)}, "
                        f"expected {(take, feature_dim)}"
                    )
                if not torch.isfinite(feature).all():
                    raise RuntimeError("CLIP-ReID returned non-finite features")
                output = feature.cpu().numpy().astype(np_dtype, copy=False)
            except RuntimeError as error:
                if not _is_cuda_oom(error) or take <= 1:
                    raise
                new_batch_size = max(1, take // 2)
                active_batch_size = min(active_batch_size, new_batch_size)
                print(
                    "[warn] CUDA OOM during CLIP-ReID encoding; retrying with "
                    f"reid_batch_size={active_batch_size} (configured "
                    f"{configured_batch_size})",
                    flush=True,
                )
                # Drop any tensors retained by the failed CUDA call before retry.
                del feature, device_batch
                torch.cuda.empty_cache()
                continue

            mmap[write_cursor : write_cursor + take] = output
            write_cursor += take
            cursor += take
            del output, feature, device_batch

        pending = []

    try:
        for gi, row in enumerate(
            progress_bar(gallery, desc="Encode detected persons", total=len(gallery), unit="image")
        ):
            start, end = int(offsets[gi]), int(offsets[gi + 1])
            if end <= start:
                continue
            path = image_path(row, gi)
            with Image.open(path) as image_handle:
                image = image_handle.convert("RGB")
                for box in boxes[start:end]:
                    left = max(0, int(math.floor(float(box[0]))))
                    top = max(0, int(math.floor(float(box[1]))))
                    right = min(image.width, int(math.ceil(float(box[2]))))
                    bottom = min(image.height, int(math.ceil(float(box[3]))))
                    if right <= left or bottom <= top:
                        raise RuntimeError(
                            f"Invalid cached crop for gallery row {gi}: {box.tolist()}"
                        )
                    pending.append(transform(image.crop((left, top, right, bottom))))
                    if len(pending) >= active_batch_size:
                        flush_batch()
        flush_batch()
        if write_cursor != total_persons:
            raise RuntimeError(
                f"Encoded {write_cursor} persons, expected {total_persons} from detection cache"
            )
        mmap.flush()
        del mmap
        os.replace(temp, cache_path)
        write_meta(meta_path(cache_path), cache_meta)
    except Exception:
        try:
            del mmap
        except Exception:
            pass
        temp.unlink(missing_ok=True)
        raise
    finally:
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return np.load(cache_path, mmap_mode="r")


def hungarian_maximize(similarity: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    try:
        return linear_sum_assignment(similarity, maximize=True)
    except TypeError:
        return linear_sum_assignment(-similarity)


def score_one_query(
    *,
    query_features: np.ndarray,
    all_features: np.ndarray,
    offsets: np.ndarray,
    counts: np.ndarray,
    person_image_index: np.ndarray,
    unmatched_score: float,
    feature_chunk_size: int,
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
        query_vector = q[0]
        for start in range(0, total_persons, feature_chunk_size):
            end = min(start + feature_chunk_size, total_persons)
            chunk = np.asarray(all_features[start:end], dtype=np.float32)
            person_scores[start:end] = chunk @ query_vector
        image_scores = np.full(num_gallery, -np.inf, dtype=np.float32)
        np.maximum.at(image_scores, person_image_index, person_scores)
        nonempty = counts > 0
        result[nonempty] = image_scores[nonempty]
        return result

    eligible = np.flatnonzero(counts >= m)
    for gi in eligible:
        start, end = int(offsets[gi]), int(offsets[gi + 1])
        gallery_features = np.asarray(all_features[start:end], dtype=np.float32)
        similarity = q @ gallery_features.T
        rows, cols = hungarian_maximize(similarity)
        if len(rows) != m:
            raise RuntimeError(
                f"Hungarian assignment returned {len(rows)} pairs for query set size {m}"
            )
        assigned = similarity[rows, cols]
        result[gi] = float(np.min(assigned))
    return result


def compute_scores(
    *,
    cfg: dict[str, Any],
    queries: Sequence[dict[str, Any]],
    query_indices: np.ndarray,
    features: np.ndarray,
    offsets: np.ndarray,
    output_path: Path,
) -> np.ndarray:
    counts = np.diff(offsets).astype(np.int64, copy=False)
    total_persons = int(offsets[-1])
    person_image_index = np.repeat(
        np.arange(len(counts), dtype=np.int64), counts.astype(np.int64)
    )
    if person_image_index.shape != (total_persons,):
        raise RuntimeError("Internal person-to-image index shape mismatch")

    unmatched_score = float(cfg["setmatch"]["unmatched_score"])
    if not math.isfinite(unmatched_score):
        raise ValueError("setmatch.unmatched_score must be finite")
    chunk_size = int(cfg["runtime"]["score_feature_chunk_size"])
    if chunk_size <= 0:
        raise ValueError("score_feature_chunk_size must be > 0")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    scores = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=np.float32,
        shape=(len(queries), len(counts)),
    )
    for qi in progress_bar(
        range(len(queries)), desc="Hungarian SetMatch", total=len(queries), unit="query"
    ):
        source_gi = int(query_indices[qi])
        q_start, q_end = int(offsets[source_gi]), int(offsets[source_gi + 1])
        query_features = features[q_start:q_end]
        scores[qi] = score_one_query(
            query_features=query_features,
            all_features=features,
            offsets=offsets,
            counts=counts,
            person_image_index=person_image_index,
            unmatched_score=unmatched_score,
            feature_chunk_size=chunk_size,
        )
    scores.flush()
    return scores


def validate_scores(scores: np.ndarray, num_queries: int, num_gallery: int) -> None:
    if scores.shape != (num_queries, num_gallery):
        raise ValueError(
            f"scores.npy shape={scores.shape}, expected {(num_queries, num_gallery)}"
        )
    if scores.dtype.kind != "f":
        raise TypeError("scores.npy must be floating point")
    if not np.isfinite(scores).all():
        bad = int((~np.isfinite(scores)).sum())
        raise ValueError(f"scores.npy contains {bad} non-finite values")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help=(
            "Validate CLIP-ReID import/config/checkpoint/forward and report whether "
            "the Grounding DINO detection cache will be reused, without running "
            "detection, feature extraction, scoring, or evaluation."
        ),
    )
    parser.add_argument(
        "--require-detection-cache",
        action="store_true",
        help=(
            "Abort instead of launching Grounding DINO when the exact detection "
            "cache is missing or stale. Useful for safe recovery after an expensive "
            "detector pass has already completed."
        ),
    )
    args = parser.parse_args()
    tracker = PhaseTracker(METHOD_ID, total=1 if args.preflight_only else 6)

    with tracker.phase("Load config, manifests, and prepared artifacts"):
        config_path = resolve_path(args.config)
        cfg = load_yaml(config_path)
        if str(cfg.get("method")) != METHOD_ID:
            raise ValueError(
                f"Method id mismatch: config has {cfg.get('method')!r}, expected {METHOD_ID!r}"
            )

        gallery_manifest = resolve_path(str(cfg["data"]["gallery_manifest"]))
        query_manifest = resolve_path(str(cfg["data"]["query_manifest"]))
        gallery = load_jsonl(gallery_manifest)
        queries = load_jsonl(query_manifest)
        if not gallery:
            raise ValueError("Gallery manifest is empty")
        if not queries:
            raise ValueError("Query manifest is empty")
        gallery_index = build_gallery_index(gallery)
        query_indices = query_gallery_indices(queries, gallery_index)

        gdino_source = ensure_clean_pinned_source(
            cfg["source"]["groundingdino"], "Grounding DINO"
        )
        clipreid_source = ensure_clean_pinned_source(
            cfg["source"]["clip_reid"], "CLIP-ReID"
        )
        detector_config = require_file(
            gdino_source / str(cfg["detector"]["config"]), "Grounding DINO config"
        )
        detector_checkpoint = require_file(
            resolve_path(str(cfg["detector"]["checkpoint"])),
            "Grounding DINO checkpoint",
        )
        reid_checkpoint = require_file(
            resolve_path(str(cfg["reid"]["checkpoint"])), "CLIP-ReID checkpoint"
        )
        clip_backbone = require_file(
            resolve_path(str(cfg["reid"]["openai_clip_checkpoint"])),
            "OpenAI CLIP ViT-B/16 checkpoint",
        )
        configure_groundingdino_offline(cfg)
        device = device_from(str(cfg["runtime"].get("device", "cuda")))
        if device.type != "cuda":
            raise RuntimeError(
                "This S5 adapter requires CUDA because the pinned official CLIP-ReID "
                "implementation constructs the ViT backbone on CUDA."
            )
        gdino_attention_backend = configure_groundingdino_source(gdino_source)

        # Fail fast on CLIP-ReID config/import/checkpoint incompatibilities before
        # launching the expensive all-gallery Grounding DINO pass. This preflight
        # costs only model-load time and would have caught the YACS failure before
        # the 17,000-image detector run.
        preflight_model = load_clipreid_model(
            cfg=cfg,
            source_root=clipreid_source,
            checkpoint=reid_checkpoint,
            clip_backbone=clip_backbone,
            device=device,
        )
        preflight_shape = validate_clipreid_forward(
            model=preflight_model, cfg=cfg, device=device
        )
        del preflight_model
        torch.cuda.empty_cache()

        detection_cache = resolve_path(str(cfg["cache"]["detections"]))
        detect_meta = detection_fingerprint(
            cfg=cfg,
            config_path=config_path,
            gallery_manifest=gallery_manifest,
            detector_config=detector_config,
            detector_checkpoint=detector_checkpoint,
            attention_backend=gdino_attention_backend,
        )

        if args.preflight_only:
            cached = load_detection_cache(detection_cache, detect_meta, len(gallery))
            if cached is None:
                tracker.log(
                    "preflight-only: clipreid_forward=ok "
                    f"feature_shape={preflight_shape} detection_cache=MISS; "
                    "a full run would execute Grounding DINO over the gallery"
                )
            else:
                offsets, _boxes, _confidences = cached
                counts = np.diff(offsets)
                tracker.log(
                    "preflight-only: clipreid_forward=ok "
                    f"feature_shape={preflight_shape} detection_cache=HIT "
                    f"persons={int(offsets[-1]):,} "
                    f"images_with_person={int((counts > 0).sum()):,}/{len(gallery):,}"
                )
        else:
            output_dir = resolve_path(str(cfg["output"]["dir"]))
            output_dir.mkdir(parents=True, exist_ok=True)
            tracker.log(
                f"gallery={len(gallery):,} queries={len(queries):,} device={device} "
                f"gdino_attention={gdino_attention_backend} clipreid_preflight=ok "
                f"clipreid_feature_shape={preflight_shape} text_used=no"
            )

    if args.preflight_only:
        tracker.finish()
        return

    with tracker.phase("Detect all persons with Grounding DINO"):
        cached = load_detection_cache(detection_cache, detect_meta, len(gallery))
        detection_cache_hit = cached is not None
        if cached is None and args.require_detection_cache:
            raise RuntimeError(
                "Grounding DINO detection cache is missing/stale while "
                "--require-detection-cache is set; refusing to launch the expensive "
                "all-gallery detector pass. Run --preflight-only to inspect the cache "
                "fingerprint first."
            )
        if cached is None:
            offsets, boxes, confidences = compute_detections(
                cfg=cfg,
                gallery=gallery,
                detector_config=detector_config,
                detector_checkpoint=detector_checkpoint,
                device=device,
            )
            save_detection_cache(
                detection_cache, detect_meta, offsets, boxes, confidences
            )
        else:
            offsets, boxes, confidences = cached
        counts = np.diff(offsets)
        tracker.log(
            f"detected_persons={int(offsets[-1]):,} "
            f"images_with_person={int((counts > 0).sum()):,}/{len(gallery):,} "
            f"max_per_image={int(counts.max(initial=0))} cache={rel(detection_cache)}"
        )

    with tracker.phase("Encode detected persons with CLIP-ReID"):
        feature_cache = resolve_path(str(cfg["cache"]["reid_features"]))
        feat_meta = feature_fingerprint(
            cfg=cfg,
            config_path=config_path,
            detection_cache=detection_cache,
            reid_checkpoint=reid_checkpoint,
            clip_backbone=clip_backbone,
        )
        expected_shape = (int(offsets[-1]), int(cfg["reid"]["feature_dim"]))
        features = load_feature_cache(feature_cache, feat_meta, expected_shape)
        feature_cache_hit = features is not None
        if features is None:
            features = compute_reid_features(
                cfg=cfg,
                gallery=gallery,
                offsets=offsets,
                boxes=boxes,
                source_root=clipreid_source,
                checkpoint=reid_checkpoint,
                clip_backbone=clip_backbone,
                cache_path=feature_cache,
                cache_meta=feat_meta,
                device=device,
            )
        tracker.log(
            f"person_features={features.shape} dtype={features.dtype} cache={rel(feature_cache)}"
        )

    with tracker.phase("Compute Hungarian + strict-min query-gallery scores"):
        scores_path = output_dir / "scores.npy"
        scores = compute_scores(
            cfg=cfg,
            queries=queries,
            query_indices=query_indices,
            features=features,
            offsets=offsets,
            output_path=scores_path,
        )

    with tracker.phase("Validate full score-matrix contract"):
        validate_scores(scores, len(queries), len(gallery))
        query_counts = np.diff(offsets)[query_indices]
        tracker.log(
            f"scores={scores.shape} finite=yes query_sets_empty={int((query_counts == 0).sum())} "
            f"query_sets_multi_person={int((query_counts > 1).sum())}"
        )

    with tracker.phase("Write reproducibility metadata"):
        run = {
            "method": METHOD_ID,
            "display_name": str(cfg.get("display_name", "Grounding DINO + CLIP-ReID - Set")),
            "group": str(cfg.get("group", "Simple / Obvious Baselines")),
            "cpr_supervision": str(cfg.get("cpr_supervision", "No")),
            "adapter_version": ADAPTER_VERSION,
            "config": rel(config_path),
            "sources": {
                "groundingdino": {
                    "repository": str(cfg["source"]["groundingdino"]["repository"]),
                    "commit": str(cfg["source"]["groundingdino"]["commit"]),
                    "checkout": rel(gdino_source),
                },
                "clip_reid": {
                    "repository": str(cfg["source"]["clip_reid"]["repository"]),
                    "commit": str(cfg["source"]["clip_reid"]["commit"]),
                    "checkout": rel(clipreid_source),
                },
            },
            "detector": {
                **cfg["detector"],
                "config_path": rel(detector_config),
                "checkpoint_sha256": sha256_file(detector_checkpoint),
                "attention_backend": gdino_attention_backend,
                "source_import_mode": "pinned_checkout_direct",
            },
            "reid": {
                **cfg["reid"],
                "checkpoint_sha256": sha256_file(reid_checkpoint),
                "openai_clip_checkpoint_sha256": sha256_file(clip_backbone),
                "feature_semantics": (
                    "official CLIP-ReID eval feature: concat pre-BN ViT feature (768) "
                    "and projected CLIP feature (512), L2-normalized for cosine scoring"
                ),
            },
            "setmatch": cfg["setmatch"],
            "benchmark_adaptation": {
                "purpose": (
                    "Test whether scene-level CPR can be reduced to person detection plus ReID."
                ),
                "query_person_construction": (
                    "All Grounding DINO person detections in the query/reference image."
                ),
                "gallery_person_construction": (
                    "All Grounding DINO person detections in each gallery image."
                ),
                "pair_score": "cosine similarity between L2-normalized CLIP-ReID person embeddings",
                "assignment": "maximum-weight one-to-one Hungarian matching",
                "aggregation": "strict minimum over assigned pair similarities",
                "missing_person_behavior": (
                    f"score={float(cfg['setmatch']['unmatched_score'])} when gallery has fewer "
                    "detected persons than the query set or the query set is empty"
                ),
                "text_used": False,
                "forbidden_cpr_labels_used": False,
                "query_image_removed_inside_method": False,
            },
            "runtime": cfg["runtime"],
            "cache": {
                "detections": rel(detection_cache),
                "detections_hit": detection_cache_hit,
                "reid_features": rel(feature_cache),
                "reid_features_hit": feature_cache_hit,
            },
            "detection_statistics": {
                "total_persons": int(offsets[-1]),
                "images_with_person": int((counts > 0).sum()),
                "images_without_person": int((counts == 0).sum()),
                "mean_persons_per_image": float(counts.mean()),
                "max_persons_per_image": int(counts.max(initial=0)),
            },
            "num_queries": len(queries),
            "num_gallery": len(gallery),
            "scores": rel(scores_path),
            "higher_is_better": True,
        }
        run_path = output_dir / "run.json"
        run_path.write_text(
            json.dumps(run, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        tracker.log(f"scores={rel(scores_path)} run={rel(run_path)}")

    tracker.finish()


if __name__ == "__main__":
    main()
