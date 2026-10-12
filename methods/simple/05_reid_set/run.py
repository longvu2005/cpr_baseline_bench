#!/usr/bin/env python3
"""Native CLIP-ReID embeddings with shared text-based Subject localization.
The primary S5 ID is an adapted baseline; text does not enter ReID scoring.
Use --text-free-diagnostic for the separately labeled all-reference-person set.
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
from torchvision import transforms as TVT

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark_progress import progress_bar

from cpr_person.adapter import load_person_context, matching_metadata, PersonContext, manifest_identity
from cpr_person.cache import feature_config_hash
from cpr_person.matching import score_feature_sets
from cpr_person.detection import detection_fingerprint, load_detections, get_detections
from cpr_person.localization import Localization
from cpr_person.cache import StageCache
DEFAULT_CONFIG = Path(__file__).resolve().parent / "config.yaml"
METHOD_ID = "groundingdino_clipreid_set"
ADAPTER_VERSION = "2026-10-12-shared-person-v1"
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
        "feature_config_sha256": feature_config_hash(config_path),
        "detection_cache_sha256": sha256_file(detection_cache),
        "shared_detection_identity": detection_cache.parent.name,
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
        mmap = None
        os.replace(temp, cache_path)
        write_meta(meta_path(cache_path), cache_meta)
    except Exception:
        try:
            mmap = None
        except Exception:
            pass
        temp.unlink(missing_ok=True)
        raise
    finally:
        model = None
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return np.load(cache_path, mmap_mode="r")


def score_one_query(*, query_features, all_features, offsets, counts,
                    person_image_index, unmatched_score, feature_chunk_size):
    return score_feature_sets(query_features, all_features, offsets, unmatched_score, feature_chunk_size)


def compute_scores(*, cfg, queries, query_indices, features, offsets, output_path,
                   query_features_list=None):
    if query_features_list is None:
        raise ValueError("Standard S5 scoring requires shared Subject localization; pass query_features_list")
    if len(query_features_list) != len(queries):
        raise ValueError("Query component list does not match the complete manifest")
    counts = np.diff(offsets)
    person_image_index = np.repeat(np.arange(len(counts)), counts)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp = output_path.with_name(output_path.name + ".partial")
    scores = np.lib.format.open_memmap(temp, mode="w+", dtype=np.float32,
                                     shape=(len(queries), len(counts)))
    for qi in progress_bar(range(len(queries)), desc="CLIP-ReID Partial SetMatch", total=len(queries), unit="query"):
        scores[qi] = score_one_query(query_features=query_features_list[qi], all_features=features,
            offsets=offsets, counts=counts, person_image_index=person_image_index,
            unmatched_score=float(cfg["setmatch"]["unmatched_score"]),
            feature_chunk_size=int(cfg["runtime"]["score_feature_chunk_size"]))
    scores.flush()
    del scores
    os.replace(temp, output_path)
    return np.load(output_path, mmap_mode="r")


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


def prepare_person_branch(*, cfg, config_path, gallery_manifest, query_manifest, device,
                          feature_cache=None, require_cache=False, preflight=True, text_free=False):
    gallery, queries = load_jsonl(gallery_manifest), load_jsonl(query_manifest)
    source_root = ensure_clean_pinned_source(cfg["source"]["clip_reid"], "CLIP-ReID")
    checkpoint = require_file(resolve_path(cfg["reid"]["checkpoint"]), "CLIP-ReID checkpoint")
    backbone = require_file(resolve_path(cfg["reid"]["openai_clip_checkpoint"]), "CLIP backbone")
    if preflight:
        model = load_clipreid_model(cfg=cfg, source_root=source_root, checkpoint=checkpoint,
                                   clip_backbone=backbone, device=device)
        validate_clipreid_forward(model=model, cfg=cfg, device=device)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if text_free:
        d = get_detections(gallery, gallery_manifest, device, require_cache=require_cache)
        person = PersonContext(d, Localization([], {}, {}, None), gallery, queries)
    else:
        person = load_person_context(gallery, queries, gallery_manifest, query_manifest, device,
                                     require_cache=require_cache)
    d = person.detections
    cache = feature_cache or resolve_path(cfg["cache"]["reid_features"])
    meta = feature_fingerprint(cfg=cfg, config_path=config_path, detection_cache=d.path,
                               reid_checkpoint=checkpoint, clip_backbone=backbone)
    features = load_feature_cache(cache, meta, (int(d.offsets[-1]), int(cfg["reid"]["feature_dim"])))
    hit = features is not None
    if features is None:
        features = compute_reid_features(cfg=cfg, gallery=gallery, offsets=d.offsets, boxes=d.boxes,
            source_root=source_root, checkpoint=checkpoint, clip_backbone=backbone,
            cache_path=cache, cache_meta=meta, device=device)
    if text_free:
        index = build_gallery_index(gallery)
        sets = [features[d.offsets[index[q["image_id"]]]:d.offsets[index[q["image_id"]]+1]] for q in queries]
        return person, features, sets, {"detections_hit": d.cache_hit, "features_hit": hit}
    # Full-reference query padding uses the same native ReID preprocessing and
    # encoder, rather than reusing a detected person's embedding.
    gallery_index = build_gallery_index(gallery)
    missing_gi = sorted({gallery_index[queries[qi]["image_id"]]
                         for qi in range(len(queries))
                         if any(m["global_person_index"] is None for si,m in person.localization.flat(qi))})
    full_features = {}
    if missing_gi:
        subset = [gallery[gi] for gi in missing_gi]
        full_boxes = np.asarray([[0,0,*d.image_sizes[gi]] for gi in missing_gi], np.float32)
        full_meta = {**meta, "kind": "full_reference_query_padding", "gallery_indices": missing_gi}
        full_cache = cache.with_name("full_reference_" + canonical_hash(full_meta)[:20] + ".npy")
        full = load_feature_cache(full_cache, full_meta, (len(subset), int(cfg["reid"]["feature_dim"])))
        if full is None:
            full = compute_reid_features(cfg=cfg, gallery=subset, offsets=np.arange(len(subset)+1),
                boxes=full_boxes, source_root=source_root, checkpoint=checkpoint, clip_backbone=backbone,
                cache_path=full_cache, cache_meta=full_meta, device=device)
        full_features = dict(zip(missing_gi, full))
    query_sets = []
    for qi, query in enumerate(queries):
        gi = gallery_index[query["image_id"]]
        query_sets.append(np.stack([features[m["global_person_index"]] if m["global_person_index"] is not None
                                   else full_features[gi] for si,m in person.localization.flat(qi)]).astype(np.float32))
    stats = {"detections_hit": d.cache_hit, "features_hit": hit,
             "num_detected_persons": int(d.offsets[-1]), "shared_person_protocol": person.metadata()}
    return person, features, query_sets, stats


def main():
    parser = argparse.ArgumentParser(description="CLIP-ReID + shared Subject localization + Partial SetMatch")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--require-detection-cache", action="store_true")
    parser.add_argument("--text-free-diagnostic", action="store_true",
                        help="Separate diagnostic: use every reference detection; no CLIP selector/text")
    args = parser.parse_args()
    config_path = resolve_path(args.config)
    cfg = load_yaml(config_path)
    if cfg.get("method") != METHOD_ID:
        raise ValueError(f"Expected config method {METHOD_ID}")
    device = device_from(str(cfg["runtime"]["device"]))
    if device.type != "cuda":
        raise RuntimeError("Pinned official CLIP-ReID constructs its backbone on CUDA")
    gallery_manifest, query_manifest = resolve_path(cfg["data"]["gallery_manifest"]), resolve_path(cfg["data"]["query_manifest"])
    gallery, queries = load_jsonl(gallery_manifest), load_jsonl(query_manifest)
    source = ensure_clean_pinned_source(cfg["source"]["clip_reid"], "CLIP-ReID")
    checkpoint, backbone = resolve_path(cfg["reid"]["checkpoint"]), resolve_path(cfg["reid"]["openai_clip_checkpoint"])
    model = load_clipreid_model(cfg=cfg, source_root=source, checkpoint=checkpoint, clip_backbone=backbone, device=device)
    shape = validate_clipreid_forward(model=model, cfg=cfg, device=device)
    del model
    torch.cuda.empty_cache()
    if args.preflight_only:
        fp = detection_fingerprint(gallery_manifest)
        cached = load_detections(StageCache("detections", fp), len(gallery))
        print(f"CLIP-ReID forward={shape}; shared_detection_cache={'HIT' if cached else 'MISS'}")
        return
    person, features, query_sets, stats = prepare_person_branch(cfg=cfg, config_path=config_path,
        gallery_manifest=gallery_manifest, query_manifest=query_manifest, device=device,
        require_cache=args.require_detection_cache, preflight=False, text_free=args.text_free_diagnostic)
    method = METHOD_ID
    output = resolve_path(cfg["output"]["dir"])
    metadata = person.metadata()
    if args.text_free_diagnostic:
        method += "_textfree_diagnostic"
        output = ROOT / "runs" / method
        index = build_gallery_index(gallery)
        query_sets = [features[person.detections.offsets[index[q["image_id"]]]:
                              person.detections.offsets[index[q["image_id"]]+1]] for q in queries]
        metadata["query_selection_uses_text"] = False
        metadata["localization_fingerprint"] = None
        metadata["diagnostic_only"] = True
    query_indices = query_gallery_indices(queries, build_gallery_index(gallery))
    scores = compute_scores(cfg=cfg, queries=queries, query_indices=query_indices, features=features,
        offsets=person.detections.offsets, output_path=output/"scores.npy", query_features_list=query_sets)
    validate_scores(scores, len(queries), len(gallery))
    run = {"method": method, "display_name": cfg["display_name"] if method==METHOD_ID else "CLIP-ReID all-person text-free diagnostic",
           "group": cfg["group"], "cpr_supervision": cfg["cpr_supervision"], "adapter_version": ADAPTER_VERSION,
           "sources": cfg["source"], "reid": cfg["reid"], "setmatch": matching_metadata(cfg, score_space="cosine"),
           "shared_person_protocol": metadata,
           "benchmark_adaptation": {"text_used_for_localization": method==METHOD_ID,
                "text_used_for_reid_or_pairwise_scoring": False, "representation": "native CLIP-ReID MSMT17"},
           "cache": {"person_features": str(resolve_path(cfg["cache"]["reid_features"])), **stats},
           "num_queries": len(queries), "num_gallery": len(gallery), "scores": str(output/"scores.npy"),
           "higher_is_better": True, "config": rel(config_path)}
    if args.text_free_diagnostic:
        run["data_fingerprint"] = {"gallery_sha256": person.detections.fingerprint["gallery_manifest_sha256"],
                                   "queries_sha256": sha256_file(query_manifest)}
    else:
        run["data_fingerprint"] = manifest_identity(run["shared_person_protocol"])
    write_meta(output/"run.json", run)
    print(f"Saved {output/'scores.npy'} shape={scores.shape}")


if __name__ == "__main__":
    main()
