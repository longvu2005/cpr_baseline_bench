"""Canonical detections, ordered crops and explicitly flagged fallbacks."""
from __future__ import annotations

import os
import subprocess
import sys
import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from . import PROTOCOL_VERSION
from .cache import ROOT, StageCache, atomic_file, cache_lock, resolve, sha256_file

PROTOCOL_PATH = Path(__file__).with_name("protocol.yaml")


def protocol_config():
    cfg = yaml.safe_load(PROTOCOL_PATH.read_text())
    for section, key, env in [("detector", "checkpoint", "CPR_GDINO_CHECKPOINT"),
                              ("selector", "checkpoint", "CPR_CLIP_SELECTOR_CHECKPOINT")]:
        if env in os.environ:
            cfg[section][key] = os.environ[env]
    return cfg


def image_path(row):
    # Only the public manifest path is consumed; no annotations are consulted.
    return resolve(row["path"])


def read_rgb(path):
    # Same stored-pixel convention as FAFA/Grounding DINO's original adapter.
    with Image.open(path) as image:
        return image.convert("RGB")


def crop_person(image, box):
    left, top, right, bottom = np.asarray(box, dtype=float)
    coords = (max(0, int(np.floor(left))), max(0, int(np.floor(top))),
              min(image.width, int(np.ceil(right))), min(image.height, int(np.ceil(bottom))))
    if coords[2] <= coords[0] or coords[3] <= coords[1]:
        raise ValueError("Empty predicted crop")
    return image.crop(coords)


def filter_person_boxes(boxes_cxcywh, confidences, width, height, detector):
    """CPU deterministic NMS: confidence, coordinates, original index break ties."""
    boxes = np.asarray(boxes_cxcywh, dtype=np.float32).reshape(-1, 4).copy()
    scores = np.asarray(confidences, dtype=np.float32).reshape(-1)
    if len(boxes) != len(scores) or width < 1 or height < 1:
        raise ValueError("Invalid detector output/image dimensions")
    boxes *= np.array([width, height, width, height], dtype=np.float32)
    xyxy = np.column_stack((boxes[:, 0] - boxes[:, 2] / 2, boxes[:, 1] - boxes[:, 3] / 2,
                            boxes[:, 0] + boxes[:, 2] / 2, boxes[:, 1] + boxes[:, 3] / 2))
    xyxy[:, 0::2] = np.clip(xyxy[:, 0::2], 0, width)
    xyxy[:, 1::2] = np.clip(xyxy[:, 1::2], 0, height)
    valid = (np.isfinite(xyxy).all(axis=1) & np.isfinite(scores)
             & (scores > float(detector["box_threshold"]))
             & ((xyxy[:, 2] - xyxy[:, 0]) >= float(detector["min_box_size_px"]))
             & ((xyxy[:, 3] - xyxy[:, 1]) >= float(detector["min_box_size_px"])))
    ids = np.flatnonzero(valid)
    ids = sorted(ids.tolist(), key=lambda i: (-scores[i], *xyxy[i].tolist(), i))
    keep = []
    for i in ids:
        if keep:
            others = xyxy[keep]
            intersection = np.maximum(0, np.minimum(others[:, 2:], xyxy[i, 2:])
                                       - np.maximum(others[:, :2], xyxy[i, :2])).prod(axis=1)
            area = (xyxy[i, 2:] - xyxy[i, :2]).prod()
            union = area + (others[:, 2:] - others[:, :2]).prod(axis=1) - intersection
            if np.any(intersection / union > float(detector["nms_iou_threshold"])):
                continue
        keep.append(i)
        if len(keep) == int(detector["max_persons_per_image"]):
            break
    # Store exact crop boundaries, avoiding PIL round-vs-floor differences.
    selected = xyxy[keep]
    selected[:, :2] = np.floor(selected[:, :2])
    selected[:, 2:] = np.ceil(selected[:, 2:])
    return selected.astype(np.float32), scores[keep].astype(np.float32)


@dataclass(frozen=True)
class BoxCandidate:
    box: tuple[float, float, float, float]
    score: float
    fallback: bool = False


@dataclass
class Detections:
    offsets: np.ndarray
    boxes: np.ndarray
    confidences: np.ndarray
    fallback: np.ndarray
    image_sizes: np.ndarray
    fingerprint: dict
    path: Path | None = None
    cache_hit: bool = False
    gallery_manifest: Path | None = None

    def candidates(self):
        return [[BoxCandidate(tuple(self.boxes[j].tolist()), float(self.confidences[j]), bool(self.fallback[j]))
                 for j in range(int(start), int(end))]
                for start, end in zip(self.offsets[:-1], self.offsets[1:])]

    def validate(self, gallery_count):
        if (self.offsets.shape != (gallery_count + 1,) or self.offsets.dtype.kind not in "iu"
                or self.offsets[0] != 0 or np.any(np.diff(self.offsets) < 1)
                or self.boxes.shape != (int(self.offsets[-1]), 4)
                or self.confidences.shape != (len(self.boxes),)
                or self.fallback.shape != (len(self.boxes),) or self.fallback.dtype != np.bool_
                or self.image_sizes.shape != (gallery_count, 2)
                or not np.isfinite(self.boxes).all() or not np.isfinite(self.confidences).all()
                or np.any(self.image_sizes < 1)
                or np.any(self.boxes[:, 2:] <= self.boxes[:, :2])
                or not np.array_equal(self.boxes, np.round(self.boxes))):
            raise ValueError("Invalid canonical detection arrays")
        cap = int(self.fingerprint["protocol"]["max_persons_per_image"])
        for gi, (start, end) in enumerate(zip(self.offsets[:-1], self.offsets[1:])):
            width, height = self.image_sizes[gi]
            b, f, scores = self.boxes[start:end], self.fallback[start:end], self.confidences[start:end]
            if (len(b) > cap or np.any(b < 0) or np.any(b[:, 0::2] > width)
                    or np.any(b[:, 1::2] > height) or np.any(np.diff(scores) > 0)):
                raise ValueError("Invalid ordered/clipped detections")
            if np.any(f):
                if len(b) != 1 or not np.array_equal(b[0], [0, 0, width, height]) or scores[0] != -1:
                    raise ValueError("No-detection fallback must be one flagged full-image candidate")


def detection_array_sha256(detections):
    digest = hashlib.sha256()
    for name in ("offsets", "boxes", "confidences", "fallback", "image_sizes"):
        array = getattr(detections, name)
        digest.update(str((name, array.shape, array.dtype.str)).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def detection_fingerprint(gallery_manifest, cfg=None):
    cfg = cfg or protocol_config()
    detector = cfg["detector"]
    checkpoint = resolve(detector["checkpoint"])
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing shared detector checkpoint: {checkpoint}; run python -m cpr_person.prepare")
    # Trust the checksum of the pinned config even when a mounted cache avoids
    # constructing the detector. A present source checkout is checked by worker.
    import json
    gallery = [json.loads(line) for line in Path(gallery_manifest).read_text().splitlines() if line.strip()]
    images_digest = hashlib.sha256()
    for row in gallery:
        # Content, order and portable manifest paths are immutable inputs.
        images_digest.update(json.dumps([row["image_id"], row["path"], sha256_file(image_path(row))],
                                        separators=(",", ":")).encode())
    return {"adapter_version": PROTOCOL_VERSION, "gallery_manifest_sha256": sha256_file(gallery_manifest),
            "gallery_image_content_sha256": images_digest.hexdigest(),
            "source_commit": detector["source"]["commit"], "config_sha256": detector["config_sha256"],
            "checkpoint_sha256": sha256_file(checkpoint), "image": cfg["image"],
            "protocol": {k: detector[k] for k in ("backend", "text_prompt", "box_threshold", "text_threshold",
                         "nms_iou_threshold", "min_box_size_px", "max_persons_per_image", "batch_size",
                         "max_padding_ratio", "attention_backend")}}


def save_detections(cache, detections):
    path = cache.directory / "persons.npz"
    with atomic_file(path) as handle:
        np.savez_compressed(handle, offsets=detections.offsets, boxes=detections.boxes,
                            confidences=detections.confidences, fallback=detections.fallback,
                            image_sizes=detections.image_sizes)
    cache.seal(path, attention_backend=cache.fingerprint["protocol"]["attention_backend"])
    detections.path = path
    return detections


def load_detections(cache, gallery_count):
    path = cache.find("persons.npz")
    if path is None:
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            result = Detections(**{name: np.array(data[name]) for name in
                ("offsets", "boxes", "confidences", "fallback", "image_sizes")},
                fingerprint=cache.fingerprint, path=path, cache_hit=True)
        result.validate(gallery_count)
        return result
    except (OSError, ValueError, KeyError, EOFError):
        return None


def get_detections(gallery, gallery_manifest, device, *, require_cache=False, runner=None, cfg=None):
    cfg = cfg or protocol_config()
    cache = StageCache("detections", detection_fingerprint(gallery_manifest, cfg))
    result = load_detections(cache, len(gallery))
    if result is not None:
        result.gallery_manifest = resolve(gallery_manifest)
        return result
    if require_cache or os.environ.get("CPR_REQUIRE_SHARED_CACHE") == "1":
        raise RuntimeError(f"No verified shared detection cache for {cache.key}; inference disabled by cache guard")
    with cache_lock(cache.directory):
        result = load_detections(cache, len(gallery))
        if result is not None:
            result.gallery_manifest = resolve(gallery_manifest)
            return result
        if runner is None:
            subprocess.run([sys.executable, "-m", "cpr_person.worker", "--gallery-manifest", str(gallery_manifest),
                            "--cache-dir", str(cache.directory), "--device", str(device)], cwd=ROOT, check=True)
        else:
            # Injection is only for CPU protocol tests; production uses the worker.
            result = runner(gallery, cache.fingerprint)
            result.validate(len(gallery))
            save_detections(cache, result)
        result = load_detections(cache, len(gallery))
        if result is None:
            raise RuntimeError("Shared worker did not produce a valid, sealed detection cache")
        result.cache_hit = False
        result.gallery_manifest = resolve(gallery_manifest)
        return result
