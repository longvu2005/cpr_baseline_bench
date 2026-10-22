"""Canonical detections, ordered crops and explicitly flagged fallbacks."""
from __future__ import annotations

import os
import subprocess
import hashlib
import re
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
    if "CPR_CLIP_SELECTOR_CHECKPOINT" in os.environ:
        cfg["selector"]["checkpoint"] = os.environ["CPR_CLIP_SELECTOR_CHECKPOINT"]
    return cfg


def image_path(row):
    # Only the public manifest path is consumed; no annotations are consulted.
    return resolve(row["path"])


def read_rgb(path):
    # Same stored-pixel convention as FAFA/Grounding DINO's original adapter.
    with Image.open(path) as image:
        return image.convert("RGB")


def crop_person(image, box):
    # RCR Proposed calls PIL.Image.crop on the original floating pixel box.
    # PIL rounds each edge (ties to even) and pads out-of-image pixels with zero.
    return image.crop(tuple(float(value) for value in box))


DETECTION_PROCESSING = {
    "implementation": "hf-grounding-dino-rcr-v1",
    "caption": "processor(images=RGB, text=[[text_prompt]], return_tensors=pt)",
    "preprocessing": "revision-pinned AutoProcessor defaults; one image per forward",
    "postprocessing": "post_process_grounded_object_detection; target_sizes=[(height,width)]",
    "labels": "exact text_labels == text_prompt",
    "boxes": "float32 pixel xyxy; no clipping, rounding, NMS, size filter or cap",
    "ordering": "original model query order after processor/label filtering",
    "confidence": "unaltered processor scores; fallback=-1",
    "zero_detections": "one flagged full-image CPR candidate; RCR instead keeps an empty set",
    "precision": "float32; eval; inference_mode; no autocast",
}


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
                or np.any(self.boxes[:, 2:] <= self.boxes[:, :2])):
            raise ValueError("Invalid canonical detection arrays")
        protocol = self.fingerprint.get("protocol", {})
        if protocol.get("backend") != "huggingface_grounding_dino":
            raise ValueError("Retired/noncanonical person detector cache")
        for gi, (start, end) in enumerate(zip(self.offsets[:-1], self.offsets[1:])):
            width, height = self.image_sizes[gi]
            b, f, scores = self.boxes[start:end], self.fallback[start:end], self.confidences[start:end]
            if np.any(f):
                if len(b) != 1 or not np.array_equal(b[0], [0, 0, width, height]) or scores[0] != -1:
                    raise ValueError("No-detection fallback must be one flagged full-image candidate")
            elif np.any((scores <= float(protocol["threshold"])) | (scores > 1)):
                raise ValueError("Invalid HF detection confidence")


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
    if detector["backend"] != "huggingface_grounding_dino":
        raise ValueError("Only the canonical Hugging Face detector is supported")
    for key in ("revision", "processor_revision"):
        if not re.fullmatch(r"[0-9a-f]{40}", detector[key]):
            raise ValueError(f"detector.{key} must be an immutable HF commit SHA")
    # Provenance comes from immutable revisions and pinned runtime, not whatever
    # Transformers happens to be installed in a native/cache-only consumer.
    # Neither model assets nor the detector interpreter are needed on cache hits.
    import json
    gallery = [json.loads(line) for line in Path(gallery_manifest).read_text().splitlines() if line.strip()]
    images_digest = hashlib.sha256()
    for row in gallery:
        # Content, order and portable manifest paths are immutable inputs.
        images_digest.update(json.dumps([row["image_id"], row["path"], sha256_file(image_path(row))],
                                        separators=(",", ":")).encode())
    return {"adapter_version": PROTOCOL_VERSION, "gallery_manifest_sha256": sha256_file(gallery_manifest),
            "gallery_image_content_sha256": images_digest.hexdigest(),
            "image": cfg["image"], "processing": DETECTION_PROCESSING,
            "protocol": {k: detector[k] for k in ("backend", "model", "revision", "processor_revision",
                         "text_prompt", "threshold", "text_threshold", "runtime_versions", "attention_backend")}}


def save_detections(cache, detections, **metadata):
    path = cache.directory / "persons.npz"
    with atomic_file(path) as handle:
        np.savez_compressed(handle, offsets=detections.offsets, boxes=detections.boxes,
                            confidences=detections.confidences, fallback=detections.fallback,
                            image_sizes=detections.image_sizes)
    cache.seal(path, attention_backend=cache.fingerprint["protocol"]["attention_backend"], **metadata)
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
            from .runtime import detector_python, detector_env
            python = detector_python()
            subprocess.run([python, "-m", "cpr_person.worker", "--gallery-manifest", str(resolve(gallery_manifest)),
                            "--cache-dir", str(cache.directory), "--device", str(device)],
                           cwd=ROOT, env=detector_env(python), check=True)
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
