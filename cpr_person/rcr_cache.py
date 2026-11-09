"""Read the immutable RCR detection index without rerunning Grounding DINO."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from .cache import canonical_hash, sha256_file

RCR_DETECTION_ENV = "CPR_RCR_DETECTION_CACHE"
RCR_DETECTION_LAYOUT = "person-detections-v1"
RCR_DETECTOR = {
    "model": "IDEA-Research/grounding-dino-tiny",
    "threshold": 0.30,
    "text_threshold": 0.25,
}


def rcr_index_path(*, required=False):
    value = os.environ.get(RCR_DETECTION_ENV)
    if not value:
        if required:
            raise ValueError(
                "RCR person baselines require --rcr-detection-cache or "
                f"{RCR_DETECTION_ENV} pointing to RCR cache/detections."
            )
        return None
    path = Path(value).expanduser().absolute()
    if path.is_dir():
        path /= "index.pt"
    if (path.parent / ".building").exists():
        raise ValueError(f"RCR detection cache is still building: {path.parent}")
    if path.name != "index.pt" or not path.is_file():
        raise FileNotFoundError(f"Missing RCR detection index: {path}")
    return path


def _rcr_cache_id(index):
    # Matches rcr.common.detection._cache_id for the packed v1 index.
    identity = {key: index[key] for key in ("layout", "image_ids", "detector", "prompt")}
    if index.get("content_identity") == "boxes-and-source-images-v2":
        identity.update(content_identity=index["content_identity"], image_content_sha256=index["image_content_sha256"])
    digest = hashlib.sha256(json.dumps(
        identity,
        sort_keys=True, separators=(",", ":"),
    ).encode())
    for key in ("image_sizes", "offsets", "boxes_pixel"):
        array = index[key].contiguous().numpy()
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def read_rcr_index(path):
    import torch

    def stamp(stat):
        return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns

    before = stamp(path.stat())
    index = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if (not isinstance(index, dict) or index.get("layout") != RCR_DETECTION_LAYOUT
            or index.get("prompt") != "person"):
        raise ValueError("Expected the RCR person-detections-v1 index")
    detector = index.get("detector")
    if not isinstance(detector, dict) or any(
        detector.get(key) != value for key, value in RCR_DETECTOR.items()
    ):
        raise ValueError("RCR detection model or thresholds differ from the benchmark configuration")
    ids = index["image_ids"]
    if (not isinstance(ids, list) or not ids
            or any(not isinstance(value, str) or not value for value in ids)
            or len(set(ids)) != len(ids)):
        raise ValueError("Invalid RCR detection gallery IDs")
    sizes, offsets, boxes = (index[key] for key in ("image_sizes", "offsets", "boxes_pixel"))
    if (not isinstance(boxes, torch.Tensor) or boxes.dtype != torch.float32
            or boxes.ndim != 2 or boxes.shape[1] != 4
            or not boxes.isfinite().all() or (boxes[:, 2:] <= boxes[:, :2]).any()
            or not isinstance(sizes, torch.Tensor) or sizes.dtype != torch.int64
            or sizes.shape != (len(ids), 2) or (sizes <= 0).any()
            or not isinstance(offsets, torch.Tensor) or offsets.dtype != torch.int64
            or offsets.shape != (len(ids) + 1,) or offsets[0] != 0
            or offsets[-1] != len(boxes) or (offsets.diff() < 0).any()):
        raise ValueError("Invalid RCR detection sizes, offsets or pixel boxes")
    if index.get("cache_id") != _rcr_cache_id(index):
        raise ValueError("RCR detection boxes differ from their recorded cache_id")
    provenance = {
        "layout": index["layout"],
        "cache_id": index["cache_id"],
        "index_sha256": sha256_file(path),
        "gallery_ids_sha256": canonical_hash(ids),
    }
    if before != stamp(path.stat()) or (path.parent / ".building").exists():
        raise RuntimeError("RCR detection cache changed while reading; retry with a completed index")
    return index, provenance
