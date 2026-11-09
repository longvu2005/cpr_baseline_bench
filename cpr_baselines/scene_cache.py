"""Read proven native RCR globals in exact split-gallery order."""

import os
from pathlib import Path

from cpr_person.cache import ROOT, canonical_hash, sha256_file
from .scene_protocol import MODEL, REVISION


def scene_index_path():
    value = os.environ.get("CPR_RCR_SCENE_CACHE")
    if not value:
        return None
    path = Path(value).expanduser().absolute()
    if path.is_dir():
        path /= "index.pt"
    if not path.is_file() or (path.parent / ".building").exists():
        raise ValueError("Missing/incomplete RCR scene cache")
    return path


def read_scene_globals(path, gallery, full_gallery_ids_sha256=None):
    import torch

    before = path.stat()
    digest = sha256_file(path)
    index = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if (
        index.get("layout") != "clip-scene-v1"
        or index.get("image_encoder")
        != {"model": MODEL, "revision": REVISION, "scene_size": [224, 224]}
        or index.get("global_dim") != 512
        or index.get("scene_dim") != 768
        or tuple(index.get("patch_hw", ())) != (7, 7)
    ):
        raise ValueError("Expected native RCR HF CLIP B/32 globals")
    ids, hashes = index.get("image_ids"), index.get("image_content_sha256")
    if not isinstance(ids, list) or not ids or len(set(ids)) != len(ids):
        raise ValueError("Invalid scene gallery order")
    if (
        full_gallery_ids_sha256 is not None
        and canonical_hash(ids) != full_gallery_ids_sha256
    ):
        raise ValueError("Scene cache differs from the full final gallery")
    if not isinstance(hashes, list) or len(hashes) != len(ids):
        raise ValueError(
            "Missing build-time image digests; verify legacy tensors or rebuild with the paired RCR patch"
        )
    positions = {image_id: i for i, image_id in enumerate(ids)}
    selected = []
    for row in gallery:
        if row["image_id"] not in positions:
            raise ValueError("Missing split image in RCR scene cache")
        i = positions[row["image_id"]]
        path_image = Path(row["path"])
        if hashes[i] != sha256_file(
            path_image if path_image.is_absolute() else ROOT / path_image
        ):
            raise ValueError("Scene cache was built from different image bytes")
        selected.append(i)
    values = index.get("global_features")
    if (
        not isinstance(values, torch.Tensor)
        or values.dtype != torch.float32
        or values.shape != (len(ids), 512)
        or not values.isfinite().all()
        or (values.norm(dim=-1) < 1e-12).any()
    ):
        raise ValueError("Invalid native scene globals")
    result = values[selected].numpy().copy()
    after = path.stat()
    if (before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    ):
        raise RuntimeError("Scene index changed during reading")
    return result, {
        "index_sha256": digest,
        "cache_id": index["cache_id"],
        "image_content_verified": True,
        "native_global_features_only": True,
    }
