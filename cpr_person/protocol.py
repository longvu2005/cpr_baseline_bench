"""Lightweight method registry for reporting and person adapters.

This module deliberately uses only the standard library so scene baselines can
build tables without installing person-detector or assignment dependencies.
"""

import re
from pathlib import Path

from . import PROTOCOL_VERSION
from .cache import sha256_file
from .rcr_cache import RCR_DETECTION_LAYOUT, rcr_index_path


PERSON_METHODS = frozenset({
    "word4per_setmatch",
    "fafa_setmatch",
    "instruct_reid_setmatch",
    "basic_setmatch",
    "adafocal_setmatch",
    "groundingdino_clipreid_set",
    "groundingdino_clipreid_set_text",
    "reid_set_qwen_edit_proxy",
    "retrieve_qwen25vl_rerank",
    "per_person_clip_compose_setmatch",
    "structured_caption_retrieval",
})

TEXTFREE_DIAGNOSTIC_METHOD = "groundingdino_clipreid_set_textfree_diagnostic"

# How shared RCR caches are bound to current images (see detection.bind_rcr_images).
IMAGE_BINDINGS = frozenset({"rcr_ids_and_image_sizes", "build_time_sha256"})


RCR_CLIP_NAME = "ViT-B/32"
RCR_CLIP_SHA256 = "40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af"
RCR_SCENE_CLIP_METHODS = frozenset({
    "clip_image", "clip_text", "clip_early_fusion", "clip_late_fusion",
    "groundingdino_clipreid_set_text", "qwen25vl_rewrite_clip", "qwen_image_edit_clip",
    "reid_set_qwen_edit_proxy", "retrieve_qwen25vl_rerank",
})


def rcr_detection_source(provenance):
    if not isinstance(provenance, dict) or provenance.get("version") != PROTOCOL_VERSION:
        return None
    fingerprint = provenance.get("detections_fingerprint")
    if not isinstance(fingerprint, dict):
        return None
    detector = fingerprint.get("protocol")
    if not isinstance(detector, dict) or detector.get("backend") != "rcr_detection_cache":
        return None
    source = fingerprint.get("rcr_cache")
    if (not isinstance(source, dict) or source.get("layout") != RCR_DETECTION_LAYOUT
            or source.get("image_binding") not in IMAGE_BINDINGS
            or any(not isinstance(source.get(key), str) or not re.fullmatch(r"[0-9a-f]{64}", source[key])
                   for key in ("index_sha256", "cache_id", "gallery_ids_sha256"))):
        return None
    return source


def require_aligned_run(run, method):
    if method in RCR_SCENE_CLIP_METHODS:
        from cpr_baselines.scene_protocol import require_scene_run
        require_scene_run(run)
    if method in PERSON_METHODS:
        source = rcr_detection_source(run.get("shared_person_protocol"))
        if source is None:
            raise RuntimeError(f"{method}: rerun with the RCR detection index")
        path = rcr_index_path()
        if path is not None and source["index_sha256"] != sha256_file(path):
            raise RuntimeError(f"{method}: scores use a different RCR detection index")


def require_score_artifact(run, path):
    digest = run.get("scores_sha256")
    if (not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest)
            or sha256_file(path) != digest):
        raise RuntimeError("Source scores differ from their completed run; rerun the source baseline")


# v6: genuine RCR scene/detection indexes are read directly (no forced verified
# copy). Results from earlier protocols must be rerun, not relabelled.
RESULT_PROTOCOL = "rcr-final-hfscene-person-v6-direct-rcr-caches"


def current_implementation(run, root):
    files = run.get("implementation")
    if not isinstance(files, dict) or not files:
        return False
    for name, digest in files.items():
        path = Path(name)
        if path.is_absolute() or ".." in path.parts or not isinstance(digest, str) or not re.fullmatch("[0-9a-f]{64}", digest):
            return False
        if (root/path).is_file() and sha256_file(root/path) != digest:
            return False
    config = run.get("config_path")
    if config is not None:
        path = Path(config)
        if path.is_absolute() or ".." in path.parts:
            return False
        if (root/path).is_file() and sha256_file(root/path) != run.get("config_sha256"):
            return False
    return True
