"""Lightweight method registry for reporting and person adapters.

This module deliberately uses only the standard library so scene baselines can
build tables without installing person-detector or assignment dependencies.
"""

import re

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
            or any(not isinstance(source.get(key), str) or not re.fullmatch(r"[0-9a-f]{64}", source[key])
                   for key in ("index_sha256", "cache_id", "gallery_ids_sha256"))):
        return None
    return source


def require_aligned_run(run, method):
    if method in RCR_SCENE_CLIP_METHODS and run.get("scene_clip_sha256") != RCR_CLIP_SHA256:
        raise RuntimeError(f"{method}: rerun with the RCR OpenAI CLIP ViT-B/32 scene encoder")
    if method in PERSON_METHODS:
        source = rcr_detection_source(run.get("shared_person_protocol"))
        if source is None:
            raise RuntimeError(f"{method}: rerun with the RCR detection index")
        path = rcr_index_path()
        if path is not None and source["index_sha256"] != sha256_file(path):
            raise RuntimeError(f"{method}: scores use a different RCR detection index")
