"""Frozen native HF globals shared with RCR Proposed, without learned layers."""

from __future__ import annotations

import os
from pathlib import Path
import re

from cpr_person.cache import resolve, read_json, sha256_file, canonical_hash

MODEL = "openai/clip-vit-base-patch32"
REVISION = "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"
SPEC = {
    "version": "rcr-proposed-hf-scene-v2",
    "model": MODEL,
    "revision": REVISION,
    "scene_size": [224, 224],
    "global_dim": 512,
    "dtype": "float32",
    "image": "RGB stored pixels; PIL bicubic black letterbox; HF normalize; no resize/center crop",
    "text": "CLIPTokenizerFast; native vocabulary; right padding; EOS/truncation 77",
    "projection": "frozen_pretrained_only",
    "proposed_learned_layers": False,
}


def checkpoint_path(value=None):
    return resolve(
        os.environ.get("CPR_SCENE_CLIP_CHECKPOINT")
        or value
        or "checkpoints/clip/hf-vit-base-patch32"
    )


def check_config(cfg):
    if cfg.get("name") != MODEL or cfg.get("revision") != REVISION:
        raise ValueError("Use the pinned frozen HF CLIP from RCR Proposed")
    return checkpoint_path(cfg.get("checkpoint"))


def assets(directory):
    directory = checkpoint_path(directory)
    marker = read_json(directory / "prepared.json")
    if (
        not isinstance(marker, dict)
        or marker.get("protocol") != SPEC
        or not marker.get("files")
    ):
        raise RuntimeError(
            f"Missing/current HF scene checkpoint: {directory}; run checkpoint preparation"
        )
    names = [item["path"] for item in marker["files"]]
    if len(names) != len(set(names)) or not {
        "pytorch_model.bin",
        "config.json",
        "preprocessor_config.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
    } <= set(names):
        raise ValueError("Incomplete scene snapshot inventory")
    for item in marker["files"]:
        path = Path(item["path"])
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("Invalid snapshot path")
        path = directory / path
        if (
            not path.is_file()
            or path.stat().st_size != item["size"]
            or sha256_file(path) != item["sha256"]
        ):
            raise RuntimeError(f"Scene checkpoint asset changed: {path}")
    digest = canonical_hash({"protocol": SPEC, "files": marker["files"]})
    if digest != marker.get("assets_sha256"):
        raise ValueError("Scene snapshot inventory digest differs")
    return {"protocol": SPEC, "assets_sha256": digest}


def require_scene_run(run):
    value = run.get("scene_clip")

    def digest(x):
        return isinstance(x, str) and re.fullmatch("[0-9a-f]{64}", x)

    if (
        not isinstance(value, dict)
        or value.get("protocol") != SPEC
        or not digest(value.get("assets_sha256"))
        or not digest(value.get("implementation_sha256"))
        or not isinstance(value.get("runtime"), dict)
        or not digest(value["runtime"].get("environment_sha256"))
        or run.get("scene_clip_sha256") != value["assets_sha256"]
    ):
        raise RuntimeError(
            "Rerun with frozen HF CLIP, RCR letterbox and current scene runtime"
        )
    return value
