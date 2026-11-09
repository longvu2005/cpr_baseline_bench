"""Bind inherited scores to the current source config and actual checkpoint bytes."""

from __future__ import annotations

from functools import lru_cache
import os
from pathlib import Path

import yaml

from cpr_person.cache import ROOT, sha256_file

PATH_KEYS = {
    "checkpoint",
    "checkpoint_dir",
    "path",
    "stage2",
    "stage2_config",
    "base_clip",
    "openai_clip_checkpoint",
    "bert_dir",
    "bert_weight",
    "config_bert",
    "prepared_marker",
    "runtime_assets_marker",
    "local_snapshot",
    "local_dir",
    "cache_dir",
    "cache_root",
    "local_checkout",
    "config",
    "person_protocol",
    "protocol",
    "tokenizer_runtime_dir",
    "runtime_cache",
}


@lru_cache(maxsize=1024)
def _digest(path, signature):
    return sha256_file(Path(path))


def file_identity(path):
    before = path.stat()
    signature = (before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
    digest = _digest(str(path.resolve()), signature)
    after = path.stat()
    if signature != (after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
        raise RuntimeError("Checkpoint asset changed while fingerprinting")
    return {"size": before.st_size, "sha256": digest}


def snapshot(cfg, root=ROOT):
    result, seen_configs = {}, set()

    def visit(value, role):
        if isinstance(value, dict):
            for key, child in value.items():
                if key in {"cache", "output", "data", "isolated_env", "caption_env"}:
                    continue
                if isinstance(child, (dict, list)):
                    visit(child, role + "." + key)
                elif isinstance(child, str) and (
                    key in PATH_KEYS
                    or key.endswith(
                        ("_snapshot", "_mapper", "_cache_root", "_download_root")
                    )
                    or child.startswith(("checkpoints/", "cpr_person/"))
                ):
                    collect(
                        child,
                        role + "." + key,
                        key in {"config", "person_protocol", "protocol"},
                    )
        elif isinstance(value, list):
            for i, child in enumerate(value):
                visit(child, role + "." + str(i))

    def collect(value, role, nested=False):
        path = Path(value)
        path = path if path.is_absolute() else root / path
        if path.is_file():
            result[role] = file_identity(path)
            if (
                nested
                and path.suffix in {".yaml", ".yml"}
                and path.resolve() not in seen_configs
            ):
                seen_configs.add(path.resolve())
                try:
                    config = yaml.safe_load(path.read_text())
                except yaml.YAMLError:
                    return
                if isinstance(config, dict):
                    visit(config, role)
        elif path.is_dir():
            for file in sorted(path.rglob("*")):
                relative = file.relative_to(path)
                if any(
                    part in {".git", ".cache", "__pycache__"} for part in relative.parts
                ):
                    continue
                if file.is_file():
                    result[role + "/" + str(relative)] = file_identity(file)

    visit(cfg, "method")
    reference = cfg.get("first_stage", {}).get("method")
    if reference:
        matches = [
            path
            for path in (root / "methods").glob("*/*/config.yaml")
            if yaml.safe_load(path.read_text()).get("method") == reference
        ]
        if len(matches) != 1:
            raise ValueError("Cannot resolve the frozen first-stage config")
        collect(str(matches[0]), "first_stage_config", True)
    from cpr_person.protocol import PERSON_METHODS, RCR_SCENE_CLIP_METHODS

    if cfg["method"] in RCR_SCENE_CLIP_METHODS:
        from .scene_protocol import checkpoint_path

        collect(str(checkpoint_path()), "shared_scene")
        source = os.environ.get("CPR_RCR_SCENE_CACHE")
        if source:
            path = Path(source)
            collect(str(path / "index.pt" if path.is_dir() else path), "rcr_scene")
    if cfg["method"] in PERSON_METHODS:
        if os.environ.get("CPR_CLIP_SELECTOR_CHECKPOINT"):
            collect(os.environ["CPR_CLIP_SELECTOR_CHECKPOINT"], "shared_selector")
        source = os.environ.get("CPR_RCR_DETECTION_CACHE")
        if source:
            path = Path(source)
            collect(str(path / "index.pt" if path.is_dir() else path), "rcr_detections")
    return result


def require_current_assets(run, root=ROOT):
    config = run.get("config_path")
    if not isinstance(config, str):
        raise RuntimeError(
            "Source run has no frozen config/asset receipt; rerun through run_baseline.py"
        )
    from cpr_person.protocol import RESULT_PROTOCOL, current_implementation

    if run.get("result_protocol") != RESULT_PROTOCOL or not current_implementation(
        run, root
    ):
        raise RuntimeError(
            "Source scores have an old/different implementation; rerun the source baseline"
        )
    path = Path(config)
    path = path if path.is_absolute() else root / path
    if run.get("config_sha256") != sha256_file(path):
        raise RuntimeError("Source method config changed; rerun the source baseline")
    if run.get("assets_snapshot") != snapshot(yaml.safe_load(path.read_text()), root):
        raise RuntimeError(
            "Source checkpoint/runtime assets changed; rerun the source baseline"
        )
