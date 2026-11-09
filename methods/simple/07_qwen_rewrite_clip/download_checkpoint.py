#!/usr/bin/env python3
"""Prepare Qwen2.5-VL-7B-Instruct and OpenAI CLIP ViT-B/32 for S7.

All network access belongs here. run.py consumes only repository-local artifacts
and sets Transformers/Hugging Face offline mode before loading either model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cpr_baselines.scene_clip import prepare_scene_clip

DEFAULT_CONFIG = Path(__file__).resolve().parent / "config.yaml"


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"Expected YAML mapping: {path}")
    return value


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


def required_qwen_files() -> list[str]:
    return [
        "config.json",
        "generation_config.json",
        "preprocessor_config.json",
        "chat_template.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "merges.txt",
        "vocab.json",
        "model.safetensors.index.json",
        "model-00001-of-00005.safetensors",
        "model-00002-of-00005.safetensors",
        "model-00003-of-00005.safetensors",
        "model-00004-of-00005.safetensors",
        "model-00005-of-00005.safetensors",
    ]


def validate_qwen_snapshot(path: Path) -> list[dict[str, Any]]:
    inventory: list[dict[str, Any]] = []
    for name in required_qwen_files():
        item = path / name
        if not item.is_file() or item.stat().st_size <= 0:
            raise RuntimeError(f"Missing/incomplete Qwen artifact: {rel(item)}")
        inventory.append({"path": name, "size": int(item.stat().st_size)})
    return inventory


def prepare_qwen(cfg: dict[str, Any], force: bool) -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:
        raise RuntimeError(
            "Missing huggingface_hub. Run this method through run_baseline.py so "
            "requirements.txt is installed before checkpoint preparation."
        ) from error

    mllm = cfg["mllm"]
    repo_id = str(mllm["repo_id"])
    revision = str(mllm["revision"])
    path = resolve_path(str(mllm["checkpoint_dir"]))
    marker = resolve_path(str(mllm["prepared_marker"]))

    if marker.is_file() and path.is_dir() and not force:
        try:
            current = json.loads(marker.read_text(encoding="utf-8"))
        except Exception:
            current = None
        if (
            isinstance(current, dict)
            and current.get("repo_id") == repo_id
            and current.get("revision") == revision
        ):
            try:
                inventory = validate_qwen_snapshot(path)
            except RuntimeError:
                inventory = []
            if inventory and current.get("files") == inventory:
                print(f"[skip] {rel(path)} (pinned Qwen snapshot valid)", flush=True)
                return

    path.mkdir(parents=True, exist_ok=True)
    print(f"[download] {repo_id}@{revision}", flush=True)
    snapshot_download(
        repo_id=repo_id,
        revision=revision,
        local_dir=str(path),
        force_download=force,
    )
    inventory = validate_qwen_snapshot(path)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(
        json.dumps(
            {
                "repo_id": repo_id,
                "revision": revision,
                "checkpoint_dir": rel(path),
                "files": inventory,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"[ok] {rel(marker)}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare S7 inference artifacts")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    config_path = resolve_path(args.config)
    cfg = load_yaml(config_path)
    print("[1/2] Prepare pinned Qwen2.5-VL-7B-Instruct snapshot", flush=True)
    prepare_qwen(cfg, args.force)
    print("[2/2] Prepare OpenAI CLIP ViT-B/32", flush=True)
    prepare_scene_clip(cfg["retriever"], args.force)
    print("[status] S7 inference artifacts are ready", flush=True)


if __name__ == "__main__":
    main()
