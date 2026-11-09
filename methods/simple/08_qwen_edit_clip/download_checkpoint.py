#!/usr/bin/env python3
"""Prepare Qwen-Image-Edit-2509 and OpenAI CLIP ViT-B/32 for S8.

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
from huggingface_hub import snapshot_download

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


def inventory_files(root: Path) -> list[dict[str, Any]]:
    files: list[dict[str, Any]] = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        files.append({"path": str(path.relative_to(root)), "size": int(path.stat().st_size)})
    return files


def inventory_matches(root: Path, files: list[dict[str, Any]]) -> bool:
    if not files:
        return False
    for item in files:
        path = root / str(item.get("path", ""))
        size = item.get("size")
        if not path.is_file() or not isinstance(size, int) or path.stat().st_size != size:
            return False
    return True


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def prepare_generator(cfg: dict[str, Any], force: bool) -> None:
    gen = cfg["generator"]
    repo_id = str(gen["repo_id"])
    revision = str(gen["revision"])
    checkpoint_dir = resolve_path(str(gen["checkpoint_dir"]))
    marker = resolve_path(str(gen["prepared_marker"]))

    if marker.is_file() and checkpoint_dir.is_dir() and not force:
        try:
            current = json.loads(marker.read_text(encoding="utf-8"))
        except Exception:
            current = None
        if isinstance(current, dict):
            expected_core = {
                "repo_id": repo_id,
                "revision": revision,
                "pipeline_class": str(gen["pipeline_class"]),
            }
            if all(current.get(k) == v for k, v in expected_core.items()):
                files = current.get("files")
                if isinstance(files, list) and inventory_matches(checkpoint_dir, files):
                    print(f"[skip] {rel(marker)} and snapshot inventory are valid", flush=True)
                    return

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        revision=revision,
        local_dir=str(checkpoint_dir),
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    files = inventory_files(checkpoint_dir)
    if not files:
        raise RuntimeError(f"Downloaded snapshot is empty: {rel(checkpoint_dir)}")
    payload = {
        "repo_id": repo_id,
        "revision": revision,
        "pipeline_class": str(gen["pipeline_class"]),
        "license": str(gen.get("license", "")),
        "checkpoint_dir": rel(checkpoint_dir),
        "files": files,
    }
    write_json(marker, payload)
    print(f"[ok] {rel(marker)}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare S8 inference artifacts")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    config_path = resolve_path(args.config)
    cfg = load_yaml(config_path)
    print("[1/2] Prepare pinned Qwen-Image-Edit-2509 snapshot", flush=True)
    prepare_generator(cfg, args.force)
    print("[2/2] Prepare OpenAI CLIP ViT-B/32", flush=True)
    prepare_scene_clip(cfg["retriever"], args.force)
    print("[status] S8 inference artifacts are ready", flush=True)


if __name__ == "__main__":
    main()
