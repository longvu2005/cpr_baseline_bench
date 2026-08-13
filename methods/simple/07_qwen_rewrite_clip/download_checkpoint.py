#!/usr/bin/env python3
"""Prepare Qwen2.5-VL-7B-Instruct and OpenAI CLIP ViT-L/14 for S7.

All network access belongs here. run.py consumes only repository-local artifacts
and sets Transformers/Hugging Face offline mode before loading either model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import urllib.request
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = Path(__file__).resolve().parent / "config.yaml"
CLIP_L14_SHA256 = "b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836"
CLIP_L14_URL = (
    "https://openaipublic.azureedge.net/clip/models/"
    f"{CLIP_L14_SHA256}/ViT-L-14.pt"
)


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


def prepare_clip(cfg: dict[str, Any], force: bool) -> None:
    retriever = cfg["retriever"]
    path = resolve_path(str(retriever["checkpoint"]))
    expected = str(retriever.get("checkpoint_sha256", CLIP_L14_SHA256))
    if expected != CLIP_L14_SHA256:
        raise ValueError("Unexpected CLIP ViT-L/14 checksum in config")

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and not force:
        actual = sha256_file(path)
        if actual == expected:
            print(f"[skip] {rel(path)} (checksum valid)", flush=True)
            return
        print(f"[warn] {rel(path)} checksum invalid; replacing", flush=True)

    temp = path.with_suffix(path.suffix + ".part")
    temp.unlink(missing_ok=True)
    request = urllib.request.Request(
        CLIP_L14_URL, headers={"User-Agent": "cpr-baseline-bench/1.0"}
    )
    digest = hashlib.sha256()
    print(f"[download] {rel(path)}", flush=True)
    try:
        with urllib.request.urlopen(request) as response, temp.open("wb") as handle:
            total_header = response.headers.get("Content-Length")
            total = int(total_header) if total_header else 0
            done = 0
            while True:
                chunk = response.read(8 * 1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
                digest.update(chunk)
                done += len(chunk)
                if total:
                    print(
                        f"\r         {done / 2**20:.1f}/{total / 2**20:.1f} MiB "
                        f"({100.0 * done / total:.1f}%)",
                        end="",
                        flush=True,
                    )
        if total:
            print(flush=True)
        actual = digest.hexdigest()
        if actual != expected:
            raise RuntimeError(
                f"CLIP ViT-L/14 checksum mismatch: expected {expected}, got {actual}"
            )
        os.replace(temp, path)
    except Exception:
        temp.unlink(missing_ok=True)
        raise
    print(f"[ok] {rel(path)}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare S7 inference artifacts")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    config_path = resolve_path(args.config)
    cfg = load_yaml(config_path)
    print("[1/2] Prepare pinned Qwen2.5-VL-7B-Instruct snapshot", flush=True)
    prepare_qwen(cfg, args.force)
    print("[2/2] Prepare OpenAI CLIP ViT-L/14", flush=True)
    prepare_clip(cfg, args.force)
    print("[status] S7 inference artifacts are ready", flush=True)


if __name__ == "__main__":
    main()
