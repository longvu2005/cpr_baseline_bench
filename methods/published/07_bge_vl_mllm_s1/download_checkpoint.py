#!/usr/bin/env python3
"""Download and validate the pinned official BGE-VL-MLLM-S1 snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import yaml

os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark_progress import PhaseTracker  # noqa: E402

DEFAULT_CONFIG = Path(__file__).resolve().parent / "config.yaml"
MARKER_SCHEMA = 1


def resolve_path(value: str) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else ROOT / path).resolve()


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise TypeError(f"Expected YAML mapping: {path}")
    return data


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def validate_snapshot(snapshot: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    try:
        from safetensors import safe_open
    except ImportError as error:
        raise RuntimeError(
            "Missing safetensors; run through run_baseline.py so requirements install first"
        ) from error

    c = cfg["checkpoint"]
    required = (
        "added_tokens.json",
        "chat_template.jinja",
        "config.json",
        "generation_config.json",
        "model.safetensors.index.json",
        "modeling_llavanext_for_embedding.py",
        "preprocessor_config.json",
        "processor_config.json",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
    )
    missing = [name for name in required if not (snapshot / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Incomplete BGE-VL snapshot; missing: {', '.join(missing)}")

    checks = {
        "config.json": str(c["config_sha256"]),
        str(c["weight_index"]): str(c["weight_index_sha256"]),
        str(c["remote_code"]): str(c["remote_code_sha256"]),
    }
    hashes: dict[str, str] = {}
    for name, expected in checks.items():
        actual = sha256_file(snapshot / name)
        if actual != expected:
            raise RuntimeError(
                f"Pinned BGE-VL file checksum mismatch for {name}: expected {expected}, got {actual}"
            )
        hashes[name] = actual

    index = json.loads((snapshot / str(c["weight_index"])).read_text(encoding="utf-8"))
    if not isinstance(index, dict) or not isinstance(index.get("weight_map"), dict):
        raise TypeError("Invalid BGE-VL safetensors index")
    total_size = int(index.get("metadata", {}).get("total_size", -1))
    if total_size != int(c["tensor_bytes"]):
        raise RuntimeError(f"Unexpected indexed tensor bytes: {total_size}")
    shards = sorted(set(index["weight_map"].values()))
    if len(shards) != int(c["num_shards"]):
        raise RuntimeError(f"Expected {c['num_shards']} weight shards, found {len(shards)}")
    for name in shards:
        path = snapshot / name
        if not path.is_file() or path.stat().st_size <= 0:
            raise FileNotFoundError(f"Missing or empty BGE-VL weight shard: {name}")
        expected_keys = {key for key, shard in index["weight_map"].items() if shard == name}
        try:
            with safe_open(path, framework="pt", device="cpu") as handle:
                actual_keys = set(handle.keys())
        except Exception as error:
            raise RuntimeError(f"Unreadable BGE-VL safetensors shard {name}: {error}") from error
        if actual_keys != expected_keys:
            raise RuntimeError(
                f"BGE-VL shard/index key mismatch for {name}: "
                f"expected {len(expected_keys)}, found {len(actual_keys)}"
            )

    return {
        "schema": MARKER_SCHEMA,
        "hf_repo_id": str(c["hf_repo_id"]),
        "hf_revision": str(c["hf_revision"]),
        "tensor_bytes": total_size,
        "weight_shards": shards,
        "validated_sha256": hashes,
    }


def remove_snapshot(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    cfg = load_yaml(resolve_path(args.config))
    c = cfg["checkpoint"]
    snapshot = resolve_path(str(c["local_snapshot"]))
    marker_path = resolve_path(str(c["prepared_marker"]))
    tracker = PhaseTracker("bge_vl_mllm_s1 checkpoint", total=2)

    with tracker.phase("Validate or download pinned Hugging Face snapshot"):
        if snapshot.is_dir() and not args.force:
            try:
                marker = validate_snapshot(snapshot, cfg)
                tracker.log(f"using valid snapshot: {rel(snapshot)}")
            except Exception as error:
                tracker.log(f"existing snapshot is invalid and will be replaced: {error}")
                remove_snapshot(snapshot)
                marker = None
        else:
            marker = None
            if snapshot.exists():
                remove_snapshot(snapshot)

        if marker is None:
            try:
                from huggingface_hub import snapshot_download
            except ImportError as error:
                raise RuntimeError(
                    "Missing huggingface_hub; run through run_baseline.py so requirements install first"
                ) from error
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            tracker.log(f"downloading {c['hf_repo_id']}@{c['hf_revision']} (~15.1 GB tensors)")
            snapshot_download(
                repo_id=str(c["hf_repo_id"]),
                revision=str(c["hf_revision"]),
                local_dir=str(snapshot),
                ignore_patterns=["*.md", "assets/*", "*.png"],
            )
            marker = validate_snapshot(snapshot, cfg)
            tracker.log(f"validated {len(marker['weight_shards'])} official weight shards")

    with tracker.phase("Write prepared marker"):
        marker.update(
            {
                "status": str(c["status"]),
                "snapshot": rel(snapshot),
                "source_repository": str(cfg["source"]["repository"]),
                "source_commit": str(cfg["source"]["commit"]),
            }
        )
        write_json(marker_path, marker)
        tracker.log(f"marker={rel(marker_path)}")

    tracker.finish()


if __name__ == "__main__":
    main()
