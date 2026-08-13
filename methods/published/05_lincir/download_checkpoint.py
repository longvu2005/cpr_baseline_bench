#!/usr/bin/env python3
"""Prepare all external artifacts required by LinCIR P5 inference."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch
import yaml

os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark_progress import PhaseTracker  # noqa: E402

DEFAULT_CONFIG = Path(__file__).resolve().parent / "config.yaml"


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


def is_generated_python_artifact(path: str) -> bool:
    normalized = path.strip().strip('"').replace("\\", "/")
    return normalized.lower().endswith((".pyc", ".pyo"))


def tracked_dirty(checkout: Path) -> str:
    output = subprocess.check_output(
        ["git", "-C", str(checkout), "status", "--porcelain", "--untracked-files=no"],
        text=True,
    )
    dirty: list[str] = []
    for line in output.splitlines():
        if not line.strip():
            continue
        if len(line) < 4:
            dirty.append(line)
            continue
        status_paths = line[3:].strip().split(" -> ")
        if status_paths and all(is_generated_python_artifact(x) for x in status_paths):
            continue
        dirty.append(line)
    return "\n".join(dirty).strip()


def prepare_source(cfg: dict[str, Any]) -> Path:
    if shutil.which("git") is None:
        raise RuntimeError("System tool 'git' is required to pin the official LinCIR source")
    source = cfg["source"]
    checkout = resolve_path(str(source["local_checkout"]))
    expected = str(source["commit"])
    if not checkout.exists():
        if not bool(source.get("auto_clone", True)):
            raise FileNotFoundError(f"Missing official source checkout: {rel(checkout)}")
        checkout.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", str(source["repository"]), str(checkout)], check=True)

    dirty = tracked_dirty(checkout)
    if dirty:
        raise RuntimeError(
            f"Pinned LinCIR source has tracked local modifications: {rel(checkout)}\n{dirty}"
        )
    actual = subprocess.check_output(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual != expected:
        subprocess.run(["git", "-C", str(checkout), "fetch", "--all", "--tags"], check=True)
        subprocess.run(["git", "-C", str(checkout), "checkout", "--detach", expected], check=True)
        actual = subprocess.check_output(
            ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True
        ).strip()
    if actual != expected:
        raise RuntimeError(f"LinCIR source commit mismatch: expected {expected}, got {actual}")
    for required in ("models.py", "encode_with_pseudo_tokens.py", "utils.py", "validate.py"):
        if not (checkout / required).is_file():
            raise FileNotFoundError(f"Pinned source missing {required}")
    print(f"[ok] official LinCIR source {expected[:12]}", flush=True)
    return checkout


def torch_load_full(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def validate_lincir_checkpoint(path: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    checkpoint_cfg = cfg["checkpoint"]
    if not path.is_file() or path.stat().st_size <= 0:
        raise FileNotFoundError(path)
    actual_size = int(path.stat().st_size)
    expected_size = int(checkpoint_cfg["size_bytes"])
    if actual_size != expected_size:
        raise ValueError(
            f"LinCIR checkpoint size mismatch: expected {expected_size}, got {actual_size}"
        )
    actual_hash = sha256_file(path)
    expected_hash = str(checkpoint_cfg["sha256"])
    if actual_hash != expected_hash:
        raise ValueError(
            f"LinCIR checkpoint checksum mismatch: expected {expected_hash}, got {actual_hash}"
        )

    raw = torch_load_full(path)
    if not isinstance(raw, dict) or "Phi" not in raw:
        raise KeyError("Official lincir_large.pt must contain state_dict key 'Phi'")
    state = raw["Phi"]
    if not isinstance(state, dict):
        raise TypeError("Checkpoint['Phi'] must be a state dict")
    expected_shapes = {
        "layers.0.weight": (3072, 768),
        "layers.0.bias": (3072,),
        "layers.3.weight": (3072, 3072),
        "layers.3.bias": (3072,),
        "layers.6.weight": (768, 3072),
        "layers.6.bias": (768,),
    }
    for key, expected in expected_shapes.items():
        tensor = state.get(key)
        if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != expected:
            raise ValueError(
                f"Unexpected LinCIR Phi tensor {key}: "
                f"shape={None if tensor is None else tuple(tensor.shape)}, expected={expected}"
            )
    info = {
        "sha256": actual_hash,
        "size": actual_size,
        "phi_tensors": len(state),
    }
    del raw, state
    gc.collect()
    return info


def download_lincir_checkpoint(cfg: dict[str, Any], force: bool) -> dict[str, Any]:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as error:
        raise RuntimeError(
            "Missing huggingface_hub. Run through run_baseline.py so requirements are installed first."
        ) from error

    c = cfg["checkpoint"]
    path = resolve_path(str(c["path"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and not force:
        try:
            info = validate_lincir_checkpoint(path, cfg)
            print(f"[skip] valid official LinCIR checkpoint: {rel(path)}", flush=True)
            return info
        except Exception as error:
            print(f"[warn] replacing invalid LinCIR checkpoint: {error}", flush=True)
            path.unlink(missing_ok=True)

    print(f"[download] official {c['hf_repo_id']}@{c['hf_revision']}/{c['hf_filename']}", flush=True)
    cached = Path(
        hf_hub_download(
            repo_id=str(c["hf_repo_id"]),
            filename=str(c["hf_filename"]),
            revision=str(c["hf_revision"]),
        )
    )
    temp = path.with_name(path.name + ".part")
    temp.unlink(missing_ok=True)
    try:
        shutil.copyfile(cached, temp)
        info = validate_lincir_checkpoint(temp, cfg)
        os.replace(temp, path)
    except Exception:
        temp.unlink(missing_ok=True)
        raise
    print(f"[ok] LinCIR checkpoint sha256={info['sha256']}", flush=True)
    return info


def prepare_clip_snapshot(cfg: dict[str, Any], force: bool) -> dict[str, Any]:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:
        raise RuntimeError("Missing huggingface_hub") from error

    b = cfg["backbone"]
    local = resolve_path(str(b["local_snapshot"]))
    model_file = local / str(b["model_file"])
    config_file = local / "config.json"
    expected_hash = str(b["model_sha256"])

    valid = (
        model_file.is_file()
        and config_file.is_file()
        and sha256_file(model_file) == expected_hash
    )
    if valid and not force:
        print(f"[skip] valid pinned CLIP ViT-L/14 snapshot: {rel(local)}", flush=True)
        return {
            "model_sha256": expected_hash,
            "model_size": int(model_file.stat().st_size),
        }

    if force and local.exists():
        shutil.rmtree(local)
    local.mkdir(parents=True, exist_ok=True)
    print(
        f"[download] {b['hf_repo_id']}@{b['hf_revision']} inference snapshot",
        flush=True,
    )
    snapshot_download(
        repo_id=str(b["hf_repo_id"]),
        revision=str(b["hf_revision"]),
        local_dir=str(local),
        allow_patterns=["config.json", "model.safetensors"],
    )
    if not model_file.is_file() or not config_file.is_file():
        raise FileNotFoundError("Pinned CLIP snapshot is incomplete")
    actual = sha256_file(model_file)
    if actual != expected_hash:
        raise RuntimeError(
            f"CLIP model.safetensors checksum mismatch: expected {expected_hash}, got {actual}"
        )
    return {"model_sha256": actual, "model_size": int(model_file.stat().st_size)}


def write_marker(cfg: dict[str, Any], source: Path, checkpoint_info: dict[str, Any], clip_info: dict[str, Any]) -> None:
    marker = resolve_path(str(cfg["checkpoint"]["prepared_marker"]))
    payload = {
        "schema": 1,
        "source_commit": str(cfg["source"]["commit"]),
        "source_checkout": rel(source),
        "checkpoint": {
            "path": rel(resolve_path(str(cfg["checkpoint"]["path"]))),
            **checkpoint_info,
            "hf_repo_id": str(cfg["checkpoint"]["hf_repo_id"]),
            "hf_revision": str(cfg["checkpoint"]["hf_revision"]),
        },
        "backbone": {
            "path": rel(resolve_path(str(cfg["backbone"]["local_snapshot"]))),
            **clip_info,
            "hf_repo_id": str(cfg["backbone"]["hf_repo_id"]),
            "hf_revision": str(cfg["backbone"]["hf_revision"]),
        },
    }
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[ok] prepared marker: {rel(marker)}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare official LinCIR artifacts")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config_path = resolve_path(args.config)
    cfg = load_yaml(config_path)

    tracker = PhaseTracker("lincir_prepare", total=4)
    tracker.advance("Pin official LinCIR source")
    source = prepare_source(cfg)

    tracker.advance("Download and validate official lincir_large.pt")
    checkpoint_info = download_lincir_checkpoint(cfg, args.force)

    tracker.advance("Prepare exact CLIP ViT-L/14 backbone snapshot")
    clip_info = prepare_clip_snapshot(cfg, args.force)

    tracker.advance("Write reproducibility marker")
    write_marker(cfg, source, checkpoint_info, clip_info)
    tracker.finish()


if __name__ == "__main__":
    main()
