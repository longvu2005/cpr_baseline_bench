#!/usr/bin/env python3
"""Prepare official Pic2Word inference artifacts for the CPR benchmark."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import subprocess
import sys
import shutil
import urllib.request
from pathlib import Path
from typing import Any

import torch
import yaml

os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark_progress import PhaseTracker, byte_progress  # noqa: E402

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
        raise RuntimeError("System tool 'git' is required to pin the official Pic2Word source checkout")
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
            f"Pinned Pic2Word source has tracked local modifications: {rel(checkout)}\n{dirty}"
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
        raise RuntimeError(f"Pic2Word source commit mismatch: expected {expected}, got {actual}")
    for required in ("model/clip.py", "model/model.py", "third_party/open_clip/simple_tokenizer.py"):
        if not (checkout / required).is_file():
            raise FileNotFoundError(f"Pinned source missing {required}")
    print(f"[ok] official Pic2Word source {expected[:12]}", flush=True)
    return checkout


def strip_module_prefix(state: dict[str, Any]) -> dict[str, Any]:
    if state and next(iter(state)).startswith("module."):
        return {key[len("module.") :]: value for key, value in state.items()}
    return state


def torch_load_full(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def validate_pic2word_checkpoint(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size <= 0:
        raise FileNotFoundError(path)

    checkpoint = torch_load_full(path)
    if not isinstance(checkpoint, dict):
        raise TypeError("Pic2Word checkpoint must be a dict")
    if "state_dict_img2text" not in checkpoint:
        raise KeyError("Pic2Word checkpoint is missing state_dict_img2text")

    mapper_raw = checkpoint["state_dict_img2text"]
    if not isinstance(mapper_raw, dict):
        raise TypeError("Invalid Pic2Word state_dict_img2text")
    mapper = strip_module_prefix(mapper_raw)

    # The original Google Drive artifact is a full training checkpoint, while
    # the pinned fallback mirror contains only the trained IM2TEXT mapper.
    # Both are valid for inference because the official training code optimizes
    # img2text only; the CLIP backbone is loaded separately from OpenAI weights.
    state_raw = checkpoint.get("state_dict")
    state: dict[str, Any] = {}
    if state_raw is not None:
        if not isinstance(state_raw, dict):
            raise TypeError("Invalid Pic2Word state_dict")
        state = strip_module_prefix(state_raw)
        expected_model_shapes = {
            "visual.conv1.weight": (1024, 3, 14, 14),
            "text_projection": (768, 768),
            "token_embedding.weight": (49408, 768),
        }
        for key, expected in expected_model_shapes.items():
            tensor = state.get(key)
            if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != expected:
                raise ValueError(
                    "Pic2Word checkpoint does not match official ViT-L/14 config: "
                    f"{key} shape={None if tensor is None else tuple(tensor.shape)}, "
                    f"expected={expected}"
                )

    expected_mapper_shapes = {
        "layers.0.0.weight": (512, 768),
        "layers.1.0.weight": (512, 512),
        "fc_out.weight": (768, 512),
    }
    for key, expected in expected_mapper_shapes.items():
        tensor = mapper.get(key)
        if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != expected:
            raise ValueError(
                f"Unexpected official IM2TEXT structure: {key} "
                f"shape={None if tensor is None else tuple(tensor.shape)}, expected={expected}"
            )

    info = {
        "sha256": sha256_file(path),
        "size": int(path.stat().st_size),
        "epoch": checkpoint.get("epoch"),
        "artifact_kind": "full_checkpoint" if state else "img2text_only",
        "num_model_tensors": sum(isinstance(x, torch.Tensor) for x in state.values()),
        "num_img2text_tensors": sum(isinstance(x, torch.Tensor) for x in mapper.values()),
    }
    del checkpoint, state, mapper
    gc.collect()
    return info


def download_pic2word(
    path: Path,
    checkpoint_cfg: dict[str, Any],
    force: bool,
) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and not force:
        try:
            info = validate_pic2word_checkpoint(path)
            info["download_source"] = "existing"
            print(f"[skip] valid Pic2Word checkpoint: {rel(path)}", flush=True)
            return info
        except Exception as error:
            print(f"[warn] replacing invalid Pic2Word checkpoint: {error}", flush=True)
            path.unlink(missing_ok=True)

    temp = path.with_name(path.name + ".part")
    temp.unlink(missing_ok=True)

    drive_error: Exception | None = None
    drive_id = str(checkpoint_cfg["google_drive_id"])
    print(f"[download] official Pic2Word checkpoint -> {rel(path)}", flush=True)
    try:
        import gdown

        result = gdown.download(id=drive_id, output=str(temp), quiet=False)
        if result is None or not temp.is_file() or temp.stat().st_size <= 0:
            raise RuntimeError("Google Drive download did not produce a valid file")
        info = validate_pic2word_checkpoint(temp)
        info["download_source"] = "official_google_drive"
        info["source_url"] = str(checkpoint_cfg["source_url"])
        os.replace(temp, path)
        print(f"[ok] Pic2Word checkpoint sha256={info['sha256']}", flush=True)
        return info
    except Exception as error:
        drive_error = error
        temp.unlink(missing_ok=True)
        print(
            f"[warn] official Google Drive download failed: "
            f"{type(error).__name__}: {error}",
            flush=True,
        )

    fallback = checkpoint_cfg.get("fallback")
    if not isinstance(fallback, dict):
        raise RuntimeError(
            "Official Pic2Word Google Drive download failed and no fallback is configured"
        ) from drive_error

    print(
        "[download] Google Drive unavailable; using pinned Pic2Word mapper mirror "
        f"{fallback['repo_id']}@{str(fallback['revision'])[:12]}",
        flush=True,
    )
    try:
        from huggingface_hub import hf_hub_download

        cached = Path(
            hf_hub_download(
                repo_id=str(fallback["repo_id"]),
                filename=str(fallback["filename"]),
                revision=str(fallback["revision"]),
            )
        )
        if not cached.is_file() or cached.stat().st_size <= 0:
            raise RuntimeError(f"Hugging Face download produced no file: {cached}")
        shutil.copyfile(cached, temp)
        info = validate_pic2word_checkpoint(temp)
        if info["artifact_kind"] != "img2text_only":
            raise RuntimeError(
                "Pinned fallback must be the mapper-only Pic2Word artifact"
            )
        info["download_source"] = "pinned_huggingface_mapper_mirror"
        info["source_url"] = str(fallback["source_url"])
        os.replace(temp, path)
    except Exception as fallback_error:
        temp.unlink(missing_ok=True)
        raise RuntimeError(
            "Unable to prepare Pic2Word checkpoint: official Google Drive failed "
            "and the pinned Hugging Face fallback also failed. "
            f"Google Drive error: {type(drive_error).__name__}: {drive_error}"
        ) from fallback_error

    print(
        f"[ok] Pic2Word mapper fallback sha256={info['sha256']} "
        f"size={info['size']} bytes",
        flush=True,
    )
    return info


def download_clip(path: Path, expected_sha256: str, force: bool) -> dict[str, Any]:
    url = (
        "https://openaipublic.azureedge.net/clip/models/"
        f"{expected_sha256}/ViT-L-14.pt"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and not force and sha256_file(path) == expected_sha256:
        print(f"[skip] valid OpenAI CLIP ViT-L/14: {rel(path)}", flush=True)
        return {"sha256": expected_sha256, "size": int(path.stat().st_size)}

    temp = path.with_name(path.name + ".part")
    temp.unlink(missing_ok=True)
    digest = hashlib.sha256()
    request = urllib.request.Request(url, headers={"User-Agent": "cpr-baseline-bench/1.0"})
    print(f"[download] OpenAI CLIP ViT-L/14 -> {rel(path)}", flush=True)
    try:
        with urllib.request.urlopen(request) as response, temp.open("wb") as handle:
            total_header = response.headers.get("Content-Length")
            total = int(total_header) if total_header else None
            with byte_progress(desc="Download ViT-L-14.pt", total=total) as bar:
                while True:
                    chunk = response.read(8 * 1024 * 1024)
                    if not chunk:
                        break
                    handle.write(chunk)
                    digest.update(chunk)
                    bar.update(len(chunk))
        actual = digest.hexdigest()
        if actual != expected_sha256:
            raise RuntimeError(
                f"CLIP checksum mismatch: expected {expected_sha256}, got {actual}"
            )
        os.replace(temp, path)
    except Exception:
        temp.unlink(missing_ok=True)
        raise
    return {"sha256": expected_sha256, "size": int(path.stat().st_size)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare official Pic2Word artifacts")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    config_path = resolve_path(args.config)
    cfg = load_yaml(config_path)
    tracker = PhaseTracker("pic2word_prepare", total=4)

    tracker.advance("Pin official Pic2Word source")
    source_root = prepare_source(cfg)

    tracker.advance("Download/validate official Pic2Word checkpoint")
    checkpoint_path = resolve_path(str(cfg["checkpoint"]["path"]))
    checkpoint_info = download_pic2word(
        checkpoint_path,
        cfg["checkpoint"],
        bool(args.force),
    )

    tracker.advance("Prepare OpenAI CLIP ViT-L/14")
    clip_path = resolve_path(str(cfg["model"]["openai_clip_checkpoint"]))
    clip_info = download_clip(
        clip_path,
        str(cfg["model"]["openai_clip_sha256"]),
        bool(args.force),
    )

    tracker.advance("Write prepared marker")
    marker = resolve_path(str(cfg["checkpoint"]["prepared_marker"]))
    marker.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "method": str(cfg["method"]),
        "source": {
            "repository": str(cfg["source"]["repository"]),
            "commit": str(cfg["source"]["commit"]),
            "checkout": rel(source_root),
        },
        "checkpoint": {
            "path": rel(checkpoint_path),
            "official_source_url": str(cfg["checkpoint"]["source_url"]),
            "status": str(cfg["checkpoint"]["status"]),
            **checkpoint_info,
        },
        "openai_clip": {"path": rel(clip_path), **clip_info},
    }
    marker.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[ok] prepared marker: {rel(marker)}", flush=True)
    tracker.finish()


if __name__ == "__main__":
    main()
