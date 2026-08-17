#!/usr/bin/env python3
"""Prepare external artifacts for the P10 Imagine-and-Seek reproduction.

No IP-CIR author checkpoint is claimed. This prepares:
  * the repository's pinned P5 LinCIR assets;
  * a pinned public MIGC source checkout + public MIGC_SD14.ckpt;
  * local HF snapshots for BLIP2, Qwen1.5-32B, and Stable Diffusion 1.5.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark_progress import PhaseTracker  # noqa: E402

DEFAULT_CONFIG = Path(__file__).resolve().parent / "config.yaml"
METHOD_ID = "imagine_seek_prepare"


def load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError(f"Expected YAML mapping: {path}")
    return data


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


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_lincir_preparer(cfg: dict[str, Any], force: bool) -> None:
    method_dir = resolve_path(str(cfg["base_retriever"]["method_dir"]))
    script = method_dir / "download_checkpoint.py"
    if not script.is_file():
        raise FileNotFoundError(f"Missing P5 preparer: {rel(script)}")
    command = [sys.executable, str(script)]
    if force:
        command.append("--force")
    subprocess.run(command, cwd=str(ROOT), check=True)


def tracked_dirty(checkout: Path) -> str:
    output = subprocess.check_output(
        ["git", "-C", str(checkout), "status", "--porcelain", "--untracked-files=no"],
        text=True,
    )
    return output.strip()


def prepare_migc_source(cfg: dict[str, Any], force: bool) -> Path:
    if shutil.which("git") is None:
        raise RuntimeError("System tool 'git' is required for the pinned MIGC checkout")
    c = cfg["migc"]
    checkout = resolve_path(str(c["local_checkout"]))
    expected = str(c["commit"])
    if force and checkout.exists():
        if tracked_dirty(checkout):
            raise RuntimeError(f"Refusing to delete dirty MIGC checkout: {rel(checkout)}")
        shutil.rmtree(checkout)
    if not checkout.exists():
        checkout.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", str(c["repository"]), str(checkout)], check=True)
    dirty = tracked_dirty(checkout)
    if dirty:
        raise RuntimeError(f"Pinned MIGC checkout has tracked modifications:\n{dirty}")
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
        raise RuntimeError(f"MIGC source commit mismatch: expected {expected}, got {actual}")
    for required in ("migc/migc_pipeline.py", "migc/migc_utils.py", "inference_single_image.py"):
        if not (checkout / required).is_file():
            raise FileNotFoundError(f"Pinned MIGC source missing {required}")
    return checkout


def prepare_migc_checkpoint(cfg: dict[str, Any], force: bool) -> dict[str, Any]:
    try:
        import gdown
    except ImportError as error:
        raise RuntimeError("Missing gdown; install method requirements first") from error

    c = cfg["migc"]
    path = resolve_path(str(c["checkpoint"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    marker_path = resolve_path(str(c["prepared_marker"]))
    old_marker: dict[str, Any] = {}
    if marker_path.is_file():
        try:
            old_marker = json.loads(marker_path.read_text(encoding="utf-8"))
        except Exception:
            old_marker = {}

    if path.is_file() and not force:
        actual = sha256_file(path)
        recorded = old_marker.get("migc_checkpoint", {}).get("sha256")
        if path.stat().st_size > 100_000_000 and (recorded is None or recorded == actual):
            print(f"[skip] MIGC checkpoint: {rel(path)}", flush=True)
            return {"path": rel(path), "sha256": actual, "size_bytes": path.stat().st_size}
        print("[warn] existing MIGC checkpoint failed validation; replacing", flush=True)
        path.unlink(missing_ok=True)

    temp = path.with_name(path.name + ".part")
    temp.unlink(missing_ok=True)
    print("[download] public MIGC_SD14.ckpt from upstream Google Drive", flush=True)
    result = gdown.download(id=str(c["google_drive_id"]), output=str(temp), quiet=False)
    if result is None or not temp.is_file():
        temp.unlink(missing_ok=True)
        raise RuntimeError("gdown failed to download MIGC_SD14.ckpt")
    if temp.stat().st_size <= 100_000_000:
        size = temp.stat().st_size
        temp.unlink(missing_ok=True)
        raise RuntimeError(f"MIGC checkpoint is unexpectedly small: {size} bytes")
    os.replace(temp, path)
    return {"path": rel(path), "sha256": sha256_file(path), "size_bytes": path.stat().st_size}


def snapshot_has_weights(path: Path) -> bool:
    if not (path / "config.json").is_file() and not (path / "model_index.json").is_file():
        return False
    patterns = ("*.safetensors", "*.bin")
    return any(any(path.rglob(pattern)) for pattern in patterns)


def prepare_hf_snapshot(*, repo_id: str, revision: str, local: Path, force: bool, label: str) -> dict[str, Any]:
    try:
        from huggingface_hub import HfApi, snapshot_download
    except ImportError as error:
        raise RuntimeError("Missing huggingface_hub") from error

    if local.is_dir() and snapshot_has_weights(local) and not force:
        meta_path = local / ".cpr_snapshot.json"
        if meta_path.is_file():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                if meta.get("repo_id") == repo_id:
                    print(f"[skip] {label}: {rel(local)}", flush=True)
                    return meta
            except Exception:
                pass

    if force and local.exists():
        shutil.rmtree(local)
    local.mkdir(parents=True, exist_ok=True)

    api = HfApi()
    info = api.model_info(repo_id, revision=revision)
    resolved_revision = str(info.sha)
    print(f"[download] {label}: {repo_id}@{resolved_revision}", flush=True)
    snapshot_download(
        repo_id=repo_id,
        revision=resolved_revision,
        local_dir=str(local),
        local_dir_use_symlinks=False,
        ignore_patterns=["*.h5", "*.msgpack", "*.onnx", "*.tflite", "*.ckpt", "*.bin"],
    )
    if not snapshot_has_weights(local):
        raise FileNotFoundError(f"Incomplete {label} snapshot: {rel(local)}")
    meta = {
        "repo_id": repo_id,
        "requested_revision": revision,
        "resolved_revision": resolved_revision,
        "path": rel(local),
    }
    write_json(local / ".cpr_snapshot.json", meta)
    return meta


def prepare_large_models(cfg: dict[str, Any], force: bool) -> dict[str, Any]:
    models: dict[str, Any] = {}
    for key, label in (
        ("captioner", "BLIP2 captioner"),
        ("layout_llm", "Qwen1.5-32B layout LLM"),
        ("stable_diffusion", "Stable Diffusion 1.5"),
    ):
        c = cfg[key]
        models[key] = prepare_hf_snapshot(
            repo_id=str(c["repo_id"]),
            revision=str(c.get("revision", "main")),
            local=resolve_path(str(c["local_snapshot"])),
            force=force,
            label=label,
        )
    return models


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare P10 Imagine-and-Seek reproduction assets")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--skip-large-models",
        action="store_true",
        help="Prepare LinCIR/MIGC only. Useful only with proxy.mode=precomputed or manual model snapshots.",
    )
    args = parser.parse_args()
    config_path = resolve_path(args.config)
    cfg = load_yaml(config_path)
    proxy_mode = str(cfg["proxy"]["mode"])
    need_generator = proxy_mode == "generate"
    skip_large = bool(args.skip_large_models) or not need_generator

    tracker = PhaseTracker(METHOD_ID, total=5 if need_generator else 2)

    with tracker.phase("Prepare pinned P5 LinCIR dependency"):
        run_lincir_preparer(cfg, args.force)

    if not need_generator:
        with tracker.phase("Validate precomputed-proxy mode"):
            manifest = resolve_path(str(cfg["proxy"]["manifest"]))
            print(f"precomputed proxy manifest expected at {rel(manifest)}", flush=True)
        tracker.finish()
        return

    with tracker.phase("Pin public MIGC source"):
        migc_source = prepare_migc_source(cfg, args.force)

    with tracker.phase("Prepare public MIGC checkpoint"):
        migc_ckpt = prepare_migc_checkpoint(cfg, args.force)

    with tracker.phase("Prepare BLIP2, Qwen1.5-32B, and SD1.5 snapshots"):
        if skip_large:
            models = {}
            missing = []
            for key in ("captioner", "layout_llm", "stable_diffusion"):
                local = resolve_path(str(cfg[key]["local_snapshot"]))
                if not snapshot_has_weights(local):
                    missing.append(rel(local))
                else:
                    meta_path = local / ".cpr_snapshot.json"
                    models[key] = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {"path": rel(local)}
            if missing:
                raise FileNotFoundError(
                    "Large model snapshots are missing while --skip-large-models was requested:\n  - "
                    + "\n  - ".join(missing)
                )
        else:
            models = prepare_large_models(cfg, args.force)

    with tracker.phase("Write P10 reproduction marker"):
        marker = resolve_path(str(cfg["migc"]["prepared_marker"]))
        payload = {
            "schema": 1,
            "method": str(cfg["method"]),
            "implementation_status": "REPRODUCED",
            "config": rel(config_path),
            "migc_source": {
                "repository": str(cfg["migc"]["repository"]),
                "commit": str(cfg["migc"]["commit"]),
                "checkout": rel(migc_source),
            },
            "migc_checkpoint": {
                **migc_ckpt,
                "status": str(cfg["migc"]["checkpoint_status"]),
            },
            "foundation_models": models,
            "reference_conditioning": str(cfg["migc"]["reference_conditioning"]),
        }
        write_json(marker, payload)
        print(f"[ok] prepared marker: {rel(marker)}", flush=True)

    tracker.finish()


if __name__ == "__main__":
    main()
