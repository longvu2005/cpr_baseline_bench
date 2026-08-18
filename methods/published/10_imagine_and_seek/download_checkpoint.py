#!/usr/bin/env python3
"""Prepare external artifacts for P10 Imagine-and-Seek official-source CPR adapter."""

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


def tracked_dirty(checkout: Path) -> str:
    output = subprocess.check_output(
        ["git", "-C", str(checkout), "status", "--porcelain", "--untracked-files=no"],
        text=True,
    )
    return output.strip()


def prepare_author_source(cfg: dict[str, Any], force: bool) -> Path:
    if shutil.which("git") is None:
        raise RuntimeError("System tool 'git' is required to pin the released IP-CIR source")
    c = cfg["author_source"]
    checkout = resolve_path(str(c["local_checkout"]))
    expected = str(c["commit"])

    if force and checkout.exists():
        dirty = tracked_dirty(checkout)
        if dirty:
            raise RuntimeError(f"Refusing to replace dirty IP-CIR checkout: {rel(checkout)}\n{dirty}")
        shutil.rmtree(checkout)

    if not checkout.exists():
        checkout.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", str(c["repository"]), str(checkout)], check=True)

    dirty = tracked_dirty(checkout)
    if dirty:
        raise RuntimeError(f"Pinned IP-CIR checkout has tracked modifications:\n{dirty}")

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
        raise RuntimeError(f"IP-CIR source commit mismatch: expected {expected}, got {actual}")

    required = (
        "generate_proxy_migc_elite.py",
        "generate_layout.py",
        "MIGC/migc/migc_pipeline.py",
        "MIGC/migc/migc_utils.py",
        "MIGC/migc_gui_weights/v1-inference.yaml",
    )
    for value in required:
        if not (checkout / value).is_file():
            raise FileNotFoundError(f"Released IP-CIR source missing {value}")
    print(f"[ok] released IP-CIR source {expected[:12]}: {rel(checkout)}", flush=True)
    return checkout


def run_lincir_preparer(cfg: dict[str, Any], force: bool) -> None:
    method_dir = resolve_path(str(cfg["base_retriever"]["method_dir"]))
    script = method_dir / "download_checkpoint.py"
    if not script.is_file():
        raise FileNotFoundError(f"Missing P5 preparer: {rel(script)}")
    command = [sys.executable, str(script)]
    if force:
        command.append("--force")
    subprocess.run(command, cwd=str(ROOT), check=True)


def download_gdrive_file(*, file_id: str, path: Path, force: bool, min_size: int) -> dict[str, Any]:
    try:
        import gdown
    except ImportError as error:
        raise RuntimeError("Missing gdown; run through run_baseline.py so requirements install first") from error

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and not force and path.stat().st_size >= min_size:
        return {"path": rel(path), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}

    temp = path.with_name(path.name + ".part")
    temp.unlink(missing_ok=True)
    result = gdown.download(id=file_id, output=str(temp), quiet=False)
    if result is None or not temp.is_file():
        temp.unlink(missing_ok=True)
        raise RuntimeError(f"gdown failed for Google Drive id={file_id}")
    if temp.stat().st_size < min_size:
        size = temp.stat().st_size
        temp.unlink(missing_ok=True)
        raise RuntimeError(f"Downloaded file is unexpectedly small: {size} bytes")
    os.replace(temp, path)
    return {"path": rel(path), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}


def prepare_migc(cfg: dict[str, Any], force: bool) -> dict[str, Any]:
    c = cfg["migc"]
    path = resolve_path(str(c["checkpoint"]))
    print("[download] MIGC_SD14.ckpt", flush=True)
    info = download_gdrive_file(
        file_id=str(c["google_drive_id"]),
        path=path,
        force=force,
        min_size=100_000_000,
    )
    print(f"[ok] MIGC sha256={info['sha256']}", flush=True)
    return info


def prepare_elite(cfg: dict[str, Any], force: bool) -> dict[str, Any]:
    try:
        import gdown
    except ImportError as error:
        raise RuntimeError("Missing gdown") from error

    c = cfg["elite"]
    global_path = resolve_path(str(c["global_mapper"]))
    local_path = resolve_path(str(c["local_mapper"]))
    if (
        global_path.is_file()
        and local_path.is_file()
        and global_path.stat().st_size > 1_000_000
        and local_path.stat().st_size > 1_000_000
        and not force
    ):
        print(f"[skip] ELITE mappers: {rel(global_path)}, {rel(local_path)}", flush=True)
    else:
        download_dir = resolve_path(str(c["download_dir"]))
        if force and download_dir.exists():
            shutil.rmtree(download_dir)
        download_dir.mkdir(parents=True, exist_ok=True)
        print("[download] ELITE pretrained checkpoint folder", flush=True)
        result = gdown.download_folder(
            url=str(c["checkpoint_folder_url"]),
            output=str(download_dir),
            quiet=False,
            use_cookies=False,
        )
        if not result:
            raise RuntimeError(
                "Could not download the public ELITE checkpoint folder. "
                "Download global_mapper.pt and local_mapper.pt from the ELITE project and place them under "
                f"{rel(global_path.parent)}."
            )
        found_global = next(download_dir.rglob("global_mapper.pt"), None)
        found_local = next(download_dir.rglob("local_mapper.pt"), None)
        if found_global is None or found_local is None:
            raise FileNotFoundError(
                "ELITE checkpoint folder downloaded, but global_mapper.pt/local_mapper.pt were not found"
            )
        global_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(found_global, global_path)
        shutil.copy2(found_local, local_path)

    for path in (global_path, local_path):
        if not path.is_file() or path.stat().st_size <= 1_000_000:
            raise FileNotFoundError(f"Invalid ELITE mapper: {rel(path)}")
    return {
        "global_mapper": {"path": rel(global_path), "sha256": sha256_file(global_path), "size_bytes": global_path.stat().st_size},
        "local_mapper": {"path": rel(local_path), "sha256": sha256_file(local_path), "size_bytes": local_path.stat().st_size},
    }


def prepare_realistic_vision(cfg: dict[str, Any], force: bool) -> dict[str, Any]:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as error:
        raise RuntimeError("Missing huggingface_hub") from error

    c = cfg["realistic_vision"]
    path = resolve_path(str(c["path"]))
    expected = str(c["sha256"])
    if path.is_file() and not force and sha256_file(path) == expected:
        print(f"[skip] Realistic Vision: {rel(path)}", flush=True)
        return {"path": rel(path), "sha256": expected, "size_bytes": path.stat().st_size}

    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[download] {c['repo_id']}/{c['filename']}", flush=True)
    cached = Path(
        hf_hub_download(
            repo_id=str(c["repo_id"]),
            filename=str(c["filename"]),
            revision=str(c.get("revision", "main")),
        )
    )
    temp = path.with_name(path.name + ".part")
    temp.unlink(missing_ok=True)
    shutil.copyfile(cached, temp)
    actual = sha256_file(temp)
    if actual != expected:
        temp.unlink(missing_ok=True)
        raise RuntimeError(f"Realistic Vision checksum mismatch: expected {expected}, got {actual}")
    os.replace(temp, path)
    return {"path": rel(path), "sha256": actual, "size_bytes": path.stat().st_size}


def snapshot_has_weights(path: Path) -> bool:
    if not path.is_dir():
        return False
    return any(path.rglob("*.safetensors")) or any(path.rglob("*.bin"))


def prepare_snapshot(
    *, repo_id: str, revision: str, local: Path, force: bool, label: str, allow_patterns: list[str] | None = None
) -> dict[str, Any]:
    try:
        from huggingface_hub import HfApi, snapshot_download
    except ImportError as error:
        raise RuntimeError("Missing huggingface_hub") from error

    meta_path = local / ".cpr_snapshot.json"
    if local.is_dir() and snapshot_has_weights(local) and meta_path.is_file() and not force:
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("repo_id") == repo_id and meta.get("requested_revision") == revision:
                print(f"[skip] {label}: {rel(local)}", flush=True)
                return meta
        except Exception:
            pass

    if force and local.exists():
        shutil.rmtree(local)
    local.mkdir(parents=True, exist_ok=True)
    info = HfApi().model_info(repo_id, revision=revision)
    resolved_revision = str(info.sha)
    print(f"[download] {label}: {repo_id}@{resolved_revision}", flush=True)
    kwargs: dict[str, Any] = {
        "repo_id": repo_id,
        "revision": resolved_revision,
        "local_dir": str(local),
        "local_dir_use_symlinks": False,
    }
    if allow_patterns is not None:
        kwargs["allow_patterns"] = allow_patterns
    snapshot_download(**kwargs)
    if not snapshot_has_weights(local):
        raise FileNotFoundError(f"Incomplete {label} snapshot: {rel(local)}")
    meta = {
        "repo_id": repo_id,
        "requested_revision": revision,
        "resolved_revision": resolved_revision,
        "path": rel(local),
    }
    write_json(meta_path, meta)
    return meta


def prepare_foundation_models(cfg: dict[str, Any], force: bool) -> dict[str, Any]:
    models: dict[str, Any] = {}
    for key, label in (("captioner", "BLIP2 captioner"), ("layout_llm", "Qwen layout LLM")):
        c = cfg[key]
        models[key] = prepare_snapshot(
            repo_id=str(c["repo_id"]),
            revision=str(c.get("revision", "main")),
            local=resolve_path(str(c["local_snapshot"])),
            force=force,
            label=label,
        )

    c = cfg["sd15_components"]
    models["sd15_components"] = prepare_snapshot(
        repo_id=str(c["repo_id"]),
        revision=str(c.get("revision", "main")),
        local=resolve_path(str(c["local_snapshot"])),
        force=force,
        label="Stable Diffusion 1.5 text components",
        allow_patterns=[
            "text_encoder/*",
            "tokenizer/*",
            "scheduler/*",
            "feature_extractor/*",
            "model_index.json",
        ],
    )
    return models


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare P10 Imagine-and-Seek official-source CPR assets")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    config_path = resolve_path(args.config)
    cfg = load_yaml(config_path)
    tracker = PhaseTracker(METHOD_ID, total=7)

    with tracker.phase("Prepare pinned P5 LinCIR dependency"):
        run_lincir_preparer(cfg, args.force)

    with tracker.phase("Pin released Imagine-and-Seek source"):
        source = prepare_author_source(cfg, args.force)

    with tracker.phase("Prepare public MIGC checkpoint"):
        migc = prepare_migc(cfg, args.force)

    with tracker.phase("Prepare public ELITE global/local mappers"):
        elite = prepare_elite(cfg, args.force)

    with tracker.phase("Prepare released Realistic Vision generator"):
        realistic_vision = prepare_realistic_vision(cfg, args.force)

    with tracker.phase("Prepare BLIP2, Qwen and SD1.5 components"):
        foundation = prepare_foundation_models(cfg, args.force)

    with tracker.phase("Write reproducibility marker"):
        marker = resolve_path(str(cfg["migc"]["prepared_marker"]))
        payload = {
            "schema": 2,
            "method": str(cfg["method"]),
            "implementation_status": "OFFICIAL_SOURCE_ADAPTED",
            "config": rel(config_path),
            "author_source": {
                "repository": str(cfg["author_source"]["repository"]),
                "commit": str(cfg["author_source"]["commit"]),
                "checkout": rel(source),
            },
            "migc_checkpoint": migc,
            "elite": elite,
            "realistic_vision": realistic_vision,
            "foundation_models": foundation,
            "cpr_adaptation": {
                "query_mode": "direct_full_scene",
                "elite_reference_mask": str(cfg["proxy"]["reference_mask"]),
                "uses_gt_target_box": False,
                "uses_cpr_labels": False,
            },
        }
        write_json(marker, payload)
        print(f"[ok] prepared marker: {rel(marker)}", flush=True)

    tracker.finish()


if __name__ == "__main__":
    main()
