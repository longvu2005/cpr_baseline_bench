"""Pinned Hugging Face detector assets, prepared explicitly before inference."""
from __future__ import annotations

import os
import urllib.request
import hashlib
from pathlib import Path

from benchmark_progress import byte_progress
from .cache import ROOT, atomic_json, read_json, resolve, sha256_file


def rel(path):
    path = Path(path).resolve()
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def download_stream(
    *,
    url: str,
    path: Path,
    force: bool,
    expected_sha256: str | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.is_file() and path.stat().st_size > 0 and not force:
        if expected_sha256 is None:
            print(f"[skip] {rel(path)} already exists", flush=True)
            return
        actual = sha256_file(path)
        if actual == expected_sha256:
            print(f"[skip] {rel(path)} (checksum valid)", flush=True)
            return
        print(f"[warn] checksum mismatch for {rel(path)}; replacing", flush=True)

    temp = path.with_suffix(path.suffix + ".part")
    temp.unlink(missing_ok=True)
    request = urllib.request.Request(
        url, headers={"User-Agent": "cpr-baseline-bench/1.0"}
    )
    digest = hashlib.sha256()
    print(f"[download] {rel(path)}", flush=True)

    try:
        with urllib.request.urlopen(request) as response, temp.open("wb") as handle:
            total_header = response.headers.get("Content-Length")
            total = int(total_header) if total_header else None
            with byte_progress(desc=f"Download {path.name}", total=total) as bar:
                while True:
                    chunk = response.read(8 * 1024 * 1024)
                    if not chunk:
                        break
                    handle.write(chunk)
                    digest.update(chunk)
                    bar.update(len(chunk))

        if temp.stat().st_size <= 0:
            raise RuntimeError(f"Downloaded empty file for {rel(path)}")

        actual = digest.hexdigest()
        if expected_sha256 is not None and actual != expected_sha256:
            raise RuntimeError(
                f"Checksum mismatch for {rel(path)}: expected {expected_sha256}, got {actual}"
            )
        os.replace(temp, path)
    except Exception:
        temp.unlink(missing_ok=True)
        raise

    print(f"[ok] {rel(path)}", flush=True)


def asset_identity(detector):
    return {key: detector[key] for key in
            ("backend", "model", "revision", "processor_revision", "runtime_versions")}


def require_snapshot(detector):
    marker = read_json(resolve(detector["runtime_assets_marker"]))
    root = resolve(detector["runtime_cache"])
    if not isinstance(marker, dict) or marker.get("identity") != asset_identity(detector):
        raise FileNotFoundError("Missing/stale HF detector assets; run python -m cpr_person.prepare")
    if not marker.get("files"):
        raise RuntimeError("HF detector asset inventory is empty")
    for item in marker["files"]:
        path = root / item["path"]
        if not path.is_file() or sha256_file(path) != item["sha256"]:
            raise RuntimeError(f"Changed/incomplete HF detector asset: {path}; rerun cpr_person.prepare")
    return marker


def prepare_detector(detector, force=False):
    from .runtime import check_runtime
    check_runtime(detector)
    if not force:
        try:
            require_snapshot(detector)
            return
        except (FileNotFoundError, RuntimeError):
            pass
    from huggingface_hub import snapshot_download
    root = resolve(detector["runtime_cache"])
    files = {}
    for revision in dict.fromkeys([detector["revision"], detector["processor_revision"]]):
        snapshot = Path(snapshot_download(repo_id=detector["model"], revision=revision,
            cache_dir=str(root), force_download=force,
            allow_patterns=["*.json", "vocab.txt", "model.safetensors"]))
        for path in snapshot.iterdir():
            if path.is_file():
                files[str(path.relative_to(root))] = sha256_file(path)
    if not resolve(detector["checkpoint"]).is_file():
        raise FileNotFoundError("Configured HF checkpoint differs from the downloaded snapshot")
    atomic_json(resolve(detector["runtime_assets_marker"]), {
        "identity": asset_identity(detector),
        "files": [{"path": name, "sha256": digest} for name, digest in sorted(files.items())],
    })


def load_detector(detector, device):
    from .runtime import check_runtime
    check_runtime(detector)
    require_snapshot(detector)
    # An inference worker can only consume previously prepared local assets.
    # Native FAFA's HF_HOME/TRANSFORMERS_CACHE never redirect this snapshot.
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
    common = {"cache_dir": str(resolve(detector["runtime_cache"])), "local_files_only": True}
    processor = AutoProcessor.from_pretrained(detector["model"],
        revision=detector["processor_revision"], **common)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(detector["model"],
        revision=detector["revision"], **common).to(device).eval()
    return model, processor
