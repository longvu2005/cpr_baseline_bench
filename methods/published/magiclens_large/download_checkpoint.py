#!/usr/bin/env python3
"""Prepare official MagicLens Large inference artifacts for P6."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
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


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
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


def tracked_dirty(checkout: Path) -> str:
    output = subprocess.check_output(
        ["git", "-C", str(checkout), "status", "--porcelain", "--untracked-files=no"],
        text=True,
    )
    return output.strip()


def prepare_git_source(label: str, spec: dict[str, Any]) -> Path:
    if shutil.which("git") is None:
        raise RuntimeError("System tool 'git' is required for MagicLens source preparation")

    checkout = resolve_path(str(spec["local_checkout"]))
    expected = str(spec["commit"])
    repository = str(spec["repository"])

    if not checkout.exists():
        checkout.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--filter=blob:none", repository, str(checkout)], check=True)

    if not (checkout / ".git").exists():
        raise RuntimeError(f"{label} checkout is not a git repository: {rel(checkout)}")

    dirty = tracked_dirty(checkout)
    if dirty:
        raise RuntimeError(f"{label} source has tracked local modifications:\n{dirty}")

    actual = subprocess.check_output(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual != expected:
        subprocess.run(
            ["git", "-C", str(checkout), "fetch", "origin", expected],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(checkout), "checkout", "--detach", expected],
            check=True,
        )
        actual = subprocess.check_output(
            ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True
        ).strip()
    if actual != expected:
        raise RuntimeError(f"{label} commit mismatch: expected {expected}, got {actual}")

    print(f"[ok] {label}: {expected[:12]} -> {rel(checkout)}", flush=True)
    return checkout


def validate_source_layout(
    magiclens: Path,
    scenic: Path,
    openai_clip: Path,
) -> None:
    required = [
        magiclens / "model.py",
        magiclens / "layers.py",
        magiclens / "data_utils.py",
        scenic / "scenic/projects/baselines/clip/model.py",
        scenic / "scenic/projects/baselines/clip/layers.py",
        scenic / "scenic/projects/baselines/clip/tokenizer.py",
        openai_clip / "clip/simple_tokenizer.py",
        openai_clip / "clip/bpe_simple_vocab_16e6.txt.gz",
    ]
    missing = [rel(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Pinned source checkout is incomplete:\n  - " + "\n  - ".join(missing))


def checkpoint_basic_info(path: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    size = int(path.stat().st_size)
    minimum = int(cfg["checkpoint"]["min_size_bytes"])
    if size < minimum:
        raise ValueError(
            f"MagicLens checkpoint is suspiciously small: {size:,} bytes < {minimum:,}"
        )
    with path.open("rb") as handle:
        prefix = handle.read(2)
    if len(prefix) != 2 or prefix[0] != 0x80:
        raise ValueError("MagicLens checkpoint does not look like the official pickle container")
    return {"size": size, "sha256": sha256_file(path)}


def download_http(url: str, target: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "cpr-baseline-bench/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response, target.open("wb") as output:
        total_raw = response.headers.get("Content-Length")
        total = int(total_raw) if total_raw and total_raw.isdigit() else None
        downloaded = 0
        while True:
            chunk = response.read(8 * 1024 * 1024)
            if not chunk:
                break
            output.write(chunk)
            downloaded += len(chunk)
            if total:
                pct = 100.0 * downloaded / total
                print(
                    f"\r[download] {downloaded / 2**20:,.1f}/{total / 2**20:,.1f} MiB ({pct:5.1f}%)",
                    end="",
                    flush=True,
                )
        if total:
            print(flush=True)


def copy_with_cli(command: list[str], target: Path) -> bool:
    executable = command[0]
    if shutil.which(executable) is None:
        return False
    subprocess.run(command + [str(target)], check=True)
    return target.is_file()


def download_from_drive(folder_id: str, filename: str, target: Path) -> bool:
    try:
        import gdown
    except ImportError as error:
        raise RuntimeError(
            "Missing gdown. Run through run_baseline.py so method requirements are installed first."
        ) from error

    with tempfile.TemporaryDirectory(prefix="magiclens_drive_") as temp_dir:
        output_dir = Path(temp_dir)
        files = gdown.download_folder(
            id=folder_id,
            output=str(output_dir),
            quiet=False,
            use_cookies=False,
        )
        candidates = list(output_dir.rglob(filename))
        if not candidates and files:
            candidates = [Path(p) for p in files if Path(p).name == filename]
        if not candidates:
            return False
        shutil.copyfile(candidates[0], target)
        return True


def prepare_checkpoint(cfg: dict[str, Any], force: bool) -> dict[str, Any]:
    c = cfg["checkpoint"]
    path = resolve_path(str(c["path"]))
    marker_path = resolve_path(str(c["prepared_marker"]))
    path.parent.mkdir(parents=True, exist_ok=True)

    if force:
        path.unlink(missing_ok=True)
        marker_path.unlink(missing_ok=True)

    if path.is_file():
        try:
            info = checkpoint_basic_info(path, cfg)
            print(
                f"[skip] existing MagicLens Large checkpoint sha256={info['sha256']}",
                flush=True,
            )
            return info
        except Exception as error:
            print(f"[warn] replacing invalid checkpoint: {error}", flush=True)
            path.unlink(missing_ok=True)

    temp = path.with_name(path.name + ".part")
    temp.unlink(missing_ok=True)
    errors: list[str] = []

    try:
        print(f"[download] official GCS HTTPS: {c['gcs_https_url']}", flush=True)
        download_http(str(c["gcs_https_url"]), temp)
        info = checkpoint_basic_info(temp, cfg)
        os.replace(temp, path)
        return info
    except Exception as error:
        temp.unlink(missing_ok=True)
        errors.append(f"HTTPS: {type(error).__name__}: {error}")
        print(f"[warn] unauthenticated GCS download failed: {error}", flush=True)

    try:
        print(f"[download] authenticated gsutil: {c['gcs_uri']}", flush=True)
        if copy_with_cli(["gsutil", "cp", str(c["gcs_uri"])], temp):
            info = checkpoint_basic_info(temp, cfg)
            os.replace(temp, path)
            return info
    except Exception as error:
        temp.unlink(missing_ok=True)
        errors.append(f"gsutil: {type(error).__name__}: {error}")

    try:
        print(f"[download] authenticated gcloud storage: {c['gcs_uri']}", flush=True)
        if copy_with_cli(["gcloud", "storage", "cp", str(c["gcs_uri"])], temp):
            info = checkpoint_basic_info(temp, cfg)
            os.replace(temp, path)
            return info
    except Exception as error:
        temp.unlink(missing_ok=True)
        errors.append(f"gcloud: {type(error).__name__}: {error}")

    try:
        print("[download] official Google Drive folder fallback", flush=True)
        if download_from_drive(
            str(c["google_drive_folder_id"]), str(c["filename"]), temp
        ):
            info = checkpoint_basic_info(temp, cfg)
            os.replace(temp, path)
            return info
    except Exception as error:
        temp.unlink(missing_ok=True)
        errors.append(f"Google Drive: {type(error).__name__}: {error}")

    detail = "\n".join(f"  - {item}" for item in errors)
    raise RuntimeError(
        "Could not obtain the official MagicLens Large checkpoint automatically.\n"
        f"Place {c['filename']} at {rel(path)} or authenticate `gcloud auth login`/`gsutil`, "
        "then rerun the baseline.\nAttempts:\n"
        f"{detail}"
    )


def write_marker(
    cfg: dict[str, Any],
    checkouts: dict[str, Path],
    checkpoint_info: dict[str, Any],
) -> Path:
    marker_path = resolve_path(str(cfg["checkpoint"]["prepared_marker"]))
    clip_bpe = checkouts["openai_clip"] / "clip/bpe_simple_vocab_16e6.txt.gz"
    payload = {
        "schema": 1,
        "checkpoint": {
            "path": rel(resolve_path(str(cfg["checkpoint"]["path"]))),
            "status": str(cfg["checkpoint"]["status"]),
            "variant": str(cfg["checkpoint"]["variant"]),
            **checkpoint_info,
        },
        "source": {
            name: {
                "checkout": rel(path),
                "commit": str(cfg["source"][name]["commit"]),
                "repository": str(cfg["source"][name]["repository"]),
            }
            for name, path in checkouts.items()
        },
        "tokenizer_bpe": {
            "path": rel(clip_bpe),
            "sha256": sha256_file(clip_bpe),
        },
    }
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    marker_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[ok] prepared marker: {rel(marker_path)}", flush=True)
    return marker_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare official MagicLens Large artifacts")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    cfg = load_yaml(resolve_path(args.config))
    tracker = PhaseTracker("magiclens_large_prepare", total=3)

    tracker.advance("Pin official MagicLens, Scenic, and OpenAI CLIP sources")
    checkouts = {
        name: prepare_git_source(name, cfg["source"][name])
        for name in ("magiclens", "scenic", "openai_clip")
    }
    validate_source_layout(
        checkouts["magiclens"], checkouts["scenic"], checkouts["openai_clip"]
    )

    tracker.advance("Download/validate official MagicLens Large checkpoint")
    checkpoint_info = prepare_checkpoint(cfg, args.force)
    print(
        f"[ok] checkpoint size={checkpoint_info['size']:,} sha256={checkpoint_info['sha256']}",
        flush=True,
    )

    tracker.advance("Write reproducibility marker")
    write_marker(cfg, checkouts, checkpoint_info)
    tracker.finish()


if __name__ == "__main__":
    main()
