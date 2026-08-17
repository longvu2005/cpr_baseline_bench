#!/usr/bin/env python3
"""Prepare P9 AdaFocal + SetMatch external assets."""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark_progress import PhaseTracker  # noqa: E402

DEFAULT_CONFIG = Path(__file__).resolve().parent / "config.yaml"


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


def download_url(url: str, path: Path, expected_sha256: str, force: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and not force:
        actual = sha256_file(path)
        if actual == expected_sha256:
            print(f"[skip] valid {rel(path)}", flush=True)
            return
        print(f"[warn] replacing bad checksum: {rel(path)}", flush=True)

    temp = path.with_name(path.name + ".part")
    temp.unlink(missing_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "cpr-baseline-bench/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=120) as response, temp.open("wb") as handle:
            total_raw = response.headers.get("Content-Length")
            total = int(total_raw) if total_raw and total_raw.isdigit() else 0
            done = 0
            while True:
                chunk = response.read(8 * 1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
                done += len(chunk)
                if total:
                    print(
                        f"\r  {done / 2**20:,.1f}/{total / 2**20:,.1f} MiB "
                        f"({100.0 * done / total:5.1f}%)",
                        end="",
                        flush=True,
                    )
        if total:
            print(flush=True)
        actual = sha256_file(temp)
        if actual != expected_sha256:
            raise RuntimeError(
                f"Checksum mismatch for {rel(path)}: expected {expected_sha256}, got {actual}"
            )
        os.replace(temp, path)
    except Exception:
        temp.unlink(missing_ok=True)
        raise
    print(f"[ok] {rel(path)}", flush=True)


def tracked_dirty(checkout: Path) -> str:
    out = subprocess.check_output(
        ["git", "-C", str(checkout), "status", "--porcelain", "--untracked-files=no"],
        text=True,
    )
    return "\n".join(line for line in out.splitlines() if line.strip())


def prepare_official_source(cfg: dict[str, Any]) -> Path:
    source = cfg["official_source"]
    checkout = resolve_path(str(source["local_checkout"]))
    expected = str(source["commit"])
    repository = str(source["repository"])

    if shutil.which("git") is None:
        raise RuntimeError("git is required to prepare the official OACIR source")

    if not checkout.exists():
        if not bool(source.get("auto_clone", True)):
            raise FileNotFoundError(f"Missing official source: {rel(checkout)}")
        checkout.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", repository, str(checkout)], check=True)

    if not (checkout / ".git").is_dir():
        raise RuntimeError(f"Not a git checkout: {rel(checkout)}")
    dirty = tracked_dirty(checkout)
    if dirty:
        raise RuntimeError(f"Official OACIR checkout has tracked local changes:\n{dirty}")

    actual = subprocess.check_output(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual != expected:
        subprocess.run(["git", "-C", str(checkout), "fetch", "origin", expected], check=True)
        subprocess.run(["git", "-C", str(checkout), "checkout", "--detach", expected], check=True)
        actual = subprocess.check_output(
            ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True
        ).strip()
    if actual != expected:
        raise RuntimeError(f"OACIR source mismatch: expected {expected}, got {actual}")

    required = [
        checkout / "data_utils.py",
        checkout / "lavis/models/blip2_models/blip2_qformer_oacir_adafocal.py",
        checkout / "lavis/configs/models/blip2/blip2_pretrain.yaml",
    ]
    missing = [rel(p) for p in required if not p.is_file()]
    if missing:
        raise FileNotFoundError("Incomplete OACIR checkout:\n  - " + "\n  - ".join(missing))
    print(f"[ok] OACIR source @ {actual[:12]} -> {rel(checkout)}", flush=True)
    return checkout


def prepare_adafocal_checkpoint(cfg: dict[str, Any], force: bool) -> None:
    from huggingface_hub import hf_hub_download

    model_cfg = cfg["adafocal"]
    path = resolve_path(str(model_cfg["checkpoint"]))
    expected = str(model_cfg["checkpoint_sha256"])
    if path.is_file() and not force and sha256_file(path) == expected:
        print(f"[skip] valid {rel(path)}", flush=True)
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    print(
        f"[download] {model_cfg['hf_repo']}:{model_cfg['hf_filename']} "
        f"@ {model_cfg['hf_revision']}",
        flush=True,
    )
    downloaded = Path(
        hf_hub_download(
            repo_id=str(model_cfg["hf_repo"]),
            filename=str(model_cfg["hf_filename"]),
            revision=str(model_cfg["hf_revision"]),
            local_dir=str(path.parent),
        )
    ).resolve()
    if downloaded != path:
        shutil.copy2(downloaded, path)

    actual = sha256_file(path)
    if actual != expected:
        path.unlink(missing_ok=True)
        raise RuntimeError(
            f"AdaFocal checkpoint checksum mismatch: expected {expected}, got {actual}"
        )
    print(f"[ok] {rel(path)} sha256={actual}", flush=True)


def run_step(command: list[str]) -> None:
    print(f"$ {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare P9 AdaFocal + SetMatch assets")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    config_path = resolve_path(args.config)
    cfg = load_yaml(config_path)
    tracker = PhaseTracker("p9_adafocal_prepare", total=4)

    tracker.advance("Prepare shared Grounding DINO assets")
    cmd = [sys.executable, "-u", "methods/simple/05_reid_set/download_checkpoint.py"]
    if args.force:
        cmd.append("--force")
    run_step(cmd)

    tracker.advance("Prepare OpenAI CLIP ViT-B/32 selector")
    selector = cfg["selector"]
    download_url(
        str(selector["checkpoint_url"]),
        resolve_path(str(selector["checkpoint"])),
        str(selector["checkpoint_sha256"]),
        args.force,
    )

    tracker.advance("Prepare pinned official OACIR source")
    prepare_official_source(cfg)

    tracker.advance("Download official AdaFocal scalar checkpoint")
    prepare_adafocal_checkpoint(cfg, args.force)
    tracker.finish()


if __name__ == "__main__":
    main()
