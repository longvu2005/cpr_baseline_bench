#!/usr/bin/env python3
"""Prepare every external artifact required by FAFA + SetMatch inference.

This stage runs after method requirements are installed. It downloads the
released FAFA checkpoint, pins the official source checkout, prepares the
OpenAI CLIP selector and torchvision person detector weights, and pre-warms the
exact LAVIS/Transformers runtime assets used to construct the official model.
Inference then runs with the same repository-local cache in offline mode.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark_progress import PhaseTracker, byte_progress  # noqa: E402
DEFAULT_CONFIG = Path(__file__).resolve().parent / "config.yaml"
GOOGLE_DRIVE_ID = "1Bf2Ia7zmxx5k3Dj-nRr3CLbAqc_zkM0y"
CLIP_B32_SHA256 = "40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af"
CLIP_B32_URL = (
    "https://openaipublic.azureedge.net/clip/models/"
    f"{CLIP_B32_SHA256}/ViT-B-32.pt"
)
DETECTOR_URL = (
    "https://download.pytorch.org/models/"
    "fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth"
)
DETECTOR_HASH_PREFIX = "dd69338a"


def resolve_path(value: str) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else ROOT / path).resolve()


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise TypeError(f"Expected YAML mapping: {path}")
    return data


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def download_with_hash(
    *,
    url: str,
    path: Path,
    force: bool,
    expected_sha256: str | None = None,
    expected_hash_prefix: str | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and path.stat().st_size > 0 and not force:
        actual = sha256_file(path)
        valid = True
        if expected_sha256 is not None:
            valid = actual == expected_sha256
        if expected_hash_prefix is not None:
            valid = valid and actual.startswith(expected_hash_prefix)
        if valid:
            print(f"[skip] {rel(path)} (checksum valid)")
            return
        print(f"[warn] {rel(path)} has an invalid checksum; replacing it")

    temp = path.with_suffix(path.suffix + ".part")
    temp.unlink(missing_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "cpr-baseline-bench/1.0"})
    digest = hashlib.sha256()
    print(f"[download] {rel(path)}")
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
        actual = digest.hexdigest()
        if expected_sha256 is not None and actual != expected_sha256:
            raise RuntimeError(
                f"Checksum mismatch for {rel(path)}: expected {expected_sha256}, got {actual}"
            )
        if expected_hash_prefix is not None and not actual.startswith(expected_hash_prefix):
            raise RuntimeError(
                f"Checksum prefix mismatch for {rel(path)}: "
                f"expected {expected_hash_prefix}, got {actual}"
            )
        os.replace(temp, path)
    except Exception:
        temp.unlink(missing_ok=True)
        raise
    print(f"[ok] {rel(path)}")


def download_fafa_checkpoint(path: Path, force: bool) -> None:
    try:
        import gdown
    except ImportError as error:
        raise RuntimeError(
            "Missing dependency 'gdown'. Run this method through run_baseline.py so "
            "requirements.txt is installed before checkpoint preparation."
        ) from error

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and path.stat().st_size > 0 and not force:
        print(f"[skip] {rel(path)} already exists")
        return

    temp = path.with_suffix(path.suffix + ".part")
    temp.unlink(missing_ok=True)
    print(f"[download] {rel(path)}")
    try:
        result = gdown.download(id=GOOGLE_DRIVE_ID, output=str(temp), quiet=False)
        if result is None or not temp.is_file() or temp.stat().st_size <= 0:
            raise RuntimeError("FAFA download completed without producing a valid file")
        os.replace(temp, path)
    except Exception:
        temp.unlink(missing_ok=True)
        raise
    print(f"[ok] {rel(path)}")


def tracked_dirty(checkout: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(checkout), "status", "--porcelain", "--untracked-files=no"],
        text=True,
    ).strip()


def prepare_source(cfg: dict[str, Any]) -> Path:
    source = cfg["source"]
    checkout = resolve_path(str(source["local_checkout"]))
    expected = str(source["commit"])
    if not checkout.exists():
        if not bool(source.get("auto_clone", True)):
            raise FileNotFoundError(f"Missing pinned official source: {rel(checkout)}")
        checkout.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", str(source["repository"]), str(checkout)], check=True)

    dirty = tracked_dirty(checkout)
    if dirty:
        raise RuntimeError(
            f"Pinned official source has tracked local modifications: {rel(checkout)}\n{dirty}"
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
        raise RuntimeError(f"Source commit mismatch: expected {expected}, got {actual}")

    fafa_dir = checkout / str(source.get("subdir", "FAFA_SynCPR"))
    if not fafa_dir.is_dir():
        raise FileNotFoundError(f"Missing official source subdir: {fafa_dir}")
    print(f"[ok] pinned official source {expected[:12]}")
    return fafa_dir


def configure_runtime_cache(cache_root: Path, *, offline: bool) -> None:
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ["TORCH_HOME"] = str(cache_root / "torch")
    os.environ["HF_HOME"] = str(cache_root / "huggingface")
    os.environ["XDG_CACHE_HOME"] = str(cache_root / "xdg")
    if offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    else:
        os.environ.pop("HF_HUB_OFFLINE", None)
        os.environ.pop("TRANSFORMERS_OFFLINE", None)


def runtime_cache_inventory(cache_root: Path) -> list[dict[str, Any]]:
    inventory: list[dict[str, Any]] = []
    if not cache_root.is_dir():
        return inventory
    for path in sorted(p for p in cache_root.rglob("*") if p.is_file()):
        inventory.append(
            {
                "path": str(path.relative_to(cache_root)),
                "size": int(path.stat().st_size),
            }
        )
    return inventory


def runtime_cache_matches(marker_data: dict[str, Any], cache_root: Path) -> bool:
    files = marker_data.get("files")
    if not isinstance(files, list) or not files:
        return False
    for item in files:
        if not isinstance(item, dict):
            return False
        rel_path = item.get("path")
        size = item.get("size")
        if not isinstance(rel_path, str) or not isinstance(size, int) or size < 0:
            return False
        path = cache_root / rel_path
        if not path.is_file() or path.stat().st_size != size:
            return False
    return True


def prewarm_fafa_runtime_assets(
    cfg: dict[str, Any], fafa_dir: Path, marker: Path, cache_root: Path, force: bool
) -> None:
    ckpt = cfg["checkpoint"]
    expected_core = {
        "source_commit": str(cfg["source"]["commit"]),
        "model_name": str(ckpt.get("model_name", "blip2_fafa_cpr")),
        "model_type": str(ckpt.get("model_type", "pretrain")),
        "cache_root": rel(cache_root),
    }
    if marker.is_file() and not force:
        try:
            current = json.loads(marker.read_text(encoding="utf-8"))
        except Exception:
            current = None
        core_matches = isinstance(current, dict) and all(
            current.get(key) == value for key, value in expected_core.items()
        )
        if core_matches and runtime_cache_matches(current, cache_root):
            print(f"[skip] {rel(marker)} and FAFA runtime cache inventory are valid")
            return

    configure_runtime_cache(cache_root, offline=False)
    src = fafa_dir / "src"
    if not src.is_dir():
        raise FileNotFoundError(src)
    sys.path.insert(0, str(src))
    try:
        from lavis.models import load_model_and_preprocess  # type: ignore

        print("[prepare] pre-warming official FAFA/LAVIS runtime assets on CPU")
        model, _, _ = load_model_and_preprocess(
            name=expected_core["model_name"],
            model_type=expected_core["model_type"],
            is_eval=True,
            device="cpu",
        )
        del model
        gc.collect()
    finally:
        try:
            sys.path.remove(str(src))
        except ValueError:
            pass

    inventory = runtime_cache_inventory(cache_root)
    if not inventory:
        raise RuntimeError("FAFA model construction succeeded but runtime cache is empty")
    marker_data = {**expected_core, "files": inventory}
    marker.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(marker_data, indent=2, sort_keys=True) + "\n"
    if not marker.is_file() or marker.read_text(encoding="utf-8") != payload:
        marker.write_text(payload, encoding="utf-8")
    print(f"[ok] {rel(marker)}")


def prepare(config_path: Path, force: bool) -> None:
    tracker = PhaseTracker("fafa_prepare", total=5)
    cfg = load_yaml(config_path)
    ckpt = cfg["checkpoint"]

    tracker.advance("Pin official FAFA source checkout")
    fafa_dir = prepare_source(cfg)

    tracker.advance("Prepare released FAFA checkpoint")
    download_fafa_checkpoint(resolve_path(str(ckpt["path"])), force)

    tracker.advance("Prepare CLIP query-selector checkpoint")
    selector = cfg["localization"]["query_selector"]
    if str(selector["model"]) != "ViT-B/32":
        raise ValueError("FAFA query selector currently supports only OpenAI CLIP ViT-B/32")
    download_with_hash(
        url=CLIP_B32_URL,
        path=resolve_path(str(selector["checkpoint"])),
        expected_sha256=CLIP_B32_SHA256,
        force=force,
    )

    tracker.advance("Prepare person-detector checkpoint")
    detector = cfg["localization"]["detector"]
    download_with_hash(
        url=DETECTOR_URL,
        path=resolve_path(str(detector["checkpoint"])),
        expected_hash_prefix=DETECTOR_HASH_PREFIX,
        force=force,
    )

    tracker.advance("Pre-warm FAFA/LAVIS runtime assets")
    cache_root = resolve_path(str(ckpt["cache_root"]))
    marker = resolve_path(str(ckpt["runtime_assets_marker"]))
    prewarm_fafa_runtime_assets(cfg, fafa_dir, marker, cache_root, force)
    print("[status] FAFA inference artifacts are ready", flush=True)
    tracker.finish()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare FAFA + SetMatch inference artifacts.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download public weights and re-warm official runtime assets.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prepare(resolve_path(args.config), args.force)


if __name__ == "__main__":
    main()
