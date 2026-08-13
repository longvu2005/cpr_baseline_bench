#!/usr/bin/env python3
"""Prepare every external artifact required by Word4Per + SetMatch inference.

The final Word4Per Stage-2 checkpoint is a REPRODUCED artifact because the
pinned public ``old_project`` does not document a released final Stage-2
``best.pth`` download. This script validates that reproduced pair, pins the
official source checkout, and downloads/checks all public auxiliary weights so
``run.py`` never needs to download model weights during benchmark inference.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import urllib.request
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = Path(__file__).resolve().parent / "config.yaml"

CLIP_ASSETS = {
    "ViT-B/16": (
        "5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f",
        "ViT-B-16.pt",
    ),
    "ViT-B/32": (
        "40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af",
        "ViT-B-32.pt",
    ),
}
CLIP_BASE_URL = "https://openaipublic.azureedge.net/clip/models"
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


def load_training_yaml(path: Path) -> dict[str, Any]:
    # Match old_project/utils/iotools.py, which saves argparse configs with
    # yaml.dump() and reloads them with FullLoader (e.g. Python tuple tags).
    with path.open("r", encoding="utf-8") as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    if not isinstance(data, dict):
        raise TypeError(f"Expected Word4Per training YAML mapping: {path}")
    return data


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def validate_nonempty(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {rel(path)}")
    if path.stat().st_size <= 0:
        raise RuntimeError(f"Empty {label}: {rel(path)}")


def download_with_sha256(
    *,
    url: str,
    path: Path,
    expected_sha256: str | None = None,
    expected_hash_prefix: str | None = None,
    force: bool,
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
            while True:
                chunk = response.read(8 * 1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
                digest.update(chunk)
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


def prepare_clip(model_name: str, path: Path, force: bool) -> None:
    try:
        checksum, filename = CLIP_ASSETS[model_name]
    except KeyError as error:
        raise ValueError(f"Unsupported OpenAI CLIP asset: {model_name!r}") from error
    url = f"{CLIP_BASE_URL}/{checksum}/{filename}"
    download_with_sha256(
        url=url,
        path=path,
        expected_sha256=checksum,
        force=force,
    )


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

    subdir = checkout / str(source.get("subdir", "old_project"))
    if not subdir.is_dir():
        raise FileNotFoundError(f"Missing official source subdir: {subdir}")
    print(f"[ok] pinned official source {expected[:12]}")
    return subdir


def validate_stage2_recipe(data: dict[str, Any], source: Path) -> None:
    expected = {
        "dataset_name": "CUHK-PEDES",
        "loss_names": "sdm+id",
        "toword_loss": "text",
        "batch_size": 128,
        "num_epoch": 60,
    }
    problems: list[str] = []
    for key, value in expected.items():
        if data.get(key) != value:
            problems.append(f"{key}: expected {value!r}, got {data.get(key)!r}")
    optimizer = str(data.get("optimizer", "")).lower()
    if optimizer != "adamw":
        problems.append(f"optimizer: expected 'AdamW', got {data.get('optimizer')!r}")
    try:
        lr = float(data.get("lr"))
    except (TypeError, ValueError):
        lr = float("nan")
    if not (abs(lr - 1e-4) <= 1e-12):
        problems.append(f"lr: expected 0.0001, got {data.get('lr')!r}")
    if bool(data.get("MLM", False)):
        problems.append("MLM: expected false for the documented Stage-2 recipe")
    if problems:
        details = "\n".join(f"  - {item}" for item in problems)
        raise RuntimeError(
            f"Word4Per Stage-2 config does not match the pinned official recipe: {rel(source)}\n"
            f"{details}"
        )


def validate_reproduced_stage2(cfg: dict[str, Any]) -> tuple[Path, Path]:
    checkpoint = cfg.get("checkpoint")
    if not isinstance(checkpoint, dict):
        raise KeyError("config.yaml must define a checkpoint mapping")

    stage2 = resolve_path(str(checkpoint["stage2"]))
    stage2_config = resolve_path(str(checkpoint["stage2_config"]))
    missing: list[str] = []
    for path, label in ((stage2, "Stage-2 checkpoint"), (stage2_config, "Stage-2 config")):
        if not path.is_file() or path.stat().st_size <= 0:
            missing.append(f"{label}: {rel(path)}")
    if missing:
        details = "\n".join(f"  - {item}" for item in missing)
        raise SystemExit(
            "Word4Per final Stage-2 inference weights are not published as a documented "
            "official download in the pinned old_project. Reproduce Stage 2 on CUHK-PEDES "
            "using the authors' recipe and place the artifacts at:\n"
            f"{details}\n"
            "Do not train, tune, or select this checkpoint on the CPR benchmark."
        )

    stage2_data = load_training_yaml(stage2_config)
    validate_stage2_recipe(stage2_data, stage2_config)
    configured_backbone = str(checkpoint["base_clip_model"])
    trained_backbone = str(stage2_data.get("pretrain_choice", "")).strip()
    if not trained_backbone:
        raise KeyError(f"{rel(stage2_config)} must contain pretrain_choice")
    if trained_backbone != configured_backbone:
        raise RuntimeError(
            "Word4Per backbone mismatch: Stage-2 config was trained with "
            f"{trained_backbone!r}, but config.yaml declares {configured_backbone!r}."
        )

    print(f"[ok] {rel(stage2)}")
    print(f"[ok] {rel(stage2_config)}")
    return stage2, stage2_config


def prepare(config_path: Path, force: bool) -> None:
    cfg = load_yaml(config_path)
    validate_reproduced_stage2(cfg)
    prepare_source(cfg)

    ckpt = cfg["checkpoint"]
    prepare_clip(str(ckpt["base_clip_model"]), resolve_path(str(ckpt["base_clip"])), force)

    selector = cfg["localization"]["query_selector"]
    prepare_clip(str(selector["model"]), resolve_path(str(selector["checkpoint"])), force)

    detector = cfg["localization"]["detector"]
    download_with_sha256(
        url=DETECTOR_URL,
        path=resolve_path(str(detector["checkpoint"])),
        expected_hash_prefix=DETECTOR_HASH_PREFIX,
        force=force,
    )
    print("[status] Word4Per inference artifacts are ready")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Word4Per + SetMatch inference artifacts.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download public auxiliary weights and refresh the pinned source checkout.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prepare(resolve_path(args.config), args.force)


if __name__ == "__main__":
    main()
