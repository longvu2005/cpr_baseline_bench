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
import shutil
import subprocess
import sys
import urllib.request
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import yaml

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark_progress import PhaseTracker, byte_progress  # noqa: E402
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

STAGE2_CHECKPOINT_ENV = "WORD4PER_STAGE2_CHECKPOINT"
STAGE2_CONFIG_ENV = "WORD4PER_STAGE2_CONFIG"
KAGGLE_INPUT_ROOT = Path("/kaggle/input")


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


def kaggle_artifact_matches(filename: str) -> list[Path]:
    """Find a uniquely named reproduced artifact in shallow Kaggle mounts.

    Kaggle datasets are normally mounted as ``/kaggle/input/<dataset>/...``.
    Keep the search deliberately shallow so preparing Word4Per does not walk
    the large image dataset just to locate two model artifacts.
    """
    if not KAGGLE_INPUT_ROOT.is_dir():
        return []

    matches: set[Path] = set()
    direct = KAGGLE_INPUT_ROOT / filename
    if direct.is_file():
        matches.add(direct.resolve())

    for pattern in (f"*/{filename}", f"*/*/{filename}"):
        for path in KAGGLE_INPUT_ROOT.glob(pattern):
            if path.is_file():
                matches.add(path.resolve())
    return sorted(matches, key=str)


def materialize_reproduced_artifact(
    canonical: Path,
    *,
    env_var: str,
    label: str,
) -> Path:
    """Expose an externally reproduced artifact at the canonical repo path.

    The benchmark must not fabricate Word4Per Stage-2 weights. This helper
    only imports an artifact supplied by the user (explicit environment
    variable) or a uniquely named artifact already mounted by Kaggle.
    """
    if canonical.is_file():
        return canonical
    if canonical.exists() and not canonical.is_file():
        raise RuntimeError(f"Expected {label} to be a file: {rel(canonical)}")
    if canonical.is_symlink():
        canonical.unlink()

    source: Path | None = None
    configured = os.environ.get(env_var, "").strip()
    if configured:
        source = resolve_path(configured)
        validate_nonempty(source, f"{label} from ${env_var}")
    else:
        matches = kaggle_artifact_matches(canonical.name)
        if len(matches) > 1:
            options = "\n".join(f"  - {path}" for path in matches)
            raise RuntimeError(
                f"Found multiple Kaggle candidates for {label}:\n{options}\n"
                f"Set ${env_var} to the exact artifact that must be used."
            )
        if matches:
            source = matches[0]

    if source is None:
        return canonical

    canonical.parent.mkdir(parents=True, exist_ok=True)
    try:
        canonical.symlink_to(source)
        mode = "symlink"
    except OSError:
        shutil.copy2(source, canonical)
        mode = "copy"
    print(f"[import] {label}: {source} -> {rel(canonical)} ({mode})")
    return canonical


def validate_stage2_checkpoint_structure(path: Path) -> None:
    """Reject Stage-1 or unrelated checkpoints before expensive inference."""
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    except Exception as error:
        raise RuntimeError(
            f"Could not safely read reproduced Word4Per Stage-2 checkpoint: {rel(path)}"
        ) from error

    if not isinstance(checkpoint, Mapping):
        raise RuntimeError(
            f"Invalid Word4Per Stage-2 checkpoint {rel(path)}: expected a mapping"
        )
    for key in ("model", "img2text"):
        state = checkpoint.get(key)
        if not isinstance(state, Mapping) or not state:
            raise RuntimeError(
                f"Invalid Word4Per Stage-2 checkpoint {rel(path)}: "
                f"missing non-empty {key!r} state_dict"
            )
        if not any(isinstance(value, torch.Tensor) for value in state.values()):
            raise RuntimeError(
                f"Invalid Word4Per Stage-2 checkpoint {rel(path)}: "
                f"{key!r} contains no tensors"
            )
    print(f"[ok] {rel(path)} contains Word4Per model + img2text states")


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
    stage2 = materialize_reproduced_artifact(
        stage2,
        env_var=STAGE2_CHECKPOINT_ENV,
        label="Word4Per Stage-2 checkpoint",
    )
    stage2_config = materialize_reproduced_artifact(
        stage2_config,
        env_var=STAGE2_CONFIG_ENV,
        label="Word4Per Stage-2 config",
    )

    missing: list[str] = []
    for path, label in ((stage2, "Stage-2 checkpoint"), (stage2_config, "Stage-2 config")):
        if not path.is_file() or path.stat().st_size <= 0:
            missing.append(f"{label}: {rel(path)}")
    if missing:
        details = "\n".join(f"  - {item}" for item in missing)
        raise SystemExit(
            "Word4Per final Stage-2 inference weights are not published as a documented "
            "official download in the pinned old_project. Do not substitute the published "
            "Stage-1 checkpoint: Word4Per inference needs the learned Stage-2 img2text/TINet.\n\n"
            "Reproduce Stage 2 on CUHK-PEDES using the authors' recipe, then either place "
            "the artifacts at the canonical paths below, mount them as a Kaggle dataset "
            "with these filenames, or point the two environment variables at them:\n"
            f"  export {STAGE2_CHECKPOINT_ENV}=/path/to/best.pth\n"
            f"  export {STAGE2_CONFIG_ENV}=/path/to/configs.yaml\n\n"
            "Expected canonical paths:\n"
            f"{details}\n"
            "Do not train, tune, or select this checkpoint on the CPR benchmark."
        )

    validate_stage2_checkpoint_structure(stage2)
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
    tracker = PhaseTracker("word4per_prepare", total=5)

    tracker.advance("Validate reproduced Word4Per Stage-2 artifacts")
    cfg = load_yaml(config_path)
    validate_reproduced_stage2(cfg)

    tracker.advance("Pin official Word4Per source checkout")
    prepare_source(cfg)

    tracker.advance("Prepare Word4Per base CLIP checkpoint")
    ckpt = cfg["checkpoint"]
    prepare_clip(str(ckpt["base_clip_model"]), resolve_path(str(ckpt["base_clip"])), force)

    tracker.advance("Prepare CLIP query-selector checkpoint")
    selector = cfg["localization"]["query_selector"]
    prepare_clip(str(selector["model"]), resolve_path(str(selector["checkpoint"])), force)

    tracker.advance("Prepare person-detector checkpoint")
    detector = cfg["localization"]["detector"]
    download_with_sha256(
        url=DETECTOR_URL,
        path=resolve_path(str(detector["checkpoint"])),
        expected_hash_prefix=DETECTOR_HASH_PREFIX,
        force=force,
    )
    print("[status] Word4Per inference artifacts are ready", flush=True)
    tracker.finish()


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
