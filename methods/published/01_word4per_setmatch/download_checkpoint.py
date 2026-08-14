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
METHOD_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = METHOD_DIR / "config.yaml"

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
STAGE1_CHECKPOINT_ENV = "WORD4PER_STAGE1_CHECKPOINT"
REPRO_DATA_ROOT_ENV = "WORD4PER_REPRO_DATA_ROOT"
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
    """Find a named artifact in standard and owner-qualified Kaggle mounts."""
    if not KAGGLE_INPUT_ROOT.is_dir():
        return []

    matches: set[Path] = set()
    direct = KAGGLE_INPUT_ROOT / filename
    if direct.is_file():
        matches.add(direct.resolve())

    # Support both /kaggle/input/<dataset>/... and
    # /kaggle/input/datasets/<owner>/<dataset>/..., with a few wrapper dirs.
    for depth in range(1, 7):
        pattern = "/".join(["*"] * depth + [filename])
        for path in KAGGLE_INPUT_ROOT.glob(pattern):
            if path.is_file():
                matches.add(path.resolve())
    return sorted(matches, key=str)


def kaggle_stage1_matches() -> list[Path]:
    candidates: set[Path] = set(kaggle_artifact_matches("stage1_model_vitb.pth"))

    # The authors' Stage-1 output may be mounted as generic best.pth.
    # Only consider it when the path clearly identifies Word4Per Stage-1.
    for path in kaggle_artifact_matches("best.pth"):
        lowered = str(path).lower()
        if "word4per" in lowered and ("stage1" in lowered or "stage-1" in lowered):
            candidates.add(path)

    valid: list[Path] = []
    for path in sorted(candidates, key=str):
        try:
            validate_stage1_checkpoint_structure(path)
        except Exception:
            continue
        valid.append(path)
    return valid


def kaggle_repro_data_root_matches() -> list[Path]:
    """Find roots with CUHK-PEDES training data plus ITCPR annotations."""
    if not KAGGLE_INPUT_ROOT.is_dir():
        return []

    matches: set[Path] = set()
    for depth in range(1, 7):
        pattern = "/".join(["*"] * depth + ["CUHK-PEDES", "reid_raw.json"])
        for annotation in KAGGLE_INPUT_ROOT.glob(pattern):
            if not annotation.is_file():
                continue
            root = annotation.parent.parent
            if (
                (root / "CUHK-PEDES" / "imgs").is_dir()
                and (root / "query.json").is_file()
                and (root / "gallery.json").is_file()
            ):
                matches.add(root.resolve())
    return sorted(matches, key=str)


def kaggle_mount_summary(limit: int = 30) -> str:
    if not KAGGLE_INPUT_ROOT.is_dir():
        return "  <no /kaggle/input directory>"
    mounts = sorted(
        (path.resolve() for path in KAGGLE_INPUT_ROOT.iterdir() if path.exists()),
        key=str,
    )
    if not mounts:
        return "  <no mounted Kaggle inputs>"
    lines = [f"  - {path}" for path in mounts[:limit]]
    if len(mounts) > limit:
        lines.append(f"  ... and {len(mounts) - limit} more")
    return "\n".join(lines)


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


def validate_stage1_checkpoint_structure(path: Path) -> None:
    """Accept an authors Stage-1 checkpoint and reject Stage-2/TINet."""
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    except Exception as error:
        raise RuntimeError(f"Could not read Word4Per Stage-1 checkpoint: {path}") from error

    if not isinstance(checkpoint, Mapping):
        raise RuntimeError(f"Invalid Word4Per Stage-1 checkpoint: {path}")
    model_state = checkpoint.get("model")
    if not isinstance(model_state, Mapping) or not model_state:
        raise RuntimeError(f"Stage-1 checkpoint has no non-empty 'model' state: {path}")
    if not any(isinstance(value, torch.Tensor) for value in model_state.values()):
        raise RuntimeError(f"Stage-1 checkpoint 'model' contains no tensors: {path}")
    if "img2text" in checkpoint:
        raise RuntimeError(f"Expected Stage-1 but found Stage-2 img2text state: {path}")


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


def try_reproduce_stage2(config_path: Path) -> bool:
    """Auto-resolve official prerequisites, then run the pinned Stage-2 recipe."""
    stage1_value = os.environ.get(STAGE1_CHECKPOINT_ENV, "").strip()
    data_root_value = os.environ.get(REPRO_DATA_ROOT_ENV, "").strip()

    if stage1_value:
        stage1 = resolve_path(stage1_value)
        validate_stage1_checkpoint_structure(stage1)
        print(f"[explicit] Word4Per Stage-1: {stage1}", flush=True)
    else:
        stage1_matches = kaggle_stage1_matches()
        if len(stage1_matches) > 1:
            options = "\n".join(f"  - {path}" for path in stage1_matches)
            raise RuntimeError(
                "Multiple valid Word4Per Stage-1 checkpoints were found:\n"
                f"{options}\n"
                f"Set ${STAGE1_CHECKPOINT_ENV} to the exact official checkpoint."
            )
        stage1 = stage1_matches[0] if stage1_matches else None
        if stage1 is not None:
            print(f"[auto] Word4Per Stage-1: {stage1}", flush=True)

    if data_root_value:
        data_root = resolve_path(data_root_value)
        if not data_root.is_dir():
            raise FileNotFoundError(
                f"${REPRO_DATA_ROOT_ENV} is not a directory: {data_root}"
            )
        print(f"[explicit] Word4Per reproduction data root: {data_root}", flush=True)
    else:
        data_matches = kaggle_repro_data_root_matches()
        if len(data_matches) > 1:
            options = "\n".join(f"  - {path}" for path in data_matches)
            raise RuntimeError(
                "Multiple valid Word4Per reproduction data roots were found:\n"
                f"{options}\n"
                f"Set ${REPRO_DATA_ROOT_ENV} to the exact root."
            )
        data_root = data_matches[0] if data_matches else None
        if data_root is not None:
            print(f"[auto] Word4Per reproduction data root: {data_root}", flush=True)

    missing: list[str] = []
    if stage1 is None:
        missing.append(
            "official Word4Per Stage-1 checkpoint "
            "(stage1_model_vitb.pth or Word4Per/Stage1 best.pth)"
        )
    if data_root is None:
        missing.append(
            "data root containing CUHK-PEDES/imgs, CUHK-PEDES/reid_raw.json, "
            "query.json and gallery.json"
        )
    if missing:
        details = "\n".join(f"  - {item}" for item in missing)
        raise RuntimeError(
            "Word4Per Stage-2 cannot be reproduced because required official "
            f"external inputs are missing:\n{details}\n\n"
            "Detected Kaggle mounts:\n"
            f"{kaggle_mount_summary()}\n\n"
            "Mount the missing official artifacts or set the two WORD4PER_* "
            "environment variables explicitly."
        )

    reproducer = METHOD_DIR / "reproduce_stage2.py"
    if not reproducer.is_file():
        raise FileNotFoundError(
            f"Missing Word4Per Stage-2 reproduction helper: {rel(reproducer)}"
        )

    command = [
        sys.executable,
        "-u",
        str(reproducer.relative_to(ROOT)),
        "--config",
        str(config_path),
        "--stage1-checkpoint",
        str(stage1),
        "--data-root",
        str(data_root),
    ]
    print(
        "[reproduce] Stage-2 artifacts are absent; running the pinned official recipe",
        flush=True,
    )
    print("$ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)
    return True


def validate_reproduced_stage2(
    cfg: dict[str, Any], *, config_path: Path = DEFAULT_CONFIG
) -> tuple[Path, Path]:
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

    if missing and try_reproduce_stage2(config_path):
        missing = []
        for path, label in (
            (stage2, "Stage-2 checkpoint"),
            (stage2_config, "Stage-2 config"),
        ):
            if not path.is_file() or path.stat().st_size <= 0:
                missing.append(f"{label}: {rel(path)}")

    if missing:
        details = "\n".join(f"  - {item}" for item in missing)
        message = (
            "Word4Per final Stage-2 inference weights are not published as a documented "
            "official download in the pinned old_project. Do not substitute the published "
            "Stage-1 checkpoint: Word4Per inference needs the learned Stage-2 img2text/TINet.\n\n"
            "Reproduce Stage 2 on CUHK-PEDES using the authors' recipe, then either place "
            "the artifacts at the canonical paths below, mount them as a Kaggle dataset "
            "with these filenames, or point the two environment variables at them:\n"
            f"  export {STAGE2_CHECKPOINT_ENV}=/path/to/best.pth\n"
            f"  export {STAGE2_CONFIG_ENV}=/path/to/configs.yaml\n\n"
            "To reproduce automatically before the normal benchmark run, set both:\n"
            f"  export {STAGE1_CHECKPOINT_ENV}=/path/to/stage1_model_vitb.pth\n"
            f"  export {REPRO_DATA_ROOT_ENV}=/path/to/word4per_reproduction_data\n"
            "  python run_baseline.py word4per_setmatch\n\n"
            "Expected canonical paths:\n"
            f"{details}\n"
            "Do not train, tune, or select this checkpoint on the CPR benchmark."
        )
        print(message, file=sys.stderr, flush=True)
        raise SystemExit(42)

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
    validate_reproduced_stage2(cfg, config_path=config_path)

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
