#!/usr/bin/env python3
"""Reproduce the missing official Word4Per Stage-2 artifact for CPR evaluation.

This script intentionally does NOT train on the CPR benchmark.  It runs the
pinned authors' ``old_project/train_stage2.py`` recipe using:

* the authors' released Word4Per Stage-1 checkpoint (supplied by the user),
* CUHK-PEDES for Stage-2 training, and
* the original ITCPR query/gallery data used by the authors' Stage-2 evaluator
  to select ``best.pth``.

After the official training process finishes, the generated ``best.pth`` and
``configs.yaml`` are validated and copied to the canonical paths expected by
``run_baseline.py word4per_setmatch``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import yaml

ROOT = Path(__file__).resolve().parents[3]
METHOD_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = METHOD_DIR / "config.yaml"

# Import the checkpoint preparer's source-pinning and Stage-2 validators rather
# than duplicating benchmark policy in two places.  Importing this module does
# not execute its CLI main().
sys.path.insert(0, str(METHOD_DIR))
from download_checkpoint import (  # noqa: E402
    load_yaml,
    prepare_clip,
    prepare_source,
    resolve_path,
    validate_stage2_checkpoint_structure,
    validate_stage2_recipe,
)

STAGE1_ENV = "WORD4PER_STAGE1_CHECKPOINT"
DATA_ROOT_ENV = "WORD4PER_REPRO_DATA_ROOT"


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def validate_stage1_checkpoint(path: Path) -> None:
    if not path.is_file() or path.stat().st_size <= 0:
        raise FileNotFoundError(f"Missing Word4Per Stage-1 checkpoint: {path}")
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    except Exception as error:
        raise RuntimeError(f"Could not read Word4Per Stage-1 checkpoint: {path}") from error

    if not isinstance(checkpoint, Mapping):
        raise RuntimeError("Stage-1 checkpoint must be a mapping")
    model_state = checkpoint.get("model")
    if not isinstance(model_state, Mapping) or not model_state:
        raise RuntimeError("Stage-1 checkpoint is missing a non-empty 'model' state_dict")
    if not any(isinstance(value, torch.Tensor) for value in model_state.values()):
        raise RuntimeError("Stage-1 'model' state_dict contains no tensors")
    if "img2text" in checkpoint:
        raise RuntimeError(
            "The supplied file already contains 'img2text'; expected the authors' Stage-1 "
            "checkpoint, not a Stage-2 checkpoint."
        )


def validate_external_data_root(data_root: Path) -> dict[str, Any]:
    """Validate the exact external data layout consumed by official train_stage2.py."""
    data_root = data_root.resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(f"Missing Word4Per reproduction data root: {data_root}")

    cuhk_dir = data_root / "CUHK-PEDES"
    cuhk_images = cuhk_dir / "imgs"
    cuhk_annotations = cuhk_dir / "reid_raw.json"
    itcpr_query = data_root / "query.json"
    itcpr_gallery = data_root / "gallery.json"

    required = [cuhk_images, cuhk_annotations, itcpr_query, itcpr_gallery]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError(
            "Word4Per Stage-2 reproduction requires the authors' external data layout.\n"
            "Missing:\n  - " + "\n  - ".join(missing)
        )

    cuhk = load_json(cuhk_annotations)
    if not isinstance(cuhk, list) or not cuhk:
        raise RuntimeError(f"Invalid CUHK-PEDES annotation file: {cuhk_annotations}")
    train_ids: set[int] = set()
    missing_cuhk_images: list[str] = []
    for index, row in enumerate(cuhk):
        if not isinstance(row, dict):
            raise RuntimeError(f"CUHK-PEDES annotation row {index} is not an object")
        if row.get("split") == "train":
            try:
                train_ids.add(int(row["id"]) - 1)
            except (KeyError, TypeError, ValueError) as error:
                raise RuntimeError(f"Invalid CUHK-PEDES identity at row {index}") from error
        file_path = row.get("file_path")
        if not isinstance(file_path, str) or not file_path:
            raise RuntimeError(f"CUHK-PEDES row {index} has no file_path")
        image = cuhk_images / file_path
        if not image.is_file() and len(missing_cuhk_images) < 10:
            missing_cuhk_images.append(str(image))

    # The authors' CUHK-PEDES loader asserts contiguous zero-based training IDs.
    # Enforce the same precondition before spending hours on Stage-2 training.
    if train_ids:
        ordered = sorted(train_ids)
        if ordered != list(range(len(ordered))):
            raise RuntimeError("CUHK-PEDES training identities are not contiguous after id-1")
    if len(train_ids) != 11003:
        raise RuntimeError(
            f"Unexpected CUHK-PEDES train identity count: {len(train_ids)}; expected 11003"
        )
    if missing_cuhk_images:
        raise RuntimeError(
            "CUHK-PEDES image files referenced by reid_raw.json are missing, e.g.:\n  - "
            + "\n  - ".join(missing_cuhk_images)
        )

    query_rows = load_json(itcpr_query)
    gallery_rows = load_json(itcpr_gallery)
    if not isinstance(query_rows, list) or not query_rows:
        raise RuntimeError(f"Invalid ITCPR query file: {itcpr_query}")
    if not isinstance(gallery_rows, list) or not gallery_rows:
        raise RuntimeError(f"Invalid ITCPR gallery file: {itcpr_gallery}")

    def validate_itcpr_rows(rows: list[Any], label: str, require_caption: bool) -> None:
        missing_images: list[str] = []
        for index, row in enumerate(rows):
            if not isinstance(row, dict):
                raise RuntimeError(f"ITCPR {label} row {index} is not an object")
            required_keys = {"file_path", "person_id", "instance_id"}
            if require_caption:
                required_keys.add("caption")
            absent = sorted(key for key in required_keys if key not in row)
            if absent:
                raise RuntimeError(
                    f"ITCPR {label} row {index} is missing: {', '.join(absent)}"
                )
            file_path = row["file_path"]
            if not isinstance(file_path, str) or not file_path:
                raise RuntimeError(f"ITCPR {label} row {index} has invalid file_path")
            image = data_root / file_path
            if not image.is_file() and len(missing_images) < 10:
                missing_images.append(str(image))
        if missing_images:
            raise RuntimeError(
                f"ITCPR {label} image files are missing, e.g.:\n  - "
                + "\n  - ".join(missing_images)
            )

    validate_itcpr_rows(query_rows, "query", require_caption=True)
    validate_itcpr_rows(gallery_rows, "gallery", require_caption=False)

    return {
        "data_root": str(data_root),
        "cuhk_rows": len(cuhk),
        "cuhk_train_ids": len(train_ids),
        "itcpr_queries": len(query_rows),
        "itcpr_gallery": len(gallery_rows),
    }


def expose_stage1_to_official_source(old_project: Path, stage1: Path) -> Path:
    target = old_project / "models" / "stage1_model_vitb.pth"
    target.parent.mkdir(parents=True, exist_ok=True)

    if target.is_symlink():
        if target.resolve() == stage1.resolve():
            return target
        target.unlink()
    elif target.exists():
        # Never silently overwrite a pre-existing checkpoint in the pinned source.
        if target.is_file() and sha256_file(target) == sha256_file(stage1):
            return target
        raise RuntimeError(
            f"Official source already contains a different Stage-1 artifact: {target}. "
            "Remove it explicitly before reproducing."
        )

    try:
        target.symlink_to(stage1.resolve())
    except OSError:
        shutil.copy2(stage1, target)
    return target


def prepare_official_clip_cache(cfg: dict[str, Any]) -> Path:
    """Seed the exact OpenAI ViT-B/16 asset into old_project's default CLIP cache."""
    checkpoint_cfg = cfg["checkpoint"]
    model_name = str(checkpoint_cfg["base_clip_model"])
    if model_name != "ViT-B/16":
        raise RuntimeError(f"Unsupported Word4Per reproduction backbone: {model_name!r}")

    canonical = resolve_path(str(checkpoint_cfg["base_clip"]))
    prepare_clip(model_name, canonical, force=False)
    expected_sha256 = sha256_file(canonical)

    cache_target = Path.home() / ".cache" / "clip" / "ViT-B-16.pt"
    cache_target.parent.mkdir(parents=True, exist_ok=True)
    if cache_target.is_symlink():
        if cache_target.resolve() == canonical.resolve():
            return cache_target
        cache_target.unlink()
    elif cache_target.exists():
        if not cache_target.is_file():
            raise RuntimeError(f"CLIP cache target is not a file: {cache_target}")
        if sha256_file(cache_target) == expected_sha256:
            return cache_target
        cache_target.unlink()

    try:
        cache_target.symlink_to(canonical.resolve())
    except OSError:
        shutil.copy2(canonical, cache_target)
    if sha256_file(cache_target) != expected_sha256:
        raise RuntimeError(f"Failed to seed a checksum-identical CLIP cache at {cache_target}")
    return cache_target


def newest_completed_run(output_base: Path) -> tuple[Path, Path, Path]:
    dataset_dir = output_base / "CUHK-PEDES"
    candidates: list[tuple[int, Path, Path, Path]] = []
    if dataset_dir.is_dir():
        for run_dir in dataset_dir.iterdir():
            if not run_dir.is_dir():
                continue
            best = run_dir / "best.pth"
            configs = run_dir / "configs.yaml"
            if best.is_file() and best.stat().st_size > 0 and configs.is_file():
                candidates.append((best.stat().st_mtime_ns, run_dir, best, configs))
    if not candidates:
        raise RuntimeError(
            f"Official Stage-2 training finished without a usable best.pth under {dataset_dir}"
        )
    _, run_dir, best, configs = max(candidates, key=lambda item: item[0])
    return run_dir, best, configs


def copy_stage2_artifacts(
    cfg: dict[str, Any], best: Path, configs: Path
) -> tuple[Path, Path]:
    checkpoint_cfg = cfg["checkpoint"]
    canonical_best = resolve_path(str(checkpoint_cfg["stage2"]))
    canonical_configs = resolve_path(str(checkpoint_cfg["stage2_config"]))
    canonical_best.parent.mkdir(parents=True, exist_ok=True)
    canonical_configs.parent.mkdir(parents=True, exist_ok=True)

    for target in (canonical_best, canonical_configs):
        if target.is_symlink():
            target.unlink()

    shutil.copy2(best, canonical_best)
    shutil.copy2(configs, canonical_configs)

    validate_stage2_checkpoint_structure(canonical_best)
    with canonical_configs.open("r", encoding="utf-8") as handle:
        stage2_data = yaml.load(handle, Loader=yaml.FullLoader)
    if not isinstance(stage2_data, dict):
        raise RuntimeError(f"Invalid reproduced Stage-2 configs.yaml: {canonical_configs}")
    validate_stage2_recipe(stage2_data, canonical_configs)
    if str(stage2_data.get("pretrain_choice")) != str(checkpoint_cfg["base_clip_model"]):
        raise RuntimeError(
            "Reproduced Stage-2 backbone mismatch: "
            f"{stage2_data.get('pretrain_choice')!r} != {checkpoint_cfg['base_clip_model']!r}"
        )
    return canonical_best, canonical_configs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reproduce the authors' Word4Per Stage-2 best.pth on external "
            "CUHK-PEDES + ITCPR data, then install it for CPR evaluation."
        )
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument(
        "--stage1-checkpoint",
        default=os.environ.get(STAGE1_ENV, ""),
        help=f"Authors' released Stage-1 checkpoint, or set ${STAGE1_ENV}.",
    )
    parser.add_argument(
        "--data-root",
        default=os.environ.get(DATA_ROOT_ENV, ""),
        help=(
            "External root containing CUHK-PEDES/, query.json, gallery.json and "
            f"ITCPR source images, or set ${DATA_ROOT_ENV}."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "runs" / "word4per_setmatch" / "stage2_reproduction"),
        help="Base output directory passed to official train_stage2.py.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.stage1_checkpoint:
        raise SystemExit(
            f"Missing Stage-1 checkpoint. Pass --stage1-checkpoint or set ${STAGE1_ENV}."
        )
    if not args.data_root:
        raise SystemExit(
            f"Missing external reproduction data root. Pass --data-root or set ${DATA_ROOT_ENV}."
        )
    if not torch.cuda.is_available():
        raise SystemExit("Official Word4Per Stage-2 training requires CUDA in the pinned code.")

    config_path = resolve_path(str(args.config))
    cfg = load_yaml(config_path)
    if str(cfg.get("method")) != "word4per_setmatch":
        raise RuntimeError(f"Unexpected method config: {cfg.get('method')!r}")
    if str(cfg["checkpoint"]["base_clip_model"]) != "ViT-B/16":
        raise RuntimeError("This reproduction helper is pinned to Word4Per ViT-B/16")

    stage1 = Path(args.stage1_checkpoint).expanduser().resolve()
    data_root = Path(args.data_root).expanduser().resolve()
    output_base = Path(args.output_dir).expanduser().resolve()

    print("[1/7] Validate official Stage-1 checkpoint", flush=True)
    validate_stage1_checkpoint(stage1)
    stage1_sha256 = sha256_file(stage1)
    print(f"[ok] {stage1} sha256={stage1_sha256}", flush=True)

    print("[2/7] Validate external CUHK-PEDES + ITCPR data", flush=True)
    data_stats = validate_external_data_root(data_root)
    print("[ok] " + json.dumps(data_stats, sort_keys=True), flush=True)

    print("[3/7] Pin official Word4Per source", flush=True)
    old_project = prepare_source(cfg)
    exposed_stage1 = expose_stage1_to_official_source(old_project, stage1)
    print(f"[ok] Stage-1 exposed at {exposed_stage1}", flush=True)

    print("[4/7] Prepare checksum-pinned OpenAI CLIP base", flush=True)
    clip_cache = prepare_official_clip_cache(cfg)
    print(f"[ok] official CLIP cache: {clip_cache}", flush=True)

    print("[5/7] Run official Word4Per Stage-2 training", flush=True)
    output_base.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-u",
        "train_stage2.py",
        "--name",
        "cpr_reproduce_word4per_stage2",
        "--output_dir",
        str(output_base),
        "--root_dir",
        str(data_root),
        "--pretrain_choice",
        "ViT-B/16",
        "--img_aug",
        "--batch_size",
        "128",
        "--lr",
        "1e-4",
        "--optimizer",
        "AdamW",
        "--dataset_name",
        "CUHK-PEDES",
        "--loss_names",
        "sdm+id",
        "--toword_loss",
        "text",
        "--num_epoch",
        "60",
    ]
    print("$ " + " ".join(command), flush=True)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    # The helper intentionally reproduces the authors' single-GPU path.
    # Avoid accidentally entering DDP from notebook/container environment variables.
    env["WORLD_SIZE"] = "1"
    subprocess.run(command, cwd=old_project, env=env, check=True)

    print("[6/7] Validate and install reproduced Stage-2 artifacts", flush=True)
    run_dir, best, configs = newest_completed_run(output_base)
    canonical_best, canonical_configs = copy_stage2_artifacts(cfg, best, configs)
    print(f"[ok] {canonical_best}", flush=True)
    print(f"[ok] {canonical_configs}", flush=True)

    print("[7/7] Write provenance", flush=True)
    provenance_path = canonical_best.parent / "word4per_cuhk_pedes_stage2_reproduction.json"
    provenance = {
        "method": "word4per_setmatch",
        "official_repository": cfg["source"]["repository"],
        "official_commit": cfg["source"]["commit"],
        "official_training_script": "old_project/train_stage2.py",
        "stage1_checkpoint": str(stage1),
        "stage1_sha256": stage1_sha256,
        "openai_clip_cache": str(clip_cache),
        "openai_clip_sha256": sha256_file(clip_cache),
        "external_data": data_stats,
        "official_run_dir": str(run_dir),
        "stage2_checkpoint": str(canonical_best),
        "stage2_checkpoint_sha256": sha256_file(canonical_best),
        "stage2_config": str(canonical_configs),
        "selection": "official Evaluator_toword R@1 best checkpoint",
        "cpr_supervision": False,
    }
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(f"[ok] {provenance_path}", flush=True)
    print()
    print("Word4Per Stage-2 is ready for benchmark inference.")
    print("Run:")
    print("  python run_baseline.py word4per_setmatch --skip-install")


if __name__ == "__main__":
    main()
