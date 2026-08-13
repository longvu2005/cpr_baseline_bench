#!/usr/bin/env python3
"""Prepare every external artifact required by Grounding DINO + CLIP-ReID - Set.

The root runner installs requirements before this script. This preparer then:
1) pins both official source repositories,
2) downloads the official Grounding DINO Swin-T checkpoint,
3) downloads the official MSMT17 CLIP-ReID ViT-B/16 checkpoint,
4) downloads the exact OpenAI CLIP ViT-B/16 backbone used by CLIP-ReID, and
5) pre-warms Grounding DINO's bert-base-uncased runtime assets into a
   repository-local Hugging Face cache.

Inference is intentionally offline and consumes only these prepared artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
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

from benchmark_progress import PhaseTracker, byte_progress  # noqa: E402

DEFAULT_CONFIG = Path(__file__).resolve().parent / "config.yaml"
OPENAI_CLIP_B16_SHA256 = "5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f"
OPENAI_CLIP_B16_URL = (
    "https://openaipublic.azureedge.net/clip/models/"
    f"{OPENAI_CLIP_B16_SHA256}/ViT-B-16.pt"
)
BERT_MODEL_ID = "bert-base-uncased"
PREPARER_VERSION = "2026-08-13-v1"


def resolve_path(value: str) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else ROOT / path).resolve()


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise TypeError(f"Expected YAML mapping: {path}")
    return data


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def require_git() -> None:
    if shutil.which("git") is None:
        raise RuntimeError(
            "System tool 'git' is required to prepare pinned official source checkouts."
        )


def tracked_dirty(checkout: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(checkout), "status", "--porcelain", "--untracked-files=no"],
        text=True,
    ).strip()


def prepare_source(source: dict[str, Any], label: str) -> Path:
    require_git()
    checkout = resolve_path(str(source["local_checkout"]))
    expected = str(source["commit"])

    if checkout.exists() and not (checkout / ".git").is_dir():
        raise RuntimeError(f"{rel(checkout)} exists but is not a git checkout")

    if not checkout.exists():
        if not bool(source.get("auto_clone", True)):
            raise FileNotFoundError(f"Missing pinned {label} source: {rel(checkout)}")
        checkout.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", str(source["repository"]), str(checkout)],
            check=True,
        )

    dirty = tracked_dirty(checkout)
    if dirty:
        raise RuntimeError(
            f"Pinned {label} source has tracked local modifications: {rel(checkout)}\n{dirty}"
        )

    actual = subprocess.check_output(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual != expected:
        subprocess.run(
            ["git", "-C", str(checkout), "fetch", "--all", "--tags"], check=True
        )
        subprocess.run(
            ["git", "-C", str(checkout), "checkout", "--detach", expected], check=True
        )
        actual = subprocess.check_output(
            ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True
        ).strip()

    if actual != expected:
        raise RuntimeError(
            f"{label} source commit mismatch: expected {expected}, got {actual}"
        )

    print(f"[ok] {label} source pinned at {expected[:12]} ({rel(checkout)})", flush=True)
    return checkout


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


def torch_load_cpu(path: Path) -> Any:
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def validate_groundingdino_checkpoint(path: Path) -> None:
    if not path.is_file() or path.stat().st_size <= 0:
        raise FileNotFoundError(path)
    checkpoint = torch_load_cpu(path)
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("model"), dict):
        raise RuntimeError(
            f"Invalid Grounding DINO checkpoint structure: {rel(path)}; expected dict['model']"
        )
    model_state = checkpoint["model"]
    if len(model_state) < 100:
        raise RuntimeError(
            f"Grounding DINO checkpoint looks incomplete: only {len(model_state)} model tensors"
        )
    print(
        f"[ok] detector checkpoint structure valid ({len(model_state):,} tensors)",
        flush=True,
    )


def download_clipreid_checkpoint(path: Path, drive_id: str, force: bool) -> None:
    try:
        import gdown
    except ImportError as error:
        raise RuntimeError(
            "Missing dependency 'gdown'. Run this method through run_baseline.py so "
            "requirements.txt is installed first."
        ) from error

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and path.stat().st_size > 0 and not force:
        try:
            validate_clipreid_checkpoint(path)
        except Exception:
            print(f"[warn] invalid existing CLIP-ReID checkpoint; replacing {rel(path)}")
            path.unlink(missing_ok=True)
        else:
            print(f"[skip] {rel(path)} (structure valid)", flush=True)
            return

    temp = path.with_suffix(path.suffix + ".part")
    temp.unlink(missing_ok=True)
    print(f"[download] {rel(path)}", flush=True)
    try:
        result = gdown.download(id=drive_id, output=str(temp), quiet=False)
        if result is None or not temp.is_file() or temp.stat().st_size <= 0:
            raise RuntimeError("CLIP-ReID download completed without a valid file")
        validate_clipreid_checkpoint(temp)
        os.replace(temp, path)
    except Exception:
        temp.unlink(missing_ok=True)
        if path.is_file() and path.stat().st_size <= 0:
            path.unlink(missing_ok=True)
        raise
    print(f"[ok] {rel(path)}", flush=True)


def validate_clipreid_checkpoint(path: Path) -> None:
    if not path.is_file() or path.stat().st_size <= 0:
        raise FileNotFoundError(path)
    state = torch_load_cpu(path)
    if not isinstance(state, dict):
        raise RuntimeError(f"Invalid CLIP-ReID state dict: {rel(path)}")

    required = {
        "classifier.weight": (1041, 768),
        "classifier_proj.weight": (1041, 512),
        "bottleneck.weight": (768,),
        "bottleneck_proj.weight": (512,),
    }
    normalized = {str(key).removeprefix("module."): value for key, value in state.items()}
    for key, expected_shape in required.items():
        tensor = normalized.get(key)
        if tensor is None:
            raise RuntimeError(f"CLIP-ReID checkpoint missing key {key!r}: {rel(path)}")
        shape = tuple(int(x) for x in tensor.shape)
        if shape != expected_shape:
            raise RuntimeError(
                f"CLIP-ReID checkpoint shape mismatch for {key}: "
                f"expected {expected_shape}, got {shape}"
            )
    print("[ok] CLIP-ReID MSMT17 checkpoint structure valid", flush=True)


def configure_hf_cache(cache_root: Path, *, offline: bool) -> Path:
    hf_home = cache_root / "huggingface"
    hf_home.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(hf_home)
    os.environ["XDG_CACHE_HOME"] = str(cache_root / "xdg")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    if offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    else:
        os.environ.pop("HF_HUB_OFFLINE", None)
        os.environ.pop("TRANSFORMERS_OFFLINE", None)
    return hf_home


def runtime_cache_inventory(root: Path) -> list[dict[str, Any]]:
    files: list[dict[str, Any]] = []
    if not root.is_dir():
        return files
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        size = int(path.stat().st_size)
        # Hugging Face lock files may legitimately be zero bytes and are not
        # runtime assets. Excluding them keeps the inventory stable/reusable.
        if size <= 0 or ".locks" in path.parts:
            continue
        files.append(
            {
                "path": str(path.relative_to(root)),
                "size": size,
            }
        )
    return files


def inventory_is_valid(root: Path, files: Any) -> bool:
    if not isinstance(files, list) or not files:
        return False
    for item in files:
        if not isinstance(item, dict):
            return False
        name = item.get("path")
        size = item.get("size")
        if not isinstance(name, str) or not isinstance(size, int) or size <= 0:
            return False
        path = root / name
        if not path.is_file() or path.stat().st_size != size:
            return False
    return True


def prewarm_groundingdino_runtime(
    *,
    cache_root: Path,
    marker: Path,
    source_commit: str,
    force: bool,
) -> None:
    expected_core = {
        "preparer_version": PREPARER_VERSION,
        "source_commit": source_commit,
        "text_encoder": BERT_MODEL_ID,
        "hf_home": rel(cache_root / "huggingface"),
    }

    if marker.is_file() and not force:
        try:
            current = json.loads(marker.read_text(encoding="utf-8"))
        except Exception:
            current = None
        if isinstance(current, dict) and all(
            current.get(key) == value for key, value in expected_core.items()
        ) and inventory_is_valid(cache_root, current.get("files")):
            print(f"[skip] {rel(marker)} and runtime cache inventory are valid", flush=True)
            return

    configure_hf_cache(cache_root, offline=False)
    try:
        from transformers import AutoTokenizer, BertModel
    except ImportError as error:
        raise RuntimeError(
            "Missing transformers. Requirements must be installed before checkpoint preparation."
        ) from error

    print(f"[prepare] Hugging Face assets for {BERT_MODEL_ID}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_ID)
    model = BertModel.from_pretrained(BERT_MODEL_ID)
    del tokenizer, model

    files = runtime_cache_inventory(cache_root)
    if not files:
        raise RuntimeError("Grounding DINO runtime cache is empty after BERT pre-warm")

    marker.parent.mkdir(parents=True, exist_ok=True)
    payload = {**expected_core, "files": files}
    marker.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"[ok] {rel(marker)}", flush=True)


def prepare(config_path: Path, force: bool) -> None:
    tracker = PhaseTracker("groundingdino_clipreid_prepare", total=6)
    cfg = load_yaml(config_path)

    tracker.advance("Pin official Grounding DINO source")
    gdino_source = prepare_source(cfg["source"]["groundingdino"], "Grounding DINO")
    detector_config = gdino_source / str(cfg["detector"]["config"])
    if not detector_config.is_file():
        raise FileNotFoundError(detector_config)

    tracker.advance("Pin official CLIP-ReID source")
    clipreid_source = prepare_source(cfg["source"]["clip_reid"], "CLIP-ReID")
    clipreid_config = clipreid_source / str(cfg["reid"]["official_config"])
    if not clipreid_config.is_file():
        raise FileNotFoundError(clipreid_config)

    tracker.advance("Prepare Grounding DINO Swin-T checkpoint")
    detector_checkpoint = resolve_path(str(cfg["detector"]["checkpoint"]))
    download_stream(
        url=str(cfg["detector"]["checkpoint_url"]),
        path=detector_checkpoint,
        force=force,
    )
    try:
        validate_groundingdino_checkpoint(detector_checkpoint)
    except Exception:
        # Do not leave a structurally invalid file that would be mistaken for
        # an already-prepared checkpoint on the next run.
        detector_checkpoint.unlink(missing_ok=True)
        raise

    tracker.advance("Prepare CLIP-ReID MSMT17 checkpoint")
    reid_checkpoint = resolve_path(str(cfg["reid"]["checkpoint"]))
    download_clipreid_checkpoint(
        reid_checkpoint,
        str(cfg["reid"]["checkpoint_drive_id"]),
        force,
    )

    tracker.advance("Prepare OpenAI CLIP ViT-B/16 backbone")
    clip_backbone = resolve_path(str(cfg["reid"]["openai_clip_checkpoint"]))
    download_stream(
        url=OPENAI_CLIP_B16_URL,
        path=clip_backbone,
        force=force,
        expected_sha256=OPENAI_CLIP_B16_SHA256,
    )

    tracker.advance("Pre-warm Grounding DINO BERT runtime assets")
    cache_root = resolve_path(str(cfg["detector"]["runtime_cache"]))
    marker = resolve_path(str(cfg["detector"]["runtime_assets_marker"]))
    prewarm_groundingdino_runtime(
        cache_root=cache_root,
        marker=marker,
        source_commit=str(cfg["source"]["groundingdino"]["commit"]),
        force=force,
    )

    print("[status] Grounding DINO + CLIP-ReID inference artifacts are ready", flush=True)
    tracker.finish()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare Grounding DINO + CLIP-ReID - Set inference artifacts."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download public weights and refresh runtime assets.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prepare(resolve_path(args.config), args.force)


if __name__ == "__main__":
    main()
