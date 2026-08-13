#!/usr/bin/env python3
"""Prepare all external artifacts for P3 Instruct-ReID + SetMatch.

The important distinction is between the official *final task checkpoint* and
bootstrap/runtime assets.  The final Instruct-ReID checkpoint is resolved from
the official inference-model Google Drive folder.  BERT, the COCO detector, and
CLIP selector are prepared only as runtime dependencies.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import re
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Any

import torch
import yaml

os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark_progress import PhaseTracker, byte_progress  # noqa: E402

DEFAULT_CONFIG = Path(__file__).resolve().parent / "config.yaml"
CLIP_B32_SHA256 = "40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af"
CLIP_B32_URL = (
    "https://openaipublic.azureedge.net/clip/models/"
    f"{CLIP_B32_SHA256}/ViT-B-32.pt"
)
DETECTOR_HASH_PREFIX = "dd69338a"
EXPECTED_RUNTIME_VERSIONS = {
    "transformers": "4.39.3",
    "tokenizers": "0.15.2",
    "scikit-learn": "1.3.2",
}


def validate_runtime_versions() -> None:
    problems: list[str] = []
    resolved: list[str] = []
    for package, expected in EXPECTED_RUNTIME_VERSIONS.items():
        try:
            actual = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            problems.append(f"{package}: missing (expected {expected})")
            continue
        resolved.append(f"{package}={actual}")
        if actual != expected:
            problems.append(f"{package}: found {actual}, expected {expected}")

    if problems:
        details = "\n".join(f"  - {item}" for item in problems)
        raise RuntimeError(
            "P3 dependency preflight failed. The active Python environment does not "
            "match the pinned Instruct-ReID runtime:\n"
            f"{details}\n"
            "Re-run `python run_baseline.py instruct_reid_setmatch` without "
            "`--skip-install`. On Python 3.12 the method requirements force binary "
            "wheels for tokenizers/scikit-learn and must not compile tokenizers from source."
        )

    print(f"[ok] dependency preflight: {', '.join(resolved)}", flush=True)


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
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"Expected YAML mapping: {path}")
    return value


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def download_http(
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
            print(f"[skip] {rel(path)} (existing artifact valid)", flush=True)
            return
        print(f"[warn] replacing invalid {rel(path)}", flush=True)

    temp = path.with_name(path.name + ".part")
    temp.unlink(missing_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "cpr-baseline-bench/1.0"})
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
        actual = digest.hexdigest()
        if expected_sha256 is not None and actual != expected_sha256:
            raise RuntimeError(
                f"Checksum mismatch for {rel(path)}: expected {expected_sha256}, got {actual}"
            )
        if expected_hash_prefix is not None and not actual.startswith(expected_hash_prefix):
            raise RuntimeError(
                f"Checksum prefix mismatch for {rel(path)}: expected {expected_hash_prefix}, got {actual}"
            )
        if temp.stat().st_size <= 0:
            raise RuntimeError(f"Downloaded zero-byte artifact: {rel(path)}")
        os.replace(temp, path)
    except Exception:
        temp.unlink(missing_ok=True)
        raise
    print(f"[ok] {rel(path)}", flush=True)


def generated_python_artifact(path: str) -> bool:
    normalized = path.strip().strip('"').replace("\\", "/")
    return normalized.lower().endswith((".pyc", ".pyo"))


def tracked_dirty(checkout: Path) -> str:
    output = subprocess.check_output(
        ["git", "-C", str(checkout), "status", "--porcelain", "--untracked-files=no"],
        text=True,
    )
    dirty: list[str] = []
    for line in output.splitlines():
        if not line.strip():
            continue
        path = line[3:].strip() if len(line) >= 4 else line
        parts = path.split(" -> ")
        if parts and all(generated_python_artifact(x) for x in parts):
            continue
        dirty.append(line)
    return "\n".join(dirty)


def prepare_source(cfg: dict[str, Any]) -> Path:
    source = cfg["source"]
    checkout = resolve_path(str(source["local_checkout"]))
    expected_commit = str(source["commit"])
    if not checkout.exists():
        if not bool(source.get("auto_clone", True)):
            raise FileNotFoundError(f"Missing official source checkout: {rel(checkout)}")
        checkout.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", str(source["repository"]), str(checkout)], check=True)

    dirty = tracked_dirty(checkout)
    if dirty:
        raise RuntimeError(
            f"Official source has tracked local modifications: {rel(checkout)}\n{dirty}"
        )

    actual = subprocess.check_output(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual != expected_commit:
        subprocess.run(["git", "-C", str(checkout), "fetch", "--all", "--tags"], check=True)
        subprocess.run(
            ["git", "-C", str(checkout), "checkout", "--detach", expected_commit], check=True
        )
        actual = subprocess.check_output(
            ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True
        ).strip()
    if actual != expected_commit:
        raise RuntimeError(f"Source commit mismatch: expected {expected_commit}, got {actual}")
    print(f"[ok] official Instruct-ReID source @ {actual[:12]}", flush=True)
    return checkout


def list_official_inference_folder(folder_url: str) -> list[dict[str, str]]:
    command = [
        sys.executable,
        "-m",
        "gdown",
        folder_url,
        "--folder",
        "--json",
        "--quiet",
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError(
            "Could not parse `gdown --folder --json` output from the official Instruct-ReID folder.\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        ) from error
    if not isinstance(data, list):
        raise RuntimeError("Unexpected gdown listing format: expected a JSON list")
    entries: list[dict[str, str]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        url = item.get("url")
        path = item.get("path")
        if isinstance(url, str) and isinstance(path, str):
            entries.append({"url": url, "path": path})
    if not entries:
        raise RuntimeError("Official Instruct-ReID inference folder listing is empty")
    return entries


def choose_language_checkpoint(cfg: dict[str, Any]) -> tuple[str, str, list[str]]:
    checkpoint_cfg = cfg["checkpoint"]
    direct = checkpoint_cfg.get("direct_file_url")
    if isinstance(direct, str) and direct.strip():
        return direct.strip(), "direct_file_url", []

    entries = list_official_inference_folder(str(checkpoint_cfg["inference_folder_url"]))
    discovery = checkpoint_cfg["discovery"]
    checkpoint_re = re.compile(str(discovery["checkpoint_regex"]))
    all_checkpoints = [item for item in entries if checkpoint_re.search(item["path"])]

    # The official inference folder currently publishes the language-instructed
    # model as `checkpoint_li.pth.tar`. Prefer an exact configured basename so
    # the adapter cannot silently switch tasks if upstream adds similarly named
    # checkpoints. The regex path is retained only for backward compatibility
    # with configs that predate `expected_filename`.
    expected_filename = str(discovery.get("expected_filename") or "").strip()
    if expected_filename:
        candidates = [
            item
            for item in all_checkpoints
            if Path(item["path"].replace("\\", "/")).name == expected_filename
        ]
        if len(candidates) != 1:
            discovered = [item["path"] for item in all_checkpoints]
            raise RuntimeError(
                "Could not find the configured official language-instructed Instruct-ReID checkpoint.\n"
                f"Expected exact filename: {expected_filename!r}\n"
                f"Exact matches ({len(candidates)}): {[item['path'] for item in candidates]}\n"
                f"All checkpoint-like files in the official folder: {discovered}\n"
                "This adapter refuses to guess or substitute another ReID task. "
                "If the official release is renamed, verify the upstream folder and update "
                "checkpoint.discovery.expected_filename (or set checkpoint.direct_file_url "
                "to that exact official file)."
            )
        chosen = candidates[0]
        return chosen["url"], chosen["path"], [item["path"] for item in all_checkpoints]

    required_re = re.compile(str(discovery["required_regex"]))
    exclude_re = re.compile(str(discovery["exclude_regex"]))
    candidates = [
        item
        for item in all_checkpoints
        if required_re.search(item["path"]) and not exclude_re.search(item["path"])
    ]
    if len(candidates) != 1:
        discovered = [item["path"] for item in all_checkpoints]
        candidate_paths = [item["path"] for item in candidates]
        raise RuntimeError(
            "Could not uniquely identify the official language/attribute Instruct-ReID inference checkpoint.\n"
            f"Matching candidates ({len(candidate_paths)}): {candidate_paths}\n"
            f"All checkpoint-like files in the official folder: {discovered}\n"
            "This adapter refuses to guess. If upstream renamed the file, set checkpoint.direct_file_url "
            "in config.yaml to the exact file URL from the same official inference-model folder."
        )
    chosen = candidates[0]
    return chosen["url"], chosen["path"], [item["path"] for item in all_checkpoints]


def validate_final_checkpoint(path: Path) -> dict[str, Any]:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("state_dict"), dict):
        raise RuntimeError(
            f"{rel(path)} is not an official-test-style Instruct-ReID checkpoint: missing state_dict"
        )
    state = checkpoint["state_dict"]
    keys = [str(k) for k in state.keys()]
    required_families = {
        "visual_encoder": any("visual_encoder" in key for key in keys),
        "text_encoder": any("text_encoder" in key for key in keys),
        "fusion": any("fusion" in key for key in keys),
    }
    if not all(required_families.values()):
        raise RuntimeError(
            f"Checkpoint structure does not look like final Instruct-ReID IRM weights: {required_families}"
        )
    if len(keys) < 100:
        raise RuntimeError(f"Checkpoint state_dict is unexpectedly small ({len(keys)} tensors)")
    return {
        "num_state_tensors": len(keys),
        "required_families": required_families,
    }


def download_final_checkpoint(cfg: dict[str, Any], force: bool) -> tuple[Path, dict[str, Any]]:
    path = resolve_path(str(cfg["checkpoint"]["path"]))
    marker_path = resolve_path(str(cfg["checkpoint"]["prepared_marker"]))
    if path.is_file() and marker_path.is_file() and not force:
        try:
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            marker = None
        if isinstance(marker, dict) and marker.get("checkpoint_sha256") == sha256_file(path):
            structure = validate_final_checkpoint(path)
            print(f"[skip] {rel(path)} (official final checkpoint already validated)", flush=True)
            return path, {**marker, "structure": structure}

    try:
        import gdown
    except ImportError as error:
        raise RuntimeError(
            "Missing gdown>=6.1. Run P3 through run_baseline.py so requirements are installed first."
        ) from error

    source_url, source_path, all_checkpoint_paths = choose_language_checkpoint(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".part")
    temp.unlink(missing_ok=True)
    if force:
        path.unlink(missing_ok=True)
    print(f"[download] official final language-ReID checkpoint: {source_path}", flush=True)
    try:
        result = gdown.download(url=source_url, output=str(temp), quiet=False)
        if result is None or not temp.is_file() or temp.stat().st_size <= 0:
            raise RuntimeError("gdown completed without producing a checkpoint file")
        structure = validate_final_checkpoint(temp)
        os.replace(temp, path)
    except Exception:
        temp.unlink(missing_ok=True)
        raise

    marker = {
        "status": str(cfg["checkpoint"]["status"]),
        "task": str(cfg["checkpoint"]["task"]),
        "official_test_task_type": str(cfg["checkpoint"]["official_test_task_type"]),
        "official_inference_folder_url": str(cfg["checkpoint"]["inference_folder_url"]),
        "source_url": source_url,
        "source_path": source_path,
        "all_checkpoint_paths_seen": all_checkpoint_paths,
        "checkpoint": rel(path),
        "checkpoint_sha256": sha256_file(path),
        "checkpoint_size": int(path.stat().st_size),
        "structure": structure,
    }
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    marker_path.write_text(json.dumps(marker, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[ok] {rel(path)}", flush=True)
    return path, marker


def validate_bert_weight(path: Path) -> None:
    if not path.is_file() or path.stat().st_size < 100_000_000:
        raise RuntimeError(f"BERT runtime weight appears incomplete: {rel(path)}")
    try:
        state = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        state = torch.load(path, map_location="cpu")
    if not isinstance(state, dict) or len(state) < 100:
        raise RuntimeError(f"BERT runtime weight has unexpected structure: {rel(path)}")


def prepare_bert_runtime(cfg: dict[str, Any], checkout: Path, force: bool) -> dict[str, Any]:
    runtime = cfg["runtime_assets"]
    dst_dir = resolve_path(str(runtime["bert_dir"]))
    dst_dir.mkdir(parents=True, exist_ok=True)
    src_dir = checkout / "bert-base-uncased"
    if not src_dir.is_dir():
        raise FileNotFoundError(f"Pinned source is missing bert-base-uncased metadata: {src_dir}")

    copied: list[str] = []
    for source in sorted(p for p in src_dir.iterdir() if p.is_file()):
        target = dst_dir / source.name
        if force or not target.is_file() or sha256_file(target) != sha256_file(source):
            shutil.copy2(source, target)
        copied.append(source.name)

    config_src = checkout / "config_bert.json"
    config_dst = resolve_path(str(runtime["config_bert"]))
    config_dst.parent.mkdir(parents=True, exist_ok=True)
    if force or not config_dst.is_file() or sha256_file(config_dst) != sha256_file(config_src):
        shutil.copy2(config_src, config_dst)

    weight = dst_dir / "pytorch_model.bin"
    if force:
        weight.unlink(missing_ok=True)
    if not weight.is_file():
        download_http(url=str(runtime["bert_weight_url"]), path=weight, force=False)
    validate_bert_weight(weight)
    return {
        "bert_dir": rel(dst_dir),
        "bert_weight_sha256": sha256_file(weight),
        "config_bert": rel(config_dst),
        "config_bert_sha256": sha256_file(config_dst),
        "tokenizer_files": copied,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare P3 Instruct-ReID + SetMatch artifacts")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    validate_runtime_versions()

    config_path = resolve_path(args.config)
    cfg = load_yaml(config_path)
    tracker = PhaseTracker("instruct_reid_setmatch_prepare", total=5)

    tracker.advance("Pin official Instruct-ReID source")
    checkout = prepare_source(cfg)

    tracker.advance("Resolve and download official final language-ReID checkpoint")
    checkpoint_path, checkpoint_meta = download_final_checkpoint(cfg, args.force)

    tracker.advance("Prepare BERT runtime assets required by official constructor")
    bert_meta = prepare_bert_runtime(cfg, checkout, args.force)

    tracker.advance("Prepare predicted-person detector checkpoint")
    detector_cfg = cfg["localization"]["detector"]
    download_http(
        url=str(detector_cfg["checkpoint_url"]),
        path=resolve_path(str(detector_cfg["checkpoint"])),
        force=args.force,
        expected_hash_prefix=DETECTOR_HASH_PREFIX,
    )

    tracker.advance("Prepare OpenAI CLIP ViT-B/32 target selector")
    selector_path = resolve_path(str(cfg["localization"]["query_selector"]["checkpoint"]))
    download_http(
        url=CLIP_B32_URL,
        path=selector_path,
        force=args.force,
        expected_sha256=CLIP_B32_SHA256,
    )

    summary = {
        "source_commit": str(cfg["source"]["commit"]),
        "final_checkpoint": checkpoint_meta,
        "final_checkpoint_sha256": sha256_file(checkpoint_path),
        "bert_runtime": bert_meta,
        "detector_sha256": sha256_file(resolve_path(str(detector_cfg["checkpoint"]))),
        "selector_sha256": sha256_file(selector_path),
    }
    summary_path = resolve_path(str(cfg["checkpoint"]["prepared_marker"])).with_name(
        "all_runtime_assets.prepared.json"
    )
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tracker.finish()


if __name__ == "__main__":
    main()
