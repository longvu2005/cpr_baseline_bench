"""Pinned Grounding DINO asset helpers extracted from the existing S5 adapter."""
from __future__ import annotations
import os
import sys
import json
import hashlib
import shutil
import subprocess
import urllib.request
from pathlib import Path
from typing import Any
from benchmark_progress import byte_progress
from .cache import ROOT, resolve as resolve_path, sha256_file, read_json as read_meta
PREPARER_VERSION = "2026-10-12-shared-person-assets-v1"
BERT_MODEL_ID = "bert-base-uncased"
def rel(path):
    path = Path(path).resolve()
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def git_head(checkout: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True
    ).strip()

def ensure_clean_pinned_source(source: dict[str, Any], label: str) -> Path:
    checkout = resolve_path(str(source["local_checkout"]))
    expected = str(source["commit"])
    if not checkout.is_dir():
        raise FileNotFoundError(
            f"Missing pinned {label} source: {rel(checkout)}. "
            "Run this baseline through run_baseline.py first."
        )
    actual = git_head(checkout)
    if actual != expected:
        raise RuntimeError(
            f"{label} source commit mismatch: expected {expected}, got {actual}. "
            "Re-run download_checkpoint.py."
        )
    dirty = subprocess.check_output(
        ["git", "-C", str(checkout), "status", "--porcelain", "--untracked-files=no"],
        text=True,
    ).strip()
    if dirty:
        raise RuntimeError(
            f"Pinned {label} source has tracked local modifications: {rel(checkout)}\n{dirty}"
        )
    return checkout

def configure_groundingdino_source(source_root: Path) -> str:
    """Import Grounding DINO from the pinned checkout without building a wheel.

    The upstream package tries to compile its optional CUDA/C++ extension during
    pip installation.  That wheel build is fragile on hosted Python 3.12
    environments.  The pinned source already contains an equivalent PyTorch
    implementation of multi-scale deformable attention, so use it when the
    custom extension is unavailable.

    Returns the attention backend name for cache/reproducibility metadata.
    """

    source_text = str(source_root)
    if source_text in sys.path:
        sys.path.remove(source_text)
    sys.path.insert(0, source_text)

    # Import the exact module from the pinned checkout.  It may emit the
    # upstream "custom C++ ops" warning before we install the explicit fallback.
    import groundingdino  # type: ignore
    from groundingdino.models.GroundingDINO import ms_deform_attn as msda  # type: ignore

    package_root = Path(groundingdino.__file__).resolve().parent
    expected_root = (source_root / "groundingdino").resolve()
    if package_root != expected_root:
        raise RuntimeError(
            "Grounding DINO import did not resolve to the pinned checkout: "
            f"expected {rel(expected_root)}, got {rel(package_root)}"
        )

    class _PytorchMSDeformAttnFunction:
        @staticmethod
        def apply(value, value_spatial_shapes, value_level_start_index,
                  sampling_locations, attention_weights, im2col_step):
            return msda.multi_scale_deformable_attn_pytorch(
                value, value_spatial_shapes, sampling_locations, attention_weights)
    msda.MultiScaleDeformableAttnFunction = _PytorchMSDeformAttnFunction
    return "official_pytorch_fallback"


def require_file(path: Path, label: str) -> Path:
    if not path.is_file() or path.stat().st_size <= 0:
        raise FileNotFoundError(
            f"Missing {label}: {rel(path)}. Run this baseline through run_baseline.py first."
        )
    return path

def configure_groundingdino_offline(cfg: dict[str, Any]) -> Path:
    cache_root = resolve_path(str(cfg["detector"]["runtime_cache"]))
    marker = resolve_path(str(cfg["detector"]["runtime_assets_marker"]))
    if not marker.is_file():
        raise FileNotFoundError(
            f"Missing Grounding DINO runtime-assets marker: {rel(marker)}. "
            "Run download_checkpoint.py first."
        )
    marker_data = read_meta(marker)
    expected = {
        "source_commit": str(cfg["source"]["groundingdino"]["commit"]),
        "text_encoder": "bert-base-uncased",
        "hf_home": rel(cache_root / "huggingface"),
    }
    if marker_data is None or any(marker_data.get(k) != v for k, v in expected.items()):
        raise RuntimeError(
            f"Stale Grounding DINO runtime-assets marker: {rel(marker)}. "
            "Re-run download_checkpoint.py."
        )
    files = marker_data.get("files")
    if not isinstance(files, list) or not files:
        raise RuntimeError(f"Runtime-assets marker has no cache inventory: {rel(marker)}")
    for item in files:
        if not isinstance(item, dict):
            raise RuntimeError(f"Invalid runtime cache inventory in {rel(marker)}")
        name = item.get("path")
        size = item.get("size")
        if not isinstance(name, str) or not isinstance(size, int):
            raise RuntimeError(f"Invalid runtime cache inventory in {rel(marker)}")
        path = cache_root / name
        if not path.is_file() or path.stat().st_size != size:
            raise RuntimeError(
                f"Grounding DINO runtime cache is incomplete: {rel(path)}. "
                "Re-run download_checkpoint.py."
            )

    os.environ["HF_HOME"] = str(cache_root / "huggingface")
    os.environ["XDG_CACHE_HOME"] = str(cache_root / "xdg")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    return cache_root

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

    # The official released Grounding DINO checkpoint may contain Namespace
    # metadata. This loader is used only for that trusted release artifact.
    return torch.load(path, map_location="cpu", weights_only=False)

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
