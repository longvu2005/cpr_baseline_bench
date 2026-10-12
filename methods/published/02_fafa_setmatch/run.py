#!/usr/bin/env python3
"""FAFA + Partial SetMatch adapter for the CPR baseline benchmark.

The retrieval model/scoring path is imported from the official FAFA_SynCPR
implementation. Person boxes and query-target localization are predicted; no GT
identity-to-box mapping is consumed by this adapter.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import socket
import subprocess
import sys
from contextlib import contextmanager
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from PIL import Image
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader, Dataset

# The official FAFA checkout is immutable benchmark input. Disable Python bytecode
# writes both in this process and inherited worker processes so importing upstream
# modules cannot mutate tracked __pycache__ files.
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmark_progress import PhaseTracker, progress_bar  # noqa: E402
from groundingdino_adapter import preflight as detector_preflight  # noqa: E402
DEFAULT_CONFIG = Path(__file__).resolve().parent / "config.yaml"
METHOD_ID = "fafa_setmatch"
ADAPTER_VERSION = "2026-10-12-v5-groundingdino-stage-cache"
# Runtime-only guard revision. Deliberately excluded from stage cache keys so this
# non-semantic guard revision does not invalidate inference caches by itself.
RUNTIME_GUARD_VERSION = "2026-08-13-v1-readonly-official-source"


@dataclass(frozen=True)
class QueryTarget:
    modify_text: str
    select_text: str
    subject_id: Any = None


@dataclass(frozen=True)
class BoxCandidate:
    box: tuple[float, float, float, float]
    score: float
    fallback: bool = False


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise TypeError(f"{path}:{lineno}: JSONL row must be an object")
            rows.append(row)
    return rows


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise TypeError(f"Expected YAML mapping: {path}")
    return cfg


def resolve_config_path(value: str) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else ROOT / path).resolve()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_identity(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    stat = path.stat()
    return {
        "path": rel(path),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".partial")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temp, path)


def atomic_save(path: Path, array: np.ndarray) -> None:
    temp = path.with_name(path.name + ".partial")
    with temp.open("wb") as handle:
        np.save(handle, array, allow_pickle=False)
    os.replace(temp, path)


def read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def build_stage_caches(
    cfg: dict[str, Any], gallery_path: Path, queries_path: Path, output: Path,
    detector_config: Path, detector_checkout: Path,
) -> tuple[dict[str, Path], dict[str, Any]]:
    """Separate each stage's input identity from execution/aggregation settings.

    Batch sizes, workers, devices, output paths and SetMatch background scores
    do not change embeddings/detections and must not rerun those stages.
    """
    gallery_hash, query_hash = sha256_file(gallery_path), sha256_file(queries_path)
    detector = cfg["localization"]["detector"]
    ckpt = cfg["checkpoint"]
    keys, fingerprints, directories = {}, {}, {}

    def stage(name: str, payload: dict[str, Any]):
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        keys[name] = hashlib.sha256(raw).hexdigest()[:16]
        fingerprints[name] = payload
        directories[name] = output / "cache" / name / keys[name]
        directories[name].mkdir(parents=True, exist_ok=True)

    helper = ROOT / "methods/simple/05_reid_set/run.py"
    stage("detections", {
        "schema": "groundingdino-batched-v1", "gallery": gallery_hash,
        "source": detector["source"]["commit"], "config": sha256_file(detector_config),
        "checkpoint": checkpoint_identity(resolve_config_path(detector["checkpoint"])),
        "helper_sha256": sha256_file(helper),
        "worker_sha256": sha256_file(Path(__file__).with_name("groundingdino_adapter.py")),
        "native_extensions": [checkpoint_identity(p) for p in sorted((detector_checkout / "groundingdino").glob("_C*.so"))],
        "protocol": {k: detector[k] for k in ("text_prompt", "box_threshold", "text_threshold",
                    "nms_iou_threshold", "min_box_size_px", "max_persons_per_image")},
    })
    selector = cfg["localization"]["query_selector"]
    stage("localization", {
        "schema": "clip-anchor-expansion-v1", "detections": keys["detections"], "queries": query_hash,
        "score_threshold": detector["score_threshold"], "selector": selector,
        "checkpoint": checkpoint_identity(resolve_config_path(selector["checkpoint"])),
    })
    encoder = {
        "schema": "official-fafa-features-v1", "source": cfg["source"]["commit"],
        "subdir": cfg["source"].get("subdir", "FAFA_SynCPR"),
        "checkpoint": checkpoint_identity(resolve_config_path(ckpt["path"])),
        "model": {k: ckpt[k] for k in ("model_name", "model_type", "fda_k", "fda_alpha",
                  "fd_margin", "use_soft", "transform", "image_size", "test_resize_hw") if k in ckpt},
    }
    stage("gallery_features", {**encoder, "detections": keys["detections"],
          "score_threshold": detector["score_threshold"],
          "dtype": cfg.get("runtime", {}).get("gallery_feature_dtype", "float16")})
    stage("query_features", {**encoder, "localization": keys["localization"]})
    stage("component_scores", {"schema": "soft-fda-v1", "gallery": keys["gallery_features"],
          "queries": keys["query_features"], "fda_k": int(ckpt.get("fda_k", 6))})
    return directories, {"keys": keys, "stages": fingerprints}


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def device_from(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but unavailable")
    return device


def image_path(row: dict[str, Any], index: int) -> Path:
    value = row.get("path")
    if not isinstance(value, str) or not value:
        raise KeyError(f"Gallery row {index} has no usable 'path': {row!r}")
    path = (ROOT / value).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def build_gallery_index(gallery: Sequence[dict[str, Any]]) -> dict[Any, int]:
    result: dict[Any, int] = {}
    for i, row in enumerate(gallery):
        if "image_id" not in row:
            raise KeyError(f"Gallery row {i} has no image_id")
        image_id = row["image_id"]
        if image_id in result:
            raise ValueError(f"Duplicate gallery image_id: {image_id!r}")
        result[image_id] = i
    return result


def parse_query_targets(query: dict[str, Any], qi: int) -> list[QueryTarget]:
    subjects = query.get("subjects")
    if not isinstance(subjects, list) or not subjects:
        raise ValueError(f"Query {qi}: subjects must be a non-empty list")

    relation_text = str(query.get("relation_text") or "").strip()
    query_text = str(query.get("text") or "").strip()
    targets: list[QueryTarget] = []

    for si, subject in enumerate(subjects):
        if not isinstance(subject, dict):
            raise TypeError(f"Query {qi} subject {si} must be an object")

        modify_text = str(subject.get("modify_text") or "").strip()
        if not modify_text:
            modify_text = relation_text or query_text
        if not modify_text:
            raise ValueError(f"Query {qi} subject {si}: no usable modification text")

        select_text = str(subject.get("select_text") or "").strip()
        if not select_text:
            # Only a localization fallback. This text is never used as FAFA's
            # final retrieval caption when select_text is present.
            select_text = str(subject.get("modify_text") or "").strip() or query_text
        if not select_text:
            raise ValueError(f"Query {qi} subject {si}: no usable selection text")

        targets.append(
            QueryTarget(
                modify_text=modify_text,
                select_text=select_text,
                subject_id=subject.get("subject_id"),
            )
        )

    return targets


def is_generated_python_artifact(path: str) -> bool:
    """Return True only for Python bytecode/cache paths, never real source files."""
    normalized = path.strip().strip('"').replace("\\", "/")
    return normalized.lower().endswith((".pyc", ".pyo"))


def tracked_dirty(checkout: Path) -> str:
    """Report tracked source changes while ignoring generated Python bytecode."""
    output = subprocess.check_output(
        ["git", "-C", str(checkout), "status", "--porcelain", "--untracked-files=no"],
        text=True,
    )
    dirty: list[str] = []
    for line in output.splitlines():
        if not line.strip():
            continue
        if len(line) < 4:
            dirty.append(line)
            continue
        path = line[3:].strip()
        status_paths = path.split(" -> ")
        if status_paths and all(is_generated_python_artifact(x) for x in status_paths):
            continue
        dirty.append(line)
    return "\n".join(dirty).strip()


def ensure_official_source(cfg: dict[str, Any]) -> Path:
    """Verify the pinned source without networking or accepting real source edits."""
    source = cfg["source"]
    checkout = resolve_config_path(str(source["local_checkout"]))
    subdir = checkout / str(source.get("subdir", "FAFA_SynCPR"))
    expected = str(source["commit"])

    if not checkout.is_dir():
        raise FileNotFoundError(
            f"Missing pinned FAFA source: {rel(checkout)}. "
            "Run the baseline through run_baseline.py first."
        )
    actual = subprocess.check_output(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual != expected:
        raise RuntimeError(
            f"Source commit mismatch: expected {expected}, got {actual}. "
            "Re-run download_checkpoint.py to repair the pinned checkout."
        )
    dirty = tracked_dirty(checkout)
    if dirty:
        raise RuntimeError(
            f"Pinned FAFA source has tracked local modifications: {rel(checkout)}\n{dirty}"
        )
    if not subdir.is_dir():
        raise FileNotFoundError(subdir)
    return subdir


def configure_runtime_cache(cfg: dict[str, Any]) -> Path:
    ckpt = cfg["checkpoint"]
    cache_root = resolve_config_path(str(ckpt["cache_root"]))
    marker = resolve_config_path(str(ckpt["runtime_assets_marker"]))
    if not marker.is_file():
        raise FileNotFoundError(
            f"Missing FAFA runtime-assets marker: {rel(marker)}. "
            "Run download_checkpoint.py first."
        )
    try:
        marker_data = json.loads(marker.read_text(encoding="utf-8"))
    except Exception as error:
        raise RuntimeError(f"Invalid FAFA runtime-assets marker: {rel(marker)}") from error
    expected = {
        "source_commit": str(cfg["source"]["commit"]),
        "model_name": str(ckpt.get("model_name", "blip2_fafa_cpr")),
        "model_type": str(ckpt.get("model_type", "pretrain")),
        "cache_root": rel(cache_root),
    }
    if any(marker_data.get(key) != value for key, value in expected.items()):
        raise RuntimeError(
            f"Stale FAFA runtime-assets marker: {rel(marker)}. "
            "Re-run download_checkpoint.py for the current config."
        )
    files = marker_data.get("files")
    if not isinstance(files, list) or not files:
        raise RuntimeError(
            f"FAFA runtime-assets marker has no cache inventory: {rel(marker)}. "
            "Re-run download_checkpoint.py."
        )
    for item in files:
        if not isinstance(item, dict):
            raise RuntimeError(f"Invalid FAFA cache inventory entry in {rel(marker)}")
        rel_path = item.get("path")
        size = item.get("size")
        if not isinstance(rel_path, str) or not isinstance(size, int):
            raise RuntimeError(f"Invalid FAFA cache inventory entry in {rel(marker)}")
        asset = cache_root / rel_path
        if not asset.is_file() or asset.stat().st_size != size:
            raise RuntimeError(
                f"Missing/stale FAFA runtime asset: {rel(asset)}. "
                "Re-run download_checkpoint.py to repair the cache."
            )
    os.environ["TORCH_HOME"] = str(cache_root / "torch")
    os.environ["HF_HOME"] = str(cache_root / "huggingface")
    os.environ["XDG_CACHE_HOME"] = str(cache_root / "xdg")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    return marker


@contextmanager
def block_network_during_model_load():
    """Make missing FAFA runtime assets fail instead of downloading silently."""
    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex

    def blocked_connect(self, address):  # noqa: ANN001
        raise RuntimeError(
            f"Network access blocked during FAFA inference model loading: {address!r}. "
            "Run download_checkpoint.py to prepare every runtime asset first."
        )

    def blocked_connect_ex(self, address):  # noqa: ANN001
        blocked_connect(self, address)
        return 1

    socket.socket.connect = blocked_connect
    socket.socket.connect_ex = blocked_connect_ex
    try:
        yield
    finally:
        socket.socket.connect = original_connect
        socket.socket.connect_ex = original_connect_ex


def import_official(fafa_dir: Path):
    src = fafa_dir / "src"
    if not src.is_dir():
        raise FileNotFoundError(src)
    sys.path.insert(0, str(src))

    from data_utils import squarepad_transform_test, targetpad_transform  # type: ignore
    from lavis.models import load_model_and_preprocess  # type: ignore

    return squarepad_transform_test, targetpad_transform, load_model_and_preprocess


def load_fafa(
    cfg: dict[str, Any], fafa_dir: Path, device: torch.device
):
    configure_runtime_cache(cfg)
    squarepad_transform_test, targetpad_transform, load_model_and_preprocess = (
        import_official(fafa_dir)
    )
    ckpt_cfg = cfg["checkpoint"]
    ckpt_path = (ROOT / ckpt_cfg["path"]).resolve()
    if not ckpt_path.is_file():
        raise FileNotFoundError(
            f"Missing official FAFA checkpoint: {ckpt_path}\n"
            "Run `python methods/published/02_fafa_setmatch/download_checkpoint.py` from the repository root."
        )

    try:
        with block_network_during_model_load():
            model, _, txt_processors = load_model_and_preprocess(
                name=str(ckpt_cfg.get("model_name", "blip2_fafa_cpr")),
                model_type=str(ckpt_cfg.get("model_type", "pretrain")),
                is_eval=True,
                device=str(device),
            )
    except Exception as error:
        raise RuntimeError(
            "FAFA model construction failed in offline inference mode. "
            "Re-run download_checkpoint.py to prepare the official LAVIS/Transformers assets."
        ) from error

    # Avoid a second copy of the complete released checkpoint in GPU memory.
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict):
        if "model" in checkpoint:
            state_dict = checkpoint["model"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    incompatible = model.load_state_dict(state_dict, strict=False)
    model = model.to(device).eval()

    fda_k = int(ckpt_cfg.get("fda_k", 6))
    fda_alpha = float(ckpt_cfg.get("fda_alpha", 0.5))
    if hasattr(model, "fda_k"):
        model.fda_k = fda_k
    if hasattr(model, "fda_alpha"):
        model.fda_alpha = fda_alpha
    if hasattr(model, "use_soft"):
        model.use_soft = bool(ckpt_cfg.get("use_soft", True))

    transform_name = str(ckpt_cfg.get("transform", "squarepad")).lower()
    image_size = int(ckpt_cfg.get("image_size", 224))
    if transform_name == "squarepad":
        resize_hw = tuple(int(x) for x in ckpt_cfg.get("test_resize_hw", [384, 192]))
        preprocess = squarepad_transform_test(image_size, need_size=resize_hw)
    elif transform_name == "targetpad":
        preprocess = targetpad_transform(1.25, image_size)
    else:
        raise ValueError(f"Unsupported official FAFA transform: {transform_name}")

    load_info = {
        "missing_keys": list(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
    }
    return model, txt_processors, preprocess, ckpt_path, load_info


def choose_person_boxes(
    candidates: Sequence[BoxCandidate],
    score_threshold: float,
    min_required: int,
) -> list[BoxCandidate]:
    selected = [c for c in candidates if c.score >= score_threshold]
    if len(selected) < min_required:
        used = {c.box for c in selected}
        for candidate in candidates:
            if candidate.box in used:
                continue
            selected.append(candidate)
            used.add(candidate.box)
            if len(selected) >= min_required:
                break
    return selected


def save_candidate_cache(
    path: Path,
    gallery: Sequence[dict[str, Any]],
    candidates: Sequence[Sequence[BoxCandidate]],
) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row, boxes in zip(gallery, candidates):
            payload = {
                "image_id": row["image_id"],
                "boxes": [
                    {
                        "xyxy": list(candidate.box),
                        "score": candidate.score,
                        "fallback": candidate.fallback,
                    }
                    for candidate in boxes
                ],
            }
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def load_candidate_cache(
    path: Path, gallery: Sequence[dict[str, Any]]
) -> list[list[BoxCandidate]] | None:
    if not path.is_file():
        return None
    try:
        rows = load_jsonl(path)
    except (OSError, ValueError, TypeError):
        return None
    if len(rows) != len(gallery):
        return None

    result: list[list[BoxCandidate]] = []
    for expected, row in zip(gallery, rows):
        if row.get("image_id") != expected.get("image_id"):
            return None
        candidates = []
        for item in row.get("boxes", []):
            if not isinstance(item, dict):
                return None
            xyxy = item.get("xyxy")
            if not isinstance(xyxy, list) or len(xyxy) != 4:
                return None
            try:
                box = tuple(float(x) for x in xyxy)
                score = float(item["score"])
            except (TypeError, ValueError, KeyError):
                return None
            if not np.isfinite([*box, score]).all() or box[2] <= box[0] or box[3] <= box[1]:
                return None
            candidates.append(BoxCandidate(box, score, bool(item.get("fallback", False))))
        if not candidates:
            return None
        result.append(candidates)
    return result


def get_or_create_detection_cache(
    gallery: Sequence[dict[str, Any]],
    cache_path: Path,
    config_path: Path,
    detector_cfg: dict[str, Any],
    device: torch.device,
) -> tuple[list[list[BoxCandidate]], dict[str, Any]]:
    cached = load_candidate_cache(cache_path, gallery)
    if cached is None:
        # Process isolation both frees detector GPU memory and prevents its BERT
        # cache constants from redirecting FAFA's later offline model loading.
        subprocess.run([
            sys.executable, "-u", str(Path(__file__).with_name("groundingdino_adapter.py")),
            "--config", str(config_path), "--cache-path", str(cache_path), "--device", str(device),
        ], cwd=ROOT, check=True)
        cached = load_candidate_cache(cache_path, gallery)
        if cached is None:
            raise RuntimeError("Grounding DINO worker wrote an invalid candidate cache")
    else:
        print(f"[cache] Grounding DINO candidates: {rel(cache_path)}", flush=True)
    info = read_json(cache_path.with_suffix(".meta.json")) or {}
    info["checkpoint"] = rel(resolve_config_path(detector_cfg["checkpoint"]))
    return cached, info


def load_clip_selector(selector_cfg: dict[str, Any], device: torch.device):
    import clip  # type: ignore

    checkpoint = resolve_config_path(str(selector_cfg["checkpoint"]))
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"Missing query-selector CLIP checkpoint: {rel(checkpoint)}. "
            "Run download_checkpoint.py first."
        )
    model, preprocess = clip.load(str(checkpoint), device=device, jit=False)
    model.eval()
    return clip, model, preprocess


def select_subject_members(
    similarity: np.ndarray,
    candidates: Sequence[BoxCandidate],
    *,
    threshold: float,
    margin: float,
    max_members: int,
) -> tuple[list[list[int]], list[int]]:
    """Select disjoint query people without case labels or oracle cardinalities.

    First anchor every textual Subject with a maximum-sum Hungarian assignment,
    ignoring thresholds. Then expand from unused, non-fallback person crops if
    their CLIP similarity passes both an absolute threshold and the Subject's
    assigned *anchor* score minus ``margin``. Resolve contested extras by the
    highest similarity, breaking ties by Subject order.

    All Subjects follow the same policy, including single-person queries.
    """
    if similarity.ndim != 2 or not np.isfinite(similarity).all():
        raise ValueError("CLIP similarity must be a finite 2D matrix")
    nsubjects, npeople = similarity.shape
    if nsubjects < 1 or npeople < nsubjects or len(candidates) != npeople:
        raise ValueError("Need at least one candidate per Subject")
    if not np.isfinite([threshold, margin]).all() or margin < 0 or max_members < 1:
        raise ValueError("Invalid membership threshold, margin, or member cap")

    rows, cols = linear_sum_assignment(-similarity)
    anchors = [-1] * nsubjects
    for row, col in zip(rows, cols):
        anchors[int(row)] = int(col)
    if any(col < 0 for col in anchors):
        raise RuntimeError("Hungarian did not assign every Subject an anchor")
    members = [[col] for col in anchors]
    used = set(anchors)

    # Stable order ensures deterministic cap behavior when scores tie.
    for col in np.argsort(-similarity.max(axis=0), kind="stable"):
        col = int(col)
        if col in used or candidates[col].fallback:
            continue
        eligible = [
            row for row in range(nsubjects)
            if len(members[row]) < max_members
            and similarity[row, col] >= threshold
            and similarity[row, col] >= similarity[row, anchors[row]] - margin
        ]
        if eligible:
            owner = max(eligible, key=lambda row: (similarity[row, col], -row))
            members[owner].append(col)
            used.add(col)
    return members, anchors


@torch.no_grad()
def select_query_target_boxes(
    queries: Sequence[dict[str, Any]],
    query_targets: Sequence[Sequence[QueryTarget]],
    gallery: Sequence[dict[str, Any]],
    gallery_index: dict[Any, int],
    all_candidates: Sequence[Sequence[BoxCandidate]],
    localization_cfg: dict[str, Any],
    device: torch.device,
    cache_path: Path | None = None,
) -> tuple[list[list[list[tuple[float, float, float, float]]]], dict[str, Any]]:
    cached = read_json(cache_path) if cache_path is not None else None
    if isinstance(cached, dict):
        try:
            cached_boxes = cached["boxes"]
            if len(cached_boxes) != len(query_targets):
                raise ValueError("Query count mismatch")
            for subjects, targets in zip(cached_boxes, query_targets):
                if len(subjects) != len(targets):
                    raise ValueError("Subject count mismatch")
                for members in subjects:
                    if not members:
                        raise ValueError("No predicted members")
                    for box in members:
                        if (len(box) != 4 or not np.isfinite(box).all()
                                or box[2] <= box[0] or box[3] <= box[1]):
                            raise ValueError("Invalid box")
            stats = cached["stats"]
            count = sum(len(members) for subjects in cached_boxes for members in subjects)
            if not isinstance(stats, dict) or stats.get("predicted_total_query_components") != count:
                raise ValueError("Invalid localization stats")
            print(f"[cache] query localization: {rel(cache_path)}", flush=True)
            return cached_boxes, stats
        except (KeyError, TypeError, ValueError):
            pass
    selector_cfg = localization_cfg["query_selector"]
    detector_cfg = localization_cfg["detector"]
    clip_module, clip_model, clip_preprocess = load_clip_selector(selector_cfg, device)
    threshold = float(detector_cfg["score_threshold"])

    selected_per_query: list[list[list[tuple[float, float, float, float]]]] = []
    localization_scores: list[float] = []
    predicted_member_counts: list[int] = []
    extra_member_count = 0
    selected_fallback_anchors = 0
    low_candidate_queries = 0
    full_scene_fallback_slots = 0
    membership_threshold = float(selector_cfg["membership_threshold"])
    membership_margin = float(selector_cfg["membership_margin"])
    member_cap = int(selector_cfg["max_members_per_subject"])
    # Several queries may reuse a reference scene or a Subject description.
    # Keep these small CLIP embeddings until localization finishes.
    crop_cache = {}
    text_cache = {}

    for qi, (query, targets) in enumerate(
        progress_bar(
            zip(queries, query_targets),
            total=len(queries),
            desc="Select query targets",
            unit="query",
        )
    ):
        image_id = query.get("image_id")
        if image_id not in gallery_index:
            raise ValueError(f"Query {qi}: image_id not found in gallery: {image_id!r}")
        gi = gallery_index[image_id]
        boxes = choose_person_boxes(all_candidates[gi], threshold, len(targets))
        missing = max(0, len(targets) - len(boxes))
        if missing:
            low_candidate_queries += 1
            full_scene_fallback_slots += missing
        crop_key = (gi, len(targets))
        if crop_key not in crop_cache:
            path = image_path(gallery[gi], gi)
            with Image.open(path) as source:
                image = source.convert("RGB")
                # A detector miss must not abort the entire benchmark. Fill only
                # missing query-target slots with the full reference scene; this
                # uses no GT box/identity information and is explicitly logged.
                if missing:
                    full_box = (0.0, 0.0, float(image.width), float(image.height))
                    boxes = list(boxes) + [
                        BoxCandidate(full_box, -1.0, fallback=True) for _ in range(missing)
                    ]
                crop_tensors = [clip_preprocess(image.crop(candidate.box)) for candidate in boxes]
            crop_batch = torch.stack(crop_tensors).to(device)
            image_features = clip_model.encode_image(crop_batch).float()
            image_features = image_features / image_features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            crop_cache[crop_key] = (boxes, image_features)
        boxes, image_features = crop_cache[crop_key]
        text_key = tuple(target.select_text for target in targets)
        if text_key not in text_cache:
            text_tokens = clip_module.tokenize(list(text_key)).to(device)
            text_features = clip_model.encode_text(text_tokens).float()
            text_features = text_features / text_features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            text_cache[text_key] = text_features
        text_features = text_cache[text_key]
        sim = (text_features @ image_features.T).cpu().numpy()

        memberships, anchors = select_subject_members(
            sim,
            boxes,
            threshold=membership_threshold,
            margin=membership_margin,
            max_members=member_cap,
        )
        # Keep the Subject dimension: one Subject may map to several person crops.
        selected_per_query.append(
            [[boxes[person].box for person in members] for members in memberships]
        )
        predicted_member_counts.extend(len(members) for members in memberships)
        extra_member_count += sum(len(members) - 1 for members in memberships)
        selected_fallback_anchors += sum(boxes[col].fallback for col in anchors)
        localization_scores.extend(float(sim[i, col]) for i, col in enumerate(anchors))

    stats = {
        "selector": str(selector_cfg.get("backend", "openai_clip")),
        "selector_model": str(selector_cfg.get("model", "ViT-B/32")),
        "assignment": "hungarian_anchors_then_threshold_expansion",
        "membership_threshold": membership_threshold,
        "membership_margin": membership_margin,
        "max_members_per_subject": member_cap,
        "membership_policy": "case_agnostic_anchor_first_disjoint_extras",
        "uses_case_labels_or_identity_counts": False,
        "predicted_member_count_histogram": dict(sorted(Counter(predicted_member_counts).items())),
        "predicted_total_query_components": sum(predicted_member_counts),
        "extra_member_components": extra_member_count,
        "selected_fallback_anchors": selected_fallback_anchors,
        "mean_assigned_clip_similarity": (
            float(np.mean(localization_scores)) if localization_scores else None
        ),
        "min_assigned_clip_similarity": (
            float(np.min(localization_scores)) if localization_scores else None
        ),
        "queries_with_too_few_candidates": low_candidate_queries,
        "full_scene_fallback_slots": full_scene_fallback_slots,
        "shortage_policy": "pad missing query-target slots with the full reference scene",
    }
    if cache_path is not None:
        atomic_json(cache_path, {"boxes": selected_per_query, "stats": stats})
    return selected_per_query, stats


class PersonCropDataset(Dataset):
    """Decode an adjacent scene once per worker, then crop its person instances."""
    def crop(self, gi: int, box):
        if getattr(self, "_last_gi", None) != gi:
            path = image_path(self.gallery[gi], gi)
            with Image.open(path) as source:
                self._last_image = source.convert("RGB")
            self._last_gi = gi
        return self._last_image.crop(box)


class GalleryPersonDataset(PersonCropDataset):
    def __init__(
        self,
        gallery: Sequence[dict[str, Any]],
        selected_boxes: Sequence[Sequence[BoxCandidate]],
        preprocess,
    ):
        self.gallery = gallery
        self.preprocess = preprocess
        self.items: list[tuple[int, tuple[float, float, float, float]]] = []
        self.offsets = np.zeros(len(gallery) + 1, dtype=np.int64)
        for gi, boxes in enumerate(selected_boxes):
            for candidate in boxes:
                self.items.append((gi, candidate.box))
            self.offsets[gi + 1] = len(self.items)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> torch.Tensor:
        gi, box = self.items[index]
        return self.preprocess(self.crop(gi, box))


def load_gallery_feature_cache(cache_path: Path, expected_offsets: np.ndarray):
    try:
        offsets = np.load(cache_path.with_name("gallery_person_offsets.npy"), allow_pickle=False)
        features = np.load(cache_path, mmap_mode="r", allow_pickle=False)
        if (features.ndim == 3 and features.shape[0] == expected_offsets[-1]
                and features.shape[1] > 0 and features.shape[2] > 0
                and features.dtype in (np.float16, np.float32)
                and np.array_equal(offsets, expected_offsets)):
            return offsets, (int(features.shape[1]), int(features.shape[2]))
    except (OSError, ValueError, EOFError):
        pass
    return None


@torch.no_grad()
def encode_gallery_persons(
    model,
    dataset: GalleryPersonDataset,
    cache_path: Path,
    runtime: dict[str, Any],
    device: torch.device,
) -> tuple[Path, np.ndarray, tuple[int, int]]:
    offsets_path = cache_path.with_name("gallery_person_offsets.npy")
    cached = load_gallery_feature_cache(cache_path, dataset.offsets)
    if cached is not None:
        print(f"[cache] gallery person features: {rel(cache_path)}", flush=True)
        return cache_path, *cached

    loader = DataLoader(
        dataset,
        batch_size=int(runtime.get("image_batch_size", 64)),
        shuffle=False,
        num_workers=int(runtime.get("num_workers", 4)),
        pin_memory=(device.type == "cuda"),
    )

    feature_memmap = None
    token_dim: tuple[int, int] | None = None
    cursor = 0
    output_dtype = str(runtime.get("gallery_feature_dtype", "float16"))
    if output_dtype not in {"float16", "float32"}:
        raise ValueError("runtime.gallery_feature_dtype must be float16 or float32")
    np_dtype = np.float16 if output_dtype == "float16" else np.float32
    temp_path = cache_path.with_name(cache_path.name + ".partial")

    for images in progress_bar(
        loader, desc="Encode FAFA gallery persons", total=len(loader), unit="batch"
    ):
        images = images.to(device, non_blocking=(device.type == "cuda"))
        model_images = images.half() if device.type == "cuda" else images.float()
        image_features, _ = model.extract_target_features(model_images, mode="mean")
        image_features = image_features.float()
        if image_features.ndim == 2:
            image_features = image_features.unsqueeze(1)
        if image_features.ndim != 3:
            raise RuntimeError(
                f"Unexpected FAFA target feature shape: {tuple(image_features.shape)}"
            )

        if feature_memmap is None:
            token_dim = (int(image_features.shape[1]), int(image_features.shape[2]))
            feature_memmap = np.lib.format.open_memmap(
                temp_path,
                mode="w+",
                dtype=np_dtype,
                shape=(len(dataset), token_dim[0], token_dim[1]),
            )

        batch_np = image_features.cpu().numpy().astype(np_dtype, copy=False)
        if not np.isfinite(batch_np).all():
            raise FloatingPointError("FAFA gallery features contain NaN/Inf")
        feature_memmap[cursor : cursor + len(batch_np)] = batch_np
        cursor += len(batch_np)

    if feature_memmap is None or token_dim is None:
        raise RuntimeError("Gallery person dataset is empty")
    feature_memmap.flush()
    del feature_memmap
    atomic_save(offsets_path, dataset.offsets)
    os.replace(temp_path, cache_path)
    return cache_path, dataset.offsets, token_dim


class QueryTargetDataset(PersonCropDataset):
    def __init__(
        self,
        queries: Sequence[dict[str, Any]],
        targets: Sequence[Sequence[QueryTarget]],
        boxes: Sequence[Sequence[Sequence[tuple[float, float, float, float]]]],
        gallery: Sequence[dict[str, Any]],
        gallery_index: dict[Any, int],
        preprocess,
    ):
        self.gallery = gallery
        self.preprocess = preprocess
        self.items: list[tuple[int, tuple[float, float, float, float], str, int]] = []
        for qi, (query, query_targets, query_boxes) in enumerate(
            zip(queries, targets, boxes)
        ):
            gi = gallery_index[query["image_id"]]
            if len(query_targets) != len(query_boxes):
                raise AssertionError(f"Query {qi}: target/box count mismatch")
            for target, member_boxes in zip(query_targets, query_boxes, strict=True):
                if not member_boxes:
                    raise AssertionError(f"Query {qi}: Subject has no selected person")
                for box in member_boxes:
                    # FAFA remains a single-person encoder. Every predicted group
                    # member reuses its Subject's public modification text.
                    self.items.append((gi, box, target.modify_text, qi))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        gi, box, caption, owner = self.items[index]
        return self.preprocess(self.crop(gi, box)), caption, owner


def load_query_feature_cache(path: Path, expected_counts: Sequence[int]):
    try:
        with np.load(path, allow_pickle=False) as data:
            features, offsets = data["features"], data["offsets"]
        expected_offsets = np.concatenate([[0], np.cumsum(expected_counts)]).astype(np.int64)
        if (features.ndim != 2 or features.dtype != np.float32 or features.shape[1] < 1
                or features.shape[0] != expected_offsets[-1]
                or not np.array_equal(offsets, expected_offsets) or not np.isfinite(features).all()):
            return None
        return [features[int(a):int(b)] for a, b in zip(offsets[:-1], offsets[1:])]
    except (OSError, ValueError, EOFError, KeyError):
        return None


def save_query_feature_cache(path: Path, features: Sequence[np.ndarray]) -> None:
    components, offsets = flatten_query_features(features)
    temp = path.with_name(path.name + ".partial")
    with temp.open("wb") as handle:
        np.savez(handle, features=components, offsets=offsets)
    os.replace(temp, path)


@torch.no_grad()
def encode_query_targets(
    model,
    txt_processors,
    dataset: QueryTargetDataset,
    num_queries: int,
    runtime: dict[str, Any],
    device: torch.device,
) -> list[np.ndarray]:
    loader = DataLoader(
        dataset,
        batch_size=int(runtime.get("query_batch_size", 64)),
        shuffle=False,
        num_workers=int(runtime.get("num_workers", 4)),
        pin_memory=(device.type == "cuda"),
    )

    grouped: list[list[np.ndarray]] = [[] for _ in range(num_queries)]
    for images, captions, owners in progress_bar(
        loader, desc="Encode FAFA query targets", total=len(loader), unit="batch"
    ):
        images = images.to(device, non_blocking=(device.type == "cuda"))
        model_images = images.half() if device.type == "cuda" else images.float()
        processed = [txt_processors["eval"](str(caption)) for caption in captions]
        features = model.extract_features(
            {"image": model_images, "text_input": processed}
        ).multimodal_embeds
        features = features.float()
        if features.ndim == 3 and features.shape[1] == 1:
            features = features[:, 0]
        if features.ndim != 2:
            raise RuntimeError(
                f"Unexpected FAFA query feature shape: {tuple(features.shape)}"
            )
        features_np = features.cpu().numpy().astype(np.float32, copy=False)
        if not np.isfinite(features_np).all():
            raise FloatingPointError("FAFA query features contain NaN/Inf")
        for feature, owner in zip(features_np, owners.numpy().tolist()):
            grouped[int(owner)].append(feature)

    result: list[np.ndarray] = []
    for qi, group in enumerate(grouped):
        if not group:
            raise RuntimeError(f"Query {qi} has no FAFA target feature")
        result.append(np.stack(group, axis=0).astype(np.float32, copy=False))
    return result


def flatten_query_features(
    query_features: Sequence[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    offsets = np.zeros(len(query_features) + 1, dtype=np.int64)
    chunks: list[np.ndarray] = []
    for qi, features in enumerate(query_features):
        if features.ndim != 2 or features.shape[0] < 1:
            raise ValueError(f"Query {qi}: invalid feature shape {features.shape}")
        chunks.append(features.astype(np.float32, copy=False))
        offsets[qi + 1] = offsets[qi] + features.shape[0]
    return np.concatenate(chunks, axis=0), offsets


@torch.no_grad()
def compute_component_person_scores(
    component_features: np.ndarray,
    gallery_feature_path: Path,
    cache_path: Path,
    fda_k: int,
    person_batch_size: int,
    query_batch_size: int,
    device: torch.device,
) -> Path:
    """Cache official soft-FDA scores for every target component x gallery person.

    Gallery person features are transferred once per person batch; all query
    component blocks are then scored against that batch. This is much faster
    than re-streaming the whole gallery once per benchmark query.
    """
    if person_batch_size <= 0 or query_batch_size <= 0 or fda_k <= 0:
        raise ValueError("Scoring batch sizes and fda_k must be positive")
    gallery_features = np.load(gallery_feature_path, mmap_mode="r")
    if (component_features.ndim != 2 or gallery_features.ndim != 3
            or component_features.shape[1] != gallery_features.shape[2]
            or gallery_features.shape[1] < 1):
        raise ValueError("Query/gallery feature dimensions do not match")
    expected_shape = (len(component_features), int(gallery_features.shape[0]))
    if cache_path.is_file():
        try:
            cached = np.load(cache_path, mmap_mode="r", allow_pickle=False)
            if cached.shape == expected_shape and cached.dtype == np.float32:
                print(f"[cache] component-person scores: {rel(cache_path)}", flush=True)
                return cache_path
        except (OSError, ValueError, EOFError):
            pass

    temp_path = cache_path.with_name(cache_path.name + ".partial")
    scores = np.lib.format.open_memmap(
        temp_path,
        mode="w+",
        dtype=np.float32,
        shape=expected_shape,
    )
    q_all = torch.from_numpy(np.asarray(component_features)).to(
        device=device, dtype=torch.float32
    )

    for p_start in progress_bar(
        range(0, gallery_features.shape[0], person_batch_size),
        desc="FAFA component-person scores",
        total=(gallery_features.shape[0] + person_batch_size - 1) // person_batch_size,
        unit="person-batch",
    ):
        p_end = min(p_start + person_batch_size, gallery_features.shape[0])
        g = torch.from_numpy(np.array(gallery_features[p_start:p_end], copy=True)).to(
            device=device, dtype=torch.float32
        )
        block = torch.empty((len(component_features), p_end - p_start), device=device, dtype=torch.float32)
        for q_start in range(0, len(component_features), query_batch_size):
            q_end = min(q_start + query_batch_size, len(component_features))
            q = q_all[q_start:q_end]
            # q: [Q,D], g: [P,K,D] -> [Q,P,K]
            token_similarity = torch.einsum("qd,pkd->qpk", q, g)
            k = min(fda_k, int(token_similarity.shape[-1]))
            topk = torch.topk(token_similarity, k=k, dim=-1).values
            block[q_start:q_end] = topk.mean(dim=-1)
        # One synchronization/CPU transfer per person block, rather than one
        # per tiny query block. Keep the original top-k mean score unchanged.
        block_np = block.cpu().numpy()
        if not np.isfinite(block_np).all():
            raise FloatingPointError("FAFA component scores contain NaN/Inf")
        scores[:, p_start:p_end] = block_np
    scores.flush()
    del scores
    os.replace(temp_path, cache_path)
    return cache_path


def setmatch_image_score(
    matrix: np.ndarray,
    unmatched_score: float,
) -> float:
    """Mean over all query components, including fixed scores for missing people.

    Rectangular Hungarian selects exactly min(K, N) distinct real pairs. Missing
    scores are a constant for a given K and N, so this assignment also maximizes
    the final image score. Do not add optional dummy slots when N >= K.
    """
    num_targets, num_persons = matrix.shape
    if num_targets < 1:
        raise ValueError("SetMatch requires at least one target")
    if num_persons == 0:
        return float(unmatched_score)

    rows, cols = linear_sum_assignment(-matrix)
    num_matched = min(num_targets, num_persons)
    if len(rows) != num_matched:
        raise RuntimeError("Hungarian assignment has the wrong number of real pairs")
    assigned_sum = matrix[rows, cols].sum(dtype=np.float64)
    missing_sum = (num_targets - num_matched) * unmatched_score
    return float((assigned_sum + missing_sum) / num_targets)


def validate_setmatch_config(setmatch_cfg: dict[str, Any]) -> float:
    """Reject legacy min configs instead of silently using different semantics."""
    if setmatch_cfg.get("enabled", True) is not True:
        raise ValueError("This adapter requires setmatch.enabled: true")
    if setmatch_cfg.get("assignment", "hungarian") != "hungarian":
        raise ValueError("Partial SetMatch requires assignment: hungarian")
    if setmatch_cfg.get("aggregation", "mean") != "mean":
        raise ValueError("Partial SetMatch requires aggregation: mean; update config.yaml")
    unmatched_score = float(setmatch_cfg.get("unmatched_score", 0.0))
    if not np.isfinite(unmatched_score):
        raise ValueError("setmatch.unmatched_score must be finite")
    return unmatched_score


def build_person_index(offsets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if (
        offsets.ndim != 1
        or len(offsets) < 2
        or offsets.dtype.kind not in "iu"
        or offsets[0] != 0
        or np.any(offsets[1:] < offsets[:-1])
    ):
        raise ValueError("Gallery offsets must be nondecreasing integers starting at zero")
    counts = np.diff(offsets).astype(np.int64, copy=False)
    max_persons = int(counts.max(initial=0))
    index = np.full((len(counts), max_persons), -1, dtype=np.int64)
    for gi, count in enumerate(counts):
        if count:
            index[gi, :count] = np.arange(offsets[gi], offsets[gi + 1])
    return index, counts


def maximum_two_match_sum(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Exact batched two-row assignment to distinct columns (at least two)."""
    def top2(x: np.ndarray):
        idx = np.argpartition(-x, kth=1, axis=1)[:, :2]
        vals = np.take_along_axis(x, idx, axis=1)
        order = np.argsort(-vals, axis=1)
        idx = np.take_along_axis(idx, order, axis=1)
        vals = np.take_along_axis(vals, order, axis=1)
        return vals[:, 0], idx[:, 0], vals[:, 1]

    a1, ai1, a2 = top2(a)
    b1, bi1, b2 = top2(b)
    b_excl_a = np.where(bi1 != ai1, b1, b2)
    a_excl_b = np.where(ai1 != bi1, a1, a2)
    # Every optimum includes a row's best column: if neither best is occupied,
    # that row can improve its score by taking it. Check both possibilities.
    return np.maximum(a1 + b_excl_a, b1 + a_excl_b)


def setmatch_two_targets_all_images(
    target_person_scores: np.ndarray,
    person_index: np.ndarray,
    counts: np.ndarray,
    unmatched_score: float,
) -> np.ndarray:
    """Exact two-component specialization of Partial SetMatch.

    For two rows, the maximum-weight one-to-one assignment can be solved from each
    row's best/second-best columns, yielding exactly the same assignment
    objective as Hungarian while avoiding millions of tiny SciPy calls.
    """
    if target_person_scores.shape[0] != 2:
        raise ValueError("Expected exactly two target rows")

    out = np.full(len(counts), unmatched_score, dtype=np.float32)
    if target_person_scores.shape[1] == 0:
        return out

    safe_index = np.where(person_index >= 0, person_index, 0)
    values = target_person_scores[:, safe_index]  # [2, G, max_persons]
    values = np.asarray(values, dtype=np.float32)
    values[:, person_index < 0] = -np.inf

    one = counts == 1
    if np.any(one):
        real_idx = safe_index[one, 0]
        best_real = np.maximum(
            target_person_scores[0, real_idx], target_person_scores[1, real_idx]
        )
        # The one real person goes to the better component; the other is missing.
        out[one] = (best_real + unmatched_score) / 2.0

    many = counts >= 2
    if np.any(many):
        a = values[0, many]
        b = values[1, many]

        out[many] = maximum_two_match_sum(a, b) / 2.0
    return out


def aggregate_component_scores(
    component_score_path: Path,
    query_offsets: np.ndarray,
    gallery_offsets: np.ndarray,
    scores_path: Path,
    unmatched_score: float,
) -> None:
    component_scores = np.load(component_score_path, mmap_mode="r")
    person_index, counts = build_person_index(gallery_offsets)
    if (
        query_offsets.ndim != 1
        or len(query_offsets) < 2
        or query_offsets.dtype.kind not in "iu"
        or query_offsets[0] != 0
        or np.any(query_offsets[1:] <= query_offsets[:-1])
    ):
        raise ValueError("Query offsets must increase and start at zero")
    expected_shape = (int(query_offsets[-1]), int(gallery_offsets[-1]))
    if component_scores.shape != expected_shape:
        raise ValueError(f"Component score shape must be {expected_shape}, got {component_scores.shape}")
    if not np.isfinite(unmatched_score):
        raise ValueError("unmatched_score must be finite")
    scores = np.lib.format.open_memmap(
        scores_path.with_name(scores_path.name + ".partial"),
        mode="w+",
        dtype=np.float32,
        shape=(len(query_offsets) - 1, len(gallery_offsets) - 1),
    )

    for qi in progress_bar(
        range(len(query_offsets) - 1),
        desc="Partial SetMatch aggregation",
        total=len(query_offsets) - 1,
        unit="query",
    ):
        start, end = int(query_offsets[qi]), int(query_offsets[qi + 1])
        target_scores = np.asarray(component_scores[start:end], dtype=np.float32)
        num_targets = end - start

        if num_targets == 1:
            # One-target Hungarian == choose the best predicted person in each
            # gallery image. Exclude zero-person images from reduceat indices.
            scores[qi] = unmatched_score
            nonempty = counts > 0
            if np.any(nonempty):
                scores[qi, nonempty] = np.maximum.reduceat(
                    target_scores[0], gallery_offsets[:-1][nonempty]
                )
        elif num_targets == 2:
            scores[qi] = setmatch_two_targets_all_images(
                target_scores, person_index, counts, unmatched_score
            )
        else:
            scores[qi] = unmatched_score
            one = counts == 1
            if np.any(one):
                best_real = target_scores[:, person_index[one, 0]].max(axis=0)
                scores[qi, one] = (
                    best_real + (num_targets - 1) * unmatched_score
                ) / num_targets
            two = counts == 2
            if np.any(two):
                # Transpose K x 2 to 2 x K: each gallery person chooses a
                # different query component. This is the same exact objective.
                values = target_scores[:, person_index[two, :2]].transpose(1, 2, 0)
                best_sum = maximum_two_match_sum(values[:, 0], values[:, 1])
                scores[qi, two] = (
                    best_sum + (num_targets - 2) * unmatched_score
                ) / num_targets
            for gi in np.flatnonzero(counts >= 3):
                p_start, p_end = int(gallery_offsets[gi]), int(gallery_offsets[gi + 1])
                scores[qi, gi] = setmatch_image_score(
                    target_scores[:, p_start:p_end], unmatched_score
                )
    scores.flush()
    del scores
    os.replace(scores_path.with_name(scores_path.name + ".partial"), scores_path)


def score_all_queries(
    query_features: Sequence[np.ndarray],
    gallery_feature_path: Path,
    offsets: np.ndarray,
    component_score_path: Path,
    scores_path: Path,
    fda_k: int,
    person_batch_size: int,
    query_batch_size: int,
    unmatched_score: float,
    device: torch.device,
) -> None:
    components, query_offsets = flatten_query_features(query_features)
    compute_component_person_scores(
        component_features=components,
        gallery_feature_path=gallery_feature_path,
        cache_path=component_score_path,
        fda_k=fda_k,
        person_batch_size=person_batch_size,
        query_batch_size=query_batch_size,
        device=device,
    )
    # Persist row boundaries for future CPU-only aggregation/calibration of the
    # raw component scores. This records predictions, not ground-truth members.
    atomic_save(component_score_path.with_name("query_component_offsets.npy"), query_offsets)
    aggregate_component_scores(
        component_score_path=component_score_path,
        query_offsets=query_offsets,
        gallery_offsets=offsets,
        scores_path=scores_path,
        unmatched_score=unmatched_score,
    )


def validate_scores(scores_path: Path, nq: int, ng: int) -> None:
    scores = np.load(scores_path, mmap_mode="r")
    if scores.shape != (nq, ng):
        raise AssertionError(f"Wrong score shape: {scores.shape}, expected {(nq, ng)}")
    for start in range(0, nq, 256):
        block = np.asarray(scores[start : start + 256])
        if not np.isfinite(block).all():
            raise FloatingPointError(f"NaN/Inf in score rows {start}:{start + 256}")


def main() -> None:
    tracker = PhaseTracker(METHOD_ID, total=10)
    tracker.advance("Load config and manifests")
    print(f"FAFA + Partial SetMatch adapter: {ADAPTER_VERSION}", flush=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()

    config_path = resolve_config_path(args.config)
    cfg = load_yaml(config_path)
    method = str(cfg.get("method", METHOD_ID))
    if method != METHOD_ID:
        raise ValueError(f"Expected method={METHOD_ID!r}, got {method!r}")
    setmatch_cfg = cfg["setmatch"]
    unmatched_score = validate_setmatch_config(setmatch_cfg)

    output = (ROOT / cfg["output"]["dir"]).resolve()
    output.mkdir(parents=True, exist_ok=True)

    data_cfg = cfg.get("data", {})
    gallery_path = (ROOT / data_cfg.get("gallery_manifest", "data/gallery.jsonl")).resolve()
    queries_path = (ROOT / data_cfg.get("query_manifest", "data/queries.jsonl")).resolve()
    gallery = load_jsonl(gallery_path)
    queries = load_jsonl(queries_path)
    if not gallery or not queries:
        raise RuntimeError("Canonical gallery/query manifests must be non-empty")

    # Fail fast before detector/CLIP work. This verifies the pinned commit, rejects
    # real tracked source edits, validates the pre-warmed runtime inventory, and
    # configures offline caches. Generated Python bytecode changes are ignored.
    tracker.advance("Validate FAFA artifacts and build cache identity")
    fafa_dir = ensure_official_source(cfg)
    runtime_marker_for_cache = configure_runtime_cache(cfg)
    tracker.log(
        f"FAFA preflight passed: source={cfg['source']['commit'][:12]} "
        f"runtime_marker={rel(runtime_marker_for_cache)} guard={RUNTIME_GUARD_VERSION}"
    )

    ckpt_cfg = cfg["checkpoint"]
    if ckpt_cfg.get("use_soft", True) is not True:
        raise ValueError("This FAFA adapter supports only checkpoint.use_soft: true")
    ckpt_path = resolve_config_path(str(ckpt_cfg["path"]))
    localization_cfg = cfg["localization"]
    detector_cfg = localization_cfg["detector"]
    detector_checkout, detector_config, _ = detector_preflight(detector_cfg)
    stage_dirs, cache_fingerprint = build_stage_caches(
        cfg, gallery_path, queries_path, output, detector_config, detector_checkout,
    )
    cache_key = cache_fingerprint["keys"]["component_scores"]
    cache_dir = stage_dirs["component_scores"]

    gallery_index = build_gallery_index(gallery)
    query_targets = [parse_query_targets(q, qi) for qi, q in enumerate(queries)]

    runtime = cfg.get("runtime", {})
    device = device_from(str(runtime.get("device", "cuda")))
    detector_device_name = str(runtime.get("detector_device", str(device)))
    detector_device = device_from(detector_device_name)
    tracker.log(
        f"gallery={len(gallery):,} queries={len(queries):,} "
        f"targets={sum(len(x) for x in query_targets):,} device={device} "
        f"detector_device={detector_device} cache={cache_key}"
    )

    tracker.advance("Detect gallery person candidates")
    candidate_cache_path = stage_dirs["detections"] / "person_candidates.jsonl"
    all_candidates, detector_info = get_or_create_detection_cache(
        gallery,
        candidate_cache_path,
        config_path,
        detector_cfg,
        detector_device,
    )
    if detector_device.type == "cuda":
        torch.cuda.empty_cache()

    threshold = float(detector_cfg["score_threshold"])
    gallery_boxes = [
        choose_person_boxes(candidates, threshold, min_required=1)
        for candidates in all_candidates
    ]
    tracker.log(
        f"person candidates ready: {sum(len(x) for x in gallery_boxes):,} crops "
        f"across {len(gallery_boxes):,} gallery images (attention={detector_info.get('attention_backend')})"
    )

    tracker.advance("Select query members: Hungarian anchors + CLIP thresholds")
    selected_query_boxes, selector_stats = select_query_target_boxes(
        queries=queries,
        query_targets=query_targets,
        gallery=gallery,
        gallery_index=gallery_index,
        all_candidates=all_candidates,
        localization_cfg=localization_cfg,
        device=device,
        cache_path=stage_dirs["localization"] / "query_localization.json",
    )
    if device.type == "cuda":
        torch.cuda.empty_cache()
    tracker.log(
        f"query localization ready: mean_clip={selector_stats.get('mean_assigned_clip_similarity')} "
        f"components={selector_stats['predicted_total_query_components']} "
        f"extras={selector_stats['extra_member_components']} "
        f"fallback_slots={selector_stats.get('full_scene_fallback_slots')}"
    )

    tracker.advance("Load FAFA only if feature extraction is needed")
    gallery_dataset = GalleryPersonDataset(gallery, gallery_boxes, preprocess=None)
    gallery_feature_path = stage_dirs["gallery_features"] / "gallery_person_features.npy"
    query_feature_path = stage_dirs["query_features"] / "query_features.npz"
    cached_gallery = load_gallery_feature_cache(gallery_feature_path, gallery_dataset.offsets)
    query_counts = [sum(len(members) for members in subjects) for subjects in selected_query_boxes]
    query_features = load_query_feature_cache(query_feature_path, query_counts)
    model_info_path = stage_dirs["gallery_features"] / "model_load_info.json"
    model = txt_processors = preprocess = None
    model_loaded = cached_gallery is None or query_features is None
    if model_loaded:
        model, txt_processors, preprocess, ckpt_path, load_info = load_fafa(cfg, fafa_dir, device)
        atomic_json(model_info_path, load_info)
        tracker.log(f"FAFA loaded: missing_keys={len(load_info['missing_keys'])} "
                    f"unexpected_keys={len(load_info['unexpected_keys'])}")
    else:
        load_info = read_json(model_info_path) or {"missing_keys": None, "unexpected_keys": None}
        tracker.log("[cache] complete gallery/query features; skipping FAFA model construction")

    tracker.advance("Encode predicted gallery persons")
    gallery_dataset.preprocess = preprocess
    gallery_feature_path, offsets, token_dim = encode_gallery_persons(
        model=model, dataset=gallery_dataset, cache_path=gallery_feature_path,
        runtime=runtime, device=device,
    )
    tracker.log(f"gallery person features ready: persons={int(offsets[-1]):,} "
                f"tokens={token_dim[0]} dim={token_dim[1]} path={rel(gallery_feature_path)}")

    tracker.advance("Encode composed query targets")
    if query_features is None:
        query_dataset = QueryTargetDataset(
            queries=queries, targets=query_targets, boxes=selected_query_boxes,
            gallery=gallery, gallery_index=gallery_index, preprocess=preprocess,
        )
        query_features = encode_query_targets(
            model=model, txt_processors=txt_processors, dataset=query_dataset,
            num_queries=len(queries), runtime=runtime, device=device,
        )
        save_query_feature_cache(query_feature_path, query_features)
    else:
        tracker.log(f"[cache] composed query features: {rel(query_feature_path)}")
    tracker.log(f"query target features ready: {sum(len(x) for x in query_features):,} components")
    # The scoring path consumes cached embeddings only; do not keep the large
    # FAFA encoder on the GPU during component scoring and CPU Hungarian.
    del model, txt_processors, preprocess
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    tracker.advance("Compute component-person scores and Partial SetMatch aggregation")
    ckpt_cfg = cfg["checkpoint"]
    fda_k = int(ckpt_cfg.get("fda_k", 6))
    scores_path = output / "scores.npy"
    component_score_path = cache_dir / "component_person_scores.npy"
    score_all_queries(
        query_features=query_features,
        gallery_feature_path=gallery_feature_path,
        offsets=offsets,
        component_score_path=component_score_path,
        scores_path=scores_path,
        fda_k=fda_k,
        person_batch_size=int(runtime.get("score_person_batch_size", 2048)),
        query_batch_size=int(runtime.get("score_query_batch_size", 128)),
        unmatched_score=unmatched_score,
        device=device,
    )
    tracker.log(f"raw score matrix written: {rel(scores_path)}")

    tracker.advance("Validate scores and summarize benchmark metadata")
    validate_scores(scores_path, len(queries), len(gallery))

    case_counts = Counter(str(q.get("case", "UNKNOWN")).upper() for q in queries)
    target_hist = Counter(len(targets) for targets in query_targets)
    gallery_person_hist = Counter(len(boxes) for boxes in gallery_boxes)
    relation_fallbacks = sum(
        1
        for q in queries
        for subject in q["subjects"]
        if not str(subject.get("modify_text") or "").strip()
    )

    run = {
        "adapter_version": ADAPTER_VERSION,
        "method": method,
        "display_name": cfg.get("display_name", "FAFA + Partial SetMatch"),
        "group": cfg.get("group", "Published / SOTA Baselines"),
        "cpr_supervision": cfg.get("cpr_supervision", "No"),
        "paper": cfg.get("paper", {}),
        "official_source": {
            "repository": cfg["source"]["repository"],
            "commit": cfg["source"]["commit"],
            "subdir": cfg["source"].get("subdir", "FAFA_SynCPR"),
            "runtime_guard_version": RUNTIME_GUARD_VERSION,
            "python_bytecode_writes_disabled": True,
            "integrity_ignores_generated_python_bytecode_only": True,
            "preflight_before_localization": True,
        },
        "checkpoint": {
            "path": rel(ckpt_path),
            "status": cfg["checkpoint"].get("status", "OFFICIAL_RELEASED"),
            "source_url": cfg["checkpoint"].get("source_url"),
            "model_name": cfg["checkpoint"].get("model_name", "blip2_fafa_cpr"),
            "model_type": cfg["checkpoint"].get("model_type", "pretrain"),
            "load_strict": False,
            "missing_key_count": len(load_info["missing_keys"]) if load_info["missing_keys"] is not None else None,
            "unexpected_key_count": len(load_info["unexpected_keys"]) if load_info["unexpected_keys"] is not None else None,
            "model_loaded_this_run": model_loaded,
        },
        "fafa_scoring": {
            "use_soft": bool(ckpt_cfg.get("use_soft", True)),
            "fda_k": fda_k,
            "fda_alpha": float(ckpt_cfg.get("fda_alpha", 0.5)),
            "fd_margin": float(ckpt_cfg.get("fd_margin", 0.5)),
            "definition": "top-k mean similarity between composed query feature and FAFA target feature tokens",
            "target_token_count": token_dim[0],
            "embedding_dim": token_dim[1],
        },
        "localization": {
            "uses_gt_target_boxes": False,
            "uses_target_ids_for_box_selection": False,
            "uses_positive_labels_for_scoring": False,
            "person_detector": detector_cfg.get("backend"),
            "detector_weights": detector_info["checkpoint"],
            "detector_source_commit": detector_cfg["source"]["commit"],
            "attention_backend": detector_info.get("attention_backend"),
            "detector_protocol": {k: detector_cfg[k] for k in ("text_prompt", "box_threshold", "text_threshold", "nms_iou_threshold", "min_box_size_px", "max_persons_per_image")},
            "score_threshold": threshold,
            "candidate_cache": rel(candidate_cache_path),
            **selector_stats,
        },
        "setmatch": {
            "assignment": "maximum-weight one-to-one rectangular Hungarian",
            "real_pair_count": "min(query_components, gallery_persons)",
            "aggregation": "mean over all query components, including missing scores",
            "definition": "(sum of matched FAFA scores + (K - min(K, N)) * b) / K",
            "unmatched_score": unmatched_score,
            "unmatched_score_source": setmatch_cfg.get("unmatched_score_source", "fixed_config_value"),
            "no_partial_credit": False,
            "allows_optional_unmatching": False,
            "query_component_offsets": rel(component_score_path.with_name("query_component_offsets.npy")),
        },
        "data_schema": {
            "gallery": "gallery.jsonl image_id + path; person_ids are not used for scoring/localization",
            "query_reference": "queries.jsonl image_id -> gallery.image_id",
            "target_selection_text": "subjects[].select_text",
            "target_modification_text": "subjects[].modify_text; relation_text then query text fallback",
            "membership": "inferred from CLIP; never use case, IDs, or GT counts",
        },
        "query_case_counts": dict(case_counts),
        "query_target_count_histogram": {str(k): v for k, v in sorted(target_hist.items())},
        "query_predicted_member_count_histogram": {
            str(k): v for k, v in selector_stats["predicted_member_count_histogram"].items()
        },
        "gallery_predicted_person_count_histogram": {
            str(k): v for k, v in sorted(gallery_person_hist.items())
        },
        "relation_text_fallback_subjects": relation_fallbacks,
        "runtime": {
            "device": str(device),
            "detector_device": str(detector_device),
            "detector_batch_size": int(runtime.get("detector_batch_size", 2)),
            "detector_num_workers": int(runtime.get("detector_num_workers", 4)),
            "image_batch_size": int(runtime.get("image_batch_size", 64)),
            "query_batch_size": int(runtime.get("query_batch_size", 64)),
            "score_person_batch_size": int(runtime.get("score_person_batch_size", 2048)),
            "score_query_batch_size": int(runtime.get("score_query_batch_size", 128)),
            "num_workers": int(runtime.get("num_workers", 4)),
        },
        "config": rel(config_path),
        "cache": {
            "key": cache_key,
            "dir": rel(cache_dir),
            "fingerprint": cache_fingerprint,
            "stage_dirs": {name: rel(path) for name, path in stage_dirs.items()},
            "query_features": rel(query_feature_path),
        },
        "num_queries": len(queries),
        "num_gallery": len(gallery),
        "num_predicted_gallery_persons": int(offsets[-1]),
        "scores": rel(scores_path),
        "higher_is_better": True,
        "notes": [
            "Canonical gallery/query ordering is preserved.",
            "The query image remains in the score matrix; evaluate.py handles exclusion.",
            "FAFA is loaded from the authors' pinned official implementation and released checkpoint.",
            (
                "Pinned-source and runtime-asset integrity are preflighted before "
                "expensive detector/CLIP localization."
            ),
            (
                "Python bytecode writes are disabled; integrity checks ignore only "
                "generated .pyc/.pyo bytecode files and still reject real tracked "
                "source edits."
            ),
            "No CPR benchmark training, fine-tuning, checkpoint selection, or hyperparameter tuning is performed.",
            "No GT PIPA target boxes or identity-to-box mapping are used in the main adapter.",
            "Subject anchors are assigned without threshold; extra members are selected by absolute CLIP threshold and distance from the assigned anchor.",
            "The same case-agnostic query membership policy is applied to every Subject; public Subject slots are used but case and GT cardinality are not.",
            "Partial SetMatch maximizes the sum of min(K, N) one-to-one real pairs and averages over all K query components; each missing person contributes the fixed configured background score.",
            "RELATIONAL text is not modeled jointly by native single-person FAFA; relation_text is only a per-target fallback when modify_text is empty.",
        ],
    }

    tracker.log(
        f"score validation passed: shape=({len(queries):,}, {len(gallery):,}); cases={dict(case_counts)}"
    )

    tracker.advance("Write run metadata and final outputs")
    run_path = output / "run.json"
    with run_path.open("w", encoding="utf-8") as f:
        json.dump(run, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(f"Saved: {scores_path}  shape=({len(queries)}, {len(gallery)})", flush=True)
    print(f"Saved: {run_path}", flush=True)
    print(f"Evaluate: python evaluate.py --method {method}", flush=True)
    tracker.finish()


if __name__ == "__main__":
    main()
