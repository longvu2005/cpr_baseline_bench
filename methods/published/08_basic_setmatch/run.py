#!/usr/bin/env python3
"""P8: official BASIC pairwise scoring + predicted-person CPR SetMatch."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import pickle
import shutil
import socket
import subprocess
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

import numpy as np
import torch
import yaml
from PIL import Image

os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark_progress import PhaseTracker, progress_bar  # noqa: E402

from cpr_person.adapter import load_person_context, matching_metadata, manifest_identity
from cpr_person.cache import feature_config_hash
from cpr_person.detection import protocol_config, PROTOCOL_PATH
from cpr_person.matching import aggregate_people
from cpr_person.adapter import expand_targets
from cpr_person.detection import crop_person as crop_box
from cpr_person.matching import build_person_index as padded_person_indices
from cpr_person.penalties import basic_unmatched_bound
DEFAULT_CONFIG = Path(__file__).resolve().parent / "config.yaml"
METHOD_ID = "basic_setmatch"
ADAPTER_VERSION = "2026-10-12-shared-person-v1"
CACHE_SCHEMA = 1


@dataclass(frozen=True)
class QueryTarget:
    modify_text: str
    select_text: str
    subject_id: Any = None


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise TypeError(f"Expected YAML mapping: {path}")
    return data


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise TypeError(f"{path}:{line_no}: JSONL row must be an object")
            rows.append(row)
    return rows


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


def canonical_hash(payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".part")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temp, path)


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


def configure_basic_cache(cache_root: Path) -> None:
    home = cache_root / "home"
    if not home.is_dir():
        raise FileNotFoundError(f"Missing prepared BASIC cache home: {rel(home)}")
    os.environ["HOME"] = str(home)
    os.environ["TORCH_HOME"] = str(cache_root / "torch")
    os.environ["HF_HOME"] = str(cache_root / "huggingface")
    os.environ["XDG_CACHE_HOME"] = str(cache_root / "xdg")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"


@contextmanager
def block_network_model_load():
    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex

    def blocked_connect(self, address):  # noqa: ANN001
        raise RuntimeError(
            f"Network access blocked during BASIC inference model loading: {address!r}. "
            "Run download_checkpoint.py to prepare all model assets."
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


def verify_prepared_assets(
    cfg: dict[str, Any], config_path: Path
) -> tuple[Path, dict[str, Any], Path]:
    marker_path = resolve_path(str(cfg["runtime_assets"]["prepared_marker"]))
    marker = read_json(marker_path)
    if marker is None:
        raise FileNotFoundError(
            f"Missing/invalid prepared marker: {rel(marker_path)}. Run download_checkpoint.py first."
        )
    if marker.get("method") != METHOD_ID:
        raise RuntimeError(f"Prepared marker method mismatch: {marker.get('method')!r}")

    checkout = resolve_path(str(cfg["source"]["local_checkout"]))
    if not (checkout / ".git").is_dir():
        raise FileNotFoundError(f"Missing official i-CIR source checkout: {rel(checkout)}")
    expected_commit = str(cfg["source"]["commit"])
    actual_commit = subprocess.check_output(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual_commit != expected_commit:
        raise RuntimeError(
            f"Official source commit mismatch: expected {expected_commit}, got {actual_commit}"
        )
    dirty = tracked_dirty(checkout)
    if dirty:
        raise RuntimeError(
            f"Pinned official i-CIR source has tracked local modifications:\n{dirty}"
        )

    source_marker = marker.get("source", {})
    if source_marker.get("commit") != expected_commit:
        raise RuntimeError("Prepared marker was created for another official source commit")
    inventory = source_marker.get("tracked_inventory")
    if not isinstance(inventory, dict) or not inventory:
        raise RuntimeError("Prepared marker has no official source inventory")
    for relative, info in inventory.items():
        path = checkout / relative
        if not path.is_file():
            raise FileNotFoundError(f"Missing official source/resource file: {rel(path)}")
        if not isinstance(info, dict):
            raise RuntimeError(f"Invalid marker entry for {relative}")
        if int(info.get("size", -1)) != path.stat().st_size:
            raise RuntimeError(f"Official source/resource size changed: {rel(path)}")
        if str(info.get("sha256")) != sha256_file(path):
            raise RuntimeError(f"Official source/resource checksum changed: {rel(path)}")

    cache_root = resolve_path(str(cfg["runtime_assets"]["basic_model_cache_root"]))
    basic_marker = marker.get("basic", {})
    if basic_marker.get("model_cache_root") != rel(cache_root):
        raise RuntimeError("Prepared BASIC cache root does not match config.yaml")
    cache_files = basic_marker.get("model_cache_files")
    if not isinstance(cache_files, list) or not cache_files:
        raise RuntimeError("Prepared marker has no BASIC model cache inventory")
    for item in cache_files:
        if not isinstance(item, dict):
            raise RuntimeError("Invalid BASIC model cache marker entry")
        path = cache_root / str(item.get("path", ""))
        expected_size = int(item.get("size", -1))
        if not path.is_file() or path.stat().st_size != expected_size:
            raise RuntimeError(
                f"Missing/stale BASIC model cache asset: {rel(path)}. "
                "Re-run download_checkpoint.py."
            )

    for name in ("selector", "detector"):
        item = marker.get(name, {})
        path = resolve_path(str(cfg["runtime_assets"][name]["path"]))
        if not path.is_file() or path.stat().st_size <= 0:
            raise FileNotFoundError(f"Missing {name} artifact: {rel(path)}")
        if int(item.get("size", -1)) != path.stat().st_size:
            raise RuntimeError(f"{name} artifact size does not match prepared marker")
        if str(item.get("sha256")) != sha256_file(path):
            raise RuntimeError(f"{name} artifact checksum does not match prepared marker")

    configure_basic_cache(cache_root)
    return checkout, marker, marker_path


def import_official(checkout: Path):
    checkout_str = str(checkout)
    if checkout_str not in sys.path:
        sys.path.insert(0, checkout_str)
    utils_features = importlib.import_module("utils_features")
    utils_retrieval = importlib.import_module("utils_retrieval")
    run_retrieval = importlib.import_module("run_retrieval")
    return utils_features, utils_retrieval, run_retrieval


def verify_and_get_official_preset(cfg: dict[str, Any], run_retrieval) -> dict[str, Any]:
    presets = getattr(run_retrieval, "METHOD_PRESETS", None)
    if not isinstance(presets, dict) or "basic" not in presets:
        raise RuntimeError("Pinned official code has no METHOD_PRESETS['basic']")
    preset = dict(presets["basic"])
    preset.pop("description", None)
    expected = dict(cfg["basic"]["expected_preset"])
    if preset != expected:
        raise RuntimeError(
            "Pinned official BASIC preset no longer matches config.yaml. "
            "Refusing to run a silently changed method.\n"
            f"official={preset}\nexpected={expected}"
        )
    return preset


def build_gallery_index(gallery: Sequence[dict[str, Any]]) -> dict[Any, int]:
    result: dict[Any, int] = {}
    for gi, row in enumerate(gallery):
        if "image_id" not in row:
            raise KeyError(f"Gallery row {gi} missing image_id")
        image_id = row["image_id"]
        if image_id in result:
            raise ValueError(f"Duplicate gallery image_id: {image_id!r}")
        result[image_id] = gi
    return result


def image_path(row: dict[str, Any], index: int) -> Path:
    value = row.get("path")
    if not isinstance(value, str) or not value.strip():
        raise KeyError(f"Gallery row {index} has no usable path")
    path = resolve_path(value)
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def query_gallery_indices(
    queries: Sequence[dict[str, Any]], gallery_index: dict[Any, int]
) -> np.ndarray:
    result: list[int] = []
    for qi, query in enumerate(queries):
        image_id = query.get("image_id")
        if image_id not in gallery_index:
            raise ValueError(f"Query row {qi}: image_id {image_id!r} missing from gallery")
        result.append(gallery_index[image_id])
    return np.asarray(result, dtype=np.int64)


def parse_query_targets(query: dict[str, Any], qi: int) -> list[QueryTarget]:
    subjects = query.get("subjects")
    if not isinstance(subjects, list) or not subjects:
        raise ValueError(f"Query {qi}: subjects must be a non-empty list")

    full_text = str(query.get("text") or "").strip()
    relation_text = str(query.get("relation_text") or "").strip()
    targets: list[QueryTarget] = []

    for si, subject in enumerate(subjects):
        if not isinstance(subject, dict):
            raise TypeError(f"Query {qi} subject {si} must be an object")

        if relation_text:
            modify_text = full_text or relation_text
        else:
            modify_text = str(subject.get("modify_text") or "").strip() or full_text
        if not modify_text:
            raise ValueError(f"Query {qi} subject {si}: no usable BASIC modification text")

        select_text = str(subject.get("select_text") or "").strip()
        if not select_text:
            select_text = (
                str(subject.get("modify_text") or "").strip()
                or full_text
                or relation_text
            )
        if not select_text:
            raise ValueError(f"Query {qi} subject {si}: no usable localization text")

        targets.append(
            QueryTarget(
                modify_text=modify_text,
                select_text=select_text,
                subject_id=subject.get("subject_id"),
            )
        )
    return targets


def build_cache_key(
    *,
    person,
    config_path: Path,
    gallery_path: Path,
    queries_path: Path,
    prepared_marker_path: Path,
) -> tuple[str, dict[str, Any]]:
    payload = {
        "schema": CACHE_SCHEMA,
        "adapter_version": ADAPTER_VERSION,
        "feature_config_sha256": feature_config_hash(config_path),
        "shared_protocol_sha256": sha256_file(PROTOCOL_PATH),
        "detections": person.detections.fingerprint,
        "memberships": person.localization.fingerprint,
        "gallery_sha256": sha256_file(gallery_path),
        "queries_sha256": sha256_file(queries_path),
        "prepared_marker_sha256": sha256_file(prepared_marker_path),
    }
    return canonical_hash(payload)[:20], payload


def load_feature_cache(
    path: Path, expected_rows: int, label: str
) -> np.ndarray | None:
    if not path.is_file():
        return None
    try:
        array = np.load(path, mmap_mode="r", allow_pickle=False)
    except Exception as error:
        print(f"Ignoring invalid {label} cache: {error}", flush=True)
        return None
    if array.ndim != 2 or array.shape[0] != expected_rows or array.dtype.kind != "f":
        print(f"Ignoring incompatible {label} cache: shape={array.shape}", flush=True)
        return None
    if not np.isfinite(np.asarray(array[: min(32, len(array))], dtype=np.float32)).all():
        print(f"Ignoring non-finite {label} cache", flush=True)
        return None
    print(f"Using {label} cache: {rel(path)}", flush=True)
    return array


@torch.no_grad()
def encode_person_crops(
    *,
    gallery: Sequence[dict[str, Any]],
    offsets: np.ndarray,
    boxes: np.ndarray,
    model,
    preprocess,
    device: torch.device,
    batch_size: int,
    encoder_kind: str,
    desc: str,
) -> np.ndarray:
    outputs: list[np.ndarray] = []
    pending: list[torch.Tensor] = []

    def flush() -> None:
        nonlocal pending
        if not pending:
            return
        batch = torch.stack(pending).to(device, non_blocking=True)
        if encoder_kind == "selector":
            feat = model.encode_image(batch).float()
        elif encoder_kind == "basic":
            feat = model.encode_image(batch).float()
        else:
            raise ValueError(encoder_kind)
        feat = feat / feat.norm(dim=-1, keepdim=True)
        outputs.append(feat.detach().cpu().numpy().astype(np.float16, copy=False))
        pending = []

    for gi, row in enumerate(
        progress_bar(gallery, total=len(gallery), desc=desc, unit="image")
    ):
        with Image.open(image_path(row, gi)) as handle:
            rgb = handle.convert("RGB")
            start, end = int(offsets[gi]), int(offsets[gi + 1])
            for person_index in range(start, end):
                pending.append(preprocess(crop_box(rgb, boxes[person_index])))
                if len(pending) >= batch_size:
                    flush()
    flush()
    result = np.concatenate(outputs, axis=0) if outputs else np.empty((0, 0), np.float16)
    if result.shape[0] != int(offsets[-1]):
        raise RuntimeError(
            f"{desc} feature count mismatch: {result.shape[0]} vs {int(offsets[-1])}"
        )
    return result


def save_npy_atomic(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".part")
    with temp.open("wb") as handle:
        np.save(handle, array, allow_pickle=False)
    os.replace(temp, path)


@torch.no_grad()
def encode_full_scenes(
    *,
    model,
    preprocess,
    gallery: Sequence[dict[str, Any]],
    gallery_indices: Sequence[int],
    device: torch.device,
    batch_size: int,
    desc: str,
) -> dict[int, np.ndarray]:
    unique = list(dict.fromkeys(int(x) for x in gallery_indices))
    result: dict[int, np.ndarray] = {}
    pending_tensors: list[torch.Tensor] = []
    pending_indices: list[int] = []

    def flush() -> None:
        nonlocal pending_tensors, pending_indices
        if not pending_tensors:
            return
        batch = torch.stack(pending_tensors).to(device, non_blocking=True)
        feat = model.encode_image(batch).float()
        feat = feat / feat.norm(dim=-1, keepdim=True)
        array = feat.detach().cpu().numpy().astype(np.float32, copy=False)
        for gi, vector in zip(pending_indices, array):
            result[gi] = vector
        pending_tensors = []
        pending_indices = []

    for gi in progress_bar(unique, total=len(unique), desc=desc, unit="image"):
        with Image.open(image_path(gallery[gi], gi)) as handle:
            pending_tensors.append(preprocess(handle.convert("RGB")))
        pending_indices.append(gi)
        if len(pending_tensors) >= batch_size:
            flush()
    flush()
    return result


def ensure_corpus_features(
    *,
    utils_features,
    checkout: Path,
    cfg: dict[str, Any],
    cache_dir: Path,
    model,
    tokenizer,
    device: torch.device,
) -> tuple[Path, Path]:
    corpus_dir = cache_dir / "corpus"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    resources = cfg["basic"]["official_resources"]
    names = [
        (str(cfg["basic"]["expected_preset"]["specified_corpus"]), resources["positive_corpus"]),
        (str(cfg["basic"]["expected_preset"]["specified_ncorpus"]), resources["negative_corpus"]),
    ]
    outputs: list[Path] = []
    for name, relative in names:
        output = corpus_dir / f"{name}.pkl"
        if output.is_file() and output.stat().st_size > 0:
            print(f"Using BASIC corpus feature cache: {rel(output)}", flush=True)
        else:
            print(f"Encoding official BASIC corpus: {relative}", flush=True)
            utils_features.save_corpus_features(
                model=model,
                tokenizer=tokenizer,
                corpus_path=str(checkout / str(relative)),
                save_file=str(output),
                device=device,
                batch_size=int(cfg["runtime"]["basic_text_batch_size"]),
            )
        outputs.append(output)
    return outputs[0], outputs[1]


def contextualized_text_cache(
    *,
    utils_features,
    checkout: Path,
    cfg: dict[str, Any],
    cache_dir: Path,
    model,
    tokenizer,
    device: torch.device,
    texts: Sequence[str],
) -> tuple[list[str], np.ndarray]:
    unique = list(dict.fromkeys(texts))
    text_list_path = cache_dir / "basic_context_texts.json"
    feature_path = cache_dir / "basic_context_text_features.npy"

    cached_texts: list[str] | None = None
    if text_list_path.is_file():
        try:
            value = json.loads(text_list_path.read_text(encoding="utf-8"))
            if isinstance(value, list) and all(isinstance(x, str) for x in value):
                cached_texts = value
        except Exception:
            cached_texts = None
    if cached_texts == unique and feature_path.is_file():
        features = np.load(feature_path, mmap_mode="r", allow_pickle=False)
        if features.ndim == 2 and features.shape[0] == len(unique):
            print(f"Using BASIC contextualized-text cache: {rel(feature_path)}", flush=True)
            return unique, features

    positive_corpus = checkout / str(cfg["basic"]["official_resources"]["positive_corpus"])
    dim = 768
    if unique:
        # Exact official contextualization helper: first 100 positive-corpus words,
        # both word orders, then average.
        features_t = utils_features.contextualize(
            model,
            tokenizer,
            dim,
            unique,
            str(positive_corpus),
            int(cfg["basic"]["contextualization_corpus_words"]),
            device,
            batch_size=int(cfg["runtime"]["basic_text_batch_size"]),
        )
        features = features_t.detach().cpu().numpy().astype(np.float32, copy=False)
    else:
        features = np.empty((0, dim), dtype=np.float32)

    save_npy_atomic(feature_path, features)
    text_list_path.write_text(json.dumps(unique, ensure_ascii=False) + "\n", encoding="utf-8")
    return unique, np.load(feature_path, mmap_mode="r", allow_pickle=False)


class BasicScorer:
    """Score-returning transcription of the pinned official BASIC branch."""

    def __init__(
        self,
        *,
        args: SimpleNamespace,
        database_features: torch.Tensor,
        text_corpus_pos: torch.Tensor,
        text_corpus_neg: torch.Tensor,
        laion_mean_path: Path,
        synthetic_path: Path,
        device: torch.device,
    ) -> None:
        self.args = args
        self.device = device

        database_features = database_features.float().to(device)
        database_features = database_features / database_features.norm(dim=1, keepdim=True)
        text_corpus_pos = text_corpus_pos.float().to(device)
        text_corpus_neg = text_corpus_neg.float().to(device)
        text_corpus_pos = text_corpus_pos / text_corpus_pos.norm(dim=1, keepdim=True)
        text_corpus_neg = text_corpus_neg / text_corpus_neg.norm(dim=1, keepdim=True)

        mean_img = database_features.mean(0, keepdim=True)
        mean_txt = text_corpus_pos.mean(0, keepdim=True)

        if bool(args.standardize_features):
            if bool(args.use_laion_mean):
                with laion_mean_path.open("rb") as handle:
                    data = pickle.load(handle)
                value = data["laion_1m_mean"]
                if torch.is_tensor(value):
                    mean_img = value.to(device=device, dtype=torch.float32)
                else:
                    mean_img = torch.as_tensor(value, dtype=torch.float32, device=device)
                mean_img = mean_img.reshape(1, -1)

            centered_database = database_features - mean_img
            centered_pos = text_corpus_pos - mean_txt
            centered_neg = text_corpus_neg - mean_txt
        else:
            centered_database = database_features
            centered_pos = text_corpus_pos
            centered_neg = text_corpus_neg

        if bool(args.project_features):
            aa = float(args.aa)
            sa = centered_pos.T @ centered_pos / (centered_pos.size(0) - 1)
            sb = centered_neg.T @ centered_neg / (centered_neg.size(0) - 1)
            # Intentionally mirror upstream exactly: scalar 1e-5 is added to C.
            c = (1.0 - aa) * sa - aa * sb + 1e-5
            eigenvalues, eigenvectors_asc = torch.linalg.eigh(c)
            eigenvalues = eigenvalues.flip(dims=[0])
            vy_t = -eigenvectors_asc.flip(dims=[1]).T
            nc = int(args.num_principal_components_for_projection)
            nc = min(nc, int((eigenvalues > 0).sum().item()))
            vy_t = vy_t[:nc]
            projection = vy_t.T @ vy_t
        else:
            projection = torch.eye(centered_database.shape[1], device=device)

        generated = np.load(synthetic_path, allow_pickle=True).item()
        image_generated = torch.tensor(
            generated["image_features"], dtype=torch.float32, device=device
        )
        text_generated = torch.tensor(
            generated["text_features"], dtype=torch.float32, device=device
        )
        image_generated = image_generated / image_generated.norm(dim=1, keepdim=True)
        text_generated = text_generated / text_generated.norm(dim=1, keepdim=True)
        if bool(args.standardize_features):
            image_generated = image_generated - mean_img
            text_generated = text_generated - mean_txt

        sim_img_gen = image_generated @ image_generated.T
        sim_text_gen = text_generated @ image_generated.T
        self.sim_img_min = float(sim_img_gen.cpu().min().item())
        self.sim_text_min = float(sim_text_gen.cpu().min().item())
        if abs(self.sim_img_min) <= 1e-12:
            raise RuntimeError("Official BASIC synthetic image similarity minimum is zero")

        self.mean_img = mean_img
        self.mean_txt = mean_txt
        self.centered_database = centered_database
        self.projection = projection

    @torch.no_grad()
    def score(
        self, image_features: torch.Tensor, text_features: torch.Tensor
    ) -> torch.Tensor:
        args = self.args
        image_features = image_features.float().to(self.device)
        text_features = text_features.float().to(self.device)
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        if bool(args.standardize_features):
            centered_image = image_features - self.mean_img
            centered_text = text_features - self.mean_txt
        else:
            centered_image = image_features
            centered_text = text_features

        projected_image = centered_image @ self.projection

        if bool(args.do_query_expansion):
            if self.centered_database.shape[0] < 25:
                raise RuntimeError(
                    "Official BASIC hardcodes top-25 query expansion, but the database "
                    f"contains only {self.centered_database.shape[0]} person features."
                )
            init_sim = projected_image @ self.centered_database.T
            top_values, top_indices = torch.topk(init_sim, 25)
            # Mirror upstream CPU weighting exactly.
            top_features = self.centered_database.cpu()[top_indices.cpu()]
            top_features = torch.cat(
                (top_features, centered_image.unsqueeze(1).cpu()), dim=1
            )
            top_values = torch.cat(
                (
                    top_values,
                    torch.ones((top_values.shape[0], 1), device=self.device),
                ),
                dim=1,
            )
            top_values = torch.exp(0.1 * top_values)
            top_values = top_values / top_values.sum(dim=1, keepdim=True)
            top_features = top_features * top_values.unsqueeze(-1).cpu()
            expanded = top_features.sum(dim=1).to(self.device)
            projected_image = expanded @ self.projection

        sim_img = (projected_image @ self.centered_database.T).cpu()
        sim_text = (centered_text @ self.centered_database.T).cpu()

        if bool(args.normalize_similarities):
            denom = abs(self.sim_img_min)
            sim_text = (sim_text - self.sim_text_min) / denom
            sim_img = (sim_img - self.sim_img_min) / denom

        sim_text = torch.clamp(sim_text, min=0)
        sim_img = torch.clamp(sim_img, min=0)
        return sim_text * sim_img - float(args.harris_lambda) * (sim_text + sim_img) ** 2


def build_basic_args(preset: dict[str, Any], checkout: Path) -> SimpleNamespace:
    values = dict(preset)
    values["method"] = "basic"
    values["backbone"] = "clip"
    values["norm"] = True
    values["path_to_synthetic_data"] = str(
        checkout / "synthetic_data"
    )
    return SimpleNamespace(**values)


def load_corpus_tensor(utils_features, path: Path, device: torch.device) -> torch.Tensor:
    features, _ = utils_features.read_corpus(str(path), device, norm=True)
    return features


@contextmanager
def working_directory(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def official_parity_check(
    *,
    cfg: dict[str, Any],
    checkout: Path,
    cache_dir: Path,
    utils_features,
    utils_retrieval,
    basic_args: SimpleNamespace,
    corpus_pos_path: Path,
    corpus_neg_path: Path,
    image_features: np.ndarray,
    text_features: np.ndarray,
    database_features: np.ndarray,
    device: torch.device,
) -> None:
    if not bool(cfg["basic"].get("official_parity_check", True)):
        return

    num_q = min(int(cfg["basic"]["parity_queries"]), len(image_features))
    num_db = min(int(cfg["basic"]["parity_database_size"]), len(database_features))
    if num_q <= 0 or num_db < 25:
        raise RuntimeError("Not enough real features for the official BASIC parity check")

    q_img = torch.from_numpy(np.asarray(image_features[:num_q], dtype=np.float32)).to(device)
    q_txt = torch.from_numpy(np.asarray(text_features[:num_q], dtype=np.float32)).to(device)
    db = torch.from_numpy(np.asarray(database_features[:num_db], dtype=np.float32)).to(device)

    pos = load_corpus_tensor(utils_features, corpus_pos_path, device)
    neg = load_corpus_tensor(utils_features, corpus_neg_path, device)
    sample_scorer = BasicScorer(
        args=basic_args,
        database_features=db,
        text_corpus_pos=pos,
        text_corpus_neg=neg,
        laion_mean_path=checkout / str(cfg["basic"]["official_resources"]["laion_mean"]),
        synthetic_path=checkout
        / str(cfg["basic"]["official_resources"]["synthetic_normalization"]),
        device=device,
    )
    adapter_ranks = torch.argsort(sample_scorer.score(q_img, q_txt), descending=True)

    fixture = cache_dir / "official_parity_fixture"
    corpus_dir = fixture / "features" / "clip_features" / "corpus"
    mean_dir = fixture / "data" / "laion_mean"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    mean_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(corpus_pos_path, corpus_dir / corpus_pos_path.name)
    shutil.copyfile(corpus_neg_path, corpus_dir / corpus_neg_path.name)
    shutil.copyfile(
        checkout / str(cfg["basic"]["official_resources"]["laion_mean"]),
        mean_dir / "laion_1m_mean_clip.pkl",
    )

    official_args = SimpleNamespace(**vars(basic_args))
    official_args.path_to_synthetic_data = str(
        checkout / "synthetic_data"
    )
    with working_directory(fixture):
        official_ranks = utils_retrieval.calculate_rankings(
            official_args, q_img, q_txt, db
        )
    if not torch.equal(adapter_ranks.cpu(), official_ranks.cpu()):
        mismatch = int((adapter_ranks.cpu() != official_ranks.cpu()).sum().item())
        raise RuntimeError(
            "BASIC score adapter failed official rank parity check "
            f"({mismatch} rank positions differ). Refusing to produce benchmark scores."
        )
    print(
        f"[ok] BASIC adapter rank parity with pinned official calculate_rankings "
        f"({num_q} queries x {num_db} database features)",
        flush=True,
    )


def collapse_setmatch(target_scores, padded, counts, offsets, unmatched_score):
    return aggregate_people(target_scores, offsets, unmatched_score)


def save_run_metadata(
    *,
    path: Path,
    cfg: dict[str, Any],
    cache_key: str,
    cache_payload: dict[str, Any],
    preset: dict[str, Any],
    source_commit: str,
    gallery_count: int,
    query_count: int,
    person_count: int,
    detector_fallback_scenes: int,
    selector_stats: dict[str, Any],
    parity_checked: bool,
    shared_person_protocol: dict,
) -> None:
    payload = {
        "method": METHOD_ID,
        "display_name": str(cfg["display_name"]),
        "group": str(cfg["group"]),
        "cpr_supervision": str(cfg["cpr_supervision"]),
        # Required by the repository-wide score-matrix contract and evaluate.py.
        # BASIC parity is checked with descending=True, so larger scores are better.
        "higher_is_better": True,
        "adapter_version": ADAPTER_VERSION,
        "setmatch": matching_metadata(cfg, score_space="basic_official_score"),
        "shared_person_protocol": shared_person_protocol,
        "paper": cfg["paper"],
        "source": {
            "repository": str(cfg["source"]["repository"]),
            "commit": source_commit,
            "status": "OFFICIAL_RELEASED_SOURCE",
        },
        "checkpoint": {
            "status": "TRAINING_FREE_NO_BASIC_NEURAL_CHECKPOINT",
            "backbone": "OpenCLIP ViT-L/14 OpenAI pretrained",
            "resources": "official pinned i-CIR corpora/LAION mean/synthetic normalization",
        },
        "official_basic": {
            "method": "basic",
            "backbone": "clip",
            "preset": preset,
            "contextualization_corpus_words": int(
                cfg["basic"]["contextualization_corpus_words"]
            ),
            "score_adapter_parity_checked": bool(parity_checked),
        },
        "cpr_adapter": {
            "person_detection": cfg["localization"]["detector"],
            "query_selector": {
                **cfg["localization"]["query_selector"],
                "backend": str(cfg["runtime_assets"]["selector"]["backend"]),
                "model": str(cfg["runtime_assets"]["selector"]["model"]),
            },
            "query_text_adapter": cfg["query_text_adapter"],
            "setmatch": cfg["setmatch"],
            "uses_gt_boxes": False,
            "uses_target_ids": False,
            "uses_positive_labels": False,
        },
        "stats": {
            "gallery_images": int(gallery_count),
            "queries": int(query_count),
            "gallery_person_candidates": int(person_count),
            "detector_full_scene_fallback_scenes": int(detector_fallback_scenes),
            "query_selector": selector_stats,
        },
        "cache": {
            "key": cache_key,
            "fingerprint": cache_payload,
        },
        "score_contract": {
            "shape": [int(query_count), int(gallery_count)],
            "higher_is_better": True,
            "query_image_removed_inside_method": False,
            "finite_only": True,
        },
    }
    payload["data_fingerprint"] = manifest_identity(payload["shared_person_protocol"])
    write_json_atomic(path, payload)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run P8 BASIC + SetMatch")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()

    config_path = resolve_path(args.config)
    cfg = load_yaml(config_path)
    cfg["runtime_assets"]["detector"]["path"] = protocol_config()["detector"]["checkpoint"]
    cfg["runtime_assets"]["selector"]["path"] = protocol_config()["selector"]["checkpoint"]
    if str(cfg.get("method")) != METHOD_ID:
        raise RuntimeError(f"config method must be {METHOD_ID!r}")

    tracker = PhaseTracker(METHOD_ID, total=7)

    tracker.advance("Validate manifests, pinned source, and prepared artifacts")
    gallery_path = resolve_path(str(cfg["data"]["gallery_manifest"]))
    queries_path = resolve_path(str(cfg["data"]["query_manifest"]))
    gallery = load_jsonl(gallery_path)
    queries = load_jsonl(queries_path)
    if not gallery or not queries:
        raise RuntimeError("Gallery/query manifests must be non-empty")
    gallery_index = build_gallery_index(gallery)
    query_gallery_idx = query_gallery_indices(queries, gallery_index)
    query_targets = [parse_query_targets(query, qi) for qi, query in enumerate(queries)]
    max_targets = max(len(x) for x in query_targets)

    checkout, prepared_marker, prepared_marker_path = verify_prepared_assets(cfg, config_path)
    device = device_from(str(cfg["runtime"]["device"]))
    person = load_person_context(gallery, queries, gallery_path, queries_path, device)
    cache_key, cache_payload = build_cache_key(
        person=person,
        config_path=config_path,
        gallery_path=gallery_path,
        queries_path=queries_path,
        prepared_marker_path=prepared_marker_path,
    )
    cache_dir = resolve_path(str(cfg["cache"]["root"])) / cache_key
    cache_dir.mkdir(parents=True, exist_ok=True)

    utils_features, utils_retrieval, run_retrieval = import_official(checkout)
    preset = verify_and_get_official_preset(cfg, run_retrieval)
    basic_args = build_basic_args(preset, checkout)
    tracker.log(
        f"gallery={len(gallery)} queries={len(queries)} max_targets={max_targets} "
        f"cache={rel(cache_dir)}"
    )

    tracker.advance("Consume shared gallery detections")
    device = device_from(str(cfg["runtime"]["device"]))
    offsets, boxes = person.detections.offsets, person.detections.boxes
    detector_fallback = np.array([person.detections.fallback[start:end].any()
                                  for start, end in zip(offsets[:-1], offsets[1:])])
    tracker.advance("Consume shared Subject memberships")
    selected_refs = [[member["global_person_index"] for si, member in person.localization.flat(qi)]
                     for qi in range(len(queries))]
    query_targets = expand_targets(query_targets, person.localization)
    selector_stats = person.localization.stats

    tracker.advance("Load official BASIC CLIP and encode person/text features")
    with block_network_model_load():
        basic_bundle = utils_features.load_model("clip", device)
    basic_model = basic_bundle["model"]
    basic_preprocess = basic_bundle["preprocess"]
    basic_tokenizer = basic_bundle["tokenizer"]

    corpus_pos_path, corpus_neg_path = ensure_corpus_features(
        utils_features=utils_features,
        checkout=checkout,
        cfg=cfg,
        cache_dir=cache_dir,
        model=basic_model,
        tokenizer=basic_tokenizer,
        device=device,
    )

    basic_gallery_cache_path = cache_dir / "basic_gallery_person_features.npy"
    basic_gallery_features = load_feature_cache(
        basic_gallery_cache_path, int(offsets[-1]), "BASIC gallery-person"
    )
    if basic_gallery_features is None:
        basic_gallery_array = encode_person_crops(
            gallery=gallery,
            offsets=offsets,
            boxes=boxes,
            model=basic_model,
            preprocess=basic_preprocess,
            device=device,
            batch_size=int(cfg["runtime"]["basic_image_batch_size"]),
            encoder_kind="basic",
            desc="Encode BASIC persons",
        )
        save_npy_atomic(basic_gallery_cache_path, basic_gallery_array)
        basic_gallery_features = np.load(
            basic_gallery_cache_path, mmap_mode="r", allow_pickle=False
        )

    all_modify_texts = [
        target.modify_text for targets in query_targets for target in targets
    ]
    context_texts, context_features = contextualized_text_cache(
        utils_features=utils_features,
        checkout=checkout,
        cfg=cfg,
        cache_dir=cache_dir,
        model=basic_model,
        tokenizer=basic_tokenizer,
        device=device,
        texts=all_modify_texts,
    )
    context_index = {text: idx for idx, text in enumerate(context_texts)}

    temp_fallback_gallery_indices = [
        int(query_gallery_idx[qi])
        for qi, refs in enumerate(selected_refs)
        if any(ref is None for ref in refs)
    ]
    full_scene_basic = encode_full_scenes(
        model=basic_model,
        preprocess=basic_preprocess,
        gallery=gallery,
        gallery_indices=temp_fallback_gallery_indices,
        device=device,
        batch_size=int(cfg["runtime"]["basic_image_batch_size"]),
        desc="BASIC fallback scenes",
    )

    tracker.advance("Build official BASIC scorer and verify parity")
    pos_tensor = load_corpus_tensor(utils_features, corpus_pos_path, device)
    neg_tensor = load_corpus_tensor(utils_features, corpus_neg_path, device)
    db_tensor = torch.from_numpy(
        np.asarray(basic_gallery_features, dtype=np.float32)
    ).to(device)
    scorer = BasicScorer(
        args=basic_args,
        database_features=db_tensor,
        text_corpus_pos=pos_tensor,
        text_corpus_neg=neg_tensor,
        laion_mean_path=checkout / str(cfg["basic"]["official_resources"]["laion_mean"]),
        synthetic_path=checkout
        / str(cfg["basic"]["official_resources"]["synthetic_normalization"]),
        device=device,
    )

    # Bound the unchanged official score using only released external resources.
    cfg["setmatch"]["unmatched_score"] = basic_unmatched_bound(
        float(scorer.mean_img.norm().item()), float(scorer.mean_txt.norm().item()),
        scorer.sim_img_min, scorer.sim_text_min, float(basic_args.harris_lambda))
    parity_img: list[np.ndarray] = []
    parity_txt: list[np.ndarray] = []
    needed_parity = int(cfg["basic"]["parity_queries"])
    for qi, refs in enumerate(selected_refs):
        for ti, ref_idx in enumerate(refs):
            if ref_idx is None:
                img_feat = full_scene_basic[int(query_gallery_idx[qi])]
            else:
                img_feat = np.asarray(basic_gallery_features[ref_idx], dtype=np.float32)
            text_feat = np.asarray(
                context_features[context_index[query_targets[qi][ti].modify_text]],
                dtype=np.float32,
            )
            parity_img.append(img_feat)
            parity_txt.append(text_feat)
            if len(parity_img) >= needed_parity:
                break
        if len(parity_img) >= needed_parity:
            break

    official_parity_check(
        cfg=cfg,
        checkout=checkout,
        cache_dir=cache_dir,
        utils_features=utils_features,
        utils_retrieval=utils_retrieval,
        basic_args=basic_args,
        corpus_pos_path=corpus_pos_path,
        corpus_neg_path=corpus_neg_path,
        image_features=np.stack(parity_img, axis=0),
        text_features=np.stack(parity_txt, axis=0),
        database_features=np.asarray(
            basic_gallery_features[
                : min(int(cfg["basic"]["parity_database_size"]), len(basic_gallery_features))
            ],
            dtype=np.float32,
        ),
        device=device,
    )

    tracker.advance("Score complete gallery with BASIC + SetMatch")
    output_dir = resolve_path(str(cfg["output"]["dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    scores_path = output_dir / "scores.npy"
    temp_scores = output_dir / "scores.npy.part"
    temp_scores.unlink(missing_ok=True)
    score_mmap = np.lib.format.open_memmap(
        temp_scores,
        mode="w+",
        dtype=np.float32,
        shape=(len(queries), len(gallery)),
    )

    padded, counts = padded_person_indices(offsets)
    unmatched_score = float(cfg["setmatch"]["unmatched_score"])
    query_batch_size = int(cfg["runtime"]["score_query_batch_size"])

    for batch_start in progress_bar(
        range(0, len(queries), query_batch_size),
        total=(len(queries) + query_batch_size - 1) // query_batch_size,
        desc="BASIC + SetMatch",
        unit="batch",
    ):
        batch_end = min(batch_start + query_batch_size, len(queries))
        flat_images: list[np.ndarray] = []
        flat_texts: list[np.ndarray] = []
        slices: list[tuple[int, int, int]] = []
        cursor = 0

        for qi in range(batch_start, batch_end):
            refs = selected_refs[qi]
            targets = query_targets[qi]
            start = cursor
            for ti, ref_idx in enumerate(refs):
                if ref_idx is None:
                    img_feat = full_scene_basic[int(query_gallery_idx[qi])]
                else:
                    img_feat = np.asarray(
                        basic_gallery_features[ref_idx], dtype=np.float32
                    )
                txt_feat = np.asarray(
                    context_features[context_index[targets[ti].modify_text]],
                    dtype=np.float32,
                )
                flat_images.append(img_feat)
                flat_texts.append(txt_feat)
                cursor += 1
            slices.append((qi, start, cursor))

        image_batch = torch.from_numpy(np.stack(flat_images).astype(np.float32)).to(device)
        text_batch = torch.from_numpy(np.stack(flat_texts).astype(np.float32)).to(device)
        person_scores = scorer.score(image_batch, text_batch).numpy()
        if not np.isfinite(person_scores).all():
            raise RuntimeError(
                f"Non-finite BASIC person scores in query batch {batch_start}:{batch_end}"
            )

        for qi, start, end in slices:
            scene_scores = collapse_setmatch(
                person_scores[start:end],
                padded,
                counts,
                offsets,
                unmatched_score,
            )
            if scene_scores.shape != (len(gallery),) or not np.isfinite(scene_scores).all():
                raise RuntimeError(f"Invalid SetMatch scene scores for query {qi}")
            score_mmap[qi] = scene_scores

        score_mmap.flush()

    del score_mmap
    os.replace(temp_scores, scores_path)

    tracker.advance("Validate outputs and write run metadata")
    saved = np.load(scores_path, mmap_mode="r", allow_pickle=False)
    expected_shape = (len(queries), len(gallery))
    if saved.shape != expected_shape:
        raise RuntimeError(f"scores.npy shape={saved.shape}, expected={expected_shape}")
    # Scan in chunks without forcing the whole ~200 MB matrix into RAM.
    scan_rows = 128
    for start in range(0, len(saved), scan_rows):
        if not np.isfinite(np.asarray(saved[start : start + scan_rows])).all():
            raise RuntimeError(f"scores.npy contains NaN/Inf near row {start}")

    run_path = output_dir / "run.json"
    save_run_metadata(
        path=run_path,
        cfg=cfg,
        cache_key=cache_key,
        cache_payload=cache_payload,
        preset=preset,
        source_commit=str(cfg["source"]["commit"]),
        gallery_count=len(gallery),
        query_count=len(queries),
        person_count=int(offsets[-1]),
        detector_fallback_scenes=int(detector_fallback.sum()),
        selector_stats=selector_stats,
        parity_checked=bool(cfg["basic"].get("official_parity_check", True)),
        shared_person_protocol=person.metadata(),
    )

    tracker.log(f"saved {rel(scores_path)}")
    tracker.log(f"saved {rel(run_path)}")
    tracker.finish()


if __name__ == "__main__":
    main()
