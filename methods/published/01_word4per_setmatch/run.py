#!/usr/bin/env python3
"""Word4Per + SetMatch adapter for the CPR baseline benchmark.

The Word4Per retrieval path is imported from the authors' pinned old_project
implementation. Because the benchmark contains full scene images rather than
person crops, person instances and query-target boxes are predicted. No GT
identity-to-box mapping, target_ids, or positive labels are used for scoring.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark_progress import PhaseTracker, progress_bar  # noqa: E402
from cpr_person.adapter import load_person_context, matching_metadata, manifest_identity
from cpr_person.cache import feature_config_hash
from cpr_person.detection import protocol_config, PROTOCOL_PATH
from cpr_person.detection import BoxCandidate
from cpr_person.matching import aggregate_component_scores
DEFAULT_CONFIG = Path(__file__).resolve().parent / "config.yaml"
METHOD_ID = "word4per_setmatch"
ADAPTER_VERSION = "2026-10-12-shared-person-v1"


@dataclass(frozen=True)
class QueryTarget:
    modify_text: str
    select_text: str
    subject_id: Any = None


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


def build_cache_key(
    *,
    person,
    config_path: Path,
    gallery_path: Path,
    queries_path: Path,
    checkpoint_paths: Sequence[Path],
) -> tuple[str, dict[str, Any]]:
    payload = {
        "adapter_version": ADAPTER_VERSION,
        "feature_config_sha256": feature_config_hash(config_path),
        "shared_protocol_sha256": sha256_file(PROTOCOL_PATH),
        "detections": person.detections.fingerprint,
        "memberships": person.localization.fingerprint,
        "gallery_sha256": sha256_file(gallery_path),
        "queries_sha256": sha256_file(queries_path),
        "checkpoints": [checkpoint_identity(path) for path in checkpoint_paths],
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    key = hashlib.sha256(raw).hexdigest()[:16]
    return key, payload


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


def ensure_official_source(cfg: dict[str, Any]) -> Path:
    """Verify the source prepared by download_checkpoint.py without networking."""
    source = cfg["source"]
    checkout = resolve_config_path(str(source["local_checkout"]))
    old_project = checkout / str(source.get("subdir", "old_project"))
    expected = str(source["commit"])

    if not checkout.is_dir():
        raise FileNotFoundError(
            f"Missing pinned Word4Per source: {rel(checkout)}. "
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
    dirty = subprocess.check_output(
        ["git", "-C", str(checkout), "status", "--porcelain", "--untracked-files=no"],
        text=True,
    ).strip()
    if dirty:
        raise RuntimeError(
            f"Pinned Word4Per source has tracked local modifications: {rel(checkout)}\n{dirty}"
        )
    if not old_project.is_dir():
        raise FileNotFoundError(old_project)
    return old_project


def import_official(old_project: Path):
    sys.path.insert(0, str(old_project))
    from datasets.bases import tokenize  # type: ignore
    from datasets.build import build_transforms  # type: ignore
    from model import build_model  # type: ignore
    from model.word4per import IM2TEXT  # type: ignore
    from utils.checkpoint import Checkpointer_Toword  # type: ignore
    from utils.iotools import load_train_configs  # type: ignore
    from utils.simple_tokenizer import SimpleTokenizer  # type: ignore

    return (
        tokenize,
        build_transforms,
        build_model,
        IM2TEXT,
        Checkpointer_Toword,
        load_train_configs,
        SimpleTokenizer,
    )


def validate_stage2_args(args, stage2_cfg: Path) -> None:
    expected = {
        "dataset_name": "CUHK-PEDES",
        "loss_names": "sdm+id",
        "toword_loss": "text",
        "batch_size": 128,
        "num_epoch": 60,
    }
    problems: list[str] = []
    for key, value in expected.items():
        if getattr(args, key, None) != value:
            problems.append(
                f"{key}: expected {value!r}, got {getattr(args, key, None)!r}"
            )
    if str(getattr(args, "optimizer", "")).lower() != "adamw":
        problems.append(
            f"optimizer: expected 'AdamW', got {getattr(args, 'optimizer', None)!r}"
        )
    try:
        lr = float(getattr(args, "lr", None))
    except (TypeError, ValueError):
        lr = float("nan")
    if not (abs(lr - 1e-4) <= 1e-12):
        problems.append(f"lr: expected 0.0001, got {getattr(args, 'lr', None)!r}")
    if bool(getattr(args, "MLM", False)):
        problems.append("MLM: expected false for the documented Stage-2 recipe")
    if problems:
        details = "\n".join(f"  - {item}" for item in problems)
        raise RuntimeError(
            f"Word4Per Stage-2 config does not match the pinned official recipe: {rel(stage2_cfg)}\n"
            f"{details}"
        )


def load_word4per(cfg: dict[str, Any], old_project: Path, device: torch.device):
    (
        tokenize,
        build_transforms,
        build_model,
        IM2TEXT,
        Checkpointer_Toword,
        load_train_configs,
        SimpleTokenizer,
    ) = import_official(old_project)

    stage2_cfg = (ROOT / cfg["checkpoint"]["stage2_config"]).resolve()
    stage2_ckpt = (ROOT / cfg["checkpoint"]["stage2"]).resolve()
    if not stage2_cfg.is_file():
        raise FileNotFoundError(f"Missing Stage-2 config: {stage2_cfg}")
    if not stage2_ckpt.is_file():
        raise FileNotFoundError(f"Missing Stage-2 checkpoint: {stage2_ckpt}")

    args = load_train_configs(str(stage2_cfg))
    validate_stage2_args(args, stage2_cfg)
    args.training = False
    backbone = str(args.pretrain_choice)
    configured_backbone = str(cfg["checkpoint"]["base_clip_model"])
    if backbone != configured_backbone:
        raise RuntimeError(
            f"Word4Per Stage-2 backbone mismatch: {backbone!r} != {configured_backbone!r}"
        )
    base_clip_path = resolve_config_path(str(cfg["checkpoint"]["base_clip"]))
    if not base_clip_path.is_file():
        raise FileNotFoundError(
            f"Missing Word4Per base CLIP checkpoint: {rel(base_clip_path)}. "
            "Run download_checkpoint.py first."
        )
    # old_project accepts a local OpenAI CLIP checkpoint path. Replacing the
    # model name prevents build_model() from downloading weights at inference.
    args.pretrain_choice = str(base_clip_path)
    model = build_model(args, num_classes=int(cfg["checkpoint"].get("num_classes", 11003)))
    args.pretrain_choice = backbone

    if backbone == "ViT-L/14":
        dim = 768
    elif backbone.startswith("ViT-B"):
        dim = 512
    else:
        raise ValueError(f"Unsupported Word4Per backbone: {backbone}")

    img2text = IM2TEXT(
        embed_dim=dim,
        middle_dim=512,
        output_dim=dim,
        n_layer=int(args.mlp_depth),
    )
    Checkpointer_Toword(model, img2text).load(f=str(stage2_ckpt))

    model.to(device).eval()
    img2text.to(device).eval()
    transform = build_transforms(img_size=args.img_size, is_train=False)
    tokenizer = SimpleTokenizer()
    split_ind = tokenize("*", tokenizer)[1]

    return (
        model,
        img2text,
        transform,
        tokenizer,
        tokenize,
        split_ind,
        args,
        stage2_cfg,
        stage2_ckpt,
    )


class GalleryPersonDataset(Dataset):
    def __init__(
        self,
        gallery: Sequence[dict[str, Any]],
        selected_boxes: Sequence[Sequence[BoxCandidate]],
        transform,
    ):
        self.gallery = gallery
        self.transform = transform
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
        path = image_path(self.gallery[gi], gi)
        with Image.open(path) as source:
            crop = source.convert("RGB").crop(box)
            return self.transform(crop)


@torch.no_grad()
def encode_gallery_persons(
    model,
    dataset: GalleryPersonDataset,
    cache_path: Path,
    runtime: dict[str, Any],
    device: torch.device,
) -> tuple[Path, np.ndarray, int]:
    offsets_path = cache_path.with_name("gallery_person_offsets.npy")
    if cache_path.is_file() and offsets_path.is_file():
        offsets = np.load(offsets_path)
        features = np.load(cache_path, mmap_mode="r")
        if (
            features.ndim == 2
            and features.shape[0] == len(dataset)
            and offsets.shape == dataset.offsets.shape
            and np.array_equal(offsets, dataset.offsets)
        ):
            print(f"[cache] gallery person features: {rel(cache_path)}", flush=True)
            return cache_path, offsets, int(features.shape[1])

    loader = DataLoader(
        dataset,
        batch_size=int(runtime.get("image_batch_size", 64)),
        shuffle=False,
        num_workers=int(runtime.get("num_workers", 4)),
        pin_memory=(device.type == "cuda"),
    )

    feature_memmap = None
    feature_dim: int | None = None
    cursor = 0
    output_dtype = str(runtime.get("gallery_feature_dtype", "float16")).lower()
    if output_dtype not in {"float16", "float32"}:
        raise ValueError("runtime.gallery_feature_dtype must be float16 or float32")
    np_dtype = np.float16 if output_dtype == "float16" else np.float32

    for images in progress_bar(
        loader, desc="Encode Word4Per gallery persons", total=len(loader), unit="batch"
    ):
        images = images.to(device, non_blocking=(device.type == "cuda"))
        features = F.normalize(model.encode_image(images).float(), p=2, dim=-1)
        if features.ndim != 2:
            raise RuntimeError(
                f"Unexpected Word4Per gallery feature shape: {tuple(features.shape)}"
            )

        if feature_memmap is None:
            feature_dim = int(features.shape[1])
            feature_memmap = np.lib.format.open_memmap(
                cache_path,
                mode="w+",
                dtype=np_dtype,
                shape=(len(dataset), feature_dim),
            )

        batch_np = features.cpu().numpy().astype(np_dtype, copy=False)
        feature_memmap[cursor : cursor + len(batch_np)] = batch_np
        cursor += len(batch_np)

    if feature_memmap is None or feature_dim is None:
        raise RuntimeError("Gallery person dataset is empty")
    feature_memmap.flush()
    np.save(offsets_path, dataset.offsets)
    return cache_path, dataset.offsets, feature_dim


class QueryTargetDataset(Dataset):
    def __init__(
        self,
        queries: Sequence[dict[str, Any]],
        targets: Sequence[Sequence[QueryTarget]],
        boxes: Sequence[Sequence[tuple[float, float, float, float]]],
        gallery: Sequence[dict[str, Any]],
        gallery_index: dict[Any, int],
        transform,
    ):
        self.gallery = gallery
        self.transform = transform
        self.items: list[tuple[int, tuple[float, float, float, float], str, int]] = []
        for qi, (query, query_targets, query_boxes) in enumerate(
            zip(queries, targets, boxes)
        ):
            gi = gallery_index[query["image_id"]]
            if len(query_targets) != len(query_boxes):
                raise AssertionError(f"Query {qi}: target/box count mismatch")
            for target, member_boxes in zip(query_targets, query_boxes, strict=True):
                for box in member_boxes:
                    self.items.append((gi, box, target.modify_text, qi))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        gi, box, caption, owner = self.items[index]
        path = image_path(self.gallery[gi], gi)
        with Image.open(path) as source:
            crop = source.convert("RGB").crop(box)
            return self.transform(crop), caption, owner


@torch.no_grad()
def encode_query_targets(
    model,
    img2text,
    tokenizer,
    tokenize,
    split_ind,
    text_length: int,
    dataset: QueryTargetDataset,
    num_queries: int,
    runtime: dict[str, Any],
    device: torch.device,
) -> list[np.ndarray]:
    loader = DataLoader(
        dataset,
        batch_size=int(runtime.get("query_batch_size", 128)),
        shuffle=False,
        num_workers=int(runtime.get("num_workers", 4)),
        pin_memory=(device.type == "cuda"),
    )

    grouped: list[list[np.ndarray]] = [[] for _ in range(num_queries)]
    mapper_dtype = next(img2text.parameters()).dtype

    for images, captions, owners in progress_bar(
        loader, desc="Encode Word4Per query targets", total=len(loader), unit="batch"
    ):
        images = images.to(device, non_blocking=(device.type == "cuda"))
        raw = model.encode_image(images)
        image_tokens = img2text(raw.to(dtype=mapper_dtype))
        text_tokens = torch.stack(
            [
                tokenize(
                    f"a * is , {str(caption)}",
                    tokenizer=tokenizer,
                    text_length=text_length,
                    truncate=True,
                )
                for caption in captions
            ]
        ).to(device)
        features = model.encode_text_img_retrieval(
            text_tokens,
            image_tokens,
            split_ind=split_ind,
            repeat=False,
        )
        features = F.normalize(features.float(), p=2, dim=-1)
        if features.ndim != 2:
            raise RuntimeError(
                f"Unexpected Word4Per query feature shape: {tuple(features.shape)}"
            )

        features_np = features.cpu().numpy().astype(np.float32, copy=False)
        for feature, owner in zip(features_np, owners.numpy().tolist()):
            grouped[int(owner)].append(feature)

    result: list[np.ndarray] = []
    for qi, group in enumerate(grouped):
        if not group:
            raise RuntimeError(f"Query {qi} has no Word4Per target feature")
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
    person_batch_size: int,
    query_batch_size: int,
    device: torch.device,
) -> Path:
    gallery_features = np.load(gallery_feature_path, mmap_mode="r")
    expected_shape = (len(component_features), int(gallery_features.shape[0]))
    if cache_path.is_file():
        cached = np.load(cache_path, mmap_mode="r")
        if cached.shape == expected_shape and cached.dtype == np.float32:
            print(f"[cache] component-person scores: {rel(cache_path)}", flush=True)
            return cache_path

    scores = np.lib.format.open_memmap(
        cache_path,
        mode="w+",
        dtype=np.float32,
        shape=expected_shape,
    )
    q_all = torch.from_numpy(np.asarray(component_features)).to(
        device=device, dtype=torch.float32
    )

    for p_start in progress_bar(
        range(0, gallery_features.shape[0], person_batch_size),
        desc="Word4Per component-person scores",
        total=(gallery_features.shape[0] + person_batch_size - 1) // person_batch_size,
        unit="person-batch",
    ):
        p_end = min(p_start + person_batch_size, gallery_features.shape[0])
        g = torch.from_numpy(np.asarray(gallery_features[p_start:p_end])).to(
            device=device, dtype=torch.float32
        )
        for q_start in range(0, len(component_features), query_batch_size):
            q_end = min(q_start + query_batch_size, len(component_features))
            q = q_all[q_start:q_end]
            scores[q_start:q_end, p_start:p_end] = (
                (q @ g.T).cpu().numpy().astype(np.float32, copy=False)
            )
    scores.flush()
    return cache_path


def score_all_queries(
    query_features: Sequence[np.ndarray],
    gallery_feature_path: Path,
    offsets: np.ndarray,
    component_score_path: Path,
    scores_path: Path,
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
        person_batch_size=person_batch_size,
        query_batch_size=query_batch_size,
        device=device,
    )
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
    tracker = PhaseTracker(METHOD_ID, total=9)
    tracker.advance("Load config, manifests, and cache identity")
    print(f"Word4Per + SetMatch adapter: {ADAPTER_VERSION}", flush=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()

    config_path = resolve_config_path(args.config)
    cfg = load_yaml(config_path)
    method = str(cfg.get("method", METHOD_ID))
    if method != METHOD_ID:
        raise ValueError(f"Expected method={METHOD_ID!r}, got {method!r}")

    output = (ROOT / cfg["output"]["dir"]).resolve()
    output.mkdir(parents=True, exist_ok=True)

    data_cfg = cfg.get("data", {})
    gallery_path = (ROOT / data_cfg.get("gallery_manifest", "data/gallery.jsonl")).resolve()
    queries_path = (ROOT / data_cfg.get("query_manifest", "data/queries.jsonl")).resolve()
    gallery = load_jsonl(gallery_path)
    queries = load_jsonl(queries_path)
    if not gallery or not queries:
        raise RuntimeError("Canonical gallery/query manifests must be non-empty")

    device = device_from(str(cfg.get("runtime", {}).get("device", "cuda")))
    person = load_person_context(gallery, queries, gallery_path, queries_path, device)
    ckpt_cfg = cfg["checkpoint"]
    stage2_ckpt_for_cache = resolve_config_path(str(ckpt_cfg["stage2"]))
    stage2_cfg_for_cache = resolve_config_path(str(ckpt_cfg["stage2_config"]))
    base_clip_for_cache = resolve_config_path(str(ckpt_cfg["base_clip"]))
    selector_for_cache = resolve_config_path(
        str(protocol_config()["selector"]["checkpoint"])
    )
    detector_for_cache = resolve_config_path(
        str(protocol_config()["detector"]["checkpoint"])
    )
    cache_key, cache_fingerprint = build_cache_key(
        person=person,
        config_path=config_path,
        gallery_path=gallery_path,
        queries_path=queries_path,
        checkpoint_paths=[
            stage2_ckpt_for_cache,
            stage2_cfg_for_cache,
            base_clip_for_cache,
            selector_for_cache,
            detector_for_cache,
        ],
    )
    cache_dir = output / "cache" / cache_key
    cache_dir.mkdir(parents=True, exist_ok=True)

    gallery_index = build_gallery_index(gallery)
    query_targets = [parse_query_targets(q, qi) for qi, q in enumerate(queries)]

    runtime = cfg.get("runtime", {})
    device = device_from(str(runtime.get("device", "cuda")))
    detector_device = device_from(str(runtime.get("detector_device", str(device))))
    tracker.log(
        f"gallery={len(gallery):,} queries={len(queries):,} "
        f"targets={sum(len(x) for x in query_targets):,} device={device} "
        f"detector_device={detector_device} cache={cache_key}"
    )

    tracker.advance("Consume shared ordered gallery persons")
    detector_cfg = protocol_config()["detector"]
    candidate_cache_path = person.detections.path
    gallery_boxes = person.detections.candidates()
    detector_weights = str(resolve_config_path(detector_cfg["checkpoint"]))
    threshold = float(detector_cfg["score_threshold"])
    tracker.advance("Consume shared Subject memberships")
    selected_query_boxes = person.localization.boxes()
    selector_stats = person.localization.stats

    tracker.advance("Load pinned Word4Per Stage-2 model")
    old_project = ensure_official_source(cfg)
    (
        model,
        img2text,
        transform,
        tokenizer,
        tokenize,
        split_ind,
        word4per_args,
        stage2_cfg,
        stage2_ckpt,
    ) = load_word4per(cfg, old_project, device)
    tracker.log(f"Word4Per model loaded from {rel(stage2_ckpt)}")

    tracker.advance("Encode predicted gallery persons")
    gallery_dataset = GalleryPersonDataset(gallery, gallery_boxes, transform)
    gallery_feature_path = cache_dir / "gallery_person_features.npy"
    gallery_feature_path, offsets, feature_dim = encode_gallery_persons(
        model=model,
        dataset=gallery_dataset,
        cache_path=gallery_feature_path,
        runtime=runtime,
        device=device,
    )
    tracker.log(
        f"gallery person features ready: persons={int(offsets[-1]):,} dim={feature_dim} "
        f"path={rel(gallery_feature_path)}"
    )

    tracker.advance("Encode composed query targets")
    query_dataset = QueryTargetDataset(
        queries=queries,
        targets=query_targets,
        boxes=selected_query_boxes,
        gallery=gallery,
        gallery_index=gallery_index,
        transform=transform,
    )
    query_features = encode_query_targets(
        model=model,
        img2text=img2text,
        tokenizer=tokenizer,
        tokenize=tokenize,
        split_ind=split_ind,
        text_length=int(word4per_args.text_length),
        dataset=query_dataset,
        num_queries=len(queries),
        runtime=runtime,
        device=device,
    )
    tracker.log(
        f"query target features ready: {sum(len(x) for x in query_features):,} target components"
    )

    tracker.advance("Compute component-person scores and SetMatch aggregation")
    setmatch_cfg = cfg["setmatch"]
    unmatched_score = float(setmatch_cfg.get("unmatched_score", -1.0))
    scores_path = output / "scores.npy"
    component_score_path = cache_dir / "component_person_scores.npy"
    score_all_queries(
        query_features=query_features,
        gallery_feature_path=gallery_feature_path,
        offsets=offsets,
        component_score_path=component_score_path,
        scores_path=scores_path,
        person_batch_size=int(runtime.get("score_person_batch_size", 4096)),
        query_batch_size=int(runtime.get("score_query_batch_size", 256)),
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
        "display_name": cfg.get("display_name", "Word4Per + SetMatch"),
        "group": cfg.get("group", "Published / SOTA Baselines"),
        "cpr_supervision": cfg.get("cpr_supervision", "No"),
        "paper": cfg.get("paper", {}),
        "official_source": {
            "repository": cfg["source"]["repository"],
            "commit": cfg["source"]["commit"],
            "subdir": cfg["source"].get("subdir", "old_project"),
        },
        "checkpoint": {
            "stage2_checkpoint": rel(stage2_ckpt),
            "stage2_config": rel(stage2_cfg),
            "status": cfg["checkpoint"].get("status", "REPRODUCED"),
            "training_dataset": cfg["checkpoint"].get("training_dataset", "CUHK-PEDES"),
        },
        "word4per": {
            "pretrain_choice": str(word4per_args.pretrain_choice),
            "img_size": list(word4per_args.img_size),
            "text_length": int(word4per_args.text_length),
            "mlp_depth": int(word4per_args.mlp_depth),
            "prompt_template": "a * is , {relative_caption}",
            "embedding_dim": feature_dim,
        },
        "localization": {
            "uses_gt_target_boxes": False,
            "uses_target_ids_for_box_selection": False,
            "uses_positive_labels_for_scoring": False,
            "person_detector": detector_cfg.get("backend"),
            "detector_weights": detector_weights,
            "score_threshold": threshold,
            "candidate_cache": rel(candidate_cache_path),
            **selector_stats,
        },
        "setmatch": matching_metadata(cfg, score_space="cosine"),
        "shared_person_protocol": person.metadata(),
        "data_schema": {
            "gallery": "gallery.jsonl image_id + path; person_ids are not used for scoring/localization",
            "query_reference": "queries.jsonl image_id -> gallery.image_id",
            "target_selection_text": "subjects[].select_text",
            "target_modification_text": "subjects[].modify_text; relation_text then query text fallback",
        },
        "query_case_counts": dict(case_counts),
        "query_target_count_histogram": {str(k): v for k, v in sorted(target_hist.items())},
        "gallery_predicted_person_count_histogram": {
            str(k): v for k, v in sorted(gallery_person_hist.items())
        },
        "relation_text_fallback_subjects": relation_fallbacks,
        "runtime": {
            "device": str(device),
            "detector_device": str(detector_device),
            "image_batch_size": int(runtime.get("image_batch_size", 64)),
            "query_batch_size": int(runtime.get("query_batch_size", 128)),
            "score_person_batch_size": int(runtime.get("score_person_batch_size", 4096)),
            "score_query_batch_size": int(runtime.get("score_query_batch_size", 256)),
            "num_workers": int(runtime.get("num_workers", 4)),
            "gallery_feature_dtype": str(runtime.get("gallery_feature_dtype", "float16")),
        },
        "config": rel(config_path),
        "cache": {
            "key": cache_key,
            "dir": rel(cache_dir),
            "fingerprint": cache_fingerprint,
        },
        "num_queries": len(queries),
        "num_gallery": len(gallery),
        "scores": rel(scores_path),
        "higher_is_better": True,
        "notes": [
            "Canonical gallery/query ordering is preserved.",
            "The query image remains in scores.npy; evaluate.py handles exclusion.",
            "Person instances and query-target boxes are predicted; no GT boxes or identity labels are used for localization/scoring.",
            "Word4Per is applied independently to each predicted Subject member and SetMatch enforces one-to-one person coverage.",
            "No CPR benchmark training, fine-tuning, checkpoint selection, or hyperparameter tuning is performed.",
        ],
    }

    tracker.log(
        f"score validation passed: shape=({len(queries):,}, {len(gallery):,}); cases={dict(case_counts)}"
    )

    tracker.advance("Write run metadata and final outputs")
    run["data_fingerprint"] = manifest_identity(run["shared_person_protocol"])
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
