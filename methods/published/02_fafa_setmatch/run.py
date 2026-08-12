#!/usr/bin/env python3
"""FAFA + SetMatch adapter for the CPR baseline benchmark.

The retrieval model/scoring path is imported from the official FAFA_SynCPR
implementation. Person boxes and query-target localization are predicted; no GT
identity-to-box mapping is consumed by this adapter.
"""

from __future__ import annotations

import argparse
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
import yaml
from PIL import Image
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = Path(__file__).resolve().parent / "config.yaml"
METHOD_ID = "fafa_setmatch"
ADAPTER_VERSION = "2026-08-12-v1-official-fafa-setmatch"


@dataclass(frozen=True)
class QueryTarget:
    modify_text: str
    select_text: str
    subject_id: Any = None
    identity_id: Any = None


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
                identity_id=subject.get("identity_id"),
            )
        )

    return targets


def ensure_official_source(cfg: dict[str, Any]) -> Path:
    source = cfg["source"]
    checkout = (ROOT / source["local_checkout"]).resolve()
    subdir = checkout / source.get("subdir", "FAFA_SynCPR")
    expected = str(source["commit"])

    if not checkout.exists():
        if not source.get("auto_clone", True):
            raise FileNotFoundError(checkout)
        checkout.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", source["repository"], str(checkout)], check=True
        )

    actual = subprocess.check_output(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual != expected:
        subprocess.run(
            ["git", "-C", str(checkout), "fetch", "--all", "--tags"], check=True
        )
        subprocess.run(
            ["git", "-C", str(checkout), "checkout", "--detach", expected],
            check=True,
        )
        actual = subprocess.check_output(
            ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True
        ).strip()

    if actual != expected:
        raise RuntimeError(f"Source commit mismatch: expected {expected}, got {actual}")
    if not subdir.is_dir():
        raise FileNotFoundError(subdir)
    return subdir


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

    model, _, txt_processors = load_model_and_preprocess(
        name=str(ckpt_cfg.get("model_name", "blip2_fafa_cpr")),
        model_type=str(ckpt_cfg.get("model_type", "pretrain")),
        is_eval=True,
        device=str(device),
    )

    checkpoint = torch.load(ckpt_path, map_location=device)
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


def load_detector(device: torch.device):
    from torchvision.models.detection import (
        FasterRCNN_ResNet50_FPN_V2_Weights,
        fasterrcnn_resnet50_fpn_v2,
    )

    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn_v2(weights=weights).to(device).eval()
    return model, weights


@torch.no_grad()
def detect_candidates(
    detector,
    weights,
    path: Path,
    detector_cfg: dict[str, Any],
    device: torch.device,
) -> list[BoxCandidate]:
    from torchvision.transforms.functional import pil_to_tensor

    with Image.open(path) as source:
        image = source.convert("RGB")
        width, height = image.size
        tensor = pil_to_tensor(image).float().div_(255.0).to(device)

    output = detector([tensor])[0]
    labels = output["labels"].detach().cpu().numpy()
    scores = output["scores"].detach().cpu().numpy()
    boxes = output["boxes"].detach().cpu().numpy()

    person_label = int(detector_cfg.get("person_label", 1))
    max_persons = int(detector_cfg.get("max_persons_per_image", 10))
    result: list[BoxCandidate] = []

    order = np.argsort(-scores)
    for idx in order:
        if int(labels[idx]) != person_label:
            continue
        x1, y1, x2, y2 = (float(x) for x in boxes[idx])
        x1 = max(0.0, min(x1, width - 1.0))
        y1 = max(0.0, min(y1, height - 1.0))
        x2 = max(x1 + 1.0, min(x2, float(width)))
        y2 = max(y1 + 1.0, min(y2, float(height)))
        result.append(BoxCandidate((x1, y1, x2, y2), float(scores[idx])))
        if len(result) >= max_persons:
            break

    if not result:
        result = [
            BoxCandidate((0.0, 0.0, float(width), float(height)), -1.0, fallback=True)
        ]
    return result


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
    rows = load_jsonl(path)
    if len(rows) != len(gallery):
        return None

    result: list[list[BoxCandidate]] = []
    for expected, row in zip(gallery, rows):
        if row.get("image_id") != expected.get("image_id"):
            return None
        candidates = []
        for item in row.get("boxes", []):
            xyxy = item.get("xyxy")
            if not isinstance(xyxy, list) or len(xyxy) != 4:
                return None
            candidates.append(
                BoxCandidate(
                    tuple(float(x) for x in xyxy),
                    float(item.get("score", -1.0)),
                    bool(item.get("fallback", False)),
                )
            )
        if not candidates:
            return None
        result.append(candidates)
    return result


def get_or_create_detection_cache(
    gallery: Sequence[dict[str, Any]],
    cache_path: Path,
    detector_cfg: dict[str, Any],
    device: torch.device,
) -> tuple[list[list[BoxCandidate]], str]:
    cached = load_candidate_cache(cache_path, gallery)
    if cached is not None:
        return cached, "cache"

    detector, weights = load_detector(device)
    candidates: list[list[BoxCandidate]] = []
    for i, row in enumerate(tqdm(gallery, desc="Predict person boxes")):
        candidates.append(
            detect_candidates(
                detector,
                weights,
                image_path(row, i),
                detector_cfg,
                device,
            )
        )
    save_candidate_cache(cache_path, gallery, candidates)
    return candidates, str(weights)


def load_clip_selector(model_name: str, device: torch.device):
    import clip  # type: ignore

    model, preprocess = clip.load(model_name, device=device, jit=False)
    model.eval()
    return clip, model, preprocess


@torch.no_grad()
def select_query_target_boxes(
    queries: Sequence[dict[str, Any]],
    query_targets: Sequence[Sequence[QueryTarget]],
    gallery: Sequence[dict[str, Any]],
    gallery_index: dict[Any, int],
    all_candidates: Sequence[Sequence[BoxCandidate]],
    localization_cfg: dict[str, Any],
    device: torch.device,
) -> tuple[list[list[tuple[float, float, float, float]]], dict[str, Any]]:
    selector_cfg = localization_cfg["query_selector"]
    detector_cfg = localization_cfg["detector"]
    clip_module, clip_model, clip_preprocess = load_clip_selector(
        str(selector_cfg.get("model", "ViT-B/32")), device
    )
    threshold = float(detector_cfg.get("score_threshold", 0.55))

    selected_per_query: list[list[tuple[float, float, float, float]]] = []
    localization_scores: list[float] = []
    low_candidate_queries = 0

    for qi, (query, targets) in enumerate(
        tqdm(zip(queries, query_targets), total=len(queries), desc="Select query targets")
    ):
        image_id = query.get("image_id")
        if image_id not in gallery_index:
            raise ValueError(f"Query {qi}: image_id not found in gallery: {image_id!r}")
        gi = gallery_index[image_id]
        boxes = choose_person_boxes(all_candidates[gi], threshold, len(targets))
        if len(boxes) < len(targets):
            low_candidate_queries += 1
            raise RuntimeError(
                f"Query {qi} ({query.get('query_id')}): detector produced only "
                f"{len(boxes)} person candidate(s) for {len(targets)} target(s). "
                "Lower localization.detector.score_threshold or increase "
                "max_persons_per_image; GT target boxes are intentionally not used."
            )

        path = image_path(gallery[gi], gi)
        with Image.open(path) as source:
            image = source.convert("RGB")
            crop_tensors = [clip_preprocess(image.crop(candidate.box)) for candidate in boxes]

        crop_batch = torch.stack(crop_tensors).to(device)
        text_tokens = clip_module.tokenize([target.select_text for target in targets]).to(device)
        image_features = clip_model.encode_image(crop_batch).float()
        text_features = clip_model.encode_text(text_tokens).float()
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        sim = (text_features @ image_features.T).cpu().numpy()

        rows, cols = linear_sum_assignment(-sim)
        mapping = {int(r): int(c) for r, c in zip(rows, cols)}
        if len(mapping) != len(targets):
            raise RuntimeError(f"Query {qi}: incomplete target-localization assignment")

        chosen = [boxes[mapping[i]].box for i in range(len(targets))]
        selected_per_query.append(chosen)
        localization_scores.extend(float(sim[i, mapping[i]]) for i in range(len(targets)))

    stats = {
        "selector": str(selector_cfg.get("backend", "openai_clip")),
        "selector_model": str(selector_cfg.get("model", "ViT-B/32")),
        "assignment": "hungarian",
        "mean_assigned_clip_similarity": (
            float(np.mean(localization_scores)) if localization_scores else None
        ),
        "min_assigned_clip_similarity": (
            float(np.min(localization_scores)) if localization_scores else None
        ),
        "queries_with_too_few_candidates": low_candidate_queries,
    }
    return selected_per_query, stats


class GalleryPersonDataset(Dataset):
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
        path = image_path(self.gallery[gi], gi)
        with Image.open(path) as source:
            crop = source.convert("RGB").crop(box)
            return self.preprocess(crop)


@torch.no_grad()
def encode_gallery_persons(
    model,
    dataset: GalleryPersonDataset,
    cache_path: Path,
    runtime: dict[str, Any],
    device: torch.device,
) -> tuple[Path, np.ndarray, tuple[int, int]]:
    offsets_path = cache_path.with_name("gallery_person_offsets.npy")
    if cache_path.is_file() and offsets_path.is_file():
        offsets = np.load(offsets_path)
        features = np.load(cache_path, mmap_mode="r")
        if (
            features.ndim == 3
            and features.shape[0] == len(dataset)
            and offsets.shape == dataset.offsets.shape
            and np.array_equal(offsets, dataset.offsets)
        ):
            return cache_path, offsets, (int(features.shape[1]), int(features.shape[2]))

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
    np_dtype = np.float16 if output_dtype == "float16" else np.float32

    for images in tqdm(loader, desc="FAFA gallery persons"):
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
                cache_path,
                mode="w+",
                dtype=np_dtype,
                shape=(len(dataset), token_dim[0], token_dim[1]),
            )

        batch_np = image_features.cpu().numpy().astype(np_dtype, copy=False)
        feature_memmap[cursor : cursor + len(batch_np)] = batch_np
        cursor += len(batch_np)

    if feature_memmap is None or token_dim is None:
        raise RuntimeError("Gallery person dataset is empty")
    feature_memmap.flush()
    np.save(offsets_path, dataset.offsets)
    return cache_path, dataset.offsets, token_dim


class QueryTargetDataset(Dataset):
    def __init__(
        self,
        queries: Sequence[dict[str, Any]],
        targets: Sequence[Sequence[QueryTarget]],
        boxes: Sequence[Sequence[tuple[float, float, float, float]]],
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
            for target, box in zip(query_targets, query_boxes):
                self.items.append((gi, box, target.modify_text, qi))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        gi, box, caption, owner = self.items[index]
        path = image_path(self.gallery[gi], gi)
        with Image.open(path) as source:
            crop = source.convert("RGB").crop(box)
            return self.preprocess(crop), caption, owner


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
    for images, captions, owners in tqdm(loader, desc="FAFA query targets"):
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
    gallery_features = np.load(gallery_feature_path, mmap_mode="r")
    expected_shape = (len(component_features), int(gallery_features.shape[0]))
    if cache_path.is_file():
        cached = np.load(cache_path, mmap_mode="r")
        if cached.shape == expected_shape and cached.dtype == np.float32:
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

    for p_start in tqdm(
        range(0, gallery_features.shape[0], person_batch_size),
        desc="FAFA component-person scores",
    ):
        p_end = min(p_start + person_batch_size, gallery_features.shape[0])
        g = torch.from_numpy(np.asarray(gallery_features[p_start:p_end])).to(
            device=device, dtype=torch.float32
        )
        for q_start in range(0, len(component_features), query_batch_size):
            q_end = min(q_start + query_batch_size, len(component_features))
            q = q_all[q_start:q_end]
            # q: [Q,D], g: [P,K,D] -> [Q,P,K]
            token_similarity = torch.einsum("qd,pkd->qpk", q, g)
            k = min(fda_k, int(token_similarity.shape[-1]))
            topk = torch.topk(token_similarity, k=k, dim=-1).values
            scores[q_start:q_end, p_start:p_end] = (
                topk.mean(dim=-1).cpu().numpy().astype(np.float32, copy=False)
            )
    scores.flush()
    return cache_path


def setmatch_image_score(
    matrix: np.ndarray,
    unmatched_score: float,
) -> float:
    """Maximum-weight Hungarian assignment followed by minimum matched score."""
    num_targets, num_persons = matrix.shape
    if num_targets < 1:
        raise ValueError("SetMatch requires at least one target")

    if num_persons < num_targets:
        padding = np.full(
            (num_targets, num_targets - num_persons),
            unmatched_score,
            dtype=np.float32,
        )
        matrix = np.concatenate([matrix, padding], axis=1)

    rows, cols = linear_sum_assignment(-matrix)
    if len(rows) != num_targets:
        raise RuntimeError("Hungarian assignment did not cover every target")
    assigned = matrix[rows, cols]
    return float(np.min(assigned))


def build_person_index(offsets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    counts = np.diff(offsets).astype(np.int64, copy=False)
    max_persons = int(counts.max(initial=0))
    if max_persons < 1:
        raise RuntimeError("Every gallery image must have at least one person crop")
    index = np.full((len(counts), max_persons), -1, dtype=np.int64)
    for gi, count in enumerate(counts):
        if count:
            index[gi, :count] = np.arange(offsets[gi], offsets[gi + 1])
    return index, counts


def setmatch_two_targets_all_images(
    target_person_scores: np.ndarray,
    person_index: np.ndarray,
    counts: np.ndarray,
    unmatched_score: float,
) -> np.ndarray:
    """Exact two-target specialization of maximum-weight Hungarian SetMatch.

    The pilot MULTI/RELATIONAL schema has exactly two target subjects. For two
    rows, the maximum-weight one-to-one assignment can be solved from each
    row's best/second-best columns, yielding exactly the same assignment
    objective as Hungarian while avoiding millions of tiny SciPy calls.
    """
    if target_person_scores.shape[0] != 2:
        raise ValueError("Expected exactly two target rows")

    safe_index = np.where(person_index >= 0, person_index, 0)
    values = target_person_scores[:, safe_index]  # [2, G, max_persons]
    values = np.asarray(values, dtype=np.float32)
    values[:, person_index < 0] = -np.inf
    out = np.empty(len(counts), dtype=np.float32)

    one = counts == 1
    if np.any(one):
        real_idx = safe_index[one, 0]
        best_real = np.maximum(
            target_person_scores[0, real_idx], target_person_scores[1, real_idx]
        )
        # With one gallery person and two targets, the other target is forced
        # onto the padded unmatched slot. This is the same padded-Hungarian
        # rule used by setmatch_image_score().
        out[one] = np.minimum(best_real, unmatched_score)

    many = counts >= 2
    if np.any(many):
        a = values[0, many]
        b = values[1, many]

        def top2(x: np.ndarray):
            idx = np.argpartition(-x, kth=1, axis=1)[:, :2]
            vals = np.take_along_axis(x, idx, axis=1)
            order = np.argsort(-vals, axis=1)
            idx = np.take_along_axis(idx, order, axis=1)
            vals = np.take_along_axis(vals, order, axis=1)
            return vals[:, 0], idx[:, 0], vals[:, 1], idx[:, 1]

        a1, ai1, a2, ai2 = top2(a)
        b1, bi1, b2, bi2 = top2(b)

        b_excl_a = np.where(bi1 != ai1, b1, b2)
        a_excl_b = np.where(ai1 != bi1, a1, a2)

        sum_a_first = a1 + b_excl_a
        sum_b_first = b1 + a_excl_b
        choose_a = sum_a_first >= sum_b_first
        out[many] = np.where(
            choose_a,
            np.minimum(a1, b_excl_a),
            np.minimum(b1, a_excl_b),
        )

    zero = counts == 0
    if np.any(zero):
        out[zero] = unmatched_score
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
    scores = np.lib.format.open_memmap(
        scores_path,
        mode="w+",
        dtype=np.float32,
        shape=(len(query_offsets) - 1, len(gallery_offsets) - 1),
    )

    for qi in tqdm(range(len(query_offsets) - 1), desc="SetMatch aggregation"):
        start, end = int(query_offsets[qi]), int(query_offsets[qi + 1])
        target_scores = np.asarray(component_scores[start:end], dtype=np.float32)
        num_targets = end - start

        if num_targets == 1:
            # One-target Hungarian == choose the best predicted person in each
            # gallery image. Person crops are contiguous per image.
            scores[qi] = np.maximum.reduceat(target_scores[0], gallery_offsets[:-1])
        elif num_targets == 2:
            scores[qi] = setmatch_two_targets_all_images(
                target_scores, person_index, counts, unmatched_score
            )
        else:
            # Generic exact Hungarian fallback for future schemas with >2
            # targets. Current pilot data never enters this branch.
            for gi in range(len(gallery_offsets) - 1):
                p_start, p_end = int(gallery_offsets[gi]), int(gallery_offsets[gi + 1])
                scores[qi, gi] = setmatch_image_score(
                    target_scores[:, p_start:p_end], unmatched_score
                )
    scores.flush()


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
    print(f"FAFA + SetMatch adapter: {ADAPTER_VERSION}")
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
    cache_dir = output / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    data_cfg = cfg.get("data", {})
    gallery_path = (ROOT / data_cfg.get("gallery_manifest", "data/gallery.jsonl")).resolve()
    queries_path = (ROOT / data_cfg.get("query_manifest", "data/queries.jsonl")).resolve()
    gallery = load_jsonl(gallery_path)
    queries = load_jsonl(queries_path)
    if not gallery or not queries:
        raise RuntimeError("Canonical gallery/query manifests must be non-empty")

    gallery_index = build_gallery_index(gallery)
    query_targets = [parse_query_targets(q, qi) for qi, q in enumerate(queries)]

    runtime = cfg.get("runtime", {})
    device = device_from(str(runtime.get("device", "cuda")))
    detector_device_name = str(runtime.get("detector_device", str(device)))
    detector_device = device_from(detector_device_name)

    localization_cfg = cfg["localization"]
    detector_cfg = localization_cfg["detector"]
    candidate_cache_path = cache_dir / "person_candidates.jsonl"
    all_candidates, detector_weights = get_or_create_detection_cache(
        gallery,
        candidate_cache_path,
        detector_cfg,
        detector_device,
    )
    if detector_device.type == "cuda":
        torch.cuda.empty_cache()

    threshold = float(detector_cfg.get("score_threshold", 0.55))
    gallery_boxes = [
        choose_person_boxes(candidates, threshold, min_required=1)
        for candidates in all_candidates
    ]

    selected_query_boxes, selector_stats = select_query_target_boxes(
        queries=queries,
        query_targets=query_targets,
        gallery=gallery,
        gallery_index=gallery_index,
        all_candidates=all_candidates,
        localization_cfg=localization_cfg,
        device=device,
    )
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Load the large FAFA model only after detector/CLIP target localization to
    # avoid keeping the auxiliary localization models resident on the GPU.
    fafa_dir = ensure_official_source(cfg)
    model, txt_processors, preprocess, ckpt_path, load_info = load_fafa(
        cfg, fafa_dir, device
    )

    gallery_dataset = GalleryPersonDataset(gallery, gallery_boxes, preprocess)
    gallery_feature_path = cache_dir / "gallery_person_features.npy"
    gallery_feature_path, offsets, token_dim = encode_gallery_persons(
        model=model,
        dataset=gallery_dataset,
        cache_path=gallery_feature_path,
        runtime=runtime,
        device=device,
    )

    query_dataset = QueryTargetDataset(
        queries=queries,
        targets=query_targets,
        boxes=selected_query_boxes,
        gallery=gallery,
        gallery_index=gallery_index,
        preprocess=preprocess,
    )
    query_features = encode_query_targets(
        model=model,
        txt_processors=txt_processors,
        dataset=query_dataset,
        num_queries=len(queries),
        runtime=runtime,
        device=device,
    )

    ckpt_cfg = cfg["checkpoint"]
    setmatch_cfg = cfg["setmatch"]
    fda_k = int(ckpt_cfg.get("fda_k", 6))
    unmatched_score = float(setmatch_cfg.get("unmatched_score", -1.0))
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
        "display_name": cfg.get("display_name", "FAFA + SetMatch"),
        "group": cfg.get("group", "Published / SOTA Baselines"),
        "cpr_supervision": cfg.get("cpr_supervision", "No"),
        "paper": cfg.get("paper", {}),
        "official_source": {
            "repository": cfg["source"]["repository"],
            "commit": cfg["source"]["commit"],
            "subdir": cfg["source"].get("subdir", "FAFA_SynCPR"),
        },
        "checkpoint": {
            "path": rel(ckpt_path),
            "status": cfg["checkpoint"].get("status", "OFFICIAL_RELEASED"),
            "source_url": cfg["checkpoint"].get("source_url"),
            "model_name": cfg["checkpoint"].get("model_name", "blip2_fafa_cpr"),
            "model_type": cfg["checkpoint"].get("model_type", "pretrain"),
            "load_strict": False,
            "missing_key_count": len(load_info["missing_keys"]),
            "unexpected_key_count": len(load_info["unexpected_keys"]),
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
            "detector_weights": detector_weights,
            "score_threshold": threshold,
            "candidate_cache": rel(candidate_cache_path),
            **selector_stats,
        },
        "setmatch": {
            "assignment": "maximum-weight one-to-one Hungarian",
            "aggregation": "minimum assigned target score",
            "unmatched_score": unmatched_score,
            "no_partial_credit": True,
        },
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
            "query_batch_size": int(runtime.get("query_batch_size", 64)),
            "score_person_batch_size": int(runtime.get("score_person_batch_size", 2048)),
            "score_query_batch_size": int(runtime.get("score_query_batch_size", 128)),
            "num_workers": int(runtime.get("num_workers", 4)),
        },
        "config": rel(config_path),
        "num_queries": len(queries),
        "num_gallery": len(gallery),
        "num_predicted_gallery_persons": int(offsets[-1]),
        "scores": rel(scores_path),
        "higher_is_better": True,
        "notes": [
            "Canonical gallery/query ordering is preserved.",
            "The query image remains in the score matrix; evaluate.py handles exclusion.",
            "FAFA is loaded from the authors' pinned official implementation and released checkpoint.",
            "No CPR benchmark training, fine-tuning, checkpoint selection, or hyperparameter tuning is performed.",
            "No GT PIPA target boxes or identity-to-box mapping are used in the main adapter.",
            "SetMatch uses maximum-weight Hungarian matching followed by the minimum matched target score.",
            "RELATIONAL text is not modeled jointly by native single-person FAFA; relation_text is only a per-target fallback when modify_text is empty.",
        ],
    }

    run_path = output / "run.json"
    with run_path.open("w", encoding="utf-8") as f:
        json.dump(run, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(f"Saved: {scores_path}  shape=({len(queries)}, {len(gallery)})")
    print(f"Saved: {run_path}")
    print(f"Evaluate: python evaluate.py --method {method}")


if __name__ == "__main__":
    main()
