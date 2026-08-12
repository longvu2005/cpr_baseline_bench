#!/usr/bin/env python3
"""Word4Per + SetMatch adapter for the CPR baseline benchmark."""

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
import torch.nn.functional as F
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = Path(__file__).resolve().parent / "config.yaml"
METHOD_ID = "word4per_setmatch"


@dataclass(frozen=True)
class QueryComponent:
    modify_text: str
    subject_id: Any = None
    identity_id: Any = None
    select_text: str | None = None


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
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


def device_from(name: str) -> torch.device:
    if name != "auto":
        device = torch.device(name)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        if device.type == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but unavailable")
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


class GalleryDataset(Dataset):
    """Canonical gallery: each row must contain image_id and path."""

    def __init__(self, rows: Sequence[dict[str, Any]], transform):
        self.rows = rows
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> torch.Tensor:
        row = self.rows[i]
        if "path" not in row:
            raise KeyError(f"Gallery row {i} has no 'path': {row!r}")
        path = ROOT / row["path"]
        if not path.is_file():
            raise FileNotFoundError(path)
        with Image.open(path) as image:
            return self.transform(image.convert("RGB"))


def build_gallery_index(gallery: Sequence[dict[str, Any]]) -> dict[Any, int]:
    index = {}
    for i, row in enumerate(gallery):
        if "image_id" not in row:
            raise KeyError(f"Gallery row {i} has no 'image_id': {row!r}")
        image_id = row["image_id"]
        if image_id in index:
            raise ValueError(f"Duplicate gallery image_id: {image_id!r}")
        index[image_id] = i
    return index


def resolve_query_indices(
    queries: Sequence[dict[str, Any]], gallery_index: dict[Any, int]
) -> np.ndarray:
    indices = []
    for i, query in enumerate(queries):
        if "image_id" not in query:
            raise KeyError(f"Query row {i} has no 'image_id': {query!r}")
        image_id = query["image_id"]
        if image_id not in gallery_index:
            raise ValueError(f"Query image missing from gallery: {image_id!r}")
        indices.append(gallery_index[image_id])
    return np.asarray(indices, dtype=np.int64)


def parse_query_components(query: dict[str, Any], qi: int) -> list[QueryComponent]:
    raw = query.get("components")

    if raw is None:
        text = query.get("modify_text")
        if not isinstance(text, str) or not text.strip():
            raise KeyError(
                f"Query row {qi} has neither components nor usable modify_text"
            )
        return [QueryComponent(
            modify_text=text.strip(),
            subject_id=query.get("subject_id"),
            identity_id=query.get("identity_id"),
            select_text=query.get("select_text"),
        )]

    if not isinstance(raw, list) or not raw:
        raise ValueError(f"Query row {qi}: 'components' must be a non-empty list")

    result = []
    for ci, comp in enumerate(raw):
        if not isinstance(comp, dict):
            raise TypeError(f"Query {qi} component {ci} must be an object")
        text = comp.get("modify_text")
        if not isinstance(text, str) or not text.strip():
            raise KeyError(
                f"Query {qi} component {ci} has no usable modify_text: {comp!r}"
            )
        result.append(QueryComponent(
            modify_text=text.strip(),
            subject_id=comp.get("subject_id"),
            identity_id=comp.get("identity_id"),
            select_text=comp.get("select_text"),
        ))
    return result


def ensure_official_source(cfg: dict[str, Any]) -> Path:
    source = cfg["source"]
    checkout = (ROOT / source["local_checkout"]).resolve()
    old_project = checkout / source.get("subdir", "old_project")
    expected = str(source["commit"])

    if not checkout.exists():
        if not source.get("auto_clone", True):
            raise FileNotFoundError(checkout)
        checkout.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", source["repository"], str(checkout)], check=True)

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
    args.training = False
    model = build_model(args, num_classes=int(cfg["checkpoint"].get("num_classes", 11003)))

    backbone = str(args.pretrain_choice)
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
        model, img2text, transform, tokenizer, tokenize, split_ind,
        args, stage2_cfg, stage2_ckpt,
    )


@torch.no_grad()
def encode_gallery(
    model,
    transform,
    gallery: Sequence[dict[str, Any]],
    runtime: dict[str, Any],
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Return raw gallery features for img2text and normalized features for scoring."""
    loader = DataLoader(
        GalleryDataset(gallery, transform),
        batch_size=int(runtime.get("image_batch_size", 256)),
        shuffle=False,
        num_workers=int(runtime.get("num_workers", 4)),
        pin_memory=(device.type == "cuda"),
    )

    raw_chunks, norm_chunks = [], []
    for images in tqdm(loader, desc="Word4Per gallery"):
        images = images.to(device, non_blocking=(device.type == "cuda"))
        raw = model.encode_image(images)
        norm = F.normalize(raw.float(), p=2, dim=-1)
        raw_chunks.append(raw.float().cpu().numpy().astype(np.float32, copy=False))
        norm_chunks.append(norm.cpu().numpy().astype(np.float32, copy=False))

    raw = np.concatenate(raw_chunks, axis=0)
    norm = np.concatenate(norm_chunks, axis=0)
    if raw.shape[0] != len(gallery) or norm.shape[0] != len(gallery):
        raise AssertionError("Gallery feature count mismatch")
    return raw, norm


def flatten_components(
    query_components: Sequence[Sequence[QueryComponent]],
    query_gallery_indices: np.ndarray,
) -> tuple[list[QueryComponent], np.ndarray, np.ndarray]:
    components, refs, owners = [], [], []
    for qi, comps in enumerate(query_components):
        ref = int(query_gallery_indices[qi])
        for comp in comps:
            components.append(comp)
            refs.append(ref)
            owners.append(qi)
    return (
        components,
        np.asarray(refs, dtype=np.int64),
        np.asarray(owners, dtype=np.int64),
    )


@torch.no_grad()
def encode_components(
    model,
    img2text,
    tokenizer,
    tokenize,
    split_ind,
    components: Sequence[QueryComponent],
    ref_indices: np.ndarray,
    gallery_raw: np.ndarray,
    text_length: int,
    runtime: dict[str, Any],
    device: torch.device,
) -> np.ndarray:
    """Official path: raw image feature -> img2text -> composed feature -> normalize."""
    batch_size = int(runtime.get("query_batch_size", 128))
    mapper_dtype = next(img2text.parameters()).dtype
    chunks = []

    for start in tqdm(range(0, len(components), batch_size), desc="Word4Per queries"):
        end = min(start + batch_size, len(components))
        batch_components = components[start:end]
        batch_refs = ref_indices[start:end]

        ref_raw = torch.from_numpy(np.asarray(gallery_raw[batch_refs])).to(
            device=device, dtype=mapper_dtype
        )
        image_tokens = img2text(ref_raw)

        text_tokens = torch.stack([
            tokenize(
                f"a * is , {comp.modify_text}",
                tokenizer=tokenizer,
                text_length=text_length,
                truncate=True,
            )
            for comp in batch_components
        ]).to(device)

        composed = model.encode_text_img_retrieval(
            text_tokens,
            image_tokens,
            split_ind=split_ind,
            repeat=False,
        )
        composed = F.normalize(composed.float(), p=2, dim=-1)
        chunks.append(composed.cpu().numpy().astype(np.float32, copy=False))

    return np.concatenate(chunks, axis=0)


def group_component_features(
    features: np.ndarray, owners: np.ndarray, num_queries: int
) -> list[np.ndarray]:
    groups: list[list[np.ndarray]] = [[] for _ in range(num_queries)]
    for feat, owner in zip(features, owners):
        groups[int(owner)].append(feat)
    result = []
    for qi, group in enumerate(groups):
        if not group:
            raise RuntimeError(f"Query {qi} has zero component features")
        result.append(np.stack(group, axis=0).astype(np.float32, copy=False))
    return result


def score_setmatch(
    query_sets: Sequence[np.ndarray],
    gallery_norm: np.ndarray,
    scores_path: Path,
    aggregation: str,
    score_batch_size: int,
    device: torch.device,
) -> None:
    """Set-to-image aggregation for canonical image-level gallery rows.

    mean: mean cosine over all query components.
    min:  minimum cosine over all query components (stricter AND-style option).
    """
    aggregation = aggregation.lower()
    if aggregation not in {"mean", "min"}:
        raise ValueError("setmatch.aggregation must be 'mean' or 'min'")

    scores = np.lib.format.open_memmap(
        scores_path,
        mode="w+",
        dtype=np.float32,
        shape=(len(query_sets), len(gallery_norm)),
    )
    gallery = torch.from_numpy(np.asarray(gallery_norm)).to(device)

    if aggregation == "mean":
        # mean(component cosine) == mean(component feature) @ normalized gallery.
        query_matrix = np.stack([q.mean(axis=0) for q in query_sets], axis=0).astype(
            np.float32, copy=False
        )
        for start in tqdm(range(0, len(query_matrix), score_batch_size), desc="Scores"):
            end = min(start + score_batch_size, len(query_matrix))
            q = torch.from_numpy(np.asarray(query_matrix[start:end])).to(device)
            scores[start:end] = (q @ gallery.T).float().cpu().numpy()
    else:
        for qi, query_set in enumerate(tqdm(query_sets, desc="Scores")):
            q = torch.from_numpy(np.asarray(query_set)).to(device)
            component_scores = q @ gallery.T
            scores[qi] = component_scores.min(dim=0).values.float().cpu().numpy()

    scores.flush()


def validate_scores(scores_path: Path, nq: int, ng: int) -> None:
    scores = np.load(scores_path, mmap_mode="r")
    if scores.shape != (nq, ng):
        raise AssertionError(f"Wrong score shape: {scores.shape}, expected {(nq, ng)}")
    for start in range(0, nq, 256):
        if not np.isfinite(np.asarray(scores[start:start + 256])).all():
            raise FloatingPointError(f"NaN/Inf in score rows {start}:{start + 256}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    cli = parser.parse_args()

    config_path = resolve_config_path(cli.config)
    cfg = load_yaml(config_path)
    method = str(cfg.get("method", METHOD_ID))
    if method != METHOD_ID:
        raise ValueError(f"Expected method={METHOD_ID!r}, got {method!r}")

    output = (ROOT / cfg["output"]["dir"]).resolve()
    output.mkdir(parents=True, exist_ok=True)

    data_cfg = cfg.get("data", {})
    gallery_path = ROOT / data_cfg.get("gallery_manifest", "data/gallery.jsonl")
    queries_path = ROOT / data_cfg.get("query_manifest", "data/queries.jsonl")
    gallery = load_jsonl(gallery_path)
    queries = load_jsonl(queries_path)
    if not gallery or not queries:
        raise RuntimeError("Canonical gallery/query manifests must be non-empty")

    # Same gallery construction as the simple CLIP baseline:
    # gallery.jsonl owns image paths, queries only reference image_id.
    gallery_index = build_gallery_index(gallery)
    query_gallery_indices = resolve_query_indices(queries, gallery_index)
    query_components = [parse_query_components(q, i) for i, q in enumerate(queries)]

    runtime = cfg.get("runtime", {})
    device = device_from(str(runtime.get("device", "auto")))
    old_project = ensure_official_source(cfg)
    (
        model, img2text, transform, tokenizer, tokenize, split_ind,
        word4per_args, stage2_cfg, stage2_ckpt,
    ) = load_word4per(cfg, old_project, device)

    # Encode gallery once. Raw features are used only for img2text; normalized
    # features are used for final cosine retrieval scores.
    gallery_raw, gallery_norm = encode_gallery(
        model, transform, gallery, runtime, device
    )

    flat_components, ref_indices, owners = flatten_components(
        query_components, query_gallery_indices
    )
    component_features = encode_components(
        model=model,
        img2text=img2text,
        tokenizer=tokenizer,
        tokenize=tokenize,
        split_ind=split_ind,
        components=flat_components,
        ref_indices=ref_indices,
        gallery_raw=gallery_raw,
        text_length=int(word4per_args.text_length),
        runtime=runtime,
        device=device,
    )
    query_sets = group_component_features(component_features, owners, len(queries))

    aggregation = str(cfg.get("setmatch", {}).get("aggregation", "mean")).lower()
    scores_path = output / "scores.npy"
    score_setmatch(
        query_sets=query_sets,
        gallery_norm=gallery_norm,
        scores_path=scores_path,
        aggregation=aggregation,
        score_batch_size=int(runtime.get("score_batch_size", 256)),
        device=device,
    )
    validate_scores(scores_path, len(queries), len(gallery))

    case_counts = Counter(
        str(q.get("case_type", q.get("case", "UNKNOWN"))).upper() for q in queries
    )
    component_hist = Counter(len(x) for x in query_components)

    run = {
        "method": method,
        "display_name": cfg.get("display_name", "Word4Per + SetMatch"),
        "group": cfg.get("group", "Published Baselines"),
        "cpr_supervision": cfg.get("cpr_supervision", "No"),
        "paper": cfg.get("paper", {}),
        "official_source": {
            "repository": cfg["source"]["repository"],
            "commit": cfg["source"]["commit"],
            "subdir": cfg["source"].get("subdir", "old_project"),
        },
        "model": {
            "pretrain_choice": str(word4per_args.pretrain_choice),
            "img_size": list(word4per_args.img_size),
            "text_length": int(word4per_args.text_length),
            "mlp_depth": int(word4per_args.mlp_depth),
            "stage2_checkpoint": rel(stage2_ckpt),
            "stage2_config": rel(stage2_cfg),
        },
        "data_schema": {
            "gallery": "gallery.jsonl: image_id + path",
            "query_reference": "queries.jsonl image_id -> gallery.image_id",
            "query_components": "components",
            "component_text": "modify_text",
            "select_text": "metadata only",
        },
        "setmatch": {
            "aggregation": aggregation,
            "definition": (
                "mean cosine across all Word4Per query components per gallery image"
                if aggregation == "mean"
                else "minimum cosine across all Word4Per query components per gallery image"
            ),
        },
        "query_case_counts": dict(case_counts),
        "query_component_count_histogram": {
            str(k): v for k, v in sorted(component_hist.items())
        },
        "runtime": {
            "device": str(device),
            "image_batch_size": int(runtime.get("image_batch_size", 256)),
            "query_batch_size": int(runtime.get("query_batch_size", 128)),
            "score_batch_size": int(runtime.get("score_batch_size", 256)),
            "num_workers": int(runtime.get("num_workers", 4)),
        },
        "config": rel(config_path),
        "num_queries": len(queries),
        "num_gallery": len(gallery),
        "scores": rel(scores_path),
        "higher_is_better": True,
        "notes": [
            "Canonical gallery/query ordering is preserved.",
            "The query image remains in the score matrix; evaluate.py handles exclusion.",
            "All components of a query share the query-level reference image resolved by image_id.",
            "select_text is not concatenated into the Word4Per modification prompt.",
            "No CPR-pilot training or tuning is performed.",
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
