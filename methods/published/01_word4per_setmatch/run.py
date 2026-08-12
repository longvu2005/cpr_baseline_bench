#!/usr/bin/env python3
"""Word4Per + SetMatch benchmark adapter.

This file intentionally wraps the authors' official Word4Per implementation
instead of reimplementing the model. It preserves canonical query/gallery order,
scores the complete gallery, and writes the benchmark score-matrix contract.

For SINGLE entries the score is the official Word4Per composed-query cosine
similarity. For set-valued entries, SetMatch computes a maximum-weight one-to-one
assignment between query components and gallery components and averages matched
scores. Missing gallery components receive `unmatched_score`.
"""

from __future__ import annotations

import itertools
import json
import math
import os
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm


METHOD_ID = "word4per_setmatch"

GALLERY_IMAGE_CANDIDATES = (
    "file_path", "image_path", "path", "gallery_path", "image", "rel_path"
)
GALLERY_COMPONENTS_CANDIDATES = (
    "components", "people", "persons", "members", "crops", "person_crops"
)
QUERY_IMAGE_CANDIDATES = (
    "reference_image", "reference_image_path", "query_image", "query_image_path",
    "source_image", "source_image_path", "file_path", "image_path", "path"
)
QUERY_TEXT_CANDIDATES = (
    "relative_text", "relative_caption", "caption", "modifier_text", "modification",
    "instruction", "description", "text"
)
QUERY_CASE_CANDIDATES = ("case_type", "case", "query_type", "type")
QUERY_COMPONENTS_CANDIDATES = (
    "components", "subjects", "queries", "references", "people", "persons"
)
COMPONENT_IMAGE_CANDIDATES = (
    "reference_image", "reference_image_path", "image", "image_path", "file_path",
    "path", "crop", "crop_path"
)
COMPONENT_TEXT_CANDIDATES = QUERY_TEXT_CANDIDATES


@dataclass(frozen=True)
class QueryComponent:
    image_path: Path
    text: str


@dataclass(frozen=True)
class GalleryComponent:
    image_path: Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise TypeError(f"Expected mapping in {path}, got {type(data).__name__}")
    return data


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                raise TypeError(f"{path}:{lineno}: each JSONL row must be an object")
            rows.append(obj)
    return rows


def get_dot(obj: Any, key: str | None) -> Any:
    if key is None:
        return None
    cur = obj
    for part in key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def find_key(obj: dict[str, Any], explicit: str | None, candidates: Sequence[str]) -> tuple[str | None, Any]:
    if explicit:
        return explicit, get_dot(obj, explicit)
    for key in candidates:
        value = get_dot(obj, key)
        if value is not None:
            return key, value
    return None, None


def resolve_image_path(root: Path, image_root: Path, manifest_path: Path, value: Any) -> Path:
    if isinstance(value, dict):
        for key in COMPONENT_IMAGE_CANDIDATES:
            if key in value:
                value = value[key]
                break
    if not isinstance(value, (str, os.PathLike)):
        raise TypeError(f"Image path must be a string/path, got {value!r}")
    p = Path(value)
    candidates = [p] if p.is_absolute() else [root / p, image_root / p, manifest_path.parent / p]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    # Keep the deterministic preferred path in the error for easier debugging.
    preferred = candidates[0]
    raise FileNotFoundError(f"Image not found for manifest value {value!r}; tried: {candidates}")


def normalize_component_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    # A single dict/path is one component, not an iterable of characters/keys.
    return [value]


def parse_gallery_rows(rows: list[dict[str, Any]], cfg: dict[str, Any], root: Path, manifest: Path) -> tuple[list[list[GalleryComponent]], dict[str, Any]]:
    data_cfg = cfg["data"]
    image_root = (root / data_cfg.get("gallery_image_root", "data/gallery")).resolve()
    resolved: dict[str, Any] = {}
    entries: list[list[GalleryComponent]] = []

    for idx, row in enumerate(rows):
        comp_key, comps = find_key(row, data_cfg.get("gallery_components_key"), GALLERY_COMPONENTS_CANDIDATES)
        if idx == 0 and comp_key:
            resolved["gallery_components_key"] = comp_key

        components = normalize_component_list(comps)
        if components:
            parsed: list[GalleryComponent] = []
            for comp in components:
                if isinstance(comp, dict):
                    key, value = find_key(comp, data_cfg.get("component_image_key"), COMPONENT_IMAGE_CANDIDATES)
                    if idx == 0 and key:
                        resolved["component_image_key"] = key
                else:
                    value = comp
                if value is None:
                    raise KeyError(f"Gallery row {idx} component has no image field: {comp!r}")
                parsed.append(GalleryComponent(resolve_image_path(root, image_root, manifest, value)))
            entries.append(parsed)
            continue

        image_key, image_value = find_key(row, data_cfg.get("gallery_image_key"), GALLERY_IMAGE_CANDIDATES)
        if image_value is None:
            raise KeyError(
                f"Gallery row {idx} has no resolvable image field. "
                f"Set data.gallery_image_key or data.gallery_components_key in config.yaml. Keys={list(row)}"
            )
        if idx == 0 and image_key:
            resolved["gallery_image_key"] = image_key
        entries.append([GalleryComponent(resolve_image_path(root, image_root, manifest, image_value))])

    return entries, resolved


def parse_query_rows(rows: list[dict[str, Any]], cfg: dict[str, Any], root: Path, manifest: Path) -> tuple[list[list[QueryComponent]], list[str], dict[str, Any]]:
    data_cfg = cfg["data"]
    image_root = (root / data_cfg.get("query_image_root", ".")).resolve()
    resolved: dict[str, Any] = {}
    entries: list[list[QueryComponent]] = []
    cases: list[str] = []

    for idx, row in enumerate(rows):
        case_key, case_value = find_key(row, data_cfg.get("query_case_key"), QUERY_CASE_CANDIDATES)
        if idx == 0 and case_key:
            resolved["query_case_key"] = case_key
        case = str(case_value).upper() if case_value is not None else "UNKNOWN"
        cases.append(case)

        parent_text_key, parent_text = find_key(row, data_cfg.get("query_text_key"), QUERY_TEXT_CANDIDATES)
        if idx == 0 and parent_text_key:
            resolved["query_text_key"] = parent_text_key

        comp_key, comps = find_key(row, data_cfg.get("query_components_key"), QUERY_COMPONENTS_CANDIDATES)
        if idx == 0 and comp_key:
            resolved["query_components_key"] = comp_key

        components = normalize_component_list(comps)
        if components:
            parsed: list[QueryComponent] = []
            for comp in components:
                if isinstance(comp, dict):
                    img_key, image_value = find_key(comp, data_cfg.get("component_image_key"), COMPONENT_IMAGE_CANDIDATES)
                    txt_key, text_value = find_key(comp, data_cfg.get("component_text_key"), COMPONENT_TEXT_CANDIDATES)
                    if idx == 0 and img_key:
                        resolved["component_image_key"] = img_key
                    if idx == 0 and txt_key:
                        resolved["component_text_key"] = txt_key
                else:
                    image_value, text_value = comp, None
                if image_value is None:
                    raise KeyError(f"Query row {idx} component has no image field: {comp!r}")
                if text_value is None:
                    text_value = parent_text
                if not isinstance(text_value, str) or not text_value.strip():
                    raise KeyError(f"Query row {idx} component has no relative text and no usable parent text")
                parsed.append(QueryComponent(
                    resolve_image_path(root, image_root, manifest, image_value),
                    text_value.strip(),
                ))
            entries.append(parsed)
            continue

        image_key, image_value = find_key(row, data_cfg.get("query_reference_image_key"), QUERY_IMAGE_CANDIDATES)
        if image_value is None:
            raise KeyError(
                f"Query row {idx} has no resolvable reference image field. "
                f"Set data.query_reference_image_key/query_components_key. Keys={list(row)}"
            )
        if not isinstance(parent_text, str) or not parent_text.strip():
            raise KeyError(
                f"Query row {idx} has no resolvable relative text field. "
                f"Set data.query_text_key. Keys={list(row)}"
            )
        if idx == 0 and image_key:
            resolved["query_reference_image_key"] = image_key
        entries.append([QueryComponent(
            resolve_image_path(root, image_root, manifest, image_value),
            parent_text.strip(),
        )])

    return entries, cases, resolved


def ensure_official_source(cfg: dict[str, Any], root: Path) -> Path:
    source = cfg["source"]
    checkout = (root / source["local_checkout"]).resolve()
    old_project = checkout / source.get("subdir", "old_project")
    commit = source["commit"]

    if not checkout.exists():
        if not source.get("auto_clone", True):
            raise FileNotFoundError(f"Official source checkout missing: {checkout}")
        checkout.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", source["repository"], str(checkout)], check=True)

    subprocess.run(["git", "-C", str(checkout), "fetch", "--all", "--tags"], check=True)
    subprocess.run(["git", "-C", str(checkout), "checkout", "--detach", commit], check=True)
    actual = subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True).strip()
    if actual != commit:
        raise RuntimeError(f"Official source commit mismatch: expected {commit}, got {actual}")
    if not old_project.is_dir():
        raise FileNotFoundError(f"Official Word4Per source directory missing: {old_project}")
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
    return tokenize, build_transforms, build_model, IM2TEXT, Checkpointer_Toword, load_train_configs, SimpleTokenizer


def load_word4per(cfg: dict[str, Any], root: Path, old_project: Path, device: torch.device):
    tokenize, build_transforms, build_model, IM2TEXT, Checkpointer_Toword, load_train_configs, SimpleTokenizer = import_official(old_project)

    ckpt_cfg = (root / cfg["checkpoint"]["stage2_config"]).resolve()
    ckpt_path = (root / cfg["checkpoint"]["stage2"]).resolve()
    if not ckpt_cfg.is_file():
        raise FileNotFoundError(
            f"Missing reproduced Word4Per Stage-2 config: {ckpt_cfg}. "
            "See this baseline README/checkpoints documentation."
        )
    if not ckpt_path.is_file():
        raise FileNotFoundError(
            f"Missing reproduced Word4Per Stage-2 checkpoint: {ckpt_path}. "
            "The official public README documents a Stage-1 download, not a final Stage-2 best.pth."
        )

    args = load_train_configs(str(ckpt_cfg))
    args.training = False
    num_classes = int(cfg["checkpoint"].get("num_classes", 11003))
    model = build_model(args, num_classes=num_classes)

    if args.pretrain_choice == "ViT-L/14":
        dim = 768
    elif str(args.pretrain_choice).startswith("ViT-B"):
        dim = 512
    else:
        raise ValueError(f"Unsupported Word4Per pretrain_choice for TINet: {args.pretrain_choice}")

    img2text = IM2TEXT(
        embed_dim=dim,
        middle_dim=512,
        output_dim=dim,
        n_layer=int(args.mlp_depth),
    )
    Checkpointer_Toword(model, img2text).load(f=str(ckpt_path))
    model.to(device).eval()
    img2text.to(device).eval()
    transform = build_transforms(img_size=args.img_size, is_train=False)
    tokenizer = SimpleTokenizer()
    split_ind = tokenize("*", tokenizer)[1]
    return model, img2text, transform, tokenizer, tokenize, split_ind, args


def batched(seq: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def load_image_tensor(path: Path, transform) -> torch.Tensor:
    with Image.open(path) as im:
        image = im.convert("RGB")
        return transform(image)


def encode_gallery_images(
    model,
    transform,
    paths: Sequence[Path],
    device: torch.device,
    batch_size: int,
) -> dict[Path, np.ndarray]:
    unique = list(dict.fromkeys(paths))
    out: dict[Path, np.ndarray] = {}
    for chunk in tqdm(list(batched(unique, batch_size)), desc="Word4Per gallery", unit="batch"):
        images = torch.stack([load_image_tensor(p, transform) for p in chunk]).to(device)
        with torch.inference_mode():
            feats = F.normalize(model.encode_image(images), p=2, dim=1).float().cpu().numpy()
        for p, f in zip(chunk, feats):
            out[p] = f.astype(np.float32, copy=False)
    return out


def encode_query_components(
    model,
    img2text,
    transform,
    tokenizer,
    tokenize,
    split_ind,
    components: Sequence[QueryComponent],
    text_length: int,
    device: torch.device,
    batch_size: int,
) -> dict[QueryComponent, np.ndarray]:
    unique = list(dict.fromkeys(components))
    out: dict[QueryComponent, np.ndarray] = {}
    for chunk in tqdm(list(batched(unique, batch_size)), desc="Word4Per queries", unit="batch"):
        images = torch.stack([load_image_tensor(c.image_path, transform) for c in chunk]).to(device)
        blank_tokens = torch.stack([
            tokenize(
                f"a * is , {c.text}",
                tokenizer=tokenizer,
                text_length=text_length,
                truncate=True,
            )
            for c in chunk
        ]).to(device)
        with torch.inference_mode():
            image_feat = model.encode_image(images)
            image_token = img2text(image_feat)
            composed = model.encode_text_img_retrieval(
                blank_tokens,
                image_token,
                split_ind=split_ind,
                repeat=False,
            )
            composed = F.normalize(composed, p=2, dim=1).float().cpu().numpy()
        for c, f in zip(chunk, composed):
            out[c] = f.astype(np.float32, copy=False)
    return out


def assignment_count(k: int, q: int) -> int:
    if k < q:
        return math.factorial(q)
    return math.factorial(k) // math.factorial(k - q)


def setmatch_group_scores(
    query: np.ndarray,
    galleries: np.ndarray,
    unmatched_score: float,
    max_vectorized_assignments: int,
) -> np.ndarray:
    """Score one query set against a stack of equal-size gallery sets.

    query: [Q, D], galleries: [G, K, D]. Returns [G].
    """
    q = query.shape[0]
    g_count, k, _ = galleries.shape
    sims = np.einsum("qd,gkd->gqk", query, galleries, optimize=True)

    if k < q:
        pad = np.full((g_count, q, q - k), unmatched_score, dtype=np.float32)
        sims = np.concatenate([sims.astype(np.float32, copy=False), pad], axis=2)
        k_eff = q
    else:
        k_eff = k

    n_assign = assignment_count(k_eff, q)
    if n_assign <= max_vectorized_assignments:
        perms = list(itertools.permutations(range(k_eff), q))
        best = np.full(g_count, -np.inf, dtype=np.float32)
        rows = np.arange(q)
        for perm in perms:
            score = sims[:, rows, np.asarray(perm)].sum(axis=1) / float(q)
            best = np.maximum(best, score.astype(np.float32, copy=False))
        return best

    # Rare fallback for large sets. This is exact but slower.
    result = np.empty(g_count, dtype=np.float32)
    for gi in range(g_count):
        cost = -sims[gi]
        row_ind, col_ind = linear_sum_assignment(cost)
        # With K padded to >= Q, every query row receives exactly one assignment.
        result[gi] = float(sims[gi][row_ind, col_ind].sum() / q)
    return result


def build_score_matrix(
    query_sets: list[np.ndarray],
    gallery_sets: list[np.ndarray],
    unmatched_score: float,
    max_vectorized_assignments: int,
) -> np.ndarray:
    nq, ng = len(query_sets), len(gallery_sets)
    if all(q.shape[0] == 1 for q in query_sets) and all(g.shape[0] == 1 for g in gallery_sets):
        q = np.stack([x[0] for x in query_sets], axis=0)
        g = np.stack([x[0] for x in gallery_sets], axis=0)
        return (q @ g.T).astype(np.float32, copy=False)

    groups: dict[int, list[int]] = defaultdict(list)
    for gi, gf in enumerate(gallery_sets):
        groups[gf.shape[0]].append(gi)

    scores = np.empty((nq, ng), dtype=np.float32)
    for qi, qf in enumerate(tqdm(query_sets, desc="SetMatch", unit="query")):
        for k, indices in groups.items():
            stack = np.stack([gallery_sets[i] for i in indices], axis=0)
            group_scores = setmatch_group_scores(
                qf, stack, unmatched_score, max_vectorized_assignments
            )
            scores[qi, np.asarray(indices)] = group_scores
    return scores


def relative_to_root(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def main() -> None:
    root = repo_root()
    config_path = Path(__file__).with_name("config.yaml")
    cfg = load_yaml(config_path)

    gallery_manifest = (root / cfg["data"]["gallery_manifest"]).resolve()
    query_manifest = (root / cfg["data"]["query_manifest"]).resolve()
    gallery_rows = read_jsonl(gallery_manifest)
    query_rows = read_jsonl(query_manifest)
    if not gallery_rows or not query_rows:
        raise RuntimeError("Canonical query/gallery manifests must be non-empty")

    gallery_entries, gallery_schema = parse_gallery_rows(gallery_rows, cfg, root, gallery_manifest)
    query_entries, query_cases, query_schema = parse_query_rows(query_rows, cfg, root, query_manifest)

    source_dir = ensure_official_source(cfg, root)
    device_cfg = str(cfg["runtime"].get("device", "cuda"))
    if device_cfg.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("config requests CUDA but torch.cuda.is_available() is false")
    device = torch.device(device_cfg)

    model, img2text, transform, tokenizer, tokenize, split_ind, args = load_word4per(
        cfg, root, source_dir, device
    )

    all_gallery_paths = [c.image_path for entry in gallery_entries for c in entry]
    all_query_components = [c for entry in query_entries for c in entry]

    gallery_cache = encode_gallery_images(
        model,
        transform,
        all_gallery_paths,
        device,
        int(cfg["runtime"].get("image_batch_size", 256)),
    )
    query_cache = encode_query_components(
        model,
        img2text,
        transform,
        tokenizer,
        tokenize,
        split_ind,
        all_query_components,
        int(args.text_length),
        device,
        int(cfg["runtime"].get("query_batch_size", 128)),
    )

    gallery_sets = [np.stack([gallery_cache[c.image_path] for c in entry], axis=0) for entry in gallery_entries]
    query_sets = [np.stack([query_cache[c] for c in entry], axis=0) for entry in query_entries]

    set_cfg = cfg.get("setmatch", {})
    scores = build_score_matrix(
        query_sets,
        gallery_sets,
        unmatched_score=float(set_cfg.get("unmatched_score", -1.0)),
        max_vectorized_assignments=int(set_cfg.get("max_vectorized_assignments", 4096)),
    )

    expected_shape = (len(query_rows), len(gallery_rows))
    if scores.shape != expected_shape:
        raise AssertionError(f"scores shape {scores.shape} != canonical shape {expected_shape}")
    if not np.isfinite(scores).all():
        raise FloatingPointError("scores.npy contains NaN/Inf")

    out_dir = (root / cfg["output"]["dir"]).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    scores_path = out_dir / "scores.npy"
    np.save(scores_path, scores.astype(np.float32, copy=False))

    case_counts: dict[str, int] = defaultdict(int)
    component_counts: dict[str, int] = defaultdict(int)
    for case, comps in zip(query_cases, query_entries):
        case_counts[case] += 1
        component_counts[str(len(comps))] += 1

    run_meta = {
        "method": cfg["method"],
        "display_name": cfg["display_name"],
        "group": cfg["group"],
        "cpr_supervision": cfg["cpr_supervision"],
        "paper": cfg.get("paper", {}),
        "official_source": {
            "repository": cfg["source"]["repository"],
            "commit": cfg["source"]["commit"],
            "subdir": cfg["source"].get("subdir", "old_project"),
        },
        "model": {
            "pretrain_choice": str(args.pretrain_choice),
            "img_size": list(args.img_size),
            "text_length": int(args.text_length),
            "mlp_depth": int(args.mlp_depth),
            "stage2_checkpoint": relative_to_root((root / cfg["checkpoint"]["stage2"]), root),
            "stage2_config": relative_to_root((root / cfg["checkpoint"]["stage2_config"]), root),
        },
        "adaptation": {
            "name": "SetMatch",
            "definition": "maximum-weight one-to-one assignment over component cosine scores; mean over query components",
            "unmatched_score": float(set_cfg.get("unmatched_score", -1.0)),
        },
        "manifest_schema": {**gallery_schema, **query_schema},
        "query_case_counts": dict(case_counts),
        "query_component_count_histogram": dict(component_counts),
        "runtime": {
            "device": str(device),
            "image_batch_size": int(cfg["runtime"].get("image_batch_size", 256)),
            "query_batch_size": int(cfg["runtime"].get("query_batch_size", 128)),
        },
        "config": relative_to_root(config_path, root),
        "num_queries": len(query_rows),
        "num_gallery": len(gallery_rows),
        "scores": relative_to_root(scores_path, root),
        "higher_is_better": True,
        "notes": [
            "Scores include the complete canonical gallery; exact query-image exclusion is left to evaluate.py.",
            "No CPR-pilot training or tuning is performed by this inference adapter.",
        ],
    }
    with (out_dir / "run.json").open("w", encoding="utf-8") as f:
        json.dump(run_meta, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(f"Saved {scores_path} with shape {scores.shape}")
    print(f"Saved {out_dir / 'run.json'}")


if __name__ == "__main__":
    main()
