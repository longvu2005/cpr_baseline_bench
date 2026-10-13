"""Label-free input views, verified feature caches and bounded score writers."""
from __future__ import annotations

import importlib.util
import hashlib
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import yaml

from benchmark_progress import progress_bar
from cpr_person.cache import StageCache, sha256_file
from cpr_person.detection import image_path, read_rgb
from . import VERSION


def load_config(path):
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError("Expected a YAML object")
    return cfg


def load_jsonl(path):
    rows = []
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{number}: expected an object")
            rows.append(value)
    if not rows:
        raise ValueError(f"Empty manifest: {path}")
    return rows


def public_inputs(gallery, queries):
    """Allowlist: relevance, identity IDs and case labels never reach a model."""
    g = [{"image_id": row["image_id"], "path": row["path"]} for row in gallery]
    q = []
    for row in queries:
        q.append({key: row[key] for key in
                  ("query_id", "image_id", "text", "final_desc", "final_change", "relation_text")
                  if key in row})
        q[-1]["subjects"] = [{key: subject[key] for key in
                              ("subject_id", "select_text", "modify_text") if key in subject}
                             for subject in row.get("subjects", [])]
    ids = [row["image_id"] for row in g]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate gallery image IDs")
    index = {key: i for i, key in enumerate(ids)}
    query_ids = [row["query_id"] for row in q]
    if len(query_ids) != len(set(query_ids)):
        raise ValueError("Duplicate query IDs")
    if any(row["image_id"] not in index for row in q):
        raise ValueError("A query reference is absent from the complete gallery")
    return g, q, index


def query_text(query, field="text"):
    text = query.get(field)
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"Empty public query field {field!r}: {query.get('query_id')}")
    return text.strip()


def image_digest(gallery):
    # Include image bytes, not just paths or timestamps (a mounted dataset may
    # change without changing any manifest). Keep manifest order in the digest.
    digest = hashlib.sha256()
    for row in progress_bar(gallery, desc="Fingerprint images", unit="image"):
        digest.update(json.dumps([row["image_id"], row["path"], sha256_file(image_path(row))],
                                 separators=(",", ":")).encode())
    return digest.hexdigest()


def normalized(features):
    x = np.asarray(features, dtype=np.float32)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError("Expected finite feature rows")
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    if np.any(norms < 1e-12):
        raise ValueError("A model returned a zero embedding")
    return x / norms


def compose_linear(images, texts, alpha):
    if not np.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("Fixed composition alpha must be in [0,1]")
    a, b = normalized(images), normalized(texts)
    if a.shape != b.shape:
        raise ValueError("Image/text feature dimensions differ")
    return normalized(alpha * a + (1 - alpha) * b)


def cached_features(cache, filename, rows, make_batch, batch_size):
    if rows < 1 or batch_size < 1:
        raise ValueError("Feature count and batch size must be positive")
    path = cache.find(filename)
    if path is not None:
        x = np.load(path, mmap_mode="r", allow_pickle=False)
        if x.ndim == 2 and len(x) == rows and x.dtype == np.float32:
            for start in range(0, rows, batch_size):
                if not np.isfinite(x[start:start + batch_size]).all():
                    raise ValueError("Non-finite cached features")
            return x
        raise ValueError(f"Invalid verified feature cache: {path}")
    path = cache.directory / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".partial")
    output = None
    try:
        for start in progress_bar(range(0, rows, batch_size), desc=filename, unit="batch"):
            end = min(rows, start + batch_size)
            block = normalized(make_batch(start, end))
            if len(block) != end - start:
                raise ValueError("Model changed feature row count")
            if output is None:
                output = np.lib.format.open_memmap(temp, mode="w+", dtype=np.float32, shape=(rows, block.shape[1]))
            if output.shape[1] != block.shape[1]:
                raise ValueError("Feature dimension changed between batches")
            output[start:end] = block
        output.flush()
        del output
        output = None
        os.replace(temp, path)
        cache.seal(path)
    finally:
        if output is not None:
            del output
        temp.unlink(missing_ok=True)
    return np.load(path, mmap_mode="r", allow_pickle=False)


def feature_cache(stage, fingerprint, runtime):
    # Batch size / device / dtype are relevant for actual neural outputs.
    return StageCache(stage, {"version": VERSION, **fingerprint, "runtime": runtime},
                      write_root="cache/additional")


def write_scene_scores(queries, gallery, path, query_batch_size, gallery_batch_size):
    if query_batch_size < 1 or gallery_batch_size < 1 or queries.shape[1] != gallery.shape[1]:
        raise ValueError("Invalid scoring sizes or feature dimensions")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".partial")
    scores = np.lib.format.open_memmap(temp, mode="w+", dtype=np.float32, shape=(len(queries), len(gallery)))
    try:
        for start in progress_bar(range(0, len(queries), query_batch_size), desc="Score full gallery", unit="batch"):
            q = np.asarray(queries[start:start + query_batch_size], dtype=np.float32)
            for gs in range(0, len(gallery), gallery_batch_size):
                block = q @ np.asarray(gallery[gs:gs + gallery_batch_size], dtype=np.float32).T
                if not np.isfinite(block).all():
                    raise FloatingPointError("Non-finite similarities")
                scores[start:start + len(q), gs:gs + block.shape[1]] = block
        scores.flush()
    finally:
        del scores
    os.replace(temp, path)


def load_source_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def seed_runtime(runtime):
    import torch
    seed = int(runtime["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    device = str(runtime["device"])
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(device)


def source_images(gallery, start, end):
    return [read_rgb(image_path(gallery[i])) for i in range(start, end)]
