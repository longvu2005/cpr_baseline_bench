#!/usr/bin/env python3
"""P6 MagicLens Large: direct full-scene CPR adapter around official JAX/Flax inference."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import os
import pickle
import shutil
import subprocess
import sys
import types
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Sequence

# Must be configured before JAX is imported.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.dont_write_bytecode = True

import numpy as np
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark_data import ensure_gallery_layout  # noqa: E402
from benchmark_progress import PhaseTracker, progress_bar  # noqa: E402

DEFAULT_CONFIG = Path(__file__).resolve().parent / "config.yaml"
METHOD_ID = "magiclens_large"
ADAPTER_VERSION = "2026-08-14-v3-inference-complete-source"
GALLERY_CACHE_SCHEMA = 1
QUERY_CACHE_SCHEMA = 1


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise TypeError(f"Expected YAML mapping: {path}")
    return data


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise TypeError(f"{path}:{lineno}: expected JSON object")
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


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def meta_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".meta.json")


def tracked_dirty(checkout: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(checkout), "status", "--porcelain", "--untracked-files=no"],
        text=True,
    ).strip()


def validate_checkout(name: str, spec: dict[str, Any]) -> Path:
    checkout = resolve_path(str(spec["local_checkout"]))
    if not checkout.is_dir():
        raise FileNotFoundError(
            f"Missing pinned {name} source: {rel(checkout)}. Run checkpoint preparation first."
        )
    actual = subprocess.check_output(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True
    ).strip()
    expected = str(spec["commit"])
    if actual != expected:
        raise RuntimeError(f"{name} source commit mismatch: expected {expected}, got {actual}")
    dirty = tracked_dirty(checkout)
    if dirty:
        raise RuntimeError(f"Pinned {name} source has tracked modifications:\n{dirty}")
    return checkout


def validate_prepared_artifacts(cfg: dict[str, Any]) -> tuple[Path, dict[str, Path], dict[str, Any]]:
    checkouts = {
        name: validate_checkout(name, cfg["source"][name])
        for name in ("magiclens", "scenic", "openai_clip")
    }
    checkpoint = resolve_path(str(cfg["checkpoint"]["path"]))
    marker_path = resolve_path(str(cfg["checkpoint"]["prepared_marker"]))
    marker = read_json(marker_path)
    if marker is None:
        raise FileNotFoundError(
            f"Missing MagicLens prepared marker: {rel(marker_path)}. Run checkpoint preparation first."
        )
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing MagicLens checkpoint: {rel(checkpoint)}")

    checkpoint_info = marker.get("checkpoint", {})
    if not isinstance(checkpoint_info, dict):
        raise RuntimeError("Invalid MagicLens prepared marker checkpoint metadata")
    actual_size = int(checkpoint.stat().st_size)
    expected_size = int(checkpoint_info.get("size", -1))
    if actual_size != expected_size:
        raise RuntimeError(
            f"MagicLens checkpoint size changed after preparation: {actual_size} != {expected_size}"
        )
    actual_sha = sha256_file(checkpoint)
    expected_sha = str(checkpoint_info.get("sha256", ""))
    if actual_sha != expected_sha:
        raise RuntimeError("MagicLens checkpoint SHA256 changed after preparation")

    marker_sources = marker.get("source", {})
    for name in checkouts:
        expected_commit = str(cfg["source"][name]["commit"])
        recorded = marker_sources.get(name, {}) if isinstance(marker_sources, dict) else {}
        if not isinstance(recorded, dict) or recorded.get("commit") != expected_commit:
            raise RuntimeError(f"MagicLens marker source commit mismatch for {name}")

    return checkpoint, checkouts, marker


def install_tensorflow_gfile_shim() -> None:
    """Satisfy Scenic's unused tensorflow.io.gfile imports without importing TensorFlow.

    MagicLens uses Scenic's CLIP model definitions and tokenizer only. Their modules import
    tensorflow.io.gfile for checkpoint-download helpers that are never called by this adapter.
    This shim implements those filesystem methods but changes no model/tokenizer math.
    """

    if "tensorflow.io" in sys.modules:
        return

    class _GFileNamespace:
        GFile = staticmethod(open)
        exists = staticmethod(os.path.exists)
        isdir = staticmethod(os.path.isdir)
        makedirs = staticmethod(lambda path: os.makedirs(path, exist_ok=True))
        remove = staticmethod(os.remove)
        copy = staticmethod(
            lambda src, dst, overwrite=False: shutil.copyfile(src, dst)
            if overwrite or not os.path.exists(dst)
            else (_ for _ in ()).throw(FileExistsError(dst))
        )

    tf_module = types.ModuleType("tensorflow")
    io_module = types.ModuleType("tensorflow.io")
    io_module.gfile = _GFileNamespace
    tf_module.io = io_module
    sys.modules["tensorflow"] = tf_module
    sys.modules["tensorflow.io"] = io_module


def install_openai_clip_tokenizer_shim(openai_clip_checkout: Path) -> None:
    """Expose only pinned OpenAI CLIP simple_tokenizer without importing PyTorch CLIP."""

    clip_dir = openai_clip_checkout / "clip"
    simple_tokenizer = clip_dir / "simple_tokenizer.py"
    if not simple_tokenizer.is_file():
        raise FileNotFoundError(simple_tokenizer)

    package = types.ModuleType("clip")
    package.__path__ = [str(clip_dir)]
    package.__package__ = "clip"
    sys.modules["clip"] = package

    spec = importlib.util.spec_from_file_location("clip.simple_tokenizer", simple_tokenizer)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load pinned OpenAI CLIP tokenizer: {simple_tokenizer}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["clip.simple_tokenizer"] = module
    spec.loader.exec_module(module)
    package.simple_tokenizer = module


def import_official_stack(checkouts: dict[str, Path]):
    install_tensorflow_gfile_shim()
    install_openai_clip_tokenizer_shim(checkouts["openai_clip"])

    scenic = checkouts["scenic"]
    magiclens = checkouts["magiclens"]
    sys.path.insert(0, str(scenic))
    sys.path.insert(0, str(magiclens))

    import jax
    import jax.numpy as jnp
    from flax import serialization
    from scenic.projects.baselines.clip import tokenizer as clip_tokenizer

    official_model = importlib.import_module("model")
    model_file = Path(getattr(official_model, "__file__", "")).resolve()
    if model_file.parent != magiclens.resolve():
        raise RuntimeError(f"Imported wrong MagicLens model.py: {model_file}")
    return jax, jnp, serialization, clip_tokenizer, official_model.MagicLens


def gallery_image_path(row: dict[str, Any], index: int) -> Path:
    value = row.get("path")
    if not isinstance(value, str) or not value.strip():
        raise KeyError(f"Gallery row {index} has no usable path")
    path = resolve_path(value)
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


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


def query_gallery_indices(
    queries: Sequence[dict[str, Any]], gallery_index: dict[Any, int]
) -> np.ndarray:
    indices: list[int] = []
    for qi, query in enumerate(queries):
        image_id = query.get("image_id")
        if image_id not in gallery_index:
            raise ValueError(f"Query row {qi}: image_id {image_id!r} missing from gallery")
        indices.append(gallery_index[image_id])
    return np.asarray(indices, dtype=np.int64)


def process_image_official(path: Path, size: int, jax, jnp) -> np.ndarray:
    """Match official magiclens/data_utils.py::process_img, preserving its square resize."""

    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"))
    ima = jnp.asarray(array)[jnp.newaxis, ...]
    ima = ima / (ima.max() + 1e-12)
    ima = jax.image.resize(ima, (1, size, size, 3), method="bilinear")
    return np.asarray(ima, dtype=np.float32)


def preprocess_paths(
    paths: Sequence[Path],
    *,
    size: int,
    workers: int,
    jax,
    jnp,
) -> np.ndarray:
    def one(path: Path) -> np.ndarray:
        return process_image_official(path, size, jax, jnp)

    if workers <= 1:
        arrays = [one(path) for path in paths]
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            arrays = list(executor.map(one, paths))
    return np.concatenate(arrays, axis=0)


def build_query_tokens(
    cfg: dict[str, Any], queries: Sequence[dict[str, Any]], tokenizer: Callable[[str], np.ndarray]
) -> np.ndarray:
    field = str(cfg["composition"]["query_text_field"])
    tokens: list[np.ndarray] = []
    for qi, query in enumerate(queries):
        value = query.get(field)
        if not isinstance(value, str) or not value.strip():
            raise KeyError(f"Query row {qi} has no usable {field!r}")
        text = value
        try:
            encoded = np.asarray(tokenizer(text), dtype=np.int32)
        except RuntimeError as error:
            raise RuntimeError(
                f"MagicLens query row {qi} exceeds the official CLIP 77-token context. "
                "The benchmark adapter does not silently truncate instructions."
            ) from error
        if encoded.shape != (1, int(cfg["model"]["text_context_length"])):
            raise RuntimeError(f"Unexpected MagicLens token shape at query {qi}: {encoded.shape}")
        tokens.append(encoded)
    return np.concatenate(tokens, axis=0)


def pad_first_axis(array: np.ndarray, size: int) -> np.ndarray:
    if array.shape[0] > size:
        raise ValueError("Cannot pad array to a smaller batch")
    if array.shape[0] == size:
        return array
    shape = (size,) + array.shape[1:]
    padded = np.zeros(shape, dtype=array.dtype)
    padded[: array.shape[0]] = array
    return padded


def load_feature_cache(
    path: Path,
    expected_meta: dict[str, Any],
    expected_shape: tuple[int, int],
    label: str,
) -> np.ndarray | None:
    if not path.is_file() or not meta_path(path).is_file():
        return None
    if read_json(meta_path(path)) != expected_meta:
        print(f"Ignoring stale {label} cache: {rel(path)}", flush=True)
        return None
    try:
        array = np.load(path, mmap_mode="r", allow_pickle=False)
    except Exception as error:
        print(f"Ignoring invalid {label} cache: {error}", flush=True)
        return None
    if array.shape != expected_shape or array.dtype != np.float32:
        print(f"Ignoring incompatible {label} cache: {array.shape} {array.dtype}", flush=True)
        return None
    for start in range(0, len(array), 4096):
        if not np.isfinite(np.asarray(array[start : start + 4096])).all():
            print(f"Ignoring non-finite {label} cache", flush=True)
            return None
    print(f"Using {label} cache: {rel(path)}", flush=True)
    return array


def feature_cache_meta(
    *,
    schema: int,
    cfg: dict[str, Any],
    config_path: Path,
    gallery_manifest: Path,
    query_manifest: Path | None,
    checkpoint_sha256: str,
    checkouts: dict[str, Path],
    kind: str,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": schema,
        "kind": kind,
        "adapter_version": ADAPTER_VERSION,
        "config_sha256": sha256_file(config_path),
        "gallery_manifest_sha256": sha256_file(gallery_manifest),
        "checkpoint_sha256": checkpoint_sha256,
        "source_commits": {
            name: str(cfg["source"][name]["commit"])
            for name in ("magiclens", "scenic", "openai_clip")
        },
        "model_size": str(cfg["model"]["size"]),
        "preprocessing": cfg["preprocessing"],
    }
    if query_manifest is not None:
        payload["query_manifest_sha256"] = sha256_file(query_manifest)
        payload["composition"] = cfg["composition"]
    return payload


def make_encoder(model, params, jax, embedding_key: str):
    def encode(model_params, ids, images):
        return model.apply(model_params, {"ids": ids, "image": images})[embedding_key]

    return jax.jit(encode)


def encode_gallery(
    *,
    gallery: Sequence[dict[str, Any]],
    model,
    params,
    encoder,
    tokenizer,
    cfg: dict[str, Any],
    cache_path: Path,
    cache_meta: dict[str, Any],
    jax,
    jnp,
) -> np.ndarray:
    dim = int(cfg["model"]["embedding_dim"])
    expected_shape = (len(gallery), dim)
    cached = load_feature_cache(cache_path, cache_meta, expected_shape, "MagicLens gallery-feature")
    if cached is not None:
        return cached

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path(cache_path).unlink(missing_ok=True)
    features = np.lib.format.open_memmap(
        cache_path, mode="w+", dtype=np.float32, shape=expected_shape
    )

    batch_size = int(cfg["runtime"]["gallery_batch_size"])
    size = int(cfg["model"]["image_size"])
    workers = int(cfg["runtime"]["preprocess_workers"])
    null = np.asarray(tokenizer(str(cfg["composition"]["gallery_text"])), dtype=np.int32)
    if null.shape != (1, int(cfg["model"]["text_context_length"])):
        raise RuntimeError(f"Unexpected empty-text token shape: {null.shape}")

    starts = range(0, len(gallery), batch_size)
    total = (len(gallery) + batch_size - 1) // batch_size
    for start in progress_bar(starts, desc="MagicLens encode gallery", total=total, unit="batch"):
        end = min(start + batch_size, len(gallery))
        paths = [gallery_image_path(gallery[gi], gi) for gi in range(start, end)]
        images = preprocess_paths(paths, size=size, workers=workers, jax=jax, jnp=jnp)
        ids = np.repeat(null, end - start, axis=0)
        images = pad_first_axis(images, batch_size)
        ids = pad_first_axis(ids, batch_size)
        output = encoder(params, ids, images)
        output.block_until_ready()
        array = np.asarray(output[: end - start], dtype=np.float32)
        if array.shape != (end - start, dim) or not np.isfinite(array).all():
            raise RuntimeError(f"Invalid MagicLens gallery embedding batch: {array.shape}")
        features[start:end] = array
    features.flush()
    write_json(meta_path(cache_path), cache_meta)
    return np.load(cache_path, mmap_mode="r", allow_pickle=False)


def encode_queries(
    *,
    gallery: Sequence[dict[str, Any]],
    queries: Sequence[dict[str, Any]],
    query_indices: np.ndarray,
    query_tokens: np.ndarray,
    model,
    params,
    encoder,
    cfg: dict[str, Any],
    cache_path: Path,
    cache_meta: dict[str, Any],
    jax,
    jnp,
) -> np.ndarray:
    dim = int(cfg["model"]["embedding_dim"])
    expected_shape = (len(queries), dim)
    cached = load_feature_cache(cache_path, cache_meta, expected_shape, "MagicLens query-feature")
    if cached is not None:
        return cached

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path(cache_path).unlink(missing_ok=True)
    features = np.lib.format.open_memmap(
        cache_path, mode="w+", dtype=np.float32, shape=expected_shape
    )

    batch_size = int(cfg["runtime"]["query_batch_size"])
    size = int(cfg["model"]["image_size"])
    workers = int(cfg["runtime"]["preprocess_workers"])
    starts = range(0, len(queries), batch_size)
    total = (len(queries) + batch_size - 1) // batch_size
    for start in progress_bar(starts, desc="MagicLens compose queries", total=total, unit="batch"):
        end = min(start + batch_size, len(queries))
        indices = query_indices[start:end]
        paths = [gallery_image_path(gallery[int(gi)], int(gi)) for gi in indices]
        images = preprocess_paths(paths, size=size, workers=workers, jax=jax, jnp=jnp)
        ids = np.asarray(query_tokens[start:end], dtype=np.int32)
        images = pad_first_axis(images, batch_size)
        ids = pad_first_axis(ids, batch_size)
        output = encoder(params, ids, images)
        output.block_until_ready()
        array = np.asarray(output[: end - start], dtype=np.float32)
        if array.shape != (end - start, dim) or not np.isfinite(array).all():
            raise RuntimeError(f"Invalid MagicLens query embedding batch: {array.shape}")
        features[start:end] = array
    features.flush()
    write_json(meta_path(cache_path), cache_meta)
    return np.load(cache_path, mmap_mode="r", allow_pickle=False)


def load_official_model(cfg: dict[str, Any], checkpoint: Path, MagicLens, serialization, jax, jnp):
    model_size = str(cfg["model"]["size"])
    model = MagicLens(model_size)
    rng = jax.random.PRNGKey(0)
    dummy = {
        "ids": jnp.ones((1, 1, int(cfg["model"]["text_context_length"])), dtype=jnp.int32),
        "image": jnp.ones(
            (1, int(cfg["model"]["image_size"]), int(cfg["model"]["image_size"]), 3),
            dtype=jnp.float32,
        ),
    }
    params = model.init(rng, dummy)
    with checkpoint.open("rb") as handle:
        model_bytes = pickle.load(handle)
    if not isinstance(model_bytes, (bytes, bytearray)):
        raise TypeError(
            f"Official MagicLens checkpoint outer pickle must contain bytes, got {type(model_bytes).__name__}"
        )
    params = serialization.from_bytes(params, model_bytes)
    params = jax.device_put(params)
    return model, params


def validate_scores(scores: np.ndarray, shape: tuple[int, int]) -> None:
    if scores.shape != shape:
        raise ValueError(f"scores shape {scores.shape}, expected {shape}")
    if scores.dtype != np.float32:
        raise TypeError(f"scores.npy must be float32, got {scores.dtype}")
    for start in range(0, shape[0], 256):
        if not np.isfinite(np.asarray(scores[start : start + 256])).all():
            raise ValueError("scores.npy contains NaN/Inf")


def score_matrix(
    query_features: np.ndarray,
    gallery_features: np.ndarray,
    output_path: Path,
    batch_size: int,
    jax,
) -> np.ndarray:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scores = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=np.float32,
        shape=(len(query_features), len(gallery_features)),
    )
    gallery_device = jax.device_put(np.asarray(gallery_features, dtype=np.float32))

    @jax.jit
    def dot(query_batch, gallery_matrix):
        return query_batch @ gallery_matrix.T

    starts = range(0, len(query_features), batch_size)
    total = (len(query_features) + batch_size - 1) // batch_size
    for start in progress_bar(starts, desc="MagicLens score queries", total=total, unit="batch"):
        end = min(start + batch_size, len(query_features))
        query = np.asarray(query_features[start:end], dtype=np.float32)
        query = pad_first_axis(query, batch_size)
        output = dot(query, gallery_device)
        output.block_until_ready()
        batch_scores = np.asarray(output[: end - start], dtype=np.float32)
        if not np.isfinite(batch_scores).all():
            raise RuntimeError("MagicLens scoring produced NaN/Inf")
        scores[start:end] = batch_scores
    scores.flush()
    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description="Run P6 MagicLens Large CPR baseline")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()
    config_path = resolve_path(args.config)
    cfg = load_yaml(config_path)

    if bool(cfg["runtime"].get("jax_preallocate", False)):
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"

    tracker = PhaseTracker(METHOD_ID, total=7)

    with tracker.phase("Load config, canonical manifests, and prepared artifacts"):
        checkpoint, checkouts, marker = validate_prepared_artifacts(cfg)
        gallery_manifest = resolve_path(str(cfg["data"]["gallery_manifest"]))
        query_manifest = resolve_path(str(cfg["data"]["query_manifest"]))
        gallery = load_jsonl(gallery_manifest)
        queries = load_jsonl(query_manifest)
        gallery_root = ensure_gallery_layout(ROOT, gallery_rows=gallery, repair=True)
        gallery_index = build_gallery_index(gallery)
        query_indices = query_gallery_indices(queries, gallery_index)
        tracker.log(f"gallery={len(gallery):,} queries={len(queries):,}")
        tracker.log(f"gallery_root={gallery_root}")

    with tracker.phase("Import pinned official JAX/Flax MagicLens stack"):
        jax, jnp, serialization, clip_tokenizer, MagicLens = import_official_stack(checkouts)
        backend = jax.default_backend()
        devices = [str(device) for device in jax.devices()]
        tracker.log(f"jax={jax.__version__} backend={backend} devices={devices}")
        if str(cfg["runtime"]["backend"]) not in ("auto", backend):
            raise RuntimeError(
                f"MagicLens config requests backend={cfg['runtime']['backend']!r}, active JAX backend={backend!r}"
            )

    with tracker.phase("Load official MagicLens Large checkpoint and tokenizer"):
        bpe_path = checkouts["openai_clip"] / "clip/bpe_simple_vocab_16e6.txt.gz"
        tokenizer = clip_tokenizer.build_tokenizer(
            bpe_path=str(bpe_path),
            truncate=bool(cfg["composition"]["tokenizer_truncate"]),
        )
        query_tokens = build_query_tokens(cfg, queries, tokenizer)
        model, params = load_official_model(
            cfg, checkpoint, MagicLens, serialization, jax, jnp
        )
        encoder = make_encoder(
            model, params, jax, str(cfg["model"]["normalized_embedding_key"])
        )
        tracker.log(
            f"model=MagicLens-{cfg['model']['size']} backbone={cfg['model']['backbone']} "
            f"checkpoint_sha256={marker['checkpoint']['sha256']}"
        )

    with tracker.phase("Encode complete canonical gallery with empty instruction"):
        gallery_cache = resolve_path(str(cfg["cache"]["gallery_features"]))
        gallery_meta = feature_cache_meta(
            schema=GALLERY_CACHE_SCHEMA,
            cfg=cfg,
            config_path=config_path,
            gallery_manifest=gallery_manifest,
            query_manifest=None,
            checkpoint_sha256=str(marker["checkpoint"]["sha256"]),
            checkouts=checkouts,
            kind="gallery",
        )
        gallery_features = encode_gallery(
            gallery=gallery,
            model=model,
            params=params,
            encoder=encoder,
            tokenizer=tokenizer,
            cfg=cfg,
            cache_path=gallery_cache,
            cache_meta=gallery_meta,
            jax=jax,
            jnp=jnp,
        )
        tracker.log(f"gallery_features={gallery_features.shape} cache={rel(gallery_cache)}")

    with tracker.phase("Encode full-scene queries with full textual instruction"):
        query_cache = resolve_path(str(cfg["cache"]["query_features"]))
        query_meta = feature_cache_meta(
            schema=QUERY_CACHE_SCHEMA,
            cfg=cfg,
            config_path=config_path,
            gallery_manifest=gallery_manifest,
            query_manifest=query_manifest,
            checkpoint_sha256=str(marker["checkpoint"]["sha256"]),
            checkouts=checkouts,
            kind="query",
        )
        query_features = encode_queries(
            gallery=gallery,
            queries=queries,
            query_indices=query_indices,
            query_tokens=query_tokens,
            model=model,
            params=params,
            encoder=encoder,
            cfg=cfg,
            cache_path=query_cache,
            cache_meta=query_meta,
            jax=jax,
            jnp=jnp,
        )
        tracker.log(f"query_features={query_features.shape} cache={rel(query_cache)}")

    with tracker.phase("Compute complete query-gallery cosine score matrix"):
        output_dir = resolve_path(str(cfg["output"]["dir"]))
        output_dir.mkdir(parents=True, exist_ok=True)
        scores_path = output_dir / "scores.npy"
        scores = score_matrix(
            query_features,
            gallery_features,
            scores_path,
            int(cfg["runtime"]["score_batch_size"]),
            jax,
        )
        validate_scores(scores, (len(queries), len(gallery)))

    with tracker.phase("Write reproducibility metadata"):
        output_dir = resolve_path(str(cfg["output"]["dir"]))
        payload = {
            "method": cfg["method"],
            "display_name": cfg["display_name"],
            "group": cfg["group"],
            "cpr_supervision": cfg["cpr_supervision"],
            "paper": cfg["paper"],
            "source": {
                name: {
                    "repository": cfg["source"][name]["repository"],
                    "commit": cfg["source"][name]["commit"],
                    "checkout": rel(checkouts[name]),
                }
                for name in ("magiclens", "scenic", "openai_clip")
            },
            "checkpoint": {
                "path": rel(checkpoint),
                "sha256": marker["checkpoint"]["sha256"],
                "size": marker["checkpoint"]["size"],
                "status": cfg["checkpoint"]["status"],
                "variant": cfg["checkpoint"]["variant"],
            },
            "model": cfg["model"],
            "preprocessing": cfg["preprocessing"],
            "composition": cfg["composition"],
            "runtime": {**cfg["runtime"], "resolved_jax_backend": jax.default_backend()},
            "cpr_adapter": {
                "query": "full canonical scene image + canonical query.text instruction",
                "gallery": "full canonical scene image + empty instruction",
                "localization": "none",
                "setmatch": False,
                "uses_target_ids": False,
                "uses_positives": False,
                "query_image_excluded_inside_method": False,
                "single_multi_relational_policy": "same scene-level MagicLens scoring for every case",
            },
            "gallery_features": rel(resolve_path(str(cfg["cache"]["gallery_features"]))),
            "query_features": rel(resolve_path(str(cfg["cache"]["query_features"]))),
            "config": rel(config_path),
            "num_queries": len(queries),
            "num_gallery": len(gallery),
            "scores": rel(output_dir / "scores.npy"),
            "higher_is_better": True,
            "adapter_version": ADAPTER_VERSION,
        }
        write_json(output_dir / "run.json", payload)
        tracker.log(f"scores={rel(output_dir / 'scores.npy')} run={rel(output_dir / 'run.json')}")

    tracker.finish()


if __name__ == "__main__":
    main()
