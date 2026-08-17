#!/usr/bin/env python3
"""P9: AdaFocal + SetMatch.

Main-table adaptation:
  * shared Grounding DINO person detections (no GT boxes)
  * CLIP ViT-B/32 + Hungarian for subject -> predicted query-person anchor
  * official AdaFocal scalar query branch on full reference image + predicted bbox
  * official AdaFocal target branch on detected gallery-person crops
  * SetMatch: Hungarian max-sum followed by strict-min aggregation
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

import clip
import numpy as np
import torch
import yaml
from PIL import Image
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark_progress import PhaseTracker, progress_bar  # noqa: E402

DEFAULT_CONFIG = Path(__file__).resolve().parent / "config.yaml"
METHOD_ID = "adafocal_setmatch"
ADAPTER_VERSION = "2026-08-17-v1-official-scalar-pred-anchor-person-target-setmatch"
TARGET_FEATURE_SCHEMA = 1


def load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
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
                raise TypeError(f"{path}:{lineno}: JSONL row must be an object")
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


def device_from(name: str) -> torch.device:
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return device


def load_module(method_dir: Path, module_name: str):
    run_path = method_dir / "run.py"
    if not run_path.is_file():
        raise FileNotFoundError(f"Missing source adapter: {rel(run_path)}")
    spec = importlib.util.spec_from_file_location(module_name, run_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import adapter from {rel(run_path)}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def build_gallery_index(gallery: Sequence[dict[str, Any]]) -> dict[Any, int]:
    index: dict[Any, int] = {}
    for gi, row in enumerate(gallery):
        image_id = row.get("image_id")
        if image_id is None:
            raise KeyError(f"Gallery row {gi} missing image_id")
        if image_id in index:
            raise ValueError(f"Duplicate gallery image_id: {image_id!r}")
        index[image_id] = gi
    return index


def query_gallery_indices(
    queries: Sequence[dict[str, Any]], gallery_index: dict[Any, int]
) -> np.ndarray:
    result = []
    for qi, query in enumerate(queries):
        image_id = query.get("image_id")
        if image_id not in gallery_index:
            raise ValueError(f"Query row {qi}: image_id {image_id!r} missing from gallery")
        result.append(gallery_index[image_id])
    return np.asarray(result, dtype=np.int64)


def image_path(row: dict[str, Any], index: int) -> Path:
    value = row.get("path")
    if not isinstance(value, str) or not value:
        raise KeyError(f"Gallery row {index} has no usable path")
    path = (ROOT / value).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def ensure_clean_pinned_source(cfg: dict[str, Any]) -> Path:
    source = cfg["official_source"]
    checkout = resolve_path(str(source["local_checkout"]))
    expected = str(source["commit"])
    repository = str(source["repository"])

    if not checkout.exists():
        if not bool(source.get("auto_clone", True)):
            raise FileNotFoundError(f"Missing official OACIR checkout: {rel(checkout)}")
        if shutil.which("git") is None:
            raise RuntimeError("git is required to clone the official OACIR source")
        checkout.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", repository, str(checkout)], check=True)

    if not (checkout / ".git").is_dir():
        raise RuntimeError(f"Official OACIR path is not a git checkout: {rel(checkout)}")

    dirty = subprocess.check_output(
        ["git", "-C", str(checkout), "status", "--porcelain", "--untracked-files=no"],
        text=True,
    ).strip()
    if dirty:
        raise RuntimeError(f"Pinned OACIR source has tracked local modifications:\n{dirty}")

    actual = subprocess.check_output(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual != expected:
        subprocess.run(["git", "-C", str(checkout), "fetch", "origin", expected], check=True)
        subprocess.run(["git", "-C", str(checkout), "checkout", "--detach", expected], check=True)
        actual = subprocess.check_output(
            ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True
        ).strip()
    if actual != expected:
        raise RuntimeError(f"OACIR source commit mismatch: expected {expected}, got {actual}")

    required = [
        checkout / "data_utils.py",
        checkout / "lavis/models/blip2_models/blip2_qformer_oacir_adafocal.py",
    ]
    missing = [rel(p) for p in required if not p.is_file()]
    if missing:
        raise FileNotFoundError("Incomplete OACIR source:\n  - " + "\n  - ".join(missing))
    return checkout


def import_official_adafocal(checkout: Path):
    checkout_str = str(checkout)
    if checkout_str not in sys.path:
        sys.path.insert(0, checkout_str)

    # The pinned checkout must win over any pip-installed LAVIS.
    for name in list(sys.modules):
        if name == "lavis" or name.startswith("lavis.") or name == "data_utils":
            del sys.modules[name]

    from lavis.models import load_model_and_preprocess
    from data_utils import targetpad_transform, transform_bbox_targetpad
    from lavis.models.blip2_models.blip2_qformer_oacir_adafocal import bbox_to_patch_mask

    return load_model_and_preprocess, targetpad_transform, transform_bbox_targetpad, bbox_to_patch_mask


def load_adafocal(
    cfg: dict[str, Any],
    checkout: Path,
    device: torch.device,
):
    (
        load_model_and_preprocess,
        targetpad_transform,
        transform_bbox_targetpad,
        bbox_to_patch_mask,
    ) = import_official_adafocal(checkout)

    model_cfg = cfg["adafocal"]
    checkpoint = resolve_path(str(model_cfg["checkpoint"]))
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"Missing AdaFocal checkpoint: {rel(checkpoint)}. Run download_checkpoint.py first."
        )
    actual_sha = sha256_file(checkpoint)
    expected_sha = str(model_cfg["checkpoint_sha256"])
    if actual_sha != expected_sha:
        raise RuntimeError(
            f"AdaFocal checkpoint checksum mismatch: expected {expected_sha}, got {actual_sha}"
        )

    model, _, txt_processors = load_model_and_preprocess(
        name=str(model_cfg["model_name"]),
        model_type=str(model_cfg["model_type"]),
        is_eval=False,
        device=device,
    )
    checkpoint_obj = torch.load(checkpoint, map_location="cpu")
    class_key = model.__class__.__name__
    if isinstance(checkpoint_obj, dict) and class_key in checkpoint_obj:
        state_dict = checkpoint_obj[class_key]
    elif isinstance(checkpoint_obj, dict) and "model" in checkpoint_obj:
        state_dict = checkpoint_obj["model"]
    elif isinstance(checkpoint_obj, dict):
        # Last-resort compatibility for a raw state_dict checkpoint.
        state_dict = checkpoint_obj
    else:
        raise TypeError("Unsupported AdaFocal checkpoint format")

    msg = model.load_state_dict(state_dict, strict=False)
    model.eval()

    # These should not be missing for the released scalar checkpoint.
    critical_prefixes = ("crm_module.", "contextual_probe_tokens", "text_proj.", "vision_proj.")
    critical_missing = [
        key for key in msg.missing_keys if key.startswith(critical_prefixes)
    ]
    if critical_missing:
        raise RuntimeError(
            "AdaFocal checkpoint is missing critical scalar-model weights: "
            + ", ".join(critical_missing[:12])
        )

    preprocess = targetpad_transform(
        float(model_cfg["target_ratio"]), int(model_cfg["input_size"])
    )
    return model, txt_processors, preprocess, transform_bbox_targetpad, bbox_to_patch_mask, msg


def query_compose_text(query: dict[str, Any], subject: dict[str, Any]) -> str:
    # Preserve the benchmark's established RELATIONAL behavior: every anchored
    # subject receives the complete relational instruction.
    if str(query.get("case", "")).strip() == "RELATIONAL":
        full = str(query.get("text") or "").strip()
        if full:
            return full

    modify = str(subject.get("modify_text") or "").strip()
    if modify:
        return modify
    relation = str(query.get("relation_text") or "").strip()
    if relation:
        return relation
    full = str(query.get("text") or "").strip()
    if full:
        return full
    raise ValueError(f"Query {query.get('query_id')} has no usable modification text")


@torch.no_grad()
def clip_image_features(
    model,
    preprocess,
    crops: list[Image.Image],
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    chunks: list[np.ndarray] = []
    for start in range(0, len(crops), batch_size):
        batch = torch.stack(
            [preprocess(im) for im in crops[start : start + batch_size]], dim=0
        ).to(device, non_blocking=True)
        feat = model.encode_image(batch).float()
        feat /= feat.norm(dim=1, keepdim=True).clamp_min(1e-12)
        chunks.append(feat.cpu().numpy())
    return np.concatenate(chunks, axis=0).astype(np.float32, copy=False)


@torch.no_grad()
def clip_text_features(
    model,
    texts: list[str],
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    chunks: list[np.ndarray] = []
    for start in range(0, len(texts), batch_size):
        tokens = clip.tokenize(texts[start : start + batch_size], truncate=True).to(device)
        feat = model.encode_text(tokens).float()
        feat /= feat.norm(dim=1, keepdim=True).clamp_min(1e-12)
        chunks.append(feat.cpu().numpy())
    return np.concatenate(chunks, axis=0).astype(np.float32, copy=False)


def hungarian_maximize(similarity: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    try:
        return linear_sum_assignment(similarity, maximize=True)
    except TypeError:
        return linear_sum_assignment(-similarity)


def select_subject_boxes(
    *,
    query: dict[str, Any],
    image: Image.Image,
    boxes: np.ndarray,
    selector_model,
    selector_preprocess,
    selector_device: torch.device,
    image_batch_size: int,
    text_batch_size: int,
) -> tuple[list[list[float]], list[str]] | None:
    subjects = query.get("subjects")
    if not isinstance(subjects, list) or not subjects:
        return None
    m = len(subjects)
    if len(boxes) < m:
        return None

    crops: list[Image.Image] = []
    valid_boxes: list[list[float]] = []
    for box in boxes:
        left = max(0, int(math.floor(float(box[0]))))
        top = max(0, int(math.floor(float(box[1]))))
        right = min(image.width, int(math.ceil(float(box[2]))))
        bottom = min(image.height, int(math.ceil(float(box[3]))))
        if right <= left or bottom <= top:
            continue
        crops.append(image.crop((left, top, right, bottom)))
        valid_boxes.append([float(box[0]), float(box[1]), float(box[2]), float(box[3])])

    if len(crops) < m:
        return None

    select_texts: list[str] = []
    compose_texts: list[str] = []
    for si, subject in enumerate(subjects):
        if not isinstance(subject, dict):
            raise TypeError(f"Query {query.get('query_id')}: subject {si} is not an object")
        select_text = str(subject.get("select_text") or "").strip()
        if not select_text:
            raise ValueError(f"Query {query.get('query_id')}: subject {si} has empty select_text")
        select_texts.append(select_text)
        compose_texts.append(query_compose_text(query, subject))

    person_feat = clip_image_features(
        selector_model,
        selector_preprocess,
        crops,
        image_batch_size,
        selector_device,
    )
    text_feat = clip_text_features(
        selector_model, select_texts, text_batch_size, selector_device
    )
    sim = text_feat @ person_feat.T

    if m == 1:
        assignment = np.asarray([int(np.argmax(sim[0]))], dtype=np.int64)
    else:
        rows, cols = hungarian_maximize(sim)
        if len(rows) != m:
            return None
        assignment = cols[np.argsort(rows)].astype(np.int64, copy=False)

    selected = [valid_boxes[int(i)] for i in assignment]
    return selected, compose_texts


def target_feature_fingerprint(
    *,
    cfg: dict[str, Any],
    config_path: Path,
    detection_cache: Path,
    checkpoint: Path,
    source_commit: str,
) -> dict[str, Any]:
    return {
        "schema": TARGET_FEATURE_SCHEMA,
        "adapter_version": ADAPTER_VERSION,
        "config_sha256": sha256_file(config_path),
        "detection_cache_sha256": sha256_file(detection_cache),
        "adafocal_checkpoint_sha256": sha256_file(checkpoint),
        "official_source_commit": source_commit,
        "target_candidate": str(cfg["adaptation"]["target_candidate"]),
        "transform": str(cfg["adafocal"]["transform"]),
        "target_ratio": float(cfg["adafocal"]["target_ratio"]),
        "input_size": int(cfg["adafocal"]["input_size"]),
    }


def load_cached_target_features(
    cache_path: Path,
    expected_meta: dict[str, Any],
    expected_count: int,
) -> np.ndarray | None:
    if not cache_path.is_file() or not meta_path(cache_path).is_file():
        return None
    if read_json(meta_path(cache_path)) != expected_meta:
        print(f"Ignoring stale AdaFocal target-feature cache: {rel(cache_path)}", flush=True)
        return None
    try:
        features = np.load(cache_path, mmap_mode="r", allow_pickle=False)
    except Exception as error:
        print(f"Ignoring invalid target-feature cache: {error}", flush=True)
        return None
    if features.ndim != 3 or features.shape[0] != expected_count:
        print(
            f"Ignoring incompatible target-feature cache shape={features.shape}; "
            f"expected first dim={expected_count}",
            flush=True,
        )
        return None
    if features.dtype.kind != "f":
        return None
    print(f"Using AdaFocal target-feature cache: {rel(cache_path)}", flush=True)
    return features


@torch.no_grad()
def compute_target_person_features(
    *,
    gallery: Sequence[dict[str, Any]],
    offsets: np.ndarray,
    boxes: np.ndarray,
    model,
    preprocess,
    batch_size: int,
    device: torch.device,
    cache_path: Path,
    cache_meta: dict[str, Any],
    np_dtype: np.dtype,
) -> np.ndarray:
    total_persons = int(offsets[-1])
    if total_persons <= 0:
        raise RuntimeError("Shared detector produced zero gallery persons")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temp = cache_path.with_name(cache_path.name + ".part")
    temp.unlink(missing_ok=True)

    mmap = None
    pending: list[torch.Tensor] = []
    write_cursor = 0
    token_count = None
    feature_dim = None

    def flush() -> None:
        nonlocal pending, write_cursor, mmap, token_count, feature_dim
        if not pending:
            return
        batch = torch.stack(pending, dim=0).to(device, non_blocking=True)
        target_features, image_embeds_frozen = model.extract_target_features(batch)
        target_features_cpu = target_features.detach().cpu()
        del target_features, image_embeds_frozen, batch
        if target_features_cpu.ndim != 3:
            raise RuntimeError(
                f"Unexpected AdaFocal target feature shape: {tuple(target_features_cpu.shape)}"
            )

        if mmap is None:
            token_count = int(target_features_cpu.shape[1])
            feature_dim = int(target_features_cpu.shape[2])
            mmap = np.lib.format.open_memmap(
                temp,
                mode="w+",
                dtype=np_dtype,
                shape=(total_persons, token_count, feature_dim),
            )

        count = int(target_features_cpu.shape[0])
        mmap[write_cursor : write_cursor + count] = (
            target_features_cpu.numpy().astype(np_dtype, copy=False)
        )
        write_cursor += count
        del target_features_cpu
        pending = []

    try:
        for gi, row in enumerate(
            progress_bar(
                gallery,
                desc="Encode detected gallery persons with AdaFocal target branch",
                total=len(gallery),
                unit="image",
            )
        ):
            start, end = int(offsets[gi]), int(offsets[gi + 1])
            if end <= start:
                continue
            with Image.open(image_path(row, gi)) as opened:
                image = opened.convert("RGB")
                for box in boxes[start:end]:
                    left = max(0, int(math.floor(float(box[0]))))
                    top = max(0, int(math.floor(float(box[1]))))
                    right = min(image.width, int(math.ceil(float(box[2]))))
                    bottom = min(image.height, int(math.ceil(float(box[3]))))
                    if right <= left or bottom <= top:
                        raise RuntimeError(
                            f"Invalid shared detector crop for gallery row {gi}: {box.tolist()}"
                        )
                    pending.append(preprocess(image.crop((left, top, right, bottom))))
                    if len(pending) >= batch_size:
                        flush()
        flush()

        if mmap is None or write_cursor != total_persons:
            raise RuntimeError(
                f"Encoded {write_cursor} target persons, expected {total_persons}"
            )
        mmap.flush()
        del mmap
        mmap = None
        os.replace(temp, cache_path)
        write_json(meta_path(cache_path), cache_meta)
    except Exception:
        if mmap is not None:
            try:
                del mmap
            except Exception:
                pass
        temp.unlink(missing_ok=True)
        raise

    return np.load(cache_path, mmap_mode="r", allow_pickle=False)


@torch.no_grad()
def encode_reference_image(model, image_tensor: torch.Tensor) -> torch.Tensor:
    image_tensor = image_tensor.to(model.device, non_blocking=True)
    with model.maybe_autocast():
        embeds = model.ln_vision(model.visual_encoder(image_tensor))
    return embeds.float()


@torch.no_grad()
def extract_anchored_query_features(
    *,
    model,
    reference_image_embeds: torch.Tensor,
    modification_texts: list[str],
    transformed_bboxes: list[list[int]],
    bbox_to_patch_mask,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact scalar AdaFocal query branch, returning fusion features before gallery matmul."""
    batch = len(modification_texts)
    if batch == 0 or len(transformed_bboxes) != batch:
        raise ValueError("Anchor/text batch mismatch")

    if reference_image_embeds.shape[0] == 1 and batch > 1:
        reference_image_embeds = reference_image_embeds.expand(batch, -1, -1)
    elif reference_image_embeds.shape[0] != batch:
        raise ValueError("Reference embedding batch must be 1 or match number of anchors")

    device = reference_image_embeds.device
    reference_image_atts = torch.ones(
        reference_image_embeds.size()[:-1], dtype=torch.long, device=device
    )

    query_tokens = model.query_tokens.expand(batch, -1, -1)
    query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long, device=device)

    text_tokens = model.tokenizer(
        modification_texts,
        padding="max_length",
        truncation=True,
        max_length=model.max_txt_len,
        return_tensors="pt",
    ).to(device)

    probe_tokens = model.contextual_probe_tokens.expand(batch, -1, -1)
    probe_atts = torch.ones(probe_tokens.size()[:-1], dtype=torch.long, device=device)
    pre_fusion_atts = torch.cat([probe_atts, text_tokens.attention_mask], dim=1)

    pre_fusion_output = model.Qformer.bert(
        text_tokens.input_ids,
        query_embeds=probe_tokens,
        attention_mask=pre_fusion_atts,
        encoder_hidden_states=reference_image_embeds,
        encoder_attention_mask=reference_image_atts,
        return_dict=True,
    )
    pre_fusion_features = pre_fusion_output.last_hidden_state[:, : model.num_probe_token, :]
    adaptive_scalar = model.crm_module(pre_fusion_features)

    patch_mask = bbox_to_patch_mask(
        transformed_bboxes,
        patch_size=model.patch_size,
        device=device,
    )
    attention_bias = (adaptive_scalar * patch_mask).unsqueeze(1).unsqueeze(1)

    fusion_atts = torch.cat([query_atts, text_tokens.attention_mask], dim=1)
    fusion_output = model.Qformer.bert(
        text_tokens.input_ids,
        query_embeds=query_tokens,
        attention_mask=fusion_atts,
        encoder_hidden_states=reference_image_embeds,
        encoder_attention_mask=reference_image_atts,
        return_dict=True,
        attention_bias=attention_bias,
    )
    fusion_features = torch.nn.functional.normalize(
        model.text_proj(
            fusion_output.last_hidden_state[:, model.num_query_token, :]
        ),
        dim=-1,
    )
    return fusion_features, adaptive_scalar


@torch.no_grad()
def score_query_anchors_against_persons(
    *,
    fusion_features: torch.Tensor,
    target_person_features: np.ndarray,
    device: torch.device,
    chunk_size: int,
) -> np.ndarray:
    m = int(fusion_features.shape[0])
    total_persons = int(target_person_features.shape[0])
    result = np.empty((m, total_persons), dtype=np.float32)
    fusion = fusion_features.float()

    for start in range(0, total_persons, chunk_size):
        end = min(start + chunk_size, total_persons)
        chunk_np = np.array(target_person_features[start:end], dtype=np.float32, copy=True)
        chunk = torch.from_numpy(chunk_np).to(device, non_blocking=True)
        # Official AdaFocal similarity: cosine(query fusion, each target query token),
        # followed by max over the target query-token dimension.
        token_sim = torch.einsum("md,ctd->mct", fusion, chunk)
        pair_sim = token_sim.max(dim=-1).values
        result[:, start:end] = pair_sim.cpu().numpy()
        del chunk, token_sim, pair_sim
    return result


def aggregate_setmatch(
    *,
    pair_scores: np.ndarray,
    offsets: np.ndarray,
    unmatched_score: float,
) -> np.ndarray:
    """Image-level SetMatch: max-sum assignment then strict minimum of assigned pairs."""
    counts = np.diff(offsets).astype(np.int64, copy=False)
    num_gallery = len(counts)
    m = int(pair_scores.shape[0])
    result = np.full(num_gallery, unmatched_score, dtype=np.float32)

    if m == 0:
        return result

    if m == 1:
        total_persons = int(offsets[-1])
        person_image_index = np.repeat(
            np.arange(num_gallery, dtype=np.int64), counts
        )
        if person_image_index.shape != (total_persons,):
            raise RuntimeError("Person-to-image index shape mismatch")
        image_scores = np.full(num_gallery, -np.inf, dtype=np.float32)
        np.maximum.at(image_scores, person_image_index, pair_scores[0])
        nonempty = counts > 0
        result[nonempty] = image_scores[nonempty]
        return result

    eligible = np.flatnonzero(counts >= m)

    # Fast exact specialization for the common two-person MULTI case.
    if m == 2:
        for gi in eligible:
            start, end = int(offsets[gi]), int(offsets[gi + 1])
            s0 = pair_scores[0, start:end]
            s1 = pair_scores[1, start:end]
            i0 = int(np.argmax(s0))
            i1 = int(np.argmax(s1))
            if i0 != i1:
                assigned0, assigned1 = float(s0[i0]), float(s1[i1])
            else:
                if len(s0) < 2:
                    continue
                second0 = int(np.argpartition(s0, -2)[-2])
                second1 = int(np.argpartition(s1, -2)[-2])
                option_a = float(s0[i0] + s1[second1])
                option_b = float(s0[second0] + s1[i1])
                if option_a >= option_b:
                    assigned0, assigned1 = float(s0[i0]), float(s1[second1])
                else:
                    assigned0, assigned1 = float(s0[second0]), float(s1[i1])
            result[gi] = min(assigned0, assigned1)
        return result

    for gi in eligible:
        start, end = int(offsets[gi]), int(offsets[gi + 1])
        similarity = pair_scores[:, start:end]
        rows, cols = hungarian_maximize(similarity)
        if len(rows) != m:
            raise RuntimeError(
                f"Hungarian returned {len(rows)} assignments for query set size {m}"
            )
        result[gi] = float(np.min(similarity[rows, cols]))
    return result


def validate_scores(scores: np.ndarray, expected_shape: tuple[int, int]) -> None:
    if scores.shape != expected_shape:
        raise ValueError(f"scores shape={scores.shape}, expected={expected_shape}")
    for start in range(0, len(scores), 256):
        if not np.isfinite(np.asarray(scores[start : start + 256])).all():
            raise ValueError("scores contains NaN/Inf")


def main() -> None:
    tracker = PhaseTracker(METHOD_ID, total=8)

    with tracker.phase("Load config and manifests"):
        parser = argparse.ArgumentParser()
        parser.add_argument("--config", default=str(DEFAULT_CONFIG))
        args = parser.parse_args()
        config_path = resolve_path(args.config)
        cfg = load_yaml(config_path)

        gallery_manifest = resolve_path(str(cfg["data"]["gallery_manifest"]))
        query_manifest = resolve_path(str(cfg["data"]["query_manifest"]))
        gallery = load_jsonl(gallery_manifest)
        queries = load_jsonl(query_manifest)
        gallery_index = build_gallery_index(gallery)
        query_indices = query_gallery_indices(queries, gallery_index)
        tracker.log(f"gallery={len(gallery):,} queries={len(queries):,}")

    with tracker.phase("Prepare shared predicted person detections"):
        s5_method_dir = resolve_path(str(cfg["shared_protocol"]["method_dir"]))
        s5_config_path = resolve_path(str(cfg["shared_protocol"]["config"]))
        s5_cfg = load_yaml(s5_config_path)
        s5 = load_module(s5_method_dir, "cpr_p9_source_s5")
        detector_device = s5.device_from(str(s5_cfg["runtime"]["device"]))

        gd_checkout = s5.ensure_clean_pinned_source(
            s5_cfg["source"]["groundingdino"], "Grounding DINO"
        )
        detector_config = s5.require_file(
            gd_checkout / str(s5_cfg["detector"]["config"]), "Grounding DINO config"
        )
        detector_checkpoint = s5.require_file(
            resolve_path(str(s5_cfg["detector"]["checkpoint"])),
            "Grounding DINO checkpoint",
        )
        s5.configure_groundingdino_offline(s5_cfg)
        attention_backend = s5.configure_groundingdino_source(gd_checkout)

        detection_cache = resolve_path(str(cfg["cache"]["detections"]))
        det_meta = s5.detection_fingerprint(
            cfg=s5_cfg,
            config_path=s5_config_path,
            gallery_manifest=gallery_manifest,
            detector_config=detector_config,
            detector_checkpoint=detector_checkpoint,
            attention_backend=attention_backend,
        )
        loaded = s5.load_detection_cache(detection_cache, det_meta, len(gallery))
        if loaded is None:
            offsets, boxes, confidence = s5.compute_detections(
                cfg=s5_cfg,
                gallery=gallery,
                detector_config=detector_config,
                detector_checkpoint=detector_checkpoint,
                device=detector_device,
            )
            s5.save_detection_cache(
                detection_cache, det_meta, offsets, boxes, confidence
            )
        else:
            offsets, boxes, confidence = loaded
        del confidence
        tracker.log(f"predicted_persons={int(offsets[-1]):,}")

    with tracker.phase("Load selector and official AdaFocal"):
        device = device_from(str(cfg["runtime"]["device"]))

        selector_checkpoint = resolve_path(str(cfg["selector"]["checkpoint"]))
        if not selector_checkpoint.is_file():
            raise FileNotFoundError(
                f"Missing selector checkpoint: {rel(selector_checkpoint)}"
            )
        selector_sha = sha256_file(selector_checkpoint)
        if selector_sha != str(cfg["selector"]["checkpoint_sha256"]):
            raise RuntimeError("CLIP ViT-B/32 selector checkpoint checksum mismatch")
        selector_model, selector_preprocess = clip.load(
            str(selector_checkpoint), device=device, jit=False
        )
        selector_model.eval()
        if device.type != "cuda":
            selector_model.float()

        oacir_checkout = ensure_clean_pinned_source(cfg)
        (
            adafocal,
            txt_processors,
            adafocal_preprocess,
            transform_bbox_targetpad,
            bbox_to_patch_mask,
            load_msg,
        ) = load_adafocal(cfg, oacir_checkout, device)
        tracker.log(
            f"AdaFocal missing_keys={len(load_msg.missing_keys)} "
            f"unexpected_keys={len(load_msg.unexpected_keys)}"
        )

    with tracker.phase("Prepare AdaFocal gallery-person target features"):
        checkpoint = resolve_path(str(cfg["adafocal"]["checkpoint"]))
        cache_path = resolve_path(str(cfg["cache"]["target_person_features"]))
        cache_meta = target_feature_fingerprint(
            cfg=cfg,
            config_path=config_path,
            detection_cache=detection_cache,
            checkpoint=checkpoint,
            source_commit=str(cfg["official_source"]["commit"]),
        )
        target_person_features = load_cached_target_features(
            cache_path, cache_meta, int(offsets[-1])
        )
        if target_person_features is None:
            dtype_name = str(cfg["runtime"]["target_feature_cache_dtype"])
            if dtype_name not in {"float16", "float32"}:
                raise ValueError(
                    "runtime.target_feature_cache_dtype must be float16 or float32"
                )
            np_dtype = np.float16 if dtype_name == "float16" else np.float32
            target_person_features = compute_target_person_features(
                gallery=gallery,
                offsets=offsets,
                boxes=boxes,
                model=adafocal,
                preprocess=adafocal_preprocess,
                batch_size=int(cfg["runtime"]["gallery_person_batch_size"]),
                device=device,
                cache_path=cache_path,
                cache_meta=cache_meta,
                np_dtype=np_dtype,
            )
        tracker.log(f"target_feature_shape={tuple(target_person_features.shape)}")

    with tracker.phase("Score anchored CPR queries"):
        output_dir = resolve_path(str(cfg["output"]["dir"]))
        output_dir.mkdir(parents=True, exist_ok=True)
        scores_path = output_dir / "scores.npy"
        temp_path = output_dir / "scores.npy.part"
        temp_path.unlink(missing_ok=True)
        score_mmap = np.lib.format.open_memmap(
            temp_path,
            mode="w+",
            dtype=np.float32,
            shape=(len(queries), len(gallery)),
        )

        invalid_queries: list[dict[str, Any]] = []
        activation_scalars: list[dict[str, Any]] = []
        unmatched_score = float(cfg["setmatch"]["unmatched_score"])
        target_ratio = float(cfg["adafocal"]["target_ratio"])
        input_size = int(cfg["adafocal"]["input_size"])

        for qi, query in enumerate(
            progress_bar(
                queries,
                desc="AdaFocal predicted-anchor + SetMatch",
                total=len(queries),
                unit="query",
            )
        ):
            source_gi = int(query_indices[qi])
            start, end = int(offsets[source_gi]), int(offsets[source_gi + 1])
            source_boxes = np.asarray(boxes[start:end], dtype=np.float32)

            try:
                with Image.open(image_path(gallery[source_gi], source_gi)) as opened:
                    image = opened.convert("RGB")
                    selected = select_subject_boxes(
                        query=query,
                        image=image,
                        boxes=source_boxes,
                        selector_model=selector_model,
                        selector_preprocess=selector_preprocess,
                        selector_device=device,
                        image_batch_size=int(cfg["runtime"]["selector_image_batch_size"]),
                        text_batch_size=int(cfg["runtime"]["selector_text_batch_size"]),
                    )
                    if selected is None:
                        raise RuntimeError("could_not_localize_all_subjects")
                    selected_boxes, modification_texts = selected

                    transformed_boxes: list[list[int]] = []
                    for box in selected_boxes:
                        transformed = transform_bbox_targetpad(
                            box,
                            image.size,
                            target_ratio=target_ratio,
                            final_size=input_size,
                        )
                        if transformed is None:
                            raise RuntimeError("predicted_anchor_removed_by_targetpad_crop")
                        transformed_boxes.append(transformed)

                    reference_tensor = adafocal_preprocess(image).unsqueeze(0)

                reference_embeds = encode_reference_image(adafocal, reference_tensor)
                processed_texts = [
                    txt_processors["eval"](text) for text in modification_texts
                ]
                fusion_features, scalar = extract_anchored_query_features(
                    model=adafocal,
                    reference_image_embeds=reference_embeds,
                    modification_texts=processed_texts,
                    transformed_bboxes=transformed_boxes,
                    bbox_to_patch_mask=bbox_to_patch_mask,
                )
                pair_scores = score_query_anchors_against_persons(
                    fusion_features=fusion_features,
                    target_person_features=target_person_features,
                    device=device,
                    chunk_size=int(cfg["runtime"]["candidate_feature_chunk_size"]),
                )
                score_mmap[qi] = aggregate_setmatch(
                    pair_scores=pair_scores,
                    offsets=offsets,
                    unmatched_score=unmatched_score,
                )
                activation_scalars.append(
                    {
                        "query_id": query.get("query_id"),
                        "beta": [float(x) for x in scalar.detach().cpu().flatten().tolist()],
                    }
                )
                del reference_embeds, fusion_features, pair_scores, scalar
            except Exception as error:
                # Deterministic benchmark behavior for detector/localizer failures.
                # Model/runtime failures must remain fatal rather than silently degrading.
                message = str(error)
                expected_localization_failures = {
                    "could_not_localize_all_subjects",
                    "predicted_anchor_removed_by_targetpad_crop",
                }
                if message not in expected_localization_failures:
                    raise
                score_mmap[qi].fill(unmatched_score)
                invalid_queries.append(
                    {"query_id": query.get("query_id"), "reason": message}
                )

        score_mmap.flush()
        del score_mmap
        os.replace(temp_path, scores_path)

    with tracker.phase("Validate score matrix"):
        scores = np.load(scores_path, mmap_mode="r", allow_pickle=False)
        validate_scores(scores, (len(queries), len(gallery)))
        tracker.log(
            f"invalid_localization_queries={len(invalid_queries)} "
            f"scores={rel(scores_path)}"
        )

    with tracker.phase("Write run metadata"):
        run_payload = {
            "method": cfg["method"],
            "display_name": cfg["display_name"],
            "group": cfg["group"],
            "cpr_supervision": "No",
            "adapter_version": ADAPTER_VERSION,
            "official_source": {
                "repository": cfg["official_source"]["repository"],
                "commit": cfg["official_source"]["commit"],
            },
            "adafocal": {
                "model_name": cfg["adafocal"]["model_name"],
                "model_type": cfg["adafocal"]["model_type"],
                "checkpoint": rel(resolve_path(str(cfg["adafocal"]["checkpoint"]))),
                "checkpoint_sha256": cfg["adafocal"]["checkpoint_sha256"],
                "variant": "scalar_beta_default",
                "transform": cfg["adafocal"]["transform"],
                "target_ratio": cfg["adafocal"]["target_ratio"],
            },
            "query_anchor": {
                "source": "shared Grounding DINO predicted person boxes",
                "selector": cfg["selector"]["name"],
                "assignment": cfg["selector"]["assignment"],
                "gt_box_used": False,
            },
            "gallery_candidates": "shared-detector person crops encoded by official AdaFocal target branch",
            "setmatch": cfg["setmatch"],
            "relational_text_behavior": cfg["adaptation"]["relational_text_behavior"],
            "groundingdino_attention_backend": attention_backend,
            "num_queries": len(queries),
            "num_gallery": len(gallery),
            "num_detected_persons": int(offsets[-1]),
            "invalid_localization_queries": invalid_queries,
            "activation_scalars": activation_scalars,
            "scores": rel(scores_path),
            "higher_is_better": True,
            "config": rel(config_path),
        }
        write_json(output_dir / "run.json", run_payload)
        tracker.log(f"run={rel(output_dir / 'run.json')}")

    with tracker.phase("Finish"):
        tracker.log("P9 main-table path uses predicted boxes only; no GT anchor is read.")

    tracker.finish()


if __name__ == "__main__":
    main()
