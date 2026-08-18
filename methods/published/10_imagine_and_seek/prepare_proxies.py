#!/usr/bin/env python3
"""Prepare imagined proxies for the P10 official-source CPR adapter."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark_progress import PhaseTracker, progress_bar  # noqa: E402

DEFAULT_CONFIG = Path(__file__).resolve().parent / "config.yaml"
METHOD_ID = "imagine_seek_proxy"
CAPTION_SCHEMA = 2
LAYOUT_SCHEMA = 2
PROMPT_VERSION = "2026-08-18-v2-official-source-cpr"


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
                raise TypeError(f"{path}:{lineno}: expected JSON object")
            rows.append(row)
    return rows


def write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".part")
    with temp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(temp, path)


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


def stable_hash(payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def read_marker(cfg: dict[str, Any]) -> dict[str, Any]:
    path = resolve_path(str(cfg["migc"]["prepared_marker"]))
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing P10 prepared marker: {rel(path)}. Run download_checkpoint.py first."
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError("Invalid P10 prepared marker")
    return data


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


def gallery_image_path(row: dict[str, Any], index: int) -> Path:
    value = row.get("path")
    if not isinstance(value, str) or not value.strip():
        raise KeyError(f"Gallery row {index} has no usable path")
    path = resolve_path(value)
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def query_image_paths(queries: Sequence[dict[str, Any]], gallery: Sequence[dict[str, Any]]) -> list[Path]:
    index = build_gallery_index(gallery)
    paths: list[Path] = []
    for qi, query in enumerate(queries):
        image_id = query.get("image_id")
        if image_id not in index:
            raise ValueError(f"Query row {qi}: image_id {image_id!r} missing from gallery")
        gi = index[image_id]
        paths.append(gallery_image_path(gallery[gi], gi))
    return paths


def device_from(cfg: dict[str, Any]) -> torch.device:
    device = torch.device(str(cfg["runtime"]["device"]))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Imagine-and-Seek proxy generation requires CUDA")
    return device


def torch_dtype(name: str, device: torch.device) -> torch.dtype:
    if device.type != "cuda":
        return torch.float32
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    value = name.lower()
    if value not in mapping:
        raise ValueError(f"Unsupported dtype: {name}")
    return mapping[value]


def unload(*objects: Any) -> None:
    for obj in objects:
        del obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def cached_rows_valid(path: Path, expected_count: int, fingerprint: str, queries: Sequence[dict[str, Any]]) -> bool:
    if not path.is_file():
        return False
    try:
        rows = load_jsonl(path)
    except Exception:
        return False
    if len(rows) != expected_count:
        return False
    for qi, row in enumerate(rows):
        if row.get("query_index") != qi:
            return False
        if row.get("image_id") != queries[qi].get("image_id"):
            return False
        if row.get("fingerprint") != fingerprint:
            return False
    return True


def caption_fingerprint(cfg: dict[str, Any], query_manifest: Path, marker: dict[str, Any]) -> str:
    foundation = marker.get("foundation_models", {}).get("captioner", {})
    return stable_hash(
        {
            "schema": CAPTION_SCHEMA,
            "query_manifest_sha256": sha256_file(query_manifest),
            "captioner": cfg["captioner"],
            "resolved_revision": foundation.get("resolved_revision"),
        }
    )


def layout_fingerprint(cfg: dict[str, Any], query_manifest: Path, captions_path: Path, marker: dict[str, Any]) -> str:
    foundation = marker.get("foundation_models", {}).get("layout_llm", {})
    return stable_hash(
        {
            "schema": LAYOUT_SCHEMA,
            "prompt_version": PROMPT_VERSION,
            "query_manifest_sha256": sha256_file(query_manifest),
            "captions_sha256": sha256_file(captions_path),
            "layout_llm": cfg["layout_llm"],
            "resolved_revision": foundation.get("resolved_revision"),
            "author_commit": cfg["author_source"]["commit"],
        }
    )


@torch.no_grad()
def generate_captions(
    *, cfg: dict[str, Any], queries: Sequence[dict[str, Any]], image_paths: Sequence[Path],
    query_manifest: Path, marker: dict[str, Any], force: bool,
) -> list[dict[str, Any]]:
    output = resolve_path(str(cfg["cache"]["captions"]))
    fingerprint = caption_fingerprint(cfg, query_manifest, marker)
    if not force and cached_rows_valid(output, len(queries), fingerprint, queries):
        print(f"[skip] valid BLIP2 caption cache: {rel(output)}", flush=True)
        return load_jsonl(output)

    from transformers import Blip2ForConditionalGeneration, Blip2Processor

    c = cfg["captioner"]
    snapshot = resolve_path(str(c["local_snapshot"]))
    if not snapshot.is_dir():
        raise FileNotFoundError(f"Missing BLIP2 snapshot: {rel(snapshot)}")
    device = device_from(cfg)
    dtype = torch_dtype(str(cfg["runtime"]["caption_dtype"]), device)
    processor = Blip2Processor.from_pretrained(str(snapshot), local_files_only=True)
    model = Blip2ForConditionalGeneration.from_pretrained(
        str(snapshot), local_files_only=True, torch_dtype=dtype, low_cpu_mem_usage=True
    ).to(device).eval()

    count = int(c["captions_per_query"])
    rows: list[dict[str, Any]] = []
    for qi in progress_bar(range(len(queries)), desc="IP-CIR BLIP2 captions", total=len(queries), unit="query"):
        with Image.open(image_paths[qi]) as image:
            inputs = processor(images=image.convert("RGB"), return_tensors="pt")
        prepared: dict[str, torch.Tensor] = {}
        for key, value in inputs.items():
            if not isinstance(value, torch.Tensor):
                continue
            prepared[key] = value.to(device=device, dtype=dtype if value.dtype.is_floating_point else value.dtype)
        torch.manual_seed(10_000 + qi)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(10_000 + qi)
        generated = model.generate(
            **prepared,
            max_new_tokens=int(c["max_new_tokens"]),
            do_sample=bool(c["do_sample"]),
            top_p=float(c["top_p"]),
            temperature=float(c["temperature"]),
            num_return_sequences=count,
        )
        captions = [x.strip() for x in processor.batch_decode(generated, skip_special_tokens=True) if x.strip()]
        if not captions:
            raise RuntimeError(f"BLIP2 produced no usable caption for query {qi}")
        while len(captions) < count:
            captions.append(captions[-1])
        rows.append(
            {
                "schema": CAPTION_SCHEMA,
                "query_index": qi,
                "image_id": queries[qi].get("image_id"),
                "captions": captions[:count],
                "fingerprint": fingerprint,
            }
        )
    write_jsonl(output, rows)
    unload(model, processor)
    return rows


def layout_prompt(captions: Sequence[str], modification: str, target_caption_count: int) -> str:
    caption_text = "\n".join(f"- {caption}" for caption in captions)
    return f"""You are the layout planner for Imagine-and-Seek (IP-CIR).

Reference full-scene captions:
{caption_text}

Composed retrieval modification:
{modification}

Infer one plausible target full-scene image after applying the modification. Preserve the identity of the relevant person or people from the reference image unless the instruction explicitly replaces/removes them. Preserve unaffected people and relationships. The output will drive MIGC + ELITE generation.

Return ONLY one JSON object with exactly this structure:
{{
  "scene_prompt": "realistic global target-scene description",
  "layout": [
    {{
      "label": "short class label such as person, backpack, street",
      "cate": "same short class label",
      "desc": "visual description after applying the modification",
      "bbox": [x0, y0, x1, y1],
      "ref": "image or text",
      "is_scene": false
    }}
  ],
  "target_captions": ["caption 1", "caption 2"]
}}

Rules:
- bbox coordinates are normalized floats in [0,1], with x0 < x1 and y0 < y1.
- include ONE global scene item with bbox [0,0,1,1], is_scene=true and ref="text".
- include every important person as a separate non-scene item.
- set ref="image" for a person whose identity should come from the reference image, EVEN IF clothes/pose/action are modified.
- set ref="text" for newly introduced objects/background/context.
- for MULTI-person or relational instructions, keep all required people and encode their relation spatially.
- use at most 10 non-scene items.
- target_captions must contain exactly {target_caption_count} concise descriptions of the desired target image.
- no Markdown, comments, code fences, or explanation.
"""


def extract_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start < 0 or end <= start:
        raise ValueError(f"No JSON object in response: {cleaned[:300]!r}")
    data = json.loads(cleaned[start : end + 1])
    if not isinstance(data, dict):
        raise TypeError("Layout response must be a JSON object")
    return data


def clamp_bbox(value: Any) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError(f"Invalid bbox: {value!r}")
    coords = [float(x) for x in value]
    if not all(math.isfinite(x) for x in coords):
        raise ValueError(f"Non-finite bbox: {value!r}")
    x0, y0, x1, y1 = [min(1.0, max(0.0, x)) for x in coords]
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"Degenerate bbox: {value!r}")
    return [x0, y0, x1, y1]


def sanitize_layout(raw: dict[str, Any], target_caption_count: int) -> dict[str, Any]:
    scene_prompt = str(raw.get("scene_prompt", "")).strip()
    if not scene_prompt:
        raise ValueError("Layout missing scene_prompt")
    items_raw = raw.get("layout")
    if not isinstance(items_raw, list):
        raise ValueError("Layout must contain a layout list")

    items: list[dict[str, Any]] = []
    for value in items_raw[:11]:
        if not isinstance(value, dict):
            continue
        label = str(value.get("label") or value.get("cate") or "").strip().lower()
        desc = str(value.get("desc") or value.get("description") or "").strip()
        if not label or not desc:
            continue
        is_scene = bool(value.get("is_scene", False))
        bbox = [0.0, 0.0, 1.0, 1.0] if is_scene else clamp_bbox(value.get("bbox"))
        ref = str(value.get("ref", "text")).strip().lower()
        if ref not in {"image", "text"}:
            ref = "text"
        items.append(
            {
                "label": label,
                "cate": label,
                "desc": desc,
                "bbox": bbox,
                "ref": "text" if is_scene else ref,
                "is_scene": is_scene,
            }
        )

    if not any(item["is_scene"] for item in items):
        items.insert(
            0,
            {
                "label": "scene",
                "cate": "scene",
                "desc": scene_prompt,
                "bbox": [0.0, 0.0, 1.0, 1.0],
                "ref": "text",
                "is_scene": True,
            },
        )

    non_scene = [item for item in items if not item["is_scene"]]
    if not non_scene:
        non_scene = [
            {
                "label": "person",
                "cate": "person",
                "desc": "the relevant person from the reference image with the requested modification",
                "bbox": [0.15, 0.05, 0.85, 0.95],
                "ref": "image",
                "is_scene": False,
            }
        ]
        items.extend(non_scene)

    # Direct CPR full-scene adapter should preserve identity visually. If the LLM forgot to
    # mark any image reference, use the first person-like instance as the ELITE anchor.
    if not any(item["ref"] == "image" and not item["is_scene"] for item in items):
        candidate = next(
            (item for item in items if not item["is_scene"] and "person" in item["label"]),
            next(item for item in items if not item["is_scene"]),
        )
        candidate["ref"] = "image"

    captions_raw = raw.get("target_captions")
    captions = [] if not isinstance(captions_raw, list) else [str(x).strip() for x in captions_raw if str(x).strip()]
    if not captions:
        captions = [scene_prompt]
    while len(captions) < target_caption_count:
        captions.append(captions[-1])

    return {
        "scene_prompt": scene_prompt,
        "layout": items,
        "target_captions": captions[:target_caption_count],
    }


@torch.no_grad()
def generate_layouts(
    *, cfg: dict[str, Any], queries: Sequence[dict[str, Any]], captions: Sequence[dict[str, Any]],
    query_manifest: Path, marker: dict[str, Any], force: bool,
) -> list[dict[str, Any]]:
    captions_path = resolve_path(str(cfg["cache"]["captions"]))
    output = resolve_path(str(cfg["cache"]["layouts"]))
    fingerprint = layout_fingerprint(cfg, query_manifest, captions_path, marker)
    if not force and cached_rows_valid(output, len(queries), fingerprint, queries):
        print(f"[skip] valid Qwen layout cache: {rel(output)}", flush=True)
        return load_jsonl(output)

    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    c = cfg["layout_llm"]
    snapshot = resolve_path(str(c["local_snapshot"]))
    if not snapshot.is_dir():
        raise FileNotFoundError(f"Missing Qwen snapshot: {rel(snapshot)}")
    device = device_from(cfg)
    dtype = torch_dtype(str(cfg["runtime"]["llm_dtype"]), device)
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True, trust_remote_code=True)
    kwargs: dict[str, Any] = {
        "local_files_only": True,
        "torch_dtype": dtype,
        "low_cpu_mem_usage": True,
        "trust_remote_code": True,
    }
    if bool(c.get("load_in_4bit", False)):
        if device.type != "cuda":
            raise RuntimeError("4-bit Qwen loading requires CUDA")
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(str(snapshot), **kwargs).eval()
    if "device_map" not in kwargs:
        model = model.to(device)

    target_caption_count = int(c["target_captions_per_query"])
    rows: list[dict[str, Any]] = []
    for qi in progress_bar(range(len(queries)), desc="IP-CIR Qwen layouts", total=len(queries), unit="query"):
        modification = queries[qi].get("text")
        if not isinstance(modification, str) or not modification.strip():
            raise KeyError(f"Query row {qi} has no usable text")
        user_prompt = layout_prompt(captions[qi]["captions"], modification.strip(), target_caption_count)
        messages = [
            {"role": "system", "content": "You are a precise image-layout planner. Return strict JSON only."},
            {"role": "user", "content": user_prompt},
        ]
        layout: dict[str, Any] | None = None
        last_error: Exception | None = None
        last_text = ""
        for attempt in range(3):
            rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            tokens = tokenizer(rendered, return_tensors="pt")
            model_device = next(model.parameters()).device
            tokens = {key: value.to(model_device) for key, value in tokens.items()}
            gen_kwargs: dict[str, Any] = {
                "max_new_tokens": int(c["max_new_tokens"]),
                "do_sample": bool(c["do_sample"]),
                "pad_token_id": tokenizer.eos_token_id,
            }
            if bool(c["do_sample"]):
                gen_kwargs["temperature"] = max(float(c.get("temperature", 0.7)), 1e-5)
            output_ids = model.generate(**tokens, **gen_kwargs)
            completion = output_ids[0, tokens["input_ids"].shape[1] :]
            last_text = tokenizer.decode(completion, skip_special_tokens=True)
            try:
                layout = sanitize_layout(extract_json_object(last_text), target_caption_count)
                break
            except Exception as error:
                last_error = error
                messages.extend(
                    [
                        {"role": "assistant", "content": last_text},
                        {
                            "role": "user",
                            "content": "Repair the JSON/layout. Return ONLY a valid object matching the requested schema. Error: " + str(error),
                        },
                    ]
                )
        if layout is None:
            raise RuntimeError(
                f"Qwen failed to produce a valid layout for query {qi}: {last_error}; last={last_text[:500]!r}"
            )
        rows.append(
            {
                "schema": LAYOUT_SCHEMA,
                "query_index": qi,
                "image_id": queries[qi].get("image_id"),
                "modification": modification.strip(),
                **layout,
                "fingerprint": fingerprint,
            }
        )
    write_jsonl(output, rows)
    unload(model, tokenizer)
    return rows


def build_jobs(
    *, cfg: dict[str, Any], queries: Sequence[dict[str, Any]], image_paths: Sequence[Path],
    captions: Sequence[dict[str, Any]], layouts: Sequence[dict[str, Any]],
) -> Path:
    path = resolve_path(str(cfg["proxy"]["jobs"]))
    rows: list[dict[str, Any]] = []
    for qi in range(len(queries)):
        rows.append(
            {
                "query_index": qi,
                "image_id": queries[qi].get("image_id"),
                "query_image": rel(image_paths[qi]),
                "original_captions": list(captions[qi]["captions"]),
                "target_captions": list(layouts[qi]["target_captions"]),
                "scene_prompt": layouts[qi]["scene_prompt"],
                "layout": layouts[qi]["layout"],
            }
        )
    write_jsonl(path, rows)
    return path


def validate_proxy_manifest(cfg: dict[str, Any], queries: Sequence[dict[str, Any]], path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing proxy manifest: {rel(path)}")
    rows = load_jsonl(path)
    if len(rows) != len(queries):
        raise ValueError(f"Proxy manifest has {len(rows)} rows, expected {len(queries)}")
    expected = int(cfg["proxy"]["count_per_query"])
    for qi, row in enumerate(rows):
        if row.get("query_index") != qi or row.get("image_id") != queries[qi].get("image_id"):
            raise ValueError(f"Proxy manifest row {qi} is not aligned")
        for key in ("original_captions", "target_captions"):
            if not isinstance(row.get(key), list) or not row[key]:
                raise ValueError(f"Proxy manifest row {qi} missing {key}")
        paths = row.get("proxy_paths")
        if not isinstance(paths, list) or len(paths) != expected:
            raise ValueError(f"Proxy manifest row {qi}: expected {expected} proxy paths")
        for value in paths:
            if not isinstance(value, str) or not resolve_path(value).is_file():
                raise FileNotFoundError(f"Proxy manifest row {qi}: missing {value!r}")
    return rows


def generate_proxies(
    *, cfg: dict[str, Any], config_path: Path, queries: Sequence[dict[str, Any]], image_paths: Sequence[Path],
    captions: Sequence[dict[str, Any]], layouts: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    build_jobs(cfg=cfg, queries=queries, image_paths=image_paths, captions=captions, layouts=layouts)
    worker = Path(__file__).resolve().parent / "official_proxy_worker.py"
    subprocess.run([sys.executable, str(worker), "--config", str(config_path)], cwd=str(ROOT), check=True)
    return validate_proxy_manifest(cfg, queries, resolve_path(str(cfg["proxy"]["manifest"])))


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare P10 Imagine-and-Seek imagined proxies")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--stage", choices=["captions", "layouts", "generate", "all"], default="all")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    config_path = resolve_path(args.config)
    cfg = load_yaml(config_path)
    query_manifest = resolve_path(str(cfg["data"]["query_manifest"]))
    gallery_manifest = resolve_path(str(cfg["data"]["gallery_manifest"]))
    queries = load_jsonl(query_manifest)
    gallery = load_jsonl(gallery_manifest)

    if str(cfg["proxy"]["mode"]) == "precomputed":
        validate_proxy_manifest(cfg, queries, resolve_path(str(cfg["proxy"]["manifest"])))
        print("[ok] precomputed proxy manifest", flush=True)
        return

    marker = read_marker(cfg)
    if marker.get("author_source", {}).get("commit") != str(cfg["author_source"]["commit"]):
        raise RuntimeError("Prepared marker does not match configured author-source commit")
    image_paths = query_image_paths(queries, gallery)

    stages = [args.stage] if args.stage != "all" else ["captions", "layouts", "generate"]
    tracker = PhaseTracker(METHOD_ID, total=len(stages))
    captions: list[dict[str, Any]] | None = None
    layouts: list[dict[str, Any]] | None = None

    for stage in stages:
        if stage == "captions":
            with tracker.phase("Generate BLIP2 reference-scene captions"):
                captions = generate_captions(
                    cfg=cfg,
                    queries=queries,
                    image_paths=image_paths,
                    query_manifest=query_manifest,
                    marker=marker,
                    force=args.force,
                )
        elif stage == "layouts":
            with tracker.phase("Infer IP-CIR target layouts with Qwen"):
                if captions is None:
                    captions_path = resolve_path(str(cfg["cache"]["captions"]))
                    if not captions_path.is_file():
                        raise FileNotFoundError("Caption cache missing; run captions first")
                    captions = load_jsonl(captions_path)
                layouts = generate_layouts(
                    cfg=cfg,
                    queries=queries,
                    captions=captions,
                    query_manifest=query_manifest,
                    marker=marker,
                    force=args.force,
                )
        elif stage == "generate":
            with tracker.phase("Generate five released-MIGC+ELITE proxies per query"):
                if captions is None:
                    captions = load_jsonl(resolve_path(str(cfg["cache"]["captions"])))
                if layouts is None:
                    layouts = load_jsonl(resolve_path(str(cfg["cache"]["layouts"])))
                generate_proxies(
                    cfg=cfg,
                    config_path=config_path,
                    queries=queries,
                    image_paths=image_paths,
                    captions=captions,
                    layouts=layouts,
                )
        else:
            raise AssertionError(stage)

    tracker.finish()


if __name__ == "__main__":
    main()
