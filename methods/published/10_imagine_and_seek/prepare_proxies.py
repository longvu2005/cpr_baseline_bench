#!/usr/bin/env python3
"""Generate/validate imagined proxies for the P10 IP-CIR reproduction."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import re
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
CAPTION_SCHEMA = 1
LAYOUT_SCHEMA = 1
PROXY_SCHEMA = 1
PROMPT_VERSION = "2026-08-18-v1-cpr-fullscene"


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


def read_prepared_marker(cfg: dict[str, Any]) -> dict[str, Any]:
    marker = resolve_path(str(cfg["migc"]["prepared_marker"]))
    if not marker.is_file():
        raise FileNotFoundError(
            f"Missing P10 prepared marker: {rel(marker)}. Run download_checkpoint.py first."
        )
    data = json.loads(marker.read_text(encoding="utf-8"))
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
    gallery_index = build_gallery_index(gallery)
    paths: list[Path] = []
    for qi, query in enumerate(queries):
        image_id = query.get("image_id")
        if image_id not in gallery_index:
            raise ValueError(f"Query row {qi}: image_id {image_id!r} missing from gallery")
        gi = gallery_index[image_id]
        paths.append(gallery_image_path(gallery[gi], gi))
    return paths


def torch_dtype(name: str) -> torch.dtype:
    value = name.lower()
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if value not in mapping:
        raise ValueError(f"Unsupported dtype: {name}")
    return mapping[value]


def device_from(cfg: dict[str, Any]) -> torch.device:
    device = torch.device(str(cfg["runtime"]["device"]))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Imagine-and-Seek proxy generation requires CUDA but CUDA is unavailable")
    return device


def unload(*objects: Any) -> None:
    for obj in objects:
        try:
            del obj
        except Exception:
            pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


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
        }
    )


def proxy_fingerprint(cfg: dict[str, Any], layouts_path: Path, marker: dict[str, Any]) -> str:
    foundation = marker.get("foundation_models", {}).get("stable_diffusion", {})
    return stable_hash(
        {
            "schema": PROXY_SCHEMA,
            "layouts_sha256": sha256_file(layouts_path),
            "count_per_query": int(cfg["proxy"]["count_per_query"]),
            "migc_commit": cfg["migc"]["commit"],
            "migc_checkpoint_sha256": marker.get("migc_checkpoint", {}).get("sha256"),
            "reference_conditioning": cfg["migc"]["reference_conditioning"],
            "stable_diffusion_revision": foundation.get("resolved_revision"),
            "generation": cfg["migc"]["generation"],
        }
    )


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


@torch.no_grad()
def generate_captions(
    *,
    cfg: dict[str, Any],
    queries: Sequence[dict[str, Any]],
    image_paths: Sequence[Path],
    query_manifest: Path,
    marker: dict[str, Any],
    force: bool,
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
    dtype = torch_dtype(str(cfg["runtime"]["caption_dtype"])) if device.type == "cuda" else torch.float32
    processor = Blip2Processor.from_pretrained(str(snapshot), local_files_only=True)
    model = Blip2ForConditionalGeneration.from_pretrained(
        str(snapshot), local_files_only=True, torch_dtype=dtype
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
            if value.dtype.is_floating_point:
                prepared[key] = value.to(device=device, dtype=dtype)
            else:
                prepared[key] = value.to(device=device)
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
        captions = [x.strip() for x in processor.batch_decode(generated, skip_special_tokens=True)]
        captions = [x for x in captions if x]
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
    return f"""You are reconstructing the desired target scene for composed person retrieval.

Reference-scene captions:
{caption_text}

Modification instruction:
{modification}

Infer ONE plausible target full-scene layout after applying the modification. Preserve people, identity-relevant visual details, unaffected objects, and relationships unless the instruction changes them. For MULTI-person or relational instructions, keep all required people and encode the requested relation in the global scene description.

Return ONLY one valid JSON object with this exact structure:
{{
  "scene_prompt": "detailed realistic target-scene prompt",
  "instances": [
    {{"description": "short visual description of one important person/object", "bbox": [x0, y0, x1, y1]}}
  ],
  "target_captions": ["caption 1", "caption 2"]
}}

Rules:
- bbox coordinates are normalized floats in [0,1], with x0 < x1 and y0 < y1.
- include all important people as separate instances; add only visually important context objects.
- use at most 10 instances.
- target_captions must contain exactly {target_caption_count} concise captions of the desired target scene.
- no Markdown, comments, explanation, or code fences.
"""


def extract_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start < 0 or end <= start:
        raise ValueError(f"LLM response contains no JSON object: {cleaned[:300]!r}")
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
        raise ValueError(f"Degenerate bbox after clamping: {value!r}")
    return [x0, y0, x1, y1]


def sanitize_layout(raw: dict[str, Any], target_caption_count: int) -> dict[str, Any]:
    scene = str(raw.get("scene_prompt", "")).strip()
    if not scene:
        raise ValueError("Layout missing scene_prompt")
    instances_raw = raw.get("instances")
    if not isinstance(instances_raw, list) or not instances_raw:
        raise ValueError("Layout must contain at least one instance")
    instances: list[dict[str, Any]] = []
    for item in instances_raw[:10]:
        if not isinstance(item, dict):
            continue
        description = str(item.get("description", "")).strip()
        if not description:
            continue
        bbox = clamp_bbox(item.get("bbox"))
        instances.append({"description": description, "bbox": bbox})
    if not instances:
        raise ValueError("Layout contains no valid instances")

    captions_raw = raw.get("target_captions")
    captions = [] if not isinstance(captions_raw, list) else [str(x).strip() for x in captions_raw if str(x).strip()]
    if not captions:
        captions = [scene]
    while len(captions) < target_caption_count:
        captions.append(captions[-1])
    return {
        "scene_prompt": scene,
        "instances": instances,
        "target_captions": captions[:target_caption_count],
    }


@torch.no_grad()
def generate_layouts(
    *,
    cfg: dict[str, Any],
    queries: Sequence[dict[str, Any]],
    captions: Sequence[dict[str, Any]],
    query_manifest: Path,
    marker: dict[str, Any],
    force: bool,
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
        raise FileNotFoundError(f"Missing Qwen1.5 snapshot: {rel(snapshot)}")
    device = device_from(cfg)
    dtype = torch_dtype(str(cfg["runtime"]["llm_dtype"])) if device.type == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True, trust_remote_code=False)

    model_kwargs: dict[str, Any] = {
        "local_files_only": True,
        "torch_dtype": dtype,
        "low_cpu_mem_usage": True,
    }
    if bool(c.get("load_in_4bit", False)):
        if device.type != "cuda":
            raise RuntimeError("Qwen 4-bit loading requires CUDA")
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(str(snapshot), **model_kwargs).eval()
    if "device_map" not in model_kwargs:
        model = model.to(device)

    target_caption_count = int(c["target_captions_per_query"])
    rows: list[dict[str, Any]] = []
    for qi in progress_bar(range(len(queries)), desc="IP-CIR Qwen layouts", total=len(queries), unit="query"):
        modification = queries[qi].get("text")
        if not isinstance(modification, str) or not modification.strip():
            raise KeyError(f"Query row {qi} has no usable text")
        prompt = layout_prompt(captions[qi]["captions"], modification.strip(), target_caption_count)
        messages = [
            {"role": "system", "content": "You are a precise vision-language scene planner. Output strict JSON only."},
            {"role": "user", "content": prompt},
        ]
        layout: dict[str, Any] | None = None
        last_error: Exception | None = None
        last_text = ""
        for attempt in range(3):
            rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(rendered, return_tensors="pt")
            model_device = next(model.parameters()).device
            inputs = {key: value.to(model_device) for key, value in inputs.items()}
            generate_kwargs: dict[str, Any] = {
                "max_new_tokens": int(c["max_new_tokens"]),
                "do_sample": bool(c["do_sample"]),
                "pad_token_id": tokenizer.eos_token_id,
            }
            if bool(c["do_sample"]):
                generate_kwargs["temperature"] = max(float(c.get("temperature", 0.7)), 1e-5)
            output_ids = model.generate(**inputs, **generate_kwargs)
            completion = output_ids[0, inputs["input_ids"].shape[1] :]
            last_text = tokenizer.decode(completion, skip_special_tokens=True)
            try:
                layout = sanitize_layout(extract_json_object(last_text), target_caption_count)
                break
            except Exception as error:
                last_error = error
                if attempt == 2:
                    break
                messages.extend(
                    [
                        {"role": "assistant", "content": last_text},
                        {
                            "role": "user",
                            "content": (
                                "That response was invalid JSON/layout: " + str(error)
                                + ". Repair it and return ONLY one JSON object matching the requested schema. "
                                "Do not add Markdown or explanation."
                            ),
                        },
                    ]
                )
        if layout is None:
            raise RuntimeError(
                f"Qwen failed to produce a valid layout for query {qi} after 3 attempts: {last_error}; "
                f"last response={last_text[:500]!r}"
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


def validate_proxy_manifest(cfg: dict[str, Any], queries: Sequence[dict[str, Any]], path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing proxy manifest: {rel(path)}")
    rows = load_jsonl(path)
    if len(rows) != len(queries):
        raise ValueError(f"Proxy manifest has {len(rows)} rows, expected {len(queries)}")
    expected_proxies = int(cfg["proxy"]["count_per_query"])
    for qi, row in enumerate(rows):
        if row.get("query_index") != qi:
            raise ValueError(f"Proxy manifest row {qi}: query_index mismatch")
        if row.get("image_id") != queries[qi].get("image_id"):
            raise ValueError(f"Proxy manifest row {qi}: image_id mismatch")
        originals = row.get("original_captions")
        targets = row.get("target_captions")
        paths = row.get("proxy_paths")
        if not isinstance(originals, list) or not originals:
            raise ValueError(f"Proxy manifest row {qi}: missing original_captions")
        if not isinstance(targets, list) or not targets:
            raise ValueError(f"Proxy manifest row {qi}: missing target_captions")
        if not isinstance(paths, list) or len(paths) != expected_proxies:
            raise ValueError(
                f"Proxy manifest row {qi}: expected {expected_proxies} proxy_paths, got {0 if not isinstance(paths, list) else len(paths)}"
            )
        for value in paths:
            if not isinstance(value, str) or not resolve_path(value).is_file():
                raise FileNotFoundError(f"Proxy manifest row {qi}: missing proxy {value!r}")
    return rows


@torch.no_grad()
def generate_proxies(
    *,
    cfg: dict[str, Any],
    queries: Sequence[dict[str, Any]],
    captions: Sequence[dict[str, Any]],
    layouts: Sequence[dict[str, Any]],
    marker: dict[str, Any],
    force: bool,
) -> list[dict[str, Any]]:
    manifest_path = resolve_path(str(cfg["proxy"]["manifest"]))
    layouts_path = resolve_path(str(cfg["cache"]["layouts"]))
    fingerprint = proxy_fingerprint(cfg, layouts_path, marker)
    if not force and manifest_path.is_file():
        try:
            rows = validate_proxy_manifest(cfg, queries, manifest_path)
            if all(row.get("fingerprint") == fingerprint for row in rows):
                print(f"[skip] valid imagined-proxy cache: {rel(manifest_path)}", flush=True)
                return rows
        except Exception as error:
            print(f"[warn] ignoring stale proxy manifest: {error}", flush=True)

    migc_source = resolve_path(str(cfg["migc"]["local_checkout"]))
    migc_checkpoint = resolve_path(str(cfg["migc"]["checkpoint"]))
    sd_snapshot = resolve_path(str(cfg["stable_diffusion"]["local_snapshot"]))
    if not migc_source.is_dir() or not migc_checkpoint.is_file() or not sd_snapshot.is_dir():
        raise FileNotFoundError("MIGC/Stable Diffusion artifacts are incomplete; run download_checkpoint.py")
    if str(migc_source) not in sys.path:
        sys.path.insert(0, str(migc_source))

    from diffusers import EulerDiscreteScheduler
    from migc.migc_pipeline import AttentionStore, MIGCProcessor, StableDiffusionMIGCPipeline
    from migc.migc_utils import load_migc, seed_everything

    device = device_from(cfg)
    dtype = torch_dtype(str(cfg["runtime"]["diffusion_dtype"])) if device.type == "cuda" else torch.float32
    pipe = StableDiffusionMIGCPipeline.from_pretrained(
        str(sd_snapshot), local_files_only=True, torch_dtype=dtype
    )
    pipe.attention_store = AttentionStore()
    load_migc(pipe.unet, pipe.attention_store, str(migc_checkpoint), attn_processor=MIGCProcessor)
    pipe = pipe.to(device)
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
    if hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing()

    count = int(cfg["proxy"]["count_per_query"])
    image_dir = resolve_path(str(cfg["proxy"]["image_dir"]))
    image_dir.mkdir(parents=True, exist_ok=True)
    generation = cfg["migc"]["generation"]
    base_seed = int(generation["seed"])
    rows: list[dict[str, Any]] = []

    for qi in progress_bar(range(len(queries)), desc="IP-CIR MIGC proxies", total=len(queries), unit="query"):
        layout = layouts[qi]
        instances = layout["instances"]
        global_prompt = "masterpiece, best quality, realistic photograph, " + str(layout["scene_prompt"])
        prompt_final = [[global_prompt] + [str(item["description"]) for item in instances]]
        bboxes = [[[float(x) for x in item["bbox"]] for item in instances]]
        query_dir = image_dir / f"{qi:06d}"
        query_dir.mkdir(parents=True, exist_ok=True)
        proxy_paths: list[str] = []
        proxy_hashes: list[str] = []
        for pi in range(count):
            output_path = query_dir / f"proxy_{pi:02d}.png"
            if force:
                output_path.unlink(missing_ok=True)
            if not output_path.is_file():
                seed = base_seed + qi * count + pi
                seed_everything(seed)
                image = pipe(
                    prompt_final,
                    bboxes,
                    num_inference_steps=int(generation["num_inference_steps"]),
                    guidance_scale=float(generation["guidance_scale"]),
                    MIGCsteps=int(generation["migc_steps"]),
                    aug_phase_with_and=False,
                    negative_prompt=str(cfg["migc"]["negative_prompt"]),
                ).images[0]
                image.save(output_path, format="PNG")
            proxy_paths.append(rel(output_path))
            proxy_hashes.append(sha256_file(output_path))
        rows.append(
            {
                "schema": PROXY_SCHEMA,
                "query_index": qi,
                "image_id": queries[qi].get("image_id"),
                "original_captions": list(captions[qi]["captions"]),
                "target_captions": list(layout["target_captions"]),
                "scene_prompt": layout["scene_prompt"],
                "instances": instances,
                "proxy_paths": proxy_paths,
                "proxy_sha256": proxy_hashes,
                "reference_conditioning": str(cfg["migc"]["reference_conditioning"]),
                "fingerprint": fingerprint,
            }
        )
        # Write incrementally so a long generation can be resumed after interruption.
        write_jsonl(manifest_path, rows)

    unload(pipe)
    return validate_proxy_manifest(cfg, queries, manifest_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare IP-CIR imagined proxies")
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
        path = resolve_path(str(cfg["proxy"]["manifest"]))
        validate_proxy_manifest(cfg, queries, path)
        print(f"[ok] precomputed proxy manifest: {rel(path)}", flush=True)
        return

    marker = read_prepared_marker(cfg)
    image_paths = query_image_paths(queries, gallery)

    stages = [args.stage] if args.stage != "all" else ["captions", "layouts", "generate"]
    tracker = PhaseTracker(METHOD_ID, total=len(stages))
    captions: list[dict[str, Any]] | None = None
    layouts: list[dict[str, Any]] | None = None

    for stage in stages:
        if stage == "captions":
            with tracker.phase("Generate BLIP2 reference captions"):
                captions = generate_captions(
                    cfg=cfg,
                    queries=queries,
                    image_paths=image_paths,
                    query_manifest=query_manifest,
                    marker=marker,
                    force=args.force,
                )
        elif stage == "layouts":
            with tracker.phase("Infer target layouts with Qwen1.5-32B"):
                if captions is None:
                    captions_path = resolve_path(str(cfg["cache"]["captions"]))
                    if not captions_path.is_file():
                        raise FileNotFoundError("Caption cache missing; run --stage captions first")
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
            with tracker.phase("Generate five MIGC imagined proxies per query"):
                if captions is None:
                    captions = load_jsonl(resolve_path(str(cfg["cache"]["captions"])))
                if layouts is None:
                    layouts = load_jsonl(resolve_path(str(cfg["cache"]["layouts"])))
                generate_proxies(
                    cfg=cfg,
                    queries=queries,
                    captions=captions,
                    layouts=layouts,
                    marker=marker,
                    force=args.force,
                )
        else:
            raise AssertionError(stage)

    tracker.finish()


if __name__ == "__main__":
    main()
