#!/usr/bin/env python3
"""Generate CPR proxy images with the released IP-CIR MIGC+ELITE implementation.

This worker imports the pinned author source but deliberately bypasses GLIP/SAM/GroundingDINO:
for the benchmark's direct-full-scene adapter, every image-referenced layout instance uses the
whole query scene as its ELITE visual concept/mask. No GT target box or identity mapping is used.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import os
import random
import sys
import types
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = Path(__file__).resolve().parent / "config.yaml"


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
                raise TypeError(f"{path}:{lineno}: expected object")
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




def stable_hash(payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".part")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temp, path)

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


class _DummyCfg:
    local_rank = 0
    num_gpus = 1

    def merge_from_file(self, *_args, **_kwargs):
        return None

    def merge_from_list(self, *_args, **_kwargs):
        return None


def _install_import_stubs() -> None:
    """Satisfy author-file imports that are not used in the full-scene adapter."""

    def module(name: str) -> types.ModuleType:
        value = sys.modules.get(name)
        if value is None:
            value = types.ModuleType(name)
            sys.modules[name] = value
        return value

    # Aesthetic predictor is imported by the released file but only used by its
    # optional generation-quality filter, which this adapter bypasses.
    predictor = module("predictor")
    predictor_simple = module("predictor.simple_inference")
    predictor.simple_inference = predictor_simple

    class DummyMLP:
        def __init__(self, *_args, **_kwargs):
            pass

    predictor_simple.MLP = DummyMLP
    predictor_simple.normalized = lambda x, *args, **kwargs: x

    # The released layout_utils module imports the full detector/segmentation stack.
    # Only its optional filtering helpers are referenced by generate_proxy_migc_elite.py.
    layout_utils = module("layout_utils")
    layout_utils_utils = module("layout_utils.utils")
    layout_utils.utils = layout_utils_utils
    layout_utils_utils.filter_image_position = lambda *_args, **_kwargs: True
    layout_utils_utils.detect_on_image = lambda *_args, **_kwargs: None
    layout_utils_utils.segment_on_bbox = lambda *_args, **_kwargs: None

    pycocotools = module("pycocotools")
    pycocotools_mask = module("pycocotools.mask")
    pycocotools.mask = pycocotools_mask

    groundingdino = module("groundingdino")
    gd_util = module("groundingdino.util")
    gd_infer = module("groundingdino.util.inference")
    gd_data = module("groundingdino.datasets")
    gd_transforms = module("groundingdino.datasets.transforms")
    groundingdino.util = gd_util
    groundingdino.datasets = gd_data
    gd_util.inference = gd_infer
    gd_data.transforms = gd_transforms

    class DummyModel:
        def __init__(self, *_args, **_kwargs):
            pass

    gd_infer.Model = DummyModel

    segment_anything = module("segment_anything")
    segment_anything.sam_model_registry = {}

    class DummySamPredictor:
        def __init__(self, *_args, **_kwargs):
            pass

    segment_anything.SamPredictor = DummySamPredictor

    mrb = module("maskrcnn_benchmark")
    mrb_config = module("maskrcnn_benchmark.config")
    mrb_engine = module("maskrcnn_benchmark.engine")
    mrb_predictor = module("maskrcnn_benchmark.engine.predictor_glip")
    mrb.config = mrb_config
    mrb.engine = mrb_engine
    mrb_config.cfg = _DummyCfg()

    class DummyGLIPDemo:
        def __init__(self, *_args, **_kwargs):
            pass

    mrb_predictor.GLIPDemo = DummyGLIPDemo


def load_author_module(source: Path):
    _install_import_stubs()
    # The release imports both `MIGC.migc...` and `migc...` namespaces.
    for path in (source, source / "MIGC"):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)

    target = source / "generate_proxy_migc_elite.py"
    spec = importlib.util.spec_from_file_location("ipcir_official_proxy", target)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import released proxy generator: {target}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    # Released validation already has skip_mode=True but evaluates the filter first.
    # The CPR full-scene adapter intentionally has no GLIP/SAM/DINO filter stage.
    module.filter_image_position = lambda *_args, **_kwargs: True
    return module


def load_pipeline(cfg: dict[str, Any], marker: dict[str, Any], author, device: torch.device):
    from diffusers import EulerDiscreteScheduler
    from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModel

    source = resolve_path(str(cfg["author_source"]["local_checkout"]))
    if str(source / "MIGC") not in sys.path:
        sys.path.insert(0, str(source / "MIGC"))
    from migc.migc_pipeline import AttentionStore, StableDiffusionMIGCPipeline
    from migc.migc_utils import load_migc

    dtype = torch_dtype(str(cfg["runtime"]["diffusion_dtype"]), device)
    components = resolve_path(str(cfg["sd15_components"]["local_snapshot"]))
    text_encoder_dir = components / "text_encoder"
    tokenizer_dir = components / "tokenizer"
    rv_path = resolve_path(str(cfg["realistic_vision"]["path"]))
    migc_path = resolve_path(str(cfg["migc"]["checkpoint"]))
    original_config = source / "MIGC" / "migc_gui_weights" / "v1-inference.yaml"

    for path in (text_encoder_dir, tokenizer_dir):
        if not path.is_dir():
            raise FileNotFoundError(f"Missing Stable Diffusion component: {rel(path)}")
    for path in (rv_path, migc_path, original_config):
        if not path.is_file():
            raise FileNotFoundError(path)

    text_encoder = CLIPTextModel.from_pretrained(
        str(text_encoder_dir), local_files_only=True, torch_dtype=dtype
    )
    tokenizer = CLIPTokenizer.from_pretrained(str(tokenizer_dir), local_files_only=True)

    pipe = StableDiffusionMIGCPipeline.from_single_file(
        str(rv_path),
        original_config_file=str(original_config),
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        load_safety_checker=False,
        torch_dtype=dtype,
    )
    pipe.attention_store = AttentionStore()
    load_migc(
        pipe.unet,
        pipe.attention_store,
        str(migc_path),
        attn_processor=author.MIGCProcessorELITE,
    )
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)

    # Same text-embedding injection hook used by the released generator.
    for submodule in pipe.text_encoder.modules():
        if submodule.__class__.__name__ == "CLIPTextTransformer":
            submodule.__class__.__call__ = author.inj_forward_text

    mapper = author.Mapper(input_dim=1024, output_dim=768)
    mapper_local = author.MapperLocal(input_dim=1024, output_dim=768)
    clip_snapshot = resolve_path(str(cfg["elite"]["clip_vision_snapshot"]))
    image_encoder = CLIPVisionModel.from_pretrained(
        str(clip_snapshot), local_files_only=True, torch_dtype=dtype
    ).eval()

    # Build ELITE projection modules with exactly the released UNet-attention shapes.
    import torch.nn as nn

    for name, submodule in pipe.unet.named_modules():
        if submodule.__class__.__name__ != "Attention" or "attn1" in name:
            continue
        shape_k = submodule.to_k.weight.shape
        shape_v = submodule.to_v.weight.shape
        mapper.add_module(
            f"{name.replace('.', '_')}_to_k",
            nn.Linear(shape_k[1], shape_k[0], bias=False),
        )
        mapper.add_module(
            f"{name.replace('.', '_')}_to_v",
            nn.Linear(shape_v[1], shape_v[0], bias=False),
        )
        mapper_local.add_module(
            f"{name.replace('.', '_')}_to_v",
            nn.Linear(shape_v[1], shape_v[0], bias=False),
        )
        mapper_local.add_module(
            f"{name.replace('.', '_')}_to_k",
            nn.Linear(shape_k[1], shape_k[0], bias=False),
        )

    global_mapper = resolve_path(str(cfg["elite"]["global_mapper"]))
    local_mapper = resolve_path(str(cfg["elite"]["local_mapper"]))
    mapper.load_state_dict(torch.load(global_mapper, map_location="cpu"), strict=True)
    mapper_local.load_state_dict(torch.load(local_mapper, map_location="cpu"), strict=True)

    # Attach the learned ELITE K/V projections to the released MIGC+ELITE processors.
    for name, submodule in pipe.unet.named_modules():
        if "attn1" in name or submodule.__class__.__name__ != "MIGCProcessorELITE":
            continue
        attention_name = ".".join(name.split(".")[:-1])
        key = attention_name.replace(".", "_")
        submodule.add_module("to_k_global", getattr(mapper, f"{key}_to_k"))
        submodule.add_module("to_v_global", getattr(mapper, f"{key}_to_v"))
        submodule.add_module("to_k_local", getattr(mapper_local, f"{key}_to_k"))
        submodule.add_module("to_v_local", getattr(mapper_local, f"{key}_to_v"))

    pipe = pipe.to(device)
    image_encoder = image_encoder.to(device)
    mapper = mapper.to(device).eval()
    mapper_local = mapper_local.to(device).eval()
    pipe.set_progress_bar_config(disable=True)

    negative = "worst quality, low quality, bad anatomy"
    uncond_input = pipe.tokenizer(
        [negative],
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        return_tensors="pt",
    )
    uncond_output = pipe.text_encoder({"input_ids": uncond_input.input_ids.to(device)})
    uncond = {
        "embed": uncond_output[0],
        "pooler": uncond_output["pooler_output"],
    }
    return pipe, image_encoder, mapper, mapper_local, uncond


def _valid_bbox(value: Any) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        x0, y0, x1, y1 = [float(x) for x in value]
    except Exception:
        return None
    x0, y0, x1, y1 = [min(1.0, max(0.0, x)) for x in (x0, y0, x1, y1)]
    if x1 <= x0 or y1 <= y0:
        return None
    return [x0, y0, x1, y1]


def build_author_input(author, layout: Sequence[dict[str, Any]], query_image: Path, tokenizer):
    with Image.open(query_image) as image:
        width, height = image.size
    full_mask = np.ones((height, width, 3), dtype=np.uint8)

    prompt_final: list[list[str]] = [[]]
    bboxes: list[list[list[float]]] = [[]]
    refs: list[str] = []
    image_paths: list[str] = []
    masks: list[np.ndarray | None] = []
    has_global = False

    for item in layout:
        if not isinstance(item, dict):
            continue
        desc = str(item.get("desc") or item.get("description") or "").strip()
        label = str(item.get("cate") or item.get("label") or "person").strip() or "person"
        bbox = _valid_bbox(item.get("bbox"))
        is_scene = bool(item.get("is_scene", False))
        ref = str(item.get("ref", "text")).strip().lower()

        if is_scene:
            if has_global or not desc:
                continue
            has_global = True
            prompt_final[0].insert(0, desc)
            refs.insert(0, "text")
            image_paths.insert(0, "_")
            masks.insert(0, None)
            continue

        if bbox is None or not desc:
            continue

        bboxes[0].append(bbox)
        if ref == "image":
            # Same construction as the released generate_proxy_migc_elite.py:
            # one textual instance plus one ELITE image token instance at the same bbox.
            prompt_final[0].append(desc)
            prompt_final[0].append(f"a * {label}")
            refs.extend(["text", "image"])
            image_paths.extend(["_", str(query_image)])
            masks.extend([None, full_mask])
            bboxes[0].append(bbox)
        else:
            prompt_final[0].append(desc)
            refs.append("text")
            image_paths.append("_")
            masks.append(None)

    if not has_global:
        prompt_final[0].insert(0, "a high quality realistic image")
        refs.insert(0, "text")
        image_paths.insert(0, "_")
        masks.insert(0, None)

    if not bboxes[0]:
        # MIGC expects at least one controlled instance. Use a full-scene text instance as fallback.
        prompt_final[0].append("the people and scene described by the instruction")
        refs.append("text")
        image_paths.append("_")
        masks.append(None)
        bboxes[0].append([0.05, 0.05, 0.95, 0.95])

    example = author.load_input(prompt_final[0], refs, image_paths, tokenizer, masks)
    return example, prompt_final, bboxes


def generation_fingerprint(cfg: dict[str, Any], marker: dict[str, Any], job: dict[str, Any]) -> str:
    return stable_hash(
        {
            "schema": 2,
            "author_commit": cfg["author_source"]["commit"],
            "job": job,
            "generation": cfg["migc"]["generation"],
            "reference_mask": cfg["proxy"]["reference_mask"],
            "migc_sha256": marker.get("migc_checkpoint", {}).get("sha256"),
            "elite_global_sha256": marker.get("elite", {}).get("global_mapper", {}).get("sha256"),
            "elite_local_sha256": marker.get("elite", {}).get("local_mapper", {}).get("sha256"),
            "realistic_vision_sha256": marker.get("realistic_vision", {}).get("sha256"),
        }
    )


def manifest_row(job: dict[str, Any], paths: list[Path], fingerprint: str) -> dict[str, Any]:
    return {
        "schema": 2,
        "query_index": int(job["query_index"]),
        "image_id": job.get("image_id"),
        "original_captions": list(job["original_captions"]),
        "target_captions": list(job["target_captions"]),
        "scene_prompt": str(job.get("scene_prompt", "")),
        "layout": job["layout"],
        "proxy_paths": [rel(path) for path in paths],
        "author_source": "LeyRio/Imagine-and-Seek",
        "reference_conditioning": "MIGC+ELITE_full_scene_mask",
        "generation_fingerprint": fingerprint,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Official-source IP-CIR MIGC+ELITE proxy worker")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()

    cfg = load_yaml(resolve_path(args.config))
    device = torch.device(str(cfg["runtime"]["device"]))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Proxy generation requires CUDA")

    marker_path = resolve_path(str(cfg["migc"]["prepared_marker"]))
    if not marker_path.is_file():
        raise FileNotFoundError(f"Missing prepared marker: {rel(marker_path)}")
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if marker.get("author_source", {}).get("commit") != str(cfg["author_source"]["commit"]):
        raise RuntimeError("Prepared marker/source commit mismatch")

    jobs_path = resolve_path(str(cfg["proxy"]["jobs"]))
    jobs = load_jsonl(jobs_path)
    output_dir = resolve_path(str(cfg["proxy"]["image_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = resolve_path(str(cfg["proxy"]["manifest"]))
    count = int(cfg["proxy"]["count_per_query"])

    source = resolve_path(str(cfg["author_source"]["local_checkout"]))
    author = load_author_module(source)
    pipe, image_encoder, mapper, mapper_local, uncond = load_pipeline(cfg, marker, author, device)

    generation = cfg["migc"]["generation"]
    migc_param = {
        "MIGCsteps": int(generation["migc_steps"]),
        "NaiveFuserSteps": int(generation["naive_fuser_steps"]),
        "negative_prompt": "worst quality, low quality, bad anatomy, watermark, text, blurry",
    }
    base_seed = int(generation["seed"])

    rows: list[dict[str, Any]] = []
    for job in jobs:
        qi = int(job["query_index"])
        query_image = resolve_path(str(job["query_image"]))
        if not query_image.is_file():
            raise FileNotFoundError(query_image)
        paths = [output_dir / f"q{qi:05d}_p{pi:02d}.png" for pi in range(count)]
        meta_path = output_dir / f"q{qi:05d}.meta.json"
        fingerprint = generation_fingerprint(cfg, marker, job)
        cache_valid = (
            read_json(meta_path) == {"generation_fingerprint": fingerprint}
            and all(path.is_file() and path.stat().st_size > 0 for path in paths)
        )
        if cache_valid:
            missing: list[int] = []
            print(f"[proxy] cache hit query={qi}", flush=True)
        else:
            for path in paths:
                path.unlink(missing_ok=True)
            meta_path.unlink(missing_ok=True)
            missing = list(range(count))

        if missing:
            example, prompt, bboxes = build_author_input(
                author, job["layout"], query_image, pipe.tokenizer
            )
            for pi in missing:
                seed = base_seed + qi * 1009 + pi
                random.seed(seed)
                np.random.seed(seed % (2**32 - 1))
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
                with torch.inference_mode():
                    image = author.validation(
                        example,
                        pipe,
                        image_encoder,
                        mapper,
                        mapper_local,
                        device,
                        float(generation["guidance_scale"]),
                        prompt,
                        bboxes,
                        migc_param,
                        uncond,
                        job["layout"],
                        None,
                        None,
                        None,
                        llambda=1.0,
                        num_steps=int(generation["num_inference_steps"]),
                    )
                image.save(paths[pi])
                print(f"[proxy] query={qi} proxy={pi + 1}/{count} -> {rel(paths[pi])}", flush=True)
            write_json(meta_path, {"generation_fingerprint": fingerprint})

        rows.append(manifest_row(job, paths, fingerprint))
        write_jsonl(manifest_path, rows)

    del pipe, image_encoder, mapper, mapper_local, uncond
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"[ok] proxy manifest: {rel(manifest_path)} ({len(rows)} queries)", flush=True)


if __name__ == "__main__":
    main()
