#!/usr/bin/env python3
"""Released-source MIGC+ELITE worker for the CPR adaptation.

Only the dataset-specific mask/filter boundary is adapted: CPR supplies no author
object mask, so every image-referenced instance uses the complete query image as
ELITE's visual concept. Detector/GLIP modules used only by the released optional
quality filter are stubbed; the released MIGCProcessorELITE, Mapper, MapperLocal,
inj_forward_text, load_input, and validation functions are imported from the
pinned author source.
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

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms.functional as TF
import yaml
from PIL import Image
from torchvision.transforms import CenterCrop, Compose, InterpolationMode, Normalize, Resize, ToTensor

ROOT = Path(__file__).resolve().parents[3]
METHOD_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = METHOD_DIR / "config.yaml"


def load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError(f"Expected mapping: {path}")
    return data


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise TypeError(f"{path}:{lineno}: expected object")
            rows.append(row)
    return rows


def write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(tmp, path)


def resolve_path(value: str) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else ROOT / path).resolve()


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except Exception:
        return str(path.resolve())


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def stable_hash(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _dummy_module(name: str) -> types.ModuleType:
    value = sys.modules.get(name)
    if value is None:
        value = types.ModuleType(name)
        sys.modules[name] = value
    return value


class _DummyCfg:
    local_rank = 0
    num_gpus = 1
    def merge_from_file(self, *_a, **_kw):
        return None
    def merge_from_list(self, *_a, **_kw):
        return None


def install_optional_import_stubs() -> None:
    """Stub author imports that are not executed by the CPR full-scene path."""
    # predictor.simple_inference
    predictor = _dummy_module("predictor")
    simple = _dummy_module("predictor.simple_inference")
    predictor.simple_inference = simple
    class DummyMLP:
        def __init__(self, *_a, **_kw):
            pass
    simple.MLP = DummyMLP
    simple.normalized = lambda x, *a, **kw: x

    # layout_utils.utils is imported with `*` by the author generator. Crucially,
    # load_input() expects cv2 and torchvision to have arrived through that star import.
    layout_utils = _dummy_module("layout_utils")
    utils = _dummy_module("layout_utils.utils")
    layout_utils.utils = utils
    utils.cv2 = cv2
    utils.torchvision = torchvision
    utils.filter_image_position = lambda *_a, **_kw: True
    utils.detect_on_image = lambda *_a, **_kw: None
    utils.segment_on_bbox = lambda *_a, **_kw: None

    # optional detector stack
    pycoco = _dummy_module("pycocotools")
    pycoco_mask = _dummy_module("pycocotools.mask")
    pycoco.mask = pycoco_mask

    grounding = _dummy_module("groundingdino")
    gd_util = _dummy_module("groundingdino.util")
    gd_inf = _dummy_module("groundingdino.util.inference")
    grounding.util = gd_util
    gd_util.inference = gd_inf
    class DummyGroundingModel:
        def __init__(self, *_a, **_kw):
            pass
    gd_inf.Model = DummyGroundingModel

    segment = _dummy_module("segment_anything")
    segment.sam_model_registry = {}
    class DummySamPredictor:
        def __init__(self, *_a, **_kw):
            pass
    segment.SamPredictor = DummySamPredictor

    mrb = _dummy_module("maskrcnn_benchmark")
    mrb_cfg = _dummy_module("maskrcnn_benchmark.config")
    mrb_engine = _dummy_module("maskrcnn_benchmark.engine")
    mrb_pred = _dummy_module("maskrcnn_benchmark.engine.predictor_glip")
    mrb.config = mrb_cfg
    mrb.engine = mrb_engine
    mrb_cfg.cfg = _DummyCfg()
    class DummyGLIPDemo:
        def __init__(self, *_a, **_kw):
            pass
    mrb_pred.GLIPDemo = DummyGLIPDemo

    # matplotlib is imported but only used for optional visualization/filtering.
    matplotlib = _dummy_module("matplotlib")
    pyplot = _dummy_module("matplotlib.pyplot")
    pylab = _dummy_module("matplotlib.pylab")
    matplotlib.pyplot = pyplot
    matplotlib.pylab = pylab


def load_author_module(source: Path):
    install_optional_import_stubs()
    for path in (source, source / "MIGC"):
        s = str(path)
        if s not in sys.path:
            sys.path.insert(0, s)
    target = source / "generate_proxy_migc_elite.py"
    spec = importlib.util.spec_from_file_location("ipcir_released_proxy", target)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {target}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    # Make the implicit globals used by released load_input explicit and deterministic.
    module.cv2 = cv2
    module.torchvision = torchvision
    module.filter_image_position = lambda *_a, **_kw: True

    required = [
        "MIGCProcessorELITE", "Mapper", "MapperLocal", "inj_forward_text",
        "load_input", "validation",
    ]
    missing = [name for name in required if not hasattr(module, name)]
    if missing:
        raise RuntimeError(f"Released generator missing required symbols: {missing}")
    return module


class TargetPad:
    """Released LDRE target-pad transform used for CLIP-L retrieval features."""
    def __init__(self, target_ratio: float, size: int):
        self.target_ratio = float(target_ratio)
        self.size = int(size)

    def __call__(self, image: Image.Image) -> Image.Image:
        w, h = image.size
        actual_ratio = max(w, h) / min(w, h)
        if actual_ratio < self.target_ratio:
            return image
        scaled_max_wh = max(w, h) / self.target_ratio
        hp = max(int((scaled_max_wh - w) / 2), 0)
        vp = max(int((scaled_max_wh - h) / 2), 0)
        return TF.pad(image, [hp, vp, hp, vp], 0, "constant")


def targetpad_transform(dim: int):
    return Compose([
        TargetPad(1.25, dim),
        Resize(dim, interpolation=InterpolationMode.BICUBIC),
        CenterCrop(dim),
        lambda im: im.convert("RGB"),
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])


def load_prepared_marker(cfg: dict[str, Any]) -> dict[str, Any]:
    path = resolve_path(str(cfg["migc"]["prepared_marker"]))
    data = read_json(path)
    if data is None:
        raise FileNotFoundError(f"Missing prepared marker: {path}")
    return data


def load_retrieval_clip(cfg: dict[str, Any], marker: dict[str, Any], device: torch.device):
    import clip
    info = marker.get("assets", {}).get("openai_clip", {})
    path = resolve_path(str(info.get("path", "")))
    if not path.is_file():
        raise FileNotFoundError(f"Missing OpenAI CLIP-L checkpoint in marker: {path}")
    expected = str(info.get("sha256", ""))
    if expected and sha256_file(path) != expected:
        raise RuntimeError("OpenAI CLIP-L checkpoint checksum mismatch")
    model, _ = clip.load(str(path), device=device, jit=False)
    model.eval()
    transform = targetpad_transform(int(model.visual.input_resolution))
    return model, transform, expected


def encode_proxy_image(image: Image.Image, model, transform, device: torch.device) -> np.ndarray:
    tensor = transform(image).unsqueeze(0).to(device, non_blocking=True)
    with torch.inference_mode():
        feature = F.normalize(model.encode_image(tensor).float(), dim=-1)[0]
    return feature.cpu().numpy().astype(np.float32, copy=False)


def load_proxy_feature_store(cfg: dict[str, Any], nq: int, count: int, dim: int, clip_sha: str):
    path = resolve_path(str(cfg["retrieval"]["proxy_features"]))
    state_path = resolve_path(str(cfg["retrieval"]["proxy_features_state"]))
    shape = (nq, count, dim)
    state = read_json(state_path)
    valid_state = (
        isinstance(state, dict)
        and state.get("schema") == 5
        and state.get("shape") == list(shape)
        and state.get("clip_sha256") == clip_sha
        and isinstance(state.get("fingerprints"), list)
        and len(state["fingerprints"]) == nq
    )
    if path.is_file() and valid_state:
        try:
            array = np.load(path, mmap_mode="r+", allow_pickle=False)
            if array.shape == shape and array.dtype == np.float32:
                return path, state_path, array, state
        except Exception:
            pass
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=shape)
    array[:] = np.nan
    array.flush()
    state = {
        "schema": 5,
        "shape": list(shape),
        "clip_sha256": clip_sha,
        "fingerprints": [None] * nq,
    }
    write_json(state_path, state)
    return path, state_path, array, state


def diffusion_dtype(cfg: dict[str, Any]) -> torch.dtype:
    name = str(cfg["runtime"]["diffusion_dtype"]).lower()
    if name in {"float32", "fp32"}:
        return torch.float32
    if name in {"float16", "fp16"}:
        return torch.float16
    raise ValueError(f"Unsupported diffusion dtype: {name}")


def load_pipeline(cfg: dict[str, Any], author, device: torch.device):
    from diffusers import EulerDiscreteScheduler
    from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModel

    source = resolve_path(str(cfg["author_source"]["local_checkout"]))
    migc_root = str(source / "MIGC")
    if migc_root not in sys.path:
        sys.path.insert(0, migc_root)
    from migc.migc_pipeline import AttentionStore, StableDiffusionMIGCPipeline
    from migc.migc_utils import load_migc

    dtype = diffusion_dtype(cfg)
    components = resolve_path(str(cfg["sd15_components"]["local_snapshot"]))
    text_dir = components / "text_encoder"
    tok_dir = components / "tokenizer"
    rv = resolve_path(str(cfg["realistic_vision"]["path"]))
    migc = resolve_path(str(cfg["migc"]["checkpoint"]))
    original_config = source / "MIGC" / "migc_gui_weights" / "v1-inference.yaml"
    for path in (text_dir, tok_dir):
        if not path.is_dir():
            raise FileNotFoundError(path)
    for path in (rv, migc, original_config):
        if not path.is_file():
            raise FileNotFoundError(path)

    text_encoder = CLIPTextModel.from_pretrained(str(text_dir), local_files_only=True, torch_dtype=dtype)
    tokenizer = CLIPTokenizer.from_pretrained(str(tok_dir), local_files_only=True)
    pipe = StableDiffusionMIGCPipeline.from_single_file(
        str(rv),
        original_config_file=str(original_config),
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        load_safety_checker=False,
        torch_dtype=dtype,
    )
    pipe.attention_store = AttentionStore()
    load_migc(pipe.unet, pipe.attention_store, str(migc), attn_processor=author.MIGCProcessorELITE)
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)

    for submodule in pipe.text_encoder.modules():
        if submodule.__class__.__name__ == "CLIPTextTransformer":
            submodule.__class__.__call__ = author.inj_forward_text

    # Match released ELITE path: vision encoder and mappers are float32; the released
    # code converts the produced injected embedding to fp16 only after the mapper.
    mapper = author.Mapper(input_dim=1024, output_dim=768)
    mapper_local = author.MapperLocal(input_dim=1024, output_dim=768)
    clip_snapshot = resolve_path(str(cfg["elite"]["clip_vision_snapshot"]))
    image_encoder = CLIPVisionModel.from_pretrained(str(clip_snapshot), local_files_only=True).eval()

    import torch.nn as nn
    for name, submodule in pipe.unet.named_modules():
        if submodule.__class__.__name__ != "Attention" or "attn1" in name:
            continue
        k_shape = submodule.to_k.weight.shape
        v_shape = submodule.to_v.weight.shape
        key = name.replace(".", "_")
        mapper.add_module(f"{key}_to_k", nn.Linear(k_shape[1], k_shape[0], bias=False))
        mapper.add_module(f"{key}_to_v", nn.Linear(v_shape[1], v_shape[0], bias=False))
        mapper_local.add_module(f"{key}_to_k", nn.Linear(k_shape[1], k_shape[0], bias=False))
        mapper_local.add_module(f"{key}_to_v", nn.Linear(v_shape[1], v_shape[0], bias=False))

    global_mapper = resolve_path(str(cfg["elite"]["global_mapper"]))
    local_mapper = resolve_path(str(cfg["elite"]["local_mapper"]))
    mapper.load_state_dict(torch.load(global_mapper, map_location="cpu"), strict=True)
    mapper_local.load_state_dict(torch.load(local_mapper, map_location="cpu"), strict=True)

    for name, submodule in pipe.unet.named_modules():
        if submodule.__class__.__name__ != "MIGCProcessorELITE" or "attn1" in name:
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
    tokens = pipe.tokenizer(
        [negative], padding="max_length", max_length=pipe.tokenizer.model_max_length, return_tensors="pt"
    )
    uncond_out = pipe.text_encoder({"input_ids": tokens.input_ids.to(device)})
    uncond = {"embed": uncond_out[0], "pooler": uncond_out["pooler_output"]}
    return pipe, image_encoder, mapper, mapper_local, uncond


def _bbox(value: Any) -> list[float] | None:
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
    with Image.open(query_image) as im:
        width, height = im.size
    full_mask = np.ones((height, width, 3), dtype=np.uint8)
    prompts: list[str] = []
    refs: list[str] = []
    image_paths: list[str] = []
    masks: list[np.ndarray | None] = []
    bboxes: list[list[list[float]]] = [[]]

    scene = next((x for x in layout if isinstance(x, dict) and x.get("is_scene")), None)
    scene_desc = str((scene or {}).get("desc") or "a high quality realistic image").strip()
    prompts.append(scene_desc)
    refs.append("text")
    image_paths.append("_")
    masks.append(None)

    for item in layout:
        if not isinstance(item, dict) or item.get("is_scene"):
            continue
        box = _bbox(item.get("bbox"))
        desc = str(item.get("desc") or "").strip()
        label = str(item.get("cate") or item.get("label") or "person").strip() or "person"
        if box is None or not desc:
            continue
        ref = str(item.get("ref", "text")).lower()
        if ref == "image":
            # Released generator represents an image-referenced object as a text instance
            # plus an ELITE placeholder instance at the same box.
            prompts.extend([desc, f"a * {label}"])
            refs.extend(["text", "image"])
            image_paths.extend(["_", str(query_image)])
            masks.extend([None, full_mask])
            bboxes[0].extend([box, box])
        else:
            prompts.append(desc)
            refs.append("text")
            image_paths.append("_")
            masks.append(None)
            bboxes[0].append(box)

    if not bboxes[0]:
        prompts.extend(["the relevant person from the reference image", "a * person"])
        refs.extend(["text", "image"])
        image_paths.extend(["_", str(query_image)])
        masks.extend([None, full_mask])
        bboxes[0].extend([[0.08, 0.04, 0.92, 0.96], [0.08, 0.04, 0.92, 0.96]])

    example = author.load_input(prompts, refs, image_paths, tokenizer, masks)
    return example, [prompts], bboxes


def generation_fingerprint(cfg: dict[str, Any], job: dict[str, Any]) -> str:
    return stable_hash({
        "schema": 4,
        "author_commit": cfg["author_source"]["commit"],
        "job": job,
        "generation": cfg["migc"]["generation"],
        "reference_mask": cfg["proxy"]["reference_mask"],
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--stage", choices=["import", "pipeline", "generate"], default="generate")
    args = parser.parse_args()
    cfg = load_yaml(resolve_path(args.config))
    source = resolve_path(str(cfg["author_source"]["local_checkout"]))
    if not source.is_dir():
        raise FileNotFoundError(source)
    author = load_author_module(source)
    print("[ok] released generator import preflight", flush=True)
    if args.stage == "import":
        return

    device = torch.device(str(cfg["runtime"]["device"]))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("MIGC+ELITE requires CUDA")
    pipe, image_encoder, mapper, mapper_local, uncond = load_pipeline(cfg, author, device)
    print("[ok] MIGC+ELITE+RealisticVision pipeline preflight", flush=True)
    if args.stage == "pipeline":
        del pipe, image_encoder, mapper, mapper_local, uncond
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return

    jobs_path = resolve_path(str(cfg["cache"]["proxy_jobs"]))
    jobs = load_jsonl(jobs_path)
    manifest_path = resolve_path(str(cfg["proxy"]["manifest"]))
    count = int(cfg["proxy"]["count_per_query"])
    dim = int(cfg["retrieval"]["projection_dim"])
    gen = cfg["migc"]["generation"]
    migc_param = {
        "MIGCsteps": int(gen["migc_steps"]),
        "NaiveFuserSteps": int(gen["naive_fuser_steps"]),
        "negative_prompt": "worst quality, low quality, bad anatomy, watermark, text, blurry",
    }
    base_seed = int(gen["seed"])

    marker = load_prepared_marker(cfg)
    retrieval_model, retrieval_transform, clip_sha = load_retrieval_clip(cfg, marker, device)
    feature_path, state_path, feature_store, state = load_proxy_feature_store(
        cfg, len(jobs), count, dim, clip_sha
    )
    audit_stride = int(cfg["proxy"].get("audit_image_stride", 0))
    audit_dir = resolve_path(str(cfg["proxy"].get("audit_image_dir", "runs/imagine_seek/cache/proxy_audit")))
    if audit_stride > 0:
        audit_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for job in jobs:
        qi = int(job["query_index"])
        query_image = resolve_path(str(job["query_image"]))
        fp = generation_fingerprint(cfg, job)
        current = np.asarray(feature_store[qi])
        valid = state["fingerprints"][qi] == fp and np.isfinite(current).all()
        audit_path: Path | None = None

        if not valid:
            example, prompt, bboxes = build_author_input(author, job["layout"], query_image, pipe.tokenizer)
            query_features = np.empty((count, dim), dtype=np.float32)
            for pi in range(count):
                seed = base_seed + qi * 1009 + pi
                random.seed(seed)
                np.random.seed(seed % (2**32 - 1))
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                with torch.inference_mode():
                    image = author.validation(
                        example, pipe, image_encoder, mapper, mapper_local, device,
                        float(gen["guidance_scale"]), prompt, bboxes, migc_param,
                        uncond, job["layout"], None, None, None,
                        llambda=1.0, num_steps=int(gen["num_inference_steps"]),
                    )
                query_features[pi] = encode_proxy_image(
                    image, retrieval_model, retrieval_transform, device
                )
                if audit_stride > 0 and pi == 0 and qi % audit_stride == 0:
                    audit_path = audit_dir / f"q{qi:05d}_p00.png"
                    image.save(audit_path)
                del image
                print(f"[proxy->feature] q={qi} {pi+1}/{count}", flush=True)
            feature_store[qi] = query_features
            feature_store.flush()
            state["fingerprints"][qi] = fp
            write_json(state_path, state)
        else:
            if audit_stride > 0 and qi % audit_stride == 0:
                candidate = audit_dir / f"q{qi:05d}_p00.png"
                if candidate.is_file():
                    audit_path = candidate
            print(f"[proxy-feature] cache q={qi}", flush=True)

        rows.append({
            "schema": 5,
            "query_index": qi,
            "image_id": job.get("image_id"),
            "original_captions": list(job["original_captions"]),
            "target_captions": list(job["target_captions"]),
            "scene_prompt": job.get("scene_prompt", ""),
            "layout": job["layout"],
            "proxy_feature_file": rel(feature_path),
            "proxy_feature_index": qi,
            "proxy_count": count,
            "audit_proxy_path": rel(audit_path) if audit_path is not None else None,
            "author_source": "LeyRio/Imagine-and-Seek",
            "reference_conditioning": "MIGC+ELITE_full_scene_mask_no_GT",
            "generation_fingerprint": fp,
            "storage_mode": "stream_generate_encode_discard",
        })
        write_jsonl(manifest_path, rows)

    if not np.isfinite(np.asarray(feature_store)).all():
        raise RuntimeError("Proxy feature store contains incomplete/non-finite rows")
    print(
        f"[ok] proxy manifest={rel(manifest_path)} rows={len(rows)}; "
        f"features={rel(feature_path)}; persisted_proxy_pngs<=~{(len(rows)+max(audit_stride,1)-1)//max(audit_stride,1) if audit_stride else 0}",
        flush=True,
    )
    del retrieval_model, pipe, image_encoder, mapper, mapper_local, uncond
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
