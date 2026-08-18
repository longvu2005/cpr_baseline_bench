#!/usr/bin/env python3
"""Build IP-CIR dense captions, edited captions, layouts, and imagined proxies.

This is the dataset-boundary adapter for CPR. It intentionally never reads target
IDs, target images, positive labels, identity annotations, or GT boxes.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import re
import subprocess

import numpy as np
import sys
from pathlib import Path
from typing import Any, Sequence

import yaml

ROOT = Path(__file__).resolve().parents[3]
METHOD_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = METHOD_DIR / "config.yaml"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from benchmark_progress import PhaseTracker, progress_bar  # noqa: E402

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"


def load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError(f"Expected mapping: {path}")
    return data


def resolve_path(value: str) -> Path:
    p = Path(value)
    return (p if p.is_absolute() else ROOT / p).resolve()


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except Exception:
        return str(path.resolve())


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


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def stable_hash(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def query_text(row: dict[str, Any], index: int) -> str:
    for key in ("text", "relative_caption", "caption", "instruction"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise KeyError(f"Query row {index} has no usable composition instruction")


def gallery_image_path(row: dict[str, Any], index: int) -> Path:
    value = row.get("path")
    if not isinstance(value, str) or not value.strip():
        raise KeyError(f"Gallery row {index} has no path")
    p = resolve_path(value)
    if not p.is_file():
        raise FileNotFoundError(p)
    return p


def query_image_paths(queries: Sequence[dict[str, Any]], gallery: Sequence[dict[str, Any]]) -> list[Path]:
    by_id: dict[Any, int] = {}
    for i, row in enumerate(gallery):
        image_id = row.get("image_id")
        if image_id in by_id:
            raise ValueError(f"Duplicate gallery image_id {image_id!r}")
        by_id[image_id] = i
    paths = []
    for qi, q in enumerate(queries):
        image_id = q.get("image_id")
        if image_id not in by_id:
            raise ValueError(f"Query {qi} image_id={image_id!r} absent from gallery")
        gi = by_id[image_id]
        paths.append(gallery_image_path(gallery[gi], gi))
    return paths


def read_marker(cfg: dict[str, Any]) -> dict[str, Any]:
    path = resolve_path(str(cfg["migc"]["prepared_marker"]))
    if not path.is_file():
        raise FileNotFoundError(f"Missing P10 prepared marker: {rel(path)}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("author_source", {}).get("commit") != str(cfg["author_source"]["commit"]):
        raise RuntimeError("P10 marker/source commit mismatch")
    return data


def valid_prefix(path: Path, queries: Sequence[dict[str, Any]], fingerprint: str) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    try:
        rows = load_jsonl(path)
    except Exception:
        return []
    valid = []
    for qi, row in enumerate(rows):
        if qi >= len(queries):
            break
        if row.get("query_index") != qi or row.get("image_id") != queries[qi].get("image_id"):
            break
        if row.get("fingerprint") != fingerprint:
            break
        valid.append(row)
    return valid


def unload(*objects: Any) -> None:
    for obj in objects:
        del obj
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def caption_fingerprint(cfg: dict[str, Any], query_manifest: Path, marker: dict[str, Any]) -> str:
    rev = marker.get("assets", {}).get("captioner", {}).get("resolved_revision")
    return stable_hash({
        "schema": 4,
        "query_manifest": sha256_file(query_manifest),
        "captioner": cfg["captioner"],
        "resolved_revision": rev,
    })


def generate_dense_captions(
    cfg: dict[str, Any], queries: Sequence[dict[str, Any]], image_paths: Sequence[Path],
    query_manifest: Path, marker: dict[str, Any], force: bool,
) -> list[dict[str, Any]]:
    import torch
    from PIL import Image
    from transformers import Blip2ForConditionalGeneration, Blip2Processor

    out = resolve_path(str(cfg["cache"]["captions"]))
    fp = caption_fingerprint(cfg, query_manifest, marker)
    rows = [] if force else valid_prefix(out, queries, fp)
    start = len(rows)
    if start == len(queries):
        print(f"[skip] dense captions complete: {rel(out)}", flush=True)
        return rows

    c = cfg["captioner"]
    snapshot = resolve_path(str(c["local_snapshot"]))
    processor = Blip2Processor.from_pretrained(str(snapshot), local_files_only=True)
    # Device-map allows a 2-GPU runtime; no smaller model fallback is used.
    model = Blip2ForConditionalGeneration.from_pretrained(
        str(snapshot), local_files_only=True, torch_dtype=torch.float16,
        low_cpu_mem_usage=True, device_map="auto",
    ).eval()
    # Pixel inputs belong on the vision tower device when device_map="auto" shards BLIP-2.
    input_device = next(model.vision_model.parameters()).device
    count = int(c["captions_per_query"])

    for qi in progress_bar(range(start, len(queries)), desc="IP-CIR BLIP2-6.7B dense captions", total=len(queries)-start, unit="query"):
        with Image.open(image_paths[qi]) as image:
            inputs = processor(images=image.convert("RGB"), return_tensors="pt")
        prepared = {}
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor):
                if value.dtype.is_floating_point:
                    value = value.to(dtype=torch.float16)
                prepared[key] = value.to(input_device)
        torch.manual_seed(10_000 + qi)
        torch.cuda.manual_seed_all(10_000 + qi)
        # Match released LAVIS blip2_opt.generate defaults used by
        # dense_caption_generator.py: nucleus sampling, 5 beams, top_p=.9,
        # temperature=1, max_length=30, and 15 returned captions.
        generated = model.generate(
            **prepared,
            max_length=int(c["max_length"]),
            min_length=int(c["min_length"]),
            num_beams=int(c["num_beams"]),
            do_sample=bool(c["do_sample"]),
            top_p=float(c["top_p"]),
            temperature=float(c["temperature"]),
            repetition_penalty=float(c["repetition_penalty"]),
            length_penalty=float(c["length_penalty"]),
            num_return_sequences=count,
        )
        captions = [x.strip() for x in processor.batch_decode(generated, skip_special_tokens=True) if x.strip()]
        if not captions:
            raise RuntimeError(f"BLIP2 produced no caption at query {qi}")
        while len(captions) < count:
            captions.append(captions[-1])
        rows.append({
            "schema": 4, "query_index": qi, "image_id": queries[qi].get("image_id"),
            "captions": captions[:count], "fingerprint": fp,
        })
        write_jsonl(out, rows)
    unload(model, processor)
    return rows


EDIT_SYSTEM_PROMPT = """
I have an image. Given an instruction to edit the image, carefully generate a description of the edited image. I will put my image content beginning with "Image Content:". The instruction I provide will begin with "Instruction:". The edited description you generate should begin with "Edited Description:". You just generate one edited description only begin with "Edited Description:". The edited description needs to be as simple as possible and only reflects image content. Just one line.
""".strip()


def clean_edited_caption(text: str) -> str:
    value = text.strip().replace("\r", " ").replace("\n", " ").strip()
    value = re.sub(r"^Edited Description\s*:\s*", "", value, flags=re.IGNORECASE)
    value = re.sub(r"\s+", " ", value).strip()
    if not value:
        raise ValueError("empty edited description")
    return value


def parse_bbox(text: str) -> list[float]:
    nums = re.findall(r"[-+]?(?:\d*\.\d+|\d+\.?\d*)", text)
    if len(nums) < 4:
        raise ValueError(f"bad bbox: {text!r}")
    x0, y0, x1, y1 = [min(1.0, max(0.0, float(x))) for x in nums[:4]]
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"degenerate bbox: {text!r}")
    return [x0, y0, x1, y1]


def field(block: str, name: str) -> str:
    m = re.search(r"\[##\s*" + re.escape(name) + r"\s*:\s*(.*?)\s*##\]", block, flags=re.I | re.S)
    return "" if m is None else m.group(1).strip()


def parse_author_layout(text: str) -> dict[str, Any]:
    scene_m = re.search(r"\[##\s*Scene\s*:\s*(.*?)\s*##\]", text, flags=re.I | re.S)
    if scene_m is None:
        raise ValueError("author-format response has no [## Scene: ... ##]")
    scene = re.sub(r"\s+", " ", scene_m.group(1)).strip()
    starts = list(re.finditer(r"\[##\s*Label\s*:\s*(.*?)\s*##\]", text, flags=re.I | re.S))
    items: list[dict[str, Any]] = [{
        "label": "scene", "cate": "scene", "desc": scene,
        "bbox": [0.0, 0.0, 1.0, 1.0], "ref": "text", "is_scene": True,
    }]
    for i, m in enumerate(starts[:10]):
        end = starts[i+1].start() if i + 1 < len(starts) else len(text)
        block = text[m.start():end]
        label = re.sub(r"\s+", " ", m.group(1)).strip().lower()
        cate = re.sub(r"\s+", " ", field(block, "Cate")).strip().lower() or label
        desc = re.sub(r"\s+", " ", field(block, "Desc")).strip()
        bbox_text = field(block, "bbox")
        ref = field(block, "Ref").lower()
        if not label or not desc or not bbox_text:
            continue
        items.append({
            "label": label,
            "cate": cate,
            "desc": desc,
            "bbox": parse_bbox(bbox_text),
            "ref": "image" if "image" in ref else "text",
            "is_scene": False,
        })
    if len(items) == 1:
        raise ValueError("author-format response produced no controlled instances")
    if not any(x["ref"] == "image" for x in items if not x["is_scene"]):
        # The final user prompt explicitly marks the reference concept as Image.
        # If Qwen omitted Ref, anchor the first person-like instance, otherwise first instance.
        candidates = [x for x in items if not x["is_scene"]]
        candidate = next((x for x in candidates if "person" in x["cate"] or "person" in x["label"]), candidates[0])
        candidate["ref"] = "image"
    return {"scene_prompt": scene, "layout": items}


def qwen_fingerprint(cfg: dict[str, Any], query_manifest: Path, captions_path: Path, marker: dict[str, Any]) -> str:
    rev = marker.get("assets", {}).get("layout_llm", {}).get("resolved_revision")
    return stable_hash({
        "schema": 4,
        "query_manifest": sha256_file(query_manifest),
        "captions": sha256_file(captions_path),
        "layout_llm": cfg["layout_llm"],
        "resolved_revision": rev,
        "author_commit": cfg["author_source"]["commit"],
    })


def model_input_device(model):
    try:
        return model.get_input_embeddings().weight.device
    except Exception:
        return next(model.parameters()).device


def generate_target_captions_batch(model, tokenizer, original: Sequence[str], modification: str, max_new_tokens: int) -> list[str]:
    """Reproduce released reasoning_and_editing.py one caption at a time.

    The author code runs 15 independent Qwen calls, not one 15-sample batch.
    Keeping this sequential also avoids a large GPTQ KV-cache spike.
    """
    import torch
    decoded: list[str] = []
    device = model_input_device(model)
    for cap in original:
        messages = [
            {"role": "system", "content": EDIT_SYSTEM_PROMPT},
            {"role": "user", "content": "Image Content: a man adjusting a woman's tie.\nInstruction: has the woman and the man with the roles switched."},
            {"role": "assistant", "content": "Edited Description: a woman adjusting a man's tie."},
            {"role": "user", "content": f"Image Content: {cap}\nInstruction: {modification}\nEdited Description:"},
        ]
        rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        tokens = tokenizer([rendered], return_tensors="pt")
        tokens = {k: v.to(device) for k, v in tokens.items()}
        with torch.inference_mode():
            output_ids = model.generate(
                **tokens, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        in_len = tokens["input_ids"].shape[1]
        text = tokenizer.decode(output_ids[0, in_len:], skip_special_tokens=True)
        decoded.append(clean_edited_caption(text))
    return decoded


def generate_layout_one(model, tokenizer, prompt_cfg: dict[str, Any], concept: str, modification: str, max_new_tokens: int, retries: int) -> dict[str, Any]:
    import torch
    layout_cfg = prompt_cfg["layout"]
    messages: list[dict[str, str]] = [{"role": "system", "content": str(layout_cfg["system_prompt"])}]
    for example in layout_cfg.get("examples", []):
        if isinstance(example, dict) and "input" in example and "output" in example:
            messages.append({"role": "user", "content": str(example["input"])})
            messages.append({"role": "assistant", "content": str(example["output"])})
    messages.append({
        "role": "user",
        "content": f"Object: Label: {concept}, Reference: Image\nLayout Rule: {modification}",
    })
    last = ""
    last_error: Exception | None = None
    for attempt in range(retries):
        rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        tokens = tokenizer([rendered], return_tensors="pt")
        device = model_input_device(model)
        tokens = {k: v.to(device) for k, v in tokens.items()}
        with torch.inference_mode():
            ids = model.generate(
                **tokens, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        last = tokenizer.decode(ids[0, tokens["input_ids"].shape[1]:], skip_special_tokens=True)
        try:
            return parse_author_layout(last)
        except Exception as exc:
            last_error = exc
            messages.extend([
                {"role": "assistant", "content": last},
                {"role": "user", "content": "Repair the answer using exactly the [## Scene/Label/Cate/Desc/Size/From/bbox/Ref ##] format required by the system prompt. Do not omit bbox or Ref."},
            ])
    raise RuntimeError(f"Qwen layout parse failed after {retries} attempts: {last_error}; last={last[:700]!r}")


def generate_qwen_outputs(
    cfg: dict[str, Any], queries: Sequence[dict[str, Any]], captions: Sequence[dict[str, Any]],
    query_manifest: Path, marker: dict[str, Any], force: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    captions_path = resolve_path(str(cfg["cache"]["captions"]))
    fp = qwen_fingerprint(cfg, query_manifest, captions_path, marker)
    target_path = resolve_path(str(cfg["cache"]["target_captions"]))
    layout_path = resolve_path(str(cfg["cache"]["layouts"]))
    target_rows = [] if force else valid_prefix(target_path, queries, fp)
    layout_rows = [] if force else valid_prefix(layout_path, queries, fp)
    start = min(len(target_rows), len(layout_rows))
    target_rows = target_rows[:start]
    layout_rows = layout_rows[:start]
    if start == len(queries):
        print("[skip] Qwen edited captions/layouts complete", flush=True)
        return target_rows, layout_rows

    c = cfg["layout_llm"]
    snapshot = resolve_path(str(c["local_snapshot"]))
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(snapshot), local_files_only=True, trust_remote_code=True,
        device_map="auto", low_cpu_mem_usage=True,
    ).eval()

    source = resolve_path(str(cfg["author_source"]["local_checkout"]))
    prompt_file = source / str(c["prompt_file"])
    prompt_cfg = load_yaml(prompt_file)
    expected_n = int(c["target_captions_per_query"])
    if expected_n != int(cfg["captioner"]["captions_per_query"]):
        raise RuntimeError("Original and edited caption counts must match released LDRE pairing")

    for qi in progress_bar(range(start, len(queries)), desc="IP-CIR Qwen32B LDRE+layout", total=len(queries)-start, unit="query"):
        modification = query_text(queries[qi], qi)
        originals = list(captions[qi]["captions"])
        targets = generate_target_captions_batch(
            model, tokenizer, originals, modification, int(c["edit_max_new_tokens"])
        )
        if len(targets) != expected_n:
            raise RuntimeError(f"Expected {expected_n} edited captions, got {len(targets)}")
        # Released generate_layout.py uses one dense concept for non-CIRCO paths.
        layout = generate_layout_one(
            model, tokenizer, prompt_cfg, originals[0], modification,
            int(c["layout_max_new_tokens"]), int(c["parse_retries"]),
        )
        target_rows.append({
            "schema": 4, "query_index": qi, "image_id": queries[qi].get("image_id"),
            "captions": targets, "fingerprint": fp,
        })
        layout_rows.append({
            "schema": 4, "query_index": qi, "image_id": queries[qi].get("image_id"),
            "modification": modification, **layout, "fingerprint": fp,
        })
        write_jsonl(target_path, target_rows)
        write_jsonl(layout_path, layout_rows)
    unload(model, tokenizer)
    return target_rows, layout_rows


def build_jobs(
    cfg: dict[str, Any], queries: Sequence[dict[str, Any]], image_paths: Sequence[Path],
    captions: Sequence[dict[str, Any]], targets: Sequence[dict[str, Any]], layouts: Sequence[dict[str, Any]],
) -> Path:
    out = resolve_path(str(cfg["cache"]["proxy_jobs"]))
    rows = []
    for qi in range(len(queries)):
        rows.append({
            "query_index": qi,
            "image_id": queries[qi].get("image_id"),
            "query_image": rel(image_paths[qi]),
            "original_captions": list(captions[qi]["captions"]),
            "target_captions": list(targets[qi]["captions"]),
            "scene_prompt": layouts[qi]["scene_prompt"],
            "layout": layouts[qi]["layout"],
        })
    write_jsonl(out, rows)
    return out


def validate_manifest(cfg: dict[str, Any], queries: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    path = resolve_path(str(cfg["proxy"]["manifest"]))
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = load_jsonl(path)
    if len(rows) != len(queries):
        raise ValueError(f"Proxy manifest rows={len(rows)}, expected={len(queries)}")
    count = int(cfg["proxy"]["count_per_query"])
    dim = int(cfg["retrieval"]["projection_dim"])
    feature_path = resolve_path(str(cfg["retrieval"]["proxy_features"]))
    if not feature_path.is_file():
        raise FileNotFoundError(f"Missing streamed proxy features: {feature_path}")
    features = np.load(feature_path, mmap_mode="r", allow_pickle=False)
    expected = (len(queries), count, dim)
    if features.shape != expected or features.dtype != np.float32:
        raise ValueError(f"Proxy features shape/dtype={features.shape}/{features.dtype}, expected={expected}/float32")
    for qi, row in enumerate(rows):
        if row.get("query_index") != qi or row.get("image_id") != queries[qi].get("image_id"):
            raise ValueError(f"Proxy manifest alignment error at row {qi}")
        if int(row.get("proxy_count", -1)) != count:
            raise ValueError(f"Proxy count error at query {qi}")
        if int(row.get("proxy_feature_index", -1)) != qi:
            raise ValueError(f"Proxy feature index error at query {qi}")
        if row.get("storage_mode") != "stream_generate_encode_discard":
            raise ValueError(f"Unexpected proxy storage mode at query {qi}: {row.get('storage_mode')!r}")
    # Full validation is only ~46 MB and catches interrupted rows before retrieval.
    if not np.isfinite(np.asarray(features)).all():
        raise RuntimeError("Proxy feature cache contains NaN/Inf or incomplete rows")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--stage", choices=["captions", "qwen", "proxies", "all"], default="all")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config_path = resolve_path(args.config)
    cfg = load_yaml(config_path)
    marker = read_marker(cfg)
    query_manifest = resolve_path(str(cfg["data"]["query_manifest"]))
    gallery_manifest = resolve_path(str(cfg["data"]["gallery_manifest"]))
    queries = load_jsonl(query_manifest)
    gallery = load_jsonl(gallery_manifest)
    images = query_image_paths(queries, gallery)

    stages = [args.stage] if args.stage != "all" else ["captions", "qwen", "proxies"]
    tracker = PhaseTracker("imagine_seek_proxy", total=len(stages))
    captions = targets = layouts = None
    for stage in stages:
        if stage == "captions":
            with tracker.phase("Generate/resume 15 BLIP2-OPT6.7B dense captions per query"):
                captions = generate_dense_captions(cfg, queries, images, query_manifest, marker, args.force)
        elif stage == "qwen":
            with tracker.phase("Generate/resume Qwen32B edited captions and released-format layouts"):
                if captions is None:
                    captions = load_jsonl(resolve_path(str(cfg["cache"]["captions"])))
                targets, layouts = generate_qwen_outputs(cfg, queries, captions, query_manifest, marker, args.force)
        elif stage == "proxies":
            with tracker.phase("Generate -> CLIP-L encode -> discard five MIGC+ELITE proxies per query"):
                if captions is None:
                    captions = load_jsonl(resolve_path(str(cfg["cache"]["captions"])))
                if targets is None:
                    targets = load_jsonl(resolve_path(str(cfg["cache"]["target_captions"])))
                if layouts is None:
                    layouts = load_jsonl(resolve_path(str(cfg["cache"]["layouts"])))
                build_jobs(cfg, queries, images, captions, targets, layouts)
                worker = METHOD_DIR / "official_proxy_worker.py"
                subprocess.run([sys.executable, str(worker), "--config", str(config_path), "--stage", "generate"], cwd=str(ROOT), check=True)
                validate_manifest(cfg, queries)
        else:
            raise AssertionError(stage)
    tracker.finish()


if __name__ == "__main__":
    main()
