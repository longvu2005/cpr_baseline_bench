#!/usr/bin/env python3
"""P11 WISER: paper-faithful full-scene CPR adapter."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
METHOD_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = METHOD_DIR / "config.yaml"


def _bootstrap() -> None:
    if os.environ.get("WISER_ISOLATED") == "1":
        return
    import yaml
    config_arg = None
    argv = sys.argv[1:]
    for i, token in enumerate(argv):
        if token == "--config" and i + 1 < len(argv):
            config_arg = argv[i + 1]
            break
    config_path = Path(config_arg) if config_arg else DEFAULT_CONFIG
    if not config_path.is_absolute():
        config_path = (ROOT / config_path).resolve()
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    py = Path(str(cfg["isolated_env"]["python"]))
    if not py.is_absolute():
        py = (ROOT / py).resolve()
    if not py.is_file():
        raise SystemExit(f"WISER isolated environment missing: {py}. Run checkpoint preparation first.")
    env = os.environ.copy()
    env["WISER_ISOLATED"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    os.execve(str(py), [str(py), str(Path(__file__).resolve()), *sys.argv[1:]], env)


_bootstrap()

import gc  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import subprocess  # noqa: E402
from typing import Any, Sequence  # noqa: E402

import numpy as np  # noqa: E402
import open_clip  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import yaml  # noqa: E402
from PIL import Image  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration  # noqa: E402
from qwen_vl_utils import process_vision_info  # noqa: E402
from openai import OpenAI  # noqa: E402

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from benchmark_data import ensure_gallery_layout  # noqa: E402
from benchmark_progress import PhaseTracker, progress_bar  # noqa: E402

METHOD_ID = "wiser"
ADAPTER_VERSION = "2026-08-18-v1-wiser-cpr-full-gallery"
CACHE_SCHEMA = 1


def load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected YAML mapping: {path}")
    return value


def resolve(value: str) -> Path:
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
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"{path}:{lineno}: expected object")
            rows.append(value)
    return rows


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return value if isinstance(value, dict) else None


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_name(value: Any) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))
    return text[:120] or "query"


def gallery_path(row: dict[str, Any], gi: int) -> Path:
    value = row.get("path")
    if not isinstance(value, str) or not value.strip():
        raise KeyError(f"Gallery row {gi}: missing path")
    p = resolve(value)
    if not p.is_file():
        raise FileNotFoundError(p)
    return p


def build_gallery_index(gallery: Sequence[dict[str, Any]]) -> dict[Any, int]:
    out = {}
    for gi, row in enumerate(gallery):
        image_id = row.get("image_id")
        if image_id in out:
            raise ValueError(f"Duplicate gallery image_id: {image_id!r}")
        out[image_id] = gi
    return out


def validate_queries(queries: Sequence[dict[str, Any]], gindex: dict[Any, int], text_field: str) -> None:
    for qi, q in enumerate(queries):
        if q.get("image_id") not in gindex:
            raise ValueError(f"Query {qi}: source image missing from gallery")
        text = q.get(text_field)
        if not isinstance(text, str) or not text.strip():
            raise KeyError(f"Query {qi}: no usable {text_field!r}")


def validate_prepared(cfg: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    marker_path = resolve(str(cfg["checkpoint"]["prepared_marker"]))
    marker = read_json(marker_path)
    if marker is None:
        raise FileNotFoundError(f"Missing WISER prepared marker: {marker_path}")
    checkout = resolve(str(cfg["author_source"]["local_checkout"]))
    actual = subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True).strip()
    expected = str(cfg["author_source"]["commit"])
    if actual != expected or marker.get("author_source", {}).get("commit") != expected:
        raise RuntimeError(f"Pinned WISER source mismatch: expected {expected}, got {actual}")
    dirty = subprocess.check_output(
        ["git", "-C", str(checkout), "status", "--porcelain", "--untracked-files=no"], text=True
    ).strip()
    if dirty:
        raise RuntimeError(f"Pinned WISER source has tracked modifications:\n{dirty}")
    return checkout, marker


def import_official_editor(checkout: Path):
    src = checkout / "src"
    sys.path.insert(0, str(src))
    import prompts as wiser_prompts
    from bagel_inference import BagelImageEditor
    return wiser_prompts, BagelImageEditor


class ImagePathDataset(Dataset):
    def __init__(self, paths: Sequence[Path], transform):
        self.paths = list(paths)
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index: int):
        with Image.open(self.paths[index]) as image:
            return self.transform(image.convert("RGB"))


def cache_meta(path: Path, expected: dict[str, Any]) -> bool:
    return read_json(path) == expected


@torch.no_grad()
def encode_images(
    paths: Sequence[Path],
    model,
    transform,
    device: torch.device,
    batch_size: int,
    workers: int,
    output: Path | None = None,
    meta_path: Path | None = None,
    meta: dict[str, Any] | None = None,
) -> np.ndarray:
    dim = int(model.visual.output_dim)
    if (
        output is not None
        and meta_path is not None
        and meta is not None
        and output.is_file()
        and cache_meta(meta_path, meta)
    ):
        arr = np.load(output, mmap_mode="r", allow_pickle=False)
        if arr.shape == (len(paths), dim) and arr.dtype == np.float32:
            sample = np.asarray(arr[: min(len(arr), 128)])
            if np.isfinite(sample).all():
                print(f"[cache] {rel(output)}")
                return arr

    loader = DataLoader(
        ImagePathDataset(paths, transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=(device.type == "cuda"),
    )

    if output is None:
        chunks = []
        for images in progress_bar(loader, desc="WISER encode images", total=len(loader), unit="batch"):
            images = images.to(device, non_blocking=True)
            feat = F.normalize(model.encode_image(images).float(), dim=-1)
            chunks.append(feat.cpu().numpy())
        return np.concatenate(chunks, axis=0).astype(np.float32, copy=False)

    output.parent.mkdir(parents=True, exist_ok=True)
    mmap = np.lib.format.open_memmap(output, mode="w+", dtype=np.float32, shape=(len(paths), dim))
    cursor = 0
    for images in progress_bar(loader, desc="WISER encode gallery", total=len(loader), unit="batch"):
        images = images.to(device, non_blocking=True)
        feat = F.normalize(model.encode_image(images).float(), dim=-1)
        n = feat.shape[0]
        mmap[cursor:cursor+n] = feat.cpu().numpy()
        cursor += n
    mmap.flush()
    if meta_path is not None and meta is not None:
        write_json(meta_path, meta)
    return np.load(output, mmap_mode="r", allow_pickle=False)


@torch.no_grad()
def encode_texts(texts: Sequence[str], model, tokenizer, device: torch.device, batch_size: int) -> np.ndarray:
    chunks = []
    for start in progress_bar(
        range(0, len(texts), batch_size),
        desc="WISER encode edited captions",
        total=(len(texts) + batch_size - 1) // batch_size,
        unit="batch",
    ):
        batch = list(texts[start:start+batch_size])
        tokens = tokenizer(batch, context_length=77).to(device)
        feat = F.normalize(model.encode_text(tokens).float(), dim=-1)
        chunks.append(feat.cpu().numpy())
    return np.concatenate(chunks, axis=0).astype(np.float32, copy=False)


def full_rank(features: np.ndarray, gallery: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    q = torch.from_numpy(np.asarray(features)).to(device=device, dtype=torch.float32)
    g = torch.from_numpy(np.asarray(gallery)).to(device=device, dtype=torch.float32)
    out = np.empty((len(features), len(gallery)), dtype=np.int32)
    for start in progress_bar(
        range(0, len(features), batch_size),
        desc="WISER base retrieval",
        total=(len(features) + batch_size - 1) // batch_size,
        unit="batch",
    ):
        end = min(start + batch_size, len(features))
        similarity = q[start:end] @ g.T
        out[start:end] = torch.argsort(similarity, dim=-1, descending=True).cpu().numpy().astype(np.int32)
    return out


def read_caption_map(path: Path, gallery: Sequence[dict[str, Any]]) -> dict[Any, str]:
    rows = load_jsonl(path)
    if len(rows) != len(gallery):
        raise ValueError(f"Caption count {len(rows)} != gallery count {len(gallery)}")
    mapping = {}
    for expected, row in zip(gallery, rows):
        if row.get("image_id") != expected.get("image_id"):
            raise ValueError("Gallery caption order/id mismatch")
        caption = row.get("caption")
        if not isinstance(caption, str):
            raise TypeError("Invalid gallery caption")
        mapping[row["image_id"]] = caption
    return mapping


def parse_edited_description(response: str, fallback: str) -> str:
    for line in str(response).splitlines():
        if line.strip().startswith("Edited Description:"):
            value = line.split(":", 1)[1].strip()
            return value if value else fallback
    return fallback


def generation_meta(cfg: dict[str, Any], query_manifest: Path, caption_meta: Path) -> dict[str, Any]:
    marker = read_json(resolve(str(cfg["checkpoint"]["prepared_marker"]))) or {}
    return {
        "schema": CACHE_SCHEMA,
        "adapter": ADAPTER_VERSION,
        "query_manifest_sha256": sha256(query_manifest),
        "caption_meta_sha256": sha256(caption_meta),
        "bagel_revision": marker.get("assets", {}).get("bagel", {}).get("resolved_revision"),
        "modifier_prompt": cfg["wiser"]["modifier_prompt"],
    }


def load_generation_cache(
    path: Path,
    meta_path: Path,
    expected_meta: dict[str, Any],
    queries: Sequence[dict[str, Any]],
) -> list[dict[str, Any]] | None:
    if not path.is_file() or read_json(meta_path) != expected_meta:
        return None
    rows = load_jsonl(path)
    if len(rows) != len(queries):
        return None
    for qi, (row, q) in enumerate(zip(rows, queries)):
        if row.get("query_index") != qi or row.get("image_id") != q.get("image_id"):
            return None
        p = resolve(str(row.get("edited_image", "")))
        if not p.is_file():
            return None
    print(f"[cache] {rel(path)}")
    return rows


def generate_queries(
    cfg: dict[str, Any],
    queries: Sequence[dict[str, Any]],
    gallery: Sequence[dict[str, Any]],
    gindex: dict[Any, int],
    captions: dict[Any, str],
    bagel,
    modifier_prompt: str,
    query_manifest: Path,
) -> list[dict[str, Any]]:
    path = resolve(str(cfg["cache"]["query_generation"]))
    meta_path = resolve(str(cfg["cache"]["query_generation_meta"]))
    expected_meta = generation_meta(cfg, query_manifest, resolve(str(cfg["cache"]["captions_meta"])))
    cached = load_generation_cache(path, meta_path, expected_meta, queries)
    if cached is not None:
        return cached

    out_dir = resolve(str(cfg["cache"]["edited_images_dir"]))
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    text_field = str(cfg["adaptation"]["query_text_field"])
    for qi, q in enumerate(
        progress_bar(queries, desc="WISER BAGEL edit caption+image", total=len(queries), unit="query")
    ):
        source_id = q["image_id"]
        instruction = str(q[text_field])
        source_caption = captions[source_id]
        prompt = (
            modifier_prompt
            + "\nImage Content: "
            + source_caption
            + "\nInstruction: "
            + instruction
        )
        modified = parse_edited_description(bagel.generate_caption(prompt), instruction)

        source_path = gallery_path(gallery[gindex[source_id]], gindex[source_id])
        qid = q.get("query_id", qi)
        edited_path = out_dir / f"{qi:05d}_{safe_name(qid)}.png"
        if not edited_path.is_file():
            edited = bagel.edit_image_no_think(str(source_path), instruction)["image"]
            edited.save(edited_path)
        rows.append({
            "query_index": qi,
            "query_id": qid,
            "image_id": source_id,
            "instruction": instruction,
            "source_caption": source_caption,
            "modified_caption": modified,
            "edited_image": rel(edited_path),
        })

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp, path)
    write_json(meta_path, expected_meta)
    return rows


class WiserVerifier:
    """Released Qwen2.5-VL yes/no confidence logic, without source enum bugs."""

    def __init__(self, cfg: dict[str, Any], device: torch.device):
        spec = cfg["models"]["verifier"]
        model_path = resolve(str(spec["local_dir"]))
        self.device = device
        self.processor = AutoProcessor.from_pretrained(
            str(model_path),
            local_files_only=True,
            min_pixels=int(spec["min_pixels"]),
            max_pixels=int(spec["max_pixels"]),
        )
        self.processor.tokenizer.padding_side = "left"
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            str(model_path),
            local_files_only=True,
            torch_dtype=torch.float16,
            device_map={"": str(device)},
            attn_implementation="sdpa",
        ).eval()
        self.y_id = self.processor.tokenizer.encode("yes", add_special_tokens=False)[0]
        self.n_id = self.processor.tokenizer.encode("no", add_special_tokens=False)[0]

    @staticmethod
    def load_image(path: Path, max_size: int = 1024) -> Image.Image:
        image = Image.open(path).convert("RGB")
        w, h = image.size
        if w > max_size or h > max_size:
            if w > h:
                nw, nh = max_size, int(h * max_size / w)
            else:
                nh, nw = max_size, int(w * max_size / h)
            image = image.resize((nw, nh), Image.LANCZOS)
        return image

    @torch.no_grad()
    def score(self, reference: Path, instruction: str, candidate: Path) -> float:
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": self.load_image(reference)},
                {"type": "image", "image": self.load_image(candidate)},
                {
                    "type": "text",
                    "text": (
                        "You are a strict visual verifier. Output exactly one token: yes or no (lowercase)."
                        "Do not add punctuation or explanations.\n"
                        "    Reference image: Picture1\n"
                        "    Candidate image: Picture2\n"
                        f"    Instruction:{instruction}\n"
                        "    Decide if the candidate image matches the result of applying the instruction to the reference image.\n"
                        "    Return yes if all required elements implied by the instruction are satisfied (like counts, categories, attributes, spatial relations). If any required element is missing or contradicted, answer no.\n"
                        "    Answer:"
                    ),
                },
            ],
        }]
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            add_vision_id=True,
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=text,
            images=image_inputs,
            videos=video_inputs,
            padding=False,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in inputs.items()}
        outputs = self.model(**inputs)
        logits = outputs.logits[:, -1, :]
        yn = torch.stack([logits[:, self.n_id], logits[:, self.y_id]], dim=1)
        confidence = F.softmax(yn, dim=1)[:, 1].item()
        del outputs, logits, yn, inputs
        return float(confidence)


def verify_cache_path(cfg: dict[str, Any], loop: int) -> Path:
    return resolve(str(cfg["cache"]["verifier_dir"])) / f"loop_{loop}.jsonl"


def verify_candidates(
    cfg: dict[str, Any],
    verifier: WiserVerifier,
    queries: Sequence[dict[str, Any]],
    gallery: Sequence[dict[str, Any]],
    gindex: dict[Any, int],
    t_rank: np.ndarray,
    i_rank: np.ndarray,
    loop: int,
) -> tuple[
    list[list[tuple[int, int, float]]],
    list[list[tuple[int, int, float]]],
    list[bool],
    list[bool],
]:
    topk = int(cfg["wiser"]["topk_per_path"])
    threshold = float(cfg["wiser"]["confidence_threshold"])
    cache_path = verify_cache_path(cfg, loop)
    existing = {}
    if cache_path.is_file():
        for row in load_jsonl(cache_path):
            existing[int(row["query_index"])] = row
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    all_t, all_i, t_uncertain, i_uncertain = [], [], [], []
    text_field = str(cfg["adaptation"]["query_text_field"])
    completed_rows = []

    for qi, q in enumerate(
        progress_bar(queries, desc=f"WISER verifier loop {loop}", total=len(queries), unit="query")
    ):
        k_t = min(topk, t_rank.shape[1])
        k_i = min(topk, i_rank.shape[1])
        t_ids = [gallery[int(x)]["image_id"] for x in t_rank[qi, :k_t]]
        i_ids = [gallery[int(x)]["image_id"] for x in i_rank[qi, :k_i]]
        cached = existing.get(qi)

        if cached and cached.get("t_ids") == t_ids and cached.get("i_ids") == i_ids:
            t_scores = [
                (j + 1, int(t_rank[qi, j]), float(cached["t_conf"][j]))
                for j in range(k_t)
            ]
            i_scores = [
                (j + 1, int(i_rank[qi, j]), float(cached["i_conf"][j]))
                for j in range(k_i)
            ]
        else:
            source_gi = gindex[q["image_id"]]
            reference = gallery_path(gallery[source_gi], source_gi)
            instruction = str(q[text_field])
            t_scores, i_scores = [], []
            for j in range(k_t):
                gi = int(t_rank[qi, j])
                conf = verifier.score(reference, instruction, gallery_path(gallery[gi], gi))
                t_scores.append((j + 1, gi, conf))
            for j in range(k_i):
                gi = int(i_rank[qi, j])
                conf = verifier.score(reference, instruction, gallery_path(gallery[gi], gi))
                i_scores.append((j + 1, gi, conf))

        bt = max((x[2] for x in t_scores), default=0.0)
        bi = max((x[2] for x in i_scores), default=0.0)
        all_t.append(t_scores)
        all_i.append(i_scores)
        t_uncertain.append(not (bt > threshold))
        i_uncertain.append(not (bi > threshold))
        completed_rows.append({
            "query_index": qi,
            "t_ids": t_ids,
            "i_ids": i_ids,
            "t_conf": [x[2] for x in t_scores],
            "i_conf": [x[2] for x in i_scores],
            "best_t": bt,
            "best_i": bi,
        })

        tmp = cache_path.with_suffix(cache_path.suffix + ".part")
        with tmp.open("w", encoding="utf-8") as f:
            for row in completed_rows:
                f.write(json.dumps(row) + "\n")
        os.replace(tmp, cache_path)

    return all_t, all_i, t_uncertain, i_uncertain


def extract_suggestion(text: str) -> str:
    lines = str(text).splitlines()
    total, started = "", False
    for line in lines:
        if started:
            total += line
        elif "suggestion:" in line.lower():
            started = True
            total += line.split(":", 1)[1].strip()
    return total or str(text).strip()


def t2i_refine_prompt(image_caption: str, instruction: str, top_captions: Sequence[str]) -> str:
    return f"""
        Assume you are an experienced composed image retrieval expert, skilled at precisely generating new image descriptions based on a reference image's description and the user's modification instructions.
        You excel at creating modified descriptions that can retrieve images matching the user's requested changes through vector retrieval.
        Your task is to help improve the effectiveness of compositional image retrieval by generating precise modification suggestions that will assist another large language model (LLM) in producing a better image description.
        Please note that this LLM has received the reference image's description and the user's modification instructions, and already generated a modified description.
        Moreover, a retrieval has been performed based on this modified description. Thus your task is to analyze the last retrieval result and provide modification suggestions and please follow the below steps to finish this task.

        Step 1: Identifying Modifications
        Your first task is to identify the modifications and generate corresponding modification phrases.
        Specifically, here is the description of the reference image: "{image_caption}." Here are the user's modification requests: "{instruction}"
        By deeply understanding the image description and the user's modifications, please generate the following two types of modification phrases:
        1. If the modification involves changing the characteristics of an entity in the original reference image, please specify the changes,
        2. If the modification involves adding or deleting an entity, please specify the additions or deletions.
        Please note that the user's modifications may lack a subject; in such cases, infer and supply the object corresponding to the modification.
        Only include modifications explicitly mentioned by the user. If a certain type of modification is not present, you do not need to provide it and should avoid generating unspecified content.

        Step 2: Analyzing the Retrieved Image
        Compare the modification phrases identified in Step 1 with the description of the retrieved image : "{list(top_captions)}". Note that this retrieval is performed with the modified description generated by another LLM, which has been mentioned above.
        Determine if the retrieved image meets the user's modification instructions.
        If it matches after excluding subjective modifications (e.g., "casual," "relaxed"), respond with: "Good retrieval, no more loops needed."
        If there are unmet modification phrases, proceed to Step 3.

        Step 3: Providing Modification Suggestions
        For any unmet modifications identified in Step 2, suggest targeted changes to help the LLM regenerate an improved modified description. Keep suggestions concise and specific to ensure they effectively guide the LLM.

        **Output format:**
        "Suggestion: <concise, actionable suggestion in 10-20 words>"
        """


def i2i_refine_prompt(image_caption: str, instruction: str, top_captions: Sequence[str]) -> str:
    return f"""
        Assume you are an experienced composed image retrieval expert, skilled at precisely generating new image based on a reference image's description and the user's modification instructions.
        You excel at creating modified images that can retrieve images matching the user's requested changes through vector retrieval.
        Your task is to help improve the effectiveness of compositional image retrieval by generating precise modification suggestions that will assist another multimodal large language model (MLLM) in producing a better image.
        Please note that this MLLM has received the reference image's description and the user's modification instructions, and already generated a modified image.
        Moreover, a retrieval has been performed based on this modified image. Thus your task is to analyze the last retrieval result and provide modification suggestions and please follow the below steps to finish this task.

        Step 1: Identifying Modifications
        Your first task is to identify the modifications and generate corresponding modification phrases.
        Specifically, here is the description of the reference image: "{image_caption}." Here are the user's modification requests: "{instruction}"
        By deeply understanding the image description and the user's modifications, please generate the following two types of modification phrases:
        1. If the modification involves changing the characteristics of an entity in the original reference image, please specify the changes,
        2. If the modification involves adding or deleting an entity, please specify the additions or deletions.
        Please note that the user's modifications may lack a subject; in such cases, infer and supply the object corresponding to the modification.
        Only include modifications explicitly mentioned by the user. If a certain type of modification is not present, you do not need to provide it and should avoid generating unspecified content.

        Step 2: Analyzing the Retrieved Image
        Compare the modification phrases identified in Step 1 with the description of the retrieved image : "{list(top_captions)}". Note that this retrieval is performed with the modified image generated by another MLLM, which has been mentioned above.
        Determine if the retrieved image meets the user's modification instructions.
        If it matches after excluding subjective modifications (e.g., "casual," "relaxed"), respond with: "Good retrieval, no more loops needed."
        If there are unmet modification phrases, proceed to Step 3.

        Step 3: Providing Modification Suggestions
        For any unmet modifications identified in Step 2, suggest targeted changes to help the MLLM regenerate an improved modified image. Keep suggestions concise and specific to ensure they effectively guide the MLLM.

        **Output format:**
        "Suggestion: <concise, actionable suggestion in 10-20 words>"
        """


def get_suggestions(
    cfg: dict[str, Any],
    client: OpenAI,
    queries: Sequence[dict[str, Any]],
    generation: Sequence[dict[str, Any]],
    gallery: Sequence[dict[str, Any]],
    captions: dict[Any, str],
    t_rank: np.ndarray,
    i_rank: np.ndarray,
    t_uncertain: Sequence[bool],
    i_uncertain: Sequence[bool],
    loop: int,
) -> tuple[list[str], list[str]]:
    path = resolve(str(cfg["cache"]["suggestions_dir"])) / f"loop_{loop}.jsonl"
    existing = {int(r["query_index"]): r for r in load_jsonl(path)} if path.is_file() else {}
    topk = int(cfg["wiser"]["topk_per_path"])
    model = str(cfg["models"]["refiner"]["model"])
    t_out, i_out, rows = [], [], []

    for qi, _q in enumerate(
        progress_bar(queries, desc=f"WISER refine suggestions {loop}", total=len(queries), unit="query")
    ):
        cached = existing.get(qi)
        if cached is not None:
            ts, ins = str(cached["t2i"]), str(cached["i2i"])
        else:
            source_caption = generation[qi]["source_caption"]
            instruction = generation[qi]["instruction"]
            t_caps = [
                captions[gallery[int(gi)]["image_id"]]
                for gi in t_rank[qi, : min(topk, t_rank.shape[1])]
            ]
            i_caps = [
                captions[gallery[int(gi)]["image_id"]]
                for gi in i_rank[qi, : min(topk, i_rank.shape[1])]
            ]
            if t_uncertain[qi]:
                ts = (
                    client.chat.completions.create(
                        model=model,
                        messages=[{
                            "role": "user",
                            "content": t2i_refine_prompt(source_caption, instruction, t_caps),
                        }],
                    ).choices[0].message.content
                    or ""
                )
            else:
                ts = "Good retrieval, no more loops needed"

            if i_uncertain[qi]:
                ins = (
                    client.chat.completions.create(
                        model=model,
                        messages=[{
                            "role": "user",
                            "content": i2i_refine_prompt(source_caption, instruction, i_caps),
                        }],
                    ).choices[0].message.content
                    or ""
                )
            else:
                ins = "Good retrieval, no more loops needed"

        t_out.append(ts)
        i_out.append(ins)
        rows.append({"query_index": qi, "t2i": ts, "i2i": ins})
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".part")
        with tmp.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        os.replace(tmp, path)

    return t_out, i_out


def refine_queries(
    cfg: dict[str, Any],
    queries: Sequence[dict[str, Any]],
    generation: Sequence[dict[str, Any]],
    gallery: Sequence[dict[str, Any]],
    gindex: dict[Any, int],
    bagel,
    modifier_prompt: str,
    t_suggestions: Sequence[str],
    i_suggestions: Sequence[str],
    t_uncertain: Sequence[bool],
    i_uncertain: Sequence[bool],
    loop: int,
) -> tuple[list[str], list[Path]]:
    out_dir = resolve(str(cfg["cache"]["refined_images_dir"])) / f"loop_{loop}"
    out_dir.mkdir(parents=True, exist_ok=True)
    captions_out, images_out = [], []

    for qi, q in enumerate(
        progress_bar(queries, desc=f"WISER BAGEL refine {loop}", total=len(queries), unit="query")
    ):
        last_caption = str(generation[qi]["modified_caption"])
        last_image = resolve(str(generation[qi]["edited_image"]))
        source_caption = str(generation[qi]["source_caption"])
        instruction = str(generation[qi]["instruction"])

        if t_uncertain[qi]:
            suggestion = extract_suggestion(t_suggestions[qi]).strip(".?,\"' ")
            combined = f"{instruction.strip('.?, ')} and {suggestion}."
            prompt = (
                f"{modifier_prompt}\n"
                f"Image Content: {source_caption}.\n"
                f"Instruction: {combined}."
            )
            new_caption = parse_edited_description(bagel.generate_caption(prompt), last_caption)
        else:
            new_caption = last_caption
        captions_out.append(new_caption)

        if i_uncertain[qi]:
            suggestion = extract_suggestion(i_suggestions[qi]).strip(".?,\"' ")
            combined = f"{instruction.strip('.?, ')} and {suggestion}."
            source_gi = gindex[q["image_id"]]
            source_path = gallery_path(gallery[source_gi], source_gi)
            target = out_dir / f"{qi:05d}_{safe_name(q.get('query_id', qi))}.png"
            if not target.is_file():
                bagel.edit_image_no_think(str(source_path), combined)["image"].save(target)
            images_out.append(target)
        else:
            images_out.append(last_image)

    return captions_out, images_out


def base_completion_order(t_row: np.ndarray, i_row: np.ndarray) -> list[int]:
    n = len(t_row)
    tr = np.empty(n, dtype=np.int32)
    ir = np.empty(n, dtype=np.int32)
    tr[t_row] = np.arange(n, dtype=np.int32)
    ir[i_row] = np.arange(n, dtype=np.int32)
    return sorted(
        range(n),
        key=lambda gi: (
            min(int(tr[gi]), int(ir[gi])),
            int(tr[gi]) + int(ir[gi]),
            int(tr[gi]),
            gi,
        ),
    )


def wiser_candidate_order(
    t_candidates: Sequence[tuple[int, int, float]],
    i_candidates: Sequence[tuple[int, int, float]],
    gallery: Sequence[dict[str, Any]],
) -> list[int]:
    tmap = {gi: conf for _, gi, conf in t_candidates}
    imap = {gi: conf for _, gi, conf in i_candidates}
    union = set(tmap) | set(imap)
    return sorted(
        union,
        key=lambda gi: (
            -(tmap.get(gi, 0.0) + imap.get(gi, 0.0)),
            -max(tmap.get(gi, 0.0), imap.get(gi, 0.0)),
            -tmap.get(gi, 0.0),
            str(gallery[gi]["image_id"]),
        ),
    )


def write_full_scores(
    cfg: dict[str, Any],
    gallery: Sequence[dict[str, Any]],
    t_rank: np.ndarray,
    i_rank: np.ndarray,
    t_candidates,
    i_candidates,
) -> Path:
    out_dir = resolve(str(cfg["output"]["dir"]))
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "scores.npy"
    scores = np.lib.format.open_memmap(
        path,
        mode="w+",
        dtype=np.float32,
        shape=(len(t_rank), len(gallery)),
    )
    n = len(gallery)

    for qi in progress_bar(
        range(len(t_rank)),
        desc="WISER complete full-gallery ranking",
        total=len(t_rank),
        unit="query",
    ):
        verified = wiser_candidate_order(t_candidates[qi], i_candidates[qi], gallery)
        seen = set(verified)
        remainder = [gi for gi in base_completion_order(t_rank[qi], i_rank[qi]) if gi not in seen]
        order = verified + remainder
        if len(order) != n or len(set(order)) != n:
            raise RuntimeError(f"Invalid complete ranking for query {qi}")
        row = np.empty(n, dtype=np.float32)
        row[np.asarray(order, dtype=np.int64)] = np.arange(n, 0, -1, dtype=np.float32)
        scores[qi] = row

    scores.flush()
    for start in range(0, len(t_rank), 128):
        if not np.isfinite(np.asarray(scores[start:start+128])).all():
            raise RuntimeError("Non-finite WISER scores")
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()
    config_path = resolve(args.config)
    cfg = load_yaml(config_path)
    tracker = PhaseTracker(METHOD_ID, total=9)

    with tracker.phase("Validate gallery, manifests, pinned source, and no-label adapter boundary"):
        ensure_gallery_layout(ROOT, repair=True)
        checkout, _marker = validate_prepared(cfg)
        gallery_manifest = resolve(str(cfg["data"]["gallery_manifest"]))
        query_manifest = resolve(str(cfg["data"]["query_manifest"]))
        gallery = load_jsonl(gallery_manifest)
        queries = load_jsonl(query_manifest)
        gindex = build_gallery_index(gallery)
        validate_queries(queries, gindex, str(cfg["adaptation"]["query_text_field"]))
        device = torch.device(str(cfg["runtime"]["device"]))
        if not torch.cuda.is_available():
            raise RuntimeError("WISER requires CUDA for paper-faithful BAGEL/Qwen inference")
        tracker.log(
            f"gallery={len(gallery):,} queries={len(queries):,} "
            f"source={cfg['author_source']['commit'][:12]}"
        )

    with tracker.phase("Official step-1 BLIP2-T5 gallery captions"):
        captions_path = resolve(str(cfg["cache"]["captions"]))
        caption_py = resolve(str(cfg["caption_env"]["python"]))
        subprocess.run(
            [
                str(caption_py),
                str(METHOD_DIR / "caption_gallery.py"),
                "--config",
                str(config_path),
            ],
            cwd=str(ROOT),
            check=True,
        )
        captions = read_caption_map(captions_path, gallery)

    with tracker.phase("Load released OpenCLIP ViT-B/32 and cache full gallery features"):
        clip_cfg = cfg["models"]["clip"]
        clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
            str(clip_cfg["name"]),
            pretrained=str(clip_cfg["pretrained"]),
            cache_dir=str(resolve(str(clip_cfg["cache_dir"]))),
        )
        clip_model = clip_model.eval().requires_grad_(False).to(device)
        tokenizer = open_clip.get_tokenizer(str(clip_cfg["name"]))
        clip_model.tokenizer = tokenizer
        gallery_paths = [gallery_path(row, gi) for gi, row in enumerate(gallery)]
        gmeta = {
            "schema": CACHE_SCHEMA,
            "adapter": ADAPTER_VERSION,
            "gallery_manifest_sha256": sha256(gallery_manifest),
            "clip": clip_cfg,
        }
        gallery_features = encode_images(
            gallery_paths,
            clip_model,
            clip_preprocess,
            device,
            int(cfg["runtime"]["image_batch_size"]),
            int(cfg["runtime"]["num_workers"]),
            resolve(str(cfg["cache"]["gallery_features"])),
            resolve(str(cfg["cache"]["gallery_features_meta"])),
            gmeta,
        )

    with tracker.phase("Load official BAGEL editor and generate edited text+image queries"):
        wiser_prompts, BagelImageEditor = import_official_editor(checkout)
        modifier_prompt = getattr(wiser_prompts, str(cfg["wiser"]["modifier_prompt"]))
        bagel_cfg = cfg["models"]["bagel"]
        bagel = BagelImageEditor(
            str(resolve(str(bagel_cfg["local_dir"]))),
            max_mem_per_gpu=str(bagel_cfg["max_mem_per_gpu"]),
            offload_folder=str(resolve(str(bagel_cfg["offload_folder"]))),
        )
        generation = generate_queries(
            cfg,
            queries,
            gallery,
            gindex,
            captions,
            bagel,
            modifier_prompt,
            query_manifest,
        )
        modified_captions = [r["modified_caption"] for r in generation]
        edited_paths = [resolve(str(r["edited_image"])) for r in generation]

    with tracker.phase("Wider Search: parallel T2I and I2I full-gallery retrieval"):
        txt_features = encode_texts(
            modified_captions,
            clip_model,
            tokenizer,
            device,
            int(cfg["runtime"]["text_batch_size"]),
        )
        img_features = encode_images(
            edited_paths,
            clip_model,
            clip_preprocess,
            device,
            int(cfg["runtime"]["image_batch_size"]),
            int(cfg["runtime"]["num_workers"]),
        )
        t_rank = full_rank(
            txt_features,
            np.asarray(gallery_features),
            device,
            int(cfg["runtime"]["score_batch_size"]),
        )
        i_rank = full_rank(
            img_features,
            np.asarray(gallery_features),
            device,
            int(cfg["runtime"]["score_batch_size"]),
        )

        # BAGEL and Qwen2.5-VL are both large. WISER does not require them to be
        # resident simultaneously, so release BAGEL before verification. This is
        # an execution-memory optimization only; generated queries are unchanged.
        del bagel
        gc.collect()
        torch.cuda.empty_cache()

    with tracker.phase("Adaptive Fusion verifier: score both top-50 branches"):
        verifier = WiserVerifier(cfg, device)
        t_candidates, i_candidates, t_uncertain, i_uncertain = verify_candidates(
            cfg,
            verifier,
            queries,
            gallery,
            gindex,
            t_rank,
            i_rank,
            loop=0,
        )

    with tracker.phase("Deeper Thinking: refine uncertain retrieval paths"):
        max_loops = int(cfg["wiser"]["max_check_num"])
        if max_loops > 0 and (any(t_uncertain) or any(i_uncertain)):
            api_env = str(cfg["models"]["refiner"]["api_key_env"])
            api_key = os.environ.get(api_env, "").strip()
            if not api_key:
                raise RuntimeError(
                    f"WISER released config uses GPT-4o for refinement. Set {api_env}; "
                    "no smaller/local refiner fallback is used by this adapter."
                )
            client = OpenAI(api_key=api_key)
            for loop in range(1, max_loops + 1):
                t_sug, i_sug = get_suggestions(
                    cfg,
                    client,
                    queries,
                    generation,
                    gallery,
                    captions,
                    t_rank,
                    i_rank,
                    t_uncertain,
                    i_uncertain,
                    loop,
                )

                # Refinement uses BAGEL, so release the verifier first and reload
                # the same pinned BAGEL checkpoint. This avoids changing models
                # merely to fit a smaller GPU.
                del verifier
                gc.collect()
                torch.cuda.empty_cache()
                bagel = BagelImageEditor(
                    str(resolve(str(bagel_cfg["local_dir"]))),
                    max_mem_per_gpu=str(bagel_cfg["max_mem_per_gpu"]),
                    offload_folder=str(resolve(str(bagel_cfg["offload_folder"]))),
                )
                refined_captions, refined_images = refine_queries(
                    cfg,
                    queries,
                    generation,
                    gallery,
                    gindex,
                    bagel,
                    modifier_prompt,
                    t_sug,
                    i_sug,
                    t_uncertain,
                    i_uncertain,
                    loop,
                )
                del bagel
                gc.collect()
                torch.cuda.empty_cache()
                txt_features = encode_texts(
                    refined_captions,
                    clip_model,
                    tokenizer,
                    device,
                    int(cfg["runtime"]["text_batch_size"]),
                )
                img_features = encode_images(
                    refined_images,
                    clip_model,
                    clip_preprocess,
                    device,
                    int(cfg["runtime"]["image_batch_size"]),
                    int(cfg["runtime"]["num_workers"]),
                )
                t_rank = full_rank(
                    txt_features,
                    np.asarray(gallery_features),
                    device,
                    int(cfg["runtime"]["score_batch_size"]),
                )
                i_rank = full_rank(
                    img_features,
                    np.asarray(gallery_features),
                    device,
                    int(cfg["runtime"]["score_batch_size"]),
                )
                verifier = WiserVerifier(cfg, device)
                t_candidates, i_candidates, t_uncertain, i_uncertain = verify_candidates(
                    cfg,
                    verifier,
                    queries,
                    gallery,
                    gindex,
                    t_rank,
                    i_rank,
                    loop=loop,
                )
        else:
            tracker.log("No refinement requested/needed")

    with tracker.phase("CPR completion: preserve WISER candidates, append full-gallery base order"):
        scores_path = write_full_scores(
            cfg,
            gallery,
            t_rank,
            i_rank,
            t_candidates,
            i_candidates,
        )
        tracker.log(f"scores={rel(scores_path)}")

    with tracker.phase("Write transparent run metadata"):
        out_dir = resolve(str(cfg["output"]["dir"]))
        payload = {
            "method": cfg["method"],
            "display_name": cfg["display_name"],
            "group": cfg["group"],
            "paper": cfg["paper"],
            "implementation_status": "OFFICIAL_SOURCE_ADAPTED",
            "adapter_version": ADAPTER_VERSION,
            "training_free": True,
            "cpr_supervision": "No",
            "higher_is_better": True,
            "query_image_removed_inside_method": False,
            "num_queries": len(queries),
            "num_gallery": len(gallery),
            "scores": rel(scores_path),
            "author_source": {
                "repository": cfg["author_source"]["repository"],
                "commit": cfg["author_source"]["commit"],
            },
            "released_components": {
                "captioner": cfg["models"]["captioner"],
                "editor": "BAGEL-7B-MoT",
                "retrieval": cfg["models"]["clip"],
                "verifier": cfg["models"]["verifier"]["repo_id"],
                "refiner": cfg["models"]["refiner"],
                "topk_per_path": int(cfg["wiser"]["topk_per_path"]),
                "confidence_threshold": float(cfg["wiser"]["confidence_threshold"]),
                "max_check_num": int(cfg["wiser"]["max_check_num"]),
            },
            "fusion": {
                "candidate_score": "confidence_t2i + confidence_i2i",
                "tie_break": [
                    "sum_confidence",
                    "max_path_confidence",
                    "t2i_confidence",
                    "canonical_image_id",
                ],
            },
            "cpr_adapter_boundary": {
                "query": "full canonical reference scene + query text",
                "uses_target_ids": False,
                "uses_full_positive_ids": False,
                "uses_cpr_labels": False,
                "full_gallery_completion": cfg["adaptation"]["full_gallery_completion"],
                "completion_key": [
                    "min(t2i_rank,i2i_rank)",
                    "t2i_rank+i2i_rank",
                    "t2i_rank",
                    "gallery_index",
                ],
                "verified_candidate_order_is_not_changed": True,
            },
            "release_compatibility_fixes": [
                "verifier implemented with released Qwen2.5-VL prompt/logit math because current public helper references an undefined Qwen3 enum member",
                "GPT-4o refiner prompt implemented from released source because current public helper references an undefined enum member",
                "post-refinement candidate verification is used for final fusion; public source computes it but returns stale pre-refinement candidate variables",
                "LAVIS captioner isolated from modern Qwen/BAGEL Transformers dependencies",
            ],
            "prepared_marker": rel(resolve(str(cfg["checkpoint"]["prepared_marker"]))),
        }
        write_json(out_dir / "run.json", payload)

    if "verifier" in locals():
        del verifier
    if "bagel" in locals():
        del bagel
    del clip_model
    gc.collect()
    torch.cuda.empty_cache()
    tracker.finish()


if __name__ == "__main__":
    main()
