"""Deterministic caption-to-caption and per-person structured caption retrieval."""
from __future__ import annotations

import hashlib
import json
import re

import numpy as np

from benchmark_progress import progress_bar
from cpr_person.cache import StageCache, canonical_hash, read_json
from cpr_person.detection import crop_person, image_path, read_rgb
from . import VERSION
from .common import query_text

FIELDS = ("appearance", "clothing", "action", "objects")


def parse_caption(raw):
    if not isinstance(raw, str):
        raise ValueError("Caption is not text")
    value = re.sub(r"\s+", " ", raw).strip()
    if not value:
        raise ValueError("Empty generated caption")
    return value


def parse_structure(raw):
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) < 3 or lines[-1].strip() != "```":
            raise ValueError("Malformed JSON code fence")
        text = "\n".join(lines[1:-1])
    value = json.loads(text)
    if (not isinstance(value, dict) or set(value) != set(FIELDS)
            or any(not isinstance(value[key], str) for key in FIELDS)):
        raise ValueError(f"Structured caption must contain exactly these string fields: {FIELDS}")
    cleaned = {key: re.sub(r"\s+", " ", value[key]).strip() for key in FIELDS}
    if not any(cleaned.values()):
        raise ValueError("All structured caption fields are empty")
    return cleaned


def structure_text(value):
    # Fixed field order avoids scoring JSON punctuation or inconsistent key order.
    if set(value) != set(FIELDS):
        raise ValueError("Invalid structured caption schema")
    return "; ".join(f"{key}: {value[key]}" for key in FIELDS if value[key])


def pixels_hash(image):
    return canonical_hash({"size": list(image.size), "mode": image.mode,
                           "sha256": hashlib.sha256(image.tobytes()).hexdigest()})


class CaptionStore:
    def __init__(self, cfg, asset_identity, generator_factory):
        self.cfg, self.asset_identity = cfg, asset_identity
        self.generator_factory, self.generator = generator_factory, None
        self.hits = self.generated = self.repairs = 0

    def request(self, prompt, image=None, *, structured=False):
        parser = parse_structure if structured else parse_caption
        fp = {"version": VERSION, "captioner": self.asset_identity,
              "generation": self.cfg["generation"], "runtime": self.cfg["runtime"],
              "prompt": prompt, "image_pixels": None if image is None else pixels_hash(image),
              "schema": "person-fields-v1" if structured else "scene-caption-v1"}
        cache = StageCache("qwen_caption_records", fp, write_root="cache/additional")
        path = cache.find("response.json")
        if path is not None:
            record = read_json(path)
            if not isinstance(record, dict) or parser(record["raw"]) != record["value"]:
                raise ValueError("Generated caption cache contains incompatible output")
            self.hits += 1
            return record["value"]
        if self.generator is None:
            self.generator = self.generator_factory()
        attempts = []
        retry_prompt = prompt
        for attempt in range(1 + int(self.cfg["generation"]["format_retries"])):
            raw = self.generator.generate(retry_prompt, image)
            attempts.append(raw)
            try:
                value = parser(raw)
            except (ValueError, TypeError) as error:
                if attempt >= int(self.cfg["generation"]["format_retries"]):
                    raise ValueError(f"Qwen returned an invalid caption after fixed format retries: {error}") from error
                self.repairs += 1
                retry_prompt = (prompt + "\nYour previous answer did not follow the required format. "
                                "Return only the requested output, without markdown or explanations.\n"
                                "Previous answer:\n" + raw)
                continue
            cache.write_json("response.json", {"raw": raw, "value": value, "attempts": attempts})
            self.generated += 1
            return value
        raise AssertionError("Caption generation did not finish")

    def metadata(self):
        return {"cache_hits": self.hits, "generated_records": self.generated,
                "format_repairs": self.repairs, "greedy_decoding": True,
                "label_conditioned_generation": False,
                "schema_fields": list(FIELDS) if self.cfg["backend"] == "structured_caption" else None}

    def release(self):
        self.generator = None
        import gc
        gc.collect()
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def scene_captions(cfg, gallery, queries, gallery_index, store):
    descriptions = []
    for row in progress_bar(gallery, desc="Caption gallery", unit="image"):
        image = read_rgb(image_path(row))
        descriptions.append(store.request(cfg["prompts"]["gallery"], image))
    targets = []
    for query in progress_bar(queries, desc="Compose target captions", unit="query"):
        reference = descriptions[gallery_index[query["image_id"]]]
        prompt = cfg["prompts"]["target"].format(reference=reference, instruction=query_text(query))
        targets.append(store.request(prompt))
    return descriptions, targets


def person_captions(cfg, gallery, queries, gallery_index, person, store):
    gallery_captions = []
    offsets = person.detections.offsets
    for gi, row in enumerate(progress_bar(gallery, desc="Caption gallery people", unit="image")):
        image = read_rgb(image_path(row))
        for pi in range(int(offsets[gi]), int(offsets[gi + 1])):
            crop = crop_person(image, person.detections.boxes[pi])
            gallery_captions.append(store.request(cfg["prompts"]["gallery"], crop, structured=True))
    if len(gallery_captions) != int(offsets[-1]):
        raise ValueError("Person captions differ from shared detector ordering")
    targets, query_offsets = [], [0]
    fallback_captions = {}
    for qi, query in enumerate(progress_bar(queries, desc="Compose structured people", unit="query")):
        gi = gallery_index[query["image_id"]]
        for si, member in person.localization.flat(qi):
            ref = member["global_person_index"]
            if ref is None:
                # Exactly the shared full-image padding policy when there are
                # fewer detected candidates than subjects. No person-ID lookup.
                key = (gi, tuple(member["box"]))
                if key not in fallback_captions:
                    image = crop_person(read_rgb(image_path(gallery[gi])), member["box"])
                    fallback_captions[key] = store.request(cfg["prompts"]["gallery"], image, structured=True)
                reference = fallback_captions[key]
            else:
                reference = gallery_captions[ref]
            subject = query["subjects"][si]
            modification = str(subject.get("modify_text") or query.get("relation_text") or "").strip()
            if not modification:
                modification = query_text(query)
            prompt = cfg["prompts"]["target"].format(
                reference=json.dumps(reference, ensure_ascii=False),
                subject_id=subject.get("subject_id", si + 1),
                selection=subject["select_text"], modification=modification,
                instruction=query_text(query))
            targets.append(store.request(prompt, structured=True))
        query_offsets.append(len(targets))
    if np.any(np.diff(query_offsets) < 1):
        raise ValueError("Every query must have at least one predicted member")
    return gallery_captions, targets, np.asarray(query_offsets, dtype=np.int64)
