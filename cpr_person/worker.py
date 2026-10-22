"""Offline, resumable HF Grounding DINO worker; no retrieval encoder imports."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from benchmark_progress import progress_bar
from .cache import StageCache, canonical_hash, resolve
from .detection import (Detections, detection_fingerprint, image_path, load_detections,
                        protocol_config, read_rgb, save_detections)


def recover_prefix(path, gallery, fingerprint):
    rows, valid_end = [], 0
    if not path.is_file():
        return rows
    with path.open("rb+") as handle:
        while line := handle.readline():
            try:
                entry = json.loads(line)
                digest = entry.pop("sha256")
                if (digest != canonical_hash(entry) or not line.endswith(b"\n")
                        or len(rows) >= len(gallery) or entry["image_id"] != gallery[len(rows)]["image_id"]):
                    break
                rows_to_detections([entry], fingerprint).validate(1)
            except (ValueError, TypeError, KeyError):
                break
            rows.append(entry)
            valid_end = handle.tell()
        handle.truncate(valid_end)
    return rows


def rows_to_detections(rows, fingerprint):
    counts = [len(row["boxes"]) for row in rows]
    return Detections(offsets=np.r_[0, np.cumsum(counts)].astype(np.int64),
                      boxes=np.asarray([b for row in rows for b in row["boxes"]], np.float32).reshape(-1, 4),
                      confidences=np.asarray([s for row in rows for s in row["scores"]], np.float32),
                      fallback=np.asarray([f for row in rows for f in row["fallback"]], bool),
                      image_sizes=np.asarray([row["size"] for row in rows], np.int64).reshape(-1, 2),
                      fingerprint=fingerprint)


def detect_image(model, processor, image, detector, device):
    """Same processor call, exact label equality and stable indexing as RCR detect.

    Keep scores alongside boxes; RCR's helper returns only boxes. HF already
    applies both thresholds. Do not sort, clip, round, suppress or cap its output.
    """
    import torch
    with torch.inference_mode():
        inputs = processor(images=image, text=[[detector["text_prompt"]]], return_tensors="pt").to(device)
        outputs = model(**inputs)
        result = processor.post_process_grounded_object_detection(
            outputs, input_ids=inputs.input_ids, threshold=float(detector["threshold"]),
            text_threshold=float(detector["text_threshold"]),
            target_sizes=[(image.height, image.width)])[0]
        boxes = result["boxes"]
        keep = torch.tensor([name == detector["text_prompt"] for name in result["text_labels"]],
                            device=boxes.device, dtype=torch.bool)
        return (boxes[keep].float().cpu().numpy(),
                result["scores"][keep].float().cpu().numpy())


def detection_row(image_id, image, boxes, scores):
    fallback = len(boxes) == 0
    if fallback:
        boxes = np.array([[0, 0, image.width, image.height]], np.float32)
        scores = np.array([-1], np.float32)
    return {"image_id": image_id, "size": [image.width, image.height],
            "boxes": boxes.tolist(), "scores": scores.tolist(), "fallback": [fallback] * len(boxes)}


def detect(gallery, cache, device):
    cfg = protocol_config()
    journal = cache.directory / "persons.partial.jsonl"
    rows = recover_prefix(journal, gallery, cache.fingerprint)
    start = len(rows)
    print(f"[shared detector] resume={start}/{len(gallery)} fingerprint={cache.key}", flush=True)
    if start < len(gallery):
        from .assets import load_detector
        from .runtime import check_runtime
        runtime = check_runtime(cfg["detector"])
        model, processor = load_detector(cfg["detector"], device)
        cache.directory.mkdir(parents=True, exist_ok=True)
        with journal.open("a", encoding="utf-8") as handle:
            for gi in progress_bar(range(start, len(gallery)), desc="Shared person detection", unit="image"):
                image = read_rgb(image_path(gallery[gi]))
                boxes, scores = detect_image(model, processor, image, cfg["detector"], device)
                entry = detection_row(gallery[gi]["image_id"], image, boxes, scores)
                rows_to_detections([entry], cache.fingerprint).validate(1)
                handle.write(json.dumps({**entry, "sha256": canonical_hash(entry)}) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
                rows.append(entry)
        del model, processor
    else:
        runtime = None  # A complete journal is sealable without loading a model.
    result = rows_to_detections(rows, cache.fingerprint)
    result.validate(len(gallery))
    save_detections(cache, result, runtime_versions=runtime, device=str(device))
    journal.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gallery-manifest", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    manifest = resolve(args.gallery_manifest)
    gallery = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    fingerprint = detection_fingerprint(manifest)
    directory = Path(args.cache_dir).resolve()
    cache = StageCache("detections", fingerprint, write_root=directory.parents[1])
    if cache.directory != directory:
        raise ValueError("Worker output directory does not match immutable inputs")
    if load_detections(cache, len(gallery)) is None:
        detect(gallery, cache, args.device)


if __name__ == "__main__":
    main()
