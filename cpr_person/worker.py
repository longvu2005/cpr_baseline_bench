"""Isolated, resumable Grounding DINO worker (no retrieval encoder imports)."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from benchmark_progress import progress_bar
from . import assets
from .cache import StageCache, canonical_hash, resolve, sha256_file
from .detection import (Detections, detection_fingerprint, filter_person_boxes, image_path,
                        protocol_config, save_detections)


class DetectionImages(Dataset):
    def __init__(self, gallery, start):
        self.gallery, self.start = gallery, start

    def __len__(self):
        return len(self.gallery) - self.start

    def __getitem__(self, index):
        from groundingdino.util.inference import load_image
        gi = index + self.start
        image, tensor = load_image(str(image_path(self.gallery[gi])))
        height, width = image.shape[:2]
        return tensor, width, height, gi


def collate(rows):
    return tuple(map(list, zip(*rows)))


def split_batch(tensors, widths, heights, owners, ratio):
    start = 0
    while start < len(tensors):
        end = start + 1
        while end < len(tensors):
            shapes = [x.shape[-2:] for x in tensors[start:end + 1]]
            padded = max(h for h, w in shapes) * max(w for h, w in shapes) * len(shapes)
            if padded / sum(h * w for h, w in shapes) > ratio:
                break
            end += 1
        yield tensors[start:end], widths[start:end], heights[start:end], owners[start:end]
        start = end


def recover_prefix(path, gallery):
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
                d = rows_to_detections([entry], {"protocol": {"max_persons_per_image": 10}})
                d.validate(1)
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


@torch.no_grad()
def detect(gallery, cache, device):
    cfg = protocol_config()
    detector_cfg = cfg["detector"]
    checkout = assets.ensure_clean_pinned_source(detector_cfg["source"], "Grounding DINO")
    config = checkout / detector_cfg["config"]
    if sha256_file(config) != detector_cfg["config_sha256"]:
        raise RuntimeError("Pinned Grounding DINO config checksum mismatch")
    assets.configure_groundingdino_offline({"detector": detector_cfg,
                                           "source": {"groundingdino": detector_cfg["source"]}})
    backend = assets.configure_groundingdino_source(checkout)
    if backend != detector_cfg["attention_backend"]:
        raise RuntimeError("Grounding DINO attention backend differs from shared protocol")
    journal = cache.directory / "persons.partial.jsonl"
    rows = recover_prefix(journal, gallery)
    start = len(rows)
    print(f"[shared detector] resume={start}/{len(gallery)} fingerprint={cache.key}", flush=True)
    if start < len(gallery):
        from groundingdino.models import build_model
        from groundingdino.util.inference import preprocess_caption
        from groundingdino.util.misc import clean_state_dict, nested_tensor_from_tensor_list
        from groundingdino.util.slconfig import SLConfig
        args = SLConfig.fromfile(str(config))
        args.device, args.use_checkpoint, args.use_transformer_ckpt = str(device), False, False
        model = build_model(args)
        state = torch.load(resolve(detector_cfg["checkpoint"]), map_location="cpu", weights_only=False)
        model.load_state_dict(clean_state_dict(state["model"]), strict=False)
        del state
        model.to(device).eval()
        torch.manual_seed(0)
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        batch_size = int(detector_cfg["batch_size"])
        # Fixed batch grouping. Resume at the preceding batch boundary so an
        # interrupted partial batch cannot change padding/forward numerics.
        if start % batch_size:
            start -= start % batch_size
            rows = rows[:start]
            with journal.open("w") as handle:
                for row in rows:
                    handle.write(json.dumps({**row, "sha256": canonical_hash(row)}) + "\n")
        loader = DataLoader(DetectionImages(gallery, start), batch_size=batch_size, shuffle=False,
                            num_workers=int(os.environ.get("CPR_DETECTOR_WORKERS", "4")), collate_fn=collate)
        caption = preprocess_caption(detector_cfg["text_prompt"])
        with journal.open("a", encoding="utf-8") as handle:
            for batch in progress_bar(loader, desc="Shared person detection", total=len(loader), unit="batch"):
                for tensors, widths, heights, owners in split_batch(*batch, detector_cfg["max_padding_ratio"]):
                    tensors = [x.to(device) for x in tensors]
                    output = model(nested_tensor_from_tensor_list(tensors), captions=[caption] * len(tensors))
                    confidence = output["pred_logits"].sigmoid().max(dim=-1).values.cpu().numpy()
                    boxes = output["pred_boxes"].cpu().numpy()
                    for row, width, height, gi in zip(range(len(tensors)), widths, heights, owners):
                        b, s = filter_person_boxes(boxes[row], confidence[row], width, height, detector_cfg)
                        fallback = not len(b)
                        if fallback:
                            b, s = np.array([[0, 0, width, height]], np.float32), np.array([-1], np.float32)
                        entry = {"image_id": gallery[gi]["image_id"], "size": [width, height],
                                 "boxes": b.tolist(), "scores": s.tolist(), "fallback": [fallback] * len(b)}
                        rows.append(entry)
                        handle.write(json.dumps({**entry, "sha256": canonical_hash(entry)}) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
        del model
    result = rows_to_detections(rows, cache.fingerprint)
    result.validate(len(gallery))
    save_detections(cache, result)
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
    detect(gallery, cache, torch.device(args.device))


if __name__ == "__main__":
    main()
