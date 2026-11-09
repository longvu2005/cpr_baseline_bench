"""Prepare and fingerprint the benchmark's RCR OpenAI CLIP scene encoder."""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from benchmark_data import load_jsonl
from benchmark_progress import progress_bar
from cpr_person.cache import ROOT, StageCache, atomic_file, cache_lock, resolve, sha256_file
from cpr_person.clip_model import prepare_openai_clip
from cpr_person.protocol import RCR_CLIP_NAME


def gallery_fingerprint(checkpoint: Path, gallery_manifest: Path, model_name: str) -> dict:
    image_files = []
    for row in load_jsonl(gallery_manifest):
        stat = (ROOT / row["path"]).stat()
        image_files.append([stat.st_size, stat.st_mtime_ns])
    return {
        "schema": "rcr-openai-clip-gallery-v1",
        "model_name": model_name,
        "checkpoint_sha256": sha256_file(checkpoint),
        "gallery_manifest_sha256": sha256_file(gallery_manifest),
        "image_files": image_files,
    }



class GalleryImages:
    def __init__(self, rows, preprocess):
        self.rows = rows
        self.preprocess = preprocess

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        from PIL import Image

        with Image.open(ROOT / self.rows[index]["path"]) as image:
            return self.preprocess(image.convert("RGB"))


def gallery_features(model, preprocess, rows, cache, runtime, device, fingerprint):
    import numpy as np
    import torch
    from torch.utils.data import DataLoader

    batch_size = int(runtime.get("clip_image_batch_size", runtime.get("batch_size", 32)))
    stage = StageCache(
        "scene_clip", {**fingerprint, "device": device.type, "batch_size": batch_size,
                       "torch_version": str(torch.__version__)},
        write_root=cache.parent, read_roots=[],
    )
    expected = (len(rows), int(model.text_projection.shape[1]))

    def cached():
        path = stage.find(cache.name)
        if path is not None:
            try:
                # Torch can wrap this copy-on-write map without modifying the sealed cache.
                features = np.load(path, mmap_mode="c", allow_pickle=False)
                if features.shape == expected and features.dtype == np.float32:
                    print(f"Using CLIP gallery cache: {path}", flush=True)
                    return features, path, True
            except (OSError, ValueError, EOFError):
                pass
        return None

    result = cached()
    if result is not None:
        return result
    with cache_lock(stage.directory):
        result = cached()
        if result is not None:
            return result
        loader = DataLoader(
            GalleryImages(rows, preprocess), batch_size=batch_size,
            num_workers=int(runtime.get("num_workers", 0)), shuffle=False,
            pin_memory=device.type == "cuda",
        )
        chunks = []
        with torch.inference_mode():
            for images in progress_bar(loader, desc="Encode CLIP gallery", total=len(loader), unit="batch"):
                encoded = model.encode_image(images.to(device)).float()
                chunks.append(torch.nn.functional.normalize(encoded, dim=-1).cpu().numpy())
        features = np.concatenate(chunks)
        if features.shape != expected or not np.isfinite(features).all():
            raise ValueError(f"Invalid CLIP gallery features: {features.shape}")
        path = stage.directory / cache.name
        with atomic_file(path) as handle:
            np.save(handle, features, allow_pickle=False)
        stage.seal(path)
        return features, path, False

def prepare_scene_clip(cfg, force=False):
    if cfg["name"] != RCR_CLIP_NAME:
        raise ValueError("RCR scene CLIP must use OpenAI ViT-B/32")
    prepare_openai_clip(cfg["name"], cfg["checkpoint"], force)


def main(config_path):
    parser = argparse.ArgumentParser(description="Prepare the RCR OpenAI CLIP scene checkpoint")
    parser.add_argument("--config", default=str(config_path))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    cfg = yaml.safe_load(resolve(args.config).read_text(encoding="utf-8"))
    prepare_scene_clip(cfg["model"], args.force)
