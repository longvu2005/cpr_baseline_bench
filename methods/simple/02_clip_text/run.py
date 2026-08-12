#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import clip
import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = Path(__file__).resolve().parent / "config.yaml"


def load_jsonl(path):
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def device_from(name):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class GalleryDataset(Dataset):
    def __init__(self, rows, preprocess):
        self.rows = rows
        self.preprocess = preprocess

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        with Image.open(ROOT / self.rows[i]["path"]) as image:
            return self.preprocess(image.convert("RGB"))


@torch.no_grad()
def gallery_features(model, preprocess, rows, cache, runtime, device):
    feature_dim = int(model.text_projection.shape[1])
    expected_shape = (len(rows), feature_dim)

    if cache.is_file():
        x = np.load(cache, mmap_mode="r")
        if x.shape == expected_shape:
            print(f"Using gallery cache: {cache}")
            return x, cache
        print(
            f"Ignoring incompatible gallery cache {cache}: "
            f"got {x.shape}, expected {expected_shape}"
        )

    loader = DataLoader(
        GalleryDataset(rows, preprocess),
        batch_size=runtime["batch_size"],
        shuffle=False,
        num_workers=runtime["num_workers"],
        pin_memory=(device.type == "cuda"),
    )
    chunks = []
    for images in tqdm(loader, desc="Gallery"):
        x = model.encode_image(images.to(device)).float()
        x /= x.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        chunks.append(x.cpu().numpy())
    x = np.concatenate(chunks).astype(np.float32)
    if x.shape != expected_shape:
        raise ValueError(
            f"Encoded gallery has shape {x.shape}, expected {expected_shape}"
        )
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, x)
    return np.load(cache, mmap_mode="r"), cache


@torch.no_grad()
def text_features(model, texts, batch_size, device):
    chunks = []
    for start in tqdm(range(0, len(texts), batch_size), desc="Text"):
        tokens = clip.tokenize(texts[start:start + batch_size], truncate=True).to(device)
        x = model.encode_text(tokens).float()
        x /= x.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        chunks.append(x.cpu().numpy())
    return np.concatenate(chunks).astype(np.float32)


def setup():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    output = ROOT / cfg["output"]["dir"]
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = ROOT / cfg["model"]["checkpoint"]
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"Missing CLIP checkpoint: {checkpoint}\n"
            "Run `python methods/simple/02_clip_text/download_checkpoint.py` from the repository root."
        )

    gallery = load_jsonl(ROOT / "data/gallery.jsonl")
    queries = load_jsonl(ROOT / "data/queries.jsonl")
    device = device_from(cfg["runtime"].get("device", "auto"))
    model, preprocess = clip.load(str(checkpoint), device=device, jit=False)
    model.eval()
    if device.type != "cuda":
        model.float()

    cache = ROOT / cfg["cache"]["gallery_features"]
    gfeat, cache_used = gallery_features(
        model, preprocess, gallery, cache, cfg["runtime"], device
    )
    tfeat = text_features(
        model, [q["text"] for q in queries],
        cfg["runtime"]["text_batch_size"], device
    )
    return cfg, config_path, output, gallery, queries, device, gfeat, tfeat, cache_used


@torch.no_grad()
def main():
    cfg, config_path, output, gallery, queries, device, gfeat, tfeat, cache_used = setup()
    scores_path = output / "scores.npy"
    scores = np.lib.format.open_memmap(
        scores_path, "w+", dtype=np.float32, shape=(len(queries), len(gallery))
    )
    gallery_tensor = torch.from_numpy(np.asarray(gfeat)).to(device)
    batch = cfg["runtime"]["score_batch_size"]
    for start in tqdm(range(0, len(queries), batch), desc="Scores"):
        end = min(start + batch, len(queries))
        query = torch.from_numpy(tfeat[start:end]).to(device)
        scores[start:end] = (query @ gallery_tensor.T).cpu().numpy()
    scores.flush()

    run = {
        "method": cfg["method"],
        "display_name": f"CLIP {cfg['model']['name']} - Text-only",
        "group": "Simple / Obvious Baselines",
        "cpr_supervision": "No",
        "model": cfg["model"],
        "runtime": cfg["runtime"],
        "gallery_features": str(cache_used.relative_to(ROOT)),
        "config": str(config_path.relative_to(ROOT)),
        "num_queries": len(queries),
        "num_gallery": len(gallery),
        "scores": str(scores_path.relative_to(ROOT)),
        "higher_is_better": True,
    }
    with (output / "run.json").open("w", encoding="utf-8") as f:
        json.dump(run, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
