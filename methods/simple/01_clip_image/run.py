#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import clip
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[3]

GALLERY_FILE = ROOT / "data/gallery.jsonl"
QUERIES_FILE = ROOT / "data/queries.jsonl"

CHECKPOINT = ROOT / "checkpoints/clip/ViT-B-16.pt"

OUTPUT_DIR = ROOT / "runs/clip_image"
FEATURES_FILE = OUTPUT_DIR / "gallery_features.npy"
SCORES_FILE = OUTPUT_DIR / "scores.npy"
RUN_FILE = OUTPUT_DIR / "run.json"


def read_jsonl(path):
    rows = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line:
                rows.append(json.loads(line))

    return rows


def get_device(requested):
    if requested != "auto":
        return torch.device(requested)

    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


class ImageDataset(Dataset):
    def __init__(self, rows, preprocess):
        self.rows = rows
        self.preprocess = preprocess

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        path = ROOT / self.rows[index]["path"]

        with Image.open(path) as image:
            image = image.convert("RGB")
            return self.preprocess(image)


@torch.no_grad()
def encode_gallery(
    model,
    preprocess,
    gallery,
    device,
    batch_size,
    num_workers,
):
    dataset = ImageDataset(gallery, preprocess)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    features = []

    for images in tqdm(loader, desc="Gallery"):
        images = images.to(device)

        x = model.encode_image(images).float()
        x = x / x.norm(dim=-1, keepdim=True).clamp_min(1e-12)

        features.append(x.cpu().numpy())

    return np.concatenate(features, axis=0)


@torch.no_grad()
def create_scores(
    features,
    query_indices,
    device,
    batch_size,
):
    gallery_features = torch.from_numpy(
        np.asarray(features)
    ).to(device)

    scores_file = np.lib.format.open_memmap(
        SCORES_FILE,
        mode="w+",
        dtype=np.float32,
        shape=(len(query_indices), len(features)),
    )

    for start in tqdm(
        range(0, len(query_indices), batch_size),
        desc="Scores",
    ):
        end = min(
            start + batch_size,
            len(query_indices),
        )

        idx = query_indices[start:end]

        query_features = gallery_features[idx]

        scores = query_features @ gallery_features.T

        scores_file[start:end] = (
            scores.float().cpu().numpy()
        )

    scores_file.flush()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--device",
        default="auto",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--score-batch-size",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--recompute",
        action="store_true",
    )

    args = parser.parse_args()

    if not CHECKPOINT.is_file():
        raise FileNotFoundError(
            f"Missing checkpoint: {CHECKPOINT}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    gallery = read_jsonl(GALLERY_FILE)
    queries = read_jsonl(QUERIES_FILE)

    device = get_device(args.device)

    print()
    print("CLIP ViT-B/16 - Image-only")
    print("--------------------------")
    print(f"Gallery : {len(gallery):,}")
    print(f"Queries : {len(queries):,}")
    print(f"Device  : {device}")
    print()

    gallery_index = {
        row["image_id"]: index
        for index, row in enumerate(gallery)
    }

    if len(gallery_index) != len(gallery):
        raise ValueError("Duplicate gallery image_id")

    query_indices = []

    for query in queries:
        image_id = query["image_id"]

        if image_id not in gallery_index:
            raise ValueError(
                f"Query image missing from gallery: {image_id}"
            )

        query_indices.append(
            gallery_index[image_id]
        )

    query_indices = np.asarray(
        query_indices,
        dtype=np.int64,
    )

    model, preprocess = clip.load(
        str(CHECKPOINT),
        device=device,
        jit=False,
    )

    model.eval()

    if device.type != "cuda":
        model.float()

    if FEATURES_FILE.is_file() and not args.recompute:
        print(f"Using cache: {FEATURES_FILE}")

        features = np.load(
            FEATURES_FILE,
            mmap_mode="r",
        )

        if features.shape[0] != len(gallery):
            raise ValueError(
                "Gallery feature cache has wrong size. "
                "Run again with --recompute."
            )

    else:
        features = encode_gallery(
            model=model,
            preprocess=preprocess,
            gallery=gallery,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )

        np.save(
            FEATURES_FILE,
            features.astype(np.float32),
        )

        print(f"Saved: {FEATURES_FILE}")

    create_scores(
        features=features,
        query_indices=query_indices,
        device=device,
        batch_size=args.score_batch_size,
    )

    run = {
        "method": "clip_image",
        "display_name": "CLIP ViT-B/16 - Image-only",
        "group": "Simple / Obvious Baselines",
        "cpr_supervision": "No",
        "checkpoint": "checkpoints/clip/ViT-B-16.pt",
        "num_queries": len(queries),
        "num_gallery": len(gallery),
        "scores": "runs/clip_image/scores.npy",
        "higher_is_better": True,
    }

    with RUN_FILE.open("w", encoding="utf-8") as f:
        json.dump(
            run,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print("Done")
    print("----")
    print(f"Features : {FEATURES_FILE}")
    print(f"Scores   : {SCORES_FILE}")
    print(f"Run info : {RUN_FILE}")
    print(f"Shape    : ({len(queries)}, {len(gallery)})")


if __name__ == "__main__":
    main()
