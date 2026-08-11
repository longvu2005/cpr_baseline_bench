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
METHOD_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = METHOD_DIR / "config.yaml"

GALLERY_FILE = ROOT / "data/gallery.jsonl"
QUERIES_FILE = ROOT / "data/queries.jsonl"


def read_jsonl(path):
    rows = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line:
                rows.append(json.loads(line))

    return rows


def load_config(path):
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


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
    dataset = ImageDataset(
        gallery,
        preprocess,
    )

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

        x = x / x.norm(
            dim=-1,
            keepdim=True,
        ).clamp_min(1e-12)

        features.append(
            x.cpu().numpy()
        )

    return np.concatenate(
        features,
        axis=0,
    )


@torch.no_grad()
def create_scores(
    features,
    query_indices,
    device,
    batch_size,
    scores_file,
):
    gallery_features = torch.from_numpy(
        np.asarray(features)
    ).to(device)

    output = np.lib.format.open_memmap(
        scores_file,
        mode="w+",
        dtype=np.float32,
        shape=(
            len(query_indices),
            len(features),
        ),
    )

    for start in tqdm(
        range(
            0,
            len(query_indices),
            batch_size,
        ),
        desc="Scores",
    ):
        end = min(
            start + batch_size,
            len(query_indices),
        )

        indices = query_indices[start:end]

        query_features = gallery_features[
            indices
        ]

        scores = (
            query_features
            @ gallery_features.T
        )

        output[start:end] = (
            scores.float()
            .cpu()
            .numpy()
        )

    output.flush()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
    )

    parser.add_argument(
        "--device",
        default=None,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--score-batch-size",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--recompute",
        action="store_true",
    )

    args = parser.parse_args()

    # ---------------------------------------------------------
    # Config
    # ---------------------------------------------------------

    config_path = Path(args.config)

    if not config_path.is_absolute():
        config_path = ROOT / config_path

    if not config_path.is_file():
        raise FileNotFoundError(
            f"Missing config: {config_path}"
        )

    config = load_config(config_path)

    model_config = config.get(
        "model",
        {},
    )

    runtime_config = config.get(
        "runtime",
        {},
    )

    output_config = config.get(
        "output",
        {},
    )

    method_name = config.get(
        "method",
        "clip_image",
    )

    model_name = model_config.get(
        "name",
        "ViT-B/16",
    )

    checkpoint_rel = model_config.get(
        "checkpoint",
        "checkpoints/clip/ViT-B-16.pt",
    )

    output_rel = output_config.get(
        "dir",
        f"runs/{method_name}",
    )

    checkpoint = ROOT / checkpoint_rel
    output_dir = ROOT / output_rel

    features_file = (
        output_dir
        / "gallery_features.npy"
    )

    scores_file = (
        output_dir
        / "scores.npy"
    )

    run_file = (
        output_dir
        / "run.json"
    )

    # CLI overrides config.
    device_name = (
        args.device
        if args.device is not None
        else runtime_config.get(
            "device",
            "auto",
        )
    )

    batch_size = (
        args.batch_size
        if args.batch_size is not None
        else runtime_config.get(
            "batch_size",
            128,
        )
    )

    score_batch_size = (
        args.score_batch_size
        if args.score_batch_size is not None
        else runtime_config.get(
            "score_batch_size",
            256,
        )
    )

    num_workers = (
        args.num_workers
        if args.num_workers is not None
        else runtime_config.get(
            "num_workers",
            4,
        )
    )

    # ---------------------------------------------------------
    # Setup
    # ---------------------------------------------------------

    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"Missing checkpoint: {checkpoint}"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    gallery = read_jsonl(
        GALLERY_FILE
    )

    queries = read_jsonl(
        QUERIES_FILE
    )

    device = get_device(
        device_name
    )

    print()
    print("CLIP ViT-B/16 - Image-only")
    print("--------------------------")
    print(f"Config      : {config_path}")
    print(f"Gallery     : {len(gallery):,}")
    print(f"Queries     : {len(queries):,}")
    print(f"Device      : {device}")
    print(f"Batch size  : {batch_size}")
    print(f"Score batch : {score_batch_size}")
    print(f"Workers     : {num_workers}")
    print()

    # ---------------------------------------------------------
    # Query -> gallery index
    # ---------------------------------------------------------

    gallery_index = {
        row["image_id"]: index
        for index, row in enumerate(
            gallery
        )
    }

    if len(gallery_index) != len(gallery):
        raise ValueError(
            "Duplicate gallery image_id"
        )

    query_indices = []

    for query in queries:
        image_id = query["image_id"]

        if image_id not in gallery_index:
            raise ValueError(
                "Query image missing "
                f"from gallery: {image_id}"
            )

        query_indices.append(
            gallery_index[image_id]
        )

    query_indices = np.asarray(
        query_indices,
        dtype=np.int64,
    )

    # ---------------------------------------------------------
    # Model
    # ---------------------------------------------------------

    model, preprocess = clip.load(
        str(checkpoint),
        device=device,
        jit=False,
    )

    model.eval()

    if device.type != "cuda":
        model.float()

    # ---------------------------------------------------------
    # Gallery features
    # ---------------------------------------------------------

    if (
        features_file.is_file()
        and not args.recompute
    ):
        print(
            f"Using cache: {features_file}"
        )

        features = np.load(
            features_file,
            mmap_mode="r",
        )

        if features.shape[0] != len(gallery):
            raise ValueError(
                "Gallery feature cache has "
                "wrong size. Run again "
                "with --recompute."
            )

    else:
        features = encode_gallery(
            model=model,
            preprocess=preprocess,
            gallery=gallery,
            device=device,
            batch_size=batch_size,
            num_workers=num_workers,
        )

        np.save(
            features_file,
            features.astype(
                np.float32
            ),
        )

        print(
            f"Saved: {features_file}"
        )

    # ---------------------------------------------------------
    # Scores
    # ---------------------------------------------------------

    create_scores(
        features=features,
        query_indices=query_indices,
        device=device,
        batch_size=score_batch_size,
        scores_file=scores_file,
    )

    # ---------------------------------------------------------
    # Run metadata
    # ---------------------------------------------------------

    run = {
        "method": method_name,
        "display_name": (
            "CLIP ViT-B/16 - Image-only"
        ),
        "group": (
            "Simple / Obvious Baselines"
        ),
        "cpr_supervision": "No",
        "model": {
            "name": model_name,
            "checkpoint": checkpoint_rel,
        },
        "runtime": {
            "device": str(device),
            "batch_size": batch_size,
            "score_batch_size": (
                score_batch_size
            ),
            "num_workers": num_workers,
        },
        "config": str(
            config_path.relative_to(ROOT)
        ),
        "num_queries": len(queries),
        "num_gallery": len(gallery),
        "scores": str(
            scores_file.relative_to(ROOT)
        ),
        "higher_is_better": True,
    }

    with run_file.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            run,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print("Done")
    print("----")
    print(f"Features : {features_file}")
    print(f"Scores   : {scores_file}")
    print(f"Run info : {run_file}")
    print(
        "Shape    : "
        f"({len(queries)}, "
        f"{len(gallery)})"
    )


if __name__ == "__main__":
    main()
