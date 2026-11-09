"""Run actual frozen HF CLIP forwards independently of native method packages."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from cpr_person.cache import atomic_json, canonical_hash, sha256_file
from .scene_protocol import MODEL, REVISION, SPEC, assets, checkpoint_path
from .provenance import runtime_identity


class LetterboxProcessor:
    def __init__(self, processor, size=(224, 224)):
        self.processor, self.size = processor, tuple(size)

    def __call__(self, images, return_tensors="pt"):
        single = not isinstance(images, list)
        rows = [images] if single else images
        rows = [
            ImageOps.pad(
                image, self.size, method=Image.Resampling.BICUBIC, color=(0, 0, 0)
            )
            for image in rows
        ]
        return self.processor(
            images=rows[0] if single else rows,
            do_resize=False,
            do_center_crop=False,
            return_tensors=return_tensors,
        )


def prepare(directory, force=False):
    directory = checkpoint_path(directory)
    if not force:
        try:
            assets(directory)
            return
        except (RuntimeError, ValueError, OSError):
            pass
    from huggingface_hub import snapshot_download

    directory.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=MODEL,
        revision=REVISION,
        local_dir=str(directory),
        allow_patterns=["pytorch_model.bin", "*.json", "vocab.json", "merges.txt"],
        force_download=force,
    )
    files = [
        {
            "path": str(p.relative_to(directory)),
            "size": p.stat().st_size,
            "sha256": sha256_file(p),
        }
        for p in sorted(directory.rglob("*"))
        if p.is_file() and ".cache" not in p.parts and p.name != "prepared.json"
    ]
    marker = {"protocol": SPEC, "files": files}
    atomic_json(
        directory / "prepared.json", {**marker, "assets_sha256": canonical_hash(marker)}
    )
    assets(directory)


def encode(
    request, *, image_model=None, text_model=None, processor=None, tokenizer=None
):
    import torch
    from transformers import (
        AutoImageProcessor,
        CLIPTokenizerFast,
        CLIPTextModelWithProjection,
        CLIPVisionModelWithProjection,
    )

    directory, device = str(request["checkpoint"]), torch.device(request["device"])
    rows, batch = request["inputs"], int(request["batch_size"])
    if not rows or batch < 1:
        raise ValueError("Expected non-empty public inputs and positive batch size")
    if request["kind"] == "images":
        processor = processor or LetterboxProcessor(
            AutoImageProcessor.from_pretrained(
                directory, local_files_only=True, use_fast=False
            )
        )
        model = image_model or CLIPVisionModelWithProjection.from_pretrained(
            directory, local_files_only=True
        )
        if (
            model.config.image_size != 224
            or model.config.patch_size != 32
            or model.config.projection_dim != 512
        ):
            raise ValueError("Expected native CLIP ViT-B/32")
    elif request["kind"] == "texts":
        tokenizer = tokenizer or CLIPTokenizerFast.from_pretrained(
            directory, local_files_only=True
        )
        model = text_model or CLIPTextModelWithProjection.from_pretrained(
            directory, local_files_only=True
        )
        if (
            tokenizer.padding_side != "right"
            or model.config.max_position_embeddings != 77
            or model.config.projection_dim != 512
        ):
            raise ValueError("Expected RCR native CLIP text/tokenizer")
    else:
        raise ValueError("Expected images or texts")
    model = model.float().to(device).eval().requires_grad_(False)
    chunks = []
    with torch.inference_mode():
        for start in range(0, len(rows), batch):
            values = rows[start : start + batch]
            if request["kind"] == "images":
                images = []
                for path in values:
                    with Image.open(path) as image:
                        images.append(image.convert("RGB"))
                output = model(
                    pixel_values=processor(images=images)["pixel_values"].to(device)
                ).image_embeds
            else:
                tokens = tokenizer(
                    values,
                    padding=True,
                    truncation=True,
                    max_length=77,
                    return_tensors="pt",
                ).to(device)
                output = model(**tokens).text_embeds
            chunks.append(output.float().cpu().numpy())
    result = np.concatenate(chunks)
    if result.shape != (len(rows), 512) or not np.isfinite(result).all():
        raise ValueError("Invalid native global embeddings")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--request")
    args = parser.parse_args()
    if args.prepare:
        prepare(args.prepare, args.force)
        return
    if not args.request:
        parser.error("Supply --prepare or --request")
    request = json.loads(Path(args.request).read_text())
    if (
        request["assets"] != assets(request["checkpoint"])
        or request["runtime"] != runtime_identity()
    ):
        raise RuntimeError("Scene assets/runtime differ from the parent fingerprint")
    with Path(request["output"]).open("wb") as handle:
        np.save(handle, encode(request), allow_pickle=False)


if __name__ == "__main__":
    main()
