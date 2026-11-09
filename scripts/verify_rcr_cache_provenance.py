#!/usr/bin/env python3
"""Compare every legacy cache tensor with real frozen-model outputs before binding images."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def verify_copy(index, paths, forward, source, destination, kind, runtime):
    import numpy as np
    import torch
    from PIL import Image
    from cpr_person.cache import sha256_file, atomic_file

    if source.resolve() == destination.resolve():
        raise ValueError(
            "Keep the original cache immutable; choose another output directory"
        )
    source_hash = sha256_file(source)
    hashes = [sha256_file(path) for path in paths]
    maximum = 0.0
    tolerance = {"rtol": 1e-5, "atol": 1e-5 if kind == "scene" else 1e-4}
    for i, path in enumerate(paths):
        if kind == "scene":
            expected = index["global_features"][i].numpy()
        else:
            with Image.open(path) as image:
                if tuple(index["image_sizes"][i].tolist()) != image.size:
                    raise ValueError("Detection image dimensions differ")
            a, b = index["offsets"][i : i + 2].tolist()
            expected = index["boxes_pixel"][a:b].numpy()
        actual = np.asarray(forward(path))
        if actual.shape != expected.shape or not np.isfinite(actual).all():
            raise ValueError(f"Native {kind} tensor shape/values differ: {path}")
        try:
            np.testing.assert_allclose(actual, expected, **tolerance)
        except AssertionError as error:
            raise ValueError(
                f"Native {kind} tensor differs; do not bind this cache to these images: {path}"
            ) from error
        maximum = max(maximum, float(np.abs(actual - expected).max(initial=0)))
    if (
        hashes != [sha256_file(path) for path in paths]
        or sha256_file(source) != source_hash
    ):
        raise RuntimeError("Source images/index changed during verification")
    result = {
        **index,
        "image_content_sha256": hashes,
        "image_verification": {
            "source_index_sha256": source_hash,
            "kind": kind,
            "all_original_tensors_compared": True,
            "num_images": len(paths),
            "max_absolute_error": maximum,
            "tolerance": tolerance,
            "runtime": runtime,
        },
    }
    with atomic_file(destination) as handle:
        torch.save(result, handle)
    return result["image_verification"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["rcr-repo", "rcr-final-dir", "image-root", "output"]:
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--detections", type=Path)
    parser.add_argument("--scene", type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if not args.detections and not args.scene:
        parser.error("Supply --detections or --scene")
    sys.path.insert(0, str(args.rcr_repo.resolve() / "src"))
    import gc
    import torch
    from PIL import Image
    from rcr.common.data import load_rcr_inputs
    from rcr.common.detection import detect_persons
    from cpr_person.rcr_cache import read_rcr_index
    from cpr_person.detection import protocol_config
    from cpr_baselines.provenance import execution_identity
    from cpr_baselines.scene_protocol import checkpoint_path, assets, MODEL, REVISION
    from cpr_baselines.scene_worker import encode

    data = load_rcr_inputs(args.rcr_final_dir, args.image_root)
    paths = [data.image_path(image_id) for image_id in data.gallery_ids]
    if args.detections:
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

        source = (
            args.detections / "index.pt"
            if args.detections.is_dir()
            else args.detections
        )
        index, _ = read_rcr_index(source)
        if index["image_ids"] != data.gallery_ids:
            raise ValueError("Detection gallery order differs")
        cfg = index["detector"]
        revision = cfg.get("revision", protocol_config()["detector"]["revision"])
        processor = AutoProcessor.from_pretrained(cfg["model"], revision=revision)
        model = (
            AutoModelForZeroShotObjectDetection.from_pretrained(
                cfg["model"], revision=revision
            )
            .float()
            .to(args.device)
            .eval()
        )

        @torch.inference_mode()
        def forward(path):
            with Image.open(path) as image:
                return (
                    detect_persons(
                        model,
                        processor,
                        image.convert("RGB"),
                        args.device,
                        cfg["threshold"],
                        cfg["text_threshold"],
                    )
                    .float()
                    .cpu()
                    .numpy()
                )

        print(
            verify_copy(
                index,
                paths,
                forward,
                source,
                args.output / "detections/index.pt",
                "detections",
                {**execution_identity(__file__), "revision": revision},
            ),
            flush=True,
        )
        del model, processor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if args.scene:
        from transformers import CLIPVisionModelWithProjection

        source = args.scene / "index.pt" if args.scene.is_dir() else args.scene
        if (source.parent / ".building").exists():
            raise ValueError("Scene cache is incomplete")
        index = torch.load(source, map_location="cpu", weights_only=True, mmap=True)
        if (
            index.get("layout") != "clip-scene-v1"
            or index["image_ids"] != data.gallery_ids
            or index.get("image_encoder")
            != {"model": MODEL, "revision": REVISION, "scene_size": [224, 224]}
            or index.get("global_dim") != 512
            or index.get("scene_dim") != 768
            or tuple(index.get("patch_hw", ())) != (7, 7)
        ):
            raise ValueError("Expected the complete native RCR HF scene cache")
        directory = checkpoint_path()
        identity = assets(directory)
        model = (
            CLIPVisionModelWithProjection.from_pretrained(
                str(directory), local_files_only=True
            )
            .float()
            .to(args.device)
            .eval()
        )

        def forward(path):
            return encode(
                {
                    "kind": "images",
                    "checkpoint": str(directory),
                    "device": args.device,
                    "inputs": [str(path)],
                    "batch_size": 1,
                },
                image_model=model,
            )[0]

        print(
            verify_copy(
                index,
                paths,
                forward,
                source,
                args.output / "scene/index.pt",
                "scene",
                {**execution_identity(__file__), "scene_assets": identity},
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
