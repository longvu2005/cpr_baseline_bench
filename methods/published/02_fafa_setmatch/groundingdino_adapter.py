#!/usr/bin/env python3
"""Grounding DINO worker, isolated from FAFA's distinct Hugging Face cache.

Reuse S5's pinned-source, extension, offline-assets and box-filter helpers.
Only detector assets are prepared; CLIP-ReID is neither loaded nor downloaded.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader, Dataset

os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from benchmark_progress import progress_bar


def load_s5(filename: str):
    name = "fafa_s5_" + Path(filename).stem
    if name in sys.modules:
        return sys.modules[name]
    path = ROOT / "methods/simple/05_reid_set" / filename
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load shared Grounding DINO helper: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def shared_config(detector_cfg: dict) -> dict:
    if detector_cfg.get("backend") != "groundingdino_swint_ogc":
        raise ValueError("FAFA requires localization.detector.backend: groundingdino_swint_ogc")
    for key in ("box_threshold", "text_threshold", "score_threshold", "nms_iou_threshold"):
        value = float(detector_cfg[key])
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"detector.{key} must be in [0, 1]")
    if str(detector_cfg["text_prompt"]).strip().lower().rstrip(".") != "person":
        raise ValueError("The FAFA detector prompt must be 'person'")
    if float(detector_cfg["min_box_size_px"]) <= 0:
        raise ValueError("detector.min_box_size_px must be positive")
    cap = detector_cfg.get("max_persons_per_image")
    if cap is not None and int(cap) <= 0:
        raise ValueError("detector.max_persons_per_image must be null or positive")
    return {"source": {"groundingdino": detector_cfg["source"]}, "detector": detector_cfg}


def preflight(detector_cfg: dict) -> tuple[Path, Path, Path]:
    """Validate assets without importing transformers into the parent process."""
    s5 = load_s5("run.py")
    cfg = shared_config(detector_cfg)
    checkout = s5.ensure_clean_pinned_source(detector_cfg["source"], "Grounding DINO")
    config = s5.require_file(checkout / detector_cfg["config"], "Grounding DINO config")
    checkpoint = s5.require_file(s5.resolve_path(detector_cfg["checkpoint"]), "Grounding DINO checkpoint")
    # HF constants are initialized on import. Restore FAFA's environment here;
    # the actual detector runs in a subprocess with its own HF constants.
    keys = ("HF_HOME", "XDG_CACHE_HOME", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "TOKENIZERS_PARALLELISM")
    previous = {key: os.environ.get(key) for key in keys}
    try:
        s5.configure_groundingdino_offline(cfg)
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    return checkout, config, checkpoint


def prepare(detector_cfg: dict, force: bool) -> None:
    shared_config(detector_cfg)
    s5 = load_s5("download_checkpoint.py")
    checkout = s5.prepare_source(detector_cfg["source"], "Grounding DINO")
    if not (checkout / detector_cfg["config"]).is_file():
        raise FileNotFoundError(checkout / detector_cfg["config"])
    # Avoid repeating a failed compiler build on every benchmark invocation.
    attempt_path = s5.resolve_path(detector_cfg["runtime_cache"]) / "extension_attempt.json"
    identity = {"commit": detector_cfg["source"]["commit"], "torch": torch.__version__,
                "cuda": torch.version.cuda, "python": sys.version,
                "cuda_home": os.environ.get("CUDA_HOME"), "arch": os.environ.get("TORCH_CUDA_ARCH_LIST")}
    try:
        previous = json.loads(attempt_path.read_text())
    except (OSError, ValueError):
        previous = None
    if force or previous != identity:
        s5.prepare_groundingdino_extension(checkout)
        attempt_path.parent.mkdir(parents=True, exist_ok=True)
        attempt_path.write_text(json.dumps(identity, sort_keys=True) + "\n")
    else:
        print("[skip] Grounding DINO extension preparation already attempted for this runtime", flush=True)
    checkpoint = s5.resolve_path(detector_cfg["checkpoint"])
    s5.download_stream(url=detector_cfg["checkpoint_url"], path=checkpoint, force=force)
    s5.validate_groundingdino_checkpoint(checkpoint)
    s5.prewarm_groundingdino_runtime(
        cache_root=s5.resolve_path(detector_cfg["runtime_cache"]),
        marker=s5.resolve_path(detector_cfg["runtime_assets_marker"]),
        source_commit=detector_cfg["source"]["commit"], force=force,
    )


def recover_prefix(path: Path, gallery: list[dict]) -> int:
    """Keep only complete, ordered JSONL rows after a canceled detection run."""
    if not path.is_file():
        return 0
    count, valid_end = 0, 0
    with path.open("rb+") as handle:
        while line := handle.readline():
            try:
                row = json.loads(line)
                if (count >= len(gallery) or row.get("image_id") != gallery[count]["image_id"]
                        or not isinstance(row.get("boxes"), list) or not row["boxes"]
                        or not line.endswith(b"\n")):
                    break
                for box in row["boxes"]:
                    xyxy = box["xyxy"]
                    if len(xyxy) != 4 or not all(float("-inf") < float(x) < float("inf") for x in xyxy):
                        raise ValueError("Invalid detection box")
                    if xyxy[2] <= xyxy[0] or xyxy[3] <= xyxy[1]:
                        raise ValueError("Invalid detection box")
                    if not float("-inf") < float(box["score"]) < float("inf"):
                        raise ValueError("Invalid detection score")
            except (ValueError, KeyError, TypeError, AttributeError):
                break
            count += 1
            valid_end = handle.tell()
        handle.truncate(valid_end)
    return count


class DetectionImages(Dataset):
    def __init__(self, gallery: list[dict], start: int):
        self.gallery, self.start = gallery, start

    def __len__(self):
        return len(self.gallery) - self.start

    def __getitem__(self, index: int):
        from groundingdino.util.inference import load_image
        gi = index + self.start
        image, tensor = load_image(str(ROOT / self.gallery[gi]["path"]))
        height, width = image.shape[:2]
        return tensor, width, height, gi


def collate_detection_images(rows):
    # Grounding DINO creates its own NestedTensor with padding and masks.
    tensors, widths, heights, owners = zip(*rows)
    return list(tensors), list(widths), list(heights), list(owners)


def split_detection_batch(tensors, widths, heights, owners, max_padding_ratio=1.3):
    """Keep aspect-mismatched scenes from doubling padded detector work."""
    start = 0
    while start < len(tensors):
        end = start + 1
        while end < len(tensors):
            shapes = [x.shape[-2:] for x in tensors[start:end + 1]]
            padded = max(h for h, w in shapes) * max(w for h, w in shapes) * len(shapes)
            actual = sum(h * w for h, w in shapes)
            if padded / actual > max_padding_ratio:
                break
            end += 1
        yield tensors[start:end], widths[start:end], heights[start:end], owners[start:end]
        start = end


@torch.no_grad()
def detect(cfg: dict, gallery: list[dict], cache_path: Path, device: torch.device) -> None:
    detector_cfg = cfg["localization"]["detector"]
    s5 = load_s5("run.py")
    detector_protocol = shared_config(detector_cfg)
    checkout, config, checkpoint = preflight(detector_cfg)
    s5.configure_groundingdino_offline(detector_protocol)
    attention_backend = s5.configure_groundingdino_source(checkout)
    journal = cache_path.with_name(cache_path.name + ".partial")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    start = recover_prefix(journal, gallery)
    if start:
        print(f"[resume] Grounding DINO: {start:,}/{len(gallery):,} images already detected", flush=True)
    if start < len(gallery):
        from groundingdino.util.inference import preprocess_caption
        from groundingdino.models import build_model
        from groundingdino.util.slconfig import SLConfig
        from groundingdino.util.misc import clean_state_dict, nested_tensor_from_tensor_list
        args = SLConfig.fromfile(str(config))
        args.device = str(device)
        # Gradient checkpointing only saves training activations; frozen
        # inference should not enter those wrappers for every image batch.
        args.use_checkpoint = False
        args.use_transformer_ckpt = False
        detector = build_model(args)
        # Grounding DINO's released checkpoint may include argparse.Namespace;
        # explicitly preserve the upstream loader's behavior on PyTorch >= 2.6.
        checkpoint_data = torch.load(checkpoint, map_location="cpu", weights_only=False)
        detector.load_state_dict(clean_state_dict(checkpoint_data["model"]), strict=False)
        del checkpoint_data
        detector = detector.to(device).eval()
        batch_size = int(cfg["runtime"].get("detector_batch_size", 2))
        workers = int(cfg["runtime"].get("detector_num_workers", 4))
        if batch_size <= 0 or workers < 0:
            raise ValueError("detector_batch_size must be positive and detector_num_workers nonnegative")
        loader = DataLoader(DetectionImages(gallery, start), batch_size=batch_size,
                            num_workers=workers, shuffle=False, pin_memory=device.type == "cuda",
                            collate_fn=collate_detection_images)
        caption = preprocess_caption(detector_cfg["text_prompt"])
        with journal.open("a", encoding="utf-8") as handle:
            for tensors, widths, heights, owners in progress_bar(
                    loader, desc="Grounding DINO person detection", total=len(loader), unit="batch"):
                for tensors, widths, heights, owners in split_detection_batch(tensors, widths, heights, owners):
                    tensors = [x.to(device, non_blocking=device.type == "cuda") for x in tensors]
                    # Same forward/box-threshold rule as upstream predict(), batched.
                    # The text threshold only decodes phrases, which FAFA never uses.
                    outputs = detector(nested_tensor_from_tensor_list(tensors), captions=[caption] * len(tensors))
                    confidences = outputs["pred_logits"].sigmoid().max(dim=-1).values.cpu()
                    boxes = outputs["pred_boxes"].cpu()
                    for row, width, height, gi in zip(range(len(tensors)), widths, heights, owners):
                        keep = confidences[row] > float(detector_cfg["box_threshold"])
                        xyxy, scores = s5.filter_person_boxes(
                            boxes_cxcywh=boxes[row, keep], confidences=confidences[row, keep],
                            width=width, height=height, cfg=detector_protocol)
                        candidates = [{"xyxy": box.tolist(), "score": float(score), "fallback": False}
                                      for box, score in zip(xyxy, scores)]
                        if not candidates:
                            candidates = [{"xyxy": [0.0, 0.0, float(width), float(height)],
                                           "score": -1.0, "fallback": True}]
                        handle.write(json.dumps({"image_id": gallery[gi]["image_id"], "boxes": candidates}) + "\n")
                handle.flush()
        del detector
    meta = {"checkpoint": str(checkpoint.relative_to(ROOT)) if checkpoint.is_relative_to(ROOT) else str(checkpoint),
            "attention_backend": attention_backend, "resumed_images": start}
    meta_path = cache_path.with_suffix(".meta.json")
    temp_meta = meta_path.with_name(meta_path.name + ".partial")
    temp_meta.write_text(json.dumps(meta, indent=2) + "\n")
    os.replace(temp_meta, meta_path)
    os.replace(journal, cache_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--cache-path")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    if args.prepare:
        prepare(cfg["localization"]["detector"], args.force)
    else:
        if not args.cache_path:
            parser.error("--cache-path is required for detection")
        manifest = ROOT / cfg.get("data", {}).get("gallery_manifest", "data/gallery.jsonl")
        gallery = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
        detect(cfg, gallery, Path(args.cache_path), torch.device(args.device))


if __name__ == "__main__":
    main()
