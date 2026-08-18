#!/usr/bin/env python3
"""Generate CPR gallery captions with WISER's released BLIP2-T5 step-1 setup."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import torch
import yaml
from PIL import Image
from tqdm import tqdm
import lavis

ROOT = Path(__file__).resolve().parents[3]
METHOD_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = METHOD_DIR / "config.yaml"


def load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected YAML mapping: {path}")
    return value


def resolve(value: str) -> Path:
    p = Path(value)
    return (p if p.is_absolute() else ROOT / p).resolve()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise TypeError(f"{path}:{lineno}: expected object")
            rows.append(row)
    return rows


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def gallery_path(row: dict[str, Any], index: int) -> Path:
    value = row.get("path")
    if not isinstance(value, str) or not value.strip():
        raise KeyError(f"Gallery row {index}: missing path")
    p = resolve(value)
    if not p.is_file():
        raise FileNotFoundError(p)
    return p


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()
    config_path = resolve(args.config)
    cfg = load_yaml(config_path)

    gallery_manifest = resolve(str(cfg["data"]["gallery_manifest"]))
    output = resolve(str(cfg["cache"]["captions"]))
    meta_path = resolve(str(cfg["cache"]["captions_meta"]))
    gallery = load_jsonl(gallery_manifest)
    expected_meta = {
        "schema": 1,
        "gallery_manifest_sha256": sha256(gallery_manifest),
        "captioner": cfg["models"]["captioner"],
    }

    if output.is_file() and meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            rows = load_jsonl(output)
            if meta == expected_meta and len(rows) == len(gallery):
                ids = [r.get("image_id") for r in rows]
                if ids == [g.get("image_id") for g in gallery]:
                    print(f"[cache] {output}")
                    return
        except Exception:
            pass

    device = torch.device(str(cfg["runtime"]["caption_device"]) if torch.cuda.is_available() else "cpu")
    cap = cfg["models"]["captioner"]
    model, processors, _ = lavis.models.load_model_and_preprocess(
        name=str(cap["name"]),
        model_type=str(cap["model_type"]),
        is_eval=True,
        device=device,
    )
    model = model.float()
    model.maybe_autocast = lambda dtype=None: torch.no_grad()
    prompt = str(cap["prompt"])

    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".part")
    with tmp.open("w", encoding="utf-8") as handle:
        for gi, row in enumerate(tqdm(gallery, desc="WISER BLIP2 gallery captions", unit="image")):
            path = gallery_path(row, gi)
            with Image.open(path) as image:
                tensor = processors["eval"](image.convert("RGB"))
            tensor = tensor.unsqueeze(0).to(device)
            with torch.no_grad():
                generated = model.generate({"image": tensor, "prompt": prompt})
            caption = str(generated[0])
            payload = {"image_id": row["image_id"], "caption": caption}
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            handle.flush()
    os.replace(tmp, output)
    write_json(meta_path, expected_meta)
    print(f"Saved {len(gallery):,} captions -> {output}")


if __name__ == "__main__":
    main()
