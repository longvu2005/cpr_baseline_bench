#!/usr/bin/env python3
"""Prepare all external artifacts for S6 Grounding DINO + CLIP-ReID - Set + Text.

S6 reuses S5's shared RCR boxes and pinned CLIP-ReID model. It prepares
RCR's official OpenAI CLIP ViT-B/32 for the text branch. Inference performs no network download.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cpr_baselines.scene_clip import prepare_scene_clip
from cpr_person.cache import resolve

S5_PREPARER = ROOT / "methods/simple/reid_set/download_checkpoint.py"
DEFAULT_CONFIG = Path(__file__).with_name("config.yaml")


def prepare_s5(force: bool) -> None:
    if not S5_PREPARER.is_file():
        raise FileNotFoundError(
            "S6 reuses the S5 detector/ReID implementation, but "
            f"{S5_PREPARER.relative_to(ROOT)} is missing. Install/apply S5 first."
        )
    command = [sys.executable, "-u", str(S5_PREPARER.relative_to(ROOT))]
    if force:
        command.append("--force")
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare S6 inference artifacts")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    cfg = yaml.safe_load(resolve(args.config).read_text(encoding="utf-8"))
    print("[1/2] Prepare S5 Grounding DINO + CLIP-ReID artifacts", flush=True)
    prepare_s5(args.force)
    print("[2/2] Prepare OpenAI CLIP ViT-B/32", flush=True)
    prepare_scene_clip(cfg["clip_text"], args.force)


if __name__ == "__main__":
    main()
