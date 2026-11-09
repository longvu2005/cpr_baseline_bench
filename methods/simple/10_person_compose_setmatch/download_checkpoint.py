#!/usr/bin/env python3
"""Prepare S10 external artifacts.

S10 needs:
- the RCR detection index and shared CLIP selector
- OpenAI CLIP ViT-L/14 for S10's person-crop composition
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark_progress import PhaseTracker  # noqa: E402
from cpr_person.adapter import prepare_shared
from cpr_person.cache import resolve
from cpr_person.clip_model import prepare_openai_clip

DEFAULT_CONFIG = Path(__file__).resolve().parent / "config.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare S10 artifacts.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--force", action="store_true", help="Reprepare the selector and person composition checkpoints.")
    args = parser.parse_args()
    cfg = yaml.safe_load(resolve(args.config).read_text(encoding="utf-8"))

    tracker = PhaseTracker("s10_prepare", total=2)
    tracker.advance("Prepare the shared RCR CLIP selector")
    prepare_shared(args.force)
    tracker.advance("Prepare CLIP ViT-L/14 for person-crop composition")
    prepare_openai_clip(cfg["clip"]["name"], cfg["clip"]["checkpoint"], args.force)
    tracker.finish()


if __name__ == "__main__":
    main()
