#!/usr/bin/env python3
"""Prepare the shared frozen HF CLIP scene checkpoint."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cpr_baselines.scene_clip import main


if __name__ == "__main__":
    main(Path(__file__).with_name("config.yaml"))
