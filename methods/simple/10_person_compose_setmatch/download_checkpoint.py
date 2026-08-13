#!/usr/bin/env python3
"""Prepare S10 external artifacts.

S10 needs:
- the shared Grounding DINO protocol assets from S5
- the OpenAI CLIP ViT-L/14 checkpoint used by the CLIP baselines
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark_progress import PhaseTracker  # noqa: E402

DEFAULT_CONFIG = Path(__file__).resolve().parent / "config.yaml"


def run_step(command: list[str]) -> None:
    print(f"$ {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare S10 artifacts.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--force", action="store_true", help="Pass --force to delegated preparers.")
    args = parser.parse_args()
    _ = Path(args.config)

    tracker = PhaseTracker("s10_prepare", total=2)
    tracker.advance("Prepare shared Grounding DINO protocol assets")
    cmd = [sys.executable, "-u", "methods/simple/05_reid_set/download_checkpoint.py"]
    if args.force:
        cmd.append("--force")
    run_step(cmd)

    tracker.advance("Prepare CLIP ViT-L/14 checkpoint")
    cmd = [sys.executable, "-u", "methods/simple/02_clip_text/download_checkpoint.py"]
    if args.force:
        cmd.append("--force")
    run_step(cmd)
    tracker.finish()


if __name__ == "__main__":
    main()
