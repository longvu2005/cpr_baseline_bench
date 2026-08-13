#!/usr/bin/env python3
"""Prepare all S9 external artifacts by delegating to S5 and S8.

S9 introduces no new checkpoints beyond its two source branches:
- S5: Grounding DINO + CLIP-ReID - Set
- S8: Qwen-Image-Edit-2509 + CLIP ViT-L/14

This preparer simply runs the method-local checkpoint preparers for both baselines
so S9 can later reuse their runtime assets and caches.
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


def run_step(title: str, command: list[str]) -> None:
    print(f"$ {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare S9 external artifacts.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--force", action="store_true", help="Pass --force to delegated preparers.")
    args = parser.parse_args()
    _ = Path(args.config)  # reserved for future extension; kept for interface consistency.

    tracker = PhaseTracker("s9_prepare", total=2)
    tracker.advance("Prepare S5 artifacts")
    cmd = [sys.executable, "-u", "methods/simple/05_groundingdino_clipreid_set/download_checkpoint.py"]
    if args.force:
        cmd.append("--force")
    run_step("S5", cmd)

    tracker.advance("Prepare S8 artifacts")
    cmd = [sys.executable, "-u", "methods/simple/08_qwen_image_edit_clip/download_checkpoint.py"]
    if args.force:
        cmd.append("--force")
    run_step("S8", cmd)
    tracker.finish()


if __name__ == "__main__":
    main()
