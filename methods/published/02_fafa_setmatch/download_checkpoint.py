#!/usr/bin/env python3
"""Download the official released FAFA/SynCPR checkpoint used by this baseline."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CHECKPOINT_PATH = ROOT / "checkpoints/fafa/tuned_recall_at1_step.pt"
GOOGLE_DRIVE_ID = "1Bf2Ia7zmxx5k3Dj-nRr3CLbAqc_zkM0y"


def ensure_gdown() -> None:
    try:
        import gdown  # noqa: F401
        return
    except ImportError:
        pass

    print("[setup] installing gdown for the official FAFA checkpoint")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "gdown>=5.2,<6"],
        check=True,
    )


def download(force: bool) -> None:
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    relative_path = CHECKPOINT_PATH.relative_to(ROOT)

    if CHECKPOINT_PATH.is_file() and CHECKPOINT_PATH.stat().st_size > 0 and not force:
        print(f"[skip] {relative_path} already exists")
        return

    ensure_gdown()
    temp_path = CHECKPOINT_PATH.with_suffix(CHECKPOINT_PATH.suffix + ".part")
    temp_path.unlink(missing_ok=True)

    print(f"[download] {relative_path}")
    try:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "gdown",
                GOOGLE_DRIVE_ID,
                "-O",
                str(temp_path),
            ],
            check=True,
        )
        if not temp_path.is_file() or temp_path.stat().st_size == 0:
            raise RuntimeError("FAFA download completed without producing a valid file")
        os.replace(temp_path, CHECKPOINT_PATH)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise

    print(f"[ok] {relative_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download the official released FAFA/SynCPR checkpoint for this baseline."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even when the checkpoint already exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    download(args.force)


if __name__ == "__main__":
    main()
