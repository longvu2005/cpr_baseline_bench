#!/usr/bin/env python3
"""Download the OpenAI CLIP ViT-L/14 checkpoint used by this baseline."""

from __future__ import annotations

import argparse
import hashlib
import os
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CHECKPOINT_PATH = ROOT / "checkpoints/clip/ViT-L-14.pt"
CHECKPOINT_SHA256 = "b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836"
CHECKPOINT_URL = (
    "https://openaipublic.azureedge.net/clip/models/"
    f"{CHECKPOINT_SHA256}/ViT-L-14.pt"
)


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def download(force: bool) -> None:
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    relative_path = CHECKPOINT_PATH.relative_to(ROOT)

    if CHECKPOINT_PATH.is_file() and not force:
        print(f"[check] {relative_path}")
        actual = sha256_file(CHECKPOINT_PATH)
        if actual == CHECKPOINT_SHA256:
            print("[skip] checkpoint already exists and checksum is valid")
            return
        print("[warn] existing checkpoint checksum is invalid; downloading a clean copy")

    temp_path = CHECKPOINT_PATH.with_suffix(CHECKPOINT_PATH.suffix + ".part")
    temp_path.unlink(missing_ok=True)

    print(f"[download] {relative_path}")
    request = urllib.request.Request(
        CHECKPOINT_URL,
        headers={"User-Agent": "cpr-baseline-bench/1.0"},
    )
    digest = hashlib.sha256()

    try:
        with urllib.request.urlopen(request) as response, temp_path.open("wb") as handle:
            total = int(response.headers.get("Content-Length", "0"))
            downloaded = 0
            while True:
                chunk = response.read(8 * 1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
                digest.update(chunk)
                downloaded += len(chunk)
                if total:
                    percent = 100.0 * downloaded / total
                    print(
                        f"\r         {downloaded / 2**20:.1f}/{total / 2**20:.1f} MiB "
                        f"({percent:.1f}%)",
                        end="",
                    )
        if total:
            print()

        actual = digest.hexdigest()
        if actual != CHECKPOINT_SHA256:
            raise RuntimeError(
                "CLIP checkpoint checksum mismatch: "
                f"expected {CHECKPOINT_SHA256}, got {actual}"
            )
        os.replace(temp_path, CHECKPOINT_PATH)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise

    print(f"[ok] {relative_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download the OpenAI CLIP ViT-L/14 checkpoint for this baseline."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even when a valid checkpoint already exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    download(args.force)


if __name__ == "__main__":
    main()
