#!/usr/bin/env python3
"""Prepare all external artifacts for S6 Grounding DINO + CLIP-ReID - Set + Text.

S6 intentionally reuses S5's pinned Grounding DINO and CLIP-ReID preparation,
then prepares the official OpenAI CLIP ViT-L/14 checkpoint used by the text
branch. Inference performs no network download.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
S5_PREPARER = ROOT / "methods/simple/05_groundingdino_clipreid_set/download_checkpoint.py"
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


def prepare_clip_l14(force: bool) -> None:
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    rel = CHECKPOINT_PATH.relative_to(ROOT)
    if CHECKPOINT_PATH.is_file() and not force:
        actual = sha256_file(CHECKPOINT_PATH)
        if actual == CHECKPOINT_SHA256:
            print(f"[skip] {rel} (checksum valid)", flush=True)
            return
        print(f"[warn] {rel} checksum invalid; replacing", flush=True)

    temp = CHECKPOINT_PATH.with_suffix(CHECKPOINT_PATH.suffix + ".part")
    temp.unlink(missing_ok=True)
    request = urllib.request.Request(
        CHECKPOINT_URL, headers={"User-Agent": "cpr-baseline-bench/1.0"}
    )
    digest = hashlib.sha256()
    print(f"[download] {rel}", flush=True)
    try:
        with urllib.request.urlopen(request) as response, temp.open("wb") as handle:
            total = int(response.headers.get("Content-Length", "0"))
            done = 0
            while True:
                chunk = response.read(8 * 1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
                digest.update(chunk)
                done += len(chunk)
                if total:
                    print(
                        f"\r         {done / 2**20:.1f}/{total / 2**20:.1f} MiB "
                        f"({100.0 * done / total:.1f}%)",
                        end="",
                        flush=True,
                    )
        if total:
            print(flush=True)
        actual = digest.hexdigest()
        if actual != CHECKPOINT_SHA256:
            raise RuntimeError(
                f"CLIP ViT-L/14 checksum mismatch: expected {CHECKPOINT_SHA256}, got {actual}"
            )
        os.replace(temp, CHECKPOINT_PATH)
    except Exception:
        temp.unlink(missing_ok=True)
        raise
    print(f"[ok] {rel}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare S6 inference artifacts")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print("[1/2] Prepare S5 Grounding DINO + CLIP-ReID artifacts", flush=True)
    prepare_s5(args.force)
    print("[2/2] Prepare OpenAI CLIP ViT-L/14", flush=True)
    prepare_clip_l14(args.force)


if __name__ == "__main__":
    main()
