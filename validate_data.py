#!/usr/bin/env python3
"""Validate the imported RCR validation/test manifests and optional image files."""

import argparse
from pathlib import Path

from benchmark_data import SPLITS, load_jsonl, split_paths, verify_manifest

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-image-files", action="store_true")
    args = parser.parse_args()
    for split in SPLITS:
        metadata = verify_manifest(ROOT, split)
        if not args.skip_image_files:
            gallery, _ = split_paths(ROOT, split)
            for row in load_jsonl(gallery):
                path = ROOT / row["path"]
                if not path.is_file():
                    raise FileNotFoundError(f"Missing {split} image: {path}")
        print(f"{split}: {metadata['num_queries']:,} queries, {metadata['num_gallery']:,} images")


if __name__ == "__main__":
    main()
