#!/usr/bin/env python3
"""Link the RCR image root containing val/ and test/ to data/gallery."""

import argparse
from pathlib import Path

from benchmark_data import describe_gallery_link, ensure_gallery_layout

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path, help="RCR dataset/data/raw/images directory")
    args = parser.parse_args()
    ensure_gallery_layout(ROOT, source_root=args.source)
    print(describe_gallery_link(ROOT))


if __name__ == "__main__":
    main()
