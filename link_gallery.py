#!/usr/bin/env python3
"""Create a flat gallery of image symlinks from train, val and test folders."""

import argparse
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def link_gallery(source, gallery):
    source = source.expanduser().resolve()
    targets = {}
    counts = Counter()
    for split in ("train", "val", "test"):
        directory = source / split
        if not directory.is_dir():
            raise FileNotFoundError(directory)
        images = sorted(path for path in directory.iterdir()
                        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"})
        if not images:
            raise ValueError(f"No images found in {directory}")
        for image in images:
            if image.name in targets:
                raise ValueError(f"Duplicate image filename: {targets[image.name]} and {image}")
            targets[image.name] = image
        counts[split] = len(images)

    # Preflight before changing the old folder link. Never delete source images
    # or replace an existing real image file in the destination.
    if not gallery.is_symlink() and gallery.exists():
        if not gallery.is_dir():
            raise ValueError(f"Gallery is not a directory: {gallery}")
        for name in targets:
            destination = gallery / name
            if destination.exists() and not destination.is_symlink():
                raise FileExistsError(f"Existing non-symlink image: {destination}")

    if gallery.is_symlink():
        gallery.unlink()
    gallery.mkdir(parents=True, exist_ok=True)
    for name, target in targets.items():
        destination = gallery / name
        if destination.is_symlink():
            if destination.resolve() == target:
                continue
            destination.unlink()
        destination.symlink_to(target)
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path,
                        help="Parent directory containing train/, val/ and test/.")
    args = parser.parse_args()
    gallery = ROOT / "data/gallery"
    counts = link_gallery(args.source, gallery)
    for split, count in counts.items():
        print(f"{split}: {count:,} images")
    print(f"Flat gallery: {sum(counts.values()):,} image symlinks at {gallery}")


if __name__ == "__main__":
    main()
