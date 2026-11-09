#!/usr/bin/env python3
"""Keep metadata writable while reading immutable weights from Kaggle inputs."""
import argparse
from pathlib import Path
import shutil

WEIGHTS = (
    ".pt",
    ".pth",
    ".pth.tar",
    ".ckpt",
    ".safetensors",
    ".bin",
    ".pkl",
    ".h5",
    ".msgpack",
)


def stage(source, destination):
    source, destination = Path(source).resolve(), Path(destination).absolute()
    if (
        not source.is_dir()
        or destination.is_symlink()
        or destination.is_relative_to(source)
        or source.is_relative_to(destination)
    ):
        raise ValueError("Use separate source mount and writable destination trees")
    destination.mkdir(parents=True, exist_ok=True)
    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source)
        if ".cache" in relative.parts or "__pycache__" in relative.parts:
            continue
        target = destination / relative
        if path.is_dir():
            if target.is_symlink():
                raise ValueError(
                    "Do not symlink metadata directories into read-only input"
                )
            target.mkdir(parents=True, exist_ok=True)
        elif path.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.is_symlink() and target.resolve() == path.resolve():
                continue
            if target.exists() or target.is_symlink():
                import hashlib

                def digest(p):
                    h = hashlib.sha256()
                    with p.open("rb") as f:
                        for data in iter(lambda: f.read(8 * 1024 * 1024), b""):
                            h.update(data)
                    return h.digest()

                if target.is_file() and digest(target) == digest(path):
                    continue
                raise FileExistsError(f"Different existing artifact: {target}")
            if path.name.endswith(WEIGHTS):
                target.symlink_to(path.resolve())
            else:
                shutil.copyfile(path, target)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, default=Path("checkpoints"))
    args = parser.parse_args()
    stage(args.source, args.destination)
