#!/usr/bin/env python3
"""Warm canonical grounding once before upgrading to Qwen dependencies."""
from pathlib import Path
import argparse
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    try:
        from cpr_baselines.common import load_config
        from cpr_baselines.person_cache import require_shared_context
        require_shared_context(load_config(Path(__file__).with_name("config.yaml")), "cpu")
        print("[skip] identical verified shared grounding/localization cache is ready", flush=True)
        return
    except (RuntimeError, FileNotFoundError, ImportError):
        pass
    if args.check_only:
        raise SystemExit("Shared grounding cache is missing. Prepare it first or run without --skip-install.")
    # Separate subprocesses avoid stale Transformers imports in a notebook.
    # Modern setuptools removed pkg_resources used by the pinned CLIP setup.py;
    # use a compatible builder without creating another model environment.
    commands = [
        [sys.executable, "-m", "pip", "install", "setuptools>=65,<81", "wheel"],
        [sys.executable, "-m", "pip", "install", "--no-build-isolation", "-r", "cpr_person/requirements.txt"],
        [sys.executable, "-m", "cpr_person.prepare"],
        [sys.executable, "-m", "cpr_baselines.person_cache"],
    ]
    for command in commands:
        subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
