"""Prepare the shared CLIP selector for the configured detection source."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from .cache import ROOT, cache_lock, resolve
from .rcr_cache import rcr_index_path
from .detection import protocol_config


def prepare(force=False):
    from . import assets
    from .clip_model import prepare_openai_clip
    from .tokenizer_runtime import prepare_tokenizer, validate_tokenizer
    cfg = protocol_config()
    d, s = cfg["detector"], cfg["selector"]
    if rcr_index_path() is None:
        assets.prepare_detector(d, force)
    prepare_openai_clip(s["model"], s["checkpoint"], force)
    try:
        if not force:
            validate_tokenizer(s)
            return
    except RuntimeError:
        pass
    with cache_lock(resolve(s["tokenizer_runtime_dir"]).with_name(".selector-tokenizer")):
        prepare_tokenizer(s, force)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if rcr_index_path() is not None:
        prepare(args.force)
        return
    from .runtime import detector_python, detector_env
    python = detector_python()
    if Path(sys.executable).absolute() != Path(python).absolute():
        subprocess.run([python, "-m", "cpr_person.prepare", *(["--force"] if args.force else [])],
                       cwd=ROOT, env=detector_env(python), check=True)
        return
    prepare(args.force)


if __name__ == "__main__":
    main()
