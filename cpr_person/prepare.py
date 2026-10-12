"""Prepare the one shared detector and CLIP selector, without a ReID download."""
from __future__ import annotations

import argparse

from .cache import resolve, sha256_file
from .detection import protocol_config


def prepare(force=False):
    from . import assets
    from .tokenizer_runtime import prepare_tokenizer
    cfg = protocol_config()
    d, s = cfg["detector"], cfg["selector"]
    checkout = assets.prepare_source(d["source"], "Grounding DINO")
    if sha256_file(checkout / d["config"]) != d["config_sha256"]:
        raise RuntimeError("Grounding DINO pinned config checksum mismatch")
    assets.download_stream(url=d["checkpoint_url"], path=resolve(d["checkpoint"]), force=force)
    assets.validate_groundingdino_checkpoint(resolve(d["checkpoint"]))
    # Pure-PyTorch attention is the single canonical implementation. Never
    # compile CUDA extensions merely to prepare a different retrieval method.
    assets.prewarm_groundingdino_runtime(cache_root=resolve(d["runtime_cache"]),
        marker=resolve(d["runtime_assets_marker"]), source_commit=d["source"]["commit"], force=force)
    assets.download_stream(url=s["checkpoint_url"], path=resolve(s["checkpoint"]), force=force,
                           expected_sha256=s["checkpoint_sha256"])
    prepare_tokenizer(s,force)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")
    prepare(parser.parse_args().force)


if __name__ == "__main__":
    main()
