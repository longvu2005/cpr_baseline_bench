"""Bind all WISER cache stages to the same public inputs and prepared models."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from benchmark_data import gallery_image_digest, load_jsonl
from cpr_baselines.provenance import execution_identity, probe_runtime
from cpr_person.cache import canonical_hash, sha256_file


def namespace_caches(cfg, root):
    def path(value):
        value = Path(value)
        return value if value.is_absolute() else root / value

    gallery = path(cfg["data"]["gallery_manifest"])
    queries = path(cfg["data"]["query_manifest"])
    images = gallery_image_digest(root, load_jsonl(gallery))
    fingerprint = canonical_hash({
        "execution": execution_identity(__file__)["implementation"],
        "runtimes": {name: probe_runtime(path(cfg[name]["python"])) for name in ("isolated_env", "caption_env") if name in cfg},
        "config": cfg,
        "gallery_sha256": sha256_file(gallery),
        "queries_sha256": sha256_file(queries),
        "gallery_image_content_sha256": images,
        "prepared_sha256": sha256_file(path(cfg["checkpoint"]["prepared_marker"])),
        "implementation": {name: sha256_file(Path(__file__).with_name(name))
                           for name in ("run.py", "caption_gallery.py", "cache_protocol.py")},
    })
    result = deepcopy(cfg)
    directory = path(cfg["cache"]["dir"])
    for key, value in cfg["cache"].items():
        relative = path(value).relative_to(directory)
        result["cache"][key] = str(directory / fingerprint / relative)
    return result, images
