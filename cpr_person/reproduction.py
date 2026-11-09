"""Admit reproduced weights only with matching external-data and run provenance."""

from __future__ import annotations

import json
import os
from pathlib import Path

from benchmark_data import load_jsonl, split_paths, verify_manifest
from .cache import ROOT, canonical_hash, read_json, sha256_file


def benchmark_images(root):
    hashes = set()
    for split in ("val", "test"):
        verify_manifest(root, split)
        gallery, _ = split_paths(root, split)
        for row in load_jsonl(gallery):
            path = Path(row["path"])
            hashes.add(sha256_file(path if path.is_absolute() else root / path))
    return hashes


def audit_word4per_data(data_root, benchmark_root=ROOT):
    data_root = Path(data_root)
    protected = benchmark_images(benchmark_root)
    annotations = {
        name: json.loads((data_root / name).read_text())
        for name in ("CUHK-PEDES/reid_raw.json", "query.json", "gallery.json")
    }
    paths = [
        data_root / "CUHK-PEDES/imgs" / row["file_path"]
        for row in annotations["CUHK-PEDES/reid_raw.json"]
        if row.get("split") == "train"
    ]
    paths += [
        data_root / row["file_path"]
        for name in ("query.json", "gallery.json")
        for row in annotations[name]
    ]
    hashes = sorted({sha256_file(path) for path in paths})
    if protected.intersection(hashes):
        raise ValueError(
            "Word4Per training/checkpoint-selection images overlap RCR val/test by bytes"
        )
    result = {
        "schema": "word4per-external-data-v1",
        "image_sha256s": hashes,
        "num_referenced_images": len(paths),
        "num_unique_images": len(hashes),
        "annotations": {name: sha256_file(data_root / name) for name in annotations},
        "checkpoint_selection_dataset": "external ITCPR",
        "target_benchmark_labels_used": False,
    }
    result["inventory_sha256"] = canonical_hash(result)
    return result


def admit_word4per(cfg, stage2, config_path, benchmark_root=ROOT):
    value = os.environ.get("WORD4PER_STAGE2_PROVENANCE") or cfg["checkpoint"].get(
        "reproduction_provenance"
    )
    path = (
        Path(value)
        if value
        else Path(stage2).parent / "word4per_cuhk_pedes_stage2_reproduction.json"
    )
    path = path if path.is_absolute() else ROOT / path
    record = read_json(path)
    if (
        not isinstance(record, dict)
        or record.get("official_commit") != cfg["source"]["commit"]
        or record.get("stage2_checkpoint_sha256") != sha256_file(stage2)
        or record.get("stage2_config_sha256") != sha256_file(config_path)
        or record.get("cpr_supervision") is not False
        or not record.get("stage1_sha256")
    ):
        raise ValueError(
            "Missing or incompatible Word4Per reproduction receipt; do not substitute another checkpoint"
        )
    audit = record.get("external_data_audit")
    if (
        not isinstance(audit, dict)
        or audit.get("schema") != "word4per-external-data-v1"
    ):
        raise ValueError(
            "Word4Per receipt lacks external training/selection image provenance"
        )
    payload = {key: value for key, value in audit.items() if key != "inventory_sha256"}
    if audit.get("inventory_sha256") != canonical_hash(payload) or not audit.get(
        "image_sha256s"
    ):
        raise ValueError("Invalid Word4Per external image inventory")
    if benchmark_images(benchmark_root).intersection(audit["image_sha256s"]):
        raise ValueError("Reproduction data overlaps current RCR val/test")
    return {
        "artifact": str(path),
        "artifact_sha256": sha256_file(path),
        "external_inventory_sha256": audit["inventory_sha256"],
        "checkpoint_sha256": record["stage2_checkpoint_sha256"],
        "stage2_config_sha256": record["stage2_config_sha256"],
        "selection": record["selection"],
    }
