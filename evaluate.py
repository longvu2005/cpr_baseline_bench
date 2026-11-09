#!/usr/bin/env python3
"""Evaluate the complete, ordered RCR test split."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from statistics import fmean

import numpy as np

from benchmark_data import (CASE_TYPES, DATA_PROTOCOL, gallery_image_digest,
                            load_jsonl, split_paths, verify_manifest)
from benchmark_progress import PhaseTracker, progress_bar
from cpr_person.cache import atomic_json, canonical_hash, sha256_file
from cpr_person.protocol import require_aligned_run

ROOT = Path(__file__).resolve().parent
RUNS_DIR = ROOT / "runs"
OUTPUTS_DIR = ROOT / "outputs"


def validate_score_matrix(scores, expected_shape):
    if scores.shape != expected_shape:
        raise ValueError(f"Wrong score shape: got {scores.shape}, expected {expected_shape}")
    if not np.issubdtype(scores.dtype, np.floating):
        raise TypeError(f"scores.npy must have a floating dtype, got {scores.dtype}")
    for start in range(0, expected_shape[0], 256):
        if not np.isfinite(scores[start:start + 256]).all():
            raise ValueError(f"scores.npy contains NaN/Inf in query rows starting at {start}")


def rank_scores(row, self_index):
    """Keep canonical gallery order for ties and remove the reference."""
    order = np.argsort(-row, kind="stable")
    return order[order != self_index]


def positive_ranks(order, positives):
    if not positives:
        raise ValueError("A query has no positives")
    return np.flatnonzero(np.isin(order, list(positives))) + 1


def ap_from_ranks(ranks, total_positives):
    return float(np.sum(np.arange(1, len(ranks) + 1) / ranks) / total_positives)


def average_precision(order, positives):
    return ap_from_ranks(positive_ranks(order, positives), len(positives))


def measure(order, positives):
    ranks = positive_ranks(order, positives)
    return {
        "mAP": ap_from_ranks(ranks, len(positives)),
        "R@1": float(np.any(ranks <= 1)),
        "R@5": float(np.any(ranks <= 5)),
        "R@10": float(np.any(ranks <= 10)),
    }


def full_relevance(query, gallery_index):
    self_index = gallery_index[query["image_id"]]
    positives = {gallery_index[image_id] for image_id in query["full_positive_ids"]}
    if not positives or self_index in positives:
        raise ValueError(f"{query['query_id']}: invalid Full positives")
    return self_index, positives


def full_map(scores, gallery, queries):
    """Full-mAP used by validation selectors, with the official tie and self rules."""
    validate_score_matrix(scores, (len(queries), len(gallery)))
    gallery_index = {row["image_id"]: i for i, row in enumerate(gallery)}
    relevance = (full_relevance(query, gallery_index) for query in queries)
    return fmean(
        average_precision(rank_scores(row, self_index), positives)
        for row, (self_index, positives) in zip(scores, relevance, strict=True)
    )


def validate_run_order(run, gallery, queries):
    for key, rows, id_key in (("gallery_image_ids", gallery, "image_id"),
                              ("query_ids", queries, "query_id")):
        if key in run and run[key] != [row[id_key] for row in rows]:
            raise ValueError(f"run.json {key} differ from canonical manifest ordering")
    for key, count in (("num_gallery", len(gallery)), ("num_queries", len(queries))):
        if key in run and run[key] != count:
            raise ValueError(f"run.json {key} differs from the complete split")


def evaluate_scores(scores, gallery, queries):
    validate_score_matrix(scores, (len(queries), len(gallery)))
    gallery_index = {row["image_id"]: i for i, row in enumerate(gallery)}
    if len(gallery_index) != len(gallery):
        raise ValueError("Duplicate gallery image IDs")
    images_by_identity = defaultdict(set)
    for index, row in enumerate(gallery):
        for identity in row["person_ids"]:
            images_by_identity[identity].add(index)

    all_metrics = defaultdict(list)
    case_metrics = {case: defaultdict(list) for case in CASE_TYPES}
    case_fields = ("ID-mAP", "ID-R@1", "Full-mAP", "Full-R@1")
    for row, query in progress_bar(
        zip(scores, queries, strict=True), desc="Evaluate queries", total=len(queries), unit="query"
    ):
        self_index, full_positives = full_relevance(query, gallery_index)
        if not query["target_ids"]:
            raise ValueError(f"{query['query_id']}: empty identity list")
        id_positives = set.intersection(*(images_by_identity[i] for i in query["target_ids"]))
        id_positives.discard(self_index)
        if not full_positives <= id_positives:
            raise ValueError(f"{query['query_id']}: Full positives must also be ID positives")
        order = rank_scores(row, self_index)
        metrics = {}
        for prefix, positives in (("ID", id_positives), ("Full", full_positives)):
            metrics.update({f"{prefix}-{key}": value for key, value in measure(order, positives).items()})
        for key, value in metrics.items():
            all_metrics[key].append(value)
        for key in case_fields:
            case_metrics[query["case"]][key].append(metrics[key])

    overall = {key: fmean(values) for key, values in all_metrics.items()}
    cases = {
        case: {"num_queries": len(values["Full-mAP"]), **{key: fmean(rows) for key, rows in values.items()}}
        for case, values in case_metrics.items() if values
    }
    # RCR gives every present case equal weight, independently of query count.
    overall["Macro-Full-mAP"] = fmean(row["Full-mAP"] for row in cases.values())
    return {"num_queries": len(queries), "overall": overall, "cases": cases}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True)
    method = parser.parse_args().method
    if Path(method).name != method or method in (".", ".."):
        raise ValueError("Expected a method ID")

    tracker = PhaseTracker("evaluate test", total=3)
    tracker.advance("Verify test manifests and score artifacts")
    test = verify_manifest(ROOT, "test")
    gallery_path, query_path = split_paths(ROOT)
    gallery, queries = load_jsonl(gallery_path), load_jsonl(query_path)
    directory = RUNS_DIR / method
    run = json.loads((directory / "run.json").read_text(encoding="utf-8"))
    if run.get("method") != method or run.get("higher_is_better") is not True:
        raise ValueError("run.json does not identify the requested descending-score method")
    if run.get("split", "test") != "test" or run.get("data_fingerprint") != test["fingerprint"]:
        raise ValueError("Scores do not match the current test manifests. Rerun inference.")
    require_aligned_run(run, method)
    validate_run_order(run, gallery, queries)
    image_digest = run.get("shared_person_protocol", {}).get("detections_fingerprint", {}).get(
        "gallery_image_content_sha256", run.get("image_provenance", {}).get("gallery_image_content_sha256"))
    if image_digest != gallery_image_digest(ROOT, gallery):
        raise ValueError("Run image provenance is missing or differs from current image bytes. Rerun inference.")
    validation = run.get("validation_fingerprint")
    if validation is not None or run.get("cpr_supervision") == "Val only":
        if validation != verify_manifest(ROOT, "val")["fingerprint"]:
            raise ValueError("Selection does not match the current validation split. Rerun tuning.")
    scores_path = directory / "scores.npy"
    score_digest = sha256_file(scores_path)
    if run.get("scores_sha256", score_digest) != score_digest:
        raise ValueError("Score bytes differ from run.json. Rerun inference.")
    scores = np.load(scores_path, mmap_mode="r", allow_pickle=False)
    evaluated_run = {**run, "split": "test", "data_protocol": DATA_PROTOCOL,
                     "scores_sha256": score_digest}

    tracker.advance("Compute ID, Full and macro metrics")
    metrics = {
        "method": method,
        "split": "test",
        "data_protocol": DATA_PROTOCOL,
        "data_fingerprint": test["fingerprint"],
        "run_metadata_sha256": canonical_hash(evaluated_run),
        "scores_sha256": score_digest,
        **evaluate_scores(scores, gallery, queries),
    }
    if verify_manifest(ROOT, "test") != test:
        raise RuntimeError("Test manifests changed during evaluation")
    if validation is not None and verify_manifest(ROOT, "val")["fingerprint"] != validation:
        raise RuntimeError("Validation manifests changed during evaluation")
    if sha256_file(scores_path) != score_digest:
        raise RuntimeError("Scores changed during evaluation")
    if json.loads((directory / "run.json").read_text(encoding="utf-8")) != run:
        raise RuntimeError("Run metadata changed during evaluation")
    tracker.advance("Write test metrics and evaluated run metadata")
    output = OUTPUTS_DIR / method
    (output / "metrics.json").unlink(missing_ok=True)
    atomic_json(output / "run.json", evaluated_run)
    atomic_json(output / "metrics.json", metrics)
    print(json.dumps(metrics, indent=2, ensure_ascii=False), flush=True)
    tracker.finish()


if __name__ == "__main__":
    main()
