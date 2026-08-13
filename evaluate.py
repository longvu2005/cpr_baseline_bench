#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from benchmark_progress import PhaseTracker, progress_bar

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RUNS_DIR = ROOT / "runs"
OUTPUTS_DIR = ROOT / "outputs"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_no}: invalid JSON") from error
            if not isinstance(row, dict):
                raise TypeError(f"{path}:{line_no}: JSONL row must be an object")
            rows.append(row)
    return rows


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        value = json.load(f)
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return value


def validate_score_matrix(scores: np.ndarray, expected_shape: tuple[int, int]) -> None:
    if scores.shape != expected_shape:
        raise ValueError(f"Wrong score shape: got {scores.shape}, expected {expected_shape}")
    if not np.issubdtype(scores.dtype, np.number):
        raise TypeError(f"scores.npy must have a numeric dtype, got {scores.dtype}")

    for start in range(0, expected_shape[0], 256):
        block = np.asarray(scores[start : start + 256])
        if not np.isfinite(block).all():
            raise ValueError(f"scores.npy contains NaN/Inf in query rows {start}:{min(start + 256, expected_shape[0])}")


def measure(scores: np.ndarray, positives: set[int]) -> dict[str, float]:
    order = np.argsort(-scores)
    rel = np.isin(order, list(positives))
    ranks = np.flatnonzero(rel) + 1

    if len(ranks) == 0:
        raise ValueError("No positive found for a query.")

    ap = np.mean(np.arange(1, len(ranks) + 1) / ranks)

    return {
        "mAP": float(ap),
        "R@1": float(np.any(ranks <= 1)),
        "R@5": float(np.any(ranks <= 5)),
        "R@10": float(np.any(ranks <= 10)),
    }


def main() -> None:
    tracker = PhaseTracker("evaluate", total=4)
    tracker.advance("Load manifests and validate score artifacts")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method",
        required=True,
        help="Method name, e.g. clip_image",
    )
    args = parser.parse_args()

    method = args.method
    gallery = load_jsonl(DATA_DIR / "gallery.jsonl")
    queries = load_jsonl(DATA_DIR / "queries.jsonl")

    scores_path = RUNS_DIR / method / "scores.npy"
    run_path = RUNS_DIR / method / "run.json"

    if not scores_path.is_file():
        raise FileNotFoundError(f"Missing score file: {scores_path}")
    if not run_path.is_file():
        raise FileNotFoundError(f"Missing run file: {run_path}")

    run = load_json(run_path)
    run_method = run.get("method")
    if run_method != method:
        raise ValueError(
            f"run.json method mismatch: requested {method!r}, run.json contains {run_method!r}"
        )
    if run.get("higher_is_better") is not True:
        raise ValueError(
            "run.json must declare higher_is_better=true because the official evaluator ranks scores descending."
        )

    scores = np.load(scores_path, mmap_mode="r", allow_pickle=False)
    expected_shape = (len(queries), len(gallery))
    validate_score_matrix(scores, expected_shape)

    tracker.log(
        f"method={method} queries={len(queries):,} gallery={len(gallery):,} "
        f"score_shape={scores.shape}"
    )

    tracker.advance("Build gallery identity index")
    gidx = {row["image_id"]: i for i, row in enumerate(gallery)}
    if len(gidx) != len(gallery):
        raise ValueError("Duplicate image_id in gallery.jsonl")

    pid2idx: defaultdict[str, set[int]] = defaultdict(set)
    for i, row in enumerate(gallery):
        for pid in row["person_ids"]:
            pid2idx[str(pid)].add(i)

    all_m: defaultdict[str, list[float]] = defaultdict(list)
    case_m: defaultdict[str, defaultdict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )

    tracker.advance("Evaluate retrieval metrics for every query")
    for qi, q in progress_bar(
        enumerate(queries),
        desc="Evaluate queries",
        total=len(queries),
        unit="query",
    ):
        image_id = q["image_id"]
        if image_id not in gidx:
            raise ValueError(f'{q.get("query_id", qi)}: query image {image_id!r} missing from gallery')

        s = np.array(scores[qi], copy=True)

        # Exact query-image exclusion is an evaluator responsibility, not a
        # method responsibility. Raw scores.npy must still contain this entry.
        self_idx = gidx[image_id]
        s[self_idx] = -np.inf

        ids = [str(x) for x in q["target_ids"]]
        if not ids:
            raise ValueError(f'{q.get("query_id", qi)}: empty target_ids')

        # Strict ID positive: a gallery image must contain every target identity.
        id_pos = pid2idx[ids[0]].copy()
        for pid in ids[1:]:
            id_pos &= pid2idx[pid]
        id_pos.discard(self_idx)

        if not id_pos:
            raise ValueError(f'{q["query_id"]}: empty ID positives')

        full_pos = {gidx[x] for x in q["full_positive_ids"] if x in gidx}
        full_pos.discard(self_idx)
        if not full_pos:
            raise ValueError(f'{q["query_id"]}: empty Full positives')

        im = measure(s, id_pos)
        fm = measure(s, full_pos)

        for key, value in im.items():
            all_m[f"ID-{key}"].append(value)
        for key, value in fm.items():
            all_m[f"Full-{key}"].append(value)

        case = str(q["case"])
        case_m[case]["Full-mAP"].append(fm["mAP"])
        case_m[case]["Full-R@1"].append(fm["R@1"])

    metrics = {
        "method": method,
        "overall": {key: float(np.mean(values)) for key, values in all_m.items()},
        "cases": {
            case: {key: float(np.mean(values)) for key, values in case_values.items()}
            for case, case_values in case_m.items()
        },
    }

    tracker.advance("Write evaluated metrics and run metadata")
    output_dir = OUTPUTS_DIR / method
    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
        f.write("\n")

    with (output_dir / "run.json").open("w", encoding="utf-8") as f:
        json.dump(run, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(json.dumps(metrics, indent=2, ensure_ascii=False), flush=True)
    tracker.finish()


if __name__ == "__main__":
    main()
