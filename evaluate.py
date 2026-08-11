#!/usr/bin/env python3

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RUNS_DIR = ROOT / "runs"
OUTPUTS_DIR = ROOT / "outputs"


def load_jsonl(path):
    rows = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    return rows


def measure(scores, positives):
    order = np.argsort(-scores)
    rel = np.isin(order, list(positives))
    ranks = np.flatnonzero(rel) + 1

    if len(ranks) == 0:
        raise ValueError("No positive found for a query.")

    ap = np.mean(
        np.arange(1, len(ranks) + 1) / ranks
    )

    return {
        "mAP": float(ap),
        "R@1": float(np.any(ranks <= 1)),
        "R@5": float(np.any(ranks <= 5)),
        "R@10": float(np.any(ranks <= 10)),
    }


def main():
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

    scores = np.load(scores_path, mmap_mode="r")

    expected_shape = (len(queries), len(gallery))
    if scores.shape != expected_shape:
        raise ValueError(
            f"Wrong score shape: got {scores.shape}, expected {expected_shape}"
        )

    gidx = {
        row["image_id"]: i
        for i, row in enumerate(gallery)
    }

    if len(gidx) != len(gallery):
        raise ValueError("Duplicate image_id in gallery.jsonl")

    pid2idx = defaultdict(set)
    for i, row in enumerate(gallery):
        for pid in row["person_ids"]:
            pid2idx[str(pid)].add(i)

    all_m = defaultdict(list)
    case_m = defaultdict(lambda: defaultdict(list))

    for qi, q in enumerate(queries):
        s = np.array(scores[qi], copy=True)

        # remove query image at evaluation time
        self_idx = gidx[q["image_id"]]
        s[self_idx] = -np.inf

        ids = [str(x) for x in q["target_ids"]]

        # strict ID positive: image must contain all target identities
        id_pos = pid2idx[ids[0]].copy()
        for pid in ids[1:]:
            id_pos &= pid2idx[pid]

        id_pos.discard(self_idx)

        if not id_pos:
            raise ValueError(f'{q["query_id"]}: empty ID positives')

        full_pos = {
            gidx[x]
            for x in q["full_positive_ids"]
            if x in gidx
        }
        full_pos.discard(self_idx)

        if not full_pos:
            raise ValueError(f'{q["query_id"]}: empty Full positives')

        im = measure(s, id_pos)
        fm = measure(s, full_pos)

        for k, v in im.items():
            all_m[f"ID-{k}"].append(v)

        for k, v in fm.items():
            all_m[f"Full-{k}"].append(v)

        case_m[q["case"]]["Full-mAP"].append(fm["mAP"])
        case_m[q["case"]]["Full-R@1"].append(fm["R@1"])

    metrics = {
        "method": method,
        "overall": {
            k: float(np.mean(v))
            for k, v in all_m.items()
        },
        "cases": {
            case: {
                k: float(np.mean(v))
                for k, v in values.items()
            }
            for case, values in case_m.items()
        },
    }

    output_dir = OUTPUTS_DIR / method
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    with open(run_path, "r", encoding="utf-8") as f:
        run = json.load(f)

    with open(output_dir / "run.json", "w", encoding="utf-8") as f:
        json.dump(run, f, indent=2, ensure_ascii=False)

    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
