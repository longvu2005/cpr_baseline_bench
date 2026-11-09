"""Validation-only selection in native unnormalized squared-Euclidean space."""

from __future__ import annotations

import numpy as np
from statistics import fmean

from evaluate import full_relevance, rank_scores, average_precision
from .matching import partial_setmatch_sum
from .cache import atomic_json

QUANTILES = np.linspace(0.0, 1.0, 21)
MAX_GRID_PAIRS = 200_000


def native_distance_grid(query_features, gallery_features):
    q, g = np.asarray(query_features), np.asarray(gallery_features)
    if (
        q.ndim != 2
        or g.ndim != 2
        or q.shape[1] != g.shape[1]
        or not len(q)
        or not len(g)
    ):
        raise ValueError("Expected nonempty native feature matrices of equal width")
    total = len(q) * len(g)
    indices = np.linspace(0, total - 1, min(total, MAX_GRID_PAIRS), dtype=np.int64)
    values = np.empty(len(indices), np.float64)
    for start in range(0, len(indices), 2048):
        ids = indices[start : start + 2048]
        delta = np.asarray(q[ids // len(g)], np.float64) - np.asarray(
            g[ids % len(g)], np.float64
        )
        values[start : start + len(ids)] = -(delta * delta).sum(axis=-1)
    if not np.isfinite(values).all():
        raise ValueError("Non-finite native distances")
    grid = np.unique(np.quantile(values, QUANTILES))
    return grid, {
        "quantiles": QUANTILES.tolist(),
        "sample_size": len(indices),
        "population_size": total,
        "sampling": "deterministic_evenly_spaced_cartesian_pairs",
        "feature_normalization": False,
    }


def select_unmatched(
    grid, gallery, queries, gallery_offsets, pair_scores, output, fingerprint
):
    grid = np.asarray(grid, np.float64)
    if (
        grid.ndim != 1
        or not len(grid)
        or not np.isfinite(grid).all()
        or (grid > 0).any()
    ):
        raise ValueError("Unmatched grid must contain finite nonpositive native scores")
    index = {row["image_id"]: i for i, row in enumerate(gallery)}
    counts = np.diff(gallery_offsets)
    aps = np.empty((len(grid), len(queries)), np.float64)
    affected = 0
    for qi, query in enumerate(queries):
        raw = np.asarray(pair_scores(qi))
        if (
            raw.ndim != 2
            or raw.shape[0] < 1
            or raw.shape[1] != int(gallery_offsets[-1])
            or not np.isfinite(raw).all()
        ):
            raise ValueError("Invalid native validation component scores")
        k = len(raw)
        sums = np.array(
            [
                partial_setmatch_sum(raw[:, start:stop])[0]
                for start, stop in zip(
                    gallery_offsets[:-1], gallery_offsets[1:], strict=True
                )
            ]
        )
        missing = np.maximum(k - counts, 0)
        affected += bool(missing.any())
        self_index, positives = full_relevance(query, index)
        for j, b in enumerate(grid):
            scores = ((sums + missing * b) / k).astype(np.float32)
            aps[j, qi] = average_precision(rank_scores(scores, self_index), positives)
    if not queries:
        raise ValueError("Validation queries must be nonempty")
    means = [fmean(row) for row in aps]
    best = max(range(len(grid)), key=lambda i: (means[i], -grid[i]))
    result = {
        "schema": "instruct-native-unmatched-val-v1",
        "selection_split": "val",
        "metric": "Full-mAP",
        "pairwise_metric": "negative_squared_euclidean",
        "selected_unmatched_score": float(grid[best]),
        "tie_break": "smallest_native_score",
        "validation_fingerprint": fingerprint,
        "num_queries_with_any_shortage": affected,
        "trials": [
            {"unmatched_score": float(b), "Full-mAP": float(m)}
            for b, m in zip(grid, means, strict=True)
        ],
    }
    atomic_json(output, result)
    return result
