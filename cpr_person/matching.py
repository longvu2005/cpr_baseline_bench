"""Exact maximum-sum rectangular matching, averaged over query components."""
from __future__ import annotations

import os
import numpy as np
from scipy.optimize import linear_sum_assignment


def partial_setmatch(matrix, unmatched_score):
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] < 1 or not np.isfinite(matrix).all():
        raise ValueError("SetMatch requires a finite [K,N] matrix, K >= 1")
    if not np.isfinite(unmatched_score):
        raise ValueError("Unmatched score must be finite")
    k, n = matrix.shape
    if n == 0:
        return float(unmatched_score)
    rows, cols = linear_sum_assignment(-matrix)
    # Rectangular Hungarian must take all min(K,N) real pairs, even below b.
    return float((matrix[rows, cols].sum() + (k - len(rows)) * unmatched_score) / k)


def build_person_index(offsets):
    offsets = np.asarray(offsets)
    if (offsets.ndim != 1 or len(offsets) < 1 or offsets[0] != 0
            or not np.issubdtype(offsets.dtype, np.integer) or np.any(np.diff(offsets) < 0)):
        raise ValueError("Offsets must be monotone integer boundaries starting at zero")
    counts = np.diff(offsets)
    width = max(2, int(counts.max(initial=0)))
    indices = offsets[:-1, None] + np.arange(width)[None, :]
    indices[np.arange(width)[None, :] >= counts[:, None]] = -1
    return indices, counts


def maximum_two_match_sum(a, b):
    """For every row, max_{i != j} a[i]+b[j], with at least two candidates."""
    ia, ib = np.argmax(a, axis=1), np.argmax(b, axis=1)
    r = np.arange(len(a))
    best = a[r, ia] + b[r, ib]
    same = ia == ib
    if np.any(same):
        aa, bb = a[same].copy(), b[same].copy()
        rr = np.arange(len(aa))
        aa[rr, ia[same]] = -np.inf
        bb[rr, ib[same]] = -np.inf
        best[same] = np.maximum(a[same, ia[same]] + bb.max(axis=1),
                                aa.max(axis=1) + b[same, ib[same]])
    return best


def aggregate_people(pair_scores, offsets, unmatched_score):
    """Vectorized K=1/2 and N=0/1/2 paths; Hungarian only for larger sets."""
    s = np.asarray(pair_scores, dtype=np.float64)
    indices, counts = build_person_index(offsets)
    if (s.ndim != 2 or s.shape[0] < 1 or s.shape[1] != int(offsets[-1])
            or not np.isfinite(s).all() or not np.isfinite(unmatched_score)):
        raise ValueError("Invalid pair scores, offsets, or unmatched value")
    k = s.shape[0]
    result = np.full(len(counts), unmatched_score, dtype=np.float64)
    one, two, many = counts == 1, counts == 2, counts >= 3
    if np.any(one):
        result[one] = (s[:, indices[one, 0]].max(axis=0) + (k - 1) * unmatched_score) / k
    if k == 1:
        active = counts >= 2
        if np.any(active):
            ids = indices[active]
            values = s[0, np.maximum(ids, 0)]
            values[ids < 0] = -np.inf
            result[active] = values.max(axis=1)
    elif k == 2:
        active = counts >= 2
        if np.any(active):
            ids = indices[active]
            values = s[:, np.maximum(ids, 0)].transpose(1, 0, 2)
            values[np.broadcast_to((ids < 0)[:, None, :], values.shape)] = -np.inf
            result[active] = maximum_two_match_sum(values[:, 0], values[:, 1]) / 2
    else:
        if np.any(two):
            values = s[:, indices[two, :2]].transpose(1, 2, 0)
            result[two] = (maximum_two_match_sum(values[:, 0], values[:, 1])
                           + (k - 2) * unmatched_score) / k
        for gi in np.flatnonzero(many):
            result[gi] = partial_setmatch(s[:, offsets[gi]:offsets[gi + 1]], unmatched_score)
    if not np.isfinite(result).all():
        raise FloatingPointError("Non-finite Partial SetMatch result")
    return result.astype(np.float32)


def aggregate_component_scores(component_score_path, query_offsets, gallery_offsets,
                               scores_path, unmatched_score):
    from benchmark_progress import progress_bar
    components = np.load(component_score_path, mmap_mode="r", allow_pickle=False)
    if (components.shape != (int(query_offsets[-1]), int(gallery_offsets[-1]))
            or np.any(np.diff(query_offsets) < 1)):
        raise ValueError("Component matrix/offset mismatch")
    scores_path.parent.mkdir(parents=True, exist_ok=True)
    temp = scores_path.with_name(scores_path.name + ".partial")
    scores = np.lib.format.open_memmap(temp, mode="w+", dtype=np.float32,
                                     shape=(len(query_offsets) - 1, len(gallery_offsets) - 1))
    for qi in progress_bar(range(len(scores)), desc="Partial SetMatch", total=len(scores), unit="query"):
        scores[qi] = aggregate_people(components[query_offsets[qi]:query_offsets[qi + 1]],
                                     gallery_offsets, unmatched_score)
    scores.flush()
    del scores
    os.replace(temp, scores_path)


def score_feature_sets(query_features, gallery_features, offsets, unmatched_score,
                       chunk_size, *, metric="cosine"):
    q = np.asarray(query_features, dtype=np.float32)
    if q.ndim != 2 or not len(q) or chunk_size < 1:
        raise ValueError("Need query components and a positive chunk size")
    scores = np.empty((len(q), int(offsets[-1])), np.float32)
    for start in range(0, scores.shape[1], chunk_size):
        g = np.asarray(gallery_features[start:start + chunk_size], dtype=np.float32)
        dot = q @ g.T
        if metric == "negative_squared_euclidean":
            dot = -(np.sum(q * q, axis=1, keepdims=True)
                    + np.sum(g * g, axis=1)[None, :] - 2 * dot)
        elif metric != "cosine":
            raise ValueError(f"Unsupported native metric: {metric}")
        scores[:, start:start + len(g)] = dot
    return aggregate_people(scores, offsets, unmatched_score)
