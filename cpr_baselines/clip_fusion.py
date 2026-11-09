"""Validation selection for the two global CLIP fusion baselines."""

from pathlib import Path

import numpy as np

from benchmark_data import gallery_image_digest, load_jsonl, split_paths, verify_manifest
from cpr_person.cache import atomic_json, read_json, sha256_file
from evaluate import full_map


def late_zscore(scores, self_indices, epsilon):
    import torch

    scores = torch.from_numpy(np.asarray(scores))
    indices = torch.as_tensor(self_indices, dtype=torch.long)
    if scores.shape[1] < 2 or not np.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("Late fusion needs a non-self gallery and a positive epsilon")
    if indices.shape != (len(scores),) or ((indices < 0) | (indices >= scores.shape[1])).any():
        raise ValueError("Late fusion requires one valid self index per query")
    valid = torch.ones_like(scores, dtype=torch.bool)
    valid.scatter_(1, indices[:, None], False)
    eligible = scores[valid].reshape(len(scores), scores.shape[1] - 1)
    if not torch.isfinite(eligible).all():
        raise ValueError("Non-self CLIP scores must be finite")
    mean = eligible.mean(-1, keepdim=True)
    std = eligible.std(-1, keepdim=True, unbiased=False).clamp_min(epsilon)
    return ((scores - mean) / std).masked_fill(~valid, 0.0).numpy()


def mix_early_scores(image_scores, text_scores, image_features, text_features, image_weight):
    """Mix branch dot products using the norm of the fused query, as in RCR."""
    import torch

    image = torch.from_numpy(np.asarray(image_scores))
    text = torch.from_numpy(np.asarray(text_scores))
    qi = torch.from_numpy(np.asarray(image_features))
    qt = torch.from_numpy(np.asarray(text_features))
    text_weight = 1.0 - image_weight
    norm = (image_weight * qi + text_weight * qt).norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return ((image_weight * image + text_weight * text) / norm).numpy()


def select_image_weight(image_scores, text_scores, gallery, queries, grid, *, early_features=None):
    trials = []
    for weight in grid:
        if early_features is None:
            mixed = weight * image_scores + (1.0 - weight) * text_scores
        else:
            mixed = mix_early_scores(image_scores, text_scores, *early_features, weight)
        trials.append({"image_weight": weight, "Full-mAP": full_map(mixed, gallery, queries)})
    best = max(trials, key=lambda row: (row["Full-mAP"], -abs(row["image_weight"] - 0.5), -row["image_weight"]))
    return best["image_weight"], best["Full-mAP"], trials


def tune_clip_fusion(root, cfg, output, device, encode_validation, *, epsilon=None):
    """Freeze alpha from complete val, then reuse the record on the same inputs."""
    gallery_path, query_path = split_paths(root, "val")
    validation = verify_manifest(root, "val")["fingerprint"]
    gallery = load_jsonl(gallery_path)
    grid = sorted({float(weight) for weight in cfg["tuning"]["image_weights"]})
    if not grid or any(not 0.0 <= weight <= 1.0 for weight in grid):
        raise ValueError("tuning.image_weights must be nonempty and in [0, 1]")
    # Same images and model must define both the val feature cache and lock.
    image_signature = gallery_image_digest(root, gallery)
    fingerprint = {
        "schema": "clip-fusion-rcr-val-v3",
        "method": cfg["method"],
        "checkpoint_sha256": sha256_file(root / cfg["model"]["checkpoint"]),
        "encoder_device": device.type,
        "runtime": cfg["runtime"],
        "validation_fingerprint": validation,
        "validation_image_signature": image_signature,
        "image_weights": grid,
        "epsilon": epsilon,
    }
    result_path = Path(output) / "tuning.json"
    record = read_json(result_path)
    if (
        isinstance(record, dict)
        and record.get("fingerprint") == fingerprint
        and record.get("image_weight") in grid
    ):
        return record

    import torch

    queries = load_jsonl(query_path)
    gallery_features, text_features = encode_validation(
        gallery_path, gallery, queries, image_signature
    )
    gallery_index = {row["image_id"]: index for index, row in enumerate(gallery)}
    self_indices = np.asarray([gallery_index[query["image_id"]] for query in queries])
    shape = (len(queries), len(gallery))
    image_scores, text_scores = np.empty(shape, np.float32), np.empty(shape, np.float32)
    gallery_tensor = torch.from_numpy(np.asarray(gallery_features)).to(device)
    batch_size = int(cfg["runtime"]["score_batch_size"])
    for start in range(0, len(queries), batch_size):
        stop = min(start + batch_size, len(queries))
        refs = gallery_tensor[self_indices[start:stop]]
        texts = torch.from_numpy(text_features[start:stop]).to(device)
        image_scores[start:stop] = (refs @ gallery_tensor.T).cpu().numpy()
        text_scores[start:stop] = (texts @ gallery_tensor.T).cpu().numpy()
    early_features = None
    if epsilon is not None:
        image_scores = late_zscore(image_scores, self_indices, epsilon)
        text_scores = late_zscore(text_scores, self_indices, epsilon)

    else:
        early_features = (gallery_features[self_indices], text_features)
    weight, score, trials = select_image_weight(
        image_scores, text_scores, gallery, queries, grid, early_features=early_features,
    )
    if verify_manifest(root, "val")["fingerprint"] != validation:
        raise RuntimeError("Validation manifests changed during CLIP selection")
    record = {
        "split": "val",
        "metric": "Full-mAP",
        "image_weight": weight,
        "text_weight": 1.0 - weight,
        "validation_full_map": score,
        "fingerprint": fingerprint,
        "trials": trials,
    }
    atomic_json(result_path, record)
    return record
