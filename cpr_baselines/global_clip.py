"""Four global baselines using native frozen HF CLIP and complete RCR splits."""

import argparse
import numpy as np

from benchmark_data import (
    load_jsonl,
    split_paths,
    verify_manifest,
    manifest_fingerprint,
)
from cpr_person.cache import ROOT, resolve, atomic_file, atomic_json
from .common import load_config, public_inputs
from .clip_fusion import tune_clip_fusion, mix_early_scores, late_zscore
from .scene_clip import (
    load_scene_clip,
    gallery_fingerprint,
    gallery_features,
    text_features,
)
from .scene_protocol import check_config
from .provenance import execution_identity


def run(cfg):
    import torch

    method = cfg["method"]
    modes = {
        "clip_image": "Image-only",
        "clip_text": "Text-only",
        "clip_early_fusion": "Early Fusion",
        "clip_late_fusion": "Late Fusion",
    }
    if method not in modes:
        raise ValueError("Expected global CLIP baseline")
    name = cfg["runtime"]["device"]
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if name == "auto"
        else torch.device(name)
    )
    checkpoint = check_config(cfg["model"])
    model, preprocess = load_scene_clip(checkpoint, device)
    output = resolve(cfg["output"]["dir"])
    if output != ROOT / "runs" / method:
        raise ValueError("Expected canonical output directory")
    output.mkdir(parents=True, exist_ok=True)
    (output / "run.json").unlink(missing_ok=True)
    fingerprint = verify_manifest(ROOT, "test")["fingerprint"]
    gm, qm = split_paths(ROOT)
    gallery, queries, index = public_inputs(load_jsonl(gm), load_jsonl(qm))
    selection = None
    if method in {"clip_early_fusion", "clip_late_fusion"}:
        epsilon = (
            float(cfg["fusion"]["epsilon"]) if method == "clip_late_fusion" else None
        )

        def validation(manifest, rows, qs, image_signature):
            fp = gallery_fingerprint(checkpoint, manifest, cfg["model"]["name"])
            features, _, _ = gallery_features(
                model, preprocess, rows, None, cfg["runtime"], device, fp
            )
            return features, text_features(
                model, [q["text"] for q in qs], cfg["runtime"]["text_batch_size"]
            )

        selection = tune_clip_fusion(
            ROOT, cfg, output, device, validation, epsilon=epsilon
        )
    fp = gallery_fingerprint(checkpoint, gm, cfg["model"]["name"])
    features, _, _ = gallery_features(
        model, preprocess, gallery, None, cfg["runtime"], device, fp
    )
    texts = (
        None
        if method == "clip_image"
        else text_features(
            model, [q["text"] for q in queries], cfg["runtime"]["text_batch_size"]
        )
    )
    references = np.array([index[q["image_id"]] for q in queries])
    gtensor = torch.from_numpy(np.asarray(features)).to(device)
    scores = np.empty((len(queries), len(gallery)), np.float32)
    batch = int(cfg["runtime"]["score_batch_size"])
    for start in range(0, len(queries), batch):
        stop = min(start + batch, len(queries))
        image_scores = (gtensor[references[start:stop]] @ gtensor.T).cpu().numpy()
        if method == "clip_image":
            values = image_scores
        else:
            text_scores = (
                (torch.from_numpy(texts[start:stop]).to(device) @ gtensor.T)
                .cpu()
                .numpy()
            )
            if method == "clip_text":
                values = text_scores
            elif method == "clip_early_fusion":
                values = mix_early_scores(
                    image_scores,
                    text_scores,
                    features[references[start:stop]],
                    texts[start:stop],
                    selection["image_weight"],
                )
            else:
                w, eps = selection["image_weight"], float(cfg["fusion"]["epsilon"])
                values = w * late_zscore(image_scores, references[start:stop], eps) + (
                    1 - w
                ) * late_zscore(text_scores, references[start:stop], eps)
        scores[start:stop] = values
    if not np.isfinite(scores).all() or manifest_fingerprint(ROOT) != fingerprint:
        raise RuntimeError("Invalid scores or test inputs changed during inference")
    with atomic_file(output / "scores.npy") as handle:
        np.save(handle, scores, allow_pickle=False)
    run = {
        "execution": execution_identity(__file__),
        "method": method,
        "display_name": "CLIP ViT-B/32 - " + modes[method],
        "group": "Simple / Obvious Baselines",
        "cpr_supervision": "Val only" if selection else "No",
        "higher_is_better": True,
        "scene_clip": model.identity,
        "scene_clip_sha256": model.identity["assets_sha256"],
        "scene_sources": model.sources,
        "num_queries": len(queries),
        "num_gallery": len(gallery),
        "query_ids": [q["query_id"] for q in queries],
        "gallery_image_ids": [g["image_id"] for g in gallery],
        "data_fingerprint": fingerprint,
        "image_provenance": {
            "gallery_image_content_sha256": fp["gallery_image_content_sha256"]
        },
        "selection": selection,
    }
    if selection:
        run["validation_fingerprint"] = selection["fingerprint"][
            "validation_fingerprint"
        ]
    atomic_json(output / "run.json", run)


def main(config_path):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(config_path))
    run(load_config(resolve(parser.parse_args().config)))
