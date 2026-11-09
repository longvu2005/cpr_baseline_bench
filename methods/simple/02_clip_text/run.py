#!/usr/bin/env python3

import argparse
import json
import sys
from pathlib import Path

import clip
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cpr_baselines.scene_clip import gallery_features, gallery_fingerprint as build_gallery_cache_fingerprint
from cpr_person.clip_model import load_openai_clip, verify_rcr_scene_clip
from benchmark_data import split_paths, verify_manifest
from benchmark_progress import PhaseTracker, progress_bar  # noqa: E402
DEFAULT_CONFIG = Path(__file__).resolve().parent / "config.yaml"


def load_jsonl(path):
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def device_from(name):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@torch.no_grad()
def text_features(model, texts, batch_size, device):
    chunks = []
    for start in progress_bar(
        range(0, len(texts), batch_size),
        desc="Encode query text",
        total=(len(texts) + batch_size - 1) // batch_size,
        unit="batch",
    ):
        tokens = clip.tokenize(texts[start:start + batch_size], truncate=True).to(device)
        x = model.encode_text(tokens).float()
        x /= x.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        chunks.append(x.cpu().numpy())
    return np.concatenate(chunks).astype(np.float32)



def setup(tracker: PhaseTracker):
    with tracker.phase("Load config and manifests"):
        parser = argparse.ArgumentParser()
        parser.add_argument("--config", default=str(DEFAULT_CONFIG))
        args = parser.parse_args()

        config_path = Path(args.config)
        if not config_path.is_absolute():
            config_path = ROOT / config_path
        with config_path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        output = ROOT / cfg["output"]["dir"]
        output.mkdir(parents=True, exist_ok=True)
        checkpoint = ROOT / cfg["model"]["checkpoint"]
        if not checkpoint.is_file():
            raise FileNotFoundError(
                f"Missing CLIP checkpoint: {checkpoint}\n"
                "Run `python methods/simple/02_clip_text/download_checkpoint.py` from the repository root."
            )

        data_fingerprint = verify_manifest(ROOT, "test")["fingerprint"]
        gallery_manifest, query_manifest = split_paths(ROOT)
        gallery = load_jsonl(gallery_manifest)
        queries = load_jsonl(query_manifest)
        device = device_from(cfg["runtime"].get("device", "auto"))
        tracker.log(
            f"gallery={len(gallery):,} queries={len(queries):,} device={device} "
            f"image_batch={cfg['runtime']['batch_size']} text_batch={cfg['runtime']['text_batch_size']}"
        )

    with tracker.phase("Load CLIP model", f"{cfg['model']['name']} on {device}"):
        verify_rcr_scene_clip(checkpoint, cfg["model"]["name"])
        model, preprocess = load_openai_clip(checkpoint, device)

    with tracker.phase("Prepare gallery image features"):
        cache = ROOT / cfg["cache"]["gallery_features"]
        cache_fingerprint = build_gallery_cache_fingerprint(
            checkpoint, gallery_manifest, str(cfg["model"]["name"])
        )
        gfeat, cache_used, _ = gallery_features(
            model, preprocess, gallery, cache, cfg["runtime"], device, cache_fingerprint
        )
        tracker.log(f"gallery_features={gfeat.shape} cache={cache_used.relative_to(ROOT)}")

    with tracker.phase("Encode query text"):
        tfeat = text_features(
            model,
            [q["text"] for q in queries],
            cfg["runtime"]["text_batch_size"],
            device,
        )
        tracker.log(f"query_text_features={tfeat.shape}")

    return cfg, config_path, output, gallery, queries, device, gfeat, tfeat, cache_used, data_fingerprint


@torch.no_grad()
def main():
    tracker = PhaseTracker("clip_text", total=6)
    cfg, config_path, output, gallery, queries, device, gfeat, tfeat, cache_used, data_fingerprint = setup(tracker)

    with tracker.phase("Compute query-gallery score matrix"):
        scores_path = output / "scores.npy"
        scores = np.lib.format.open_memmap(
            scores_path, "w+", dtype=np.float32, shape=(len(queries), len(gallery))
        )
        gallery_tensor = torch.from_numpy(np.asarray(gfeat)).to(device)
        batch = cfg["runtime"]["score_batch_size"]
        score_steps = (len(queries) + batch - 1) // batch
        for start in progress_bar(
            range(0, len(queries), batch),
            desc="Score queries",
            total=score_steps,
            unit="batch",
        ):
            end = min(start + batch, len(queries))
            query = torch.from_numpy(tfeat[start:end]).to(device)
            scores[start:end] = (query @ gallery_tensor.T).cpu().numpy()
        scores.flush()

    with tracker.phase("Write run metadata and outputs"):
        run = {
            "method": cfg["method"],
            "data_fingerprint": data_fingerprint,
            "display_name": f"CLIP {cfg['model']['name']} - Text-only",
            "group": "Simple / Obvious Baselines",
            "cpr_supervision": "No",
            "model": cfg["model"],
            "scene_clip_sha256": verify_rcr_scene_clip(ROOT / cfg["model"]["checkpoint"], cfg["model"]["name"]),
            "runtime": cfg["runtime"],
            "gallery_features": str(cache_used.relative_to(ROOT)),
            "config": str(config_path.relative_to(ROOT)),
            "num_queries": len(queries),
            "num_gallery": len(gallery),
            "scores": str(scores_path.relative_to(ROOT)),
            "higher_is_better": True,
        }
        run_path = output / "run.json"
        with run_path.open("w", encoding="utf-8") as f:
            json.dump(run, f, indent=2, ensure_ascii=False)
            f.write("\n")
        tracker.log(f"scores={scores_path.relative_to(ROOT)} run={run_path.relative_to(ROOT)}")

    tracker.finish()


if __name__ == "__main__":
    main()
