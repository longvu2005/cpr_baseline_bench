"""One actual HF pipeline for benchmark-owned scene CLIP branches."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory

import numpy as np
import yaml

from benchmark_data import manifest_image_digest
from cpr_person.cache import (
    ROOT,
    StageCache,
    atomic_file,
    cache_lock,
    resolve,
    sha256_file,
    canonical_hash,
    read_json,
)
from .scene_protocol import (
    check_config,
    checkpoint_path,
    assets,
    MODEL,
    require_scene_run,
)
from .scene_runtime import scene_python
from .provenance import execution_identity, probe_runtime
from .scene_cache import (
    read_scene_globals,
    scene_index_path,
    spot_check_globals,
    spot_check_rows,
)


def scene_identity(checkpoint=None):
    return {
        **assets(checkpoint_path(checkpoint)),
        "runtime": probe_runtime(scene_python()),
        "implementation_sha256": canonical_hash(
            execution_identity(__file__)["implementation"]
        ),
    }


def verify_scene_clip(checkpoint, name):
    if name != MODEL:
        raise ValueError("Expected the HF CLIP model from RCR Proposed")
    return scene_identity(checkpoint)["assets_sha256"]


def require_current_scene_run(run):
    if require_scene_run(run) != scene_identity():
        raise RuntimeError(
            "Source scores use a different scene encoder/runtime; rerun the source method"
        )


class SceneEncoder:
    def __init__(self, checkpoint, device):
        self.checkpoint = checkpoint_path(checkpoint)
        self.identity, self.device, self.sources = (
            scene_identity(self.checkpoint),
            str(device),
            {},
        )

    def encode(self, kind, values, batch):
        root = ROOT / "cache/scene_requests"
        root.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=root) as temporary:
            request, output = (
                Path(temporary) / "request.json",
                Path(temporary) / "features.npy",
            )
            request.write_text(
                json.dumps(
                    {
                        "kind": kind,
                        "inputs": list(values),
                        "batch_size": int(batch),
                        "checkpoint": str(self.checkpoint),
                        "assets": assets(self.checkpoint),
                        "runtime": self.identity["runtime"],
                        "device": self.device,
                        "output": str(output),
                    }
                )
            )
            env = os.environ.copy()
            env.pop("PYTHONPATH", None)
            env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
            subprocess.run(
                [
                    scene_python(),
                    "-u",
                    "-m",
                    "cpr_baselines.scene_worker",
                    "--request",
                    str(request),
                ],
                cwd=ROOT,
                env=env,
                check=True,
            )
            result = np.load(output, allow_pickle=False)
            if result.shape != (len(values), 512) or not np.isfinite(result).all():
                raise ValueError("Invalid scene worker features")
            return result

    def images(self, paths, batch):
        return self.encode("images", [str(resolve(path)) for path in paths], batch)

    def texts(self, texts, batch):
        return self.encode("texts", list(texts), batch)


def load_scene_clip(checkpoint, device):
    return SceneEncoder(checkpoint, device), None


def normalize(values):
    values = np.asarray(values, np.float32)
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    if values.ndim != 2 or not np.isfinite(values).all() or (norms < 1e-12).any():
        raise ValueError("Expected finite nonzero CLIP global rows")
    return values / norms


def stage_cache(stage, fingerprint):
    roots = [
        p
        for p in os.environ.get("CPR_SCENE_CACHE_READ_ROOTS", "").split(os.pathsep)
        if p
    ]
    return StageCache(
        stage,
        fingerprint,
        write_root=os.environ.get("CPR_SCENE_CACHE_ROOT", "cache/scene_clip"),
        read_roots=roots,
    )


def gallery_fingerprint(checkpoint, gallery_manifest, model_name):
    verify_scene_clip(checkpoint, model_name)
    source = scene_index_path()
    return {
        "encoder": scene_identity(checkpoint),
        "gallery_manifest_sha256": sha256_file(gallery_manifest),
        "gallery_image_content_sha256": manifest_image_digest(ROOT, gallery_manifest),
        "rcr_scene_index_sha256": None if source is None else sha256_file(source),
    }


def gallery_features(model, preprocess, rows, cache, runtime, device, fingerprint):
    batch = int(runtime.get("clip_image_batch_size", runtime.get("batch_size", 32)))
    stage = stage_cache(
        "gallery", {**fingerprint, "device": str(device), "batch_size": batch}
    )
    path = stage.find("gallery.npy")
    if path is not None:
        result = np.load(path, mmap_mode="c", allow_pickle=False)
        if result.shape == (len(rows), 512) and np.isfinite(result).all():
            model.sources[fingerprint["gallery_manifest_sha256"]] = {
                "feature_cache": str(path)
            }
            return result, path, True
    with cache_lock(stage.directory):
        source = scene_index_path()
        if source is None:
            values = model.images([row["path"] for row in rows], batch)
            provenance = {"native_hf_encoding": True}
        else:
            metadata = read_json(ROOT / "data/manifest.json") or {}
            values, provenance = read_scene_globals(
                source, rows, metadata.get("rcr_gallery_ids_sha256")
            )
            # Re-encode a few public images with the same worker; this binds an
            # index that has no build-time digests to the current pixels/pipeline.
            checked = spot_check_rows(rows)
            provenance["spot_check"] = spot_check_globals(
                values[checked],
                model.images([rows[i]["path"] for i in checked], batch),
            )
        path = stage.directory / "gallery.npy"
        with atomic_file(path) as handle:
            np.save(handle, normalize(values), allow_pickle=False)
        stage.seal(path)
        model.sources[fingerprint["gallery_manifest_sha256"]] = provenance
    return np.load(path, mmap_mode="c", allow_pickle=False), path, False


def encoded_features(model, kind, values, batch):
    identity = (
        values
        if kind == "texts"
        else [{"path": str(p), "sha256": sha256_file(resolve(p))} for p in values]
    )
    stage = stage_cache(
        kind,
        {
            "encoder": model.identity,
            "inputs": canonical_hash(identity),
            "device": model.device,
            "batch_size": int(batch),
        },
    )
    path = stage.find("features.npy")
    if path is not None:
        result = np.load(path, allow_pickle=False)
        if result.shape == (len(values), 512) and np.isfinite(result).all():
            return result, path
    with cache_lock(stage.directory):
        result = normalize(
            model.texts(values, batch)
            if kind == "texts"
            else model.images(values, batch)
        )
        path = stage.directory / "features.npy"
        with atomic_file(path) as handle:
            np.save(handle, result, allow_pickle=False)
        stage.seal(path)
    return result, path


def text_features(model, texts, batch):
    result, path = encoded_features(model, "texts", list(texts), batch)
    model.sources["texts"] = str(path)
    return result


def image_features(model, paths, batch):
    return encoded_features(model, "images", list(paths), batch)


def prepare_scene_clip(cfg, force=False):
    directory = check_config(cfg)
    command = [
        scene_python(),
        "-u",
        "-m",
        "cpr_baselines.scene_worker",
        "--prepare",
        str(directory),
    ]
    if force:
        command.append("--force")
    subprocess.run(command, cwd=ROOT, check=True)


def main(config_path):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(config_path))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    cfg = yaml.safe_load(resolve(args.config).read_text())
    prepare_scene_clip(cfg["model"], args.force)
