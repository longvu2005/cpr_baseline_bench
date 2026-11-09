"""Receipt and cache wiring checks; substituted forwards are not native evidence."""

from __future__ import annotations

import ast
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import yaml

from cpr_baselines.provenance import execution_identity, require_execution_identity
from cpr_person.cache import sha256_file
from run_baseline import ROOT, discover_methods


def result_dictionaries(path):
    tree = ast.parse(path.read_text())
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Dict)
        and {"method", "display_name"}
        <= {key.value for key in node.keys if isinstance(key, ast.Constant)}
    ]


@pytest.mark.parametrize("method", discover_methods(), ids=lambda item: item.method_id)
def test_all_29_result_writers_emit_actual_execution_identity(method):
    dictionaries = result_dictionaries(method.run_path)
    if not dictionaries:
        tree = ast.parse(method.run_path.read_text())
        providers = [
            node.module
            for node in tree.body
            if isinstance(node, ast.ImportFrom)
            and node.module in {"cpr_baselines.runner", "cpr_baselines.global_clip"}
            and any(alias.name == "main" for alias in node.names)
        ]
        assert len(providers) == 1
        provider = ROOT / (providers[0].replace(".", "/") + ".py")
        dictionaries = result_dictionaries(provider)
    assert len(dictionaries) == 1
    node = dictionaries[0]
    values = {
        key.value: value
        for key, value in zip(node.keys, node.values)
        if isinstance(key, ast.Constant)
    }
    assert "execution" in values, method.method_id
    call = values["execution"]
    assert (
        isinstance(call, ast.Call)
        and isinstance(call.func, ast.Name)
        and call.func.id == "execution_identity"
    )
    assert (
        len(call.args) == 1
        and isinstance(call.args[0], ast.Name)
        and call.args[0].id == "__file__"
    )


def test_execution_receipt_rejects_missing_or_incomplete_runtime():
    actual = execution_identity(__file__)
    assert require_execution_identity(actual) is actual
    for field in ("python", "packages", "environment_sha256", "implementation"):
        broken = deepcopy(actual)
        broken.pop(field)
        with pytest.raises(RuntimeError, match="actual method runtime"):
            require_execution_identity(broken)
    for value in (None, {}, {"environment_sha256": "a" * 64}):
        with pytest.raises(RuntimeError):
            require_execution_identity(value)


def test_person_smoke_does_not_start_instruct_validation_selection():
    import runpy

    require = runpy.run_path(str(ROOT / "scripts/smoke_person_runtime.py"))[
        "require_untuned_smoke"
    ]
    cfg = {"method": "instruct_reid_setmatch", "setmatch": {"unmatched_score": None}}
    with pytest.raises(ValueError, match="external calibration"):
        require(cfg)
    cfg["setmatch"]["unmatched_calibration"] = "already_frozen_external.json"
    require(cfg)


def load_method(relative, name):
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_s9_val_proxy_cache_revalidates_generator_images_and_rcr_globals(
    tmp_path, monkeypatch
):
    pytest.importorskip("torch")
    import contextlib

    s9 = load_method("methods/simple/reid_qwen_proxy/run.py", "s9_receipt_contract")
    s8 = load_method("methods/simple/qwen_edit_clip/run.py", "s8_receipt_contract")
    cfg = yaml.safe_load(
        (ROOT / "methods/simple/reid_qwen_proxy/config.yaml").read_text()
    )
    proxy = yaml.safe_load(
        (ROOT / "methods/simple/qwen_edit_clip/config.yaml").read_text()
    )
    cfg["runtime"]["device"] = "cpu"
    folder = tmp_path / "generator"
    folder.mkdir()
    marker = folder / "prepared.json"
    marker.write_text('{"checkpoint":"first"}')
    scene_index = tmp_path / "scene.pt"
    scene_index.write_bytes(b"first RCR globals fixture")
    proxy_config = tmp_path / "proxy.yaml"
    proxy_config.write_text(yaml.safe_dump(proxy))
    cfg["proxy"]["config"] = str(proxy_config)
    cache = {
        "proxy_scores": str(tmp_path / "scores.npy"),
        "proxy_edited_queries_dir": str(tmp_path / "edited"),
        "proxy_edited_queries_manifest": str(tmp_path / "edited.jsonl"),
        "proxy_query_features": str(tmp_path / "query.npy"),
        "proxy_gallery_features": str(tmp_path / "gallery.npy"),
    }
    cfg["cache"]["validation"] = cache
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump(cfg))
    image = tmp_path / "image.png"
    Image.new("RGB", (21, 13), "red").save(image)
    gallery = [
        {"image_id": "a", "path": str(image)},
        {"image_id": "b", "path": str(image)},
    ]
    queries = [
        {
            "query_id": "q",
            "image_id": "a",
            "text": "public instruction",
            "full_positive_ids": ["b"],
        }
    ]
    gm, qm = tmp_path / "gallery.jsonl", tmp_path / "queries.jsonl"
    for path, rows in ((gm, gallery), (qm, queries)):
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    calls = {"generation": 0, "score": 0}

    def generate(**kwargs):
        calls["generation"] += 1
        directory = Path(kwargs["cfg"]["cache"]["edited_queries_dir"])
        directory.mkdir(exist_ok=True)
        path = directory / "0.png"
        Image.new("RGB", (21, 13), (40 * calls["generation"], 10, 20)).save(path)
        manifest = Path(kwargs["cfg"]["cache"]["edited_queries_manifest"])
        manifest.write_text(
            json.dumps(
                {
                    "query_id": "q",
                    "image_id": "a",
                    "edited_image": str(path),
                    "edited_image_sha256": sha256_file(path),
                }
            )
            + "\n"
        )
        fingerprint = s8.edit_fingerprint(
            cfg=kwargs["cfg"],
            config_path=kwargs["config_path"],
            gallery_manifest=kwargs["gallery_manifest"],
            query_manifest=kwargs["query_manifest"],
            generator_marker=marker,
        )
        s8.edit_meta_path(manifest).write_text(json.dumps(fingerprint))
        return [path], manifest, s8.edit_meta_path(manifest)

    def load_clip(*args):
        calls["score"] += 1
        return object(), None

    def query_features(model, preprocess, paths, cache, runtime, device, fingerprint):
        color = np.asarray(Image.open(paths[0])).mean(axis=(0, 1)).astype(np.float32)
        return (color / np.linalg.norm(color))[None, :], tmp_path / "query.npy"

    monkeypatch.setattr(s9, "check_config", lambda _: folder)
    monkeypatch.setattr(s9, "verify_scene_clip", lambda *_: "a" * 64)
    monkeypatch.setattr(s9, "scene_identity", lambda *_: {"assets": "a" * 64})
    monkeypatch.setattr(s8, "scene_identity", lambda *_: {"assets": "a" * 64})
    monkeypatch.setattr(s9, "load_scene_clip", load_clip)
    monkeypatch.setattr(
        s8, "validate_generator_artifacts", lambda _: (folder, marker, {})
    )
    monkeypatch.setattr(s8, "generate_or_load_edited_queries", generate)
    monkeypatch.setattr(
        s8,
        "clip_gallery_cache_fingerprint",
        lambda *_: {"rcr_index_sha256": sha256_file(scene_index)},
    )
    monkeypatch.setattr(
        s8,
        "clip_gallery_features",
        lambda *args: (
            np.array([[1, 0, 0], [0, 1, 0]], np.float32),
            tmp_path / "gallery.npy",
        ),
    )
    monkeypatch.setattr(s8, "clip_query_image_features", query_features)
    tracker = SimpleNamespace(phase=lambda *_: contextlib.nullcontext())

    def run():
        return s9.build_validation_proxy_scores(
            cfg=cfg,
            config_path=config,
            val_gallery_manifest=gm,
            val_query_manifest=qm,
            val_gallery=gallery,
            val_queries=queries,
            s8=s8,
            tracker=tracker,
        )

    first = run().copy()
    np.testing.assert_array_equal(first, run())
    assert calls == {"generation": 1, "score": 1}
    marker.write_text('{"checkpoint":"changed"}')
    run()
    assert calls == {"generation": 2, "score": 2}
    Image.new("RGB", (21, 13), "black").save(tmp_path / "edited/0.png")
    run()
    assert calls == {"generation": 3, "score": 3}
    scene_index.write_bytes(b"changed RCR globals fixture")
    run()
    run()
    assert calls == {"generation": 3, "score": 4}
