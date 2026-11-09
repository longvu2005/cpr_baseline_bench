"""Actual file-content checks for inherited scores and checkpoint mounts."""

import json
import os
from pathlib import Path
import runpy

import pytest
import yaml

from cpr_baselines.native_assets import require_current_assets, snapshot
from cpr_person.cache import sha256_file
from cpr_person.protocol import RESULT_PROTOCOL


def test_source_receipt_rejects_weight_and_dependent_config_drift(tmp_path):
    weight = tmp_path / "weight.pt"
    weight.write_bytes(b"first weights")
    dependency = tmp_path / "source.yaml"
    dependency.write_text(
        yaml.safe_dump({"method": "native", "model": {"checkpoint": str(weight)}})
    )
    config = tmp_path / "config.yaml"
    cfg = {"method": "native_proxy", "base_set": {"config": str(dependency)}}
    config.write_text(yaml.safe_dump(cfg))
    receipt = {
        "config_path": config.name,
        "config_sha256": sha256_file(config),
        "assets_snapshot": snapshot(cfg, tmp_path),
        "result_protocol": RESULT_PROTOCOL,
        "implementation": {config.name: sha256_file(config)},
    }
    require_current_assets(receipt, tmp_path)
    stale = {**receipt, "result_protocol": "old"}
    with pytest.raises(RuntimeError, match="implementation"):
        require_current_assets(stale, tmp_path)
    original = weight.stat()
    weight.write_bytes(b"other weights")
    os.utime(weight, ns=(original.st_atime_ns, original.st_mtime_ns))
    with pytest.raises(RuntimeError, match="assets changed"):
        require_current_assets(receipt, tmp_path)
    weight.write_bytes(b"first weights")
    dependency.write_text(dependency.read_text() + "native_setting: changed\n")
    with pytest.raises(RuntimeError, match="assets changed"):
        require_current_assets(receipt, tmp_path)
    with pytest.raises(RuntimeError, match="receipt"):
        require_current_assets({}, tmp_path)


def test_shared_protocol_and_source_bytes_are_in_asset_snapshot(tmp_path):
    source = tmp_path / "official"
    source.mkdir()
    (source / "model.py").write_text("class Native: pass\n")
    protocol = tmp_path / "protocol.yaml"
    protocol.write_text("selector:\n  threshold: 0.2\n")
    cfg = {
        "method": "native",
        "source": {"local_checkout": str(source)},
        "person_protocol": str(protocol),
    }
    original = snapshot(cfg, tmp_path)
    (source / "model.py").write_text("class Modified: pass\n")
    assert original != snapshot(cfg, tmp_path)
    assert "method.person_protocol" in original


def test_native_auxiliary_weights_are_bound_without_scanning_virtualenvs(tmp_path):
    model = tmp_path / "mapper.pt"
    model.write_bytes(b"native mapper")
    cfg = {
        "method": "native",
        "elite": {"global_mapper": str(model)},
        "runtime_assets": {"basic_model_cache_root": str(model)},
        "isolated_env": {"path": str(tmp_path)},
    }
    result = snapshot(cfg, tmp_path)
    assert set(result) == {
        "method.elite.global_mapper",
        "method.runtime_assets.basic_model_cache_root",
    }


def test_readonly_checkpoint_mount_uses_writable_metadata_directories(tmp_path):
    root = Path(__file__).resolve().parents[1]
    module = runpy.run_path(str(root / "scripts/stage_checkpoint_mount.py"))
    source, target = tmp_path / "input", tmp_path / "working"
    (source / "native").mkdir(parents=True)
    weights = source / "native/model.pt"
    weights.write_bytes(b"weights")
    marker = source / "native/prepared.json"
    marker.write_text(json.dumps({"ready": True}))
    module["stage"](source, target)
    assert (target / "native").is_dir() and not (target / "native").is_symlink()
    assert (target / "native/model.pt").is_symlink()
    assert not (target / "native/prepared.json").is_symlink()
    (target / "native/prepared.json").write_text("{}")
    assert json.loads(marker.read_text()) == {"ready": True}


def test_all_dependency_sources_are_declared_in_their_frozen_configs():
    import run_baseline

    methods = {method.method_id: method for method in run_baseline.discover_methods()}
    assert len(methods) == 29
    nine = yaml.safe_load(methods["reid_set_qwen_edit_proxy"].config_path.read_text())
    eleven = yaml.safe_load(methods["retrieve_qwen25vl_rerank"].config_path.read_text())
    assert nine["base_set"]["config"] == str(
        methods["groundingdino_clipreid_set"].config_path.relative_to(run_baseline.ROOT)
    )
    assert nine["proxy"]["config"] == str(
        methods["qwen_image_edit_clip"].config_path.relative_to(run_baseline.ROOT)
    )
    assert "groundingdino_clipreid_set_text" in str(eleven["first_stage"])
