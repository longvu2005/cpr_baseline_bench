#!/usr/bin/env python3
"""Prepare a strict, isolated IP-CIR environment and public checkpoints for P10.

Design goals:
- Never downgrade/replace Kaggle's system torch/transformers stack.
- Use the released Qwen-32B GPTQ + MIGC + ELITE path; no hidden small-model fallback.
- Fail before large downloads when hardware/disk is insufficient.
- Pin the public Imagine-and-Seek source commit and record resolved HF revisions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[3]
METHOD_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = METHOD_DIR / "config.yaml"
P10_HF_HOME = ROOT / "runs" / "imagine_seek" / "hf_cache"
os.environ.setdefault("HF_HOME", str(P10_HF_HOME))
os.environ.setdefault("HF_HUB_CACHE", str(P10_HF_HOME / "hub"))

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from benchmark_progress import PhaseTracker  # noqa: E402


def load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError(f"Expected YAML mapping: {path}")
    return data


def resolve_path(value: str) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else ROOT / path).resolve()


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except Exception:
        return str(path.resolve())


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def run(command: list[str], *, cwd: Path = ROOT, env: dict[str, str] | None = None) -> None:
    print("$ " + " ".join(command), flush=True)
    merged = os.environ.copy()
    if env:
        merged.update(env)
    merged["PYTHONUNBUFFERED"] = "1"
    subprocess.run(command, cwd=str(cwd), env=merged, check=True)


def output(command: list[str], *, cwd: Path = ROOT) -> str:
    return subprocess.check_output(command, cwd=str(cwd), text=True).strip()


def _json_file(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return value if isinstance(value, dict) else None


def _model_weight_bytes(path: Path) -> int:
    return sum(
        item.stat().st_size
        for item in path.glob("model-*.safetensors")
        if item.is_file()
    )


def _validate_large_model(kind: str, path: Path) -> tuple[bool, str]:
    """Validate model identity without reading multi-GB tensors into memory."""
    config_path = path / "config.json"
    index_path = path / "model.safetensors.index.json"
    config = _json_file(config_path)
    if config is None or not index_path.is_file():
        return False, "missing config.json or model.safetensors.index.json"
    weight_bytes = _model_weight_bytes(path)
    if kind == "captioner":
        text = config.get("text_config") if isinstance(config.get("text_config"), dict) else {}
        vision = config.get("vision_config") if isinstance(config.get("vision_config"), dict) else {}
        ok = (
            config.get("model_type") == "blip-2"
            and int(text.get("hidden_size", -1)) == 4096
            and int(vision.get("image_size", -1)) == 364
            and weight_bytes >= 28_000_000_000
        )
        return ok, f"BLIP2 signature weight_bytes={weight_bytes}"
    if kind == "layout_llm":
        quant = config.get("quantization_config") if isinstance(config.get("quantization_config"), dict) else {}
        ok = (
            config.get("model_type") == "qwen2"
            and int(config.get("hidden_size", -1)) == 5120
            and int(config.get("num_hidden_layers", -1)) == 64
            and int(quant.get("bits", -1)) == 4
            and str(quant.get("quant_method", "")).lower() == "gptq"
            and weight_bytes >= 17_000_000_000
        )
        return ok, f"Qwen32 GPTQ signature weight_bytes={weight_bytes}"
    raise ValueError(kind)


def _candidate_model_dirs(roots: list[Path]) -> list[Path]:
    result: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        if not root.is_dir():
            continue
        for cfg_path in root.rglob("config.json"):
            parent = cfg_path.parent.resolve()
            if parent not in seen:
                seen.add(parent)
                result.append(parent)
    return result


def discover_external_large_assets(cfg: dict[str, Any], *, fail_if_missing: bool) -> dict[str, Any]:
    storage = cfg["asset_storage"]
    roots = [Path(str(x)).resolve() for x in storage.get("external_search_roots", [])]
    candidates = _candidate_model_dirs(roots)
    found: dict[str, Any] = {}
    missing: list[str] = []
    specs = (
        ("captioner", str(storage["captioner_env"]), cfg["captioner"]),
        ("layout_llm", str(storage["layout_llm_env"]), cfg["layout_llm"]),
    )
    for kind, env_name, model_cfg in specs:
        selected: Path | None = None
        source = None
        explicit = os.environ.get(env_name, "").strip()
        search = [Path(explicit).resolve()] if explicit else candidates
        diagnostics: list[str] = []
        for candidate in search:
            ok, detail = _validate_large_model(kind, candidate)
            if ok:
                selected = candidate
                source = f"env:{env_name}" if explicit else "auto:/kaggle/input"
                break
            if explicit:
                diagnostics.append(f"{candidate}: {detail}")
        if selected is None:
            missing.append(kind)
            found[kind] = {
                "found": False,
                "expected_repo_id": str(model_cfg["repo_id"]),
                "env_override": env_name,
                "diagnostics": diagnostics,
            }
        else:
            found[kind] = {
                "found": True,
                "path": str(selected),
                "source": source,
                "expected_repo_id": str(model_cfg["repo_id"]),
                "config_sha256": sha256_file(selected / "config.json"),
                "index_sha256": sha256_file(selected / "model.safetensors.index.json"),
                "weight_bytes": _model_weight_bytes(selected),
            }
            print(f"[mount] {kind}: {selected}", flush=True)

    if missing and fail_if_missing:
        names = ", ".join(missing)
        raise RuntimeError(
            "Missing paper-faithful large model input(s): " + names + ".\n"
            "On Kaggle, do NOT download these into /kaggle/working. Add the exact "
            "Hugging Face models as Notebook Inputs / Kaggle Models, then rerun.\n"
            "Required: Salesforce/blip2-opt-6.7b-coco and "
            "Qwen/Qwen1.5-32B-Chat-GPTQ-Int4.\n"
            "If Kaggle mounts them under unusual paths, set IPCIR_BLIP2_DIR and/or "
            "IPCIR_QWEN32_DIR to the model directory containing config.json and model shards."
        )
    return found


def ensure_local_free_disk(cfg: dict[str, Any], external: dict[str, Any]) -> dict[str, float | bool]:
    usage = shutil.disk_usage(ROOT)
    free_gib = usage.free / 1024**3
    local_required = float(cfg["isolated_env"]["minimum_local_free_disk_gib"])
    both_mounted = all(external.get(k, {}).get("found") for k in ("captioner", "layout_llm"))
    required = local_required if both_mounted else float(cfg["isolated_env"]["full_download_minimum_free_disk_gib"])
    print(
        f"[preflight] writable free={free_gib:.1f} GiB; required={required:.1f} GiB; "
        f"large_models_mounted={both_mounted}",
        flush=True,
    )
    if free_gib < required:
        raise RuntimeError(
            f"Insufficient writable disk for the selected P10 storage mode: need {required:.1f} GiB, "
            f"found {free_gib:.1f} GiB. Large model mounts are required on Kaggle; "
            "do not lower the threshold or switch to smaller models for final reporting."
        )
    return {
        "free_gib_at_preflight": round(free_gib, 3),
        "required_gib": required,
        "large_models_mounted": both_mounted,
    }

def env_python(cfg: dict[str, Any]) -> Path:
    return resolve_path(str(cfg["isolated_env"]["python"]))


def prepare_env(cfg: dict[str, Any], force: bool) -> dict[str, Any]:
    env_cfg = cfg["isolated_env"]
    env_dir = resolve_path(str(env_cfg["path"]))
    py = env_python(cfg)
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError("uv was not installed by P10 requirements.txt")

    # `--force` intentionally does not destroy a valid multi-GB environment. It only
    # causes the exact pins to be re-asserted and the probe to be rerun.
    if not py.is_file():
        env_dir.parent.mkdir(parents=True, exist_ok=True)
        run([uv, "python", "install", str(env_cfg["python_version"])], env={"UV_NO_CACHE": "1"})
        run([uv, "venv", "--python", str(env_cfg["python_version"]), "--seed", str(env_dir)], env={"UV_NO_CACHE": "1"})

    torch_spec = f"torch=={env_cfg['torch_version']}"
    tv_spec = f"torchvision=={env_cfg['torchvision_version']}"
    run([
        uv, "pip", "install", "--python", str(py), "--no-cache",
        "--index-url", str(env_cfg["torch_index_url"]), torch_spec, tv_spec,
    ], env={"UV_NO_CACHE": "1"})
    run([
        uv, "pip", "install", "--python", str(py), "--no-cache",
        "-r", str(METHOD_DIR / "generator_requirements.txt")
    ], env={"UV_NO_CACHE": "1"})
    # AutoGPTQ's CUDA-12.1 0.7.1 wheel is built against torch 2.2.1+cu121.
    run([
        str(py), "-m", "pip", "install", "--no-cache-dir",
        f"auto-gptq=={env_cfg['auto_gptq_version']}", "--no-build-isolation",
    ])

    probe = r'''
import json, sys
import torch, torchvision, transformers, diffusers, huggingface_hub
import auto_gptq
info = {
    "python": sys.version.split()[0],
    "torch": torch.__version__,
    "torchvision": torchvision.__version__,
    "transformers": transformers.__version__,
    "diffusers": diffusers.__version__,
    "huggingface_hub": huggingface_hub.__version__,
    "cuda_available": torch.cuda.is_available(),
    "cuda_device_count": torch.cuda.device_count(),
    "gpus": [],
}
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        info["gpus"].append({"index": i, "name": p.name, "total_gib": p.total_memory / 1024**3})
print(json.dumps(info))
'''
    raw = output([str(py), "-c", probe])
    info = json.loads(raw.splitlines()[-1])
    if not info.get("cuda_available"):
        raise RuntimeError("Isolated IP-CIR environment cannot see CUDA")
    total_gib = sum(float(g["total_gib"]) for g in info.get("gpus", []))
    minimum = float(env_cfg["minimum_total_gpu_gib"])
    if total_gib < minimum:
        raise RuntimeError(
            f"Visible GPU memory totals {total_gib:.1f} GiB, below the strict {minimum:.1f} GiB "
            "required for Qwen1.5-32B-GPTQ. Select a larger/multi-GPU Kaggle accelerator or run on the A6000-class server."
        )
    info["total_gpu_gib"] = total_gib
    print("[ok] isolated env: " + json.dumps(info, indent=2), flush=True)
    return info


def tracked_dirty(checkout: Path) -> str:
    if not (checkout / ".git").is_dir():
        return "not a git checkout"
    return output(["git", "-C", str(checkout), "status", "--porcelain", "--untracked-files=no"])


def prepare_source(cfg: dict[str, Any]) -> Path:
    c = cfg["author_source"]
    checkout = resolve_path(str(c["local_checkout"]))
    expected = str(c["commit"])
    if not checkout.exists():
        if shutil.which("git") is None:
            raise RuntimeError("git is required to clone the released Imagine-and-Seek source")
        checkout.parent.mkdir(parents=True, exist_ok=True)
        run(["git", "clone", str(c["repository"]), str(checkout)])
    dirty = tracked_dirty(checkout)
    if dirty:
        raise RuntimeError(f"Refusing to use modified released source at {rel(checkout)}:\n{dirty}")
    actual = output(["git", "-C", str(checkout), "rev-parse", "HEAD"])
    if actual != expected:
        run(["git", "-C", str(checkout), "fetch", "origin", expected])
        run(["git", "-C", str(checkout), "checkout", "--detach", expected])
        actual = output(["git", "-C", str(checkout), "rev-parse", "HEAD"])
    if actual != expected:
        raise RuntimeError(f"Author source commit mismatch: {actual} != {expected}")
    required = [
        "generate_proxy_migc_elite.py",
        "prompt/prompt_layout_v2.yaml",
        "MIGC/migc/migc_pipeline.py",
        "MIGC/migc/migc_utils.py",
        "MIGC/migc_gui_weights/v1-inference.yaml",
    ]
    for item in required:
        if not (checkout / item).is_file():
            raise FileNotFoundError(checkout / item)
    print(f"[ok] author source={expected[:12]}", flush=True)
    return checkout


def replace_with_symlink(target: Path, source: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink():
        if target.resolve() == source.resolve():
            return
        target.unlink()
    elif target.exists():
        if target.is_dir():
            shutil.rmtree(target)
        else:
            target.unlink()
    target.symlink_to(source, target_is_directory=source.is_dir())


def _isolated_json(py: Path, code: str) -> dict[str, Any]:
    raw = output([str(py), "-c", code])
    try:
        value = json.loads(raw.splitlines()[-1])
    except Exception as exc:
        raise RuntimeError(f"Isolated helper did not return JSON. Last output: {raw[-1000:]!r}") from exc
    if not isinstance(value, dict):
        raise RuntimeError("Isolated helper returned non-object JSON")
    return value


def hf_snapshot(
    py: Path, repo_id: str, revision: str, local: Path,
    allow_patterns: list[str] | None = None,
) -> dict[str, Any]:
    payload = json.dumps({
        "repo_id": repo_id,
        "revision": revision,
        "allow_patterns": allow_patterns,
    })
    code = '''
import json
from huggingface_hub import HfApi, snapshot_download
cfg = json.loads(__PAYLOAD__)
info = HfApi().model_info(cfg["repo_id"], revision=cfg["revision"])
resolved = str(info.sha)
path = snapshot_download(
    repo_id=cfg["repo_id"], revision=resolved,
    allow_patterns=cfg.get("allow_patterns"),
)
print(json.dumps({"resolved_revision": resolved, "cached_path": path}))
'''.replace("__PAYLOAD__", repr(payload))
    result = _isolated_json(py, code)
    cached = Path(str(result["cached_path"]))
    replace_with_symlink(local, cached)
    print(f"[download/cache] {repo_id}@{result['resolved_revision']}", flush=True)
    return {
        "repo_id": repo_id,
        "requested_revision": revision,
        "resolved_revision": str(result["resolved_revision"]),
        "path": rel(local),
    }


def hf_file(
    py: Path, repo_id: str, revision: str, filename: str, local: Path,
    expected_sha: str | None = None,
) -> dict[str, Any]:
    payload = json.dumps({"repo_id": repo_id, "revision": revision, "filename": filename})
    code = '''
import json
from huggingface_hub import HfApi, hf_hub_download
cfg = json.loads(__PAYLOAD__)
resolved = str(HfApi().model_info(cfg["repo_id"], revision=cfg["revision"]).sha)
path = hf_hub_download(repo_id=cfg["repo_id"], revision=resolved, filename=cfg["filename"])
print(json.dumps({"resolved_revision": resolved, "cached_path": path}))
'''.replace("__PAYLOAD__", repr(payload))
    result = _isolated_json(py, code)
    cached = Path(str(result["cached_path"]))
    actual = sha256_file(cached)
    if expected_sha and actual != expected_sha:
        raise RuntimeError(f"Checksum mismatch for {repo_id}/{filename}: {actual} != {expected_sha}")
    replace_with_symlink(local, cached)
    return {
        "repo_id": repo_id,
        "filename": filename,
        "resolved_revision": str(result["resolved_revision"]),
        "path": rel(local),
        "sha256": actual,
        "size_bytes": cached.stat().st_size,
    }


def prepare_migc(cfg: dict[str, Any], py: Path) -> dict[str, Any]:
    c = cfg["migc"]
    return hf_file(
        py,
        str(c["repo_id"]),
        str(c["revision"]),
        str(c["filename"]),
        resolve_path(str(c["checkpoint"])),
        str(c["sha256"]),
    )

def prepare_openai_clip(cfg: dict[str, Any], py: Path) -> dict[str, Any]:
    root = resolve_path(str(cfg["retrieval"]["clip_download_root"]))
    root.mkdir(parents=True, exist_ok=True)
    code = (
        "import json,clip; "
        f"p=clip._download(clip._MODELS[{cfg['retrieval']['clip_name']!r}], {str(root)!r}); "
        "print(json.dumps({'path':p}))"
    )
    raw = output([str(py), "-c", code])
    path = Path(json.loads(raw.splitlines()[-1])["path"])
    return {"name": str(cfg["retrieval"]["clip_name"]), "path": rel(path), "sha256": sha256_file(path)}


def _verify_external_snapshot_identity(
    py: Path, *, repo_id: str, revision: str, external: Path
) -> dict[str, Any]:
    """Compare tiny config/index files against the pinned HF revision."""
    payload = json.dumps({"repo_id": repo_id, "revision": revision})
    lines = [
        "import hashlib, json",
        "from huggingface_hub import HfApi, hf_hub_download",
        f"cfg=json.loads({payload!r})",
        "resolved=str(HfApi().model_info(cfg['repo_id'], revision=cfg['revision']).sha)",
        "out={'resolved_revision':resolved}",
        "for name in ('config.json','model.safetensors.index.json'):",
        "    p=hf_hub_download(repo_id=cfg['repo_id'], revision=resolved, filename=name)",
        "    out[name]=hashlib.sha256(open(p,'rb').read()).hexdigest()",
        "print(json.dumps(out))",
    ]
    reference = _isolated_json(py, "\n".join(lines))
    local = {
        "config.json": sha256_file(external / "config.json"),
        "model.safetensors.index.json": sha256_file(external / "model.safetensors.index.json"),
    }
    for name in local:
        if local[name] != reference[name]:
            raise RuntimeError(
                f"Mounted model does not match pinned {repo_id}@{reference['resolved_revision']}: "
                f"{name} sha256 {local[name]} != {reference[name]}"
            )
    return {
        "repo_id": repo_id,
        "requested_revision": revision,
        "resolved_revision": reference["resolved_revision"],
        "source": "external_read_only_mount",
        "path": str(external),
        "config_sha256": local["config.json"],
        "index_sha256": local["model.safetensors.index.json"],
        "weight_bytes": _model_weight_bytes(external),
    }


def prepare_large_model(
    cfg: dict[str, Any], py: Path, *, key: str, external: dict[str, Any]
) -> dict[str, Any]:
    c = cfg[key]
    local = resolve_path(str(c["local_snapshot"]))
    entry = external.get(key, {})
    if entry.get("found"):
        mounted = Path(str(entry["path"])).resolve()
        identity = _verify_external_snapshot_identity(
            py, repo_id=str(c["repo_id"]), revision=str(c["revision"]), external=mounted
        )
        replace_with_symlink(local, mounted)
        identity["local_alias"] = rel(local)
        print(f"[ok] {key} uses read-only mount: {mounted}", flush=True)
        return identity

    return hf_snapshot(
        py,
        str(c["repo_id"]),
        str(c["revision"]),
        local,
        allow_patterns=["*.json", "*.txt", "*.model", "*.safetensors", "*.safetensors.index.json", "*.py"],
    )


def prepare_assets(cfg: dict[str, Any], py: Path, external: dict[str, Any]) -> dict[str, Any]:
    assets: dict[str, Any] = {}
    assets["migc"] = prepare_migc(cfg, py)

    elite = cfg["elite"]
    assets["elite_global"] = hf_file(
        py, str(elite["repo_id"]), str(elite["revision"]), str(elite["global_filename"]),
        resolve_path(str(elite["global_mapper"])), None,
    )
    assets["elite_local"] = hf_file(
        py, str(elite["repo_id"]), str(elite["revision"]), str(elite["local_filename"]),
        resolve_path(str(elite["local_mapper"])), str(elite["local_sha256"]),
    )

    rv = cfg["realistic_vision"]
    assets["realistic_vision"] = hf_file(
        py, str(rv["repo_id"]), str(rv["revision"]), str(rv["filename"]),
        resolve_path(str(rv["path"])), str(rv["sha256"]),
    )

    assets["captioner"] = prepare_large_model(cfg, py, key="captioner", external=external)
    assets["layout_llm"] = prepare_large_model(cfg, py, key="layout_llm", external=external)
    clipv = cfg["elite"]
    assets["clip_vision"] = hf_snapshot(
        py, str(clipv["clip_vision_repo_id"]), str(clipv["clip_vision_revision"]),
        resolve_path(str(clipv["clip_vision_snapshot"])),
        allow_patterns=["config.json", "preprocessor_config.json", "model.safetensors"],
    )
    sd = cfg["sd15_components"]
    assets["sd15_components"] = hf_snapshot(
        py, str(sd["repo_id"]), str(sd["revision"]), resolve_path(str(sd["local_snapshot"])),
        allow_patterns=["text_encoder/*", "tokenizer/*", "scheduler/*", "feature_extractor/*", "model_index.json"],
    )
    assets["openai_clip"] = prepare_openai_clip(cfg, py)
    return assets


def worker_preflight(cfg: dict[str, Any], config_path: Path, stage: str) -> None:
    py = env_python(cfg)
    worker = METHOD_DIR / "official_proxy_worker.py"
    run([str(py), str(worker), "--config", str(config_path), "--stage", stage])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--env-only", action="store_true", help="Create and validate only the isolated environment")
    parser.add_argument("--check-inputs", action="store_true", help="Only discover mounted giant model inputs; install/download nothing")
    args = parser.parse_args()
    config_path = resolve_path(args.config)
    cfg = load_yaml(config_path)

    on_kaggle = Path("/kaggle/working").is_dir()
    external = discover_external_large_assets(
        cfg,
        fail_if_missing=(on_kaggle and not args.check_inputs and bool(cfg["asset_storage"].get("require_external_large_assets_on_kaggle", True))),
    )
    if args.check_inputs:
        print(json.dumps({"on_kaggle": on_kaggle, "external_assets": external}, indent=2), flush=True)
        if not all(external.get(k, {}).get("found") for k in ("captioner", "layout_llm")):
            raise SystemExit(2)
        return

    tracker = PhaseTracker("imagine_seek_prepare", total=2 if args.env_only else 7)
    with tracker.phase("Resource/storage preflight"):
        disk = ensure_local_free_disk(cfg, external) if not args.env_only else {"skipped_for_env_only": True}
    with tracker.phase("Create/validate isolated Python 3.10 IP-CIR environment"):
        env_info = prepare_env(cfg, args.force)
    if args.env_only:
        tracker.finish()
        return

    with tracker.phase("Pin released Imagine-and-Seek source"):
        source = prepare_source(cfg)
    with tracker.phase("Import-preflight released MIGC+ELITE adapter"):
        worker_preflight(cfg, config_path, "import")
    with tracker.phase("Download/cache exact public model assets"):
        assets = prepare_assets(cfg, env_python(cfg), external)
    with tracker.phase("Pipeline-preflight MIGC + ELITE + Realistic Vision"):
        worker_preflight(cfg, config_path, "pipeline")
    with tracker.phase("Write reproducibility marker"):
        marker = resolve_path(str(cfg["migc"]["prepared_marker"]))
        payload = {
            "schema": 4,
            "method": cfg["method"],
            "implementation_status": "OFFICIAL_SOURCE_ADAPTED",
            "config": rel(config_path),
            "author_source": {
                "repository": cfg["author_source"]["repository"],
                "commit": cfg["author_source"]["commit"],
                "checkout": rel(source),
            },
            "isolated_env": env_info,
            "resource_preflight": disk,
            "external_asset_discovery": external,
            "assets": assets,
            "cpr_adaptation": {
                "dense_caption_boundary": "HF BLIP2 OPT-6.7B COCO corresponding to released LAVIS caption_coco_opt6.7b",
                "layout_prompt": "released prompt/prompt_layout_v2.yaml",
                "proxy_reference_mask": cfg["proxy"]["reference_mask"],
                "uses_gt_target_box": False,
                "uses_gt_target_identity": False,
                "uses_cpr_labels": False,
                "small_model_fallback": False,
            },
        }
        write_json(marker, payload)
        print(f"[ok] marker={rel(marker)}", flush=True)
    tracker.finish()


if __name__ == "__main__":
    main()
