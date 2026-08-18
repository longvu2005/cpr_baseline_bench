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


def ensure_free_disk(cfg: dict[str, Any]) -> dict[str, float]:
    usage = shutil.disk_usage(ROOT)
    free_gib = usage.free / 1024**3
    required = float(cfg["isolated_env"]["minimum_free_disk_gib"])
    print(f"[preflight] free disk={free_gib:.1f} GiB; required={required:.1f} GiB", flush=True)
    if free_gib < required:
        raise RuntimeError(
            "Insufficient free disk for the paper-faithful P10 assets. "
            f"Need at least {required:.1f} GiB free, found {free_gib:.1f} GiB. "
            "Do not switch to smaller models for final benchmark reporting. "
            "Use a larger runtime/server or mount pre-downloaded checkpoints."
        )
    return {"free_gib_at_preflight": round(free_gib, 3), "required_gib": required}


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
        run([uv, "python", "install", str(env_cfg["python_version"])])
        run([uv, "venv", "--python", str(env_cfg["python_version"]), "--seed", str(env_dir)])

    torch_spec = f"torch=={env_cfg['torch_version']}"
    tv_spec = f"torchvision=={env_cfg['torchvision_version']}"
    run([
        uv, "pip", "install", "--python", str(py),
        "--index-url", str(env_cfg["torch_index_url"]), torch_spec, tv_spec,
    ])
    run([
        uv, "pip", "install", "--python", str(py), "-r", str(METHOD_DIR / "generator_requirements.txt")
    ])
    # AutoGPTQ's CUDA-12.1 0.7.1 wheel is built against torch 2.2.1+cu121.
    run([
        str(py), "-m", "pip", "install",
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
    path = resolve_path(str(c["checkpoint"]))
    if path.is_file() and path.stat().st_size > 100_000_000:
        return {"path": rel(path), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    tmp.unlink(missing_ok=True)
    code = (
        "import gdown; "
        f"r=gdown.download(id={str(c['google_drive_id'])!r}, output={str(tmp)!r}, quiet=False); "
        "print('GDOWN_RESULT='+str(r))"
    )
    run([str(py), "-c", code])
    if not tmp.is_file() or tmp.stat().st_size < 100_000_000:
        tmp.unlink(missing_ok=True)
        raise RuntimeError("Failed to download a valid public MIGC_SD14.ckpt")
    os.replace(tmp, path)
    return {"path": rel(path), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}


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


def prepare_assets(cfg: dict[str, Any], py: Path) -> dict[str, Any]:
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

    cap = cfg["captioner"]
    assets["captioner"] = hf_snapshot(
        py, str(cap["repo_id"]), str(cap["revision"]), resolve_path(str(cap["local_snapshot"])),
        allow_patterns=["*.json", "*.txt", "*.model", "*.safetensors", "*.safetensors.index.json"],
    )
    llm = cfg["layout_llm"]
    assets["layout_llm"] = hf_snapshot(
        py, str(llm["repo_id"]), str(llm["revision"]), resolve_path(str(llm["local_snapshot"])),
        allow_patterns=["*.json", "*.txt", "*.model", "*.safetensors", "*.safetensors.index.json", "*.py"],
    )
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
    args = parser.parse_args()
    config_path = resolve_path(args.config)
    cfg = load_yaml(config_path)

    tracker = PhaseTracker("imagine_seek_prepare", total=2 if args.env_only else 7)
    with tracker.phase("Resource preflight"):
        disk = ensure_free_disk(cfg) if not args.env_only else {"skipped_for_env_only": True}
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
        assets = prepare_assets(cfg, env_python(cfg))
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
