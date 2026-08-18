#!/usr/bin/env python3
"""Prepare pinned WISER source, isolated runtimes, BAGEL, and Qwen2.5-VL assets."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[3]
METHOD_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = METHOD_DIR / "config.yaml"


def load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected YAML mapping: {path}")
    return value


def resolve(value: str) -> Path:
    p = Path(value)
    return (p if p.is_absolute() else ROOT / p).resolve()


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except Exception:
        return str(path.resolve())


def run(cmd: list[str], cwd: Path = ROOT) -> None:
    print("$ " + " ".join(cmd), flush=True)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    subprocess.run(cmd, cwd=str(cwd), env=env, check=True)


def output(cmd: list[str], cwd: Path = ROOT) -> str:
    return subprocess.check_output(cmd, cwd=str(cwd), text=True).strip()


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_disk(cfg: dict[str, Any]) -> None:
    usage = shutil.disk_usage(ROOT)
    free = usage.free / 1024**3
    need = float(cfg["isolated_env"]["minimum_free_disk_gib"])
    print(f"[preflight] free disk={free:.1f} GiB, requested floor={need:.1f} GiB")
    if free < need:
        raise RuntimeError(
            "WISER uses BAGEL-7B-MoT + Qwen2.5-VL-7B + BLIP2-FlanT5-XXL caches. "
            f"Need at least {need:.1f} GiB free before preparation; found {free:.1f} GiB."
        )


def ensure_venv(cfg: dict[str, Any], section: str, req_file: Path) -> Path:
    spec = cfg[section]
    env_dir = resolve(str(spec["path"]))
    py = resolve(str(spec["python"]))
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError("uv is missing; root requirements.txt should install it first")
    if not py.is_file():
        env_dir.parent.mkdir(parents=True, exist_ok=True)
        run([uv, "python", "install", str(spec["python_version"])])
        run([uv, "venv", "--python", str(spec["python_version"]), "--seed", str(env_dir)])
    run([
        uv, "pip", "install", "--python", str(py),
        "--index-url", str(spec["torch_index_url"]),
        f"torch=={spec['torch_version']}", f"torchvision=={spec['torchvision_version']}",
    ])
    run([uv, "pip", "install", "--python", str(py), "-r", str(req_file)])
    return py


def prepare_source(cfg: dict[str, Any]) -> dict[str, Any]:
    spec = cfg["author_source"]
    checkout = resolve(str(spec["local_checkout"]))
    expected = str(spec["commit"])
    if not checkout.exists():
        checkout.parent.mkdir(parents=True, exist_ok=True)
        run(["git", "clone", str(spec["repository"]), str(checkout)])
    if not (checkout / ".git").is_dir():
        raise RuntimeError(f"Not a git checkout: {checkout}")
    dirty = output(["git", "-C", str(checkout), "status", "--porcelain", "--untracked-files=no"])
    if dirty:
        raise RuntimeError(f"Refusing modified WISER source at {checkout}:\n{dirty}")
    actual = output(["git", "-C", str(checkout), "rev-parse", "HEAD"])
    if actual != expected:
        run(["git", "-C", str(checkout), "fetch", "origin", expected])
        run(["git", "-C", str(checkout), "checkout", "--detach", expected])
        actual = output(["git", "-C", str(checkout), "rev-parse", "HEAD"])
    if actual != expected:
        raise RuntimeError(f"WISER commit mismatch: {actual} != {expected}")
    required = ["src/bagel_inference.py", "src/prompts.py", "src/generate_captions.py", "src/compute_results.py"]
    for name in required:
        if not (checkout / name).is_file():
            raise FileNotFoundError(checkout / name)
    return {"path": rel(checkout), "commit": actual}


def snapshot_download(py: Path, repo_id: str, revision: str, local_dir: Path) -> dict[str, Any]:
    local_dir.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps({"repo_id": repo_id, "revision": revision, "local_dir": str(local_dir)})
    code = r"""
import json
from huggingface_hub import HfApi, snapshot_download
cfg = json.loads(__PAYLOAD__)
sha = str(HfApi().model_info(cfg["repo_id"], revision=cfg["revision"]).sha)
path = snapshot_download(repo_id=cfg["repo_id"], revision=sha, local_dir=cfg["local_dir"])
print(json.dumps({"sha": sha, "path": path}))
""".replace("__PAYLOAD__", repr(payload))
    raw = output([str(py), "-c", code])
    result = json.loads(raw.splitlines()[-1])
    return {"repo_id": repo_id, "resolved_revision": result["sha"], "path": rel(Path(result["path"]))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config_path = resolve(args.config)
    cfg = load_yaml(config_path)
    ensure_disk(cfg)

    print("[1/5] Prepare pinned WISER source")
    source = prepare_source(cfg)

    print("[2/5] Prepare modern WISER/BAGEL/Qwen environment")
    wiser_py = ensure_venv(cfg, "isolated_env", METHOD_DIR / "wiser_requirements.txt")

    print("[3/5] Prepare isolated legacy LAVIS caption environment")
    caption_py = ensure_venv(cfg, "caption_env", METHOD_DIR / "caption_requirements.txt")

    print("[4/5] Download/cache BAGEL and Qwen verifier")
    bagel = cfg["models"]["bagel"]
    verifier = cfg["models"]["verifier"]
    bagel_asset = snapshot_download(
        wiser_py, str(bagel["repo_id"]), str(bagel["revision"]), resolve(str(bagel["local_dir"]))
    )
    verifier_asset = snapshot_download(
        wiser_py, str(verifier["repo_id"]), str(verifier["revision"]), resolve(str(verifier["local_dir"]))
    )

    print("[5/5] Smoke-test imports and write prepared marker")
    probe = r"""
import json, torch, transformers, open_clip
from qwen_vl_utils import process_vision_info
print(json.dumps({"torch": torch.__version__, "transformers": transformers.__version__, "cuda": torch.cuda.is_available()}))
"""
    raw = output([str(wiser_py), "-c", probe])
    runtime = json.loads(raw.splitlines()[-1])
    if not runtime.get("cuda"):
        raise RuntimeError("Prepared WISER runtime cannot see CUDA")
    caption_probe = output([str(caption_py), "-c", "import lavis,transformers; print(transformers.__version__)"])

    marker = resolve(str(cfg["checkpoint"]["prepared_marker"]))
    write_json(marker, {
        "schema": 1,
        "config_sha256": sha256(config_path),
        "author_source": source,
        "wiser_python": rel(wiser_py),
        "caption_python": rel(caption_py),
        "runtime": runtime,
        "caption_transformers": caption_probe.splitlines()[-1],
        "assets": {"bagel": bagel_asset, "verifier": verifier_asset},
        "note": "OpenCLIP and LAVIS model weights are lazily cached on first inference/caption run.",
    })
    print(f"Prepared WISER marker -> {rel(marker)}")


if __name__ == "__main__":
    main()
