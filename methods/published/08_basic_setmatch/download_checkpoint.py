#!/usr/bin/env python3
"""Prepare official BASIC/i-CIR source and all P8 runtime assets."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Any

import yaml

os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark_progress import PhaseTracker  # noqa: E402

DEFAULT_CONFIG = Path(__file__).resolve().parent / "config.yaml"


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise TypeError(f"Expected YAML mapping: {path}")
    return data


def resolve_path(value: str) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else ROOT / path).resolve()


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def generated_python_artifact(path: str) -> bool:
    normalized = path.strip().strip('"').replace("\\", "/")
    return normalized.lower().endswith((".pyc", ".pyo"))


def tracked_dirty(checkout: Path) -> str:
    output = subprocess.check_output(
        ["git", "-C", str(checkout), "status", "--porcelain", "--untracked-files=no"],
        text=True,
    )
    dirty: list[str] = []
    for line in output.splitlines():
        if not line.strip():
            continue
        path = line[3:].strip() if len(line) >= 4 else line
        parts = path.split(" -> ")
        if parts and all(generated_python_artifact(x) for x in parts):
            continue
        dirty.append(line)
    return "\n".join(dirty)


def prepare_source(cfg: dict[str, Any]) -> Path:
    if shutil.which("git") is None:
        raise RuntimeError("System tool 'git' is required for official i-CIR source preparation")

    source = cfg["source"]
    checkout = resolve_path(str(source["local_checkout"]))
    expected = str(source["commit"])
    repository = str(source["repository"])

    if not checkout.exists():
        if not bool(source.get("auto_clone", True)):
            raise FileNotFoundError(f"Missing official source checkout: {rel(checkout)}")
        checkout.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", repository, str(checkout)], check=True)

    if not (checkout / ".git").is_dir():
        raise RuntimeError(f"Official source path is not a git checkout: {rel(checkout)}")

    dirty = tracked_dirty(checkout)
    if dirty:
        raise RuntimeError(
            f"Pinned official i-CIR source has tracked local modifications: {rel(checkout)}\n{dirty}"
        )

    actual = subprocess.check_output(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual != expected:
        subprocess.run(["git", "-C", str(checkout), "fetch", "origin", expected], check=True)
        subprocess.run(
            ["git", "-C", str(checkout), "checkout", "--detach", expected], check=True
        )
        actual = subprocess.check_output(
            ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True
        ).strip()
    if actual != expected:
        raise RuntimeError(f"Official source commit mismatch: expected {expected}, got {actual}")

    print(f"[ok] official i-CIR source @ {actual[:12]} -> {rel(checkout)}", flush=True)
    return checkout


def validate_source_layout(cfg: dict[str, Any], checkout: Path) -> dict[str, dict[str, Any]]:
    required = [
        checkout / "run_retrieval.py",
        checkout / "utils_retrieval.py",
        checkout / "utils_features.py",
        checkout / "requirements.txt",
    ]
    resources = cfg["basic"]["official_resources"]
    for relative in resources.values():
        required.append(checkout / str(relative))

    missing = [rel(path) for path in required if not path.is_file() or path.stat().st_size <= 0]
    if missing:
        raise FileNotFoundError(
            "Pinned official i-CIR checkout is incomplete:\n  - " + "\n  - ".join(missing)
        )

    inventory: dict[str, dict[str, Any]] = {}
    for path in required:
        key = str(path.relative_to(checkout))
        inventory[key] = {
            "size": int(path.stat().st_size),
            "sha256": sha256_file(path),
        }
    return inventory


def download_http(
    *,
    url: str,
    path: Path,
    force: bool,
    expected_sha256: str | None = None,
    expected_prefix: str | None = None,
) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)

    def validate(candidate: Path) -> dict[str, Any]:
        if not candidate.is_file() or candidate.stat().st_size <= 0:
            raise RuntimeError(f"Missing/empty artifact: {rel(candidate)}")
        actual = sha256_file(candidate)
        if expected_sha256 is not None and actual != expected_sha256:
            raise RuntimeError(
                f"Checksum mismatch for {rel(candidate)}: expected {expected_sha256}, got {actual}"
            )
        if expected_prefix is not None and not actual.startswith(expected_prefix):
            raise RuntimeError(
                f"Checksum prefix mismatch for {rel(candidate)}: "
                f"expected {expected_prefix}, got {actual}"
            )
        return {"size": int(candidate.stat().st_size), "sha256": actual}

    if path.is_file() and not force:
        try:
            info = validate(path)
            print(f"[skip] valid artifact: {rel(path)}", flush=True)
            return info
        except Exception as error:
            print(f"[warn] replacing invalid {rel(path)}: {error}", flush=True)

    temp = path.with_name(path.name + ".part")
    temp.unlink(missing_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "cpr-baseline-bench/1.0"})
    print(f"[download] {url}", flush=True)
    try:
        with urllib.request.urlopen(request, timeout=120) as response, temp.open("wb") as handle:
            total_raw = response.headers.get("Content-Length")
            total = int(total_raw) if total_raw and total_raw.isdigit() else None
            downloaded = 0
            while True:
                chunk = response.read(8 * 1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
                downloaded += len(chunk)
                if total:
                    print(
                        f"\r  {downloaded / 2**20:,.1f}/{total / 2**20:,.1f} MiB "
                        f"({100.0 * downloaded / total:5.1f}%)",
                        end="",
                        flush=True,
                    )
        if total:
            print(flush=True)
        info = validate(temp)
        os.replace(temp, path)
    except Exception:
        temp.unlink(missing_ok=True)
        raise
    print(f"[ok] {rel(path)} sha256={info['sha256']}", flush=True)
    return info


def configure_basic_cache(cache_root: Path) -> None:
    cache_root.mkdir(parents=True, exist_ok=True)
    home = cache_root / "home"
    home.mkdir(parents=True, exist_ok=True)
    os.environ["HOME"] = str(home)
    os.environ["TORCH_HOME"] = str(cache_root / "torch")
    os.environ["HF_HOME"] = str(cache_root / "huggingface")
    os.environ["XDG_CACHE_HOME"] = str(cache_root / "xdg")
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"


def run_official_clip_loader(
    checkout: Path,
    *,
    offline: bool,
) -> subprocess.CompletedProcess[str]:
    """Run the pinned official CLIP loader in an isolated child process.

    Cache probing used to monkey-patch ``socket.socket.connect`` in this process.
    On Kaggle, Hugging Face/httpx can retain networking state across retries, so
    the subsequent online population attempt could still call the blocked function.
    A fresh child process makes the two modes independent and guarantees that an
    offline verification cannot contaminate cache preparation.
    """

    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["HF_HUB_DISABLE_TELEMETRY"] = "1"
    if offline:
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"
    else:
        # Preparation is the one stage where networking is intentionally allowed.
        # Do not inherit an offline shell setting into the population subprocess.
        env.pop("HF_HUB_OFFLINE", None)
        env.pop("TRANSFORMERS_OFFLINE", None)

    code = (
        "import socket\n"
        "import sys\n"
        "from pathlib import Path\n"
        "checkout = Path(sys.argv[1]).resolve()\n"
        "offline = sys.argv[2] == \"1\"\n"
        "if offline:\n"
        "    def blocked_connect(self, address):\n"
        "        raise RuntimeError(f\"Network blocked during isolated BASIC offline verification: {address!r}\")\n"
        "    socket.socket.connect = blocked_connect\n"
        "    socket.socket.connect_ex = blocked_connect\n"
        "sys.path.insert(0, str(checkout))\n"
        "from utils_features import load_model\n"
        "bundle = load_model(\"clip\", \"cpu\")\n"
        "del bundle\n"
        "print(\"OFFICIAL_BASIC_CLIP_LOADER_OK\", flush=True)\n"
    )
    return subprocess.run(
        [sys.executable, "-u", "-c", code, str(checkout), "1" if offline else "0"],
        cwd=checkout,
        env=env,
        text=True,
        # Online population inherits stdout/stderr so large model-download progress
        # stays visible. Offline probes are captured to keep expected cache misses quiet.
        capture_output=offline,
        check=False,
    )


def format_loader_failure(result: subprocess.CompletedProcess[str]) -> str:
    stdout = (result.stdout or "").strip()
    stderr = (result.stderr or "").strip()
    parts = [f"exit={result.returncode}"]
    if stdout:
        parts.append(f"stdout:\n{stdout[-4000:]}")
    if stderr:
        parts.append(f"stderr:\n{stderr[-8000:]}")
    return "\n".join(parts)


def import_official(checkout: Path):
    checkout_str = str(checkout)
    if checkout_str not in sys.path:
        sys.path.insert(0, checkout_str)
    run_retrieval = importlib.import_module("run_retrieval")
    return run_retrieval


def normalized_expected_preset(cfg: dict[str, Any]) -> dict[str, Any]:
    return dict(cfg["basic"]["expected_preset"])


def verify_official_preset(cfg: dict[str, Any], run_retrieval) -> dict[str, Any]:
    presets = getattr(run_retrieval, "METHOD_PRESETS", None)
    if not isinstance(presets, dict) or "basic" not in presets:
        raise RuntimeError("Pinned official run_retrieval.py does not expose METHOD_PRESETS['basic']")
    preset = dict(presets["basic"])
    preset.pop("description", None)
    expected = normalized_expected_preset(cfg)
    if preset != expected:
        raise RuntimeError(
            "Official BASIC preset differs from the benchmark config. "
            "Refusing to silently run a changed method.\n"
            f"official={preset}\nexpected={expected}"
        )
    print("[ok] official METHOD_PRESETS['basic'] matches config.yaml", flush=True)
    return preset


def list_cache_files(cache_root: Path) -> list[dict[str, Any]]:
    files: list[dict[str, Any]] = []
    for path in sorted(p for p in cache_root.rglob("*") if p.is_file()):
        if path.name.endswith((".lock", ".part", ".tmp")):
            continue
        files.append(
            {
                "path": str(path.relative_to(cache_root)),
                "size": int(path.stat().st_size),
            }
        )
    return files


def prepare_basic_model_cache(
    cfg: dict[str, Any], checkout: Path, force: bool
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    cache_root = resolve_path(str(cfg["runtime_assets"]["basic_model_cache_root"]))
    if force and cache_root.exists():
        shutil.rmtree(cache_root)
    configure_basic_cache(cache_root)

    run_retrieval = import_official(checkout)
    preset = verify_official_preset(cfg, run_retrieval)

    # Probe an existing cache in an isolated offline process. If it is incomplete,
    # populate it in a separate online process, then verify again offline. Never
    # monkey-patch networking in this parent process: that caused the Kaggle failure
    # where the online retry still hit ``blocked_connect``.
    offline_probe = run_official_clip_loader(checkout, offline=True)
    if offline_probe.returncode == 0:
        print("[skip] official BASIC CLIP cache is already offline-complete", flush=True)
    else:
        print(
            "[prepare] BASIC CLIP cache miss/incomplete; populating with the pinned "
            "official loader",
            flush=True,
        )
        online_load = run_official_clip_loader(checkout, offline=False)
        if online_load.returncode != 0:
            raise RuntimeError(
                "Official BASIC CLIP cache population failed.\n"
                + format_loader_failure(online_load)
            )
        offline_verify = run_official_clip_loader(checkout, offline=True)
        if offline_verify.returncode != 0:
            raise RuntimeError(
                "Official BASIC CLIP loaded online, but the prepared cache does not "
                "work offline. Refusing an unreproducible runtime cache.\n"
                + format_loader_failure(offline_verify)
            )
        print("[ok] official BASIC CLIP loader verified offline after preparation", flush=True)

    files = list_cache_files(cache_root)
    if not files:
        raise RuntimeError(
            "Official BASIC model loaded, but no files were captured under the configured "
            f"cache root {rel(cache_root)}. Refusing an unreproducible global cache."
        )
    return preset, files


def write_marker(
    cfg: dict[str, Any],
    checkout: Path,
    source_inventory: dict[str, dict[str, Any]],
    preset: dict[str, Any],
    model_cache_files: list[dict[str, Any]],
    selector_info: dict[str, Any],
    detector_info: dict[str, Any],
) -> Path:
    marker_path = resolve_path(str(cfg["runtime_assets"]["prepared_marker"]))
    cache_root = resolve_path(str(cfg["runtime_assets"]["basic_model_cache_root"]))
    payload = {
        "schema": 1,
        "method": str(cfg["method"]),
        "source": {
            "repository": str(cfg["source"]["repository"]),
            "commit": str(cfg["source"]["commit"]),
            "checkout": rel(checkout),
            "tracked_inventory": source_inventory,
        },
        "basic": {
            "backbone": str(cfg["basic"]["backbone"]),
            "preset": preset,
            "model_cache_root": rel(cache_root),
            "model_cache_files": model_cache_files,
        },
        "selector": {
            "path": rel(resolve_path(str(cfg["runtime_assets"]["selector"]["path"]))),
            **selector_info,
        },
        "detector": {
            "path": rel(resolve_path(str(cfg["runtime_assets"]["detector"]["path"]))),
            **detector_info,
        },
    }
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    marker_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[ok] prepared marker: {rel(marker_path)}", flush=True)
    return marker_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare P8 BASIC + SetMatch artifacts")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    config_path = resolve_path(args.config)
    cfg = load_yaml(config_path)
    tracker = PhaseTracker("basic_setmatch_prepare", total=4)

    tracker.advance("Pin and validate official i-CIR/BASIC source")
    checkout = prepare_source(cfg)
    source_inventory = validate_source_layout(cfg, checkout)

    tracker.advance("Prepare official BASIC CLIP model cache")
    preset, model_cache_files = prepare_basic_model_cache(cfg, checkout, args.force)

    tracker.advance("Prepare benchmark detector and query-person selector")
    selector_cfg = cfg["runtime_assets"]["selector"]
    selector_info = download_http(
        url=str(selector_cfg["url"]),
        path=resolve_path(str(selector_cfg["path"])),
        force=args.force,
        expected_sha256=str(selector_cfg["sha256"]),
    )
    detector_cfg = cfg["runtime_assets"]["detector"]
    detector_info = download_http(
        url=str(detector_cfg["url"]),
        path=resolve_path(str(detector_cfg["path"])),
        force=args.force,
        expected_prefix=str(detector_cfg["sha256_prefix"]),
    )

    tracker.advance("Write reproducibility marker")
    write_marker(
        cfg,
        checkout,
        source_inventory,
        preset,
        model_cache_files,
        selector_info,
        detector_info,
    )
    tracker.finish()


if __name__ == "__main__":
    main()
