"""Per-method, repeatable Python runtime without modifying the Kaggle base environment.

The interpreter running run_baseline.py remains responsible for dataset validation,
official evaluation, and tables. Each method gets a separate runtime for its own
requirements, checkpoint preparation, and inference. System site packages are
visible to reuse Kaggle's CUDA-enabled torch installation without copying it.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

ENV_DIR_NAME = ".baseline_venvs"
MARKER_NAME = ".requirements-ready.json"


def _requirements_digest(path: Path) -> tuple[str, str]:
    """Hash direct and nested -r/-c files, not just the top-level requirements."""
    visited: set[Path] = set()
    lines: list[str] = []
    contents: list[str] = []

    def visit(current: Path) -> None:
        current = current.resolve()
        if current in visited:
            return
        if not current.is_file():
            raise FileNotFoundError(f"Missing requirements file: {current}")
        visited.add(current)
        raw = current.read_text(encoding="utf-8")
        contents.append(raw)
        lines.append(f"{current}: {hashlib.sha256(raw.encode('utf-8')).hexdigest()}")
        for line in raw.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = shlex.split(stripped, comments=True)
            if not parts:
                continue
            nested = None
            if parts[0] in ("-r", "--requirement", "-c", "--constraint") and len(parts) > 1:
                nested = parts[1]
            elif parts[0].startswith("-r") and len(parts[0]) > 2:
                nested = parts[0][2:]
            elif parts[0].startswith("-c") and len(parts[0]) > 2:
                nested = parts[0][2:]
            if nested:
                visit(current.parent / nested)

    visit(path)
    digest = hashlib.sha256("\n".join(sorted(lines)).encode("utf-8")).hexdigest()
    return digest, "\n".join(contents)


def _method_env(root: Path, method_id: str) -> tuple[Path, Path]:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", method_id):
        raise ValueError(f"Unsafe method id: {method_id!r}")
    directory = root / ENV_DIR_NAME / method_id
    executable = directory / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    return directory, executable


def _child_env(directory: Path) -> dict[str, str]:
    env = os.environ.copy()
    env.update({
        "USE_FLAX": "0",  # Transformers 4.x must not auto-import unrelated Kaggle JAX.
        "USE_TF": "0",    # These baselines use PyTorch, not TensorFlow.
        "PYTHONUNBUFFERED": "1",
        "PYTHONNOUSERSITE": "1",
        "VIRTUAL_ENV": str(directory),
    })
    env["PATH"] = str(directory / ("Scripts" if os.name == "nt" else "bin")) + os.pathsep + env.get("PATH", "")
    return env


def _smoke_test(executable: Path, requirements: str, env: dict[str, str]) -> None:
    """Fail before downloading checkpoints if an expected import is missing."""
    imports = ["import numpy", "import torch"]
    if "github.com/openai/CLIP" in requirements or "git+https://github.com/openai/CLIP" in requirements:
        imports.append("import clip")
    if "transformers" in requirements:
        imports.append("from transformers import AutoTokenizer, BertModel")
    command = [str(executable), "-c", "; ".join(imports)]
    try:
        subprocess.run(command, check=True, env=env)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "Method environment failed its import check. "
            "Run without --skip-install to reinstall its requirements. "
            f"Interpreter: {executable}"
        ) from exc


def prepare_method_python(*, root: Path, method_id: str, requirements_path: Path,
                          skip_install: bool) -> tuple[str, dict[str, str]]:
    """Create/reuse a per-method venv; install only when its requirements changed."""
    root = root.resolve()
    directory, executable = _method_env(root, method_id)
    digest, files = _requirements_digest(requirements_path)
    env = _child_env(directory)
    marker = directory / MARKER_NAME
    expected = {
        "requirements_sha256": digest,
        "base_python": str(Path(sys.executable).resolve()),
        "base_python_version": list(sys.version_info[:3]),
    }

    if not executable.is_file():
        if skip_install:
            raise RuntimeError(
                f"Missing isolated environment for {method_id}: {executable}. "
                f"First run: python run_baseline.py {method_id} (without --skip-install)."
            )
        directory.parent.mkdir(parents=True, exist_ok=True)
        print(f"[environment] Creating isolated runtime: {directory}", flush=True)
        subprocess.run([sys.executable, "-m", "venv", "--system-site-packages", str(directory)],
                       check=True)

    try:
        ready = json.loads(marker.read_text(encoding="utf-8")) == expected
    except (OSError, ValueError):
        ready = False

    if not ready:
        if skip_install:
            raise RuntimeError(
                f"Environment for {method_id} has missing/stale requirements. "
                f"Run python run_baseline.py {method_id} without --skip-install."
            )
        marker.unlink(missing_ok=True)
        print(f"[environment] Installing method requirements into {directory}", flush=True)
        subprocess.run([str(executable), "-m", "pip", "install", "-r", str(requirements_path)],
                       check=True, cwd=root, env=env)
        _smoke_test(executable, files, env)
        marker.write_text(json.dumps(expected, indent=2) + "\n", encoding="utf-8")
    else:
        print(f"[environment] Reusing prepared runtime: {directory}", flush=True)
        _smoke_test(executable, files, env)

    return str(executable), env
