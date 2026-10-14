"""Per-method Python runtimes that do not modify Kaggle's global packages.

A Python virtual environment is created WITHOUT ensurepip because Kaggle's
system Python may not have a working ensurepip installation. The method's pip
is resolved from inherited system site-packages or managed via host pip's
--python option. Installation still targets the virtual environment.
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
VENV_CREATED_NAME = ".venv-created-ok"
BOOTSTRAP_SCHEMA = 2


def _requirements_digest(path: Path) -> tuple[str, str]:
    """Hash top-level and nested -r/-c requirements files."""
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
        "USE_FLAX": "0",
        "USE_TF": "0",
        "PYTHONUNBUFFERED": "1",
        "PYTHONNOUSERSITE": "1",
        "VIRTUAL_ENV": str(directory),
    })
    env["PATH"] = str(directory / ("Scripts" if os.name == "nt" else "bin")) + os.pathsep + env.get("PATH", "")
    # Prevent an unrelated PYTHONHOME/PYTHONPATH from redirecting venv imports.
    env.pop("PYTHONHOME", None)
    env.pop("PYTHONPATH", None)
    return env


def _venv_python_works(directory: Path, executable: Path, env: dict[str, str]) -> bool:
    if not executable.is_file() or not (directory / "pyvenv.cfg").is_file():
        return False
    code = (
        "import pathlib,sys; "
        "actual=pathlib.Path(sys.prefix).resolve(); "
        "expected=pathlib.Path(sys.argv[1]).resolve(); "
        "assert actual == expected, (actual, expected); "
        "assert sys.prefix != sys.base_prefix, 'not a virtual environment'"
    )
    try:
        result = subprocess.run(
            [str(executable), "-c", code, str(directory)],
            env=env, capture_output=True, text=True, check=False,
        )
    except OSError:
        return False
    return result.returncode == 0


def _create_venv(directory: Path, executable: Path, env: dict[str, str]) -> None:
    """Rebuild an interrupted venv without invoking Kaggle's broken ensurepip."""
    if directory.is_symlink():
        raise RuntimeError(f"Refusing to overwrite symbolic link: {directory}")
    directory.parent.mkdir(parents=True, exist_ok=True)
    print(f"[environment] Creating isolated runtime (without ensurepip): {directory}", flush=True)
    subprocess.run(
        [sys.executable, "-m", "venv", "--clear", "--system-site-packages",
         "--without-pip", str(directory)],
        check=True,
    )
    if not _venv_python_works(directory, executable, env):
        raise RuntimeError(f"venv created but its Python interpreter is not usable: {executable}")
    (directory / VENV_CREATED_NAME).write_text(str(BOOTSTRAP_SCHEMA) + "\n", encoding="utf-8")


def _pip_command(executable: Path, env: dict[str, str]) -> list[str]:
    """Choose pip for the target venv, never pip installing into the host."""
    cmd = [str(executable), "-m", "pip"]
    if subprocess.run(cmd + ["--version"], env=env, stdout=subprocess.DEVNULL,
                      stderr=subprocess.DEVNULL, check=False).returncode == 0:
        return cmd
    # pip>=22.3 can manage an interpreter even if it has no pip installed.
    cmd = [sys.executable, "-m", "pip", "--python", str(executable)]
    if subprocess.run(cmd + ["--version"], env=env, stdout=subprocess.DEVNULL,
                      stderr=subprocess.DEVNULL, check=False).returncode == 0:
        return cmd
    raise RuntimeError(
        "Could not access pip for isolated Python. Install/upgrade pip in the "
        "host kernel (pip>=22.3 supports --python); do not reinstall requirements globally."
    )


def _smoke_test(executable: Path, requirements: str, env: dict[str, str]) -> None:
    imports = ["import numpy", "import torch"]
    if "github.com/openai/CLIP" in requirements or "git+https://github.com/openai/CLIP" in requirements:
        imports.append("import clip")
    if "transformers" in requirements:
        imports.append("from transformers import AutoTokenizer, BertModel")
    try:
        subprocess.run([str(executable), "-c", "; ".join(imports)], check=True, env=env)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "Method environment failed its import check. Run without --skip-install "
            f"to reinstall its requirements. Interpreter: {executable}"
        ) from exc


def prepare_method_python(*, root: Path, method_id: str, requirements_path: Path,
                          skip_install: bool) -> tuple[str, dict[str, str]]:
    root = root.resolve()
    directory, executable = _method_env(root, method_id)
    if directory.is_symlink():
        raise RuntimeError(f"Refusing to use symbolic link as virtual environment: {directory}")
    digest, requirements = _requirements_digest(requirements_path)
    env = _child_env(directory)
    marker = directory / MARKER_NAME
    expected = {
        "bootstrap_schema": BOOTSTRAP_SCHEMA,
        "requirements_sha256": digest,
        "base_python": str(Path(sys.executable).resolve()),
        "base_python_version": list(sys.version_info[:3]),
    }

    created = (directory / VENV_CREATED_NAME).is_file()
    if not created or not _venv_python_works(directory, executable, env):
        if skip_install:
            raise RuntimeError(
                f"Missing/incomplete isolated environment for {method_id}: {directory}. "
                f"Run python run_baseline.py {method_id} without --skip-install."
            )
        # --clear repairs the directory left behind by the previous ensurepip failure.
        _create_venv(directory, executable, env)

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
        pip_cmd = _pip_command(executable, env)
        print(f"[environment] Installing requirements into {directory}", flush=True)
        subprocess.run(pip_cmd + ["install", "-r", str(requirements_path)],
                       check=True, cwd=root, env=env)
        _smoke_test(executable, requirements, env)
        marker.write_text(json.dumps(expected, indent=2) + "\n", encoding="utf-8")
    else:
        print(f"[environment] Reusing prepared runtime: {directory}", flush=True)
        _smoke_test(executable, requirements, env)

    return str(executable), env
