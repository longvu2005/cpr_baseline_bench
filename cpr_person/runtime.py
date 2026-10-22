"""Explicit shared detector interpreter; never inherit a retrieval runtime."""
from __future__ import annotations

import argparse
import os
import subprocess
from importlib.metadata import version
from pathlib import Path

from .cache import ROOT, resolve


def detector_python():
    default = ROOT / ".baseline_venvs/shared_person_detector" / (
        "Scripts/python.exe" if os.name == "nt" else "bin/python")
    # Keep the venv executable path: resolving its symlink loses the environment.
    path = Path(os.path.abspath(os.environ.get("CPR_DETECTOR_PYTHON", str(default))))
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing detector Python: {path}. Run python -m cpr_person.runtime, "
            "or set CPR_DETECTOR_PYTHON to a prepared detector interpreter.")
    return str(path)


def detector_env(python):
    env = os.environ.copy()
    directory = Path(python).parent.parent
    env.update(CPR_DETECTOR_PYTHON=str(python), VIRTUAL_ENV=str(directory),
               PYTHONNOUSERSITE="1", PYTHONUNBUFFERED="1", USE_TF="0", USE_FLAX="0",
               TOKENIZERS_PARALLELISM="false")
    env["PATH"] = str(Path(python).parent) + os.pathsep + env.get("PATH", "")
    for name in ("PYTHONPATH", "PYTHONHOME", "HF_HOME", "HF_HUB_CACHE", "TRANSFORMERS_CACHE",
                 "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        env.pop(name, None)
    return env


def check_runtime(detector):
    actual = {name: version(name) for name in detector["runtime_versions"]}
    mismatches = {name: actual[name] for name, expected in detector["runtime_versions"].items()
                  if actual[name].split("+")[0] != expected}
    if mismatches:
        raise RuntimeError(f"Incompatible shared detector runtime: {mismatches}; "
                           "install cpr_person/detector-requirements.txt in CPR_DETECTOR_PYTHON")
    return actual


def prepare_runtime(*, skip_install=False):
    if "CPR_DETECTOR_PYTHON" in os.environ:
        python = detector_python()
    else:
        from baseline_environment import prepare_method_python
        python, _ = prepare_method_python(root=ROOT, method_id="shared_person_detector",
            requirements_path=resolve("cpr_person/detector-requirements.txt"), skip_install=skip_install)
    subprocess.run([python, "-c", "from cpr_person.detection import protocol_config; "
                    "from cpr_person.runtime import check_runtime; "
                    "check_runtime(protocol_config()['detector']); "
                    "from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection"],
                   cwd=ROOT, env=detector_env(python), check=True)
    return python


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-install", action="store_true")
    args = parser.parse_args()
    print(prepare_runtime(skip_install=args.skip_install))


if __name__ == "__main__":
    main()
