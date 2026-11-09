"""One isolated runtime for shared scene CLIP and the heuristic person selector."""

import os
from pathlib import Path
import json

from cpr_person.cache import ROOT
from .provenance import probe_runtime


def prepare_runtime(*, skip_install=False):
    from baseline_environment import prepare_method_python

    python, env = prepare_method_python(
        root=ROOT,
        method_id="shared_clip",
        requirements_path=Path(__file__).with_name("scene-requirements.txt"),
        skip_install=skip_install,
    )
    os.environ["CPR_SCENE_CLIP_PYTHON"] = python
    os.environ["CPR_SELECTOR_RUNTIME"] = json.dumps(
        probe_runtime(python), sort_keys=True
    )
    return python, env


def scene_python():
    value = os.environ.get("CPR_SCENE_CLIP_PYTHON") or str(
        ROOT / ".baseline_venvs/shared_clip/bin/python"
    )
    if not Path(value).is_file():
        raise RuntimeError("Missing shared CLIP runtime; run through run_baseline.py")
    return str(Path(value).absolute())
