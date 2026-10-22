#!/usr/bin/env python3
"""Warm canonical grounding in the shared runtime before loading Qwen."""
from pathlib import Path
import argparse
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    try:
        from cpr_baselines.common import load_config
        from cpr_baselines.person_cache import require_shared_context
        require_shared_context(load_config(Path(__file__).with_name("config.yaml")), "cpu")
        print("[skip] identical verified shared grounding/localization cache is ready", flush=True)
        return
    except (RuntimeError, FileNotFoundError, ImportError):
        pass
    if args.check_only:
        raise SystemExit("Shared grounding cache is missing. Prepare it first or run without --skip-install.")
    from cpr_person.runtime import detector_python, detector_env
    python = detector_python()
    for module in ("cpr_person.prepare", "cpr_baselines.person_cache"):
        subprocess.run([python, "-m", module], cwd=ROOT, env=detector_env(python), check=True)



if __name__ == "__main__":
    main()
