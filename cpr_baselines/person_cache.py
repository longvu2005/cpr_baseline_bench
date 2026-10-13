"""Build shared grounding once in its original runtime; captioners only read it."""
from __future__ import annotations

import argparse
import os
from importlib.metadata import version

from cpr_person.adapter import load_person_context
from cpr_person.cache import ROOT, resolve
from .common import load_jsonl, seed_runtime


def require_shared_context(cfg, device, loader=load_person_context):
    gm, qm = resolve(cfg["data"]["gallery_manifest"]), resolve(cfg["data"]["query_manifest"])
    previous = os.environ.get("CPR_REQUIRE_SHARED_CACHE")
    os.environ["CPR_REQUIRE_SHARED_CACHE"] = "1"
    try:
        return loader(load_jsonl(gm), load_jsonl(qm), gm, qm, device, require_cache=True)
    except (RuntimeError, FileNotFoundError) as error:
        raise RuntimeError(
            "Structured Caption Retrieval requires the same verified grounding/localization cache as existing person methods. "
            "Prepare it once with the original detector requirements: "
            "python -m pip install -r cpr_person/requirements.txt; "
            "python -m cpr_person.prepare; python -m cpr_baselines.person_cache. "
            "Then run this method through run_baseline.py. " + str(error)) from error
    finally:
        if previous is None:
            os.environ.pop("CPR_REQUIRE_SHARED_CACHE", None)
        else:
            os.environ["CPR_REQUIRE_SHARED_CACHE"] = previous


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if version("transformers") != "4.39.3":
        raise SystemExit("Build shared grounding with cpr_person/requirements.txt (Transformers 4.39.3), then install the captioner requirements.")
    device = seed_runtime({"seed": 0, "device": args.device})
    gm, qm = ROOT / "data/gallery.jsonl", ROOT / "data/queries.jsonl"
    context = load_person_context(load_jsonl(gm), load_jsonl(qm), gm, qm, str(device))
    print("Prepared canonical shared grounding and memberships:", context.metadata())


if __name__ == "__main__":
    main()
