"""Content identities for native implementations and installed runtimes."""

from __future__ import annotations

from functools import lru_cache
import importlib.metadata
import json
from pathlib import Path
import platform
import re
import subprocess

from cpr_person.cache import ROOT, canonical_hash, sha256_file


@lru_cache(maxsize=1)
def runtime_identity():
    packages = sorted(
        (d.metadata["Name"].lower().replace("_", "-"), d.version)
        for d in importlib.metadata.distributions()
        if d.metadata["Name"]
    )
    relevant = {}
    for name in (
        "torch",
        "torchvision",
        "transformers",
        "tokenizers",
        "huggingface-hub",
        "numpy",
        "Pillow",
        "scipy",
        "clip",
        "ftfy",
        "regex",
        "wcwidth",
    ):
        try:
            relevant[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return {
        "python": platform.python_version(),
        "packages": relevant,
        "environment_sha256": canonical_hash(packages),
    }


@lru_cache(maxsize=8)
def probe_runtime(python):
    command = "import json;from cpr_baselines.provenance import runtime_identity;print(json.dumps(runtime_identity()))"
    return json.loads(
        subprocess.check_output([str(python), "-c", command], cwd=ROOT, text=True)
    )


def execution_identity(source):
    files = {
        p
        for directory in (
            ROOT / "cpr_baselines",
            ROOT / "cpr_person",
            Path(source).resolve().parent,
        )
        for p in directory.iterdir()
        if p.is_file()
        and p.name != "config.yaml"
        and p.suffix in {".py", ".yaml", ".yml", ".txt"}
    }
    files.update(ROOT.glob("*.py"))
    signature = tuple(
        (str(p), p.stat().st_size, p.stat().st_mtime_ns, p.stat().st_ctime_ns)
        for p in sorted(files)
    )
    return {
        **runtime_identity(),
        "implementation": implementation_identity(str(ROOT), signature),
    }


def require_execution_identity(value):
    if (
        not isinstance(value, dict)
        or not isinstance(value.get("python"), str)
        or not value["python"]
        or not isinstance(value.get("packages"), dict)
        or not value["packages"]
        or not isinstance(value.get("environment_sha256"), str)
        or re.fullmatch(r"[0-9a-f]{64}", value["environment_sha256"]) is None
        or not isinstance(value.get("implementation"), dict)
        or not value["implementation"]
        or any(
            not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None
            for digest in value["implementation"].values()
        )
    ):
        raise RuntimeError(
            "Missing or invalid actual method runtime/implementation identity"
        )
    return value


@lru_cache(maxsize=128)
def implementation_identity(root, signature):
    return {
        str(Path(path).relative_to(root)): sha256_file(Path(path))
        for path, *_ in signature
    }


def module_state_sha256(module):
    import hashlib
    import torch

    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        if not isinstance(value, torch.Tensor):
            digest.update(canonical_hash({"name": name, "value": value}).encode())
            continue
        tensor = value.detach().cpu().contiguous()
        digest.update(
            json.dumps([name, str(tensor.dtype), list(tensor.shape)]).encode()
        )
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()
