"""Small isolated selector dependencies, without changing native model packages."""
from __future__ import annotations

import importlib.metadata
import subprocess
import sys

from .cache import atomic_json, read_json, resolve, sha256_file


def inventory(root):
    return [{"path":str(p.relative_to(root)),"sha256":sha256_file(p)}
            for p in sorted(root.rglob("*")) if p.is_file()
            and "__pycache__" not in p.parts and p.name!="ready.json"]


def validate_tokenizer(cfg):
    root=resolve(cfg["tokenizer_runtime_dir"])
    marker=read_json(root/"ready.json")
    if not isinstance(marker,dict) or marker.get("packages")!=cfg["tokenizer_packages"]:
        raise RuntimeError("Missing/stale isolated selector tokenizer; run python -m cpr_person.prepare")
    items=marker.get("files")
    if not items or any(not (root/item["path"]).is_file()
                        or sha256_file(root/item["path"])!=item["sha256"] for item in items):
        raise RuntimeError("Isolated selector tokenizer checksum mismatch")
    distributions={d.metadata["Name"].lower():d.version for d in importlib.metadata.distributions(path=[str(root)])}
    if any(distributions.get(name)!=version for name,version in cfg["tokenizer_packages"].items()):
        raise RuntimeError("Isolated selector tokenizer package version mismatch")
    return root


def prepare_tokenizer(cfg,force=False):
    if not force:
        try:
            return validate_tokenizer(cfg)
        except RuntimeError:
            pass
    root=resolve(cfg["tokenizer_runtime_dir"])
    root.mkdir(parents=True,exist_ok=True)
    # Both packages are pure Python. This is a private target directory, not a
    # second notebook environment or an upgrade/downgrade of native dependencies.
    subprocess.run([sys.executable,"-m","pip","install","--no-deps","--upgrade",
                    "--target",str(root),*[f"{name}=={version}" for name,version in cfg["tokenizer_packages"].items()]],check=True)
    atomic_json(root/"ready.json",{"packages":cfg["tokenizer_packages"],"files":inventory(root)})
    return validate_tokenizer(cfg)
