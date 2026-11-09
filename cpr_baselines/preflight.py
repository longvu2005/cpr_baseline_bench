"""Check declared execution prerequisites without installing or fabricating models."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess


def gpu_memory():
    executable = shutil.which("nvidia-smi")
    if not executable:
        return []
    try:
        output = subprocess.check_output(
            [executable, "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            text=True,
        )
        return [int(value.strip()) for value in output.splitlines() if value.strip()]
    except (subprocess.CalledProcessError, ValueError):
        return []


def method_readiness(cfg, root, *, gpu_mib=None, free_bytes=None):
    gpu_mib = gpu_memory() if gpu_mib is None else gpu_mib
    free_bytes = shutil.disk_usage(root).free if free_bytes is None else free_bytes
    issues, notes = [], []
    runtime = cfg.get("runtime", {})
    device = str(runtime.get("device", "auto"))
    if device.startswith("cuda") and not gpu_mib:
        issues.append("configured CUDA device is unavailable")
    if cfg["method"] == "word4per_setmatch":
        weights = (
            os.environ.get("WORD4PER_STAGE2_CHECKPOINT") or cfg["checkpoint"]["stage2"]
        )
        config = (
            os.environ.get("WORD4PER_STAGE2_CONFIG")
            or cfg["checkpoint"]["stage2_config"]
        )

        def present(value):
            path = Path(value)
            return (path if path.is_absolute() else root / path).is_file()

        if present(weights) and present(config):
            receipt = (
                os.environ.get("WORD4PER_STAGE2_PROVENANCE")
                or cfg["checkpoint"].get("reproduction_provenance")
                or str(
                    Path(cfg["checkpoint"]["stage2"]).parent
                    / "word4per_cuhk_pedes_stage2_reproduction.json"
                )
            )
            if not present(receipt):
                issues.append(
                    "missing Stage-2 reproduction/external-data provenance receipt"
                )
        else:
            if not (
                os.environ.get("WORD4PER_STAGE1_CHECKPOINT")
                and os.environ.get("WORD4PER_REPRO_DATA_ROOT")
            ):
                issues.append(
                    "need reproduced Stage-2 weights/config/receipt or Stage-1 plus external CUHK-PEDES/ITCPR inputs"
                )
            else:
                notes.append(
                    "Stage-2 must be reproduced with the original external selection data; no RCR val/test overlap permitted"
                )
    if cfg["method"] == "wiser":
        key_name = cfg["models"]["refiner"]["api_key_env"]
        if not os.environ.get(key_name):
            issues.append(f"missing configured refiner credential: {key_name}")
        required = float(cfg["isolated_env"]["minimum_free_disk_gib"])
        if free_bytes < required * 1024**3:
            issues.append(
                f"declared preparation disk floor is {required:g} GiB; available {free_bytes/1024**3:.1f} GiB"
            )
        notes.append(
            "BAGEL max_mem_per_gpu is a configured budget, not proof of native inference fitting this hardware"
        )
    return {
        "method": cfg["method"],
        "prerequisites_passed": not issues,
        "blockers": issues,
        "notes": notes,
        "gpu_memory_mib": gpu_mib,
        "free_disk_bytes": free_bytes,
        "native_inference_verified": False,
    }


def require_readiness(cfg, root):
    result = method_readiness(cfg, root)
    if result["blockers"]:
        raise RuntimeError(cfg["method"] + ": " + "; ".join(result["blockers"]))
    return result
