#!/usr/bin/env python3
"""Dependency-free verifier that the strict P10 v5 replacement is actually installed."""
from __future__ import annotations

import py_compile
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent


def requirement_names() -> list[str]:
    names: list[str] = []
    for raw in (HERE / "requirements.txt").read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        match = re.match(r"([A-Za-z0-9_.-]+)", line)
        if not match:
            raise SystemExit(f"BAD P10 INSTALL: cannot parse host requirement {line!r}")
        names.append(match.group(1).lower().replace("_", "-"))
    return names


names = requirement_names()
if set(names) != {"pyyaml", "uv"} or len(names) != 2:
    raise SystemExit(
        "BAD P10 INSTALL: host requirements must be exactly PyYAML + uv; "
        f"found {names}"
    )

config = (HERE / "config.yaml").read_text(encoding="utf-8")
required_config_fragments = (
    "display_name: Imagine and Seek (LDRE-L + IP-CIR, mounted-assets streaming CPR adapter)",
    "implementation_status: OFFICIAL_SOURCE_ADAPTED",
    "minimum_local_free_disk_gib: 15.0",
    "full_download_minimum_free_disk_gib: 72.0",
    "require_external_large_assets_on_kaggle: true",
    "captioner_env: IPCIR_BLIP2_DIR",
    "layout_llm_env: IPCIR_QWEN32_DIR",
    "repo_id: Salesforce/blip2-opt-6.7b-coco",
    "captions_per_query: 15",
    "repo_id: Qwen/Qwen1.5-32B-Chat-GPTQ-Int4",
    "clip_name: ViT-L/14",
    "count_per_query: 5",
    "persist_proxy_images: false",
    "proxy_features_state: runs/imagine_seek/cache/proxy_features.state.json",
    "lambda_text: 0.3",
    "source: paper_CIRCO_fixed_no_CPR_tuning",
    "small_model_fallback: false",
)
missing = [fragment for fragment in required_config_fragments if fragment not in config]
if missing:
    raise SystemExit("BAD P10 INSTALL: config mismatch; missing " + repr(missing))

for name in (
    "download_checkpoint.py",
    "prepare_proxies.py",
    "official_proxy_worker.py",
    "run.py",
):
    py_compile.compile(str(HERE / name), doraise=True)

print("P10 STRICT V5 INSTALL OK")
print("host phase2 requirements: PyYAML + uv only")
print("paper branch: LDRE-L + IP-CIR")
print("large assets: exact BLIP2-OPT6.7B + Qwen1.5-32B-GPTQ mounted read-only on Kaggle")
print("proxies: MIGC+ELITE generate -> CLIP-L encode -> discard; 5/query")
