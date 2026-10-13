"""Thin integration helpers; these never replace a retrieval model's scoring."""
from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass

from . import PROTOCOL_VERSION
from .cache import ROOT, resolve
from .detection import get_detections, protocol_config, detection_array_sha256
from .localization import get_localization

PERSON_METHODS = frozenset({"word4per_setmatch", "fafa_setmatch", "instruct_reid_setmatch",
                           "basic_setmatch", "adafocal_setmatch", "groundingdino_clipreid_set",
                           "groundingdino_clipreid_set_text", "reid_set_qwen_edit_proxy",
                           "per_person_clip_compose_setmatch", "structured_caption_retrieval"})


@dataclass
class PersonContext:
    detections: object
    localization: object
    gallery: list
    queries: list

    def metadata(self):
        cfg = protocol_config()
        from .cache import canonical_hash
        return {"version": PROTOCOL_VERSION, "detector": cfg["detector"],
                "selector": cfg["selector"], "image": cfg["image"],
                "detections_fingerprint": self.detections.fingerprint,
                "localization_fingerprint": self.localization.fingerprint,
                "detection_array_sha256": detection_array_sha256(self.detections),
                "localization_components_sha256": canonical_hash(self.localization.members),
                "detections_cache": str(self.detections.path),
                "localization_cache": str(self.localization.path),
                "detections_cache_hit": self.detections.cache_hit,
                "localization_cache_hit": self.localization.cache_hit,
                "localization_statistics": self.localization.stats,
                "adapted_inference": True, "gt_localization_used": False,
                "query_selection_uses_text": True,
                "setmatch": "maximum-sum min(K,N) real pairs; (sum+(K-min(K,N))*b)/K"}


def load_person_context(gallery, queries, gallery_manifest, query_manifest, device,
                        *, require_cache=False):
    if not gallery or not queries:
        raise ValueError("Gallery/query manifests must be non-empty")
    # Enforce the exact full manifests and their ordering even when inherited
    # methods pass rows to helpers directly.
    for path, rows, fields in [(gallery_manifest, gallery, ("image_id", "path")),
                               (query_manifest, queries, ("image_id", "subjects"))]:
        on_disk = [json.loads(line) for line in resolve(path).read_text().splitlines() if line.strip()]
        if len(on_disk) != len(rows) or any(any(a.get(f) != b.get(f) for f in fields)
                                          for a, b in zip(on_disk, rows)):
            raise ValueError("In-memory inputs differ from the complete ordered manifests")
    detections = get_detections(gallery, resolve(gallery_manifest), device, require_cache=require_cache)
    localization = get_localization(queries, resolve(query_manifest), gallery, detections, device)
    return PersonContext(detections, localization, gallery, queries)


def matching_metadata(cfg, *, score_space=None):
    value = cfg["setmatch"]["unmatched_score"]
    return {"assignment": "rectangular_hungarian_maximum_sum", "aggregation": "mean",
            "real_pair_count": "min(K,N)", "definition": "(sum(S[M])+(K-min(K,N))*b)/K",
            "unmatched_score": float(value),
            "unmatched_score_source": cfg["setmatch"]["unmatched_score_source"],
            "pairwise_metric": score_space or cfg["setmatch"].get("pairwise_metric"),
            "calibration_provenance": cfg["setmatch"].get("calibration_provenance"),
            "no_partial_credit": False, "allows_optional_unmatching": False}


def manifest_identity(metadata):
    return {"gallery_sha256": metadata["detections_fingerprint"]["gallery_manifest_sha256"],
            "queries_sha256": metadata["localization_fingerprint"]["queries_manifest_sha256"]}


def expand_targets(targets, localization):
    """One native instruction/composition input for each predicted member."""
    return [[targets[qi][si] for si, member in localization.flat(qi)]
            for qi in range(len(targets))]


def prepare_shared(force=False):
    command = [sys.executable, "-m", "cpr_person.prepare"]
    if force:
        command.append("--force")
    subprocess.run(command, cwd=ROOT, check=True)


def require_person_branch(run, gallery_manifest, query_manifest):
    from .cache import sha256_file
    protocol = run.get("shared_person_protocol", {})
    if (protocol.get("version") != PROTOCOL_VERSION
            or protocol.get("detections_fingerprint", {}).get("gallery_manifest_sha256") != sha256_file(gallery_manifest)
            or protocol.get("localization_fingerprint", {}).get("queries_manifest_sha256") != sha256_file(query_manifest)):
        raise RuntimeError("Inherited ReID scores have an old/different person protocol or manifest; rerun S5")
    return protocol


def require_manifest_branch(run, gallery_manifest, query_manifest):
    from .cache import sha256_file
    expected={"gallery_sha256":sha256_file(gallery_manifest),"queries_sha256":sha256_file(query_manifest)}
    if not isinstance(run,dict) or run.get("data_fingerprint")!=expected:
        raise RuntimeError("Source scores lack the same complete manifest identity; rerun their method through run_baseline.py")
