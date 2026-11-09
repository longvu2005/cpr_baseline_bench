#!/usr/bin/env python3
"""Check complete experiment artifacts; never infer native success from CPU contracts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def audit_artifacts(root=ROOT, methods=None):
    import numpy as np
    import build_tables
    from benchmark_data import (
        verify_manifest,
        split_paths,
        load_jsonl,
        gallery_image_digest,
    )
    from cpr_person.cache import canonical_hash, read_json, sha256_file
    from cpr_person.protocol import (
        RESULT_PROTOCOL,
        current_implementation,
        require_aligned_run,
    )
    from evaluate import validate_score_matrix, validate_run_order
    from run_baseline import discover_methods

    methods = discover_methods() if methods is None else methods
    info = verify_manifest(root, "test")
    val = verify_manifest(root, "val")["fingerprint"]
    gm, qm = split_paths(root, "test")
    gallery, queries = load_jsonl(gm), load_jsonl(qm)
    images = gallery_image_digest(root, gallery)
    results = []
    batch = read_json(root / "runs/batch_status.json")
    for method in methods:
        name = method.method_id
        failures = []
        directory = root / "outputs" / name
        run, metrics = read_json(directory / "run.json"), read_json(
            directory / "metrics.json"
        )
        try:
            if (
                isinstance(batch, dict)
                and batch.get(name, {}).get("state") != "completed"
            ):
                raise ValueError("latest all-method batch did not complete this method")
            if not isinstance(run, dict) or not isinstance(metrics, dict):
                raise ValueError("missing completed inference/evaluation artifacts")
            if run.get(
                "result_protocol"
            ) != RESULT_PROTOCOL or not current_implementation(run, root):
                raise ValueError("old pipeline or implementation")
            if run.get("method") != name or metrics.get("method") != name:
                raise ValueError("method identity mismatch")
            if (
                run.get("data_fingerprint") != info["fingerprint"]
                or metrics.get("data_fingerprint") != info["fingerprint"]
            ):
                raise ValueError("wrong final test manifests")
            receipt = run.get("execution_receipt", {})
            if (
                receipt.get("schema") != "native-method-run-v1"
                or receipt.get("method") != name
                or receipt.get("process_succeeded") is not True
                or receipt.get("complete_canonical_test_gallery") is not True
                or receipt.get("synthetic_model_substitution") is not False
            ):
                raise ValueError("missing full native execution receipt")
            if metrics.get("run_metadata_sha256") != canonical_hash(run):
                raise ValueError("metrics/run receipt differ")
            if (
                run.get("cpr_supervision") == "Val only"
                and run.get("validation_fingerprint") != val
            ):
                raise ValueError("missing current val-only selection")
            scores = root / "runs" / name / "scores.npy"
            if (
                run.get("scores_sha256") != sha256_file(scores)
                or metrics.get("scores_sha256") != run["scores_sha256"]
            ):
                raise ValueError("score artifact checksum differs")
            validate_score_matrix(
                np.load(scores, mmap_mode="r", allow_pickle=False),
                (len(queries), len(gallery)),
            )
            validate_run_order(run, gallery, queries)
            recorded_images = (
                run.get("shared_person_protocol", {})
                .get("detections_fingerprint", {})
                .get(
                    "gallery_image_content_sha256",
                    run.get("image_provenance", {}).get("gallery_image_content_sha256"),
                )
            )
            if recorded_images != images:
                raise ValueError("wrong current gallery image bytes")
            if receipt.get("entrypoint_sha256") != sha256_file(method.run_path):
                raise ValueError("different native entrypoint")
            from cpr_baselines.provenance import require_execution_identity

            require_execution_identity(run.get("execution"))
            if receipt.get("environment") != run["execution"]:
                raise ValueError(
                    "native execution receipt and actual method runtime differ"
                )
            require_aligned_run(run, name)
            from cpr_baselines.native_assets import require_current_assets

            require_current_assets(run, root)
            build_tables.validate_evaluated_result(run, metrics, info)
            if name == "instruct_reid_setmatch":
                match = run.get("setmatch", {})
                if match.get(
                    "pairwise_metric"
                ) != "negative_squared_euclidean" or not match.get(
                    "calibration_provenance"
                ):
                    raise ValueError(
                        "unmatched native-distance penalty has no validated provenance"
                    )
                proof = match["calibration_provenance"]
                if proof.get("artifact_sha256") != sha256_file(
                    root / Path(proof["artifact"])
                ):
                    raise ValueError(
                        "native penalty selection/calibration artifact changed"
                    )
            if name == "word4per_setmatch":
                proof = run.get("reproduction_provenance")
                if not proof or proof.get("artifact_sha256") != sha256_file(
                    root / Path(proof["artifact"])
                ):
                    raise ValueError("missing/changed Stage-2 reproduction receipt")
            if name == "wiser":
                states = run.get("native_encoder_states", {})
                if not all(
                    isinstance(states.get(key), str) and len(states[key]) == 64
                    for key in ("clip", "captioner")
                ):
                    raise ValueError("missing native OpenCLIP/BLIP2 weight identities")
                trace = run.get("refiner_api_trace")
                if (
                    not isinstance(trace, dict)
                    or trace.get("schema") != "wiser-native-api-trace-v1"
                ):
                    raise ValueError("missing native API trace")
                if trace["num_calls"] and (
                    len(trace["resolved_models"]) != 1
                    or len(trace["requested_models"]) != 1
                ):
                    raise ValueError("unresolved or mixed native API models")
                for artifact in trace["files"]:
                    if sha256_file(root / artifact["path"]) != artifact["sha256"]:
                        raise ValueError("native API response artifact changed")
            for artifact in run.get("selection_artifacts", []):
                if sha256_file(root / artifact["path"]) != artifact["sha256"]:
                    raise ValueError("validation selection receipt changed")
        except (ValueError, RuntimeError, KeyError, OSError, TypeError) as error:
            failures.append(str(error))
        results.append(
            {
                "method": name,
                "artifact_checks_passed": not failures,
                "blockers": failures,
            }
        )
    return {
        "schema": "publication-artifacts-v1",
        "all_methods_complete": len(methods) == 29
        and all(x["artifact_checks_passed"] for x in results),
        "num_methods": len(methods),
        "results": results,
        "scope": "validates recorded full-run artifacts; independent native-output fidelity and external pretraining overlap still require evidence",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "runs/publication_readiness.json"
    )
    args = parser.parse_args()
    try:
        result = audit_artifacts()
    except (ValueError, RuntimeError, OSError) as error:
        result = {
            "schema": "publication-artifacts-v1",
            "all_methods_complete": False,
            "global_blockers": [str(error)],
            "results": [],
        }
    from cpr_person.cache import atomic_json

    atomic_json(args.output, result)
    print(json.dumps(result, indent=2))
    if not result["all_methods_complete"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
