#!/usr/bin/env python3

import argparse
import csv
import json
import math
import re
from pathlib import Path

from benchmark_data import CASE_TYPES, DATA_PROTOCOL, verify_manifest
from cpr_person import PROTOCOL_VERSION
from cpr_person.protocol import (
    PERSON_METHODS,
    RCR_CLIP_SHA256,
    RCR_SCENE_CLIP_METHODS,
    TEXTFREE_DIAGNOSTIC_METHOD,
    rcr_detection_source,
)
from cpr_person.cache import atomic_file, canonical_hash

ROOT = Path(__file__).resolve().parent
OUTPUTS_DIR = ROOT / "outputs"
TABLES_DIR = ROOT / "tables"


MAIN_COLUMNS = [
    "Method",
    "Protocol",
    "CPR Supervision",
    "ID-mAP",
    "ID-R@1",
    "ID-R@5",
    "ID-R@10",
    "Full-mAP",
    "Macro-Full-mAP",
    "Full-R@1",
    "Full-R@5",
    "Full-R@10",
]

CASE_METRICS = ("ID-mAP", "ID-R@1", "Full-mAP", "Full-R@1")
CASE_COLUMNS = ["Method", "Protocol"] + [
    f"{case} {metric}" for case in CASE_TYPES for metric in CASE_METRICS
]


def load_json(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def format_metric(value):
    if value is None:
        return ""

    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"Retrieval metric must be finite and in [0, 1], got {value!r}")
    return f"{100.0 * value:.2f}"


def validate_evaluated_result(run, metrics, split_info):
    method = run["method"]
    if run.get("higher_is_better") is not True:
        raise ValueError(f"Expected descending-score run metadata: {method}")
    if metrics["run_metadata_sha256"] != canonical_hash(run):
        raise ValueError(f"Metrics and run metadata were not evaluated together: {method}")
    digest = metrics.get("scores_sha256")
    if (not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest)
            or digest != run.get("scores_sha256")):
        raise ValueError(f"Metrics and run metadata refer to different score bytes: {method}")
    count = split_info["num_queries"]
    if metrics.get("num_queries") != count or run.get("num_queries", count) != count:
        raise ValueError(f"Evaluated query count differs from the complete test split: {method}")
    if run.get("num_gallery", split_info["num_gallery"]) != split_info["num_gallery"]:
        raise ValueError(f"Evaluated gallery count differs from the complete test split: {method}")
    overall = metrics.get("overall", {})
    for key in MAIN_COLUMNS[3:]:
        if key not in overall:
            raise ValueError(f"Missing overall metric {key}: {method}")
        format_metric(overall[key])
    cases = metrics.get("cases", {})
    if not cases or not set(cases) <= set(CASE_TYPES):
        raise ValueError(f"Missing or unknown cases: {method}")
    if sum(case.get("num_queries", 0) for case in cases.values()) != count:
        raise ValueError(f"Case counts differ from the complete test split: {method}")
    for case, values in cases.items():
        expected = split_info.get("case_counts", {}).get(case)
        if values.get("num_queries", 0) < 1 or (expected is not None and values["num_queries"] != expected):
            raise ValueError(f"Invalid {case} query count: {method}")
        for key in CASE_METRICS:
            if key not in values:
                raise ValueError(f"Missing {case} metric {key}: {method}")
            format_metric(values[key])
    macro = sum(case["Full-mAP"] for case in cases.values()) / len(cases)
    weighted = sum(case["Full-mAP"] * case["num_queries"] for case in cases.values()) / count
    if (not math.isclose(macro, overall["Macro-Full-mAP"], abs_tol=1e-12)
            or not math.isclose(weighted, overall["Full-mAP"], abs_tol=1e-12)):
        raise ValueError(f"Overall Full-mAP disagrees with case metrics: {method}")


def validate_person_provenance(provenance, current_data, method_id):
    detection = provenance.get("detections_fingerprint")
    localization = provenance.get("localization_fingerprint")
    if (not isinstance(detection, dict) or not detection
            or not isinstance(localization, dict) or not localization):
        raise RuntimeError(f"Missing shared detector/localization fingerprints: {method_id}")
    if (detection.get("adapter_version") != PROTOCOL_VERSION
            or localization.get("adapter_version") != PROTOCOL_VERSION):
        raise RuntimeError(f"Shared fingerprint version mismatch: {method_id}")
    if (detection.get("gallery_manifest_sha256") != current_data["gallery_sha256"]
            or localization.get("queries_manifest_sha256") != current_data["queries_sha256"]):
        raise RuntimeError(f"Shared adapter provenance does not match the evaluated manifests: {method_id}")
    for value, label in [
        (detection.get("gallery_image_content_sha256"), "gallery image bytes checksum"),
        (provenance.get("detection_array_sha256"), "detector boxes checksum"),
        (provenance.get("localization_components_sha256"), "selected membership checksum"),
    ]:
        if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
            raise RuntimeError(f"Missing/invalid {label}: {method_id}")
    if (localization.get("detections") != detection
            or localization.get("detection_array_sha256") != provenance["detection_array_sha256"]):
        raise RuntimeError(f"Shared localization refers to different detector boxes: {method_id}")
    if not isinstance(localization.get("selector"), dict) or not localization["selector"]:
        raise RuntimeError(f"Missing shared selector fingerprint: {method_id}")


def collect_results():
    results = []
    person_detection_signature = None
    person_localization_signature = None
    image_signature = None

    if not OUTPUTS_DIR.is_dir():
        return results

    split_info = verify_manifest(ROOT, "test")
    current_data = split_info["fingerprint"]

    for method_dir in sorted(OUTPUTS_DIR.iterdir()):
        if not method_dir.is_dir():
            continue
        if method_dir.name == TEXTFREE_DIAGNOSTIC_METHOD:
            print("Skipping separate text-free diagnostic from the primary comparison.")
            continue

        metrics_path = method_dir / "metrics.json"
        run_path = method_dir / "run.json"

        if not metrics_path.is_file():
            continue

        metrics = load_json(metrics_path)
        if (
            metrics.get("split") != "test"
            or metrics.get("data_protocol") != DATA_PROTOCOL
            or metrics.get("data_fingerprint") != current_data
        ):
            print(f"Skipping previous-data result: {method_dir.name}; rerun this baseline.")
            continue

        if not run_path.is_file():
            raise RuntimeError(
                f"Missing evaluated run metadata next to {metrics_path.relative_to(ROOT)}: "
                f"expected {run_path.relative_to(ROOT)}"
            )
        run = load_json(run_path)
        provenance = run.get("shared_person_protocol", {})
        if (method_dir.name in RCR_SCENE_CLIP_METHODS
                and run.get("scene_clip_sha256") != RCR_CLIP_SHA256):
            print(f"Skipping previous CLIP result: {method_dir.name}; rerun with RCR ViT-B/32.")
            continue
        if method_dir.name in PERSON_METHODS and (
                not isinstance(provenance, dict) or provenance.get("version") != PROTOCOL_VERSION):
            print(f"Skipping legacy/different-protocol result: {method_dir.name}; rerun this baseline.")
            continue
        metric_method = metrics.get("method")
        run_method = run.get("method")
        if not isinstance(run_method, str) or not run_method:
            raise ValueError(f"Missing method id in {run_path.relative_to(ROOT)}")
        if metric_method != run_method:
            raise ValueError(
                f"Method mismatch for {method_dir.name}: metrics={metric_method!r}, "
                f"run={run_method!r}"
            )
        if run_method != method_dir.name:
            raise ValueError(
                f"Output directory/method mismatch: {method_dir.name!r} vs {run_method!r}"
            )
        if run.get("data_fingerprint") != metrics["data_fingerprint"]:
            raise RuntimeError(f"Run metadata does not match the evaluated manifests: {run_method}")
        if "run_metadata_sha256" not in metrics:
            print(f"Skipping unbound metrics: {run_method}; rerun evaluate.py --method {run_method}.")
            continue
        validate_evaluated_result(run, metrics, split_info)

        if run.get("split") != "test" or run.get("data_protocol") != DATA_PROTOCOL:
            raise RuntimeError(f"Run is outside the RCR test protocol: {run_method}")
        validation = run.get("validation_fingerprint")
        if validation is not None or run.get("cpr_supervision") == "Val only":
            if validation != verify_manifest(ROOT, "val")["fingerprint"]:
                print(f"Skipping previous-validation result: {run_method}; rerun selection.")
                continue

        method_id = run_method
        if method_id in PERSON_METHODS:
            validate_person_provenance(provenance, current_data, method_id)
            if rcr_detection_source(provenance) is None:
                raise RuntimeError(f"Missing or invalid RCR detection provenance: {method_id}")
        images = (provenance["detections_fingerprint"]["gallery_image_content_sha256"]
                  if method_id in PERSON_METHODS
                  else run.get("image_provenance", {}).get("gallery_image_content_sha256"))
        if not isinstance(images, str) or not re.fullmatch(r"[0-9a-f]{64}", images):
            raise RuntimeError(f"Missing/invalid gallery image bytes checksum: {method_id}; rerun inference.")
        if image_signature is not None and images != image_signature:
            raise RuntimeError("Runs use different gallery image bytes; rerun them on the same image files before comparing.")
        image_signature = images
        if method_id in PERSON_METHODS:
            signature = (canonical_hash(provenance["detections_fingerprint"]), provenance["detection_array_sha256"])
            if person_detection_signature is not None and signature != person_detection_signature:
                raise RuntimeError("Person runs use different actual detector boxes. Rerun them with the same shared cache before comparing.")
            person_detection_signature = signature
            selection = (canonical_hash(provenance["localization_fingerprint"]),
                         provenance["localization_components_sha256"])
            if person_localization_signature is not None and selection != person_localization_signature:
                raise RuntimeError("Person runs use different selected people. Rerun them with the same shared localization cache.")
            person_localization_signature = selection

        method = (
            run.get("display_name")
            or metrics.get("display_name")
            or method_id
        )

        group = (
            run.get("group")
            or metrics.get("group")
            or "Other"
        )

        supervision = (
            run.get("cpr_supervision")
            or metrics.get("cpr_supervision")
            or ""
        )
        protocol_label = "Shared RCR boxes" if method_id in PERSON_METHODS else "Native scene v1"
        if method_id == "instruct_reid_setmatch" and not run.get("setmatch", {}).get("calibration_provenance"):
            protocol_label += "; uncalibrated unmatched"

        results.append(
            {
                "method_id": method_id,
                "method": method,
                "group": group,
                "supervision": supervision,
                "protocol": protocol_label,
                "metrics": metrics,
            }
        )

    return results


def build_main_table(results):
    rows = []

    for result in results:
        overall = result["metrics"].get("overall", {})

        rows.append(
            {
                "Method": result["method"],
                "Protocol": result["protocol"],
                "CPR Supervision": result["supervision"],
                "ID-mAP": format_metric(
                    overall.get("ID-mAP")
                ),
                "ID-R@1": format_metric(
                    overall.get("ID-R@1")
                ),
                "ID-R@5": format_metric(
                    overall.get("ID-R@5")
                ),
                "ID-R@10": format_metric(
                    overall.get("ID-R@10")
                ),
                "Full-mAP": format_metric(
                    overall.get("Full-mAP")
                ),
                "Macro-Full-mAP": format_metric(overall.get("Macro-Full-mAP")),
                "Full-R@1": format_metric(
                    overall.get("Full-R@1")
                ),
                "Full-R@5": format_metric(
                    overall.get("Full-R@5")
                ),
                "Full-R@10": format_metric(
                    overall.get("Full-R@10")
                ),
                "_group": result["group"],
            }
        )

    return rows


def build_case_table(results):
    rows = []

    for result in results:
        cases = result["metrics"].get("cases", {})

        row = {"Method": result["method"], "Protocol": result["protocol"], "_group": result["group"]}
        for case in CASE_TYPES:
            for metric in CASE_METRICS:
                row[f"{case} {metric}"] = format_metric(cases.get(case, {}).get(metric))
        rows.append(row)

    return rows


def write_csv(path, columns, rows):
    with atomic_file(path, "w") as f:
        f.reconfigure(encoding="utf-8", newline="")
        writer = csv.DictWriter(
            f,
            fieldnames=columns,
            extrasaction="ignore",
        )

        writer.writeheader()
        writer.writerows(rows)


def print_table(title, columns, rows):
    print()
    print(title)
    print("=" * len(title))

    if not rows:
        print("No results.")
        return

    widths = {
        column: len(column)
        for column in columns
    }

    for row in rows:
        for column in columns:
            widths[column] = max(
                widths[column],
                len(str(row.get(column, ""))),
            )

    header = " | ".join(
        column.ljust(widths[column])
        for column in columns
    )

    separator = "-+-".join(
        "-" * widths[column]
        for column in columns
    )

    print(header)
    print(separator)

    previous_group = None

    for row in rows:
        group = row.get("_group")

        if (
            previous_group is not None
            and group != previous_group
        ):
            print(separator)

        print(
            " | ".join(
                str(row.get(column, "")).ljust(
                    widths[column]
                )
                for column in columns
            )
        )

        previous_group = group


def main():
    parser = argparse.ArgumentParser(description="Build tables from evaluated benchmark results.")
    parser.add_argument("--check", action="store_true", help="Check reporting imports without reading manifests or results.")
    if parser.parse_args().check:
        print("Table builder imports OK")
        return

    results = collect_results()

    if not results:
        raise RuntimeError(
            "No evaluated results match the current manifests. Run a baseline first."
        )

    group_order = {
        "Simple / Obvious Baselines": 0,
        "Published / SOTA Baselines": 1,
        # Backward-compatible alias for older run.json files.
        "Published Baselines": 1,
        "Proposed": 2,
    }

    method_order = {
        "clip_image": 1,
        "clip_text": 2,
        "clip_early_fusion": 3,
        "clip_late_fusion": 4,
        "word4per_setmatch": 1,
        "fafa_setmatch": 2,
        "instruct_reid_setmatch": 3,
        "pic2word": 4,
        "lincir": 5,
        "magiclens_large": 6,
        "bge_vl_mllm_s1": 7,
    }

    results.sort(
        key=lambda x: (
            group_order.get(
                x["group"],
                99,
            ),
            method_order.get(
                x["method_id"],
                999,
            ),
            x["method"].lower(),
        )
    )

    main_rows = build_main_table(results)
    case_rows = build_case_table(results)

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    main_path = TABLES_DIR / "table1_main.csv"
    case_path = TABLES_DIR / "table2_cases.csv"

    write_csv(
        main_path,
        MAIN_COLUMNS,
        main_rows,
    )

    write_csv(
        case_path,
        CASE_COLUMNS,
        case_rows,
    )

    print_table(
        "Table 1 - RCR Test Retrieval",
        MAIN_COLUMNS,
        main_rows,
    )

    print_table(
        "Table 2 - RCR Test Retrieval by Case",
        CASE_COLUMNS,
        case_rows,
    )

    print()
    print("Saved")
    print("-----")
    print(main_path.relative_to(ROOT))
    print(case_path.relative_to(ROOT))
    print()
    print(f"Methods: {len(results)}")


if __name__ == "__main__":
    main()
