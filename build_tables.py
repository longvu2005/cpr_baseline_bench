#!/usr/bin/env python3

import csv
import json
from pathlib import Path

from benchmark_data import manifest_fingerprint
from prepare_data import CASE_TYPES
from cpr_benchmark.protocol import compatible, PERSON_METHODS, TEXT_PERSON_METHODS

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
    "Full-R@1",
    "Full-R@5",
    "Full-R@10",
]

CASE_COLUMNS = ["Method", "Protocol"] + [
    f"{case} {metric}" for case in CASE_TYPES for metric in ("mAP", "R@1")
]


def load_json(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def format_metric(value):
    if value is None:
        return ""

    return f"{100.0 * float(value):.2f}"


def collect_results():
    results = []
    person_detection_signature = None
    person_localization_signature = None
    image_signature = None

    if not OUTPUTS_DIR.is_dir():
        return results

    current_data = manifest_fingerprint(ROOT)

    for method_dir in sorted(OUTPUTS_DIR.iterdir()):
        if not method_dir.is_dir():
            continue

        metrics_path = method_dir / "metrics.json"
        run_path = method_dir / "run.json"

        if not metrics_path.is_file():
            continue

        metrics = load_json(metrics_path)
        if metrics.get("data_fingerprint") != current_data:
            print(f"Skipping previous-data result: {method_dir.name}; rerun this baseline.")
            continue

        if not run_path.is_file():
            raise RuntimeError(
                f"Missing evaluated run metadata next to {metrics_path.relative_to(ROOT)}: "
                f"expected {run_path.relative_to(ROOT)}"
            )
        run = load_json(run_path)
        if not compatible(run) or metrics.get("benchmark_protocol") != run.get("benchmark_protocol"):
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

        method_id = run_method
        images = run["benchmark_protocol"].get("image_fingerprint")
        if images:
            if image_signature is not None and images != image_signature:
                raise RuntimeError("Runs use different gallery image bytes; rerun them on the same image files before comparing.")
            image_signature = images
        if method_id in PERSON_METHODS:
            provenance = run["benchmark_protocol"]
            signature = (provenance["detections"], provenance["detection_array_sha256"])
            if person_detection_signature is not None and signature != person_detection_signature:
                raise RuntimeError("Person runs use different actual detector boxes. Rerun them with the same shared cache before comparing.")
            person_detection_signature = signature
            if method_id in TEXT_PERSON_METHODS:
                selection = (provenance.get("localization"), provenance.get("localization_components_sha256"))
                if not all(selection): raise RuntimeError(f"Missing shared membership provenance: {method_id}")
                if person_localization_signature is not None and selection != person_localization_signature:
                    raise RuntimeError("Text-person runs use different selected people. Rerun them with the same shared localization cache.")
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

        results.append(
            {
                "method_id": method_id,
                "method": method,
                "group": group,
                "supervision": supervision,
                "protocol": "Shared person v1" if method_id in PERSON_METHODS else "Native scene v1",
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
            for metric in ("mAP", "R@1"):
                row[f"{case} {metric}"] = format_metric(cases.get(case, {}).get(f"Full-{metric}"))
        rows.append(row)

    return rows


def write_csv(path, columns, rows):
    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
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
    TABLES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

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
        "Table 1 - Main CPR Retrieval Benchmark",
        MAIN_COLUMNS,
        main_rows,
    )

    print_table(
        "Table 2 - Case-wise Full CPR Retrieval",
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
