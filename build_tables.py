#!/usr/bin/env python3

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUTPUTS_DIR = ROOT / "outputs"
TABLES_DIR = ROOT / "tables"


MAIN_COLUMNS = [
    "Method",
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

CASE_COLUMNS = [
    "Method",
    "SINGLE mAP",
    "SINGLE R@1",
    "MULTI mAP",
    "MULTI R@1",
    "RELATIONAL mAP",
    "RELATIONAL R@1",
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

    if not OUTPUTS_DIR.is_dir():
        return results

    for method_dir in sorted(OUTPUTS_DIR.iterdir()):
        if not method_dir.is_dir():
            continue

        metrics_path = method_dir / "metrics.json"
        run_path = method_dir / "run.json"

        if not metrics_path.is_file():
            continue

        metrics = load_json(metrics_path)

        if run_path.is_file():
            run = load_json(run_path)
        else:
            run = {}

        method_id = (
            run.get("method")
            or metrics.get("method")
            or method_dir.name
        )

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

        single = cases.get("SINGLE", {})
        multi = cases.get("MULTI", {})
        relational = cases.get("RELATIONAL", {})

        rows.append(
            {
                "Method": result["method"],
                "SINGLE mAP": format_metric(
                    single.get("Full-mAP")
                ),
                "SINGLE R@1": format_metric(
                    single.get("Full-R@1")
                ),
                "MULTI mAP": format_metric(
                    multi.get("Full-mAP")
                ),
                "MULTI R@1": format_metric(
                    multi.get("Full-R@1")
                ),
                "RELATIONAL mAP": format_metric(
                    relational.get("Full-mAP")
                ),
                "RELATIONAL R@1": format_metric(
                    relational.get("Full-R@1")
                ),
                "_group": result["group"],
            }
        )

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
            "No outputs/*/metrics.json found."
        )

    group_order = {
        "Simple / Obvious Baselines": 0,
        "Published / SOTA Baselines": 1,
        "Proposed": 2,
    }

    method_order = {
        "clip_image": 1,
        "clip_text": 2,
        "clip_early_fusion": 3,
        "clip_late_fusion": 4,
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