#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
GALLERY_PATH = ROOT / "data/gallery.jsonl"
QUERIES_PATH = ROOT / "data/queries.jsonl"

EXPECTED_GALLERY = 17000
EXPECTED_QUERIES = 2975
EXPECTED_CASES = {
    "SINGLE": 2671,
    "MULTI": 225,
    "RELATIONAL": 79,
}


def fail(message: str) -> None:
    raise ValueError(message)


def require(condition: bool, message: str) -> None:
    if not condition:
        fail(message)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_no}: invalid JSON") from error
            if not isinstance(row, dict):
                raise TypeError(f"{path}:{line_no}: JSONL row must be an object")
            rows.append(row)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate canonical CPR benchmark manifests.")
    parser.add_argument(
        "--skip-image-files",
        action="store_true",
        help=(
            "Validate manifest structure/semantics without requiring local gallery image files. "
            "The default validation still requires every image path to exist."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gallery = read_jsonl(GALLERY_PATH)
    queries = read_jsonl(QUERIES_PATH)

    require(
        len(gallery) == EXPECTED_GALLERY,
        f"Expected {EXPECTED_GALLERY} gallery images, got {len(gallery)}",
    )
    require(
        len(queries) == EXPECTED_QUERIES,
        f"Expected {EXPECTED_QUERIES} queries, got {len(queries)}",
    )

    image_ids: list[Any] = []
    gallery_by_id: dict[Any, dict[str, Any]] = {}
    checked_images = 0

    for gi, row in enumerate(gallery):
        for key in ("image_id", "path", "person_ids"):
            require(key in row, f"Gallery row {gi}: missing {key}")

        image_id = row["image_id"]
        require(image_id not in gallery_by_id, f"Duplicate gallery image_id: {image_id!r}")
        require(
            isinstance(row["person_ids"], list) and bool(row["person_ids"]),
            f"Gallery row {gi} ({image_id}): person_ids must be a non-empty list",
        )
        require(
            len(row["person_ids"]) == len(set(map(str, row["person_ids"]))),
            f"Gallery row {gi} ({image_id}): duplicate person_ids",
        )
        require(
            isinstance(row["path"], str) and bool(row["path"].strip()),
            f"Gallery row {gi} ({image_id}): invalid path",
        )

        if not args.skip_image_files:
            path = (ROOT / row["path"]).resolve()
            require(path.is_file(), f"Missing image: {path}")
            checked_images += 1

        image_ids.append(image_id)
        gallery_by_id[image_id] = row

    query_ids: set[Any] = set()
    cases: Counter[str] = Counter()

    for qi, q in enumerate(queries):
        for key in (
            "query_id",
            "image_id",
            "text",
            "case",
            "subjects",
            "target_ids",
            "full_positive_ids",
        ):
            require(key in q, f"Query row {qi}: missing {key}")

        query_id = q["query_id"]
        require(query_id not in query_ids, f"Duplicate query_id: {query_id!r}")
        query_ids.add(query_id)

        case = str(q["case"])
        require(
            case in {"SINGLE", "MULTI", "RELATIONAL"},
            f"{query_id}: invalid case {case!r}",
        )
        cases[case] += 1

        require(q["image_id"] in gallery_by_id, f"{query_id}: query image not in gallery")
        require(
            isinstance(q["text"], str) and bool(q["text"].strip()),
            f"{query_id}: empty text",
        )

        target_ids = q["target_ids"]
        require(
            isinstance(target_ids, list) and bool(target_ids),
            f"{query_id}: target_ids must be a non-empty list",
        )
        target_ids_str = [str(x) for x in target_ids]
        require(
            len(target_ids_str) == len(set(target_ids_str)),
            f"{query_id}: duplicate target identity",
        )

        if case == "SINGLE":
            require(len(target_ids) == 1, f"{query_id}: SINGLE must have exactly 1 identity")
        else:
            require(len(target_ids) >= 2, f"{query_id}: {case} must have >=2 identities")

        subjects = q["subjects"]
        require(
            isinstance(subjects, list) and bool(subjects),
            f"{query_id}: subjects must be a non-empty list",
        )
        require(
            len(subjects) == len(target_ids),
            f"{query_id}: subjects/target_ids count mismatch",
        )

        subject_identity_ids: list[str] = []
        subject_ids: list[Any] = []
        relation_text = str(q.get("relation_text") or "").strip()
        for si, subject in enumerate(subjects):
            require(isinstance(subject, dict), f"{query_id}: subject {si} must be an object")
            require("identity_id" in subject, f"{query_id}: subject {si} missing identity_id")
            require("subject_id" in subject, f"{query_id}: subject {si} missing subject_id")
            require(
                bool(str(subject.get("select_text") or "").strip()),
                f"{query_id}: subject {si} has empty select_text",
            )
            modify_text = str(subject.get("modify_text") or "").strip()
            require(
                bool(modify_text or relation_text),
                f"{query_id}: subject {si} has no modify_text and no relation_text fallback",
            )
            subject_identity_ids.append(str(subject["identity_id"]))
            subject_ids.append(subject["subject_id"])

        require(
            subject_identity_ids == target_ids_str,
            f"{query_id}: subjects[].identity_id must match target_ids in order",
        )
        require(
            len(subject_ids) == len(set(subject_ids)),
            f"{query_id}: duplicate subject_id",
        )

        query_people = {str(x) for x in gallery_by_id[q["image_id"]]["person_ids"]}
        require(
            set(target_ids_str).issubset(query_people),
            f"{query_id}: target identity missing from query image",
        )

        positives = q["full_positive_ids"]
        require(
            isinstance(positives, list) and bool(positives),
            f"{query_id}: no Full positive",
        )
        require(
            len(positives) == len(set(positives)),
            f"{query_id}: duplicate Full positive image id",
        )

        for positive_id in positives:
            require(
                positive_id in gallery_by_id,
                f"{query_id}: positive {positive_id!r} not in gallery",
            )
            require(
                positive_id != q["image_id"],
                f"{query_id}: query image is also a Full positive",
            )
            positive_people = {
                str(x) for x in gallery_by_id[positive_id]["person_ids"]
            }
            require(
                set(target_ids_str).issubset(positive_people),
                f"{query_id}: Full positive {positive_id} does not contain all target identities",
            )

    require(
        dict(cases) == EXPECTED_CASES,
        f"Unexpected case counts: {dict(cases)}; expected {EXPECTED_CASES}",
    )

    print()
    print("CPR pilot data validation: OK")
    print("-----------------------------")
    print(f"Gallery      : {len(gallery):,}")
    print(f"Queries      : {len(queries):,}")
    print(f"SINGLE       : {cases['SINGLE']:,}")
    print(f"MULTI        : {cases['MULTI']:,}")
    print(f"RELATIONAL   : {cases['RELATIONAL']:,}")
    if args.skip_image_files:
        print("Image files  : skipped by request")
    else:
        print(f"Image files  : {checked_images:,} checked")
    print()
    print("Manifest ordering and uniqueness checks: OK")
    print("Query subjects/target identity alignment: OK")
    print("All Full positives contain all target identities: OK")


if __name__ == "__main__":
    main()
