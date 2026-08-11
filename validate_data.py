#!/usr/bin/env python3

import json
from collections import Counter
from pathlib import Path

GALLERY_PATH = Path("data/gallery.jsonl")
QUERIES_PATH = Path("data/queries.jsonl")


def read_jsonl(path):
    
    rows = []

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()

            if not line:
                continue

            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"{path}:{line_no}: invalid JSON"
                ) from e

    return rows


def main():
    gallery = read_jsonl(GALLERY_PATH)
    queries = read_jsonl(QUERIES_PATH)

    # ---------------------------------------------------------
    # Gallery
    # ---------------------------------------------------------

    assert len(gallery) == 17000, (
        f"Expected 17000 gallery images, got {len(gallery)}"
    )

    image_ids = [g["image_id"] for g in gallery]

    assert len(image_ids) == len(set(image_ids)), (
        "Duplicate image_id in gallery"
    )

    gallery_by_id = {
        g["image_id"]: g
        for g in gallery
    }

    for g in gallery:
        assert g["person_ids"], (
            f'No person_ids: {g["image_id"]}'
        )

        path = Path(g["path"])

        assert path.is_file(), (
            f"Missing image: {path}"
        )

    # ---------------------------------------------------------
    # Queries
    # ---------------------------------------------------------

    assert len(queries) == 2975, (
        f"Expected 2975 queries, got {len(queries)}"
    )

    query_ids = [q["query_id"] for q in queries]

    assert len(query_ids) == len(set(query_ids)), (
        "Duplicate query_id"
    )

    cases = Counter()

    for q in queries:
        query_id = q["query_id"]
        case = q["case"]

        assert case in {
            "SINGLE",
            "MULTI",
            "RELATIONAL",
        }, f"{query_id}: invalid case {case}"

        cases[case] += 1

        # Query image must belong to global gallery.
        assert q["image_id"] in gallery_by_id, (
            f"{query_id}: query image not in gallery"
        )

        # Full positives must exist.
        positives = q["full_positive_ids"]

        assert positives, (
            f"{query_id}: no Full positive"
        )

        for image_id in positives:
            assert image_id in gallery_by_id, (
                f"{query_id}: positive {image_id} "
                f"not in gallery"
            )

            assert image_id != q["image_id"], (
                f"{query_id}: query image is also "
                f"the Full positive"
            )

        target_ids = q["target_ids"]

        assert target_ids, (
            f"{query_id}: empty target_ids"
        )

        assert len(target_ids) == len(set(target_ids)), (
            f"{query_id}: duplicate target identity"
        )

        # Case definition.
        if case == "SINGLE":
            assert len(target_ids) == 1, (
                f"{query_id}: SINGLE must have 1 identity"
            )

        if case in {"MULTI", "RELATIONAL"}:
            assert len(target_ids) >= 2, (
                f"{query_id}: {case} must have >=2 identities"
            )

        # Target identities must exist in query image.
        query_people = set(
            gallery_by_id[q["image_id"]]["person_ids"]
        )

        assert set(target_ids).issubset(query_people), (
            f"{query_id}: target identity missing "
            f"from query image"
        )

        # Every annotated Full-positive must contain
        # the complete target identity set.
        for positive_id in positives:
            positive_people = set(
                gallery_by_id[positive_id]["person_ids"]
            )

            assert set(target_ids).issubset(
                positive_people
            ), (
                f"{query_id}: Full positive "
                f"{positive_id} does not contain "
                f"all target identities"
            )

        assert q["text"].strip(), (
            f"{query_id}: empty text"
        )

    # ---------------------------------------------------------
    # Expected pilot statistics
    # ---------------------------------------------------------

    expected = {
        "SINGLE": 2671,
        "MULTI": 225,
        "RELATIONAL": 79,
    }

    assert dict(cases) == expected, (
        f"Unexpected case counts: {dict(cases)}"
    )

    print()
    print("CPR pilot data validation: OK")
    print("-----------------------------")
    print(f"Gallery      : {len(gallery):,}")
    print(f"Queries      : {len(queries):,}")
    print(f"SINGLE       : {cases['SINGLE']:,}")
    print(f"MULTI        : {cases['MULTI']:,}")
    print(f"RELATIONAL   : {cases['RELATIONAL']:,}")
    print()
    print("All image paths exist.")
    print("All query images exist in gallery.")
    print("All Full positives exist in gallery.")
    print("All Full positives contain all target identities.")
    print("Strict MULTI identity consistency: OK")


if __name__ == "__main__":
    main()
