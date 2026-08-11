#!/usr/bin/env python3

import json
from collections import Counter, defaultdict
from pathlib import Path

DATA = Path("data")
SOURCE = DATA / "source"
GALLERY_DIR = DATA / "gallery"

RECORDS = SOURCE / "records.jsonl"
IMAGES = SOURCE / "images.jsonl"
ANNOTATIONS = SOURCE / "export_stage2.jsonl"
PIPA_INDEX = SOURCE / "index.txt"

GALLERY_OUT = DATA / "gallery.jsonl"
QUERIES_OUT = DATA / "queries.jsonl"


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


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            )


def load_identity_index(path):
    """
    PIPA index.txt format:

    album_id photo_id x y width height identity_id ...
    """

    identities = defaultdict(set)

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            parts = line.split()

            if not parts:
                continue

            if len(parts) != 8:
                raise ValueError(
                    f"{path}:{line_no}: "
                    f"expected 8 columns, got {len(parts)}"
                )

            album_id = parts[0]
            photo_id = parts[1]
            identity_id = parts[6]

            image_id = f"{album_id}_{photo_id}"
            identities[image_id].add(identity_id)

    return {
        image_id: sorted(person_ids, key=int)
        for image_id, person_ids in identities.items()
    }


def is_clean_query(record):
    """
    Pilot formulation:

    SINGLE:
        exactly one subject
        exactly one target identity

    MULTI / RELATIONAL:
        exactly two subjects
        one identity per subject
        two distinct target identities
    """

    if not record.get("qc_pass", False):
        return False

    case = record.get("case_type")
    subjects = record.get("subjects", [])

    if case == "SINGLE":
        return (
            len(subjects) == 1
            and len(subjects[0].get("identity_ids", [])) == 1
        )

    if case in {"MULTI", "RELATIONAL"}:
        if len(subjects) != 2:
            return False

        if any(
            len(subject.get("identity_ids", [])) != 1
            for subject in subjects
        ):
            return False

        target_ids = [
            subject["identity_ids"][0]
            for subject in subjects
        ]

        return len(set(target_ids)) == 2

    return False


def main():
    # ---------------------------------------------------------
    # Validate inputs
    # ---------------------------------------------------------

    required = [
        RECORDS,
        IMAGES,
        ANNOTATIONS,
        PIPA_INDEX,
    ]

    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)

    # Works for both a real directory and a directory symlink.
    if not GALLERY_DIR.exists():
        raise FileNotFoundError(
            f"{GALLERY_DIR} does not exist"
        )

    if not GALLERY_DIR.is_dir():
        raise ValueError(
            f"{GALLERY_DIR} must be a directory or directory symlink"
        )

    # ---------------------------------------------------------
    # Load source data
    # ---------------------------------------------------------

    records = read_jsonl(RECORDS)
    images = read_jsonl(IMAGES)
    annotations = read_jsonl(ANNOTATIONS)

    identities_by_image = load_identity_index(PIPA_INDEX)

    image_by_id = {
        image["image_id"]: image
        for image in images
    }

    if len(image_by_id) != len(images):
        raise ValueError("Duplicate image_id in images.jsonl")

    annotation_by_id = {}

    for row in annotations:
        submission_id = row["submission_id"]

        if submission_id in annotation_by_id:
            raise ValueError(
                f"Duplicate submission_id: {submission_id}"
            )

        annotation_by_id[submission_id] = row

    # ---------------------------------------------------------
    # Build gallery
    #
    # Current pilot gallery = source_split TRAIN = 17k images.
    # ---------------------------------------------------------

    gallery_images = [
        image
        for image in images
        if image["source_split"] == "TRAIN"
    ]

    # Preserve the original deterministic image ordering.
    gallery_images.sort(key=lambda x: int(x["image_idx"]))

    gallery = []
    gallery_ids = set()

    for gallery_idx, image in enumerate(gallery_images):
        image_id = str(image["image_id"])
        file_name = Path(image["relative_path"]).name

        image_path = GALLERY_DIR / file_name

        # Path.is_file follows directory symlinks automatically.
        if not image_path.is_file():
            raise FileNotFoundError(
                f"Missing gallery image: {image_path}"
            )

        person_ids = identities_by_image.get(image_id, [])

        if not person_ids:
            raise ValueError(
                f"No identity annotation for gallery image: "
                f"{image_id}"
            )

        if image_id in gallery_ids:
            raise ValueError(
                f"Duplicate gallery image_id: {image_id}"
            )

        gallery_ids.add(image_id)

        gallery.append(
            {
                "gallery_idx": gallery_idx,
                "image_id": image_id,
                "path": f"data/gallery/{file_name}",
                "person_ids": person_ids,
            }
        )

    # ---------------------------------------------------------
    # Build queries
    # ---------------------------------------------------------

    queries = []
    case_counts = Counter()

    skipped = Counter()

    # Stable original ordering.
    records.sort(
        key=lambda r: (
            int(r.get("query_image_idx", -1)),
            str(r["sample_id"]),
        )
    )

    for record in records:
        if not is_clean_query(record):
            skipped["not_clean"] += 1
            continue

        query_id = str(record["sample_id"])
        query_image_id = str(record["query_image_id"])
        target_image_id = str(record["target_image_id"])

        # The pilot uses a single 17k gallery.
        # Both query and annotated target must exist inside it.
        if query_image_id not in gallery_ids:
            skipped["query_outside_gallery"] += 1
            continue

        if target_image_id not in gallery_ids:
            skipped["target_outside_gallery"] += 1
            continue

        raw = annotation_by_id.get(query_id)

        if raw is None:
            raise ValueError(
                f"No raw annotation found for query: {query_id}"
            )

        annotation = raw.get("annotation", {})

        text = (
            annotation.get("captionFinal")
            or raw.get("caption")
            or ""
        ).strip()

        if not text:
            raise ValueError(
                f"Empty query text: {query_id}"
            )

        subjects = []
        target_ids = []

        for subject in record["subjects"]:
            identity_id = str(
                subject["identity_ids"][0]
            )

            target_ids.append(identity_id)

            subjects.append(
                {
                    "subject_id": int(subject["subject_id"]),
                    "identity_id": identity_id,
                    "select_text": (
                        subject.get("select_text") or ""
                    ).strip(),
                    "modify_text": (
                        subject.get("modify_text") or ""
                    ).strip(),
                }
            )

        # Deduplicate while preserving order.
        target_ids = list(dict.fromkeys(target_ids))

        # -----------------------------------------------------
        # Identity consistency checks
        # -----------------------------------------------------

        query_people = set(
            identities_by_image.get(query_image_id, [])
        )

        target_people = set(
            identities_by_image.get(target_image_id, [])
        )

        expected_people = set(target_ids)

        if not expected_people.issubset(query_people):
            raise ValueError(
                f"{query_id}: target identity not present "
                f"in query image "
                f"{query_image_id}"
            )

        if not expected_people.issubset(target_people):
            raise ValueError(
                f"{query_id}: target identity not present "
                f"in target image "
                f"{target_image_id}"
            )

        image = image_by_id[query_image_id]
        query_file_name = Path(
            image["relative_path"]
        ).name

        case = str(record["case_type"]).upper()

        queries.append(
            {
                "query_idx": len(queries),
                "query_id": query_id,

                # Query image remains in the global gallery.
                # Evaluator removes it only at ranking time.
                "image_id": query_image_id,
                "path": f"data/gallery/{query_file_name}",

                # Full natural-language CPR instruction.
                "text": text,

                "case": case,

                # Useful for person-aware baselines.
                "subjects": subjects,

                # RELATIONAL text when available.
                "relation_text": record.get(
                    "pair_modify_text"
                ),

                # Used to compute strict ID relevance.
                "target_ids": target_ids,

                # Current annotation provides one known
                # Full-CPR positive target image per query.
                "full_positive_ids": [
                    target_image_id
                ],
            }
        )

        case_counts[case] += 1

    # ---------------------------------------------------------
    # Final checks
    # ---------------------------------------------------------

    if len(gallery) != 17000:
        raise ValueError(
            f"Expected 17000 gallery images, "
            f"found {len(gallery)}"
        )

    query_ids = [
        query["query_id"]
        for query in queries
    ]

    if len(query_ids) != len(set(query_ids)):
        raise ValueError(
            "Duplicate query_id in generated queries"
        )

    # ---------------------------------------------------------
    # Write
    # ---------------------------------------------------------

    write_jsonl(GALLERY_OUT, gallery)
    write_jsonl(QUERIES_OUT, queries)

    print()
    print("CPR pilot data prepared successfully")
    print("------------------------------------")
    print(f"Gallery      : {len(gallery):,}")
    print(f"Queries      : {len(queries):,}")
    print(f"SINGLE       : {case_counts['SINGLE']:,}")
    print(f"MULTI        : {case_counts['MULTI']:,}")
    print(f"RELATIONAL   : {case_counts['RELATIONAL']:,}")

    print()
    print("Skipped")
    print("-------")

    for key, value in sorted(skipped.items()):
        print(f"{key:24s}: {value:,}")

    print()
    print("Saved")
    print("-----")
    print(GALLERY_OUT)
    print(QUERIES_OUT)


if __name__ == "__main__":
    main()
