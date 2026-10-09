#!/usr/bin/env python3
"""Build the common TRAIN/VAL/TEST gallery from the flattened Stage-2 export."""

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
SOURCE = DATA / "source"
GALLERY_DIR = DATA / "gallery"
IMAGES = SOURCE / "images.jsonl"
ANNOTATIONS = SOURCE / "export_stage2.jsonl"
PIPA_INDEX = SOURCE / "index.txt"
GALLERY_OUT = DATA / "gallery.jsonl"
QUERIES_OUT = DATA / "queries.jsonl"
CASE_TYPES = ("INDIVIDUAL", "DUAL", "GROUP", "RELATIONAL")
GALLERY_SPLITS = ("TRAIN", "VAL", "TEST")

SUBJECT_DESC = re.compile(r"\bSubject\s+(\d+)\s+as\s+", re.IGNORECASE)
SUBJECT_REF = re.compile(r"\bSubject\s+(\d+)\b", re.IGNORECASE)
CHANGE_PREFIX = re.compile(r"^then\s+retrieve\s+target\s+images\s+where\s+", re.IGNORECASE)
CLAUSE_JOIN = re.compile(r"(?:[,;]?\s+and|[,;]?\s+while|[,;])\s*$", re.IGNORECASE)


def read_jsonl(path):
    rows = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_no}: invalid JSON") from error
            if not isinstance(row, dict):
                raise TypeError(f"{path}:{line_no}: JSONL row must be an object")
            rows.append(row)
    return rows


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def select_gallery_images(images):
    """Keep all three source splits, in the canonical metadata image order."""
    return sorted((image for image in images if image["source_split"] in GALLERY_SPLITS),
                  key=lambda image: int(image["image_idx"]))


def load_identity_index(path):
    """PIPA columns: album_id photo_id x y width height identity_id split."""
    identities = defaultdict(set)
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            parts = line.split()
            if not parts:
                continue
            if len(parts) != 8:
                raise ValueError(f"{path}:{line_no}: expected 8 columns, got {len(parts)}")
            identities[f"{parts[0]}_{parts[1]}"].add(parts[6])
    return {image_id: sorted(ids, key=int) for image_id, ids in identities.items()}


def query_texts(record):
    """Derive adapter inputs from text alone; case and identity labels never route text."""
    query_id = record["sample_id"]
    texts = {}
    for key in ("final_desc", "final_change", "final_instruction"):
        value = record.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{query_id}: missing or empty {key}")
        texts[key] = value.strip()

    description = texts["final_desc"]
    markers = list(SUBJECT_DESC.finditer(description))
    if not markers or description[:markers[0].start()].strip().lower() != "identify":
        raise ValueError(f"{query_id}: final_desc must identify named Subjects")

    selectors = {}
    for index, marker in enumerate(markers):
        subject_id = int(marker.group(1))
        end = markers[index + 1].start() if index + 1 < len(markers) else len(description)
        fragment = description[marker.end():end].strip()
        if index + 1 < len(markers):
            fragment = CLAUSE_JOIN.sub("", fragment).strip()
        if subject_id in selectors or not fragment:
            raise ValueError(f"{query_id}: duplicate or empty Subject {subject_id} description")
        selectors[subject_id] = fragment

    condition = CHANGE_PREFIX.sub("", texts["final_change"]).strip()
    mentions = list(SUBJECT_REF.finditer(condition))
    if not condition or not mentions:
        raise ValueError(f"{query_id}: final_change must mention a named Subject")
    mentioned_ids = {int(marker.group(1)) for marker in mentions}
    if not mentioned_ids.issubset(selectors):
        raise ValueError(f"{query_id}: final_change mentions an undefined Subject")

    # Split only unambiguous independent clauses. Cross-Subject references,
    # shared predicates and repeated mentions keep the entire condition.
    modifiers = {}
    independent = (
        len(mentions) == len(selectors)
        and mentioned_ids == set(selectors)
        and mentions[0].start() == 0
    )
    if independent:
        for index, marker in enumerate(mentions):
            end = mentions[index + 1].start() if index + 1 < len(mentions) else len(condition)
            fragment = condition[marker.end():end].strip()
            if index + 1 < len(mentions):
                join = CLAUSE_JOIN.search(fragment)
                if join is None:
                    independent = False
                    break
                fragment = fragment[:join.start()].strip()
            if not fragment:
                independent = False
                break
            modifiers[int(marker.group(1))] = fragment

    relation_text = None
    if not independent:
        modifiers = {subject_id: condition for subject_id in selectors}
        if len(selectors) > 1:
            relation_text = condition

    return texts, selectors, modifiers, relation_text


def build_queries(records, gallery):
    for index, record in enumerate(records):
        for key in ("sample_id", "query_image_id", "target_image_id"):
            if key not in record:
                raise ValueError(f"Stage-2 row {index}: missing {key}; expected the flattened export format")
    gallery_by_id = {row["image_id"]: row for row in gallery}
    gallery_order = {row["image_id"]: index for index, row in enumerate(gallery)}
    queries = []
    skipped = Counter()
    seen = set()

    # Same deterministic convention as the previous pilot: image order, then ID.
    ordered = sorted(records, key=lambda row: (
        gallery_order.get(str(row["query_image_id"]), -1), str(row["sample_id"])
    ))
    for record in ordered:
        query_id = str(record["sample_id"])
        if query_id in seen:
            raise ValueError(f"Duplicate sample_id: {query_id}")
        seen.add(query_id)
        query_image_id = str(record["query_image_id"])
        target_image_id = str(record["target_image_id"])
        if query_image_id not in gallery_by_id:
            skipped["query_outside_gallery"] += 1
            continue
        if target_image_id not in gallery_by_id:
            skipped["target_outside_gallery"] += 1
            continue

        case = record.get("case_type")
        if case not in CASE_TYPES:
            raise ValueError(f"{query_id}: unsupported case_type {case!r}")
        texts, selectors, modifiers, relation_text = query_texts(record)
        source_subjects = record.get("subjects")
        if not isinstance(source_subjects, list) or not source_subjects:
            raise ValueError(f"{query_id}: subjects must be a non-empty list")
        labels = {}
        for subject in source_subjects:
            subject_id = subject["subject_id"]
            ids = subject.get("identity_ids")
            if not isinstance(subject_id, int) or subject_id <= 0 or subject_id in labels:
                raise ValueError(f"{query_id}: invalid or duplicate subject_id {subject_id!r}")
            if not isinstance(ids, list) or not ids:
                raise ValueError(f"{query_id}: Subject {subject_id} has no identity_ids")
            ids = [str(identity_id) for identity_id in ids]
            if len(ids) != len(set(ids)) or any(not identity_id for identity_id in ids):
                raise ValueError(f"{query_id}: Subject {subject_id} has invalid identity_ids")
            labels[subject_id] = ids
        if set(labels) != set(selectors):
            raise ValueError(f"{query_id}: subjects do not match final_desc Subject names")

        # Subject slots come from final_desc, including one slot for a group.
        # identity_ids are evaluation labels, never person counts for inference.
        subjects = [{
            "subject_id": subject_id,
            "identity_ids": labels[subject_id],
            "select_text": selector,
            "modify_text": modifiers[subject_id],
        } for subject_id, selector in selectors.items()]
        target_ids = [identity_id for subject in subjects for identity_id in subject["identity_ids"]]
        if len(target_ids) != len(set(target_ids)):
            raise ValueError(f"{query_id}: identity_ids overlap between Subjects")

        positive_ids = record.get("positive_image_ids")
        if not isinstance(positive_ids, list) or not positive_ids:
            raise ValueError(f"{query_id}: positive_image_ids must be a non-empty list")
        positive_ids = list(dict.fromkeys(str(image_id) for image_id in positive_ids))
        if target_image_id not in positive_ids or target_image_id == query_image_id:
            raise ValueError(f"{query_id}: target_image_id must be a non-self positive")
        full_positive_ids = []
        for image_id in positive_ids:
            if image_id == query_image_id:
                skipped["self_positive_removed"] += 1
            elif image_id not in gallery_by_id:
                skipped["positive_outside_gallery"] += 1
            else:
                full_positive_ids.append(image_id)

        expected_ids = set(target_ids)
        for image_id in (query_image_id, *full_positive_ids):
            if not expected_ids.issubset(gallery_by_id[image_id]["person_ids"]):
                raise ValueError(f"{query_id}: image {image_id} does not contain all target identities")

        queries.append({
            "query_idx": len(queries),
            "query_id": query_id,
            "image_id": query_image_id,
            "path": gallery_by_id[query_image_id]["path"],
            "target_image_id": target_image_id,
            "text": texts["final_instruction"],
            "final_desc": texts["final_desc"],
            "final_change": texts["final_change"],
            "case": case,
            "subjects": subjects,
            "relation_text": relation_text,
            "target_ids": target_ids,
            "full_positive_ids": full_positive_ids,
        })

    if not queries:
        raise ValueError("No Stage-2 queries belong to the gallery")
    return queries, skipped


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, default=ANNOTATIONS,
                        help="Flattened Stage-2 JSONL input (a .txt JSONL file is also accepted).")
    parser.add_argument("--skip-image-files", action="store_true",
                        help="Build manifests from metadata without requiring local image files.")
    args = parser.parse_args()
    for path in (IMAGES, args.annotations, PIPA_INDEX):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not args.skip_image_files and not GALLERY_DIR.is_dir():
        raise FileNotFoundError(f"{GALLERY_DIR} must be a directory or directory symlink")

    images = read_jsonl(IMAGES)
    if len({str(image["image_id"]) for image in images}) != len(images):
        raise ValueError("Duplicate image_id in images.jsonl")
    identities_by_image = load_identity_index(PIPA_INDEX)
    gallery_images = select_gallery_images(images)
    if not gallery_images:
        raise ValueError("No TRAIN/VAL/TEST images found in images.jsonl")

    gallery = []
    for gallery_idx, image in enumerate(gallery_images):
        image_id = str(image["image_id"])
        file_name = Path(image["relative_path"]).name
        if not args.skip_image_files and not (GALLERY_DIR / file_name).is_file():
            raise FileNotFoundError(f"Missing gallery image: {GALLERY_DIR / file_name}")
        person_ids = identities_by_image.get(image_id, [])
        if not person_ids:
            raise ValueError(f"No identity annotation for gallery image: {image_id}")
        gallery.append({"gallery_idx": gallery_idx, "image_id": image_id,
                        "path": f"data/gallery/{file_name}", "person_ids": person_ids,
                        "source_split": image["source_split"]})

    queries, skipped = build_queries(read_jsonl(args.annotations), gallery)
    write_jsonl(GALLERY_OUT, gallery)
    write_jsonl(QUERIES_OUT, queries)
    cases = Counter(query["case"] for query in queries)
    print(f"CPR data prepared: {len(gallery):,} gallery images, {len(queries):,} queries")
    image_counts = Counter(row["source_split"] for row in gallery)
    for split in GALLERY_SPLITS:
        print(f"Gallery {split:16s}: {image_counts[split]:,}")
    for case in CASE_TYPES:
        print(f"{case:24s}: {cases[case]:,}")
    for reason, count in sorted(skipped.items()):
        print(f"{reason:24s}: {count:,}")
    if args.skip_image_files:
        print("Image files: skipped by request")
    print(f"Saved: {GALLERY_OUT}\nSaved: {QUERIES_OUT}")


if __name__ == "__main__":
    main()
