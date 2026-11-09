#!/usr/bin/env python3
"""Import validation and test manifests from a finalized RCR dataset."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path, PurePosixPath
import re
from tempfile import TemporaryDirectory

from benchmark_data import CASE_TYPES, DATA_PROTOCOL, SPLITS, load_jsonl, manifest_fingerprint
from cpr_person.cache import cache_lock, canonical_hash

ROOT = Path(__file__).resolve().parent

SUBJECT_REF = re.compile(r"\bSubject\s+(\d+)\b", re.IGNORECASE)
SELECT_SPLIT = re.compile(r"\s+and\s+(?=Subject\s+\d+\s+as\s+)")
CHANGE_PREFIX = "then retrieve target images where "
CLAUSE_JOIN = re.compile(r"(?:[,;]?\s+and|[,;]?\s+while|[,;])\s*$", re.IGNORECASE)


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def index_unique(rows, key):
    indexed = {}
    for row in rows:
        value = row[key]
        if not isinstance(value, str) or not value or value in indexed:
            raise ValueError(f"Invalid or duplicate {key}: {value!r}")
        indexed[value] = row
    return indexed


def query_texts(sample, subject_ids):
    """Apply RCR's selection parser and case-independent FAFA clause routing."""
    desc, change, instruction = (
        sample[key] for key in ("final_desc", "final_change", "final_instruction")
    )
    if any(not isinstance(text, str) or not text.strip() for text in (desc, change, instruction)):
        raise ValueError(f"{sample['sample_id']}: missing finalized text")
    if not desc.startswith("Identify ") or not change.startswith(CHANGE_PREFIX):
        raise ValueError(f"{sample['sample_id']}: invalid finalized text prefix")
    parts = SELECT_SPLIT.split(desc.removeprefix("Identify "))
    if len(parts) != len(subject_ids):
        raise ValueError(f"{sample['sample_id']}: selection text does not match Subject slots")
    selectors = []
    for part, subject_id in zip(parts, subject_ids, strict=True):
        prefix = f"Subject {subject_id} as "
        if not part.startswith(prefix) or not part[len(prefix):].strip():
            raise ValueError(f"{sample['sample_id']}: missing selection for Subject {subject_id}")
        selectors.append(part[len(prefix):].strip())

    condition = change.removeprefix(CHANGE_PREFIX).strip()
    mentions = list(SUBJECT_REF.finditer(condition))
    mentioned_ids = {int(marker[1]) for marker in mentions}
    if mentioned_ids != set(subject_ids):
        raise ValueError(f"{sample['sample_id']}: target text does not match Subject slots")

    if len(mentions) == len(subject_ids) and mentions[0].start() == 0:
        modifiers = {}
        for index, marker in enumerate(mentions):
            end = mentions[index + 1].start() if index + 1 < len(mentions) else len(condition)
            fragment = condition[marker.end():end].strip()
            if index + 1 < len(mentions):
                join = CLAUSE_JOIN.search(fragment)
                if join is None:
                    break
                fragment = fragment[:join.start()].strip()
            if not fragment:
                break
            modifiers[int(marker[1])] = fragment
        if len(modifiers) == len(subject_ids):
            return selectors, [modifiers[subject_id] for subject_id in subject_ids], None
    return selectors, [condition] * len(subject_ids), condition if len(subject_ids) > 1 else None


def build_queries(samples, gallery, split):
    gallery_by_id = {row["image_id"]: row for row in gallery}
    queries = []
    for sample in samples:
        query_id = sample["sample_id"]
        image_id = sample["query_image_id"]
        target_id = sample["target_image_id"]
        if image_id not in gallery_by_id or target_id not in gallery_by_id or image_id == target_id:
            raise ValueError(f"{query_id}: query and distinct seed must belong to {split}")
        case = sample["case_type"]
        if case not in CASE_TYPES:
            raise ValueError(f"{query_id}: unknown RCR case {case!r}")
        subjects = sample["subjects"]
        subject_ids = [subject["subject_id"] for subject in subjects]
        expected = [1] if case in ("INDIVIDUAL", "GROUP") else [1, 2]
        if subject_ids != expected:
            raise ValueError(f"{query_id}: {case} requires Subject slots {expected}")
        target_ids = []
        for subject in subjects:
            ids = subject["identity_ids"]
            if not isinstance(ids, list) or not ids or any(not isinstance(i, str) or not i for i in ids):
                raise ValueError(f"{query_id}: invalid Subject identity IDs")
            target_ids.extend(ids)
        required = set(target_ids)
        if len(required) != len(target_ids):
            raise ValueError(f"{query_id}: duplicate Subject identity")
        if (case == "INDIVIDUAL" and len(required) != 1) or (case == "GROUP" and len(required) < 2):
            raise ValueError(f"{query_id}: {case} does not match the finalized Subject size")

        # Same projection as RCR split_samples(). Preserve the reviewed order.
        positives = [i for i in sample["positive_image_ids"] if i in gallery_by_id]
        if target_id not in positives or image_id in positives or len(set(positives)) != len(positives):
            raise ValueError(f"{query_id}: invalid split Full positives")
        for i in (image_id, *positives):
            if not required.issubset(gallery_by_id[i]["person_ids"]):
                raise ValueError(f"{query_id}: image {i} lacks a required identity")

        selectors, modifiers, relation = query_texts(sample, subject_ids)
        queries.append({
            "query_idx": len(queries),
            "query_id": query_id,
            "image_id": image_id,
            "path": gallery_by_id[image_id]["path"],
            "target_image_id": target_id,
            "source_split": split.upper(),
            "case": case,
            "text": sample["final_instruction"],
            "final_desc": sample["final_desc"],
            "final_change": sample["final_change"],
            "subjects": [
                {
                    "subject_id": subject["subject_id"],
                    "identity_ids": subject["identity_ids"],
                    "select_text": select,
                    "modify_text": modify,
                }
                for subject, select, modify in zip(subjects, selectors, modifiers, strict=True)
            ],
            "relation_text": relation,
            "target_ids": target_ids,
            "full_positive_ids": positives,
        })
    return queries


def import_final(final_dir, root):
    source = json.loads((final_dir / "manifest.json").read_text(encoding="utf-8"))
    if source.get("partial") is not False or not source.get("version"):
        raise ValueError("Expected a complete, versioned RCR final export")
    samples = index_unique(load_jsonl(final_dir / "samples.jsonl"), "sample_id")
    images = index_unique(load_jsonl(final_dir / "images.jsonl"), "image_id")
    gallery = index_unique(load_jsonl(final_dir / "gallery.jsonl"), "image_id")
    if gallery.keys() != images.keys():
        raise ValueError("RCR gallery must contain every registered image exactly once")

    by_split = {split: [] for split in SPLITS}
    for image_id in gallery:
        path = PurePosixPath(images[image_id]["path"])
        if path.is_absolute() or len(path.parts) < 2 or ".." in path.parts:
            raise ValueError(f"Invalid RCR image path: {path}")
        if path.parts[0] in by_split:
            by_split[path.parts[0]].append((image_id, path))
    eligible = {image_id for rows in by_split.values() for image_id, _ in rows}
    identities = defaultdict(set)
    for head in load_jsonl(final_dir / "head_boxes.jsonl"):
        if head["image_id"] in eligible:
            identities[head["image_id"]].add(head["identity_id"])

    data = root / "data"
    data.mkdir(parents=True, exist_ok=True)
    metadata = {
        "protocol": DATA_PROTOCOL,
        "rcr_version": source["version"],
        "rcr_gallery_ids_sha256": canonical_hash(list(gallery)),
        "splits": {},
    }
    seen = set()
    with TemporaryDirectory(prefix=".rcr-import-", dir=data) as temporary:
        staged_root = Path(temporary)
        for split in SPLITS:
            split_path = final_dir / "splits" / f"{split}.txt"
            sample_ids = split_path.read_text(encoding="utf-8").split()
            if (
                not sample_ids
                or len(sample_ids) != source["splits"].get(split)
                or len(set(sample_ids)) != len(sample_ids)
                or seen.intersection(sample_ids)
                or not set(sample_ids) <= samples.keys()
            ):
                raise ValueError(f"Invalid RCR {split} split: {split_path}")
            seen.update(sample_ids)
            rows = []
            for image_id, path in by_split[split]:
                rows.append({
                    "gallery_idx": len(rows),
                    "image_id": image_id,
                    "path": str(PurePosixPath("data/gallery") / path),
                    "source_split": split.upper(),
                    "person_ids": sorted(identities[image_id]),
                })
            if len(rows) < 2:
                raise ValueError(f"{split}: expected at least two gallery images")
            queries = build_queries([samples[i] for i in sample_ids], rows, split)
            staged = staged_root / "data" / split
            staged.mkdir(parents=True)
            write_jsonl(staged / "gallery.jsonl", rows)
            write_jsonl(staged / "queries.jsonl", queries)
            counts = Counter(query["case"] for query in queries)
            metadata["splits"][split] = {
                "num_gallery": len(rows),
                "num_queries": len(queries),
                "case_counts": {case: counts[case] for case in CASE_TYPES},
                "fingerprint": manifest_fingerprint(staged_root, split),
            }

        # Publish the manifest last. Readers reject mixed-generation split files.
        marker = staged_root / "manifest.json"
        marker.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        with cache_lock(data):
            for split in SPLITS:
                (data / split).mkdir(exist_ok=True)
                for name in ("gallery.jsonl", "queries.jsonl"):
                    (staged_root / "data" / split / name).replace(data / split / name)
            marker.replace(data / "manifest.json")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rcr-final-dir", type=Path, required=True, help="RCR dataset/data/final directory")
    args = parser.parse_args()
    metadata = import_final(args.rcr_final_dir.expanduser(), ROOT)
    print(f"Imported RCR {metadata['rcr_version']}")
    for split, info in metadata["splits"].items():
        print(f"{split}: {info['num_queries']:,} queries, {info['num_gallery']:,} images")


if __name__ == "__main__":
    main()
