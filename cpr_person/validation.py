"""Keep optional fusion selection and unmatched calibration off evaluation inputs."""
from __future__ import annotations

import json

from .cache import canonical_hash, resolve, sha256_file


def public_query_signature(row):
    return canonical_hash({"image_id":row["image_id"],"text":row.get("text"),
        "relation_text":row.get("relation_text"),
        "subjects":[{k:s.get(k) for k in ("select_text","modify_text")} for s in row["subjects"]]})


def ensure_separate_queries(main_query_manifest, validation_query_manifest):
    main,validation=resolve(main_query_manifest),resolve(validation_query_manifest)
    if main==validation or sha256_file(main)==sha256_file(validation):
        raise ValueError("Validation queries must be separate from evaluation queries")
    def signatures(path):
        return {public_query_signature(json.loads(line)) for line in path.read_text().splitlines() if line.strip()}
    if signatures(main)&signatures(validation):
        raise ValueError("Validation contains evaluation query inputs, even if IDs/GT labels were changed")
