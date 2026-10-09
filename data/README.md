# Canonical Stage-2 Data

`prepare_data.py` reads the flattened `data/source/export_stage2.jsonl` directly.
Each non-empty line is one JSON object; a `.txt` file containing JSONL is also
accepted. The export is already curated, so no old `qc_pass`, submission-ID join,
or nested `annotation` object is required. `records.jsonl` is no longer an input.

## Source Schema

```json
{
  "sample_id": "train__query__target",
  "case_type": "GROUP",
  "query_image_id": "query",
  "target_image_id": "target",
  "positive_image_ids": ["target", "another_positive"],
  "subjects": [{"subject_id": 1, "identity_ids": ["10", "20"]}],
  "final_desc": "Identify Subject 1 as the couple wearing white shirts",
  "final_change": "then retrieve target images where Subject 1 are wearing costumes",
  "final_instruction": "Identify Subject 1 as the couple wearing white shirts; then retrieve target images where Subject 1 are wearing costumes."
}
```

Supported case labels are `INDIVIDUAL`, `DUAL`, `GROUP`, and `RELATIONAL`.
Case labels are used for reporting only. Text routing is derived from the
Subject mentions in `final_desc` and `final_change`.

## Canonical Manifest Mapping

| Export field | `data/queries.jsonl` field |
| --- | --- |
| `sample_id` | `query_id` |
| `query_image_id` | `image_id`; `path` comes from the gallery |
| `target_image_id` | `target_image_id` |
| `case_type` | `case` (reporting only) |
| `final_instruction` | `text`, preserved as supplied |
| `final_desc` | `final_desc` plus parsed `subjects[].select_text` |
| `final_change` | `final_change` plus parsed `subjects[].modify_text` |
| `subjects[].identity_ids` | `subjects[].identity_ids` plus flattened `target_ids` |
| `positive_image_ids` | `full_positive_ids`, restricted to the evaluation gallery |

The parser recognizes `Identify Subject N as ...` selector clauses. Independent
target clauses are split only when each Subject is mentioned once and the
clauses have an explicit connector. Cross-Subject references, repeated mentions,
and shared predicates retain the complete target condition in every Subject's
`modify_text` and in `relation_text`. Existing adapters that use full instructions
for relational inputs continue to do so, without consulting `case`.

## Gallery and Relevance

The benchmark uses all TRAIN, VAL, and TEST images from `images.jsonl` in one
common gallery, ordered by `image_idx`. PIPA `index.txt` supplies evaluation identity labels.
Queries are ordered by gallery image position, then `sample_id`. Both the query
image and annotated target must belong to this gallery. Queries from all three
source splits are evaluated against the same full gallery. This is a shared-gallery
evaluation protocol, not three separate split-specific galleries.

All distinct positive images inside the gallery are retained in source order.
Out-of-gallery positives are counted and excluded from this gallery's relevance
set. The query image is removed from positives and excluded by the evaluator
when ranking. The annotated target must remain a non-self positive. Preparation
checks that the query and every retained positive contain all target identities.

With the current image metadata, the gallery contains 30,552 images: 17,000 TRAIN,
5,684 VAL, and 7,868 TEST. For the supplied 4,311-row export, the benchmark has:

| Case | Queries |
| --- | ---: |
| INDIVIDUAL | 3,671 |
| DUAL | 308 |
| GROUP | 151 |
| RELATIONAL | 181 |
| Total | 4,311 |

The 264 VAL and 14 TEST queries are included. These counts describe this export;
the validator checks membership and ordering against the source image metadata
instead of hard-coding gallery or query counts. The supplied export also has
1,759 positive assignments referring to images absent from that metadata; these
are counted as `positive_outside_gallery` and excluded from this gallery's
relevance set. Every annotated target is retained.

## GROUP Adapter Limitation

A group remains one textual Subject with multiple identity labels. Its
`target_ids` contain every member, so strict ID relevance and Full relevance
require all members. A group is never expanded into person slots using GT
identity counts or mappings.

The current text-selected person-crop adapters (Word4Per, FAFA, Instruct-ReID,
Per-Person CLIP Compose, BASIC, and AdaFocal) create one predicted crop slot per
textual Subject. For a GROUP Subject this is a single-crop approximation and
does not explicitly match every group member. Their GROUP scores are reported
with this limitation. Scene-level methods continue to consume the complete
instruction; text-free ReID-Set continues to use all predicted query persons.

## Commands

From the repository root, after replacing `data/source/export_stage2.jsonl`:

```bash
python link_gallery.py --source /path/to/images
python prepare_data.py
python validate_data.py
python run_baseline.py clip_image
```

To read a JSONL attachment directly, supply its path:

```bash
python prepare_data.py --annotations /path/to/export_stage2.txt
```

For manifest-only inspection on a machine without images:

```bash
python prepare_data.py --skip-image-files
python validate_data.py --skip-image-files
python -m unittest discover -s tests -v
```

Inference still requires the gallery image files. Rebuild `queries.jsonl` after
changing the export, and rerun inference. `run_baseline.py` validates the new
manifest before checkpoint/model work and records ordered manifest SHA-256s
with `run.json`. `evaluate.py` rejects a recorded fingerprint mismatch and
places manifest fingerprints in evaluated metrics. `build_tables.py` skips
older outputs, including results without fingerprints, instead of mixing
different query sets in the same comparison.

## Flat Symlink Gallery

`link_gallery.py --source /path/to/images` expects sibling `train/`, `val/`,
and `test/` folders. It creates one symlink per image under `data/gallery`, so
the manifests keep portable paths such as `data/gallery/<image_id>.jpg`.
It replaces an old gallery directory symlink with a real directory containing
file symlinks, without deleting or copying the source images. Repeated execution
reuses matching links. Duplicate filenames across splits and existing real image
files in the destination fail before the directory link is changed.

On Kaggle, from the repository root:

```python
!python link_gallery.py --source /kaggle/input/datasets/phmhunhlongv/cir-data/data/raw/images
!python prepare_data.py
!python validate_data.py
```

Image files not listed in `images.jsonl` are linked if present in the three source
folders, but they do not become evaluation candidates automatically. The source
metadata defines the canonical candidate set.
