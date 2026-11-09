# RCR validation and test data

Import a completed RCR `dataset/data/final` export from the repository root.

```bash
python prepare_data.py --rcr-final-dir /path/to/rcr/dataset/data/final
python link_gallery.py --source /path/to/rcr/dataset/data/raw/images
python validate_data.py
```

The same commands work with a Kaggle mount. Pass the image root that contains
both `val/` and `test/`. Remove any conflicting flat `data/gallery` link
before linking the split image root. For a metadata-only check, use
`python validate_data.py --skip-image-files`.

## Imported files

| Path | Purpose |
| --- | --- |
| `data/val/gallery.jsonl` and `data/val/queries.jsonl` | Complete validation inputs for parameter selection |
| `data/test/gallery.jsonl` and `data/test/queries.jsonl` | Complete test inputs and relevance labels |
| `data/manifest.json` | RCR version, split counts, ordered manifest checksums and full RCR gallery ID hash |

The importer reads `samples.jsonl`, `images.jsonl`, `gallery.jsonl`,
`head_boxes.jsonl`, `manifest.json` and `splits/{val,test}.txt` from RCR.
Gallery membership comes from image paths and preserves RCR gallery order.
Query membership and order come from the two split files. Images are linked
by their RCR relative paths. A new finalized release can be imported without
changing baseline configs.

| RCR field | Baseline query field |
| --- | --- |
| `sample_id` | `query_id` |
| `query_image_id` | `image_id` and its gallery `path` |
| `case_type` | `case` |
| `final_instruction` | `text` |
| `final_desc`, `final_change` | Subject selection and modification text |
| `subjects[].identity_ids` | Flattened `target_ids` for evaluation |
| `positive_image_ids` | `full_positive_ids` projected onto the query split |

Case order is INDIVIDUAL, GROUP, DUAL, RELATIONAL. GROUP has one public Subject
slot with multiple annotated identities. Person adapters predict its membership
from the reference image and selection text. Case and identity labels are used
for evaluation. ID relevance requires every target identity. Full relevance uses
the reviewed positives in the same split. The exact reference image is removed
from eligible rankings.

Reimport the dataset after finalizing new queries or relevance labels. A run
records the manifests it scored. Tuned runs also record the validation
fingerprint, which binds the result to the selection inputs.

Person methods use that gallery ID hash to verify the detector index built by RCR.
