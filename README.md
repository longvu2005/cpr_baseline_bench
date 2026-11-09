# CPR Baseline Benchmark

A common RCR evaluation pipeline for 29 retrieval baselines. Native scene
methods and person methods adapted with shared RCR boxes are labeled separately.
Published method names refer to the documented CPR adaptations, not a rerun of
the original paper's dataset or scores.

## Prepare data

Use the same completed RCR final export and image files for every method.

```bash
python prepare_data.py --rcr-final-dir /path/to/rcr/dataset/data/final
python link_gallery.py --source /path/to/rcr/dataset/data/raw/images
python validate_data.py
```

Validation selects parameters. Test is used for final reporting. The importer
preserves canonical RCR ordering and projects reviewed positives onto each
split. See [data fields and layout](data/README.md).

## Run

```bash
python run_baseline.py --list
python run_baseline.py clip_image
python run_baseline.py fafa_setmatch --rcr-detection-cache /path/to/rcr/cache/detections
```

The runner validates data, prepares an isolated method environment, prepares
checkpoints, performs validation selection where configured, scores test,
evaluates and rebuilds tables. Keep each method's requirements separate from
the notebook's global environment. See [method protocols and dependencies](docs/baselines.md)
and [additional scene/caption methods](docs/additional_baselines.md).

Person methods consume the same RCR Grounding DINO Tiny index. Set its path
with `--rcr-detection-cache` or `CPR_RCR_DETECTION_CACHE`. They share the CLIP
Subject selector and Partial SetMatch. Scene methods use their native encoders;
S1–S4, S6–S9 and S11 use the pinned frozen Hugging Face ViT-B/32 from
RCR Proposed, including letterbox preprocessing and native text globals. The
shared person selector retains RCR FAFA's separate OpenAI ViT-B/32 pipeline.
See [the 29-method audit](docs/audit_29.md) and [Kaggle execution](docs/kaggle.md).

S9 needs completed S5 and S8 source runs. S11 needs completed S6. The configured
order used by `--all` places these sources before their dependents:

```bash
python run_baseline.py --all --rcr-detection-cache /path/to/rcr/cache/detections
```

An unavailable checkpoint returns a nonzero status and produces no substitute
scores. Word4Per needs independently reproduced Stage-2 weights, training config and
external-data provenance receipt. Instruct-ReID selects its unmatched penalty
on complete val in native squared-distance space. Large
editor/VLM methods also need their documented storage and GPU resources.

The runner creates `.baseline_venvs/<method>/` and reuses it while requirements
match. `--skip-install` requires a prepared environment. `--force-checkpoint`
asks the selected checkpoint preparer to refresh supported artifacts.

## Scores and reporting

```text
runs/<method>/scores.npy
runs/<method>/run.json
outputs/<method>/metrics.json
outputs/<method>/run.json
tables/table1_main.csv
tables/table2_cases.csv
```

Scores are finite, descending similarities with shape `(queries, gallery)`.
They retain the reference column; eligible rankings exclude the reference.
Ties preserve gallery order. R@K means at least one positive in the first K
results. Full-mAP is query weighted; Macro-Full-mAP weights present cases equally.

Image content, manifests and inherited score artifacts are checked before
comparison. Evaluation binds metrics to the exact score bytes and evaluated
run metadata. Historical results without that binding are excluded from tables.
After this update, rerun affected inference and evaluation; do not edit metadata
to attach old scores to a new protocol.

```bash
python run_baseline.py --tables-only
python -S build_tables.py --check
```

Tables-only reads evaluated artifacts without models or image scans. See
[reporting checks](docs/reporting.md) and [paper comparison requirements](docs/paper_protocol.md).

## Validation and development

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
python -m compileall -q cpr_baselines cpr_person methods scripts tests *.py
python -m pyflakes .
```

CPU tests validate contracts, matching, leakage boundaries, cache invalidation,
artifact consistency and reporting. Native pretrained GPU inference requires
real model assets and gallery images; passing CPU tests does not certify accuracy.
See [validation scope](docs/additional_validation.md).

For a new method, follow [the integration guide](docs/adding_baselines.md).
Keep model scoring in its method directory and shared benchmark rules in the
common modules. Checkpoint/network preparation belongs in `download_checkpoint.py`.
