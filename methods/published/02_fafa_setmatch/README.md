# FAFA + SetMatch

Published baseline adapter for **FAFA (Fine-grained Adaptive Feature Alignment)**
from *Automatic Synthetic Data and Fine-grained Adaptive Feature Alignment for
Composed Person Retrieval* (NeurIPS 2025).

## Official implementation and checkpoint

The adapter imports the authors' implementation instead of reimplementing FAFA:

```text
Delong-liu-bupt/Composed_Person_Retrieval
commit: 0cc16936f031f7ad166be4cce1be33d0b44b728e
subdir: FAFA_SynCPR
```

The authors state that all retrieval training/inference code is open-sourced in
`FAFA_SynCPR` and release a paper-version pretrained model. Their official
`run_inference.sh` uses the checkpoint name `tuned_recall_at1_step.pt`. Put the
released weight at:

```text
checkpoints/fafa/tuned_recall_at1_step.pt
```

The adapter preserves the official inference path:

- model: `blip2_fafa_cpr` / `pretrain`;
- checkpoint loading with `strict=False`, matching `inference_fafa.py`;
- test preprocessing based on `squarepad_transform_test(224)`;
- multimodal query feature from `model.extract_features({image, text_input})`;
- target feature tokens from `model.extract_target_features(..., mode="mean")`;
- soft FDA similarity: top-`k` target-token similarities followed by their mean;
- released/default FDA settings `k=6`, `alpha=0.5` and soft aggregation.

FAFA is **not retrained, fine-tuned, checkpoint-selected, or tuned** on this CPR
benchmark. CPR Supervision is therefore **No**.

## Scene-to-person localization used by this benchmark adapter

The pilot benchmark uses full scene images and intentionally does not expose a
GT target box in `queries.jsonl`. SetMatch, however, needs person instances.
This adapter therefore uses **predicted** person localization only:

1. torchvision Faster R-CNN ResNet-50-FPN-v2 proposes person boxes in every
   gallery/query image;
2. for a query with multiple `subjects`, OpenAI CLIP (`ViT-B/32`) scores each
   `subjects[].select_text` against the predicted query-person crops;
3. a one-to-one Hungarian assignment selects one predicted reference crop per
   target subject.

No PIPA identity-to-box mapping, GT person box, `target_ids`, or positive labels
are used to localize a target or compute a retrieval score. The CLIP selector is
only a target-localization adapter; final retrieval similarities come from FAFA.

This is distinct from the benchmark's **Predicted Anchor [2]** convention for
methods that natively require an anchor/box: FAFA itself is not box-conditioned.
Predicted crops here are the instance construction required to apply **SetMatch
[1]** to multi-person scene images.

## MULTI adaptation: SetMatch [1]

FAFA is natively a one-reference-person / one-target-person CPR method. For each
benchmark query:

1. run FAFA independently for every target subject;
2. run FAFA target encoding independently for every predicted person in a
   gallery image;
3. build the target-person score matrix using the official FAFA soft-FDA score;
4. compute maximum-weight **one-to-one Hungarian matching**;
5. take the **minimum score among the matched target slots** as the image score.

The current pilot has exactly two targets for MULTI/RELATIONAL. `run.py` uses an
algebraically equivalent vectorized two-row specialization for speed and checks
the same maximum-sum one-to-one assignment objective; the generic SciPy Hungarian
path remains as the fallback for future queries with more than two targets.

If a gallery image has fewer predicted persons than the number of targets, the
matrix is padded with `setmatch.unmatched_score` (default `-1.0`). Thus every
target slot must be matched and there is no partial credit.

For SINGLE, SetMatch reduces to the best FAFA score over predicted persons in the
gallery image.

## Query text behavior

Each target subject uses:

```text
subjects[].modify_text
```

as FAFA's relative caption. If it is empty, `relation_text` is used; if both are
empty, the query-level `text` is used. RELATIONAL queries are therefore not
claimed to be jointly relation-aware: the original single-person FAFA model is
applied independently to each subject and SetMatch enforces only one-to-one
coverage.

`subjects[].select_text` is used only to select the predicted reference person
inside the query image. It is not concatenated into FAFA's modification text.

## Benchmark contract

The adapter:

- reads `data/gallery.jsonl` and `data/queries.jsonl` in exact order;
- scores every query against the complete gallery;
- leaves the exact query image in `scores.npy` (the common evaluator removes it);
- writes `runs/fafa_setmatch/scores.npy` with shape
  `(len(queries), len(gallery))` and finite scores only;
- writes reproducibility metadata to `runs/fafa_setmatch/run.json`;
- does not calculate the official benchmark metrics itself.

The adapter may also cache predicted person boxes and FAFA gallery-person
features plus component-person FAFA scores under `runs/fafa_setmatch/cache/` to
avoid recomputing them.

## Run

```bash
pip install -r methods/published/02_fafa_setmatch/requirements.txt

# Download the official released FAFA weight (see checkpoints/fafa/README.md).
python validate_data.py
python methods/published/02_fafa_setmatch/run.py
python evaluate.py --method fafa_setmatch
python build_tables.py
```

Expected benchmark outputs:

```text
runs/fafa_setmatch/
├── scores.npy
├── run.json
└── cache/

outputs/fafa_setmatch/
├── metrics.json
└── run.json
```
