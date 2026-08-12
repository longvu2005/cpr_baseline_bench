# Word4Per + SetMatch

Published baseline adapter for **Word4Per: Zero-shot Composed Person Retrieval** (arXiv:2311.16515, v1-v3 method lineage).

## What is preserved from the authors' implementation

This baseline deliberately imports the official `old_project` code at pinned commit:

```text
Delong-liu-bupt/Composed_Person_Retrieval
commit: 0cc16936f031f7ad166be4cce1be33d0b44b728e
subdir: old_project
```

The adapter uses the official:

- `Word4Per` CLIP-based model;
- `IM2TEXT` textual inversion network;
- inference image transform (`384 x 128` by the released default config);
- CLIP tokenizer;
- composed prompt format `a * is , {relative_caption}`;
- `encode_text_img_retrieval(..., repeat=False)` composed-query encoder;
- Stage-2 `best.pth` checkpoint structure (`model` + `img2text`).

It does **not** replace Word4Per with a new fusion rule.

## Checkpoint status

The current public `old_project/README.md` documents a downloadable **Stage-1** model. The released test script for Word4Per, however, loads the Stage-2 `best.pth` from the experiment output directory. A clearly documented public final Stage-2 download is not provided there.

Therefore this benchmark records the final inference weight as **REPRODUCED**, not "official final checkpoint".

Expected local files:

```text
checkpoints/word4per/word4per_cuhk_pedes_stage2_best.pth
checkpoints/word4per/word4per_cuhk_pedes_stage2_configs.yaml
```

Reproduce Stage 2 using the authors' `old_project` recipe on **CUHK-PEDES**, never on this benchmark CPR pilot.

Official recipe:

```bash
python train_stage2.py \
  --name word4per \
  --root_dir /path/to/datasets \
  --img_aug \
  --batch_size 128 \
  --MLM \
  --lr 1e-4 \
  --optimizer AdamW \
  --dataset_name CUHK-PEDES \
  --loss_names 'sdm+id+mlm' \
  --toword_loss 'text' \
  --num_epoch 60
```

Stage 2 itself starts from the Stage-1 model. The authors' repository contains both a Stage-1 training recipe and a Stage-1 download link.

## MULTI adaptation: SetMatch

Word4Per is applied independently to each composed query component. If a benchmark query or gallery row contains multiple person components, the adapter builds a component-level cosine-similarity matrix and computes the maximum-weight **one-to-one** assignment. The score is the mean assigned score over query components.

This prevents one gallery person from satisfying multiple query-person slots. If a candidate contains fewer gallery components than the query contains, missing slots receive `setmatch.unmatched_score` (default `-1.0`).

This is a benchmark adaptation, not a claim that SetMatch is part of the original Word4Per paper. The adaptation is recorded in `run.json`.

For ordinary singleton rows, this reduces exactly to Word4Per cosine similarity.

## Manifest fields

The benchmark README fixes ordering and score shape but does not define field names. `run.py` therefore resolves common field names automatically and writes the resolved schema into `run.json`. If your manifest uses different fields, set the explicit `data.*_key` entries in `config.yaml`.

A set-valued query can be represented as a list of component objects. Each component needs a reference image; component text may be supplied per component or inherited from the parent query text.

A set-valued gallery entry can be represented as a list of component image paths/objects.

## Run

```bash
pip install -r methods/published/01_word4per_setmatch/requirements.txt
python validate_data.py
python methods/published/01_word4per_setmatch/run.py
python evaluate.py --method word4per_setmatch
python build_tables.py
```

The baseline scores the **complete gallery**. It does not remove the exact query image; that remains the common evaluator's job.

## Expected raw output

```text
runs/word4per_setmatch/
├── scores.npy
└── run.json
```

The common evaluator should then create:

```text
outputs/word4per_setmatch/
├── metrics.json
└── run.json
```
