# Word4Per + SetMatch

Published baseline adapter for the earlier **Word4Per / Word for Person** zero-shot CPR method associated with arXiv:2311.16515 (v1-v3 method lineage).

## Official implementation preserved

The adapter imports the authors' pinned `old_project` implementation:

```text
Delong-liu-bupt/Composed_Person_Retrieval
commit: 0cc16936f031f7ad166be4cce1be33d0b44b728e
subdir: old_project
```

The retrieval path preserves the official Word4Per components used for Stage-2 inference:

- the authors' CLIP-based person retrieval model;
- `IM2TEXT` textual inversion network;
- official inference transform from the reproduced Stage-2 config;
- official tokenizer;
- prompt template `a * is , {relative_caption}`;
- `encode_text_img_retrieval(..., repeat=False)` composed-query encoder;
- Stage-2 checkpoint loading through `Checkpointer_Toword`.

The benchmark adapter does not replace Word4Per with an ad-hoc image/text fusion rule.

## Checkpoint status

The public `old_project` README documents a downloadable Stage-1 model, while Stage-2 testing expects the experiment's final `best.pth`. A clearly documented public final Stage-2 download is not provided there.

Therefore this benchmark labels the final inference artifacts as:

```text
REPRODUCED
```

Expected local files:

```text
checkpoints/word4per/word4per_cuhk_pedes_stage2_best.pth
checkpoints/word4per/word4per_cuhk_pedes_stage2_configs.yaml
```

Reproduce Stage 2 using the authors' `old_project` recipe on **CUHK-PEDES**, never on the CPR benchmark data. The public recipe uses the Stage-1 model as initialization and then trains the textual inversion stage with the authors' Stage-2 settings, including:

```text
batch_size 128
lr 1e-4
optimizer AdamW
dataset_name CUHK-PEDES
loss_names 'sdm+id'
toword_loss 'text'
num_epoch 60
```

`download_checkpoint.py` intentionally does not invent an official final download. It validates that the reproduced Stage-2 checkpoint and its matching config are present and fails early with their exact expected paths when they are missing. After that validation it also pins the official source checkout and prepares the public auxiliary weights used at inference: the Stage-2 base OpenAI CLIP backbone, the CLIP query-person selector, and the torchvision person detector. `run.py` loads those assets from local paths rather than downloading them silently.

## Scene-to-person localization used by this adapter

The benchmark uses full scene images, while Word4Per is natively a cropped-person CPR method. The normal benchmark therefore uses predicted person instances only:

1. torchvision Faster R-CNN ResNet-50-FPN-v2 predicts person candidates in every gallery/query scene;
2. for each query target, OpenAI CLIP (`ViT-B/32`) scores `subjects[].select_text` against the predicted query-person crops;
3. Hungarian assignment chooses one predicted reference crop per target subject;
4. Word4Per is then applied to each selected reference person crop and its relative modification text.

The adapter does **not** use PIPA GT boxes, identity-to-box mappings, `target_ids`, or positive labels for localization or retrieval scoring.

## MULTI / RELATIONAL adaptation: SetMatch

Word4Per is applied independently to every target subject. Every predicted gallery person is encoded with Word4Per's image encoder, giving a target-by-gallery-person cosine-similarity matrix.

For each gallery image:

1. compute maximum-weight one-to-one Hungarian matching between target subjects and predicted gallery persons;
2. if the gallery has fewer predicted persons than targets, pad missing slots with `setmatch.unmatched_score`;
3. use the **minimum assigned target score** as the final image score.

This is an AND-style SetMatch rule: every target must be supported, and one strong target cannot compensate for one weak/missing target.

For SINGLE, this reduces to the best Word4Per score over predicted persons in the gallery image.

SetMatch and predicted scene localization are benchmark adaptations; they are not claimed to be part of the original Word4Per paper.

## Query text behavior

Each target subject uses:

```text
subjects[].modify_text
```

as Word4Per's relative caption. If it is empty, `relation_text` is used; if both are empty, the query-level `text` is used.

`subjects[].select_text` is used only for predicted target localization inside the query scene. It is not concatenated into Word4Per's final relative-caption prompt.

RELATIONAL queries are therefore not claimed to be jointly relation-aware inside Word4Per itself: the method is applied independently per target and SetMatch enforces one-to-one target coverage.

## Benchmark contract

The adapter:

- reads `data/gallery.jsonl` and `data/queries.jsonl` in exact order;
- scores every query against the complete gallery;
- leaves the exact query image in `scores.npy` for the common evaluator to remove;
- writes finite scores with shape `(len(queries), len(gallery))`;
- records checkpoint, source, localization, SetMatch, and runtime metadata in `run.json`;
- performs no CPR benchmark training, fine-tuning, checkpoint selection, or tuning.

## Run

Normal end-to-end command:

```bash
python run_baseline.py word4per_setmatch
```

The root runner installs this method's `requirements.txt` first, validates the reproduced Stage-2 artifacts, prepares the pinned source and auxiliary weights, runs inference, evaluates the method, and rebuilds the tables.

Expected raw output:

```text
runs/word4per_setmatch/
├── scores.npy
├── run.json
└── cache/
```

Expected evaluated output:

```text
outputs/word4per_setmatch/
├── metrics.json
└── run.json
```
