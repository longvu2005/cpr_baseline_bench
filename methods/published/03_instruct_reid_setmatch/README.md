# P3. Instruct-ReID + SetMatch

## What this baseline is

This adapter integrates the official **Instruct-ReID** model from CVPR 2024 into the CPR benchmark.
The original work defines instruction-guided person ReID: a query person image and a language/image instruction are fused into a retrieval representation. Its newly introduced **language-instructed ReID** setting is especially close to CPR person-level composition.

Official source:

```text
hwz-zju/Instruct-ReID
```

Pinned source commit:

```text
8250f44a301a50d8afcd2e09a46c3e96bf52090d
```

CPR supervision:

```text
No
```

No CPR training, fine-tuning, validation tuning, target identities, positives, or GT boxes are used.

## Checkpoint policy: final model vs bootstrap weights

This distinction is important.

The official README has two different artifact categories:

1. **training/bootstrap assets** such as `pass_vit_base_full.pth`, `ALBEF.pth`, and BERT weights;
2. **final inference models for each ReID task**, provided in the official Google Drive inference-model folder.

P3 uses the **official final inference model for the language-instructed / `attr` task** as its method checkpoint. It does **not** present `pass_vit_base_full.pth`, `ALBEF.pth`, or BERT as the final Instruct-ReID checkpoint.

At inference the official model constructor still expects a local BERT directory. Therefore `download_checkpoint.py` prepares BERT only as a runtime/bootstrap dependency, then loads the official task checkpoint over the constructed model exactly as the official testing path does.

Because the official README links a Drive **folder**, not a stable single-file URL, the preparer uses `gdown --folder --json` to inspect the folder without downloading it all. It selects exactly one checkpoint whose Drive path identifies the language/attribute task. If zero or multiple candidates match, it fails and prints the discovered checkpoint paths rather than guessing. `checkpoint.direct_file_url` in `config.yaml` may be filled with the exact official file URL if the upstream folder naming changes.

## Official inference path preserved

The official testing script constructs:

```text
PASS_Transformer_DualAttn_joint
```

with the test settings:

```text
test_task_type = attr
validate_feat = fusion
attn_type = dual_attn
fusion_loss = all
fusion_branch = bio+clot
vit_type = base
vit_fusion_layer = 2
test_feat_type = f
```

P3 preserves those settings.

For language-instructed inference, the model takes:

```text
person image + language instruction
```

and returns the fusion representation:

```text
concat([bio_f, clot_f])
```

which is 1536-D for the official base model.

The official source contains placeholder strings such as:

```text
<your project root> + /Instruct-ReID/bert-base-uncased
```

P3 does not edit the pinned source checkout. Instead, the wrapper redirects only those BERT/config path lookups to the prepared local artifacts during model construction. This keeps the official source checkout clean and the architecture unchanged.

## Scene-to-person adapter

Instruct-ReID expects cropped person images, while CPR gallery entries are scenes. P3 therefore uses the same deterministic predicted-instance adapter style as the existing published `+ SetMatch` baselines.

### Person detection

All query/gallery scenes are processed with:

```text
torchvision Faster R-CNN ResNet-50-FPN-v2 (COCO)
```

Only predicted COCO `person` detections are used.

No GT target boxes or identity-to-box mapping are used.

### Query target localization

For every subject in `queries.jsonl`:

1. encode every detected query-person crop with OpenAI CLIP ViT-B/32;
2. encode `subjects[].select_text` with CLIP;
3. build the subject-text × detected-person similarity matrix;
4. use maximum-weight Hungarian assignment so multiple subjects select distinct predicted persons.

Fallback text order is:

```text
select_text -> modify_text -> query.text
```

If fewer people are detected than required targets, the query receives the fixed unmatched score against the full gallery.

This localization uses text and predicted boxes only; it never uses `target_ids`.

## CPR instruction adapter

For a selected query target person, P3 supplies the Instruct-ReID language branch with:

```text
subject.modify_text
```

when available.

If `relation_text` is present, P3 uses the complete canonical `query.text` for that subject so RELATIONAL instructions are not reduced to a local modifier. Otherwise it falls back to `query.text` when `modify_text` is absent.

Before BERT tokenization, text follows the official language-instructed preprocessing behavior: lowercase/cleanup and truncation to 50 words. The official model itself tokenizes to max length 70.

### Why the gallery uses a fixed neutral instruction

The original language-instructed benchmark ships language annotations associated with its person samples. CPR gallery scenes do not provide a per-person target-language annotation, and generating one would introduce another model and change P3.

Therefore P3 uses the native Instruct-ReID traditional-ReID instruction:

```text
Do not change clothes.
```

for every gallery person. This gives one query-independent official-model feature per detected gallery person while the query representation remains instruction-edited.

This is a **CPR benchmark adaptation**, not an exact reproduction of the COCAS+ language-instructed evaluation protocol. The source model, final language-task checkpoint, image/text encoders, editing/fusion path, and output feature remain official.

## Pairwise score and SetMatch

The official Instruct-ReID evaluator compares fusion features with squared Euclidean distance. P3 preserves that ranking geometry by defining person-pair similarity as:

```text
pair_score = - || q_i - g_j ||_2^2
```

so higher is better for the CPR score contract.

For `m` query targets and `n` gallery persons:

1. build the `m × n` person-pair score matrix;
2. if `n < m`, return the fixed unmatched score;
3. otherwise use maximum-weight Hungarian one-to-one assignment;
4. aggregate assigned scores with the strict minimum (AND-style SetMatch).

For SINGLE, this reduces to the maximum pair score over detected gallery persons.

The unmatched score is:

```text
-1000000.0
```

which is intentionally far below ordinary negative squared feature distances while remaining finite.

## MULTI and RELATIONAL

**MULTI** uses distinct query target detections, one Instruct-ReID feature per target, Hungarian matching, and strict-min aggregation.

**RELATIONAL** has no extra relation classifier. When `relation_text` is present, every target receives the full canonical query instruction. This deliberately tests how far the person-level Instruct-ReID model generalizes to CPR relations without adding a learned relation module.

## Caches

P3 caches expensive deterministic stages:

```text
runs/instruct_reid_setmatch/cache/person_detections.npz
runs/instruct_reid_setmatch/cache/selector_person_features.npy
runs/instruct_reid_setmatch/cache/gallery_instruct_features.npy
runs/instruct_reid_setmatch/cache/query_target_features.npz
```

Each model-dependent cache has a metadata fingerprint including the relevant manifest, source/config/checkpoint hashes, detector settings, selector/model settings, and adapter version.

## Run

From repository root:

```bash
python run_baseline.py instruct_reid_setmatch
```

The root runner performs:

```text
install requirements
-> prepare official source/final checkpoint/runtime assets
-> inference with progress
-> official evaluate.py
-> build_tables.py
```

P3 requires CUDA because the pinned official Instruct-ReID code hard-codes CUDA for the BERT instruction tensors during forward inference.

## Expected outputs

```text
runs/instruct_reid_setmatch/scores.npy
runs/instruct_reid_setmatch/run.json
outputs/instruct_reid_setmatch/metrics.json
```

`scores.npy` preserves the complete canonical query × gallery matrix. The method does not remove the query image; the official benchmark evaluator owns self-image exclusion.
