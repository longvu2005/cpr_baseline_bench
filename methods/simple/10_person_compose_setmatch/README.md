# S10. Per-Person CLIP Compose + SetMatch

## Purpose

S10 tests whether **person-level composition plus set matching** is already sufficient for CPR,
before any learned binding mechanism is needed.

The method does **not** compose the whole scene. Instead, it:

1. detects people in the query/reference scene;
2. localizes the target person(s) using only predicted detections and subject selection text;
3. composes each localized target-person crop with modification text in CLIP space;
4. detects people in each gallery image;
5. matches query-person compositions to gallery persons with Hungarian SetMatch and strict minimum aggregation.

This is a benchmark-defined simple baseline, not an exact reproduction of a published method.

## Pipeline

```text
query/reference scene
    ↓
Grounding DINO person detections
    ↓
subject select_text → target-person localization over predicted persons
    ↓
for each localized target person:
    CLIP image embedding of crop
    +
    CLIP text embedding of modification text
    ↓
normalized weighted composition
    ↓
person-level query set

gallery scene
    ↓
Grounding DINO person detections
    ↓
CLIP image embedding for each detected person crop
    ↓
gallery person set

query-person set × gallery-person set
    ↓
Hungarian one-to-one matching
    ↓
strict minimum aggregation
    ↓
image-level retrieval score
```

## Models

- **Detector:** Grounding DINO Swin-T using the same shared protocol as S5.
- **Retriever/composer:** OpenAI CLIP ViT-L/14.

## Shared target-person localization protocol

S10 does **not** use GT target boxes or target identity labels.
Target persons are localized only from predicted person detections and `subjects[].select_text`.

For a query with `m` subjects:

1. detect all persons in the query image;
2. embed all detected person crops with CLIP image encoder;
3. encode each subject's `select_text` with CLIP text encoder;
4. build the similarity matrix between subject texts and predicted query persons;
5. solve a Hungarian assignment so different subjects map to distinct predicted people.

If the query image contains fewer predicted people than query subjects, or localization fails,
that query receives the method's unmatched score everywhere.

## Composition rule

For each localized target person `i`, S10 forms:

```text
q_i = normalize(alpha * image_crop_i + (1 - alpha) * text_i)
```

where:

- `image_crop_i` is the CLIP image embedding of the localized target-person crop;
- `text_i` is the CLIP text embedding of the modification text used for that subject.

Text selection rule:

- **RELATIONAL:** use the full instruction `query.text` for every subject;
- otherwise: use `subject.modify_text` when present;
- otherwise fall back to `query.relation_text` if present;
- otherwise fall back to `query.text`.

## SetMatch rule

Given the composed query-person set and the detected gallery-person set:

- pairwise score = cosine similarity;
- assignment = Hungarian maximum-sum matching;
- image score = **strict minimum** across the assigned person pairs;
- if the gallery image has fewer detected persons than the number of query subjects,
  the image score is the fixed unmatched score `-1.0`.

For SINGLE queries, the image score is simply the maximum similarity over the gallery's detected persons.

## Supervision and alpha selection

```text
CPR Supervision: Val only
```

S10 uses validation labels only to choose the composition weight `alpha`.
No benchmark evaluation labels are used in the main inference path.

Required validation manifests:

```text
data/validation/gallery.jsonl
data/validation/queries.jsonl
```

If those validation manifests are missing, or if they are byte-identical to the main evaluation manifests,
S10 refuses to run rather than leaking test labels.

Default alpha grid:

```text
0.00, 0.05, 0.10, ..., 0.95, 1.00
```

Selection criterion:

```text
Full-mAP on validation split
```

Exact ties are broken by the **smallest alpha**.

## Cache design

S10 intentionally reuses the **shared detection protocol** already established by S5.

### Main split

- detections: `runs/groundingdino_clipreid_set/cache/person_detections.npz`
- CLIP person features: `runs/per_person_clip_compose_setmatch/cache/main_person_features.npy`

### Validation split

- detections: `runs/per_person_clip_compose_setmatch/cache/val_person_detections.npz`
- CLIP person features: `runs/per_person_clip_compose_setmatch/cache/val_person_features.npy`

The CLIP person-feature cache fingerprint covers:

- S10 config;
- the specific detection cache;
- the CLIP checkpoint;
- the CLIP model name;
- cache schema + adapter version.

## Important behavioral scope

- **No GT target box**
- **No GT identity label for localization**
- **No relation classifier**
- **RELATIONAL uses the full instruction directly**
- **No learned binding head**

So S10 isolates the question:

```text
Is person-level CLIP composition plus explicit SetMatch already enough?
```

## Checkpoint preparation

`download_checkpoint.py` delegates to:

- S5 checkpoint preparation for the Grounding DINO shared protocol;
- CLIP ViT-L/14 checkpoint preparation from the existing CLIP text baseline.

No model/tokenizer is silently downloaded in `run.py`.

## Run

From repository root:

```bash
python run_baseline.py per_person_clip_compose_setmatch
```

## Interpretation

S10 is the direct baseline for the objection:

```text
Maybe CPR only needs person-level composition and set matching, before any learned binding mechanism.
```

If S10 performs strongly, that suggests explicit person decomposition + CLIP composition may already capture much of the task.
If it performs weakly, that supports the need for stronger structured reasoning or learned binding beyond simple CLIP-space composition.
