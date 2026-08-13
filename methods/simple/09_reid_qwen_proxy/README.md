# S9. ReID-Set + Qwen Edit Proxy

## Purpose

S9 is a **hybrid simple baseline** that combines the strongest person-centric branch from S5
with the pixel-space generative proxy from S8.

```text
final score = alpha * ReID-Set score + (1 - alpha) * generated-proxy similarity
```

The motivation is the strongest common objection:

```text
Let explicit person ReID solve WHO, and let generative editing solve the desired change/context.
```

This baseline asks whether that hybrid is already sufficient for a large fraction of CPR.

## Components

### Identity branch

Exactly S5:

- Grounding DINO Swin-T person detector;
- CLIP-ReID ViT-B/16 trained on MSMT17;
- all detected persons in the query/reference scene and gallery scene;
- Hungarian one-to-one matching;
- strict minimum aggregation.

This produces a concrete **ReID-Set score**.

### Proxy branch

Exactly S8 as the generative proxy source:

- frozen `Qwen/Qwen-Image-Edit-2509`;
- edit the full query/reference scene according to the canonical modification text;
- use the synthetic edited image as a proxy target image;
- retrieve the full gallery with OpenAI CLIP ViT-L/14 image-to-image similarity.

This produces a **generated-proxy similarity**.

## Fusion

The final score is:

```text
alpha * ReID-Set score + (1 - alpha) * generated-proxy similarity
```

There is **no extra score normalization** beyond the branch-internal cosine-style similarities already
produced by S5 and S8.

## Alpha selection

```text
Alpha: validation Full-mAP only.
CPR Supervision: Val only.
```

S9 does **not** tune alpha on the benchmark evaluation queries.
It requires a separate validation split with the same schema:

```text
data/validation/gallery.jsonl
data/validation/queries.jsonl
```

If those validation manifests are missing, or if they are identical to the main evaluation manifests,
S9 refuses to run rather than leaking test labels.

The default alpha grid is:

```text
0.00, 0.05, 0.10, ..., 0.95, 1.00
```

The best alpha is selected by **Full-mAP**. Exact ties are broken by the **smallest alpha**.

## Reuse policy

S9 is intentionally implemented to **reuse prior branch artifacts** where appropriate.

### Main evaluation split

For the main benchmark split, S9 expects the finished branch scores from:

- `runs/groundingdino_clipreid_set/scores.npy` (S5)
- `runs/qwen_image_edit_clip/scores.npy` (S8)

So the simplest recommended order is:

```bash
python run_baseline.py groundingdino_clipreid_set
python run_baseline.py qwen_image_edit_clip
python run_baseline.py reid_set_qwen_edit_proxy
```

This is the intended meaning of “reuse S5 cache and S8 generated images”.

### Validation split

Because S5 and S8 are benchmark methods rather than generic reusable libraries, their released runs do not
already contain validation branch scores. S9 therefore reuses their **adapter logic** to build the validation
branch scores under its own cache namespace.

Validation caches live under:

```text
runs/reid_set_qwen_edit_proxy/cache/
```

including:

- validation person detections and CLIP-ReID features for the S5-style branch;
- validation generated proxy images and CLIP features for the S8-style branch.

## Canonical data contract

Main inference uses only:

```text
data/gallery.jsonl
data/queries.jsonl
```

Validation alpha selection uses only:

```text
data/validation/gallery.jsonl
data/validation/queries.jsonl
```

S9 never consumes benchmark labels for main inference.
Validation labels are used only to choose alpha.

## Failure modes to inspect

S9 is useful because its two branches fail differently.
Important observations include:

- **identity branch strong / proxy branch weak:** wrong context or change, but identities align;
- **proxy branch strong / identity branch weak:** desired change is well described, but identity consistency is poor;
- **fusion conflict:** proxy branch helps some cases but hurts concrete person matching;
- **generator failure:** identity drift, missing person, hallucinated person, wrong relation;
- **set branch failure:** detector misses a person or strict-min penalizes a good image too hard.

## Checkpoint preparation

`download_checkpoint.py` delegates to the existing S5 and S8 checkpoint preparers.
No new external checkpoints are introduced beyond those methods.

## Run

From repository root:

```bash
python run_baseline.py reid_set_qwen_edit_proxy
```

Recommended practical order:

```bash
python run_baseline.py groundingdino_clipreid_set
python run_baseline.py qwen_image_edit_clip
python run_baseline.py reid_set_qwen_edit_proxy
```

## Interpretation

S9 is the direct answer to the objection:

```text
ReID handles identity, generative editing handles requested change/context.
```

If S9 is strong, that supports the claim that a hybrid of concrete person matching and generic image-editing
composition is already a strong CPR baseline. If S9 still fails, that argues the task needs tighter structured
reasoning than “ReID + generic generative proxy”.
