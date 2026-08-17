# P9. AdaFocal + SetMatch

Published CPR adapter for **AdaFocal** from *Beyond Semantic Search: Towards Referential Anchoring in Composed Image Retrieval* (CVPR 2026 Highlight).

## Benchmark definition

This adapter intentionally separates the **official AdaFocal model** from the **CPR adaptation**:

1. Shared Grounding DINO detects persons in every query/gallery image.
2. Query subjects are localized to predicted query-person boxes with **OpenAI CLIP ViT-B/32 + Hungarian max-sum assignment**. No GT box or GT identity is used.
3. For each selected subject, the **full reference scene** and its predicted person bbox are passed through the official **AdaFocal scalar** query branch. The bbox is transformed with the official TargetPad bbox transform before CAAM.
4. Every detected gallery person is cropped and encoded by the official AdaFocal target branch.
5. AdaFocal pairwise similarity is computed between each anchored query and every gallery-person candidate. It matches the official scoring rule: cosine-like dot product to all target Q-Former tokens, then max over target tokens.
6. For MULTI queries, SetMatch performs Hungarian max-sum assignment and uses the strict minimum assigned score as the image score.
7. RELATIONAL queries use the full instruction for every anchored subject, matching the benchmark's existing SetMatch convention.

The main-table path never reads a GT target box. A future `AdaFocal-GT Oracle` should be a separate method/row.

## Official resources

- Code: `HaHaJun1101/OACIR`
- Pinned source commit: `11307ff5e31cda82fe70b0a8e5d6a9d34c130ae9`
- Official checkpoint repository: `HaHaJun1101/AdaFocal`
- Checkpoint: `adafocal_scalar.pt`
- Scalar checkpoint SHA256:
  `03e885888facffd789640fe4a4fd209e4924255124d5bee32bccfd280b52406e`
- Model-repository revision used for download:
  `c36dfffb367a9a1f95d8f40460e7d668c8f1c436`

`adafocal_vector.pt` is not used because it is the vector-beta ablation, not the default scalar configuration.

## Install

From the repository root:

```bash
pip install -r methods/published/09_adafocal_setmatch/requirements.txt
```

The adapter imports the **vendored LAVIS fork from the pinned OACIR checkout**, not the pip `salesforce-lavis` package.

## Prepare assets

```bash
python methods/published/09_adafocal_setmatch/download_checkpoint.py
```

This prepares:

- the shared Grounding DINO assets used by S5;
- OpenAI CLIP ViT-B/32 for predicted-anchor localization;
- the pinned official OACIR source checkout;
- the official `adafocal_scalar.pt` checkpoint.

AdaFocal's official ViT-G/BERT base assets may be populated by the official LAVIS loader on the first model load, so Kaggle Internet should be enabled for that first load.

## Run

```bash
python methods/published/09_adafocal_setmatch/run.py
```

Outputs:

```text
runs/adafocal_setmatch/
├── cache/
│   ├── target_person_features.npy
│   └── target_person_features.npy.meta.json
├── scores.npy
└── run.json
```

The shared detection cache remains:

```text
runs/groundingdino_clipreid_set/cache/person_detections.npz
```

## Memory notes

AdaFocal uses EVA-CLIP ViT-G and the released scalar checkpoint is about 2.77 GB. Defaults are conservative:

```yaml
gallery_person_batch_size: 8
candidate_feature_chunk_size: 2048
target_feature_cache_dtype: float16
```

If CUDA OOM occurs, lower `gallery_person_batch_size` first (for example 8 -> 4 -> 2), then lower `candidate_feature_chunk_size`.

## Scoring

For an anchored subject \(a\) and gallery-person candidate \(p\):

```text
AdaFocal(a, p)
= max over target Q-Former tokens of similarity(
    anchored query fusion feature,
    gallery-person target feature token
  )
```

For a SINGLE query:

```text
image_score = max score among detected gallery persons
```

For MULTI:

```text
pairwise matrix
-> Hungarian assignment maximizing total score
-> strict minimum of the assigned pair scores
```

A gallery image with fewer detected persons than the number of query subjects receives `unmatched_score = -1.0`.

## Why this is not an AdaFocal reimplementation

The adapter uses the official released model class, official CAAM/Attention Activation Mechanism, official TargetPad transform, official bbox transform, official scalar checkpoint, and official target feature head. The benchmark-specific code is limited to predicted-person localization, person-candidate construction, caching, and SetMatch aggregation.

## Expected benchmark supervision

```text
CPR Supervision: No
GT box in main table: No
```
