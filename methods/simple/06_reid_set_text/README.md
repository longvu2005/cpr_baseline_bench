# S6. Grounding DINO + CLIP-ReID - Set + Text

## Purpose

This is the strongest conventional simple baseline in the CPR benchmark. It tests a direct decomposition:

- **WHO:** detect every person and compare identity using CLIP-ReID.
- **WHAT:** compare the CPR query text with the full gallery scene using OpenAI CLIP.

The final score is

```text
final_score = alpha * ReID_Set_score + (1 - alpha) * CLIP_text_score
```

No score normalization is inserted between the two branches. Both branches are cosine-style similarities. By default `alpha=0.50` is fixed, which keeps the baseline label-free. An optional validation mode can select `alpha` on a separate validation Full-mAP split.

## Components

### ReID-Set branch

S6 reuses the exact S5 adapter and its caches:

- detector: Grounding DINO Swin-T;
- detector config: `groundingdino/config/GroundingDINO_SwinT_OGC.py`;
- detector checkpoint: `groundingdino_swint_ogc.pth`;
- ReID: CLIP-ReID ViT-B/16 trained on MSMT17;
- official CLIP-ReID evaluation checkpoint: `ViT-B-16_60.pth`;
- all predicted persons form the query/gallery sets;
- maximum-weight one-to-one Hungarian assignment;
- strict minimum over assigned similarities;
- unmatched score `-1.0` when the gallery has fewer detected persons than the query set.

S6 intentionally imports `methods/simple/05_reid_set/run.py` instead of cloning the S5 implementation. This keeps person detection, embedding, matching, and cache semantics identical between S5 and S6. On an S5 cache miss, S6 runs the same one-sample CLIP-ReID CUDA preflight before launching expensive detector/ReID inference. The S5 branch is completed before OpenAI CLIP ViT-L/14 is loaded, reducing unnecessary peak GPU memory.

### Text branch

- model: OpenAI CLIP ViT-L/14;
- checkpoint: official `ViT-L-14.pt`;
- query representation: canonical `queries.jsonl` field `text`;
- gallery representation: global CLIP image embedding of the complete scene;
- score: cosine similarity of L2-normalized text/image embeddings.

The main gallery CLIP cache is deliberately shared with the existing CLIP baselines when its fingerprint matches:

```text
runs/clip_image/gallery_features_vit_l14.npy
```

## Alpha selection and CPR supervision

The default configuration is:

```yaml
fusion:
  alpha_selection:
    mode: fixed
    fixed_alpha: 0.50
```

This requires no CPR labels and is reported as:

```text
CPR Supervision: No
```

To tune `alpha`, explicitly switch to:

```yaml
fusion:
  alpha_selection:
    mode: validation
```

Validation mode is **never** allowed to tune on the canonical evaluation queries.

It requires separate manifests configured as:

```text
data/validation/gallery.jsonl
data/validation/queries.jsonl
```

The validation query manifest must follow the same schema and contain `full_positive_ids`, because those labels are used only to compute **validation Full-mAP** for alpha selection.

Default search grid:

```text
0.00, 0.05, ..., 0.95, 1.00
```

The selected alpha is the value with maximum validation Full-mAP. Exact metric ties are resolved deterministically by choosing the smallest alpha.

The result is stored in:

```text
runs/groundingdino_clipreid_set_text/alpha_selection.json
```

The result is fingerprinted by the S6 config, S5 config, validation manifests, CLIP checkpoint, alpha grid and adapter version. A stale selection is recomputed.

In validation mode benchmark supervision is:

```text
CPR Supervision: Val only
```

The adapter explicitly refuses to tune if the validation query manifest is byte-identical to the canonical evaluation query manifest, regardless of which gallery manifest is supplied. Runtime metadata derives `cpr_supervision` from the active mode, so switching modes cannot silently leave a stale supervision label.

## Data contract

Main benchmark inference reads only:

```text
data/gallery.jsonl
data/queries.jsonl
```

and preserves their exact row order. The complete output matrix is:

```text
scores.shape == (len(queries), len(gallery))
```

The query image is not removed inside S6. `evaluate.py` remains responsible for self-image exclusion.

CPR labels from the canonical evaluation queries are never used to tune alpha. In fixed mode no CPR labels are used for fusion selection. In validation mode, validation `full_positive_ids` are used only in the alpha-selection stage.

## Checkpoint preparation

`download_checkpoint.py` first runs the S5 preparer so Grounding DINO, BERT runtime assets, CLIP-ReID source/checkpoint and its OpenAI ViT-B/16 backbone are prepared. It then downloads and validates OpenAI CLIP ViT-L/14.

No inference stage performs a network download.

## Caches

Main S5 detection/ReID caches are shared with S5. S6 keeps branch-score caches separately:

```text
runs/groundingdino_clipreid_set_text/cache/
  main_reid_set_scores.npy
  main_clip_text_scores.npy
  val_person_detections.npz
  val_person_features.npy
  val_reid_set_scores.npy
  val_clip_gallery_features.npy
  val_clip_text_scores.npy
```

Each cache has a metadata fingerprint. Shape-only validation is not accepted. The `val_*` caches are only used when `alpha_selection.mode: validation`.

## Run

S5 must already be present in the repository because S6 imports its implementation.

With the default fixed-alpha configuration, a clean checkout can run directly:

```bash
python run_baseline.py groundingdino_clipreid_set_text
```

The root runner will:

```text
1. install S6 requirements
2. run S6 checkpoint preparation
3. run S6 inference using fixed alpha, or validation-only selection when explicitly enabled
4. run the official evaluator
5. rebuild benchmark tables
```

## Interpretation

S5 asks whether person detection + identity matching alone is sufficient.

S6 asks whether a conventional decomposition is sufficient when both signals are available:

```text
WHO  = Grounding DINO + CLIP-ReID SetMatch
WHAT = OpenAI CLIP global text-to-gallery semantics
```

If S6 remains below a CPR-specific method, the comparison is particularly useful because it separates gains from ordinary person identity matching and ordinary global text semantics from gains that require explicit composed-person reasoning.
