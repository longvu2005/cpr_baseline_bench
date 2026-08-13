# S10. Per-Person CLIP Compose + SetMatch

## Purpose

S10 tests whether **person-level composition plus explicit set matching** is already sufficient for CPR before introducing a learned binding mechanism.

This is a benchmark-defined simple baseline, not an exact reproduction of a published method.

The method does **not** compose the whole scene directly. Instead, it:

1. detects people in the query/reference image;
2. localizes the target person(s) using predicted person crops and `subjects[].select_text`;
3. composes each selected person crop with the corresponding modification text in CLIP space;
4. detects people in every gallery image;
5. represents every detected gallery person with CLIP;
6. matches the query-person set to each gallery-person set with Hungarian assignment;
7. aggregates the matched similarities with a strict minimum.

The baseline isolates the question:

```text
Is explicit person decomposition + CLIP composition + SetMatch already enough for CPR?
```

---

## Pipeline

```text
query/reference scene
    ↓
Grounding DINO person detections
    ↓
subjects[].select_text
    ↓
CLIP text ↔ detected-person CLIP image similarity
    ↓
Hungarian assignment
    ↓
localized target-person crop(s)
    ↓
CLIP image embedding + modification-text embedding
    ↓
q_i = normalize(alpha * image_i + (1 - alpha) * text_i)
    ↓
query-person composition set


gallery scene
    ↓
Grounding DINO person detections
    ↓
CLIP image embedding for each detected person
    ↓
gallery-person set


query-person set × gallery-person set
    ↓
cosine-similarity matrix
    ↓
Hungarian maximum-sum assignment
    ↓
strict minimum over assigned pairs
    ↓
image-level retrieval score
```

---

## Models

### Detector

**Grounding DINO Swin-T**

S10 reuses the exact Grounding DINO protocol implemented by:

```text
methods/simple/05_reid_set
```

The pinned official Grounding DINO source is cloned by S5's checkpoint preparer and imported directly from that checkout.

Grounding DINO is intentionally **not installed as a pip wheel**.

This avoids fragile C++/CUDA wheel builds on hosted Python 3.12 environments such as Kaggle.

The shared S5 adapter:

- attempts to use the custom Grounding DINO CUDA/C++ op when available;
- otherwise switches to the official pure-PyTorch deformable-attention implementation;
- includes the active attention backend in the detection-cache fingerprint.

### Composer / retriever

**OpenAI CLIP ViT-L/14**

Checkpoint:

```text
checkpoints/clip/ViT-L-14.pt
```

The checkpoint is prepared through the existing CLIP baseline checkpoint preparer.

---

## Shared Grounding DINO protocol

S10 imports the S5 implementation from:

```text
methods/simple/05_reid_set/run.py
```

Configured in:

```yaml
shared_protocol:
  method: groundingdino_clipreid_set
  method_dir: methods/simple/05_reid_set
  config: methods/simple/05_reid_set/config.yaml
```

S10 therefore shares:

- the exact Grounding DINO repository;
- the exact pinned Grounding DINO commit;
- detector checkpoint;
- person prompt;
- confidence thresholds;
- NMS settings;
- runtime Hugging Face cache;
- direct-source import behavior;
- custom-op / PyTorch-fallback handling;
- detection-cache validation logic.

The main detection cache is intentionally shared with S5:

```text
runs/groundingdino_clipreid_set/cache/person_detections.npz
```

If the cache fingerprint matches the current S5 detector protocol, S10 reuses it and does **not** rerun Grounding DINO over the full gallery.

---

## Target-person localization

S10 does not use:

- GT person boxes;
- GT target-person box assignment;
- GT target identity labels for localization.

For a query containing `m` subjects:

1. detect all persons in the query/reference image;
2. encode every detected person crop with CLIP image encoder;
3. encode every `subjects[i].select_text` with CLIP text encoder;
4. compute the text-to-person cosine-similarity matrix;
5. solve maximum-weight Hungarian assignment.

This ensures different subjects are assigned to different detected people for MULTI / RELATIONAL queries.

If there are fewer detected persons than query subjects, or localization cannot produce a valid one-to-one assignment, the query is treated as unmatched.

---

## Composition rule

For each localized subject `i`, S10 forms:

```text
q_i = normalize(
    alpha * image_crop_i
    + (1 - alpha) * text_i
)
```

where:

- `image_crop_i` is the normalized CLIP image embedding of the localized person crop;
- `text_i` is the normalized CLIP text embedding used for the requested modification;
- `alpha ∈ [0, 1]`.

### Text-selection rule

For `RELATIONAL` queries:

```text
query.text
```

is used as the composition text for each subject.

For other cases, the fallback order is:

```text
subject.modify_text
→ query.relation_text
→ query.text
```

The first non-empty text is used.

---

## SetMatch rule

Let the composed query-person set be:

```text
Q = {q_1, ..., q_m}
```

and the detected gallery-person set be:

```text
G = {g_1, ..., g_n}
```

Pairwise scores are cosine similarities:

```text
S_ij = q_i^T g_j
```

For `m > 1`, S10 solves:

```text
π* = argmax_π Σ_i S_{i,π(i)}
```

subject to one-to-one Hungarian assignment.

The gallery-image score is then:

```text
score(Q, G) = min_i S_{i,π*(i)}
```

So all requested subjects must have a strong matched person.

If:

```text
n < m
```

the gallery image receives:

```text
-1.0
```

which is the configured unmatched score.

### SINGLE case

For a one-person query:

```text
score(q, G) = max_j q^T g_j
```

This avoids running Hungarian assignment unnecessarily.

---

## Alpha selection

S10 supports two modes.

### Default: fixed alpha

The repository default is:

```yaml
composition:
  alpha_selection:
    mode: fixed
    fixed_alpha: 0.50
```

Therefore the normal baseline is runnable from a clean benchmark checkout without requiring a separate validation split.

Default composition:

```text
q_i = normalize(
    0.50 * image_crop_i
    + 0.50 * text_i
)
```

With this mode:

```text
CPR Supervision: No
```

No CPR positive labels are used to choose `alpha`.

### Optional: validation-selected alpha

If a genuine separate validation split is available, change:

```yaml
composition:
  alpha_selection:
    mode: validation
```

The configured validation manifests are:

```text
data/validation/gallery.jsonl
data/validation/queries.jsonl
```

S10 then searches the configured alpha grid:

```text
0.00, 0.05, 0.10, ..., 0.95, 1.00
```

and selects the value maximizing:

```text
Full-mAP
```

Exact ties are resolved using the smallest alpha.

In this mode:

```text
CPR Supervision: Val only
```

S10 refuses validation tuning if the validation manifests are missing or identical to the canonical evaluation manifests.

The canonical evaluation positives are never used for alpha tuning.

---

## Dependencies

S10 reuses the dependency contract of S5:

```text
-r ../05_reid_set/requirements.txt
```

and additionally installs the pinned OpenAI CLIP repository.

Grounding DINO itself is **not** pip-installed.

This is deliberate.

The upstream Grounding DINO setup attempts to build optional C++/CUDA extensions during wheel installation, which can fail on environments such as Kaggle Python 3.12.

Instead:

```text
download_checkpoint.py
    ↓
S5 checkpoint preparer
    ↓
clone exact Grounding DINO commit
    ↓
best-effort in-place extension build
    ↓
direct source import
    ↓
official PyTorch fallback if custom op unavailable
```

This keeps the detector reproducible without making the benchmark depend on a successful Grounding DINO wheel build.

---

## Checkpoint preparation

S10's:

```text
methods/simple/10_person_compose_setmatch/download_checkpoint.py
```

delegates to two existing preparers.

### 1. Shared Grounding DINO assets

```text
methods/simple/05_reid_set/download_checkpoint.py
```

This prepares:

- pinned Grounding DINO source;
- Grounding DINO Swin-T checkpoint;
- Grounding DINO BERT runtime assets;
- optional Grounding DINO CUDA/C++ extension;
- S5 shared external assets.

S10 does not use the S5 CLIP-ReID features for its own retrieval representation, but it reuses the same S5 artifact-preparation protocol to guarantee a valid shared detector setup.

### 2. CLIP ViT-L/14

```text
methods/simple/02_clip_text/download_checkpoint.py
```

This prepares:

```text
checkpoints/clip/ViT-L-14.pt
```

No model checkpoint is silently downloaded during S10 inference.

---

## Cache design

### Main split

Grounding DINO detections:

```text
runs/groundingdino_clipreid_set/cache/person_detections.npz
```

S10 CLIP person features:

```text
runs/per_person_clip_compose_setmatch/cache/main_person_features.npy
```

### Validation split

Used only when:

```yaml
mode: validation
```

Validation detections:

```text
runs/per_person_clip_compose_setmatch/cache/val_person_detections.npz
```

Validation CLIP person features:

```text
runs/per_person_clip_compose_setmatch/cache/val_person_features.npy
```

### Person-feature fingerprint

The CLIP person-feature cache fingerprint covers:

- S10 adapter version;
- S10 config hash;
- detection-cache hash;
- CLIP checkpoint hash;
- CLIP model name;
- cache schema.

A stale or incompatible cache is ignored and recomputed.

---

## Runtime behavior

CLIP image/text feature extraction runs under:

```python
torch.no_grad()
```

because S10 performs inference only.

This prevents construction of unnecessary autograd graphs and reduces GPU-memory usage during person-feature extraction.

Runtime defaults include:

```yaml
runtime:
  device: cuda
  clip_image_batch_size: 128
  clip_text_batch_size: 256
  num_workers: 4
  feature_cache_dtype: float16
  score_feature_chunk_size: 65536
```

If GPU memory is limited, `clip_image_batch_size` can be reduced without changing method semantics.

---

## Data contract

Canonical manifests:

```text
data/gallery.jsonl
data/queries.jsonl
```

S10 expects query rows to provide:

```text
image_id
text
case
subjects
```

Each subject should provide:

```text
select_text
modify_text
```

For optional validation alpha selection, queries must additionally expose valid:

```text
full_positive_ids
```

The query/reference image remains inside the gallery score matrix.

S10 does not remove it internally.

Self-image exclusion remains the responsibility of the official benchmark evaluator:

```text
evaluate.py
```

---

## Output contract

The inference adapter writes:

```text
runs/per_person_clip_compose_setmatch/scores.npy
runs/per_person_clip_compose_setmatch/run.json
```

The score matrix must satisfy:

```text
scores.shape == (
    number_of_queries,
    number_of_gallery_images,
)
```

All scores must be finite floating-point values.

Higher scores indicate better matches.

---

## Run

From repository root:

```bash
python run_baseline.py person_compose_setmatch
```

Equivalent method id:

```bash
python run_baseline.py per_person_clip_compose_setmatch
```

The root runner performs:

```text
[1/6] Validate gallery data

[2/6] Install requirements
      - install S5 runtime dependencies
      - install OpenAI CLIP
      - do NOT pip-build Grounding DINO

[3/6] Prepare checkpoint
      - pin Grounding DINO source through S5
      - prepare Grounding DINO runtime assets
      - prepare CLIP ViT-L/14

[4/6] Inference
      - import pinned S5 detector protocol
      - configure Grounding DINO source
      - reuse or generate person detections
      - encode detected people with CLIP
      - localize target persons
      - compose image + text
      - SetMatch against gallery persons

[5/6] Official evaluation

[6/6] Build benchmark tables
```

---

## Expected Grounding DINO behavior on Kaggle / Python 3.12

A successful S10 environment should **not** show:

```text
Building wheel for groundingdino
```

during the requirements-install step.

Grounding DINO is instead handled through the pinned checkout prepared by S5.

If the custom Grounding DINO extension cannot be used, inference may report that it is using:

```text
official_pytorch_fallback
```

This is an expected supported backend, not an installation failure.

---

## CPR supervision

Default configuration:

```text
CPR Supervision: No
```

because `alpha=0.50` is fixed.

Only explicit:

```yaml
alpha_selection:
  mode: validation
```

changes the effective supervision to:

```text
CPR Supervision: Val only
```

No canonical evaluation labels are used by the method itself for model fitting or hyperparameter selection.

---

## Important behavioral scope

S10 uses:

- predicted person detections;
- subject selection text;
- modification/instruction text;
- OpenAI CLIP;
- Hungarian assignment;
- fixed or validation-selected composition weight.

S10 does **not** use:

- GT person boxes;
- target identity labels for localization;
- GT subject-to-box mappings;
- a learned CPR binding module;
- a relation classifier;
- training on CPR query-target triplets;
- evaluation-set positives for default hyperparameter selection.

---

## Interpretation

S10 is a direct control baseline for the hypothesis:

```text
Maybe CPR can already be solved by:
detect person
→ identify requested person
→ compose that person's image feature with text
→ explicitly match person sets.
```

If S10 performs strongly, it suggests that explicit person decomposition and CLIP-space composition already explain a meaningful portion of CPR performance.

If S10 remains substantially below learned binding approaches, that supports the need for stronger structured conditioning or learned person-text binding beyond simple feature interpolation and Hungarian matching.
