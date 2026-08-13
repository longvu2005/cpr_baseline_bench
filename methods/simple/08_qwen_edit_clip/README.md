# S8. Qwen-Image-Edit-2509 + CLIP ViT-L/14

## Purpose

S8 tests direct **pixel-space composition** instead of text-space or embedding-space composition.
A frozen generative image editor transforms the canonical query/reference scene into one synthetic target image according to the canonical modification text. That synthetic image is then used as the image query for CLIP retrieval.

The pipeline is:

```text
full query/reference scene + canonical modification/query text
                    ↓
        frozen Qwen-Image-Edit-2509 image editor
                    ↓
          one synthetic target/query image
                    ↓
             OpenAI CLIP ViT-L/14
                    ↓
            global gallery retrieval
```

This is a **simple benchmark pipeline**, not an exact reproduction of a published CPR method.
Conceptually it asks whether direct image editing of the query scene already solves a large fraction of CPR.

## Components

### Frozen image editor

- model: `Qwen/Qwen-Image-Edit-2509`;
- official source: Qwen Hugging Face repository;
- expected pipeline class: `QwenImageEditPlusPipeline`;
- intended use: image editing from reference image + instruction text;
- no fine-tuning or CPR supervision.

The prepared local snapshot lives under:

```text
checkpoints/qwen/Qwen-Image-Edit-2509/
```

`download_checkpoint.py` downloads the full Hugging Face snapshot, inventories it, and writes a prepared marker. `run.py` validates that marker and then forces Hugging Face/Transformers offline mode before inference.

### Retriever

- model: OpenAI CLIP ViT-L/14;
- checkpoint: official `ViT-L-14.pt`;
- query representation: CLIP image embedding of the synthetic edited image;
- gallery representation: CLIP image embedding of the complete canonical scene;
- score: cosine similarity between L2-normalized embeddings.

The gallery feature cache intentionally uses the same fingerprint schema and path as the existing ViT-L/14 CLIP baselines:

```text
runs/clip_image/gallery_features_vit_l14.npy
```

so a valid cache can be reused across methods.

## Fixed edit instruction template

One prompt template is used for **every** query and every case type. There are no SINGLE/MULTI/RELATIONAL-specific prompts.

The canonical field `queries.jsonl["text"]` is inserted only into the `{modification}` slot:

```text
You are editing a reference scene into the desired target image for person retrieval.
Apply the requested modification faithfully.
Preserve the identity and visually useful appearance of the relevant person or people whenever they are not supposed to change.
Keep all required people present, preserve or update their relationship when the request requires it, and avoid adding unrelated extra people.
Produce a realistic edited image of the target scene.

Modification text: {modification}
```

The benchmark also fixes one negative prompt to discourage common failure modes:

```text
blurry, low quality, distorted face, wrong identity, duplicate person, extra person,
missing person, malformed hands, malformed body, incorrect relation, text, watermark
```

Generation is fixed and not tuned on CPR labels:

```text
num_inference_steps = 30
guidance_scale = 4.0
seed = 42
```

## Canonical data contract

S8 reads only:

```text
data/gallery.jsonl
data/queries.jsonl
```

For each query:

1. read `query.image_id`;
2. map it to the exact canonical gallery row;
3. load the full image from that gallery row's `path`;
4. read the canonical query-level `text` field;
5. send only that image and text through the fixed edit prompt.

The adapter does **not** consume:

```text
target_ids
full_positive_ids
person_ids as supervision
GT boxes
case annotations for routing
```

The final artifact remains the full canonical matrix:

```text
scores.shape == (len(queries), len(gallery))
```

The method does not remove the query image. The official evaluator owns self-image exclusion.

## Edited-image cache

Image editing is expensive, so generated synthetic query images are cached at:

```text
runs/qwen_image_edit_clip/cache/edited_queries/
runs/qwen_image_edit_clip/cache/edited_queries.jsonl
```

The cache fingerprint covers:

- adapter version;
- complete S8 config;
- canonical gallery/query manifests;
- pinned generator prepared-marker inventory;
- fixed prompt template;
- fixed negative prompt;
- generation settings.

If any of these change, the cache is rejected and the edited images are regenerated.

CLIP embeddings of those edited images are cached separately at:

```text
runs/qwen_image_edit_clip/cache/edited_query_features.npy
```

## CPR supervision

```text
CPR Supervision: No
```

There is no CPR training, validation tuning, checkpoint selection by benchmark score, or target-label use.

## Case behavior

S8 has no explicit SetMatch or case-specific module:

- **SINGLE:** edit the full query scene into the desired target scene, then retrieve with CLIP image-to-image similarity.
- **MULTI:** the same prompt asks the editor to keep all required people present when the modification requires them.
- **RELATIONAL:** the same prompt asks the editor to preserve or update relations in the scene when the modification requires them.

This is deliberately a test of whether a strong generic image editor can perform CPR composition directly in pixel space.

## Failure modes to observe

S8 is especially interesting because it can fail in visible ways. Important failure classes to inspect are:

- **identity drift:** the edited person no longer matches the intended identity;
- **missing person:** a required person disappears;
- **hallucinated extra person:** the editor inserts an unrelated person;
- **relation error:** the scene violates the requested relation;
- **global over-editing:** the whole scene changes too much relative to the query image.

These are expected analysis targets when comparing S8 to embedding-space baselines.

## Checkpoint preparation

The normal root runner installs requirements first, then `download_checkpoint.py`:

1. downloads the Qwen image-edit snapshot;
2. inventories the files and writes the prepared marker;
3. downloads/checksums OpenAI CLIP ViT-L/14.

No model or tokenizer is silently downloaded in `run.py`.

## Run

From repository root:

```bash
python run_baseline.py qwen_image_edit_clip
```

The root runner performs:

```text
requirements
→ checkpoint preparation
→ synthetic image generation / cache
→ CLIP retrieval
→ official evaluate.py
→ build_tables.py
```

## Interpretation

S8 asks a different question from S5/S6/S7:

```text
Can a strong frozen image editor directly compose the target image in pixel space well enough
that ordinary CLIP image retrieval becomes sufficient?
```

If S8 performs strongly, this supports the view that generic image-editing composition can already capture much of CPR. If it performs weakly or fails visibly through identity drift, missing people, or wrong relations, that supports the need for explicit person-aware retrieval and structured composition rather than pure generative editing.
