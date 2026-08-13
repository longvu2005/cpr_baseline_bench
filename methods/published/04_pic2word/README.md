# P4. Pic2Word

## Paper

**Pic2Word: Mapping Pictures to Words for Zero-shot Composed Image Retrieval**, CVPR 2023.

Official implementation:

```text
google-research/composed_image_retrieval
```

This adapter pins the official source at:

```text
8c053297c2fae9cd17ddcded48445a4f47208dbd
```

## What Pic2Word does

Pic2Word maps the reference image into a learned pseudo-word embedding and inserts that pseudo-word into the CLIP text token sequence. CLIP then encodes the resulting image-conditioned text prompt into the retrieval query representation.

For CPR, the adapter deliberately uses the **full query/reference scene** and the **full canonical modification text**. It does not detect persons and does not use SetMatch.

```text
full query/reference scene
        ↓
CLIP ViT-L/14 image encoder
        ↓
learned official IM2TEXT mapping
        ↓
pseudo-word
        ↓
"a photo of * , {full modification text}"
        ↓ replace * with pseudo-word
CLIP text transformer
        ↓
composed query feature
        ×
full-gallery CLIP image features
        ↓
cosine similarity
```

## Official checkpoint and backbone

The official repository releases a pretrained Pic2Word model via Google Drive. The official README's evaluation command uses:

```text
--model ViT-L/14
--openai-pretrained
```

The checkpoint is stored locally as:

```text
checkpoints/pic2word/pic2word_pretrained.pt
```

Checkpoint status:

```text
OFFICIAL_RELEASED
```

`download_checkpoint.py` validates that the downloaded artifact contains both official checkpoint branches:

```text
state_dict
state_dict_img2text
```

and verifies the expected ViT-L/14 and 2-layer IM2TEXT tensor shapes. This prevents a random CLIP checkpoint or unrelated training artifact from being accepted as the released Pic2Word model.

The OpenAI CLIP ViT-L/14 base checkpoint is prepared separately at:

```text
checkpoints/clip/ViT-L-14.pt
```

with the official OpenAI SHA256.

## Why the prompt is fixed this way

The official CIRR code constructs:

```text
a photo of * , {caption}
```

where `*` is the token replaced by the image-derived pseudo-word.

The CPR adapter preserves that official composition format exactly and substitutes the canonical CPR query-level text:

```text
a photo of * , {queries.jsonl["text"]}
```

There is one fixed template for the entire benchmark. No prompt differs by SINGLE, MULTI, or RELATIONAL case.

## CPR adaptation

P4 reads only the canonical benchmark manifests:

```text
data/gallery.jsonl
data/queries.jsonl
```

For each query:

1. read `query.image_id`;
2. map it to the matching canonical gallery row;
3. load that row's full scene image;
4. read the full query-level `text` field;
5. produce the Pic2Word composed feature;
6. score it against every full gallery scene.

The method does **not** consume:

```text
target_ids
full_positive_ids
GT boxes
GT identity labels
subjects[].select_text
subjects[].modify_text
relation_text as a separate routing signal
case annotations for routing
```

The query image is not removed inside the method. `evaluate.py` owns benchmark self-image exclusion.

## SINGLE / MULTI / RELATIONAL

There is no case-specific branch.

- **SINGLE:** full scene + full instruction goes through standard Pic2Word composition.
- **MULTI:** same full scene + full instruction path; there is no person decomposition or SetMatch.
- **RELATIONAL:** same full instruction is passed directly through the same Pic2Word prompt.

This is intentional: P4 measures how far a mature zero-shot CIR method can go **without person-aware adaptation**.

## CPR supervision

```text
CPR Supervision: No
```

No CPR training, validation tuning, checkpoint selection, target localization labels, or evaluation labels are used.

## Source fidelity

Preserved from the official implementation:

- official ViT-L/14 configuration;
- official released pretrained checkpoint;
- official `IM2TEXT` class;
- official CLIP implementation bundled in the repository;
- official image preprocessing;
- official pseudo-word insertion through `encode_text_img_retrieval`;
- official CIRR-style `a photo of * , ...` prompt construction.

Benchmark-only adaptation:

- replace CIRR/Fashion-IQ data loaders with canonical CPR manifests;
- use the full CPR query scene as the Pic2Word reference image;
- use `queries.jsonl["text"]` as the modification caption;
- emit the benchmark-standard full `scores.npy` matrix and `run.json`.

No architecture is rewritten.

## Cache

```text
runs/pic2word/cache/gallery_features.npy
runs/pic2word/cache/query_features.npy
```

Fingerprints include the pinned source commit, Pic2Word checkpoint, OpenAI CLIP checkpoint, config, and canonical manifest hashes. Cache hits are reported explicitly.

## Run

From the repository root:

```bash
python run_baseline.py pic2word
```

The root runner performs:

```text
install requirements
→ pin official source / prepare official checkpoints
→ Pic2Word inference
→ official evaluate.py
→ build_tables.py
```

## Interpretation

P4 answers a clean reviewer question:

```text
How strong is a mature zero-shot composed-image retrieval method when applied directly to the full CPR scene and full instruction, without any person detection, target localization, or SetMatch?
```

A strong P4 result would show that generic language-space recomposition transfers well to CPR. A large gap to person-aware methods would quantify the value of explicit person-level reasoning/binding.
