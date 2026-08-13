# P5. LinCIR

## Paper

**Language-only Training of Zero-shot Composed Image Retrieval**, CVPR 2024.

Official code:

```text
https://github.com/navervision/lincir
```

This adapter pins the official source at:

```text
1dec42d118da816be2a43fd43cc0746e05f63881
```

## Reproducible checkpoint choice

The benchmark intentionally freezes the official **ViT-L / `large`** distribution instead of attempting to
reconstruct a larger ViT-G experiment with incompatible weights.

Official model distribution:

```text
repo:     navervision/zeroshot-cir-models
revision: 5118df4683de6efa09f8e5336d86ca626ed21c44
file:     lincir_large.pt
SHA256:   5ba98d52db7a5e9a78f9ba7436991235477b89544caf246e4659642bf9b409bd
size:     56,653,151 bytes
```

The official LinCIR README links this exact file for the ViT-L demo/application.

## Backbone

Official `large` resolves to:

```text
openai/clip-vit-large-patch14
```

The benchmark pins the Hugging Face CLIP snapshot at:

```text
32bd64288804d66eefd0ccbe215aa642df71cc41
```

and downloads only the inference files required by the vision/text encoders.

## CPR adaptation

P5 is a **direct full-scene CIR** adapter.

It does not use person detection, target localization, SetMatch, GT boxes, identities, or CPR labels.

```text
full query/reference scene
        ↓
CLIP ViT-L/14 vision encoder
        ↓
raw image projection feature
        ↓
official LinCIR Phi
        ↓
pseudo text token
        +
full canonical modification instruction
        ↓
CLIP text encoder
        ↓
normalized composed query feature
        ×
normalized full-scene gallery CLIP features
        ↓
cosine similarity
        ↓
scores.npy
```

### Fixed prompt

The official CIRR evaluation constructs:

```text
a photo of $ that {relative_caption}
```

P5 keeps this template exactly and substitutes the canonical CPR query text:

```text
a photo of $ that {queries.jsonl["text"]}
```

There is exactly one pseudo-token placeholder `$`.

## Official pseudo-token pathway

The official source uses:

```text
image_features → Phi → pseudo token
```

and `encode_with_pseudo_tokens_HF` replaces the CLIP token embedding at token id `259` with the Phi output.
The wrapper imports the official `Phi` implementation and official pseudo-token text encoder directly from the
pinned source checkout.

The adapter verifies at runtime that OpenAI CLIP tokenization still maps `$` to id `259` before inference.

## Normalization

The official LinCIR validation option `--l2_normalize` is **off by default**. Therefore P5 does **not** normalize
CLIP image features before passing them through Phi.

After composition:

- composed query embeddings are L2-normalized;
- gallery image embeddings are L2-normalized;
- retrieval score is their dot product / cosine similarity.

No CPR-specific fusion weight or validation tuning is introduced.

## CPR supervision

```text
No
```

Inference never reads:

```text
target_ids
full_positive_ids
subjects
case
GT target boxes
GT identity mappings
```

The only query-specific benchmark inputs are:

```text
image_id
text
```

`image_id` resolves the full reference scene through the canonical gallery manifest.

## Canonical benchmark contract

P5:

- reads `data/gallery.jsonl` and `data/queries.jsonl` in exact row order;
- scores every query against every gallery row;
- does not remove the reference image itself;
- writes only finite `scores.npy` values;
- lets the benchmark evaluator own self-image exclusion and metrics.

Expected raw output:

```text
runs/lincir/
├── scores.npy
└── run.json
```

## Offline preparation

`download_checkpoint.py` performs all networked preparation before inference:

1. pin the official LinCIR source checkout;
2. download and checksum the exact `lincir_large.pt` release;
3. snapshot the exact CLIP ViT-L/14 inference files;
4. validate Phi tensor shapes and CLIP weight checksum;
5. write a reproducibility marker.

`run.py` sets Hugging Face offline mode and refuses missing runtime artifacts.

## Cache

```text
runs/lincir/cache/gallery_features.npy
runs/lincir/cache/query_features.npy
```

Cache fingerprints include the canonical manifests, adapter version, source commit, official LinCIR checkpoint
hash, pinned CLIP weight hash, prompt template, and preprocessing/model configuration.

## Run

From repository root:

```bash
python run_baseline.py lincir
```

The normal root runner installs requirements first, prepares all external assets, runs inference, evaluates the
full score matrix, and rebuilds benchmark tables.
