# P6 — MagicLens Large

Published CPR baseline using the official JAX/Flax implementation and released **MagicLens Large** checkpoint.

## Reference

- **Paper:** *MagicLens: Self-Supervised Image Retrieval with Open-Ended Instructions*, ICML 2024 Oral.
- **Training data:** 36.7M `(query image, instruction, target image)` triplets, as stated by the official project.
- **Official repository:** `google-deepmind/magiclens`.
- **Pinned MagicLens source:** `a296f807d49912790cb2d915673ccab3e78df8b0`.
- **Checkpoint:** `magic_lens_clip_large.pkl`.
- **Checkpoint status:** `OFFICIAL_RELEASED`.
- **Checkpoint variant:** the official repository's converted JAX/Flax Large checkpoint. The upstream README notes that converted weights can differ slightly from the original model's reported numbers.
- **Backbone:** CLIP ViT-L/14.
- **Embedding dimension:** 768.

The checkpoint-release commit `c0770efa6d29f125ee3600fbd8b62bc66127aa04`
did not yet contain `data_utils.py`, although `inference.py` imported it. The
adapter therefore pins the first inference-complete official revision,
`a296f807d49912790cb2d915673ccab3e78df8b0`, which adds `data_utils.py` and
README edits only. `model.py`, `layers.py`, and `inference.py` are byte-identical
between those two commits, so this source correction does not alter the model
architecture, Flax parameter tree, checkpoint restoration, or scoring math.

MagicLens upstream depends on Scenic but does not pin a Scenic revision. For reproducible benchmark execution, this adapter pins Scenic commit `e08103067d2033470e5d072a0f4117a02f6f9a4a`, the latest Scenic commit preceding the **2024-05-28 official checkpoint open-source commit**. This is deliberately tied to the checkpoint-release era rather than a later Scenic update. OpenAI CLIP tokenizer source is pinned to `a1d071733d7111c9c014f024669f959182114e33`.

## What is preserved from official MagicLens

The adapter preserves the official retrieval semantics:

1. Instantiate `MagicLens("large")`.
2. Restore the official Flax checkpoint with `flax.serialization.from_bytes`.
3. Gallery/index images are paired with the CLIP tokenization of the empty string.
4. Query images are paired with the textual retrieval instruction.
5. Both sides use `multimodal_embed_norm`.
6. Retrieval score is the dot product of the normalized embeddings, i.e. cosine similarity.
7. Image preprocessing matches `magiclens/data_utils.py::process_img`: RGB conversion, per-image max scaling, and JAX bilinear resize to `224×224`, followed by MagicLens' own CLIP preprocessing.

The code uses `jax.jit` and fixed-size padded runtime batches only as execution optimizations. Valid examples are sliced back to their canonical batch length; the model inputs and score definition are unchanged.

## CPR adaptation

This is intentionally a **direct full-scene baseline**, not a person-localized or SetMatch method.

For each canonical CPR query:

```text
query image       = the full scene identified by queries.jsonl["image_id"]
query instruction = the full queries.jsonl["text"] retrieval instruction
```

For every canonical gallery row:

```text
gallery image       = full scene
gallery instruction = ""
```

Then:

```text
q = MagicLens(query_scene, query.text)["multimodal_embed_norm"]
g = MagicLens(gallery_scene, "")["multimodal_embed_norm"]
score(q, g) = q @ g
```

There is **no** detector, box crop, identity supervision, target-id lookup, positive-label use, target localization, or SetMatch. `SINGLE`, `MULTI`, and `RELATIONAL` queries all use the same scene-level MagicLens rule. This limitation is deliberate: adding person matching would change the published method into a different adapter.

The method also does **not** remove the query image from the score matrix. The repository evaluator owns query-image exclusion.

## Text length

The official Scenic/OpenAI CLIP tokenizer has a context length of 77 tokens. This adapter keeps `tokenizer_truncate: false` by default. If a CPR instruction is too long, inference fails with the exact query row instead of silently changing the instruction by truncation.

## Environment

MagicLens is a JAX/Flax method and should preferably run in a dedicated environment. The requirements pin a 2024-compatible stack:

```text
JAX 0.4.30
Flax 0.8.5
Optax 0.2.3
Orbax Checkpoint 0.6.4
Chex 0.1.86
NumPy 2.0.2
```

JAX 0.4.30 supports NumPy 2.0. Keeping NumPy 2.0.2 avoids downgrading Kaggle's
shared notebook ABI and prevents conflicts with its OpenCV, CuPy, rasterio and
other NumPy-2-dependent packages.

On Linux x86_64 the requirements install `jax[cuda12]==0.4.30`. Other platforms receive CPU JAX. A CUDA GPU is strongly recommended for the full benchmark; CPU execution is functional but slow.

Scenic imports `tensorflow.io.gfile` in download helpers even though MagicLens inference does not use those helpers. `run.py` supplies a minimal filesystem-only import shim so this baseline does not need to install TensorFlow. Similarly, only OpenAI CLIP's pinned `simple_tokenizer.py` is exposed, avoiding an unnecessary PyTorch CLIP runtime. These shims do not alter model or tokenizer math.

## Checkpoint preparation

The root runner calls `download_checkpoint.py` before inference. The preparer:

1. clones and pins MagicLens, Scenic, and OpenAI CLIP source checkouts;
2. tries the official GCS HTTPS object;
3. if needed, tries authenticated `gsutil` / `gcloud storage` using the official `gs://gresearch/magiclens/models/...` path;
4. falls back to the official Google Drive folder via `gdown`;
5. writes a marker containing the actual checkpoint SHA256, size, source commits, and tokenizer-BPE SHA256.

The official MagicLens README notes that GCS access may require `gcloud auth login` with a Google account. If automatic download is unavailable, place:

```text
magic_lens_clip_large.pkl
```

at:

```text
checkpoints/magiclens/magic_lens_clip_large.pkl
```

and rerun the normal command. The preparer will validate the local artifact and create the reproducibility marker.

## Run

From the repository root:

```bash
python run_baseline.py magiclens_large
```

Discovery can be checked with:

```bash
python run_baseline.py --list
```

To re-run artifact preparation:

```bash
python run_baseline.py magiclens_large --force-checkpoint
```

For debugging in an already prepared dedicated environment:

```bash
python run_baseline.py magiclens_large --skip-install
```

## Outputs

Raw method outputs follow the repository-wide contract:

```text
runs/magiclens_large/scores.npy
runs/magiclens_large/run.json
```

The score matrix has shape:

```text
(num_queries, num_gallery)
```

with higher score meaning a better match. `evaluate.py` and `build_tables.py` remain the only benchmark evaluation/table stages.

## Cache correctness

Gallery/query feature caches are keyed by the adapter version, full config SHA256, canonical manifest SHA256, official checkpoint SHA256, pinned source commits, preprocessing settings, and query composition settings. A stale cache is rejected rather than reused by shape alone.
