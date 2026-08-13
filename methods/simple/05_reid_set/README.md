# S5. Grounding DINO + CLIP-ReID - Set

## Purpose

This is the deliberately simple, text-free reviewer baseline for the question:

> Can CPR be solved by detecting the people in the scene and re-identifying them independently?

It does **not** use CPR supervision and it does **not** use the modification text at inference time.

## Components

### Person detector

- Model: Grounding DINO Swin-T.
- Official repository: `IDEA-Research/GroundingDINO`.
- Pinned source commit: `856dde20aee659246248e20734ef9ba5214f5e44`.
- Official config: `groundingdino/config/GroundingDINO_SwinT_OGC.py`.
- Official released checkpoint: `groundingdino_swint_ogc.pth`.
- Checkpoint status: `OFFICIAL_RELEASED`.
- Detection prompt: `person`.
- Adapter thresholds: box `0.35`, text `0.25`, followed by class-agnostic NMS at IoU `0.50`.

The thresholds are fixed adapter settings; they are not tuned on CPR labels.

### Person ReID

- Model: CLIP-ReID ViT-B/16.
- Training dataset: MSMT17.
- Paper: *CLIP-ReID: Exploiting Vision-Language Model for Image Re-Identification without Concrete Text Labels*, AAAI 2023.
- Official repository: `Syliz517/CLIP-ReID`.
- Pinned source commit: `eb1898b72c882875f478bebfc6d41644eece0a5d`.
- Official config: `configs/person/vit_clipreid.yml`.
- Official evaluation example checkpoint name: `ViT-B-16_60.pth`.
- Official trained-model table: `ViT-CLIP-ReID / MSMT17`.
- Checkpoint status: `OFFICIAL_RELEASED`.

The adapter uses the same feature returned by the official CLIP-ReID evaluation path: the concatenation of the pre-BN ViT feature (768-D) and projected CLIP feature (512-D), for 1280 dimensions total. The final person embedding is L2-normalized before cosine similarity.

## CPR adaptation

For every canonical gallery image:

1. Grounding DINO predicts all `person` instances.
2. Overlapping duplicate boxes are removed with NMS.
3. Every predicted person crop is resized and normalized with the official CLIP-ReID test preprocessing.
4. CLIP-ReID encodes every crop.

The query image is identified only by its canonical `image_id`; because every query/reference image is also a gallery row, the exact same cached detections and embeddings are reused for the query person set.

No `subjects`, `select_text`, `modify_text`, `relation_text`, `text`, `target_ids`, positives, GT boxes, or GT identity-to-box mapping is used for retrieval.

## SetMatch rule

Let a query/reference image contain predicted person features

`Q = {q_1, ..., q_m}`

and a gallery image contain

`G = {g_1, ..., g_n}`.

The person similarity matrix is cosine similarity:

`S_ij = q_i^T g_j`.

When `n >= m`, the adapter computes a maximum-weight one-to-one Hungarian assignment using `S` and then applies strict minimum aggregation:

`score(Q, G) = min_i S_{i, pi(i)}`.

This is an AND-style set score: every query person must have a strong assigned match.

When the gallery has fewer detected persons than the query (`n < m`), or when the query detector returns no persons (`m = 0`), the score is the explicit unmatched score `-1.0`.

Extra gallery persons are allowed and are left unmatched.

## SINGLE / MULTI / RELATIONAL behavior

- `SINGLE`: still uses **all predicted persons in the reference image**, not a text-selected subject. This is intentional because S5 tests a pure detect+ReID reduction.
- `MULTI`: same person-set construction; maximum-weight Hungarian assignment followed by strict minimum.
- `RELATIONAL`: relation text is ignored. The method can succeed only when person identity/set evidence alone is sufficient.

## Caches

The expensive work is cached under:

```text
runs/groundingdino_clipreid_set/cache/
├── person_detections.npz
├── person_detections.npz.meta.json
├── person_features.npy
└── person_features.npy.meta.json
```

Detection cache validity is keyed by the canonical gallery manifest, adapter config, pinned Grounding DINO source/config, and detector checkpoint.

Feature cache validity additionally depends on the exact detection cache, pinned CLIP-ReID source, CLIP-ReID checkpoint, OpenAI CLIP ViT-B/16 backbone, and ReID preprocessing settings. Shape-only cache reuse is not accepted.

## External artifact preparation

`download_checkpoint.py` prepares all networked artifacts before inference:

- pinned Grounding DINO source checkout (imported directly by `run.py`; it is not built as a pip wheel);
- pinned CLIP-ReID source checkout;
- official Grounding DINO Swin-T checkpoint;
- official MSMT17 CLIP-ReID ViT-B/16 checkpoint;
- exact OpenAI CLIP ViT-B/16 backbone used by CLIP-ReID;
- `bert-base-uncased` tokenizer/model assets required by Grounding DINO.

Grounding DINO's Hugging Face assets are stored in a repository-local cache. `run.py` enables Hugging Face/Transformers offline mode before loading Grounding DINO, so inference does not silently download runtime artifacts.

The pinned official CLIP-ReID code normally downloads the OpenAI CLIP backbone during model construction. This adapter replaces only that download helper with the already-prepared local `ViT-B-16.pt`; the official CLIP builder, CLIP-ReID architecture, checkpoint, preprocessing, and evaluation feature are preserved.

### CLIP-ReID config compatibility guard

The pinned official `configs/person/vit_clipreid.yml` contains a bare `DATASETS:` placeholder. PyYAML parses that placeholder as `null`, while the official YACS defaults define `DATASETS` as a nested config node. Direct `merge_from_file()` therefore fails on current YACS with a `CfgNode`/`NoneType` mismatch.

The adapter does **not** edit the pinned official checkout. It permits exactly the known top-level `DATASETS: null` placeholder from the pinned file, removes only that placeholder before the YACS merge, merges every real official setting unchanged, and then sets the MSMT17 dataset name required to construct the released model. Any drift from that exact placeholder shape fails immediately instead of being silently sanitized.

`run.py` also performs a CLIP-ReID preflight before the expensive Grounding DINO pass. The preflight loads the exact pinned config/model/checkpoint, verifies that the adapter still matches the pinned official ViT-B/16 recipe (`256x128`, stride `16`, mean/std `0.5`, pre-BN evaluation feature, no SIE), runs one real CUDA forward pass, verifies the expected finite `1280`-D feature, and synchronizes CUDA so asynchronous kernel failures surface immediately. Import/config/checkpoint/forward incompatibilities therefore fail before person detection starts.

The detector cache fingerprint remains tied to the unchanged detector adapter version, so applying this ReID-only compatibility fix does not invalidate an already completed `person_detections.npz` cache. ReID encoding also has CUDA-OOM backoff: it starts from the configured batch size and halves only the active runtime batch on OOM, without changing the config or detector-cache identity.

Before an expensive rerun, use the explicit smoke test:

```bash
python methods/simple/05_reid_set/run.py --preflight-only
```

This command never runs Grounding DINO detection, full ReID feature extraction, SetMatch scoring, or evaluation. It reports whether the existing `person_detections.npz` will be a cache `HIT` under the exact current fingerprint. If it reports `MISS`, do not start the full baseline unless re-running detection is intentional.

For recovery after a completed detector pass, the safest direct inference command is:

```bash
python methods/simple/05_reid_set/run.py --require-detection-cache
```

With this guard enabled, `run.py` aborts rather than silently spending GPU time on Grounding DINO if the exact cache becomes missing or stale. After direct inference, run the normal evaluator and table builder manually.

## Run

From the repository root:

```bash
python run_baseline.py reid_set
```

The root runner will execute:

```text
1. install this method's requirements
2. run download_checkpoint.py
3. run run.py
4. run evaluate.py --method groundingdino_clipreid_set
5. rebuild benchmark tables
```

For a prepared environment only:

```bash
python run_baseline.py reid_set --skip-install
```

To force public checkpoint/runtime-asset refresh:

```bash
python run_baseline.py reid_set --force-checkpoint
```

## Expected benchmark outputs

```text
runs/groundingdino_clipreid_set/scores.npy
runs/groundingdino_clipreid_set/run.json
outputs/groundingdino_clipreid_set/metrics.json
outputs/groundingdino_clipreid_set/run.json
```

`scores.npy` keeps the full canonical `(num_queries, num_gallery)` ordering and includes the query image itself. Query-image exclusion remains the responsibility of the official `evaluate.py`.

## Important runtime note

The pinned official CLIP-ReID ViT implementation constructs parts of the model directly on CUDA. Therefore this adapter intentionally requires a CUDA runtime rather than pretending to support CPU/MPS through an unverified rewrite of the official implementation.

Grounding DINO is imported directly from the pinned official checkout instead of being installed as a wheel. During artifact preparation, `download_checkpoint.py` makes a best-effort in-place build of the optional CUDA/C++ `_C` extension; build failure is non-fatal. If `_C` is usable, inference uses it. Otherwise `run.py` switches only the deformable-attention call to Grounding DINO's own pure-PyTorch fallback implementation. The selected backend is recorded in the detection-cache fingerprint and `run.json`, so caches are not silently mixed across backends.
