# S7. Qwen2.5-VL Rewrite + CLIP ViT-L/14

## Purpose

S7 tests whether a strong frozen multimodal language model can solve the hard part of CPR by converting the composed query into ordinary language-space retrieval.

The pipeline is:

```text
full query/reference scene + canonical modification/query text
                    ↓
       frozen Qwen2.5-VL-7B-Instruct
                    ↓
     one rewritten target-image caption
                    ↓
          OpenAI CLIP ViT-L/14
                    ↓
         global gallery retrieval
```

This is a **simple benchmark pipeline**, not an exact reproduction of a published CPR method. Conceptually it belongs to language-space query recomposition.

## Components

### Frozen MLLM

- model: `Qwen/Qwen2.5-VL-7B-Instruct`;
- official source: Qwen Hugging Face repository;
- pinned revision: `cc594898137f460bfe9f0759e9844b3ce807cfb5`;
- license: Apache-2.0;
- inference API: Hugging Face Transformers `Qwen2_5_VLForConditionalGeneration` + `AutoProcessor`;
- inference memory mode: bitsandbytes 4-bit NF4 weights with double quantization;
- non-quantized / compute dtype: FP16, fixed in `config.yaml`;
- the pinned Qwen checkpoint and architecture are unchanged; quantization is applied only
  when the frozen model is loaded for inference so S7 fits a 16 GB-class CUDA GPU;
- no fine-tuning or CPR supervision.

The full official snapshot is prepared under:

```text
checkpoints/qwen/Qwen2.5-VL-7B-Instruct/
```

`download_checkpoint.py` validates the required five safetensor shards and processor/tokenizer files, and writes a local inventory marker. `run.py` validates that inventory before inference and then forces Hugging Face/Transformers offline mode.

### Retriever

- model: OpenAI CLIP ViT-L/14;
- checkpoint: official `ViT-L-14.pt`;
- query representation: CLIP text embedding of the Qwen-generated target caption;
- gallery representation: CLIP image embedding of the complete canonical scene;
- score: cosine similarity between L2-normalized embeddings.

The gallery feature cache intentionally uses the same fingerprint schema and path as the existing ViT-L/14 CLIP baselines:

```text
runs/clip_image/gallery_features_vit_l14.npy
```

so a valid cache can be reused across methods.

## Fixed prompt template

One prompt is used for **every** query and every case type. There are no SINGLE/MULTI/RELATIONAL-specific prompts.

The canonical field `queries.jsonl["text"]` is inserted only into the `{modification}` slot:

```text
You are rewriting a composed person-retrieval query for image retrieval.
The provided image is the full reference scene. The modification text describes the desired target image relative to that reference.
Write exactly one concise, standalone English sentence describing the target image that should be retrieved.
Preserve visually useful appearance details of the relevant person or people from the reference scene when they remain applicable, and apply the requested modification faithfully.
Include multiple people and their relationship when the request requires them.
Describe only visible target-image content. Do not mention the reference image, the query, the modification, reasoning, uncertainty, or these instructions.
Output only the rewritten target-image description, with no label, preface, bullet, or explanation.

Modification text: {modification}
```

Generation is deterministic:

```text
do_sample = false
num_beams = 1
max_new_tokens = 96
repetition_penalty = 1.0
```

The processor uses the same fixed pixel budget for every query:

```text
min_pixels = 256 * 28 * 28
max_pixels = 1280 * 28 * 28
```

These are runtime choices fixed in `config.yaml`; they are not tuned on CPR labels.

## Canonical data contract

S7 reads only:

```text
data/gallery.jsonl
data/queries.jsonl
```

For each query:

1. read `query.image_id`;
2. map it to the exact canonical gallery row;
3. load the full image from that gallery row's `path`;
4. read the canonical query-level `text` field;
5. send only that image and text through the fixed Qwen prompt.

The adapter does **not** consume:

```text
target_ids
full_positive_ids
person_ids as supervision
GT boxes
case annotations for routing
```

`person_ids` may exist in the gallery manifest because it is part of the benchmark schema, but S7 never reads it for inference.

The final artifact remains the full canonical matrix:

```text
scores.shape == (len(queries), len(gallery))
```

The method does not remove the query image. The official evaluator owns self-image exclusion.

## Rewrite cache

Qwen inference is expensive, so generated captions are cached at:

```text
runs/qwen25vl_rewrite_clip/cache/rewritten_queries.jsonl
```

The cache fingerprint covers:

- adapter version;
- complete S7 config;
- canonical gallery/query manifests;
- pinned Qwen prepared-marker inventory;
- fixed prompt template;
- processor pixel settings;
- deterministic generation settings.

If any of these changes, the cache is rejected and Qwen rewrites the queries again.

CLIP embeddings of the rewritten captions are cached separately at:

```text
runs/qwen25vl_rewrite_clip/cache/rewritten_text_features.npy
```

## CPR supervision

```text
CPR Supervision: No
```

There is no CPR training, validation tuning, prompt selection by benchmark score, checkpoint selection, or target-label use. The one prompt template is fixed before benchmark evaluation.

## Case behavior

S7 has no explicit SetMatch or case-specific module:

- **SINGLE:** Qwen rewrites the scene + text into one target caption, then CLIP retrieves globally.
- **MULTI:** the same prompt asks Qwen to describe all required people when the modification requires them.
- **RELATIONAL:** the same prompt asks Qwen to preserve visible relationships when the modification requires them.

This is deliberately a test of whether generic multimodal understanding + language rewrite is already sufficient without a CPR-specific architecture.

## Checkpoint preparation

The normal root runner installs requirements first, then `download_checkpoint.py`:

1. downloads the exact pinned Qwen Hugging Face snapshot;
2. validates required model/processor/tokenizer files;
3. writes the prepared marker;
4. downloads/checksums OpenAI CLIP ViT-L/14.

No model or tokenizer is silently downloaded in `run.py`.

## Run

From repository root:

```bash
python run_baseline.py qwen25vl_rewrite_clip
```

The root runner performs:

```text
requirements
→ checkpoint preparation
→ Qwen rewrite / cache
→ CLIP retrieval
→ official evaluate.py
→ build_tables.py
```

## Interpretation

S7 asks a different question from S5/S6:

```text
Can a strong frozen MLLM understand the reference scene and modification well enough
that ordinary CLIP text-to-image retrieval becomes sufficient after rewriting?
```

If S7 performs strongly, a substantial part of CPR may be solvable through generic multimodal query understanding and language-space recomposition. If it remains clearly below a CPR-specific method, the result supports the need for explicit person-level composition, binding, or structured retrieval rather than caption rewriting alone.
