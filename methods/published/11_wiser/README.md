# P11 — WISER

Paper-faithful CPR adapter for **WISER: Wider Search, Deeper Thinking, and Adaptive Fusion for Training-Free Zero-Shot Composed Image Retrieval (CVPR 2026)**.

## Preserved from the released method

- BLIP2-T5 (`pretrain_flant5xxl`) gallery captions with the released detail-focused prompt.
- BAGEL-7B-MoT edited caption and edited image generation.
- OpenCLIP `ViT-B-32 / laion2b_s34b_b79k` retrieval space.
- Parallel T2I and I2I full-gallery retrieval; top-50 candidates from each branch.
- Qwen2.5-VL-7B strict yes/no verifier, using the released confidence computation.
- Branch confidence threshold `0.7`.
- One released-default refinement loop using GPT-4o and the released structured reflection prompt.
- Final candidate fusion by `c_t2i + c_i2i`, then max branch confidence, then T2I confidence.

## CPR adaptation boundary

Input is the **full canonical query image + `text` instruction**. No person detector, crop, SetMatch, target identity, `target_ids`, or `full_positive_ids` is used.

The WISER release only ranks the verified union of two top-50 branches. The benchmark evaluator requires a complete dense `(num_queries, num_gallery)` `scores.npy`. Therefore:

1. verified WISER candidates stay first in WISER's confidence-fused order;
2. non-candidates are appended by the pre-verifier dual-path base order:
   `(min(t2i_rank, i2i_rank), t2i_rank+i2i_rank, t2i_rank, gallery_index)`;
3. the complete order is converted to monotonic float32 rank scores.

The root evaluator still removes the exact query image itself.

## Two isolated environments

The released step-1 captioner uses Salesforce LAVIS/BLIP2, while the verifier uses modern Qwen2.5-VL. Their Transformers generations conflict, so preparation creates:

- `runs/wiser/.venv` — BAGEL, OpenCLIP, Qwen2.5-VL, adapter runtime.
- `runs/wiser/.venv_caption` — LAVIS BLIP2-T5 only.

This changes dependency isolation, not the retrieval method.

## Run

```bash
export OPENAI_API_KEY='...'
python run_baseline.py wiser
```

The first run downloads/caches large public assets and gallery captions. Expensive generation and verifier artifacts are cached for resume.

## Outputs

```text
runs/wiser/scores.npy
runs/wiser/run.json
outputs/wiser/metrics.json
```

## Public-release compatibility fixes

The inspected WISER revision has several execution bugs/incomplete helper edges. This adapter keeps the released models, prompts, thresholds and fusion equations, while applying only execution-level fixes:

- bypasses undefined enum references in the released verifier/refiner helper classes;
- uses the post-refinement verifier candidates for final fusion (the public source computes them but returns stale pre-refinement candidate variables);
- isolates the LAVIS caption stack from the modern Qwen/BAGEL stack.

Every boundary/fix is written to `runs/wiser/run.json`.
