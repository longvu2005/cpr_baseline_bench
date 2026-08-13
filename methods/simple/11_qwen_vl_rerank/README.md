# S11. Retrieve + Qwen2.5-VL Rerank

## Purpose

S11 is a benchmark-defined **retrieve-then-verify** baseline:

```text
S6 first-stage retrieval
        ↓
full-gallery ranking
        ↓
Qwen2.5-VL verifies only top-K candidates
        ↓
rerank top-K
        ↓
append all remaining gallery images in unchanged S6 order
```

It tests whether a strong generic MLLM can recover the remaining CPR errors by explicitly comparing the reference scene, full instruction, and candidate scene.

This is **not** a reproduction of PinPoint. PinPoint (CVPR 2026) is cited only as evidence that off-the-shelf MLLM reranking is a practical CIR evaluation direction.

## Models

- first-stage retriever: **S6 Grounding DINO + CLIP-ReID - Set + Text**;
- verifier: `Qwen/Qwen2.5-VL-7B-Instruct`;
- Qwen checkpoint/revision is exactly shared with S7.

S11 introduces no learned CPR module.

## Verifier input

For every top-K candidate, Qwen receives:

1. the full canonical query/reference scene;
2. the full candidate gallery scene;
3. the canonical query-level `text` instruction;
4. one fixed verifier prompt shared across SINGLE/MULTI/RELATIONAL.

It does **not** receive `target_ids`, `full_positive_ids`, GT boxes, or GT identity labels during main inference.

The verifier must output:

```text
SCORE=N
```

where `N` is an integer from 0 to 100. Invalid output is treated as an error rather than silently guessed.

## Reranking rule

Qwen's score is normalized as:

```text
v = SCORE / 100
```

The S6 branch is converted to a fixed rank prior inside top-K:

```text
r_j = 1 - j / (K - 1)
```

for zero-based S6 rank `j`; when `K=1`, `r_0=1`.

Top-K candidates are reranked by:

```text
rerank_score = w * r_j + (1 - w) * v_j
```

where `w` is selected only on validation Full-mAP.

The raw scales of S6 and Qwen are therefore never mixed directly.

## Complete full-gallery ranking invariant

S11 never lets a reranked top-K item fall below an item outside top-K.

For every query:

```text
final_order = reranked(top-K) + original_S6_order[K:]
```

The final permutation is converted to strictly descending rank-derived numerical scores before writing `scores.npy`. Therefore:

- every top-K candidate remains in the top-K block;
- images outside top-K keep exactly their original first-stage relative order;
- the evaluator still receives one finite full `queries × gallery` score matrix;
- self-image exclusion remains the evaluator's responsibility.

## Validation-only hyperparameters

```text
CPR Supervision: Val only
```

Both of these are chosen by **validation Full-mAP**:

- `top-K`;
- first-stage fusion weight `w`.

Default grids:

```text
K ∈ {5, 10, 20}
w ∈ {1.0, 0.9, ..., 0.0}
```

Ties are resolved by:

1. smaller K (cheaper);
2. larger first-stage weight (more conservative).

Separate manifests are required:

```text
data/validation/gallery.jsonl
data/validation/queries.jsonl
```

S11 refuses to tune if validation manifests are byte-identical to the main evaluation manifests.

## First-stage requirement

S11 deliberately treats S6 as a completed first-stage system. Run S6 first:

```bash
python run_baseline.py groundingdino_clipreid_set_text
```

S11 then reuses:

- main S6 `scores.npy`;
- S6 validation ReID and CLIP-text branch caches;
- S6 validation-selected alpha to reconstruct the exact validation S6 first-stage scores.

This avoids silently training or changing the first-stage retriever inside S11.

## Qwen verifier cache

Because MLLM verification is expensive, S11 caches Qwen scores.

Validation computes Qwen only for the maximum configured K once, then reuses subsets for all smaller K values.
Main inference computes only the selected K.

Cache fingerprints include:

- main/validation manifests;
- first-stage score artifact hash;
- Qwen prepared-marker hash;
- fixed verifier prompt;
- processor/generation settings;
- requested maximum K.

## Checkpoint preparation

`download_checkpoint.py` delegates to the existing S7 Qwen checkpoint preparer so S7 and S11 share exactly one pinned Qwen snapshot.
No model download occurs in `run.py`.

## Run

Recommended order:

```bash
python run_baseline.py groundingdino_clipreid_set_text
python run_baseline.py retrieve_qwen25vl_rerank
```

## Compute

S11 is intentionally **OPTIONAL / EXPENSIVE**.
With the default search grid, validation verifies only the maximum K=20 once per validation query. After K/w selection, the main benchmark verifies only the selected K candidates per query.

## Interpretation

S11 answers the objection:

```text
Maybe a cheap conventional retriever only needs a strong MLLM verifier on its shortlist.
```

If S11 becomes very strong, retrieval-plus-verification is an important conventional baseline. If it still fails, the remaining difficulty is not simply shortlist semantic verification.
