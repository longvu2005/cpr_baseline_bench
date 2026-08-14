# P7 — BGE-VL-MLLM-S1

Official zero-shot full-scene adapter for **BGE-VL-MLLM-S1**, released with
*MegaPairs: Massive Data Synthesis for Universal Multimodal Retrieval*
(ACL 2025 Oral).

## Provenance

- Paper: <https://arxiv.org/abs/2412.14475>
- Official code: <https://github.com/VectorSpaceLab/MegaPairs>
- Pinned code revision: `00a74a776e935b58fa95015b2aff48119c43df30`
- Official model: <https://huggingface.co/BAAI/BGE-VL-MLLM-S1>
- Pinned model revision: `455ac20c111813fbb263dd0f22d47d173a971582`
- Checkpoint status: `OFFICIAL_RELEASED`
- License: MIT

S1 is used deliberately: it is trained exclusively on MegaPairs. S2 adds an
epoch of MMEB fine-tuning, so S1 is the cleaner no-CPR-supervision transfer
baseline for this benchmark.

## What remains official

The adapter follows the official MLLM composed-image-retrieval example:

1. Load `BAAI/BGE-VL-MLLM-S1` with `AutoModel.from_pretrained(...,
   trust_remote_code=True)` using the pinned local snapshot.
2. Format query inputs with `q_or_c="q"` and the official CIR task instruction.
3. Format image-only gallery candidates with `q_or_c="c"`.
4. Use the last-token hidden state as the 4096-dimensional embedding.
5. L2-normalize both sides and score with their dot product (cosine similarity).

`output_hidden_states=False` and `use_cache=False` are inference-only memory
optimizations. The pinned remote model returns its final hidden-state tensor
directly, so they do not change the embedding used by the official example.

## CPR adaptation

| Side | BGE-VL input |
| --- | --- |
| Query | Full canonical query scene plus the complete `queries.jsonl["text"]` instruction |
| Gallery | Full canonical gallery scene, image only |
| Score | Normalized query embedding × normalized candidate embedding |

There is no detector, person crop, SetMatch, CPR training, checkpoint selection,
or label-driven localization. `target_ids`, positives, case annotations and GT
boxes are never consumed. Every query is scored against the complete canonical
gallery; query-image exclusion remains the evaluator's responsibility.

SINGLE, MULTI and RELATIONAL rows all use exactly the same full-scene query and
full textual instruction. This is intentional: the model is being measured as
a universal composed retrieval encoder, not extended with a CPR-specific set
module.

## Resources

The official safetensors index describes 15,132,528,640 bytes of tensor data
across four shards (about 15.1 GB before cache/overhead). Defaults use FP16,
SDPA and batch size 1. A CUDA GPU with roughly 24 GB VRAM is the practical
minimum; more headroom is preferable. Gallery encoding is the expensive phase,
but normalized gallery/query features are cached with config, manifest, model
revision and adapter fingerprints.

If a CUDA OOM occurs, keep both encoding batch sizes at `1`. Reducing the score
batch size only affects the much cheaper final matrix multiplication.

## Run

Use a dedicated environment because the official code recommends exactly
`transformers==4.45.2`:

```bash
conda create -n bge-vl-cpr python=3.11 -y
conda activate bge-vl-cpr
cd cpr_baseline_bench
python run_baseline.py bge_vl_mllm_s1
```

The root runner installs dependencies, downloads and validates the pinned
official Hugging Face snapshot, runs offline inference, invokes the canonical
evaluator, and rebuilds the tables. `HF_TOKEN` may be set if Hugging Face asks
for authentication or applies anonymous download limits.

Prepared artifacts:

```text
checkpoints/bge_vl_mllm_s1/
├── hf/BAAI--BGE-VL-MLLM-S1/
└── prepared.json
```

Outputs:

```text
runs/bge_vl_mllm_s1/
├── cache/gallery_features.npy
├── cache/query_features.npy
├── scores.npy
└── run.json
```

Do not commit the checkpoint, feature caches, or score matrix.
