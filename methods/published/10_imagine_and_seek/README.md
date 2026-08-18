# P10. Imagine and Seek — strict mounted-assets CPR adapter

Paper: **Imagine and Seek: Improving Composed Image Retrieval with an Imagined Proxy** (CVPR 2025).

This implementation targets the released **LDRE-L + IP-CIR** configuration while adapting only the dataset boundary needed for the CPR benchmark. It uses the released Imagine-and-Seek source pinned at `LeyRio/Imagine-and-Seek@2f615824bd7a6958083c85d8ad5e5e20549e22cb` and does **not** train or tune on CPR labels.

## Status

```text
Implementation status: OFFICIAL_SOURCE_ADAPTED
CPR supervision: No
CPR training/tuning: No
GT target identity: No
GT target box: No
Small-model fallback: No
```

## Faithful profile

The final profile is intentionally not the earlier Kaggle-small debug variant:

- dense captions: BLIP-2 OPT-6.7B COCO, **15** samples/query;
- editing/layout LLM: `Qwen/Qwen1.5-32B-Chat-GPTQ-Int4`;
- proxy generator: Realistic Vision + MIGC + ELITE;
- proxy count: **5/query**;
- retrieval: CLIP ViT-L/14 / LDRE-L;
- robust-proxy weights: `s_w=t_w=a_w=1`;
- fusion: `lambda=0.3`, fixed from the paper CIRCO setting, never tuned on CPR.

The CPR-specific boundary uses the complete reference scene as the ELITE visual reference because the CPR benchmark does not provide the author dataset's `shared_concept`/object-mask annotations. This is recorded as an adaptation rather than represented as exact dataset preprocessing.

## Why the two giant models are mounted on Kaggle

Kaggle's writable `/kaggle/working` space is too small for the exact BLIP2-OPT6.7B and Qwen32B-GPTQ snapshots plus the generator environment. V5 therefore requires those two exact snapshots to be mounted read-only under `/kaggle/input` on Kaggle. It verifies their architecture/quantization signature before doing any expensive work, and during checkpoint preparation it also verifies the mounted `config.json` and safetensor index against the pinned Hugging Face revisions.

Use:

```bash
python methods/published/10_imagine_and_seek/download_checkpoint.py --check-inputs
```

Expected discovery lines:

```text
[mount] captioner: /kaggle/input/...
[mount] layout_llm: /kaggle/input/...
```

If auto-discovery cannot locate them, set:

```bash
export IPCIR_BLIP2_DIR=/kaggle/input/.../model-directory
export IPCIR_QWEN32_DIR=/kaggle/input/.../model-directory
```

Each directory must contain `config.json`, `model.safetensors.index.json`, and all model safetensor shards.

## Streaming proxy storage

Persisting `2,975 × 5 = 14,875` generated PNGs would waste the remaining writable disk. V5 instead performs:

```text
MIGC+ELITE proxy
    -> CLIP-L targetpad encode
    -> normalized 768-D feature
    -> discard proxy image
```

The resumable feature store is:

```text
runs/imagine_seek/cache/proxy_features.npy
runs/imagine_seek/cache/proxy_features.state.json
```

Only sparse audit images are retained under `runs/imagine_seek/cache/proxy_audit/`.

## Scoring

For the five normalized proxy features, released retrieval first computes their mean `f_p`. With source image feature `f_q` and semantic direction `f_s`, the robust imagined-proxy feature is:

```text
f_RP = a_w f_p
     + s_w max(f_p)/max(f_q) f_q
     + t_w max(f_p)/max(f_s) f_s
```

Then, with baseline text similarity `S_t` and robust proxy similarity `S_p`:

```text
S_b = S_t * S_p
S_f = lambda * S_t + (1-lambda) * S_b
```

The method writes a **complete** `Q × G` score matrix and does not remove the query image internally; exclusion remains evaluator-owned.

## Environment isolation

`requirements.txt` contains only host bootstrap requirements (`PyYAML`, `uv`). `download_checkpoint.py` creates an isolated Python 3.10 environment under:

```text
runs/imagine_seek/env/ipcir_py310
```

This prevents the released legacy stack from replacing Kaggle's system Torch/Transformers environment.

## Recommended Kaggle order

```bash
python methods/published/10_imagine_and_seek/verify_install.py
python methods/published/10_imagine_and_seek/download_checkpoint.py --check-inputs
python run_baseline.py imagine_seek
```

If `--check-inputs` fails, mount the exact giant models first. Do **not** lower model size or the storage checks for a final reported P10 result.
