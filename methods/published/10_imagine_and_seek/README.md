# P10. Imagine and Seek — official-source CPR adapter

Published CPR adapter for **Imagine and Seek (IP-CIR)** from *Imagine and Seek: Improving Composed Image Retrieval with an Imagined Proxy* (CVPR 2025).

This directory replaces the earlier paper-guided reproduction. A public author project is now available at `LeyRio/Imagine-and-Seek`; this adapter pins and imports that released source while keeping the benchmark's canonical score-matrix/evaluator contract.

## Status

```text
Implementation status: OFFICIAL_SOURCE_ADAPTED
CPR supervision: No
GT target box / target identity used: No
Training on CPR: No
```

IP-CIR itself is training-free. Running this method is inference/reproduction, not training.

## What is official vs benchmark-specific

Preserved from the released project:

- pinned author source and MIGC implementation;
- released `MIGC_SD14.ckpt`;
- ELITE global/local mapper architecture and weights;
- Realistic Vision V6.0 B1 VAE generator;
- Qwen-family layout planning;
- MIGC + ELITE visual-reference injection;
- five proxy images per query;
- robust proxy representation and multiplicative retrieval fusion.

Benchmark adaptation:

- query/gallery data come from `data/queries.jsonl` and `data/gallery.jsonl`;
- query is **direct full scene**;
- instead of CIRCO/CIRR-specific object masks, every `ref=image` layout instance receives the complete query scene as the ELITE visual concept and a full-scene mask;
- no GroundingDINO/SAM/GLIP build is required for P10 because those released components are only used to derive/filter object masks in the original dataset pipeline; this CPR adapter intentionally does not use a GT/object-specific target mask;
- LinCIR P5 is the base retrieval branch;
- the normal benchmark evaluator still removes the query image and computes metrics.

The released author source is pinned to:

```text
repository: https://github.com/LeyRio/Imagine-and-Seek.git
commit: 2f615824bd7a6958083c85d8ad5e5e20549e22cb
```

## Kaggle compatibility

The released project was developed around Python 3.9 / Torch 2.2.2 and a large legacy environment. This adapter **does not downgrade Kaggle's Torch/CUDA stack**. `requirements.txt` installs only the Python packages needed by the benchmark-side adapter and MIGC/ELITE import path.

`diffusers==0.21.1` is distributed as source on newer Python versions, so pip can legitimately print:

```text
Installing build dependencies: started
```

That line alone is not an error. A real installation failure will end with `ERROR:` or a non-zero return code.

## Models/assets

`download_checkpoint.py` prepares:

1. P5 LinCIR assets;
2. pinned `LeyRio/Imagine-and-Seek` source;
3. public `MIGC_SD14.ckpt`;
4. ELITE `global_mapper.pt` and `local_mapper.pt`;
5. Realistic Vision `realisticVisionV60B1_v60B1VAE.safetensors`;
6. BLIP2 captioner;
7. Qwen layout planner;
8. Stable Diffusion 1.5 text/tokenizer components needed to load the single-file Realistic Vision checkpoint offline.

The default Qwen model is:

```text
Qwen/Qwen1.5-7B-Chat
```

loaded with bitsandbytes 4-bit. The released project primarily reports `Qwen1.5-32B-Chat-GPTQ-Int4`, but AutoGPTQ is fragile on Kaggle/Python 3.12. The author README explicitly allows other Qwen/LLM choices. If you have sufficient resources, change `layout_llm.repo_id` in `config.yaml`.

## Run

From repository root:

```bash
python run_baseline.py imagine_seek
```

The end-to-end runner performs:

```text
install requirements
-> prepare external assets
-> run P10
   -> run/load P5 LinCIR
   -> BLIP2 captions
   -> Qwen target layouts
   -> released MIGC + ELITE proxy generation
   -> CLIP-L proxy features
   -> robust proxy representation
   -> IP-CIR score fusion
-> official evaluate.py
-> build_tables.py
```

Long stages are cached. If generation is interrupted, existing proxy PNGs are reused on the next run.

## Useful debugging commands

Prepare assets only:

```bash
python methods/published/10_imagine_and_seek/download_checkpoint.py
```

Generate captions only:

```bash
python methods/published/10_imagine_and_seek/prepare_proxies.py --stage captions
```

Generate layouts only:

```bash
python methods/published/10_imagine_and_seek/prepare_proxies.py --stage layouts
```

Generate proxy images only:

```bash
python methods/published/10_imagine_and_seek/prepare_proxies.py --stage generate
```

Run scorer after proxies exist:

```bash
python methods/published/10_imagine_and_seek/run.py
```

## Retrieval computation

The released retrieval code first averages the five proxy-image features:

```text
f_p = mean(proxy_1, ..., proxy_5)
```

Then builds the robust proxy representation:

```text
f_s = f_t - f_o

f_RP = f_p
     + max(f_p)/max(f_q) * f_q
     + max(f_p)/max(f_s) * f_s
```

where:

```text
f_q = reference/query image feature
f_o = original/reference caption feature
f_t = imagined target caption feature
```

The final score follows the released source:

```text
S_p = f_RP @ gallery_features.T
S_b = S_t * S_p
S_f = lambda * S_t + (1 - lambda) * S_b
```

with default:

```text
lambda = 0.3
```

`S_t` is the P5 LinCIR complete-gallery score matrix. No query image is removed inside P10; exclusion belongs to `evaluate.py`.

## Expected output

```text
runs/imagine_seek/
├── cache/
│   ├── query_captions.jsonl
│   ├── layouts.jsonl
│   ├── proxy_jobs.jsonl
│   ├── proxy_manifest.jsonl
│   ├── proxies/
│   ├── proxy_features.npy
│   └── query_components.npz
├── official_source/
│   └── Imagine-and-Seek/
├── scores.npy
└── run.json
```

## Important limitation

The author source expects dataset-specific object masks and includes GroundingDINO/SAM/GLIP helpers. For CPR, using those helpers would introduce a second target-localization protocol and difficult legacy builds. This adapter therefore uses **full-scene ELITE conditioning**. It is a documented CPR boundary adaptation, not an oracle and not a hidden GT localization step.
