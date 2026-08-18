# P10 — Imagine and Seek (LDRE-L + IP-CIR)

Paper: **Imagine and Seek: Improving Composed Image Retrieval with an Imagined Proxy**, CVPR 2025.

This folder is a **strict official-source CPR adapter** for the paper's **CLIP-L/14 branch: LDRE + IP-CIR**. IP-CIR is training-free and plug-and-play: it improves a baseline similarity `S_t` using an imagined-proxy similarity. The adapter never reads CPR target IDs/images, positive labels, GT identity labels, or GT boxes.

## Why this replacement exists

The previous P10 in the repository could fail before inference because it installed the released legacy stack (`transformers`, `diffusers`, `numpy`, etc.) directly into Kaggle's Python 3.12 environment. It also mixed P5 LinCIR-L with IP-CIR. The paper's reported branches are **LDRE + IP-CIR with CLIP-L/14** and **LinCIR + IP-CIR with CLIP-G/14**; therefore substituting the repository's LinCIR-L is not the paper configuration.

This replacement fixes both structurally:

- phase 2 only installs tiny bootstrap packages;
- all legacy IP-CIR dependencies live in an isolated Python-3.10 environment;
- the retrieval side implements the released LDRE-L caption/debias path and then IP-CIR balancing;
- there is no automatic small-model fallback.

## Fidelity choices

Default final configuration:

- Author source: `LeyRio/Imagine-and-Seek` pinned to `2f615824bd7a6958083c85d8ad5e5e20549e22cb`.
- Dense captions: BLIP-2 OPT-6.7B COCO, **15** samples/query, matching the released `caption_coco_opt6.7b` preprocessing choice.
- Editing/layout LLM: `Qwen/Qwen1.5-32B-Chat-GPTQ-Int4`.
- Layout prompt: released `prompt/prompt_layout_v2.yaml`.
- Text baseline: **LDRE-L**, OpenAI CLIP ViT-L/14, target-pad ratio 1.25, 15 paired original/edited captions and released negative-difference debiasing.
- Proxy generator: Realistic Vision V6 + released MIGC + ELITE.
- Proxy count: **5** images/query.
- Robust-proxy weights `(source, semantic, proxy) = (1,1,1)`.
- Fusion `lambda = 0.3`, fixed from the paper's CIRCO setting and **not tuned using CPR labels**.

The final score is:

```text
St = LDRE-L edited-caption baseline similarity
Sp = robust imagined-proxy similarity
Sb = St * Sp
Sf = lambda * St + (1-lambda) * Sb
```

The robust proxy is built from the mean proxy feature, query-image feature, and semantic perturbation exactly in the released Eq. (1) form. The implementation intentionally preserves the released global scalar `max()` scaling.

## Unavoidable CPR adaptation boundary

The author's CIR datasets provide dataset-specific concepts/object masks (for example CIRCO `shared_concept`). CPR does not provide an author-compatible concept/mask, and using target identity or GT boxes would leak supervision. Therefore this adapter:

- generates dense captions from the **query image only**;
- applies the CPR instruction using the released LDRE editing prompt;
- feeds the released layout prompt with the first dense visual concept plus the CPR rule;
- uses the complete query image as ELITE's visual reference mask for image-referenced instances;
- never inspects target images/IDs/positives/GT boxes.

This is recorded as `OFFICIAL_SOURCE_ADAPTED`, not `OFFICIAL_EXACT`.

## Environment isolation

`requirements.txt` is intentionally tiny. The root benchmark's phase 2 must **not** downgrade the notebook ML stack.

Phase 3 creates:

```text
runs/imagine_seek/env/ipcir_py310
```

with:

```text
Python       3.10
Torch        2.2.1+cu121
Torchvision  0.17.1+cu121
Transformers 4.43.3
Diffusers    0.21.1
AutoGPTQ     0.7.1
```

`run.py` re-execs itself inside this environment before importing Torch/CLIP.

## Install verification

After replacing the method folder, run this **before** the baseline:

```bash
python methods/published/10_imagine_and_seek/verify_install.py
```

It fails if the host `requirements.txt` still contains `torch`, `transformers`, `diffusers`, `bitsandbytes`, or a NumPy pin. This prevents accidentally running the stale P10 again.

Environment-only diagnosis:

```bash
python methods/published/10_imagine_and_seek/download_checkpoint.py --env-only
```

## Final run

```bash
python run_baseline.py imagine_seek
```

Caption, Qwen, and proxy generation are resumable. The method writes the canonical full score matrix:

```text
runs/imagine_seek/scores.npy   # [num_queries, num_gallery], float32
runs/imagine_seek/run.json
```

The method does not remove the query image internally; the global evaluator keeps ownership of exclusion.

## Resource policy

This is the **final benchmark configuration**, not a small Kaggle approximation. The downloader performs explicit disk/GPU preflight and stops rather than silently replacing Qwen-32B or BLIP-2 OPT-6.7B with smaller models. If the runtime cannot hold the released-scale configuration, use a larger runtime or pre-mounted checkpoints; do not report a smoke-test model as P10.
