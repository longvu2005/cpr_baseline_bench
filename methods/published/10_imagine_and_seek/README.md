# P10. Imagine and Seek (IP-CIR) — reproduced full-scene CPR adapter

Paper: **Imagine and Seek: Improving Composed Image Retrieval with an Imagined Proxy** (CVPR 2025).

This directory is deliberately marked **REPRODUCED**, not `OFFICIAL_RELEASED`. No author IP-CIR repository or final IP-CIR checkpoint is assumed. IP-CIR is training-free; this adapter reconstructs its inference equations and proxy-generation recipe from the paper using public foundation-model code.

## What is preserved from the paper

The scorer follows the paper's two central equations. For each generated proxy feature `f_p`, original query-image feature `f_q`, and semantic perturbation `f_s = f_t - f_o`:

```text
f_RP = f_p
     + max(f_p)/max(f_q) * f_q
     + max(f_p)/max(f_s) * f_s
```

The proxy score `S_p` is then combined with baseline retrieval score `S_t`:

```text
S_b = S_t * S_p
S_f = lambda * S_t + (1 - lambda) * S_b
```

The paper generates **5 imagined proxies per query**. This implementation also uses five by default.

## CPR adaptation

- **Query:** complete reference scene + canonical `queries.jsonl[text]` modification.
- **Gallery:** complete scene image; no person crop and no GT person box.
- **Base retriever:** existing P5 LinCIR full-scene adapter.
- **SINGLE / MULTI / RELATIONAL:** all use the same direct full-scene IP-CIR path. There is no SetMatch layer.
- **CPR supervision:** No.
- **Query-image exclusion:** not done here; the benchmark evaluator owns exclusion.

## Important reproduction choices

The paper describes BLIP2 captions, Qwen1.5-32B layout inference, MIGC/Stable Diffusion proxy generation, and an ELITE-style reference-image conditioning path. The public MIGC repository exposes text/layout-controlled generation but does not provide an IP-CIR author implementation of that exact reference-image-conditioned branch.

Therefore the bundled generator uses:

```text
query image
 -> BLIP2 captions
captions + modification
 -> Qwen1.5-32B target scene/layout JSON
layout
 -> public MIGC + Stable Diffusion 1.5
 -> 5 proxy images
```

The original query image is still injected into `f_RP` exactly through the paper's query-image residual term. This is a **paper-guided reproduction approximation**, not a claim of bit-exact author inference.

The paper uses five proxies but does not clearly specify a multi-proxy aggregation rule in the released text. This adapter uses:

```text
S_p = mean_j cosine(f_RP_j, gallery)
```

and records `proxy.aggregation = mean_similarity` in `run.json`.

The default `lambda=0.3` is fixed in config and **not tuned on CPR labels**. It is an adapter choice and should remain frozen for the reported main result unless a separate validation-only study is explicitly declared.

## Files

```text
methods/published/10_imagine_seek/
├── config.yaml
├── requirements.txt
├── download_checkpoint.py
├── prepare_proxies.py
├── run.py
└── README.md
```

## End-to-end run

From repository root:

```bash
python run_baseline.py imagine_seek
```

The first run is expensive because the reproduction needs LinCIR, BLIP2, Qwen1.5-32B, MIGC, and Stable Diffusion assets.

For debugging individual stages:

```bash
python methods/published/10_imagine_seek/download_checkpoint.py
python methods/published/10_imagine_seek/prepare_proxies.py --stage captions
python methods/published/10_imagine_seek/prepare_proxies.py --stage layouts
python methods/published/10_imagine_seek/prepare_proxies.py --stage generate
python methods/published/10_imagine_seek/run.py
```

## Precomputed proxy mode

If exact author proxies, a better proxy generator, or proxies generated in another environment are available, set:

```yaml
proxy:
  mode: precomputed
  manifest: runs/imagine_seek/cache/proxy_manifest.jsonl
```

The manifest must have one row per canonical query in exact order:

```json
{
  "query_index": 0,
  "image_id": "...",
  "original_captions": ["..."],
  "target_captions": ["..."],
  "proxy_paths": ["path/to/proxy_00.png", "... five total ..."]
}
```

This lets the retrieval/scoring implementation stay unchanged when the proxy generator is improved.

## Output contract

```text
runs/imagine_seek/
├── cache/
│   ├── query_captions.jsonl
│   ├── layouts.jsonl
│   ├── proxy_manifest.jsonl
│   ├── proxies/...
│   ├── proxy_features.npy
│   └── query_components.npz
├── scores.npy
└── run.json
```

`scores.npy` preserves complete canonical query/gallery order and contains finite float scores only.
