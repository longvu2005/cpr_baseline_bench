# P8. BASIC + SetMatch

Benchmark adapter for **BASIC** from *Instance-Level Composed Image Retrieval* (NeurIPS 2025), with the benchmark's person-level **SetMatch** adaptation for multi-target CPR.

## Paper / official implementation

- Paper: *Instance-Level Composed Image Retrieval*, NeurIPS 2025.
- Official project: `https://vrg.fel.cvut.cz/icir/`
- Official repository: `https://github.com/billpsomas/icir`
- Pinned source commit: `28375f899d731f91857dcf7dfc9af28755cbccf7`
- BASIC is training-free. There is **no BASIC neural checkpoint**.
- The VLM backbone is the official BASIC CLIP setting: OpenCLIP **ViT-L/14, OpenAI weights**.

The checkpoint preparer clones the exact official source commit and prepares every external runtime asset before inference. `run.py` runs with network access blocked while the official BASIC model is loaded, so missing assets fail instead of downloading silently.

## What is preserved from official BASIC

The adapter reads `METHOD_PRESETS["basic"]` directly from the pinned official `run_retrieval.py` and verifies it against `config.yaml`. The following official `basic --use_preset` components are preserved:

1. **Feature standardization** using the official LAION-1M CLIP mean.
2. **Contrastive PCA projection** using:
   - `generic_subjects.csv`
   - `generic_styles.csv`
   - `aa = 0.2`
   - up to 250 positive-eigenvalue components.
3. **Query expansion** with the official top-25 database images and exponential weighting.
4. **Text contextualization** with the first 100 positive-corpus entries, in both `corpus + text` and `text + corpus` orders, averaged exactly as in the official feature creation path.
5. **Synthetic score normalization** using `dataset_1_sd_clip.pkl.npy`.
6. **Harris fusion** with `lambda = 0.1`.

The official implementation exposes rankings, not the raw BASIC similarity matrix required by SetMatch. Therefore this adapter contains a score-returning transcription of the `basic` branch of the pinned `utils_retrieval.calculate_rankings()`.

To guard against accidental drift, `run.py` performs an **official parity check** by running the pinned official `calculate_rankings()` on a small real-feature fixture and requiring its ranking to exactly match `argsort(adapter_basic_scores)`.

## CPR adaptation

BASIC itself is unchanged at the pairwise level:

```text
(reference-person crop, modification text)
                +
       gallery-person crop
                ↓
       official BASIC score
```

The benchmark adaptation is only around this pairwise scorer:

1. Detect person candidates in every canonical gallery scene with predicted Faster R-CNN boxes.
2. For each query subject, use `subjects[].select_text` with OpenAI CLIP ViT-B/32 to select the reference person. Multi-subject selection is maximum-weight one-to-one Hungarian assignment.
3. Apply BASIC to each selected query person + its modification text against every detected gallery person.
4. Collapse person-level scores to one score per gallery image with SetMatch:
   - assignment: maximum-weight one-to-one Hungarian;
   - aggregation: strict minimum of the assigned BASIC scores;
   - too few gallery persons: `unmatched_score = -1e6`.

No `target_ids`, positives, GT identity labels, or GT person boxes are used.

### Query shortage policy

If the reference image has fewer predicted person candidates than query subjects, the missing selector slots are padded with the **full reference scene**. This is a predicted-input fallback and is explicitly recorded in `run.json`.

### No-person gallery policy

If the detector finds no person in a gallery scene, that scene gets one full-scene fallback candidate. SINGLE queries can still score it; a MULTI query requiring more slots receives the normal SetMatch unmatched score.

## SINGLE / MULTI / RELATIONAL

- **SINGLE:** one selected query person; gallery score is the best BASIC person score in that scene.
- **MULTI:** one BASIC target per query subject, followed by one-to-one SetMatch.
- **RELATIONAL:** because BASIC has no dedicated relation module, when `relation_text` is present the adapter uses the full canonical query text as the BASIC text input for each selected subject. This preserves relational wording without adding a learned relation classifier. The SetMatch rule remains unchanged.

## Runtime artifacts

`download_checkpoint.py` prepares:

```text
runs/basic_setmatch/official_source/icir/
    pinned official i-CIR repository

checkpoints/basic_setmatch/openclip_cache/
    exact runtime cache populated by the official i-CIR load_model("clip", ...)

checkpoints/clip/ViT-B-32.pt
    OpenAI CLIP selector weights

checkpoints/torchvision/fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth
    COCO person detector weights

checkpoints/basic_setmatch/prepared.json
    reproducibility / asset inventory marker
```

Official BASIC resources are consumed from the pinned source checkout:

```text
corpora/generic_subjects.csv
corpora/generic_styles.csv
data/laion_mean/laion_1m_mean_clip.pkl
synthetic_data/dataset_1_sd_clip.pkl.npy
```

## Run

From repository root:

```bash
python run_baseline.py basic_setmatch
```

or equivalently:

```bash
python run_baseline.py 08_basic_setmatch
```

The normal pipeline is:

```text
install requirements
→ pin official i-CIR source + prepare runtime assets
→ run BASIC + SetMatch
→ evaluate.py
→ build_tables.py
```

Expected raw outputs:

```text
runs/basic_setmatch/scores.npy
runs/basic_setmatch/run.json
```

The method preserves the canonical query/gallery order and scores the complete gallery. Query-image exclusion remains the responsibility of `evaluate.py`.

## Cache safety

All heavy caches live below:

```text
runs/basic_setmatch/cache/<fingerprint>/
```

The fingerprint includes the adapter/config, gallery/query manifests, pinned-source marker, selector checkpoint, and detector checkpoint. A configuration or artifact change therefore cannot silently reuse stale features.

## Reproducibility status

- BASIC method checkpoint: **not applicable (training-free)**.
- CLIP ViT-L/14: pretrained OpenAI weights loaded through the pinned official i-CIR code path.
- BASIC statistics/corpora/synthetic normalization: **official repository resources**.
- Detector / target selector: benchmark adaptation only, not part of the BASIC paper.
- CPR supervision: **No**.
