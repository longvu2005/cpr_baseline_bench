# FAFA + Partial SetMatch

Drop-in replacement for `methods/published/02_fafa_setmatch/`. The method ID stays
`fafa_setmatch`, so the existing runner and evaluator commands remain valid.
This revision uses official Grounding DINO person proposals and independent stage
caches. The released FAFA checkpoint, native soft-FDA score, CLIP membership
thresholds, Partial SetMatch rule and common evaluator are preserved.

Published baseline adapter for **FAFA (Fine-grained Adaptive Feature Alignment)**
from *Automatic Synthetic Data and Fine-grained Adaptive Feature Alignment for
Composed Person Retrieval* (NeurIPS 2025).

## Official implementation and checkpoint

The adapter imports the authors' implementation instead of reimplementing FAFA:

```text
Delong-liu-bupt/Composed_Person_Retrieval
commit: 0cc16936f031f7ad166be4cce1be33d0b44b728e
subdir: FAFA_SynCPR
```

The authors state that all retrieval training/inference code is open-sourced in
`FAFA_SynCPR` and release a paper-version pretrained model. Their official
`run_inference.sh` uses the checkpoint name `tuned_recall_at1_step.pt`. The
method-local checkpoint preparer downloads the released weight to:

```text
checkpoints/fafa/tuned_recall_at1_step.pt
```

The adapter preserves the official inference path:

- model: `blip2_fafa_cpr` / `pretrain`;
- checkpoint loading with `strict=False`, matching `inference_fafa.py`;
- test preprocessing based on `squarepad_transform_test(224)`;
- multimodal query feature from `model.extract_features({image, text_input})`;
- target feature tokens from `model.extract_target_features(..., mode="mean")`;
- soft FDA similarity: top-`k` target-token similarities followed by their mean;
- released/default FDA settings `k=6`, `alpha=0.5` and soft aggregation.

FAFA is **not retrained, fine-tuned, checkpoint-selected, or tuned** on this CPR
benchmark. CPR Supervision is therefore **No**.

## Scene-to-person localization used by this benchmark adapter

The pilot benchmark uses full scene images and intentionally does not expose a
GT target box in `queries.jsonl`. SetMatch, however, needs person instances.
This adapter therefore uses **predicted** person localization only:

1. official Grounding DINO Swin-T OGC proposes person boxes with the fixed
   prompt `person.` in every gallery/query image;
2. for a query with multiple `subjects`, OpenAI CLIP (`ViT-B/32`) scores each
   `subjects[].select_text` against the predicted query-person crops;
3. a one-to-one Hungarian assignment first chooses one **anchor crop** per
   textual Subject, regardless of its absolute CLIP score;
4. from the remaining detected people, a Subject may receive extra members if
   `CLIP(subject, person) >= membership_threshold` **and**
   `CLIP(subject, person) >= CLIP(subject, assigned_anchor) - membership_margin`;
   extra people cannot be reused across Subjects and are capped by
   `max_members_per_subject`. Fallback whole-scene crops are never extras.

This rule does **not** read a case label, identity list, or ground-truth member
count. Every Subject uses the same anchor-first/threshold-expansion policy.
An INDIVIDUAL query can acquire erroneous extra members; that is a documented
limitation of the zero-shot heuristic, not an oracle-corrected prediction.

No PIPA identity-to-box mapping, GT person box, `target_ids`, or positive labels
are used to localize a target or compute a retrieval score. The CLIP selector is
only a target-localization adapter; final retrieval similarities come from FAFA.

This is distinct from the benchmark's **Predicted Anchor [2]** convention for
methods that natively require an anchor/box: FAFA itself is not box-conditioned.
Predicted crops here are the instance construction required to apply **SetMatch
[1]** to multi-person scene images.

## Multi-person adaptation: Partial SetMatch

FAFA is natively a one-reference-person / one-target-person CPR method. For each
benchmark query:

1. run FAFA independently for **each predicted member** of every Subject,
   reusing that Subject's modification text;
2. run FAFA target encoding independently for every predicted person in a
   gallery image;
3. build the target-person score matrix using the official FAFA soft-FDA score;
4. compute maximum-weight **one-to-one rectangular Hungarian matching**;
5. sum the matched scores, add a fixed background score for each missing
   query component, and divide by the **total number of query components**.

For `K` predicted query people and `N` gallery crops, let `m = min(K, N)`.
Among all one-to-one assignments of exactly `m` real pairs, choose `M*` with the
largest total FAFA similarity. The image score is:

```text
S(Q, I) = [sum_(i,j in M*) s_ij + (K - m) * b] / K
b = setmatch.unmatched_score
```

Missing components stay in the denominator. They are not dropped, and a missing
person does not force the whole image to one floor score. For example, three
components with two matched scores `0.90, 0.80` and `b=0.0` give
`(0.90 + 0.80 + 0.0) / 3 = 0.5667`. These are illustrative scores.

The matching objective and ranking score now agree: the missing term and
denominator are fixed for a given image, so maximizing the real-pair sum also
maximizes the final score. When `N >= K`, all query components must match real
crops, even if some similarities are below `b`; dummy matches are **not** optional.

`run.py` retains exact vectorized matching for two query components. For larger
queries, gallery images with one or two crops also use exact vectorized formulas;
the remaining cases use SciPy Hungarian. No greedy approximation is introduced.
Zero-crop images receive `b` if supplied to the aggregator; the existing detector
still keeps its whole-scene fallback crop when no person is found.

When exactly one query person is predicted and `N > 0`, the rule reduces to the
best FAFA score over gallery crops. Multiple predicted members are always matched
to distinct gallery people where available, even with one textual Subject.

### Fixed background score

The shipped config uses `aggregation: mean` and **`unmatched_score: 0.0`**.
This is an explicit, uncalibrated reference value, not an estimate from the
uploaded files and not a claim about the empirical FAFA background distribution.
`unmatched_score_source` records that distinction in `run.json`.

If the final protocol requires an empirical background value, estimate it once
from FAFA similarities of known non-corresponding person pairs on a **separate
calibration/validation set**. One reproducible conservative rule is a fixed 5th
percentile, chosen before inspecting test results:

```python
import numpy as np

# Prepare this array from known non-corresponding calibration pairs only.
background_scores = np.load("calibration_negative_person_scores.npy")
if background_scores.size == 0 or not np.isfinite(background_scores).all():
    raise ValueError("Need non-empty, finite calibration similarities")
b = float(np.quantile(background_scores, 0.05, method="linear"))
print(b)  # Copy into setmatch.unmatched_score in config.yaml.
```

Also set `unmatched_score_source` to describe the calibration split, sample
construction and statistic, and keep the sampled scores as experimental evidence.
Freeze `b` for every query, case and test run. Inference never estimates `b` from
the current gallery or test labels. Do not call random gallery pairs confirmed
negatives unless their non-correspondence is established.

A missing slot contributes less than a particular real match only when `b` is
below that match's similarity. The method therefore does not promise a universal
penalty relative to every possible real crop. It also does not fix CLIP selecting
too many query members or make FAFA jointly relation-aware.

## Query text behavior

Each target subject uses:

```text
subjects[].modify_text
```

as FAFA's relative caption. If it is empty, `relation_text` is used; if both are
empty, the query-level `text` is used. RELATIONAL queries are therefore not
claimed to be jointly relation-aware: the original single-person FAFA model is
applied independently to each subject and SetMatch enforces only one-to-one
coverage.

`subjects[].select_text` is used only to select the predicted reference person
inside the query image. It is not concatenated into FAFA's modification text.

## Benchmark contract

The adapter:

- reads `data/gallery.jsonl` and `data/queries.jsonl` in exact order;
- scores every query against the complete gallery;
- leaves the exact query image in `scores.npy` (the common evaluator removes it);
- writes `runs/fafa_setmatch/scores.npy` with shape
  `(len(queries), len(gallery))` and finite scores only;
- writes reproducibility metadata to `runs/fafa_setmatch/run.json`;
- does not calculate the official benchmark metrics itself.

The adapter caches each stage independently under `runs/fafa_setmatch/cache/`:

| Stage | Cache input identity | Effect of a query-manifest edit |
| --- | --- | --- |
| `detections` | Gallery, pinned detector source/config/weights, NMS/threshold/cap, helper and extension identity | Reused |
| `localization` | Detections, query manifest, CLIP weights and membership settings | Recomputed |
| `gallery_features` | Detections, gallery crop threshold, FAFA source/weights/transform, storage dtype | Reused |
| `query_features` | Localization and FAFA source/weights/transform | Recomputed |
| `component_scores` | Gallery/query features and FDA settings | Recomputed |

Changing batch sizes, workers, execution device or `setmatch.unmatched_score`
does not invalidate these stages. The background score is applied only during
final aggregation. Checkpoint identities retain the repository's path/size/mtime
policy; pinned source/config/helper content is hashed. Do not edit weights in
place while preserving both their size and timestamp.

A warm run reuses CLIP localization and both FAFA feature caches. If both feature
caches are complete, it skips constructing the large FAFA model entirely.
`query_component_offsets.npy` and `gallery_person_offsets.npy` preserve the
component/person boundaries in the raw matrix. `run.json` records every stage key,
path, detector protocol and attention backend.

## Grounding DINO assets and runtime

FAFA reuses the detector source/checkpoint locations already used by S5/S10:

```text
source: IDEA-Research/GroundingDINO
commit: 856dde20aee659246248e20734ef9ba5214f5e44
config: groundingdino/config/GroundingDINO_SwinT_OGC.py
checkpoint: checkpoints/groundingdino/groundingdino_swint_ogc.pth
prompt: person.
box threshold: 0.35
text threshold: 0.25
NMS IoU: 0.50
minimum box size: 4 px
FAFA person cap: 10 per image
```

S5 has no person cap by default; FAFA retains its existing cap of 10. Set
`max_persons_per_image: null` if that cap should also match S5. The upstream
`text_threshold` controls phrase decoding, not which boxes are retained;
FAFA needs only boxes/confidences, so it skips decoding unused phrases.

`groundingdino_adapter.py` calls only S5's detector preparation helpers. It does
not prepare CLIP-ReID assets. It tries the official optional CUDA extension and
uses the existing official PyTorch fallback when that extension is unavailable.
Failed extension preparation is remembered for the same runtime; `--force` retries
it. The fallback backend is logged because it can be slower on CUDA.

Preparation and detection run in a separate process. This is necessary because
FAFA and Grounding DINO have different BERT/Hugging Face caches, whose library
constants are initialized at import. Process exit also releases detector memory
before FAFA loads. `transformers==4.39.3` remains pinned; no Hugging Face Grounding
DINO implementation or Transformers upgrade is introduced.

Detection uses the pinned `load_image` transform (800 px short side, maximum
1333 px), the native model's padded/masked NestedTensor batch, and S5's box
conversion/NMS. `runtime.detector_batch_size` defaults to 2; set it to 1 if GPU
memory is limited. Batches are split when padding would add more than 30% extra
pixels, preserving manifest order and avoiding costly portrait/landscape padding.
Preprocessing uses `runtime.detector_num_workers` workers.

## Performance audit and interruption handling

| Previous behavior | Resulting overhead or failure | Fix |
| --- | --- | --- |
| One key hashed the entire YAML, queries and adapter version | Changing a batch size/background score reran detection and gallery encoding | Separate stage identities |
| CLIP localization and FAFA query embeddings were not cached | Every retry reran both steps and loaded FAFA | Persist localization and ragged query features; skip FAFA when features are cached |
| Every person crop reopened and decoded its scene | Multi-person images repeated CPU work | Reuse the last decoded scene per DataLoader worker; reuse repeated CLIP scene/text embeddings |
| Released checkpoint loaded directly onto CUDA | Temporary duplicate checkpoint tensors increased peak GPU memory | Load the checkpoint on CPU before copying its state |
| FAFA stayed on GPU during similarity scoring | Less free memory for scoring blocks | Release it after feature extraction |
| Every query block transferred scores to CPU separately | Repeated GPU synchronization | Transfer once per person block |
| Score/feature memmaps were written directly to their final paths | A canceled write could later look like a shape-valid cache | Publish only complete files via atomic rename |
| Detection was saved only after the last gallery image | Cancellation lost all completed detection work | Flush an ordered JSONL journal per batch and resume its valid prefix |

Gallery encoding and component scoring restart their incomplete stage after a
cancellation; detection resumes from completed journal rows. Final `scores.npy`
is also replaced atomically. These files assume one writer per output directory;
do not run two FAFA jobs into the same output directory simultaneously.

The regression suite checks numerical score equivalence, missing-person matching,
cache dependency isolation, canceled writes, detection resume, crop reuse, and
cold/warm/query-only rebuild orchestration with toy models. It does not measure
full-benchmark CUDA throughput or validate retrieval metrics with released FAFA
weights. The detector change can change predictions and retrieval scores; run the
benchmark again. The first Grounding DINO run rebuilds detections/features and
cannot reuse the previous Faster R-CNN crop cache.

## Pinned-source integrity and retry behavior

The authors' checkout is treated as immutable benchmark input. Both preparation
and inference disable Python bytecode writes (`PYTHONDONTWRITEBYTECODE=1` plus
`sys.dont_write_bytecode`) so importing LAVIS cannot rewrite tracked
`__pycache__` files. The Git integrity guard still rejects every real tracked
source/config change; it ignores only generated `.pyc` and `.pyo` bytecode files
(normally stored under `__pycache__`). This prevents Python-version-specific
bytecode from causing
a false "tracked local modifications" failure while preserving the pinned-source
reproducibility check.

Inference performs a cheap FAFA preflight **before** detector/CLIP localization:
the pinned commit, runtime-assets marker/inventory, and offline cache are checked
first. Missing/stale artifacts or real source edits therefore fail before the
expensive person detector runs. The large FAFA model itself is still loaded only
after detector/CLIP localization to avoid unnecessary GPU-memory overlap.

`ADAPTER_VERSION` is v5. Each expensive stage now owns its own schema/input key;
future aggregation-only changes do not need to invalidate person detections or
features. Old monolithic caches are not migrated into this detector protocol.

Legacy configs with `aggregation: min`, non-Hungarian assignment, disabled
SetMatch or a non-finite background score are rejected before localization.

## Run

Extract the ZIP **from the repository root**, preserving its relative paths.
It replaces the five existing FAFA method files and `.gitignore`, adds
`groundingdino_adapter.py`, and adds `tests/test_fafa_groundingdino.py`.
The existing S5 helper files are
required and already present in this repository. This patch targets main commit
`5de87cabb97c326cd4f7359e91ddb27b24040cf1`.
The included regression tests are independent of FAFA weights and CUDA:

```bash
python tests/test_fafa_groundingdino.py
```

Normal end-to-end command:

```bash
python run_baseline.py fafa_setmatch
```

The root runner installs this method's `requirements.txt` first, pins the official source, downloads the released FAFA checkpoint plus CLIP/Grounding DINO weights, pre-warms FAFA's LAVIS/Transformers runtime assets into a repository-local cache, then runs inference, evaluation, and table rebuilding. Inference preflights the pinned source and runtime inventory before expensive localization and then uses the cache in offline mode, so a missing runtime artifact fails instead of downloading silently mid-benchmark.

Expected benchmark outputs:

```text
runs/fafa_setmatch/
├── scores.npy
├── run.json
└── cache/

outputs/fafa_setmatch/
├── metrics.json
└── run.json
```


Stage-2 input mapping and GROUP behavior are described in
[data/README.md](../../../data/README.md).

## Baseline constraints

- Inference uses public `subjects[]` and their selection/modification texts.
  `case`, `case_type`, `identity_ids`, GT boxes and positive labels are not
  consulted to choose members or compute scores. The Subject-list structure
  itself must be a permitted query input in the evaluation protocol.
- Membership thresholds are **fixed heuristics** (`config.yaml`), not tuned
  on the evaluation set. FAFA/CLIP/detector weights are frozen.
- All predicted group members receive the *same* Subject modification text:
  there is no group-level or relation verifier and no guarantee that CLIP
  discovers the true number of people.
- Per-query predicted member counts, CLIP anchor similarities and fallback
  statistics are recorded in `run.json` for auditability.
- Partial SetMatch is a benchmark adapter, **not** the authors' original FAFA
  image-level method. Report the fixed background value and its provenance;
  do not describe the default `0.0` as empirically calibrated.

