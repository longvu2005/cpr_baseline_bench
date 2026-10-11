# FAFA + Partial SetMatch

Drop-in replacement for `methods/published/02_fafa_setmatch/`. The method ID stays
`fafa_setmatch`, so the existing runner and evaluator commands remain valid.
This update changes only the image-level matching/aggregation rule; FAFA,
checkpoint, detector, CLIP membership thresholds and evaluation are unchanged.

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

1. torchvision Faster R-CNN ResNet-50-FPN-v2 proposes person boxes in every
   gallery/query image;
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

The adapter may also cache predicted person boxes and FAFA gallery-person
features plus component-person FAFA scores under `runs/fafa_setmatch/cache/` to
avoid recomputing them. It also saves `query_component_offsets.npy` alongside
`gallery_person_offsets.npy` so the raw score matrix has explicit row/column
boundaries. The cache directory is fingerprinted by adapter/config/data
and checkpoint identities so changing a model/config does not silently reuse stale
features.

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

The Partial SetMatch scoring change **does** alter the semantics;
`ADAPTER_VERSION` is therefore bumped to v4. The full cache key changes, so the
prior v3 cache is not automatically reused (including its detection cache).
This is intentional for reproducibility. No official FAFA source or weights change.

Legacy configs with `aggregation: min`, non-Hungarian assignment, disabled
SetMatch or a non-finite background score are rejected before localization.

## Run

Extract the ZIP **from the repository root**, replacing the three method files.
The included regression tests are independent of FAFA weights and CUDA:

```bash
python tests/test_fafa_partial_setmatch.py
```

Normal end-to-end command:

```bash
python run_baseline.py fafa_setmatch
```

The root runner installs this method's `requirements.txt` first, pins the official source, downloads the released FAFA checkpoint plus CLIP/detector auxiliary weights, pre-warms FAFA's LAVIS/Transformers runtime assets into a repository-local cache, then runs inference, evaluation, and table rebuilding. Inference preflights the pinned source and runtime inventory before expensive localization and then uses the cache in offline mode, so a missing runtime artifact fails instead of downloading silently mid-benchmark.

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

