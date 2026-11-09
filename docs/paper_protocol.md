# Paper comparison protocol

## What this benchmark establishes

All methods use the same split-specific gallery, reference exclusion,
reviewed Full positives, identity positives, ordered tie rule and evaluator.
This supports a comparison of frozen pretrained transfer systems and explicit
CPR adapters. It does not establish equal backbone capacity, training data,
training supervision or inference cost.

On 2026-11-09, RCR commit `2fe92766d5158f8dd87297d3bc4cb6280b8ed63b`
exported these complete final splits (version 0.1.0):

| Split | Queries | Gallery | Individual | Group | Dual | Relational |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| val | 1,799 | 5,684 | 1,320 | 168 | 149 | 162 |
| test | 2,307 | 7,868 | 1,319 | 329 | 318 | 341 |

Counts come from actual finalized files, not older prose in the RCR protocol
document. Reimport the release used for the paper and retain its manifest hashes.
Matching a version string alone is insufficient because content can change.

The baseline evaluator and RCR's `evaluate_rankings` were compared on the same
synthetic full rankings for all 2,307 test queries. All eight common overall
metrics and sixteen per-case metrics agreed within 1e-12 (maximum observed
absolute difference 8.673617379884035e-19). This checks protocol equivalence;
synthetic rankings are not model predictions or publishable accuracy results.

R@K is the fraction of queries with at least one positive in the top K.
It is not the fraction of all positives recovered. Full-mAP is query weighted.
Macro-Full-mAP equally weights cases present on test. CandidateRecall and
coarse/fine diagnostics are separate from final retrieval metrics.

## Comparison groups

Use separate blocks in the paper for native scene systems and shared-person
adapters. Person methods share predicted boxes and CLIP Subject memberships;
they retain their own encoders and native pair scores. Published names with
Partial SetMatch denote a benchmark adaptation. The proposed method's learned
localizer is a method component; sharing detector boxes does not imply identical
localization or equal training supervision with the proposed model.

Early/late CLIP fusion, S6, S9, S10 and S11 select parameters on complete val.
Other defaults are predeclared. If any predeclared prompt, cap, checkpoint or
adapter setting was actually chosen using RCR val in earlier experiments,
report that validation use truthfully. No test metric or example may select
settings or the subset of methods reported as the final comparison.

`CPR Supervision: No` means no labels from this target benchmark were used to
fit/select that run. It does not mean an externally pretrained model is
training-free. A supervised RCR Proposed model and frozen transfer baselines
can share a benchmark table when their supervision is explicit; this is not an
isolation of architecture under matched training conditions.

## Remaining scientific requirements

- Instruct-ReID defaults to a native-distance unmatched value of -1,000,000,
  explicitly uncalibrated. It can dominate rankings when a person is missing.
  Keep this run diagnostic until an independently justified calibration or a
  validation-only sensitivity study is completed. Tables label uncalibrated
  results. Do not replace native Euclidean scoring with normalized cosine merely
  to reuse -1. Calibration checks image bytes against both val and test.
- Word4Per requires a reproduced Stage-2 checkpoint and matching training config.
  Report it as REPRODUCED, preserve its training/checkpoint-selection records,
  and verify it with the actual weights. Other configured checkpoint status
  labels describe intended sources; they do not prove local artifacts exist.
- The shared selector cap is five members per Subject. The inspected val export
  contains one six-member GROUP query. Test also contains one. This limits the
  representation of larger groups. Any change must be decided on val and then
  applied consistently to all shared-person baselines before a fresh test run.
- Native token budgets differ. CLIP-based methods may truncate long finalized
  instructions, and crop-only methods have limited relational context. Retain
  native preprocessing, disclose truncation/fallback counts, and label any
  alternative prompt/adapter as a separate validated variant.
- WISER and other editor/VLM systems have much larger GPU, disk and generation
  costs. Record model revisions, runtime, generated artifacts and API refiner
  identity. A GPT-4o alias and stochastic generation are not made deterministic
  merely by pinning local Hugging Face models.
- Audit overlap between external pretraining/reproduction/calibration data and
  the paper's data. Model-input allowlists prevent GT lookup in this code; they
  cannot establish the entire history of externally released weights.

Full pretrained inference, actual checkpoint availability, native-output parity,
GPU memory/runtime and statistical reliability must be verified using real
images and model assets. The CPU tests and synthetic evaluator comparison do
not establish those claims. Evaluate every test query, including localization
failures, and report failure counts rather than silently dropping queries.

## Reporting after this update

Run the normal pipeline with the exact finalized release and shared detection
index. Cache keys changed intentionally for content/precision-sensitive stages.
Old score matrices should not be relabeled with new metadata.

```bash
python run_baseline.py <method> --rcr-detection-cache /path/to/rcr/cache/detections
python run_baseline.py --tables-only
```

Evaluation writes score SHA256 and an evaluated-run digest into metrics.json.
The table builder rejects mismatched metric/run pairs, incomplete counts,
invalid metric values and different recorded image/box/membership identities.
Historical unbound metrics are skipped. Reporting validates recorded artifacts;
it does not rerun models or prove historical tuning choices.
