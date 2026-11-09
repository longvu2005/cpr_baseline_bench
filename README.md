# CPR Baseline Benchmark

Seven additional methods are available: SigLIP2 Compose, SEARLE-XL,
Context-I2W, FTI4CIR, Qwen Caption-to-Caption, Structured Caption Retrieval,
and GME-Qwen2-VL-2B. See [setup, protocol and limitations](docs/additional_baselines.md).

Benchmark-owned scene CLIP uses RCR's official OpenAI ViT-B/32. Person
adapters read the same RCR Grounding DINO Tiny detection index and share the
CLIP Subject selector. The FAFA row uses the released FAFA model and Partial
SetMatch. See [baseline protocol and run commands](docs/baselines.md).

This repository provides a common evaluation protocol for Composed Person Retrieval (CPR) baselines.

Each method may keep its own implementation and method-specific dependencies, but every method must use the same canonical data, score-matrix contract, evaluator, and table builder.

## Repository Structure

```text
data/
checkpoints/
methods/
  simple/
  published/
runs/
outputs/
tables/
run_baseline.py
benchmark_progress.py
evaluate.py
build_tables.py
validate_data.py
```

---

## Benchmark Gallery Protocol

Use one completed RCR `dataset/data/final` export for all methods. The importer
creates separate val and test manifests in canonical RCR order.

```bash
python prepare_data.py --rcr-final-dir /path/to/rcr/dataset/data/final
python link_gallery.py --source /path/to/rcr/dataset/data/raw/images
python validate_data.py
```

Only val and test are imported. Validation is used for parameter selection.
All reported scores, metrics and tables use the test split. The overall table
includes Macro-Full-mAP, the equally weighted mean of Full-mAP across present
cases. The case table reports ID and Full mAP and R@1 in RCR case order.

Global CLIP early/late fusion, S6, S9, S10 and S11 select their configured
parameters by complete-validation Full-mAP and then score test with those
settings. S9 needs completed S5 and S8 runs. S11 needs S6's validation-selected
alpha and branch scores. Other methods use their predeclared configurations.

See [data layout and field mapping](data/README.md) for Kaggle mounts and the
split contract. Reimport after finalizing a new RCR release.

---

## Run One Baseline End-to-End

From the repository root, use the method name and supply the RCR detection index for person methods.

```bash
python run_baseline.py clip_image
python run_baseline.py fafa_setmatch --rcr-detection-cache /path/to/rcr/cache/detections
```

Before model work, the runner checks reporting imports with `build_tables.py --check`.
It then performs the complete method pipeline in this order:

```text
1. verify the imported val/test manifests and validate or link the image root
2. prepare the isolated method environment and verify any required person cache
3. prepare/validate the method checkpoint when download_checkpoint.py exists
4. select parameters on val where configured, then score test
5. evaluate test
6. rebuild benchmark tables
```

**Gallery validation runs before dependency installation.** Requirements are installed
into each method's isolated environment. Person methods use RCR's existing
`cache/detections/index.pt`. Supply its directory or the file itself with
`--rcr-detection-cache` or set `CPR_RCR_DETECTION_CACHE`. Scene-only methods such
as `clip_image` can run without it. See [detector setup](docs/baselines.md#detection-and-crop-convention).

To rebuild tables from existing evaluated outputs:

```bash
python run_baseline.py --tables-only
```

This reads `outputs/<method>/metrics.json` and adjacent `run.json`; it does not
install packages, load a model, repair gallery links or rerun evaluation.
`python -S build_tables.py --check` checks reporting imports without needing
manifests or outputs. See [reporting and provenance checks](docs/reporting.md).

All Python subprocesses launched by the root runner are unbuffered, so phase messages and progress bars appear immediately instead of being held until a process exits. Each `run.py` should use the shared `benchmark_progress.py` helpers for phase-level status and batch-level progress:

```python
from benchmark_progress import PhaseTracker, progress_bar
```

Use `PhaseTracker` for coarse stages such as model loading, gallery encoding, query encoding, scoring, validation, and output writing. Use `progress_bar` only around meaningful repeated work such as image, batch, query, or person-batch loops. Progress reporting must not change ordering, batching semantics, scores, caching rules, or benchmark outputs.

Use a dedicated virtual/conda environment for a baseline when its dependencies may conflict with another method.

List discovered methods with:

```bash
python run_baseline.py --list
```

The runner discovers methods from their `config.yaml`. A configured method is considered incomplete if either `requirements.txt` or `run.py` is missing, and discovery fails clearly instead of silently hiding the method.

For debugging only, installation can be skipped when the active environment is already prepared:

```bash
python run_baseline.py clip_image --skip-install
```

To re-run a method checkpoint preparer that supports replacement/re-download:

```bash
python run_baseline.py <method_id> --force-checkpoint
```

---

## Adding a New Baseline

Adding a method means integrating the complete benchmark pipeline, not only making an inference script run.

### 1. Inspect the Benchmark First

Before coding, inspect:

```text
README.md
data/README.md
data/test/gallery.jsonl
data/test/queries.jsonl
validate_data.py
evaluate.py
build_tables.py
run_baseline.py
```

Then inspect the closest existing method:

```text
Simple method:
methods/simple/01_clip_image/

Published method:
methods/published/01_word4per_setmatch/
methods/published/02_fafa_setmatch/
```

For a published baseline, also inspect the paper, official repository, official inference path, training/checkpoint protocol, and released checkpoint status when available.

Prefer adapting the official implementation instead of reimplementing the method from scratch.

---

### 2. Create the Method Directory

Use:

```text
methods/simple/<NN_method_name>/
```

or:

```text
methods/published/<NN_method_name>/
```

Required files for every runnable method:

```text
config.yaml
requirements.txt
run.py
```

Add these when applicable:

```text
download_checkpoint.py   # external checkpoint/artifact preparation
README.md                # required for published or non-trivial adapters
```

Example:

```text
methods/published/03_new_method/
├── config.yaml
├── requirements.txt
├── download_checkpoint.py
├── run.py
└── README.md
```

Keep model parameters, checkpoint paths, method settings, and runtime settings in `config.yaml`.

The configuration must contain one stable benchmark identifier:

```yaml
method: new_method
```

The same identifier must be used by:

```text
runs/<method_id>/
outputs/<method_id>/
evaluate.py --method <method_id>
```

For published methods, document at least:

```text
paper
official repository
source commit
checkpoint source/status
original backbone
what is preserved from official code
what is adapted for this CPR benchmark
INDIVIDUAL / DUAL / GROUP / RELATIONAL behavior
```

Clearly distinguish:

```text
OFFICIAL_RELEASED
VERIFIED_MIRROR
REPRODUCED
```

Never label a reproduced or third-party checkpoint as an official released checkpoint.

---

### 3. Declare All Method Dependencies

`requirements.txt` is part of the runnable method contract.

Every Python package needed by any method-local stage must be declared there, including packages needed only by `download_checkpoint.py`.

Do not make checkpoint or inference scripts install their own missing packages at runtime. The root runner installs the method requirements before those scripts are executed.

`run.py` must also expose meaningful progress for long-running work. At minimum, report major phases and add a progress bar to any potentially long gallery/query/batch loop. Cache hits should be stated explicitly so a fast skipped phase is distinguishable from a missing computation.

Likewise, `run.py` must not silently download model weights, tokenizers, detector weights, official source code, or other runtime artifacts. Networked preparation belongs in `download_checkpoint.py`; inference should consume already prepared local artifacts and fail clearly when one is missing.

System tools that cannot be installed by pip, such as `git`, should fail with a clear error if missing.

---

### 4. Follow the Score-Matrix Contract

Every method must:

```text
read data/test/gallery.jsonl
read data/test/queries.jsonl
preserve their exact row order
score every query against the complete gallery
keep a finite placeholder for the query image column
save scores.npy
save run.json
```

Required raw output:

```text
runs/<method_id>/
├── scores.npy
└── run.json
```

`scores.npy` must satisfy:

```python
scores.shape == (len(queries), len(gallery))
```

with:

```text
scores[q, g] = retrieval score of gallery row g for query row q
higher score = better match
```

The matrix must contain only finite values: no `NaN`, `+Inf`, or `-Inf`.

The evaluator computes the official test metrics.

`evaluate.py` excludes the query image and sorts ties in canonical gallery order. Late fusion and rerankers exclude the reference before gallery statistics or candidate selection. Each method still writes the complete test score matrix.

---

### 5. Handle CPR Inputs Correctly

Use the canonical fields from `data/test/queries.jsonl`.

For ordinary methods, the query-level field:

```text
text
```

may be sufficient.

For component-aware methods, inspect fields such as:

```text
subjects[].select_text
subjects[].modify_text
relation_text
```

Use only inputs that are justified by the original method or by an explicitly documented benchmark adapter.

If a published method does not natively support scene-level target localization, `DUAL`, `GROUP`, or `RELATIONAL`, define a deterministic adaptation and document it clearly.

Do not silently change the original retrieval method while still presenting it as the published baseline.

Use public inputs for inference. Select any benchmark hyperparameters on the complete val split and freeze them before scoring test. The test evaluator uses identities, case labels and reviewed positives.

---

### 6. Checkpoints and External Source Code

Store weights under:

```text
checkpoints/<method_or_model>/
```

Do not commit model weights.

Operational checkpoint preparation belongs to:

```text
methods/<group>/<method>/download_checkpoint.py
```

The root runner calls this script after installing `requirements.txt` and before inference.

A checkpoint preparer should:

```text
create the expected checkpoint directory
obtain or validate every required artifact
skip an already valid artifact
support repeated execution safely
avoid accepting partial downloads as valid checkpoints
prepare auxiliary pretrained weights/tokenizers/source checkouts required by inference, not only the final method checkpoint
validate checksum/structure when practical
support --force when replacement is meaningful
fail clearly when preparation cannot be completed automatically
```

If the final published checkpoint is unavailable, the preparer may validate a required `REPRODUCED` artifact and explain exactly what must be reproduced. It must not pretend that an unavailable final checkpoint can be downloaded officially.

For methods sharing the same pretrained model, the physical weight may be shared under `checkpoints/`, while each method must still resolve the artifact deterministically.

Caches that contain model-dependent features or scores must be keyed by enough immutable inputs to prevent stale reuse after a checkpoint, config, manifest, adapter, or auxiliary model changes. Shape-only cache validation is not sufficient.

For published methods, pin the imported official repository to an exact commit when practical.

Training, fine-tuning, checkpoint selection, or hyperparameter tuning on the CPR evaluation data is not allowed.

---

### 7. Localization and Set-Valued Methods

The benchmark gallery contains scene images that may contain more than one person. A published method that expects one cropped person image cannot be applied to the whole scene and then described as person-level SetMatch.

If person instances are required by the adapter:

```text
use predicted person instances for the normal benchmark
record the detector and target-selection procedure in run.json
avoid target_ids / positives / GT identity-to-box mapping
keep oracle-box experiments separate and explicitly labeled
```

For adapters named `+ SetMatch`, the matching rule must be documented precisely, including:

```text
person-instance construction
query-target construction
one-to-one assignment rule
aggregation of assigned target scores
behavior when a gallery image has fewer persons than targets
```

The affected person adapters use the shared protocol in `cpr_person/`: maximum-sum rectangular Hungarian with exactly `min(K,N)` real pairs, followed by `(matched_sum + (K-min(K,N))*b) / K`. All Subjects can have multiple predicted members, including GROUP. Native model pair scores and preprocessing are preserved. Fixed, score-space-specific unmatched references are documented in [docs/baselines.md](docs/baselines.md).

---

### 8. Validate the Integration

First validate the canonical data:

```bash
python validate_data.py
```

Then verify discovery:

```bash
python run_baseline.py --list
```

Run static syntax checks on the method files:

```bash
python -m py_compile methods/<group>/<method>/download_checkpoint.py
python -m py_compile methods/<group>/<method>/run.py
```

When `download_checkpoint.py` is not required, compile only `run.py`.

Then run the complete pipeline:

```bash
python run_baseline.py <method_id>
```

Expected evaluated outputs:

```text
runs/<method_id>/scores.npy
runs/<method_id>/run.json

outputs/<method_id>/metrics.json
outputs/<method_id>/run.json

tables/table1_main.csv
tables/table2_cases.csv
```

Add the new method to `method_order` in `build_tables.py` when explicit table ordering is required.

---

## Definition of Done

A baseline integration is complete only when:

```text
[ ] the original paper/code/checkpoint protocol was inspected when applicable
[ ] config.yaml is present and has a stable method id
[ ] requirements.txt contains every method-local Python dependency
[ ] run.py is present
[ ] download_checkpoint.py is present when external checkpoint preparation is required
[ ] README.md documents published/non-trivial adapters
[ ] official / mirrored / reproduced checkpoint status is accurate
[ ] official source code is pinned when practical
[ ] canonical query/gallery order is preserved
[ ] every query scores the complete gallery
[ ] the score matrix contains a finite query-image entry and eligible shortlists exclude it
[ ] scores.npy has the exact required shape
[ ] scores.npy contains only finite values
[ ] run.json records reproducibility metadata and benchmark adaptations
[ ] no evaluation labels are used for hidden training/tuning/localization
[ ] the method appears in `python run_baseline.py --list`
[ ] dependency installation succeeds before checkpoint/inference work
[ ] runner/subprocess output is unbuffered and visible while the baseline is running
[ ] run.py reports major phases and progress for long gallery/query/batch loops
[ ] cache hits are reported clearly rather than looking like skipped/missing work
[ ] checkpoint preparation succeeds or fails early with a truthful actionable message
[ ] inference performs no silent network download of model/runtime artifacts
[ ] model-dependent caches are invalidated when their inputs change
[ ] evaluate.py succeeds
[ ] build_tables.py succeeds
[ ] final tables contain the method in the intended order/group
[ ] `python run_baseline.py <method_id>` is the normal end-to-end command
```

---

## Example End-to-End Flow

For a published method called `example_method`:

```text
methods/published/03_example_method/
├── config.yaml
├── requirements.txt
├── download_checkpoint.py
├── run.py
└── README.md
```

The normal user-facing command is:

```bash
python run_baseline.py example_method
```

Internally the runner performs:

```text
validate/repair gallery and validate manifests
→ python -m pip install -r methods/published/03_example_method/requirements.txt
→ methods/published/03_example_method/download_checkpoint.py
→ methods/published/03_example_method/run.py
→ evaluate.py --method example_method
→ build_tables.py
```

Individual stages may still be executed manually for debugging, but the end-to-end runner is the benchmark's normal execution path.


### Baseline Python environments (Kaggle)

`python run_baseline.py <method>` automatically creates or reuses
`.baseline_venvs/<method_id>/`. Dependencies are installed only into that
method's environment and are rechecked when requirements files change. The
checkpoint preparation and inference steps use **the same** method interpreter.
The host interpreter is kept for data validation, official evaluation, and
benchmark tables. Torch/CUDA from the host remains visible using
`--system-site-packages`, but conflicting pip packages are installed locally
inside the method's virtual environment. PyTorch-only subprocesses also disable
Transformers' optional Flax/JAX and TensorFlow backend discovery.

Run normally (without `--skip-install`) at least once for each method:

```bash
python run_baseline.py 10_person_compose_setmatch --rcr-detection-cache /path/to/rcr/cache/detections
```

`--skip-install` only reuses an **already prepared**, up-to-date method
environment and fails early with an actionable message if one is missing.
A successful install is recorded under `.baseline_venvs/` and reused on
later runs; the existing `checkpoints/` and `runs/` data are not cleared.

To refresh an environment, remove only `.baseline_venvs/<method_id>/` and
run the normal command again. This does not delete model checkpoints or feature
caches.
