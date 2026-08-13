# CPR Baseline Benchmark

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
evaluate.py
build_tables.py
validate_data.py
```

---

## Run One Baseline End-to-End

From the repository root, pass only the method name:

```bash
python run_baseline.py clip_image
```

The default runner performs the complete method pipeline in this order:

```text
1. install methods/<group>/<method>/requirements.txt
2. prepare/validate the method checkpoint when download_checkpoint.py exists
3. run inference
4. run the official evaluator
5. rebuild benchmark tables
```

The dependency step is always first and uses the same Python interpreter as the runner:

```text
python -m pip install -r methods/<group>/<method>/requirements.txt
```

This prevents checkpoint or inference scripts from silently depending on packages that have not been installed yet.

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
data/gallery.jsonl
data/queries.jsonl
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
SINGLE / MULTI / RELATIONAL behavior
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

Likewise, `run.py` must not silently download model weights, tokenizers, detector weights, official source code, or other runtime artifacts. Networked preparation belongs in `download_checkpoint.py`; inference should consume already prepared local artifacts and fail clearly when one is missing.

System tools that cannot be installed by pip, such as `git`, should fail with a clear error if missing.

---

### 4. Follow the Score-Matrix Contract

Every method must:

```text
read data/gallery.jsonl
read data/queries.jsonl
preserve their exact row order
score every query against the complete gallery
NOT remove the query image itself
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

The method must not calculate the official benchmark metrics itself.

`evaluate.py` owns query-image exclusion and official benchmark evaluation. Method code must preserve the complete canonical score matrix until evaluation.

---

### 5. Handle CPR Inputs Correctly

Use the canonical fields from `data/queries.jsonl`.

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

If a published method does not natively support scene-level target localization, `MULTI`, or `RELATIONAL`, define a deterministic adaptation and document it clearly.

Do not silently change the original retrieval method while still presenting it as the published baseline.

Do not use CPR evaluation labels, `target_ids`, positives, case annotations, or GT identity-to-box mappings to train, fine-tune, select checkpoints, tune hyperparameters, or secretly localize the target unless the experiment is explicitly defined as an oracle.

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

The current published adapters use maximum-weight one-to-one matching and an AND-style minimum over the assigned target scores, with an explicit unmatched score for missing person slots.

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
[ ] the query image is NOT removed inside the method
[ ] scores.npy has the exact required shape
[ ] scores.npy contains only finite values
[ ] run.json records reproducibility metadata and benchmark adaptations
[ ] no evaluation labels are used for hidden training/tuning/localization
[ ] the method appears in `python run_baseline.py --list`
[ ] dependency installation succeeds before checkpoint/inference work
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
python -m pip install -r methods/published/03_example_method/requirements.txt
→ methods/published/03_example_method/download_checkpoint.py
→ methods/published/03_example_method/run.py
→ evaluate.py --method example_method
→ build_tables.py
```

Individual stages may still be executed manually for debugging, but the end-to-end runner is the benchmark's normal execution path.
