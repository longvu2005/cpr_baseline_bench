# Adding a baseline

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
methods/simple/clip_image/

Published method:
methods/published/word4per_setmatch/
methods/published/fafa_setmatch/
```

For a published baseline, also inspect the paper, official repository, official inference path, training/checkpoint protocol, and released checkpoint status when available.

Prefer adapting the official implementation instead of reimplementing the method from scratch.

---

### 2. Create the Method Directory

Use:

```text
methods/simple/<method_name>/
```

or:

```text
methods/published/<method_name>/
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
methods/published/new_method/
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

Do not train or fine-tune on validation/test inputs. Select adapter hyperparameters only on the complete validation split, then freeze them before test. Never select a checkpoint or configuration from test results.

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

The affected person adapters use the shared protocol in `cpr_person/`: maximum-sum rectangular Hungarian with exactly `min(K,N)` real pairs, followed by `(matched_sum + (K-min(K,N))*b) / K`. All Subjects can have multiple predicted members, including GROUP. Native model pair scores and preprocessing are preserved. Fixed, score-space-specific unmatched references are documented in [docs/baselines.md](baselines.md).

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

The tables sort methods alphabetically within groups; no manual ordering registration is needed.

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
[ ] final tables contain the method under its intended group
[ ] `python run_baseline.py <method_id>` is the normal end-to-end command
```

---

## Example End-to-End Flow

For a published method called `example_method`:

```text
methods/published/example_method/
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
→ python -m pip install -r methods/published/example_method/requirements.txt
→ methods/published/example_method/download_checkpoint.py
→ methods/published/example_method/run.py
→ evaluate.py --method example_method
→ build_tables.py
```

Individual stages may still be executed manually for debugging, but the end-to-end runner is the benchmark's normal execution path.
