## Run One Baseline End-to-End

From the repository root, pass only the method name:

```bash
python run_baseline.py clip_image
```

The runner automatically performs:

```text
method-local checkpoint preparation
→ inference
→ official evaluation
→ table rebuilding
```

When the method provides:

```text
download_checkpoint.py
```

the runner executes it before `run.py`.

List discovered method names with:

```bash
python run_baseline.py --list
```

The runner discovers methods from their `config.yaml`, so newly integrated methods do not need to be manually registered in the runner.

---

## Adding a New Baseline

When adding a new method, **do not only implement `run.py`**. Complete the whole benchmark integration.

### 1. Inspect First

Before coding, inspect:

```text
README.md
data/README.md
data/gallery.jsonl
data/queries.jsonl
evaluate.py
build_tables.py
run_baseline.py
```

Then inspect the closest existing baseline:

```text
Simple method:
methods/simple/01_clip_image/

Published method:
methods/published/01_word4per_setmatch/
```

For a published method, also inspect the **paper, official repository, official inference code, training/checkpoint protocol, and released checkpoint when available**.

Prefer adapting the official implementation instead of reimplementing the method from scratch.

---

### 2. Create the Method

Use:

```text
methods/simple/<NN_method_name>/
```

or:

```text
methods/published/<NN_method_name>/
```

Recommended files:

```text
config.yaml
requirements.txt
download_checkpoint.py   # required when external checkpoints/artifacts are needed
run.py
README.md                # required for published/non-trivial methods
```

Example:

```text
methods/published/02_new_method/
├── config.yaml
├── requirements.txt
├── download_checkpoint.py
├── run.py
└── README.md
```

Keep important model parameters, checkpoint paths, method settings, and runtime settings in `config.yaml`.

The configuration must contain a stable method identifier:

```yaml
method: new_method
```

For published methods, document:

```text
paper
official repository
source commit
checkpoint source/status
original backbone
what is preserved from official code
what is adapted for CPR
SINGLE / MULTI / RELATIONAL behavior
```

Clearly distinguish between:

```text
official checkpoint
verified mirror
reproduced checkpoint
```

Do not claim a reproduced or third-party checkpoint is an official checkpoint.

---

### 3. Follow the Benchmark Contract

Every method must:

```text
read data/gallery.jsonl
read data/queries.jsonl
preserve their exact order
score every query against the complete gallery
NOT remove the query image
save scores.npy
save run.json
```

Required output:

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

Scores must contain no `NaN` or `Inf`.

The baseline must **not** calculate the official benchmark metrics itself.

`evaluate.py` handles query-image exclusion and official benchmark evaluation.

Method code must preserve the complete canonical score matrix before evaluation.

---

### 4. Handle CPR Inputs Correctly

Use the canonical fields from `queries.jsonl`.

For ordinary methods, the query-level:

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

Use only the inputs that are appropriate for the original method.

If a published method does not directly support `MULTI` or `RELATIONAL`, define a deterministic benchmark adaptation and document it clearly.

Do not silently modify the original method.

Do not use CPR evaluation labels, positives, or case annotations to train, fine-tune, select checkpoints, or tune method hyperparameters.

---

### 5. Checkpoints and External Code

Store model weights under:

```text
checkpoints/<method_or_model>/
```

Do not commit model weights.

Checkpoint preparation should belong to the method itself:

```text
methods/<group>/<method>/download_checkpoint.py
```

After:

```bash
python methods/<group>/<method>/download_checkpoint.py
```

finishes successfully, the corresponding:

```bash
python methods/<group>/<method>/run.py
```

must be runnable without additional undocumented checkpoint preparation.

A checkpoint preparer should:

```text
create the expected checkpoint directory
download or reproduce every required artifact
skip already valid artifacts
support repeated execution safely
avoid treating partial downloads as valid checkpoints
validate checkpoint structure when practical
support --force when appropriate
fail clearly when preparation cannot be completed
```

For methods sharing the same pretrained model, the physical checkpoint may be shared under `checkpoints/`, but every method must still resolve the correct artifact deterministically.

For published methods, preferably pin the official repository to an exact commit.

If the official final checkpoint is unavailable, a reproducible preparation pipeline may rebuild it from the official training procedure and permitted external training data.

Training, fine-tuning, checkpoint selection, or hyperparameter tuning on the CPR evaluation data is not allowed.

Operational checkpoint download/reproduction commands should live in `download_checkpoint.py`, not in the root README.

---

### 6. Register and Validate the Method

First verify that the method can be discovered:

```bash
python run_baseline.py --list
```

Then run basic checks:

```bash
python validate_data.py

python -m py_compile methods/<group>/<method>/download_checkpoint.py
python -m py_compile methods/<group>/<method>/run.py
```

If the method has additional dependencies:

```bash
pip install -r methods/<group>/<method>/requirements.txt
```

`run_baseline.py` intentionally does **not** automatically install method-specific requirements.

Then run the complete pipeline:

```bash
python run_baseline.py <method_id>
```

This executes:

```text
download_checkpoint.py
→ run.py
→ evaluate.py
→ build_tables.py
```

If the method does not require external checkpoints, `download_checkpoint.py` may be omitted and the runner proceeds directly to inference.

Also add the new `method_id` to `method_order` in:

```text
build_tables.py
```

when explicit table ordering is required.

Verify that the following files are produced:

```text
runs/<method_id>/scores.npy
runs/<method_id>/run.json

outputs/<method_id>/metrics.json
outputs/<method_id>/run.json

tables/table1_main.csv
tables/table2_cases.csv
```

---

### Definition of Done

A new baseline is complete only when:

```text
[ ] the original method/paper/code has been inspected when applicable
[ ] config.yaml is added
[ ] requirements.txt is added
[ ] download_checkpoint.py is added when external artifacts are required
[ ] checkpoint preparation leaves the method immediately runnable
[ ] run.py is added
[ ] README.md is added for published/non-trivial methods
[ ] checkpoint and source provenance are documented
[ ] official / mirrored / reproduced checkpoints are distinguished correctly
[ ] canonical query/gallery ordering is preserved
[ ] the complete gallery is scored
[ ] the query image is NOT removed by the method
[ ] scores.npy has the correct shape
[ ] scores.npy contains only finite values
[ ] run.json contains reproducibility metadata
[ ] the method appears in `python run_baseline.py --list`
[ ] evaluate.py succeeds
[ ] method_order is updated when required
[ ] build_tables.py succeeds
[ ] final benchmark tables contain the new method
[ ] `python run_baseline.py <method_id>` completes the full pipeline
```

---

### Example

For a new published method called `example_method`:

```text
methods/published/02_example_method/
├── config.yaml
├── requirements.txt
├── download_checkpoint.py
├── run.py
└── README.md

checkpoints/example_method/...

runs/example_method/
├── scores.npy
└── run.json

outputs/example_method/
├── metrics.json
└── run.json
```

The normal user-facing command is:

```bash
python run_baseline.py example_method
```

Internally this performs:

```text
methods/published/02_example_method/download_checkpoint.py
→ methods/published/02_example_method/run.py
→ evaluate.py --method example_method
→ build_tables.py
```

For debugging, the individual stages may still be executed manually:

```bash
python methods/published/02_example_method/download_checkpoint.py
python methods/published/02_example_method/run.py
python evaluate.py --method example_method
python build_tables.py
```

> **Rule:** Adding a method means integrating it into the complete benchmark pipeline — checkpoint preparation, inference, official evaluation, and table generation — not merely making its inference code run.