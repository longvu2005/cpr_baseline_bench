## Adding a New Baseline

When adding a new method, **do not only implement `run.py`**. Complete the whole benchmark integration.

### 1. Inspect First

Before coding, inspect:

```text
README.md
data/gallery.jsonl
data/queries.jsonl
evaluate.py
build_tables.py
```

Then inspect the closest existing baseline:

```text
Simple method:
methods/simple/01_clip_image/

Published method:
methods/published/01_word4per_setmatch/
```

For a published method, also inspect the **paper, official repository, official inference code, and checkpoint**. Prefer adapting the official implementation instead of reimplementing the method from scratch.

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

Required files:

```text
config.yaml
requirements.txt
run.py
README.md        # required for published/non-trivial methods
```

Example:

```text
methods/published/02_new_method/
├── config.yaml
├── requirements.txt
├── run.py
└── README.md
```

Keep important model parameters, checkpoint paths, method settings, and runtime settings in `config.yaml`.

For published methods, document:

```text
paper
official repository
source commit
checkpoint source/status
what is preserved from official code
what is adapted for CPR
SINGLE / MULTI / RELATIONAL behavior
```

Do not claim a reproduced checkpoint is an official checkpoint.

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

The baseline must **not** calculate the official benchmark metrics itself. `evaluate.py` handles query-image exclusion and evaluation.

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

If a published method does not directly support `MULTI` or `RELATIONAL`, define a deterministic benchmark adaptation and document it clearly. Do not silently modify the original method.

---

### 5. Checkpoints and External Code

Store model weights under:

```text
checkpoints/
```

and document new checkpoints in:

```text
checkpoints/README.md
```

Do not commit model weights.

For published methods, preferably pin the official repository to an exact commit.

Training, fine-tuning, checkpoint selection, or hyperparameter tuning on the CPR evaluation data is not allowed.

---

### 6. Register and Validate the Method

Run:

```bash
python validate_data.py

python -m py_compile methods/<group>/<method>/run.py

pip install -r methods/<group>/<method>/requirements.txt

python methods/<group>/<method>/run.py

python evaluate.py --method <method_id>

python build_tables.py
```

Also add the new `method_id` to `method_order` in:

```text
build_tables.py
```

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
[ ] run.py is added
[ ] README.md is added for published/non-trivial methods
[ ] checkpoint and source provenance are documented
[ ] canonical query/gallery ordering is preserved
[ ] the complete gallery is scored
[ ] the query image is NOT removed by the method
[ ] scores.npy has the correct shape
[ ] scores.npy contains only finite values
[ ] run.json contains reproducibility metadata
[ ] evaluate.py succeeds
[ ] method_order is updated
[ ] build_tables.py succeeds
[ ] final benchmark tables contain the new method
```

---

### Example

For a new published method called `example_method`:

```text
methods/published/02_example_method/
├── config.yaml
├── requirements.txt
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

Run:

```bash
python methods/published/02_example_method/run.py
python evaluate.py --method example_method
python build_tables.py
```

> **Rule:** Adding a method means integrating it into the complete benchmark pipeline, not merely making its inference code run.