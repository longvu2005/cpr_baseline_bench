# CPR Baseline Benchmark

This repository provides a common evaluation protocol for **Composed Person Retrieval (CPR)** baseline methods.

The goal is to compare heterogeneous retrieval methods under the same data, scoring, evaluation, and reporting protocol.

Each baseline may use its own implementation, model, dependencies, and runtime configuration. Baselines should remain independent mini-projects rather than being forced into a shared framework.

However, every method must obey the same:

- data manifests;
- query/gallery ordering;
- score-matrix contract;
- evaluation protocol;
- result format;
- benchmark table protocol.

---

## 1. Repository Structure

```text
data/
├── gallery/
├── gallery.jsonl
├── queries.jsonl
└── source/

checkpoints/
└── README.md

methods/
├── simple/
└── published/

runs/
outputs/
tables/

validate_data.py
evaluate.py
build_tables.py
README.md
```

The directories have different purposes:

```text
methods/      Method implementations
checkpoints/  Local pretrained weights
runs/         Raw method outputs and caches
outputs/      Compact evaluated benchmark results
tables/       Aggregated benchmark tables
```

### Git policy

Commit:

```text
methods/
outputs/
tables/
data/*.jsonl
checkpoints/README.md
```

Do not commit:

```text
model checkpoints
gallery images
scores.npy
feature caches
large intermediate files
```

---

## 2. Benchmark Protocol

This repository currently evaluates methods on a fixed CPR pilot evaluation pool.

The pilot is used for evaluation only.

Do not:

- train on the CPR pilot;
- fine-tune on the CPR pilot;
- tune method hyperparameters using pilot retrieval results;
- modify the benchmark protocol separately for individual methods.

Use official pretrained or released checkpoints whenever possible.

Runtime parameters such as batch size, number of workers, and score batch size may be changed for memory or throughput, but must not be tuned according to retrieval performance.

---

## 3. Common Data

Every baseline must use:

```text
data/gallery/
data/gallery.jsonl
data/queries.jsonl
```

The manifests define the canonical ordering.

### Gallery ordering

Column `g` of every score matrix corresponds exactly to line `g` in:

```text
data/gallery.jsonl
```

### Query ordering

Row `q` corresponds exactly to line `q` in:

```text
data/queries.jsonl
```

Do not reorder either manifest inside baseline implementations.

---

## 4. Query Image Policy

The original query image remains physically inside the global gallery.

Baseline code must score the complete gallery and must **not remove the query image**.

The common evaluator handles exact query-image exclusion.

Therefore:

```text
baseline:
    score complete gallery

evaluate.py:
    exclude exact query image
    compute metrics
```

This behavior must remain identical for every method.

---

## 5. Adding a New Baseline

A new method must be added under one of:

```text
methods/simple/<NN_method_name>/
```

or:

```text
methods/published/<NN_method_name>/
```

Use sequential numeric prefixes for repository organization.

Example:

```text
methods/simple/
├── 01_clip_image/
├── 02_clip_text/
├── 03_clip_early_fusion/
└── 04_clip_late_fusion/
```

The numeric prefix is for repository organization.

The actual method identifier used by the evaluator must not contain the numeric prefix unless explicitly intended.

Example:

```text
folder:
04_clip_late_fusion

method id:
clip_late_fusion
```

---

## 6. Required Method Structure

Each method should contain:

```text
methods/<group>/<NN_method_name>/
├── config.yaml
├── requirements.txt
├── run.py
└── README.md
```

`README.md` inside a method is optional for trivial baselines but recommended for published or non-trivial methods.

Each baseline should be runnable independently.

Do not create a large shared baseline framework unless there is a strong reason to do so.

Small duplication between baseline implementations is acceptable when it keeps methods self-contained and reproducible.

---

## 7. Method Configuration

Method and runtime settings belong in:

```text
config.yaml
```

Example:

```yaml
method: clip_image

model:
  name: ViT-B/16
  checkpoint: checkpoints/clip/ViT-B-16.pt

runtime:
  device: cuda
  batch_size: 256
  score_batch_size: 512
  num_workers: 4

output:
  dir: runs/clip_image
```

If a method reuses a compatible feature cache, declare it explicitly.

Example:

```yaml
cache:
  gallery_features: runs/clip_image/gallery_features.npy
```

Method-specific parameters should also be explicit.

Example:

```yaml
fusion:
  image_weight: 0.5
  text_weight: 0.5
```

Do not silently hard-code important method parameters inside `run.py`.

---

## 8. Dependencies

Each method must declare its dependencies in its own:

```text
requirements.txt
```

Example:

```text
numpy
pillow
pyyaml
torch
tqdm
git+https://github.com/openai/CLIP.git
```

On Kaggle or another fresh environment, install dependencies with:

```bash
pip install -r methods/<group>/<method>/requirements.txt
```

A baseline should not silently depend on packages installed by a previous baseline.

---

## 9. Checkpoints

Do not commit model weights to GitHub.

Store local weights under:

```text
checkpoints/
```

Every external checkpoint used by a method must be documented in:

```text
checkpoints/README.md
```

For every checkpoint record:

- model name;
- official source;
- download URL or official repository;
- expected local path;
- relevant version if applicable.

Example:

```text
Model:
OpenAI CLIP ViT-B/16

Expected path:
checkpoints/clip/ViT-B-16.pt
```

Prefer official pretrained/released checkpoints.

For published methods, reproduce the authors' released inference configuration as closely as possible.

---

## 10. Required Baseline Output

Every method must create:

```text
runs/<method_id>/
├── scores.npy
└── run.json
```

Additional local artifacts are allowed:

```text
gallery_features.npy
query_features.npy
generated_images/
detections/
rerank_cache/
```

These belong under `runs/` and are not benchmark results.

---

## 11. Score Matrix Contract

`scores.npy` must have shape:

```text
[num_queries, num_gallery]
```

where:

```text
scores[q, g]
```

is the retrieval score assigned to gallery image `g` for query `q`.

The matrix must satisfy:

```text
rows    = data/queries.jsonl order
columns = data/gallery.jsonl order
```

Higher scores must always mean better matches.

Do not return only Top-K results.

Every baseline must score the complete gallery.

---

## 12. run.json Contract

Each method must save sufficient metadata for reproducibility.

Recommended structure:

```json
{
  "method": "clip_image",
  "display_name": "CLIP ViT-B/16 - Image-only",
  "group": "Simple / Obvious Baselines",
  "cpr_supervision": "No",
  "model": {},
  "runtime": {},
  "config": "methods/simple/01_clip_image/config.yaml",
  "num_queries": 2975,
  "num_gallery": 17000,
  "scores": "runs/clip_image/scores.npy",
  "higher_is_better": true
}
```

Method-specific metadata may be added.

Examples:

```json
"fusion": {}
```

```json
"gallery_features": "runs/clip_image/gallery_features.npy"
```

The `method` field is the canonical method identifier used throughout evaluation and table generation.

---

## 13. Evaluation

Individual baselines must not implement their own final benchmark metrics.

After producing the score matrix, always run:

```bash
python evaluate.py --method <method_id>
```

Example:

```bash
python evaluate.py --method clip_image
```

The evaluator reads:

```text
runs/<method>/scores.npy
runs/<method>/run.json
data/gallery.jsonl
data/queries.jsonl
```

and writes:

```text
outputs/<method>/
├── metrics.json
└── run.json
```

Only the common evaluator defines official benchmark metrics.

---

## 14. Official Metrics

### Identity Retrieval

Report:

```text
ID-mAP
ID-R@1
ID-R@5
ID-R@10
```

For MULTI queries, identity relevance uses the strict intersection of all target identities.

Partial identity matches do not count as full identity relevance.

### Full CPR Retrieval

Report:

```text
Full-mAP
Full-R@1
Full-R@5
Full-R@10
```

### Case-wise Full CPR

Also report:

```text
SINGLE:
    Full-mAP
    Full-R@1

MULTI:
    Full-mAP
    Full-R@1

RELATIONAL:
    Full-mAP
    Full-R@1
```

Do not redefine these metrics inside individual methods.

---

## 15. Building Benchmark Tables

After evaluating one or more methods, run:

```bash
python build_tables.py
```

This generates:

```text
tables/table1_main.csv
tables/table2_cases.csv
```

### Table 1

Main retrieval metrics:

```text
Method
CPR Supervision
ID-mAP
ID-R@1
ID-R@5
ID-R@10
Full-mAP
Full-R@1
Full-R@5
Full-R@10
```

### Table 2

Case-wise Full CPR metrics:

```text
Method
SINGLE mAP
SINGLE R@1
MULTI mAP
MULTI R@1
RELATIONAL mAP
RELATIONAL R@1
```

---

## 16. Method Ordering

Benchmark tables must follow the intended baseline order rather than alphabetical display-name order.

When adding a new method, also update `method_order` inside:

```text
build_tables.py
```

Example:

```python
method_order = {
    "clip_image": 1,
    "clip_text": 2,
    "clip_early_fusion": 3,
    "clip_late_fusion": 4,
}
```

A newly added method must receive the next appropriate position.

Do not sort benchmark methods purely by display name.

---

## 17. Feature Cache Reuse

Methods may reuse computationally equivalent feature caches.

For example:

```text
runs/clip_image/gallery_features.npy
```

may be reused by other methods using exactly the same CLIP image encoder and preprocessing.

Cache reuse is allowed only when the features are mathematically identical.

A method should still be able to run independently when practical.

Preferred behavior:

```text
compatible cache exists
    -> reuse cache

cache missing
    -> compute features
    -> save local fallback cache
```

Do not reuse caches across incompatible:

- checkpoints;
- preprocessing;
- model variants;
- resolutions;
- feature definitions.

---

## 18. Simple vs Published Baselines

Use:

```text
methods/simple/
```

for obvious or constructed reference baselines.

Examples:

```text
CLIP Image-only
CLIP Text-only
CLIP Early Fusion
CLIP Late Fusion
```

Use:

```text
methods/published/
```

for methods derived from published papers or official released systems.

For published methods:

1. find the official paper;
2. find the official repository;
3. prefer official released checkpoints;
4. identify the official inference pipeline;
5. document any adaptation required for CPR;
6. avoid unnecessary reimplementation;
7. do not train on the CPR pilot unless the benchmark protocol is explicitly changed.

Any important deviation from the published method must be documented.

---

## 19. Validation Before Running a New Method

Before expensive inference, verify:

```bash
python validate_data.py
```

Then verify the new method at minimum for:

```text
config loads successfully
checkpoint exists
dependencies install
query count matches manifest
gallery count matches manifest
scores shape is correct
scores contain finite values
higher score means better match
run.json is created
```

After inference:

```bash
python evaluate.py --method <method_id>
python build_tables.py
```

Do not consider a baseline complete before both commands succeed.

---

## 20. Complete Workflow for Adding a Method

When adding a new baseline, perform **all** of the following steps.

### Step 1 - Identify the method

Decide:

```text
simple or published
folder name
method_id
display_name
official model/checkpoint
inference definition
```

For published work, inspect the official paper and repository before implementing it.

### Step 2 - Inspect existing baselines

Before writing code, inspect neighboring method implementations.

Preserve the current repository conventions for:

```text
config structure
run.py structure
requirements.txt
run.json metadata
cache handling
output paths
method naming
```

Do not invent a new architecture when the existing pattern is sufficient.

### Step 3 - Create the method folder

```text
methods/<group>/<NN_method_name>/
```

Create:

```text
config.yaml
requirements.txt
run.py
README.md if needed
```

### Step 4 - Add checkpoint documentation

If a new model is required, update:

```text
checkpoints/README.md
```

Do not commit the checkpoint itself.

### Step 5 - Implement inference

The implementation must:

```text
load canonical query manifest
load canonical gallery manifest
preserve their ordering
score the entire gallery
not exclude the query image
save scores.npy
save run.json
```

### Step 6 - Validate implementation

Check syntax:

```bash
python -m py_compile methods/<group>/<method>/run.py
```

Check data when necessary:

```bash
python validate_data.py
```

### Step 7 - Install method dependencies

```bash
pip install -r methods/<group>/<method>/requirements.txt
```

### Step 8 - Run baseline

```bash
python methods/<group>/<method>/run.py
```

### Step 9 - Evaluate

```bash
python evaluate.py --method <method_id>
```

### Step 10 - Add method ordering

Update:

```text
build_tables.py
```

so the method appears in the intended benchmark order.

### Step 11 - Rebuild tables

```bash
python build_tables.py
```

### Step 12 - Inspect results

Verify that these exist:

```text
outputs/<method>/metrics.json
outputs/<method>/run.json
tables/table1_main.csv
tables/table2_cases.csv
```

### Step 13 - Commit only reproducible code and compact results

Commit:

```text
method implementation
config
requirements
method documentation
checkpoint documentation
outputs/<method>/
updated tables
build_tables.py if ordering changed
```

Do not commit:

```text
scores.npy
feature caches
checkpoints
gallery images
generated temporary artifacts
```

---

## 21. Definition of Done

A new baseline is complete only when all of the following are true:

```text
[ ] Method source and official implementation inspected
[ ] Method folder created
[ ] config.yaml created
[ ] requirements.txt created
[ ] run.py implemented
[ ] Official checkpoint documented
[ ] Full gallery is scored
[ ] Query/gallery ordering is preserved
[ ] Query image is not removed by baseline
[ ] scores.npy has correct shape
[ ] scores contain valid finite values
[ ] run.json is created
[ ] evaluate.py succeeds
[ ] metrics.json is created
[ ] build_tables.py method order is updated
[ ] build_tables.py succeeds
[ ] Table 1 contains the method
[ ] Table 2 contains the method
[ ] Raw scores/checkpoints/images are not committed
[ ] Compact outputs are ready for GitHub
```

---

## 22. Instructions for AI Assistants

When this README is provided to an AI assistant and the user asks to **add a new baseline**, treat this document as the repository specification.

Do not stop after implementing `run.py`.

Complete the full baseline integration:

```text
method identification
        |
        v
official paper / repository / checkpoint
        |
        v
inspect neighboring baselines
        |
        v
create method folder
        |
        +-- config.yaml
        +-- requirements.txt
        +-- run.py
        +-- README.md if needed
        |
        v
document checkpoint
        |
        v
validate implementation
        |
        v
run baseline
        |
        v
runs/<method>/
        |
        +-- scores.npy
        `-- run.json
        |
        v
evaluate.py
        |
        v
outputs/<method>/
        |
        +-- metrics.json
        `-- run.json
        |
        v
update method_order
        |
        v
build_tables.py
        |
        v
tables/
        |
        v
compact reproducible benchmark result
```

Before proposing or modifying code:

1. inspect the existing neighboring baseline implementations;
2. preserve the repository's current style and conventions;
3. inspect the official source for published methods;
4. determine whether an existing feature cache is mathematically compatible;
5. determine checkpoint requirements;
6. determine required dependencies;
7. verify the score-matrix contract.

Do not:

- invent a new architecture when the current pattern works;
- modify the common evaluator for convenience;
- silently change metric definitions;
- silently change query/gallery ordering;
- remove the query image inside a baseline;
- tune method hyperparameters on the CPR pilot;
- replace an official published inference pipeline with an arbitrary approximation without explicitly documenting it.

When a published method requires adaptation to CPR, explain the adaptation explicitly.

When asked to provide a patch, the patch should include all repository changes required for a complete method integration, not only `run.py`.

---

## 23. Benchmark Rules

1. Use the same query and gallery manifests for every method.
2. Preserve manifest ordering exactly.
3. Score the complete gallery.
4. Do not remove the query image inside baseline code.
5. Higher score must mean better retrieval.
6. Do not train or tune on the CPR pilot.
7. Prefer official pretrained/released checkpoints.
8. Record checkpoint sources.
9. Record method and runtime configuration.
10. Keep each baseline independently reproducible.
11. Use `evaluate.py` for official metrics.
12. Use `build_tables.py` for official tables.
13. Update benchmark method ordering when adding a baseline.
14. Do not commit model weights.
15. Do not commit gallery images.
16. Do not commit raw score matrices or feature caches.
17. Commit compact evaluated results under `outputs/`.
18. Document meaningful deviations from published methods.

---

## 24. Benchmark Workflow

```text
                     NEW BASELINE
                          |
                          v
               identify official method
                          |
                          v
             inspect existing baselines
                          |
                          v
                 create method folder
                          |
             +------------+------------+
             |            |            |
             v            v            v
        config.yaml  requirements.txt  run.py
                          |
                          v
                 document checkpoint
                          |
                          v
                  validate method
                          |
                          v
                     run method
                          |
                          v
                 runs/<method>/
                 |-- scores.npy
                 `-- run.json
                          |
                          v
                    evaluate.py
                          |
                          v
                outputs/<method>/
                |-- metrics.json
                `-- run.json
                          |
                          v
                update method_order
                          |
                          v
                  build_tables.py
                          |
                          v
                       tables/
                          |
                          v
                benchmark complete
```
