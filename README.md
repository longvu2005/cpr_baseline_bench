
# Adding a New Baseline

Each baseline is an independent mini-project.

Baseline implementations do not need to share code, environments, or dependencies.
All methods only need to use the common benchmark data and export results in the required format.

## 1. Create a Method Folder

Choose one group:

    methods/simple/
    methods/published/

Example:

    methods/simple/new_method/

or:

    methods/published/new_method/

A method folder may contain:

    methods/simple/new_method/
    |-- run.py
    |-- requirements.txt
    |-- README.md
    `-- other_files/

Each baseline may use its own implementation.

## 2. Common Benchmark Data

All methods use:

    data/gallery/
    data/gallery.jsonl
    data/queries.jsonl

Do not create separate benchmark data for each method.

The query image remains inside the global gallery.
Do not remove it inside baseline code.
The common evaluator will remove the query image during evaluation.

## 3. Checkpoints

Do not commit model weights to GitHub.

All checkpoints must be stored under:

    checkpoints/

For every new checkpoint, add an entry to:

    checkpoints/README.md

Example:

    ## Model Name

    Source: <official source>

    Expected path: checkpoints/model_name/checkpoint.pt

## 4. Run a Baseline

Each method may use its own Python environment and dependencies.

Example:

    pip install -r methods/simple/new_method/requirements.txt

Run:

    python methods/simple/new_method/run.py

Published methods may use their official repositories and original dependencies.

## 5. Required Output

Every baseline must create:

    runs/<method_name>/
    |-- scores.npy
    `-- run.json

### scores.npy

The score matrix must have shape:

    [num_queries, num_gallery]

Definition:

    scores[q, g]

is the retrieval score assigned by query q to gallery image g.

Higher score must mean a better match.

The row order must exactly follow:

    data/queries.jsonl

The column order must exactly follow:

    data/gallery.jsonl

Do not reorder either manifest.

## 6. run.json

Example:

    {
      "method": "method_name",
      "display_name": "Method Display Name",
      "group": "Simple / Obvious Baselines",
      "cpr_supervision": "No",
      "checkpoint": "checkpoints/model_name/checkpoint.pt",
      "num_queries": "<num_queries>",
      "num_gallery": "<num_gallery>",
      "scores": "runs/method_name/scores.npy",
      "higher_is_better": true
    }

Allowed groups:

    Simple / Obvious Baselines
    Published / SOTA Baselines
    Proposed

CPR supervision:

    No
    Val only
    Train

## 7. Benchmark Rules

1. Use the same gallery.jsonl.
2. Use the same queries.jsonl.
3. Do not create a different gallery for each method.
4. Do not remove the query image inside baseline code.
5. Do not compute final benchmark metrics independently.
6. Export scores for the complete gallery.
7. Higher score must mean better retrieval.
8. Use official pretrained checkpoints whenever possible.
9. Record checkpoint sources in checkpoints/README.md.
10. Keep method-specific code inside its own method folder.

Baseline implementation is independent.
Evaluation protocol is shared.

## 8. Naming Convention

Simple baselines:

    methods/simple/clip_image/
    methods/simple/clip_text/
    methods/simple/new_method/

Published baselines:

    methods/published/word4per_setmatch/
    methods/published/fafa_setmatch/
    methods/published/new_method/

Output folders:

    runs/clip_image/
    runs/fafa_setmatch/
    runs/new_method/

## 9. Validate Baseline Output

Run:

    python - <<'PY'
    import json
    import numpy as np

    def count_jsonl(path):
        with open(path, "r", encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())

    num_queries = count_jsonl("data/queries.jsonl")
    num_gallery = count_jsonl("data/gallery.jsonl")

    scores = np.load(
        "runs/<method_name>/scores.npy",
        mmap_mode="r",
    )

    print("shape:", scores.shape)
    print("expected:", (num_queries, num_gallery))
    print("dtype:", scores.dtype)
    print("finite:", np.isfinite(scores).all())

    assert scores.shape == (num_queries, num_gallery)
    PY

Only valid score matrices should be passed to the common evaluator.

# cpr_baseline_bench
