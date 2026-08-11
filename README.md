# CPR Baseline Benchmark

This repository provides a common evaluation protocol for CPR baseline methods.

Each baseline may use its own implementation and dependencies, but all methods must use the same data, output format, and evaluator.

## Repository Structure

    data/
    checkpoints/
    methods/
      simple/
      published/
    runs/
    outputs/
    tables/
    evaluate.py
    build_tables.py

## Adding a New Baseline

Create a method folder under:

    methods/simple/<method_name>/

or:

    methods/published/<method_name>/

Recommended structure:

    methods/simple/<method_name>/
    |-- config.yaml
    |-- run.py
    |-- requirements.txt
    `-- README.md

Each baseline is independent and may use its own environment and dependencies.

## Config

Keep method and runtime settings in config.yaml.

Example:

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

Runtime parameters are used for memory and speed.
They should not be tuned using retrieval performance.

## Common Data

All methods must use:

    data/gallery/
    data/gallery.jsonl
    data/queries.jsonl

Do not reorder the query or gallery manifests.

The query image remains inside the global gallery.
Baseline code must not remove it.
The common evaluator handles query-image exclusion.

## Checkpoints

Do not commit model weights to GitHub.

Store checkpoints under:

    checkpoints/

Document every checkpoint in:

    checkpoints/README.md

Include the official source and expected local path.

## Required Baseline Output

Every baseline must create:

    runs/<method_name>/
    |-- scores.npy
    `-- run.json

scores.npy must have shape:

    [num_queries, num_gallery]

scores[q, g] is the retrieval score of gallery image g for query q.

Higher score must mean a better match.

Rows must follow:

    data/queries.jsonl

Columns must follow:

    data/gallery.jsonl

## Evaluation

Do not compute final benchmark metrics separately inside each baseline.

Run:

    python evaluate.py --method <method_name>

This creates:

    outputs/<method_name>/
    |-- metrics.json
    `-- run.json

## Build Benchmark Tables

After evaluating one or more methods, run:

    python build_tables.py

Tables are generated under:

    tables/

## Benchmark Rules

1. Use the same gallery and query manifests.
2. Score the complete gallery.
3. Do not reorder queries or gallery images.
4. Do not remove the query image inside baseline code.
5. Higher score must mean better retrieval.
6. Use official pretrained checkpoints whenever possible.
7. Record checkpoint sources and runtime configuration.
8. Use evaluate.py for final metrics.
9. Use build_tables.py for benchmark tables.
10. Do not commit checkpoints, raw score matrices, or gallery images.

## Workflow

    run baseline
        |
        v
    runs/<method>/scores.npy
        |
        v
    evaluate.py
        |
        v
    outputs/<method>/metrics.json
        |
        v
    build_tables.py
        |
        v
    tables/

