# Structured Caption Retrieval + Partial SetMatch

Run from the repository root:

```bash
python run_baseline.py structured_caption_retrieval
```

Standalone preparation/inference also accepts `--config`:

```bash
python methods/simple/14_structured_caption_retrieval/download_checkpoint.py
python methods/simple/14_structured_caption_retrieval/run.py
python evaluate.py --method structured_caption_retrieval
python build_tables.py
```

See [the integration guide](../../../docs/additional_baselines.md) for the exact protocol, asset requirements, fairness rules and test limits. Weights are downloaded separately; there is no substitute model or fabricated score if an artifact is missing.
