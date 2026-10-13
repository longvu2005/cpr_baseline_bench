# Qwen Caption-to-Caption

Run from the repository root:

```bash
python run_baseline.py qwen_caption_to_caption
```

Standalone preparation/inference also accepts `--config`:

```bash
python methods/simple/13_qwen_caption_to_caption/download_checkpoint.py
python methods/simple/13_qwen_caption_to_caption/run.py
python evaluate.py --method qwen_caption_to_caption
python build_tables.py
```

See [the integration guide](../../../docs/additional_baselines.md) for the exact protocol, asset requirements, fairness rules and test limits. Weights are downloaded separately; there is no substitute model or fabricated score if an artifact is missing.
