# Context-I2W

Run from the repository root:

```bash
python run_baseline.py context_i2w
```

Standalone preparation/inference also accepts `--config`:

```bash
python methods/published/13_context_i2w/download_checkpoint.py
python methods/published/13_context_i2w/run.py
python evaluate.py --method context_i2w
python build_tables.py
```

See [the integration guide](../../../docs/additional_baselines.md) for the exact protocol, asset requirements, fairness rules and test limits. Weights are downloaded separately; there is no substitute model or fabricated score if an artifact is missing.
