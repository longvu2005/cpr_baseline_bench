# SEARLE-XL

Run from the repository root:

```bash
python run_baseline.py searle_xl
```

Standalone preparation/inference also accepts `--config`:

```bash
python methods/published/12_searle_xl/download_checkpoint.py
python methods/published/12_searle_xl/run.py
python evaluate.py --method searle_xl
python build_tables.py
```

See [the integration guide](../../../docs/additional_baselines.md) for the exact protocol, asset requirements, fairness rules and test limits. Weights are downloaded separately; there is no substitute model or fabricated score if an artifact is missing.
