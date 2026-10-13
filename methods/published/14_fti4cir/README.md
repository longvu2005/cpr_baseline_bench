# FTI4CIR

Run from the repository root:

```bash
python run_baseline.py fti4cir
```

Standalone preparation/inference also accepts `--config`:

```bash
python methods/published/14_fti4cir/download_checkpoint.py
python methods/published/14_fti4cir/run.py
python evaluate.py --method fti4cir
python build_tables.py
```

See [the integration guide](../../../docs/additional_baselines.md) for the exact protocol, asset requirements, fairness rules and test limits. Weights are downloaded separately; there is no substitute model or fabricated score if an artifact is missing.
