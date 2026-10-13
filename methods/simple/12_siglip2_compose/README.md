# SigLIP2 Compose

Run from the repository root:

```bash
python run_baseline.py siglip2_compose
```

Standalone preparation/inference also accepts `--config`:

```bash
python methods/simple/12_siglip2_compose/download_checkpoint.py
python methods/simple/12_siglip2_compose/run.py
python evaluate.py --method siglip2_compose
python build_tables.py
```

See [the integration guide](../../../docs/additional_baselines.md) for the exact protocol, asset requirements, fairness rules and test limits. Weights are downloaded separately; there is no substitute model or fabricated score if an artifact is missing.
