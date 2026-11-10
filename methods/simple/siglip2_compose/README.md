# SigLIP2 Compose

Run from the repository root:

```bash
python run_baseline.py siglip2_compose
```

Standalone preparation/inference also accepts `--config`:

```bash
python methods/simple/siglip2_compose/download_checkpoint.py
python methods/simple/siglip2_compose/run.py
python evaluate.py --method siglip2_compose
python build_tables.py
```

SigLIP2 Compose selects one global image/text fusion weight from 21 values
(0.00 to 1.00 in steps of 0.05) by Full-mAP on the complete RCR validation split.
Frozen image/text features are cached once per split. The selected weight and
all trials are recorded in `runs/siglip2_compose/tuning.json` and frozen for
test evaluation (`CPR Supervision: Val only`). No test labels select the weight.

See [the integration guide](../../../docs/additional_baselines.md) for the exact protocol, asset requirements, fairness rules and test limits. Weights are downloaded separately; there is no substitute model or fabricated score if an artifact is missing.
