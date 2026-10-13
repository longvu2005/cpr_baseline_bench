# GME-Qwen2-VL-2B

Run from the repository root:

```bash
python run_baseline.py gme_qwen2_vl_2b
```

Standalone preparation/inference also accepts `--config`:

```bash
python methods/simple/15_gme_qwen2_vl_2b/download_checkpoint.py
python methods/simple/15_gme_qwen2_vl_2b/run.py
python evaluate.py --method gme_qwen2_vl_2b
python build_tables.py
```

See [the integration guide](../../../docs/additional_baselines.md) for the exact protocol, asset requirements, fairness rules and test limits. Weights are downloaded separately; there is no substitute model or fabricated score if an artifact is missing.
