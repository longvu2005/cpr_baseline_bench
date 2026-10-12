# P9. AdaFocal + Partial SetMatch

Official AdaFocal full-reference-image context and predicted anchor-box masks are retained for every predicted member. Gallery crops use the native target branch; pair scores remain maximum cosine over its target query tokens. Source/checkpoint compatibility handling is unchanged.

The shared CPR adapter supplies Grounding DINO detections, identical ordered crops,
CLIP ViT-B/32 Hungarian anchors with disjoint member expansion, explicitly flagged
full-image fallbacks, and rectangular Partial SetMatch with mean aggregation.
This is a benchmark adaptation of native representation/scoring. No evaluation
identities, positive sets or annotation boxes select people.

```bash
python run_baseline.py adafocal_setmatch
```

Outputs remain `runs/adafocal_setmatch/scores.npy` and `run.json`.
The unchanged evaluator handles reference-image exclusion and official positives.

Read [the shared protocol and complete method table](../../../docs/baselines.md)
for exact thresholds, unmatched references, Kaggle cache inspection/reuse,
installation, validation, dependencies, calibration and optional native smoke tests.
