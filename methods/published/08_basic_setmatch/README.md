# P8. BASIC + Partial SetMatch

Official BASIC CLIP, external corpora, LAION mean, projection, query expansion and polynomial scoring are retained. The existing official ranking-parity check still executes. The unmatched score is an analytic lower bound derived only from the official external resources.

The shared CPR adapter supplies Grounding DINO detections, identical ordered crops,
CLIP ViT-B/32 Hungarian anchors with disjoint member expansion, explicitly flagged
full-image fallbacks, and rectangular Partial SetMatch with mean aggregation.
This is a benchmark adaptation of native representation/scoring. No evaluation
identities, positive sets or annotation boxes select people.

```bash
python run_baseline.py basic_setmatch
```

Outputs remain `runs/basic_setmatch/scores.npy` and `run.json`.
The unchanged evaluator handles reference-image exclusion and official positives.

Read [the shared protocol and complete method table](../../../docs/baselines.md)
for exact thresholds, unmatched references, Kaggle cache inspection/reuse,
installation, validation, dependencies, calibration and optional native smoke tests.
