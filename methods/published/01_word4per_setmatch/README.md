# P1. Word4Per + Partial SetMatch

Official pinned `old_project` implementation and independently reproduced Stage-2 checkpoint. Pseudo-word mapping, native preprocessing, prompt and cosine pairwise scoring are unchanged. Stage-2 remains external to CPR evaluation; see `checkpoints/word4per/README.word4per.md`.

The shared CPR adapter supplies Grounding DINO detections, identical ordered crops,
CLIP ViT-B/32 Hungarian anchors with disjoint member expansion, explicitly flagged
full-image fallbacks, and rectangular Partial SetMatch with mean aggregation.
This is a benchmark adaptation of native representation/scoring. No evaluation
identities, positive sets or annotation boxes select people.

```bash
python run_baseline.py word4per_setmatch
```

Outputs remain `runs/word4per_setmatch/scores.npy` and `run.json`.
The unchanged evaluator handles reference-image exclusion and official positives.

Read [the shared protocol and complete method table](../../../docs/baselines.md)
for exact thresholds, unmatched references, Kaggle cache inspection/reuse,
installation, validation, dependencies, calibration and optional native smoke tests.
