# S10. Per-Person CLIP Compose + Partial SetMatch

Composition remains normalized `alpha * crop_image + (1-alpha) * modification_text` using native CLIP ViT-L/14. Subject localization now uses the common ViT-B/32 selector, and all group members receive their Subject modification. Fixed/independent-validation alpha modes remain.

The shared CPR adapter supplies Grounding DINO detections, identical ordered crops,
CLIP ViT-B/32 Hungarian anchors with disjoint member expansion, explicitly flagged
full-image fallbacks, and rectangular Partial SetMatch with mean aggregation.
This is a benchmark adaptation of native representation/scoring. No evaluation
identities, positive sets or annotation boxes select people.

```bash
python run_baseline.py per_person_clip_compose_setmatch
```

Outputs remain `runs/per_person_clip_compose_setmatch/scores.npy` and `run.json`.
The unchanged evaluator handles reference-image exclusion and official positives.

Read [the shared protocol and complete method table](../../../docs/baselines.md)
for exact thresholds, unmatched references, Kaggle cache inspection/reuse,
installation, validation, dependencies, calibration and optional native smoke tests.
