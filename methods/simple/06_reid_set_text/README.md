# S6. CLIP-ReID Subject Set + Global CLIP Text

The inherited ReID branch now calls the shared S5 protocol. The global CLIP text branch, fusion formula and fixed/independent-validation alpha modes are unchanged. This method has both Subject-selection text and global semantic text.

The shared CPR adapter supplies Grounding DINO detections, identical ordered crops,
CLIP ViT-B/32 Hungarian anchors with disjoint member expansion, explicitly flagged
full-image fallbacks, and rectangular Partial SetMatch with mean aggregation.
This is a benchmark adaptation of native representation/scoring. No evaluation
identities, positive sets or annotation boxes select people.

```bash
python run_baseline.py groundingdino_clipreid_set_text
```

Outputs remain `runs/groundingdino_clipreid_set_text/scores.npy` and `run.json`.
The unchanged evaluator handles reference-image exclusion and official positives.

Read [the shared protocol and complete method table](../../../docs/baselines.md)
for exact thresholds, unmatched references, Kaggle cache inspection/reuse,
installation, validation, dependencies, calibration and optional native smoke tests.
