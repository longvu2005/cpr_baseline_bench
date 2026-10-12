# S5. CLIP-ReID + CLIP Subject Localization + Partial SetMatch

Official MSMT17 CLIP-ReID ViT-B/16, native 256x128 preprocessing, pre-BN concatenated feature and normalized cosine scoring are retained. The primary ID now uses text to select Subject members; text does not enter ReID scoring. `--text-free-diagnostic` skips the selector and writes a separately labeled all-reference-person diagnostic.

The shared CPR adapter supplies Grounding DINO detections, identical ordered crops,
CLIP ViT-B/32 Hungarian anchors with disjoint member expansion, explicitly flagged
full-image fallbacks, and rectangular Partial SetMatch with mean aggregation.
This is a benchmark adaptation of native representation/scoring. No evaluation
identities, positive sets or annotation boxes select people.

```bash
python run_baseline.py groundingdino_clipreid_set
```

Outputs remain `runs/groundingdino_clipreid_set/scores.npy` and `run.json`.
The unchanged evaluator handles reference-image exclusion and official positives.

Read [the shared protocol and complete method table](../../../docs/baselines.md)
for exact thresholds, unmatched references, Kaggle cache inspection/reuse,
installation, validation, dependencies, calibration and optional native smoke tests.
