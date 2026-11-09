# P2. FAFA + Partial SetMatch

Use the same pinned FAFA source, released weights and squarepad crop transform
as RCR (`224`, resize `384 × 192`). Each predicted member uses its crop and
Subject modification. Soft FDA averages the top six target-token similarities.
The single benchmark row uses Partial SetMatch with rectangular Hungarian
assignment and a fixed unmatched component score of `-1.0`.

Both this baseline and the RCR baseline use `max_txt_len=128` (including special tokens) and set the FAFA eval caption processor to `max_words=128`. The upstream default of 32 is explicitly overridden at runtime. Changing this policy changes the query-feature cache fingerprint; rerun inference rather than relying on previously saved 32-token query features. `assignment: hungarian_max_sum` and `aggregation: mean` are required.

The shared CPR adapter supplies Grounding DINO detections, identical ordered crops,
CLIP ViT-B/32 Hungarian anchors with disjoint member expansion, explicitly flagged
full-image fallbacks, and rectangular Partial SetMatch with mean aggregation.
This is a benchmark adaptation of native representation/scoring. No evaluation
identities, positive sets or annotation boxes select people.

```bash
python run_baseline.py fafa_setmatch --rcr-detection-cache /path/to/rcr/cache/detections
```

Outputs remain `runs/fafa_setmatch/scores.npy` and `run.json`.
The unchanged evaluator handles reference-image exclusion and official positives.

Read [the shared protocol and complete method table](../../../docs/baselines.md)
for exact thresholds, unmatched references, Kaggle cache inspection/reuse,
installation, validation, dependencies, calibration and optional native smoke tests.

Shared detection reads RCR's existing `cache/detections/index.pt`.
See the [benchmark detection setup](../../../docs/baselines.md#detection-and-crop-convention).
