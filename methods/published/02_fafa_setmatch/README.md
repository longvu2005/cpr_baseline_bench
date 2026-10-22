# P2. FAFA + Partial SetMatch

Official multimodal FAFA encoder and released checkpoint. Each predicted Subject member uses its own crop and its Subject modification text. Native soft-FDA top-k token mean scoring and separate embedding/component-score stages are retained.

Both this baseline and the RCR baseline use `max_txt_len=128` (including special tokens) and set the FAFA eval caption processor to `max_words=128`. The upstream default of 32 is explicitly overridden at runtime. Changing this policy changes the query-feature cache fingerprint; rerun inference rather than relying on previously saved 32-token query features. `assignment: hungarian_max_sum` and `aggregation: mean` are required.

The shared CPR adapter supplies Grounding DINO detections, identical ordered crops,
CLIP ViT-B/32 Hungarian anchors with disjoint member expansion, explicitly flagged
full-image fallbacks, and rectangular Partial SetMatch with mean aggregation.
This is a benchmark adaptation of native representation/scoring. No evaluation
identities, positive sets or annotation boxes select people.

```bash
python run_baseline.py fafa_setmatch
```

Outputs remain `runs/fafa_setmatch/scores.npy` and `run.json`.
The unchanged evaluator handles reference-image exclusion and official positives.

Read [the shared protocol and complete method table](../../../docs/baselines.md)
for exact thresholds, unmatched references, Kaggle cache inspection/reuse,
installation, validation, dependencies, calibration and optional native smoke tests.
