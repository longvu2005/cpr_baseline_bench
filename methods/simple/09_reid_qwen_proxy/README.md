# S9. CLIP-ReID Subject Set + Qwen Edit Proxy

The inherited ReID branch uses the shared S5 protocol. Full-scene Qwen generation and original proxy fusion remain native. Existing S5/S8 source scores and separate validation inputs are prerequisites; stale S5 protocol/manifests are rejected. Directory references now point to `05_reid_set` and `08_qwen_edit_clip`.

The shared CPR adapter supplies Grounding DINO detections, identical ordered crops,
CLIP ViT-B/32 Hungarian anchors with disjoint member expansion, explicitly flagged
full-image fallbacks, and rectangular Partial SetMatch with mean aggregation.
This is a benchmark adaptation of native representation/scoring. No evaluation
identities, positive sets or annotation boxes select people.

```bash
python run_baseline.py reid_set_qwen_edit_proxy
```

Outputs remain `runs/reid_set_qwen_edit_proxy/scores.npy` and `run.json`.
The unchanged evaluator handles reference-image exclusion and official positives.

Read [the shared protocol and complete method table](../../../docs/baselines.md)
for exact thresholds, unmatched references, Kaggle cache inspection/reuse,
installation, validation, dependencies, calibration and optional native smoke tests.
