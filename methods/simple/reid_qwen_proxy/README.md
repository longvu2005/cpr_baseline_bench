# reid_set_qwen_edit_proxy. CLIP-ReID Subject Set + Qwen Edit Proxy

The inherited ReID branch uses the shared groundingdino_clipreid_set protocol. Full-scene Qwen generation and original proxy fusion remain native. Existing groundingdino_clipreid_set/qwen_image_edit_clip source scores and separate validation inputs are prerequisites; stale groundingdino_clipreid_set protocol/manifests are rejected. Directory references now point to `reid_set` and `qwen_edit_clip`.

The shared CPR adapter supplies Grounding DINO detections, identical ordered crops,
CLIP ViT-B/32 Hungarian anchors with disjoint member expansion, explicitly flagged
full-image fallbacks, and rectangular Partial SetMatch with mean aggregation.
This is a benchmark adaptation of native representation/scoring. No evaluation
identities, positive sets or annotation boxes select people.

```bash
python run_baseline.py reid_set_qwen_edit_proxy --rcr-detection-cache /path/to/rcr/cache/detections
```

Outputs remain `runs/reid_set_qwen_edit_proxy/scores.npy` and `run.json`.
The unchanged evaluator handles reference-image exclusion and official positives.

Read [the shared protocol and complete method table](../../../docs/baselines.md)
for exact thresholds, unmatched references, Kaggle cache inspection/reuse,
installation, validation, dependencies, calibration and optional native smoke tests.

Shared detection reads RCR's existing `cache/detections/index.pt`.
See the [benchmark detection setup](../../../docs/baselines.md#detection-and-crop-convention).
