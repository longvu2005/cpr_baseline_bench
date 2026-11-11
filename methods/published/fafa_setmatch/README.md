# fafa_setmatch. FAFA + Partial SetMatch

Use the same pinned FAFA source, released weights and squarepad crop transform
as RCR (`224`, resize `384 × 192`). Each predicted member uses its crop and
the **complete per-Subject condition from RCR Proposed**, constructed by the
ported `subject_conditions()` function in `cpr_person/fafa_text.py`.
Soft FDA averages the top six target-token similarities.
The single benchmark row uses Partial SetMatch with rectangular Hungarian
assignment and a fixed unmatched component score of `-1.0`.

Both this baseline and the RCR baseline use `max_txt_len=128` (including special tokens) and set the FAFA eval caption processor to `max_words=128`. The upstream default of 32 is explicitly overridden at runtime. Changing this policy changes the query-feature cache fingerprint; rerun inference rather than relying on previously saved 32-token query features. `assignment: hungarian_max_sum` and `aggregation: mean` are required.

Text routing is taken from RCR Proposed `src/rcr/proposed/cache/native.py`,
source commit `da764bc3c158dc5286e4b65bfa4c3bbd8cf22296`:
`final_change` is kept intact and each `Subject N` reference is rewritten as
`the reference person` (for the current Subject) or
`the other referenced person (Subject N)`. Only adjacent plural auxiliaries
(`are`, `were`, `have`) are singularized. **Do not** use the imported
`subjects[].modify_text`: it splits independent clauses for other baselines.
`subjects[].select_text` remains exclusively for reference CLIP localization.
This change affects only `fafa_setmatch`, not other benchmark methods.

The shared CPR adapter supplies Grounding DINO detections, identical ordered crops,
CLIP ViT-B/32 Hungarian anchors with disjoint member expansion, explicitly flagged
full-image fallbacks, and rectangular Partial SetMatch with mean aggregation.
This is a benchmark adaptation of native representation/scoring. No evaluation
identities, positive sets or annotation boxes select people.

```bash
python run_baseline.py fafa_setmatch --rcr-detection-cache /path/to/rcr/cache/detections
```

Changing FAFA text routing invalidates query features, component scores and
final rankings. Rerun validation/test inference and evaluation; never relabel
old `scores.npy` artifacts. The baseline still selects reference people by CLIP
and uses Partial SetMatch, whereas Proposed encodes all reference crops and uses
learned grounding/matching. Thus native feature parity for identical crops/text
is expected, but **identical end-to-end rankings are not**.

Outputs remain `runs/fafa_setmatch/scores.npy` and `run.json`.
The unchanged evaluator handles reference-image exclusion and official positives.

Read [the shared protocol and complete method table](../../../docs/baselines.md)
for exact thresholds, unmatched references, Kaggle cache inspection/reuse,
installation, validation, dependencies, calibration and optional native smoke tests.

Shared detection reads RCR's existing `cache/detections/index.pt`.
See the [benchmark detection setup](../../../docs/baselines.md#detection-and-crop-convention).
