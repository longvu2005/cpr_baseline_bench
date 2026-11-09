# Validation scope

Audit date: 2026-11-09. Baseline starting commit:
`783ce68`. Runtime: Python 3.12.14, NumPy 1.26.4, SciPy 1.17.1,
PyTorch 2.15.0 CPU. Method-specific GPU environments were not installed.

| Check | Result | What it establishes |
| --- | --- | --- |
| CPU suite | 265 passed, 4 skipped | Protocol, matching, shared cache, public-input boundary, evaluation, reporting and failure handling |
| Native architecture signatures | Passed in the suite | Protected native encoder/composition/pair-scoring functions are unchanged |
| Compilation and pyflakes | Passed | Repository Python compiles; no reported unused imports/undefined names |
| Reporting preflight | `python -S build_tables.py --check` passed | Table imports do not require model packages |
| Method discovery | 29 methods | All configured method entry points and aliases remain discoverable |
| Complete RCR import | Passed | Val 1,799/5,684 and test 2,307/7,868 query/gallery counts, canonical order and split-specific positives |
| RCR evaluator agreement | Maximum difference < 1e-12 | Eight overall and sixteen per-case metrics match for synthetic full rankings of every current test query |
| Patch delivery | Checked against starting commit | Changes apply to the reviewed revision; no images, checkpoints, caches or experiment outputs included |

The four skipped checks need absent upstream OpenAI CLIP source/checkpoints
(two checks), `einops_exts` (one) and Transformers detector packages (one).
No skipped check is represented as passed.

The full current final export was read from RCR commit
`2fe92766d5158f8dd87297d3bc4cb6280b8ed63b`.
The independent evaluator comparison used deterministic synthetic ranks with
ties and self-exclusion. It saved no model predictions or benchmark accuracy.

No complete pretrained GPU benchmark was run: actual gallery images, the
RCR detection index and real method checkpoints were unavailable here. Model
accuracy, checkpoint availability, native-output conformance under the user's
installed runtimes, GPU memory use, timing and external API behavior remain
unverified. Optional native smoke scripts are provided for that environment.

For publication requirements, see [paper protocol](paper_protocol.md).
