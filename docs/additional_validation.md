# Validation evidence and limits

## Current revision (2026-11-09)

Baseline base `5f7a9d395f5c98aa16ac8e3f07fe57cdc5f8dfe9`, unmodified RCR
`2fe92766d5158f8dd87297d3bc4cb6280b8ed63b`. Python 3.13.18, NumPy 2.5, SciPy
1.18; no torch/transformers and no package installs were available here.

| Check | Result | Scope |
| --- | --- | --- |
| CPU suite, no RCR checkout | 310 passed, 28 skipped | Skips need torch or opt-in pretrained assets; real `runs/` artifacts are left untouched |
| CPU suite with `CPR_TEST_RCR_REPO` | 318 passed, 20 skipped | Adds torch-free parity in `test_rcr_protocol_parity.py` |
| Final import vs RCR loader | Exact | All val/test queries and gallery rows, positives, required IDs, selection text |
| Subject text routing vs RCR FAFA | Exact on 4,106 queries | `modify_text` / `subject_changes` |
| Evaluator vs RCR | Max difference 3.5e-18 | Complete val/test with tied integer scores |
| Subject selection vs RCR FAFA | Exact; mutation detected | GROUP saturating cap 5, DUAL, empty and short references, seeded stand-in features |
| Partial SetMatch vs RCR | Bit-identical | K=1..10, N=0..8, ties, cosine and native penalties |
| Native algorithm signatures | 118/118 unchanged | Python 3.12 and 3.13 |
| Original RCR index binding | Passed contracts | Indexes without digests accepted; digests verified when present; spot-check rejects a different image/preprocessing |

The skipped torch tests (HF letterbox/text parity with real weights, packed
detector projection, Instruct-ReID glue, S9 proxy cache, cache audit) were not run
here; run them on Kaggle as described in [kaggle.md](kaggle.md#direct-parity-tests).
Stand-in selector features test the selection rule, not CLIP accuracy.

## Previous revision (from `65a7c37`)

The table below was recorded in a different environment (Python 3.12, Torch 2.6
CPU) for the previous revision and the RCR companion patch, which this revision
no longer needs. It was not rerun here.

| Check | Result | Evidence scope |
| --- | --- | --- |
| Baseline complete opt-in suite | 323 passed, 3 skipped | Unit/integration contracts, all method routes, actual CLIP components, full-split evaluator parity; see logs delivered with patch |
| RCR suite with companion patch | 431 passed | Existing RCR coverage plus image-provenance cache regressions |
| Native algorithm signatures | 118 protected nodes in 27 files | 25 method entrypoints plus native backend/published providers retain selected original encoder/tokenizer/composition/scoring bodies; S1–S4 intentionally use the new shared HF provider |
| Actual frozen HF image forward | Passed | Odd aspect ratios; exact RCR letterbox tensors; raw pretrained globals match RCR ImageEncoder; proven RCR cache/global reuse and warm cache |
| Owned CLIP fixture pipeline | Passed for S1–S4 | Actual pinned HF model inference, official evaluation and table subprocesses on isolated fixture val/test; environment preparation substituted; no final-data benchmark claim |
| All 29 metadata writers | Passed | Every direct/shared result writer records actual runtime identity; missing/malformed runtime is rejected before completed execution receipts |
| Actual frozen HF text forward | Passed | CLIPTokenizerFast/native EOS/right padding/77-token truncation; RCR native TextEncoder globals; altering a learned context projection does not affect globals; actual subprocess facade |
| Actual OpenAI B/32 selector | Passed | Same official weight SHA, native preprocessing/tokenization/crop features as RCR; actual isolated selector worker and warm cache with fixture geometry |
| Full val/test evaluator parity | Passed within 1e-12 | Synthetic full rankings for all 1,799 val and 2,307 test queries; stable ties, reference exclusion, eight overall and sixteen case metrics, Macro-Full-mAP |
| Packed detector adapter parity | Passed | Actual RCR cache builder/reader/projector and FAFA candidate adapter with fixture detector output; exact pixel order/out-of-frame boxes/empty fallback; not a pretrained DINO forward |
| Instruct validation selection | Passed contracts | Native distance math/scale, exact affine reaggregation, frozen val selection/cache and test-label perturbation isolation; heavy native encoder substituted only in glue test |
| Word4Per reproduction admission | Passed contracts | Weight/config/source/Stage-1 receipt binding, external image inventory, byte-identical renamed overlap rejection |
| WISER provenance | Passed contracts | Native prompts/request options/output handling preserved in wrapper tests, cached response integrity, resolved model drift rejection, actual module state hash sensitivity; no paid API call |
| Dependency/cache safety | Passed | S6 score keys bind gallery source; S9 validates generator marker, edited image bytes and source scene index even on warm score cache; Source config/checkpoint/code drift rejection, readonly weight mount with writable metadata, stale result/manifest/image checks, S5/S8→S9 and S6→S11 guards |
| Method discovery/reporting/lint | Passed | 29 IDs unchanged; `python -S build_tables.py --check`; compilation, pyflakes, `bash -n`, `git diff --check` |
| Patch application | Passed | Both patches checked/applied to fresh clean worktrees at their exact base commits; modified/new file bytes compared with delivered source |

Runtime used for the actual CPU component checks: Python 3.12.14,
Torch 2.6.0+cpu, torchvision 0.21.0+cpu, Transformers 4.56.2,
NumPy 1.26.4, SciPy 1.17.1, Pillow 12.3.0. Native method environments
remain separately specified; the shared HF/selector runtime does not replace them.

The three skips are the optional SEARLE/FTI4CIR upstream-source/mapper checks
(two checks) and the Context-I2W check requiring absent `einops_exts`/author
assets. They are not counted as passed. These existing optional tests use tiny
random backbones and would still not constitute full pretrained benchmark inference.

The real HF snapshot is pinned to
`3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268`; official OpenAI selector weight SHA256 is
`40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af`.
No pretrained assets are included in the patch. Fixture images and fixture
geometry are explicitly distinct from the actual final RGB gallery and native
Grounding DINO outputs. Synthetic rankings establish evaluator agreement only.

S1–S4 additionally completed isolated fixture pipeline checks with actual HF weights.
Their diagnostic metrics are not benchmark accuracy; they are excluded from the
reported final-data verification count.

No complete final-data native inference was run for any of the 29 methods:
there is no CUDA device, no actual final RGB image collection/detection cache,
no complete native checkpoint set and no WISER API credential in this environment.
Consequently final accuracy, native FAFA/ReID/pseudo-token/editor/VLM output
conformance, measured GPU memory/runtime and full-run reproducibility remain
unverified. Preflight reports prerequisites, never native inference success.
The publication checker validates recorded artifacts and intentionally fails
when completed native runs/evidence are absent.

## Reproduce the checks

Install the appropriate test/shared dependencies and prepare the two local CLIP
assets and isolated selector tokenizer. Then run from the baseline checkout:

```bash
CPR_TEST_RCR_REPO=/absolute/rcr \
CPR_TEST_PRETRAINED_CLIP=1 CPR_TEST_FULL_EVALUATOR=1 \
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 USE_TF=0 USE_FLAX=0 \
python -m pytest -q -rs
python -m pyflakes cpr_baselines cpr_person methods scripts tests run_baseline.py evaluate.py build_tables.py
python -S build_tables.py --check
bash -n scripts/run_all_kaggle.sh
```

From the RCR checkout, with its test dependencies installed:

```bash
PYTHONPATH=src OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 USE_TF=0 USE_FLAX=0 python -m pytest -q
```

Without the opt-in environment variables, actual cross-repo/pretrained checks
skip. Never cite that smaller offline suite as having run actual CLIP forwards.
For execution of the benchmark itself, see [Kaggle](kaggle.md) and the complete
[29-method audit](audit_29.md).
