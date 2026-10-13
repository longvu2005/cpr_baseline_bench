# Seven additional CIR baselines

The supplied list contains seven named methods. All seven are implemented;
an eighth method has not been invented. Base repository commit:
`ca00427fbb13e3ad68cc0449c6e4202557641588`.

| Method ID | Folder | Representation | Protocol |
| --- | --- | --- | --- |
| `siglip2_compose` | `methods/simple/12_siglip2_compose` | SigLIP2 image/text fusion, fixed alpha 0.5 | Whole scene |
| `searle_xl` | `methods/published/12_searle_xl` | Official Phi on raw CLIP features, native pseudo-word encoder | Whole scene |
| `context_i2w` | `methods/published/13_context_i2w` | Official context mapper, patch tokens and custom tokenizer | Whole scene |
| `fti4cir` | `methods/published/14_fti4cir` | Official IMG2TEXT, subject/attribute pseudo-tokens | Whole scene |
| `qwen_caption_to_caption` | `methods/simple/13_qwen_caption_to_caption` | Qwen scene captions, target rewrite, MPNet cosine | Whole scene |
| `structured_caption_retrieval` | `methods/simple/14_structured_caption_retrieval` | Person captions with shared Partial SetMatch | Shared person v1 |
| `gme_qwen2_vl_2b` | `methods/simple/15_gme_qwen2_vl_2b` | Official fused image/instruction vs image-only embedding | Whole scene |

Published CIR methods retain their native scene architectures. Structured
Caption Retrieval is the explicit person adaptation.

## Apply and run

Extract the complete update ZIP **at the root of the existing checkout**.
It preserves paths and contains code/config/docs/tests, not images, weights,
caches or old outputs. Review local edits before overwriting files listed in
`UPDATE_MANIFEST.json`.

```bash
python run_baseline.py --list
python run_baseline.py siglip2_compose
python run_baseline.py searle_xl
python run_baseline.py context_i2w
python run_baseline.py fti4cir
python run_baseline.py qwen_caption_to_caption
python run_baseline.py structured_caption_retrieval
python run_baseline.py gme_qwen2_vl_2b
```

Run one selected method per notebook/runtime. `device: auto` selects CUDA when
available; `cpu` forces CPU. CPU support does not imply that full-gallery
captioning is practical on CPU. Reduce batch sizes if needed.

The runner installs dependencies, prepares actual assets, performs inference,
then uses the existing evaluator and table builder. Direct method scripts
accept `--config` and require preparation first. Every run writes
`runs/<method>/scores.npy` and `run.json`.

The by-folder archive contains one method ZIP per folder and
`shared_code.zip`. Extract the shared ZIP and all seven method ZIPs at the same
repository root. All additions require the common `cpr_baselines/` package.

## Grounding and dependency handling

Qwen uses Transformers 4.51.3, while the fixed GroundingDINO runtime uses
4.39.3. Before installing captioner dependencies, the top-level runner
verifies the original shared detection/membership cache. If missing, a
method-local preparer installs the original detector dependencies, prepares
assets and computes the cache in fresh subprocesses. Then the runner installs
Qwen's dependencies. No extra virtual environment is created.

The captioner only reads verified shared caches and cannot recompute grounding
under the newer runtime. Existing person baselines also prepare reusable
caches. The preparer handles setuptools compatibility for the legacy CLIP
dependency. `--skip-install` never installs a missing shared runtime.

Manual preparation:
```bash
python methods/simple/14_structured_caption_retrieval/prepare_person_cache.py
python run_baseline.py structured_caption_retrieval
```

Mounted caches retain `CPR_SHARED_CACHE_READ_ROOTS` and
`CPR_SHARED_CACHE_ROOT`. Canonical manifests and detector/selector assets
must remain available for provenance checks.

## Fairness rules

- Complete ordered canonical gallery/query manifests are required. Alternative
  manifest/output paths are rejected because the evaluator uses canonical paths.
- Model input views expose only image paths/IDs, public instructions and Subject
  selection/modification text. GT person IDs, target IDs/images, positive sets
  and case labels are excluded from model input.
- All models are frozen. No benchmark training, evaluation-label tuning, retrieval
  feedback to Qwen or query-specific fusion selection is used. SigLIP2 alpha is
  fixed at 0.5.
- Scores cover the full gallery and retain a finite reference-image entry. Only
  the original evaluator excludes the reference. ID/Full mAP, Recall and
  multi-positive formulas are unchanged.
- Structured captions reuse actual cached boxes, Subject memberships, one-to-many
  expansion, full-image fallback slots and crop geometry. Membership never
  comes from GT identity counts.
- Shared matching uses `min(K,N)` real Hungarian pairs and
  `(sum(matched_cosines)+(K-min(K,N))*(-1))/K`. Missing components receive the
  analytic cosine lower-bound penalty -1; extra gallery people are allowed.
- New scene runs record the same image-byte digest as person runs. Tables reject
  different recorded image bytes and different actual person boxes/memberships.
  Existing scene results without an image digest are not retroactively verified.
- Metadata records prompts, pixel/token budgets, revisions, SHA-256 values,
  runtime/package versions, row ordering and shared-person provenance.
  Backbone sizes and compute costs still differ.

## Captioning and caches

Both caption methods use pinned Qwen2.5-VL-3B-Instruct and
all-mpnet-base-v2, with greedy decoding, one request at a time, a fixed pixel
budget and one format-repair retry. No GT is supplied. Reaching the output
token budget raises an error rather than accepting truncated text.

Caption-to-caption describes every gallery image, reuses the reference
caption, rewrites it using the public instruction and embeds both sides in
the same text space with official Sentence Transformers pooling/normalization.

Structured captions have four string fields: `appearance`, `clothing`,
`action`, `objects`. Each predicted member gets one target record. Visible
physical traits are preserved unless explicitly changed. This description
is not a face/ReID identity encoder. Crops have limited scene/relational
information; interpret ID and Full scores with that limitation.

`cache/additional/` holds integrity-checked features and individual caption
responses. Completed requests are reused after an interruption. Keys include
assets, prompt, actual crop pixels and generation/runtime settings. Failed
requests are not cached as successful captions. `captions.json` exposes
descriptions and component offsets.

## Official sources and limitations

| Asset | Official source | Detail |
| --- | --- | --- |
| SigLIP2 | [Google](https://huggingface.co/google/siglip2-so400m-patch14-384) | Fixed 384-pixel SO400M; 64 text tokens |
| SEARLE-XL | [Authors](https://github.com/miccunifi/SEARLE) | Official 768/3072/768 Phi; verified checksum in config |
| Context-I2W | [Authors](https://github.com/Pter61/context-i2w) | Custom vocabulary; both CLIP/mapper state dicts load strictly |
| FTI4CIR | [Code](https://github.com/iLearn-Lab/SIGIR24-FTI4CIR), [weights](https://huggingface.co/iLearn-Lab/SIGIR24-FTI4CIR) | Serialized IMG2TEXT; official HF checksum in config |
| Qwen | [Qwen](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) | Local generation; no paid API |
| MPNet | [Sentence Transformers](https://huggingface.co/sentence-transformers/all-mpnet-base-v2) | Official pooling; 384 text tokens |
| GME | [Alibaba](https://huggingface.co/Alibaba-NLP/gme-Qwen2-VL-2B-Instruct) | Native fused forward/final-token pooling; 256–1280 image tokens |

All HF revisions and source commits are frozen in config. SEARLE/FTI4CIR
import pinned OpenAI CLIP code directly, avoiding a legacy wheel build.
Compatible setuptools is still needed for its `pkg_resources` import and
is declared in their requirements. Author code/licenses are downloaded separately.

Context-I2W's author Drive release has no published checksum. Preparation
records the actual SHA-256 and verifies it on later runs, but cannot prove
that an initially supplied manual file is the author release. Set
`assets.checkpoint.sha256` if independently verified. Blocked/quota-limited
downloads exit with status 42 without fallback models or substitute scores.
For manual placement, use the official Drive link in the author README and
the configured checkpoint path, then rerun preparation.

Published CLIP methods keep their 77-token limit and deterministic truncation
with pseudo-token prefixes preserved. Long instructions can lose trailing
content. SEARLE/FTI4CIR use author TargetPad 1.25; Context-I2W uses its author
CLIP transform. All three published adapters default to fp32.

## Validation

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q tests
python -m compileall -q cpr_baselines cpr_person methods scripts tests run_baseline.py build_tables.py
python run_baseline.py --list
```

After preparing real assets, run a selected method's native forward smoke:
```bash
python scripts/smoke_additional_runtime.py --method siglip2_compose
python scripts/smoke_additional_runtime.py --method searle_xl
python scripts/smoke_additional_runtime.py --method context_i2w
python scripts/smoke_additional_runtime.py --method fti4cir
python scripts/smoke_additional_runtime.py --method gme_qwen2_vl_2b
```

Optional native tests check author modules/released mappers with tiny random
CLIP backbones, without computing benchmark metrics. Completed validation
and its limits are recorded in `docs/additional_validation.md`.

