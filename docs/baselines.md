# CPR baseline protocol

The seven new CIR adapters and their native/shared-person choices are
documented in [additional baselines](additional_baselines.md).

The person adapter lives in `cpr_person/`. The official model, checkpoint,
composition function and pairwise score remain in each method directory.
This is a CPR benchmark adaptation, not a claim that the published method used
these detection, localization or image aggregation procedures.

## Coverage

| ID / directory | Representation or scoring retained | Shared detection | Shared localization | Partial SetMatch |
| --- | --- | --- | --- | --- |
| S1 `clip_image` | Full-image CLIP | No | No | No |
| S2 `clip_text` | Text to full-image CLIP | No | No | No |
| S3 `clip_early_fusion` | Global early fusion | No | No | No |
| S4 `clip_late_fusion` | Global late fusion | No | No | No |
| S5 `groundingdino_clipreid_set` | Official MSMT17 CLIP-ReID, cosine | Yes | Yes | Yes |
| S6 `groundingdino_clipreid_set_text` | Adapted S5 + global CLIP text, original fusion | Inherited S5 | Inherited S5 | ReID branch |
| S7 `qwen25vl_rewrite_clip` | Text rewriting + full-image CLIP | No | No | No |
| S8 `qwen_image_edit_clip` | Generated full-scene proxy + CLIP | No | No | No |
| S9 `reid_set_qwen_edit_proxy` | Adapted S5 + original S8 proxy/fusion | Inherited S5 | Inherited S5 | ReID branch |
| S10 `per_person_clip_compose_setmatch` | Native ViT-L/14 image/text composition, cosine | Yes | Yes, B/32 | Yes |
| S11 `retrieve_qwen25vl_rerank` | Full-scene VLM reranking | S6 first stage | S6 first stage | First stage only |
| P1 `word4per_setmatch` | Official Word4Per Stage-2 pseudo-word composition, cosine | Yes | Yes | Yes |
| P2 `fafa_setmatch` | Official multimodal FAFA, native soft-FDA top-k token mean | Yes | Yes | Yes |
| P3 `instruct_reid_setmatch` | Official instruction-aware fusion, negative squared Euclidean | Yes | Yes | Yes |
| P4 `pic2word` | Native full-image composition | No | No | No |
| P5 `lincir` | Native full-image composition | No | No | No |
| P6 `magiclens_large` | Native full-image encoder | No | No | No |
| P7 `bge_vl_mllm_s1` | Native scene/image retrieval | No | No | No |
| P8 `basic_setmatch` | Official BASIC scorer and parity check | Yes | Yes | Yes |
| P9 `adafocal_setmatch` | Full-reference context + predicted anchor; native max-token similarity | Yes | Yes | Yes |
| P10 `imagine_seek` | Native full-scene generation/retrieval | No | No | No |
| P11 `wiser` | Native scene retrieval | No | No | No |

S11 continues to pass full scenes to its verifier. Its only code change is a
guard against old/incompatible S6 first-stage scores. Scene encoders and their
inference paths are otherwise unchanged.

## Detection and crop convention

`cpr_person/protocol.yaml` is the authoritative shared configuration. Method
YAMLs include matching snapshots for visibility; execution reads the shared
configuration. Change that one file, then rerun every affected baseline.

- Grounding DINO Swin-T OGC, official release `groundingdino_swint_ogc.pth`.
- Source commit `856dde20aee659246248e20734ef9ba5214f5e44`.
- Prompt `person`, box threshold `0.35`, text threshold `0.25`.
- Deterministic NMS at IoU `0.50`, minimum raw clipped box width/height `4px`.
- Maximum `10` persons per scene, ordered by confidence; coordinate order and
  original index break confidence ties.
- Load stored image pixels as RGB, without EXIF transpose, consistently with
  the previous FAFA detector's image loading convention.
- Clip boxes to the image, perform NMS, then store integer crop boundaries:
  floor left/top, ceil right/bottom. Every native crop therefore contains the
  same pixels. Model-specific resizing and normalization remain native.
- No detection produces one **flagged full-image fallback candidate**.

Forward grouping is fixed at batch size `2` with padding ratio `1.3`. The
upstream pure-PyTorch deformable-attention implementation is used consistently,
including on CUDA. This intentionally fixes the attention backend, rather than
allowing each method/environment to choose a different compiled extension.
Text threshold is retained in provenance; upstream uses it for phrase decoding,
while box acceptance uses box threshold.

Detector and selector run in isolated processes. Native FAFA/AdaFocal/Instruct-ReID
Hugging Face caches do not redirect Grounding DINO runtime assets. The selector
uses a private pure-Python `ftfy==5.8` / `wcwidth==0.2.13` target prepared once,
so Word4Per keeps its native `ftfy>=6.1` requirement and Instruct-ReID keeps
`ftfy==5.8`. Their installed tokenizer packages and native preprocessing are
not replaced. It journals
ordered, checksummed rows and replays the incomplete batch on resume.

## Subject localization and GROUP

Only `image_id` and `subjects[].select_text` enter localization. The same
OpenAI CLIP ViT-B/32 checkpoint and tokenizer pins encode predicted crops and
selection descriptions. No case, identity labels, annotation boxes, positive
sets or target image IDs infer membership.

If a reference has fewer candidates than Subject slots, add exactly that many
flagged full-reference candidates. Maximum-sum Hungarian assignment gives
every Subject a distinct anchor, including a fallback anchor when needed.
Unused real people may join a Subject when both conditions hold:

\[
S_{s,p} \ge 0.20,\qquad S_{s,p} \ge S_{s,a_s}-0.03.
\]

Each Subject has at most `5` members, including its anchor. Consider extras
in stable descending similarity order; competing Subjects use highest
similarity, then Subject order. A predicted person belongs to at most one
Subject. Fallbacks do not become extra members.

INDIVIDUAL, GROUP, DUAL and RELATIONAL use this same rule. A single Subject
can yield several components. Each native person encoder processes every
member with that Subject's native modification/instruction text. AdaFocal keeps
the complete reference image and applies its official anchor transform/mask
for each selected member. Its existing explicit handling of an anchor removed
by native TargetPad remains recorded in `invalid_localization_queries`.

**S5 now uses text for localization.** Its public ID is unchanged; display name
and metadata explicitly identify the adaptation. Text does not enter its ReID
encoder or cosine pair score. The separately labeled diagnostic is:

```bash
python methods/simple/05_reid_set/run.py --text-free-diagnostic
```

That diagnostic uses all reference detections, skips the selector, writes to
`runs/groundingdino_clipreid_set_textfree_diagnostic/`, and is not included in
the shared-person comparison table.

## Partial SetMatch and unmatched references

For the native pair-score matrix \(S\in\mathbb R^{K\times N}\), match exactly
\(m=\min(K,N)\) real one-to-one pairs to maximize their sum:

\[
\operatorname{Score}(S)=\frac{\sum_{(i,j)\in M^*}S_{i,j}+(K-m)b}{K}.
\]

This is a mean over all query components. It gives partial credit for `K>N`,
returns `b` when `N=0`, and forces the `m` real pairs even if a real score is
below `b`. A gallery person cannot be reused. Non-finite inputs fail explicitly.
Vectorized `K=1`, `K=2` and `N=1/2` specializations are exact; larger problems
use rectangular Hungarian. Native pair scores are never replaced by FAFA's.

| Methods | Fixed unmatched reference | Source |
| --- | --- | --- |
| Word4Per, FAFA, CLIP-ReID, S10, AdaFocal | `-1.0` | Lower bound of normalized cosine, top-k mean cosine or max-token cosine |
| S6, S9 | S5's `-1.0` before fusion | Same ReID branch; global/proxy fusion remains native |
| BASIC | Computed once from official external resources | Analytic lower bound of its unchanged polynomial score |
| Instruct-ReID | `-1000000.0` by default | Previous predeclared native Euclidean distance cutoff `1000`, explicitly **uncalibrated** |

BASIC bounds centered unit-vector norms using its official means, then bounds
the normalized nonnegative image/text scores. The score polynomial is separately
concave, so its minimum over that fixed rectangle occurs at a corner. No CPR
evaluation statistics or labels choose this value. The resolved numeric value
and source are recorded in `run.json`.

Instruct-ReID's native fusion vectors are **not L2-normalized**, so there is no
universal cosine bound. The legacy cutoff is not claimed to be calibrated and
can dominate missing-component rankings. For a final experiment needing an
empirical reference, freeze it once from an independent permitted dataset:

```bash
python -m cpr_person.calibration \
  --scores /absolute/external_negative_person_pair_scores.npy \
  --queries /absolute/external_queries.jsonl \
  --gallery /absolute/external_gallery.jsonl \
  --checkpoint checkpoints/instruct_reid/inference_language_reid.pth.tar \
  --source-commit 8250f44a301a50d8afcd2e09a46c3e96bf52090d \
  --dataset-id independent_calibration_dataset \
  --noncorresponding-pairs --quantile 0.01 \
  --output checkpoints/instruct_reid/unmatched_calibration.json
```

Use native negative-squared-distance scores of externally verified
noncorresponding pairs. Set `setmatch.unmatched_calibration` in P3 config to
that JSON path. Model/checkpoint identity and overlap guards are checked at
load time. Never generate this reference from CPR evaluation pairs or tune its
quantile on evaluation results. Calibration provenance is recorded in the run.

## Shared caches

```text
cache/shared/
  detections/<full-fingerprint>/persons.npz
  query_localization/<full-fingerprint>/members.json
```

Both files have content-checksummed `.meta.json` sidecars. Fingerprints include
ordered manifest hashes, actual gallery image byte hashes, detector checkpoint
checksum, pinned source/config, detector/crop/forward protocol, adapter version,
selector checksum/source and membership parameters. Localization also includes
the actual detection array digest. Atomic writes, OS locks, semantic validation
and ordered worker journals protect interruption/corruption.

Hashing gallery bytes is a CPU verification pass, not detector inference. A
matching verified cache skips detector/model construction. Native feature caches
also include their shared detection identity, native checkpoint and relevant
preprocessing configuration. SetMatch aggregation/background changes do not
invalidate detection, localization or native embeddings. FAFA retains separate
gallery/query feature and component-score stages for CPU-only reaggregation.

For Kaggle, use a writable root and separate read roots (colon separated on
Linux/macOS; OS path separator on Windows):

```bash
export CPR_SHARED_CACHE_ROOT=/kaggle/working/cpr_shared
export CPR_SHARED_CACHE_READ_ROOTS=/kaggle/input/cpr-shared-person-cache/cache/shared
# Optional mounted official weights, always checksummed:
export CPR_GDINO_CHECKPOINT=/absolute/groundingdino_swint_ogc.pth
export CPR_CLIP_SELECTOR_CHECKPOINT=/absolute/ViT-B-32.pt
```

`CPR_REQUIRE_SHARED_CACHE=1` refuses a missing detection or localization cache.
S5 additionally retains `--preflight-only` and `--require-detection-cache`.

Inspect actual mounted datasets before selecting paths:

```bash
python -m cpr_person.cache_tools inspect /kaggle/input/fafa-data /kaggle/input/rcr-proposed-cache
python -m cpr_person.cache_tools import-shared /absolute/verified_shared_root \
  --gallery-manifest data/gallery.jsonl --query-manifest data/queries.jsonl
```

Use the actual mount paths in the notebook if they differ. Import performs no
model inference and fails closed unless full fingerprints, content checksums
and semantic arrays/memberships match. Legacy files without equivalent
provenance are reported for inspection, not accepted by tensor shape.

The public Kaggle listing inspected during this refactor shows
`fafa-data/referential-composition-retrieval/...` and
`rcr-proposed-cache/features/<id>.pt`. The complete archives/provenance could
not be verified from this environment, so neither dataset was certified as an
equivalent Grounding DINO cache. DINO/DINOv3 feature tensors are not Grounding
DINO detections or FAFA features. Existing weights can be pointed to explicitly;
no notebook environment or feature tensor is automatically adopted.

## Install, validate and run

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q tests
python -m compileall -q cpr_person methods scripts tests run_baseline.py build_tables.py
python run_baseline.py --list
python validate_data.py
python run_baseline.py fafa_setmatch
# Or any stable method ID / directory alias:
python run_baseline.py word4per_setmatch
python run_baseline.py groundingdino_clipreid_set
```

The runner installs the method requirements, prepares its shared/native assets,
runs inference, invokes the **unchanged** evaluator and builds tables. It does
not fine-tune on the benchmark. Word4Per still needs the independently
reproduced Stage-2 artifacts described in
`checkpoints/word4per/README.word4per.md`.

S9 requires completed S5 and S8 source runs on exactly the same manifests:

```bash
python run_baseline.py groundingdino_clipreid_set
python run_baseline.py qwen_image_edit_clip
python run_baseline.py reid_set_qwen_edit_proxy
```

S9 and S11 need genuinely separate validation manifests. S11 requires S6's
validation-selected alpha artifact; configure S6's existing
`fusion.alpha_selection.mode: validation` before that dependency run.
Fixed-alpha S6/S10 remain label-free defaults. Validation guards reject copied
evaluation inputs even if query IDs or GT labels were changed.

To attempt every configured baseline in dependency order:

```bash
python run_baseline.py --all
```

Prepare the required checkpoints, validation inputs and S6 selection mode
first. Failures are reported and remaining methods are attempted; the command
returns nonzero if any method fails. Checkpoint-unavailable methods keep the
existing explicit unavailable status and produce no substitute scores.

Optional native CUDA smoke test after installing the selected method's
requirements and preparing its checkpoints:

```bash
python scripts/smoke_person_runtime.py --methods fafa_setmatch
python scripts/smoke_person_runtime.py --methods word4per_setmatch instruct_reid_setmatch basic_setmatch adafocal_setmatch
```

Smoke uses a temporary 40-scene/4-query subset and checks finite output shape;
it never evaluates or reports those subset scores as benchmark results.
S9/S11 are tested through their inherited branches in CPU integration tests;
their full native proxy/verifier runs still need their configured source runs.

Every normal run preserves complete gallery/query ordering and the
`scores.npy` / `run.json` contract, including the reference image column.
The evaluator alone excludes the reference. Tables skip old person-protocol
results and reject different actual detection/membership digests; rerun all
affected methods before comparing this protocol with itself.
