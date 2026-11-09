# Shared HF person detector (protocol v2)

This page records the previous standalone detector runtime for diagnostics.
Official val/test person baselines now consume the RCR detection index. Use the
[current benchmark setup](baselines.md#detection-and-crop-convention) for runs.


Audit base: CPR `0fbe43a98e920c634a9a638c4287ac6b1cd1d2f4`.
Read-only reference: RCR `def2f0d56330f5a27cd4aec52becdbbb9972a16d`.

## Original architecture and migration

The shared detector/CLIP Subject selector already lived in `cpr_person`; methods
kept their native encoders and scores. Its old worker used official GroundingDINO
Swin-T, threshold .35, confidence-sorted NMS .5, 4px minimum boxes, a 10-person
cap, clipped integer floor/ceil boxes, and the calling method's Python.
`cpr_person/requirements.txt` pinned Transformers 4.39.3, which is native FAFA's
version but lacks the required modern HF Grounding DINO API. Structured Caption
Retrieval additionally installed the shared dependencies into its parent runtime.

The new worker is the only detector constructor. It uses HF AutoProcessor and
AutoModelForZeroShotObjectDetection with model ID
`IDEA-Research/grounding-dino-tiny` and immutable model/processor revision
`a2bb814dd30d776dcf7e30523b00659f4f141c71`. The HF revision was resolved from the
model API during this audit. Its processor configuration uses bilinear resize,
shortest edge 800 / longest edge 1333, rescale 1/255, ImageNet mean/std and
processor padding. Inference uses the revision's default processor without
additional transforms, one image per forward, eval float32 without autocast.
The runtime pins are in `protocol.yaml` and `detector-requirements.txt`.

## Actual RCR processing

`configs/proposed.yaml`, `src/rcr/common/vision.py`,
`src/rcr/proposed/cache/dino.py` and `store.py` were inspected at the reference
commit. RCR supports Transformers >=4.56,<5; it does not pin its HF revision.
The CPR migration deliberately pins both version 4.56.2 and the revision.

| Step | RCR Proposed | Migrated CPR |
| --- | --- | --- |
| Input | Stored RGB pixels, no EXIF transpose | Same |
| Caption | `processor(images=image,text=[["person"]],return_tensors="pt")`; HF turns labels into `person.` | Same |
| Processor/model | HF Tiny AutoProcessor / AutoModelForZeroShotObjectDetection | Same classes/model, pinned revisions, prepared local files only |
| Postprocess | `threshold=.30`, `text_threshold=.25`, `target_sizes=[(height,width)]` | Same |
| Labels | Exact `text_labels == "person"` | Same; do not accept substring/synonym matches |
| Boxes | Pixel `xyxy` floats in model query order | Same, stored as float32 |
| Extra filtering | No NMS, clipping, min size, cap or sorting | Same |
| Confidence | Processor thresholds boxes by max sigmoid token score; helper discards scores | Keep original accepted processor score, aligned with each box |
| Crop | PIL `image.crop(tuple(box.tolist()))` | Same; nearest/even edge rounding and zero padding happen only inside PIL |
| Empty set | Zero person features/false person mask | One CPR full-image candidate with `fallback=true`, score -1 |

Text threshold selects tokens for decoded phrases; changing it can therefore
change exact-person membership. The processor's box threshold is strict `>`.
RCR normalizes a separate copy of boxes for letterboxed scene coordinates;
its `boxes_pixel` remain raw. CPR consumes the raw pixel boxes. RCR's later
fine-scoring duplicate handling is a retrieval rule, not detector postprocessing,
and has not been imported into CPR.

The fallback difference is deliberate to preserve CPR's existing native SetMatch
and Subject-selector contract. It is never counted as a genuine detection.
All real box coordinates and their order remain unchanged. Predictions can extend
outside image bounds; consumers use PIL directly instead of modifying geometry.
Matching detectors does not make CPR and RCR gallery/evaluation protocols or mAP
values directly comparable. No pretrained numerical parity is claimed here.

## Dependency map

| Method | Detector/cache route | Independent detector |
| --- | --- | --- |
| `word4per_setmatch` | `load_person_context` | No |
| `fafa_setmatch` | `load_person_context`; native query/gallery crop datasets | No |
| `instruct_reid_setmatch` | `load_person_context` | No |
| `basic_setmatch` | `load_person_context` | No |
| `adafocal_setmatch` | `load_person_context` | No |
| `groundingdino_clipreid_set` | `load_person_context` / shared cache guard | No |
| `groundingdino_clipreid_set_text` | groundingdino_clipreid_set `prepare_person_branch` | No |
| `reid_set_qwen_edit_proxy` | groundingdino_clipreid_set branch; own Qwen proxy stays native | No |
| `per_person_clip_compose_setmatch` | `load_person_context` | No |
| `structured_caption_retrieval` | Shared preflight; caption runtime cache-only | No |
| `retrieve_qwen25vl_rerank` | Completed groundingdino_clipreid_set_text scores; checks protocol version | No |
| groundingdino_clipreid_set text-free diagnostic | Same detections, no selector | No |
| All other scene baselines | Native scene inference, no shared detector | No |

The registry is centralized in `cpr_person.protocol.PERSON_METHODS`, also
re-exported by the adapter. The Imagine & Seek scene adapter stubs unused
interactive GroundingDINO imports; it does not construct a person detector.

## Install and run

Use Python 3.10-3.12 for the pinned Torch 2.5.1 / native FAFA dependencies.
From the repository root, use the current [Kaggle workflow](kaggle.md) or
pass the RCR detection cache explicitly. The original RCR index is read as is,
bound by its `cache_id`, gallery order and image sizes (digests are checked when
present); see [Kaggle workflow](kaggle.md#get-the-code).
The runner prepares `.baseline_venvs/shared_clip` for scene CLIP and the shared
selector. It consumes RCR's detector index without constructing a detector.
The standalone worker below remains a diagnostic path only. Native
FAFA remains pinned to Transformers 4.39.3 and max_txt_len/max_words 128.
The shared consumer requirements no longer include Transformers. AdaFocal keeps
its formerly inherited native 4.39.3 pin explicitly; reid_set_qwen_edit_proxy keeps native 4.50.3.
Pinned OpenAI CLIP needs pkg_resources during setup; the environment manager
installs setuptools<81/wheel into the target venv and builds without isolation.
It never installs those method requirements into the parent notebook environment.

To prepare the detector and canonical gallery/Subject cache separately:

```bash
python -m cpr_person.runtime
python -m cpr_person.prepare
.baseline_venvs/shared_person_detector/bin/python -m cpr_baselines.person_cache --device cuda
```

Use `--device cpu` for a CPU cache build. On Windows the executable is
`.baseline_venvs/shared_person_detector/Scripts/python.exe`.
For an explicitly managed external environment:

```bash
python -m venv /absolute/envs/cpr-detector
/absolute/envs/cpr-detector/bin/python -m pip install 'setuptools>=65,<81' wheel
/absolute/envs/cpr-detector/bin/python -m pip install --no-build-isolation -r cpr_person/detector-requirements.txt
export CPR_DETECTOR_PYTHON=/absolute/envs/cpr-detector/bin/python
python -m cpr_person.runtime --skip-install
python -m cpr_person.prepare
"$CPR_DETECTOR_PYTHON" -m cpr_baselines.person_cache --device cuda
python run_baseline.py fafa_setmatch
python run_baseline.py groundingdino_clipreid_set
```

`CPR_DETECTOR_PYTHON` must name an executable, not a shell command. When set,
the runner checks that environment and never installs into it. Without it,
workers explicitly select the prepared default path; they never use the native
method's sys.executable. `--skip-install` requires prepared, compatible runtimes.
The worker clears inherited native Hugging Face cache variables and uses an
explicit detector cache directory. Inference passes `local_files_only=True`.
Only `cpr_person.prepare` downloads the immutable HF snapshots and records their
content SHA256 inventory. A missing detector runtime or assets fails clearly.

## Cache regeneration and boundaries

Protocol version is `2026-10-22-person-hf-v2`. Detection fingerprints include
model ID/revision, processor revision, runtime versions, implementation/caption/
preprocessing/postprocessing conventions, thresholds, box/crop/order/fallback
rules, ordered gallery manifest SHA256 and actual image-content provenance.
Local interpreter/cache paths do not invalidate portable cache identity.

Old Swin-T caches are **not reusable**. Keep them for historical reproducibility;
normal runs write a new key. Rebuild detection and Subject localization once.
Downstream native features/component scores carry detection identity/array hashes
and localization identity, so affected methods regenerate their features/scores.
Scene-only embeddings are unaffected. groundingdino_clipreid_set_text/reid_set_qwen_edit_proxy/retrieve_qwen25vl_rerank inherited results need their
new-protocol groundingdino_clipreid_set/groundingdino_clipreid_set_text source runs. Historical evaluated mAP values are not rewritten.

Atomic data/sidecar writes, OS locks, mounted read roots, checksum validation and
ordered checksummed journals remain. The one-image forward means a resumed run
continues at the next complete image. Read-only cache reuse performs no model
imports, asset download or detector interpreter selection; it still verifies
manifest/image bytes. `CPR_REQUIRE_SHARED_CACHE=1` rejects all cache misses.

## CPU verification

```bash
python -m pytest -q
.baseline_venvs/shared_person_detector/bin/python -m pytest -q tests/test_person_hf_detector.py
python -m compileall -q cpr_person cpr_baselines methods scripts tests run_baseline.py baseline_environment.py
python -m pyflakes cpr_person baseline_environment.py run_baseline.py tests/test_person_hf_detector.py
```

The tests use synthetic outputs; the optional real HF processor test needs the
pinned detector runtime, never downloads weights, and exercises actual caption,
resize, token thresholds and normalized-to-pixel conversion. All existing native
architecture signatures are compared with this patch's base commit; only two
stale golden digests (previous FAFA text-policy and Instruct-ReID changes) were
refreshed. Their native functions are identical to the base commit. The official
evaluator, matching/aggregation functions, annotations and split files are
unchanged. The reporting follow-up fixes the table-reporting failures present
at the audit base; it adds a standard-library-only tables path and validates
shared provenance before comparison. See [reporting](reporting.md).
