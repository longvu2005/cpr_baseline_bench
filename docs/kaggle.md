# Run all 29 methods on Kaggle

Use a writable checkout under `/kaggle/working`, Internet enabled for preparation,
and the complete finalized RGB images and original RCR detection index. Native
weights and very large snapshots can remain in read-only `/kaggle/input`.
No substitute checkpoint/model is generated if any dependency is missing.

## Apply the complete patch

The delivered baseline patch is based on commit
`65a7c373b98ece2fb632242d13f2ea0cdddd4de6`; its RCR companion is based on
`2fe92766d5158f8dd87297d3bc4cb6280b8ed63b`. The package provides a complete patch and a follow-up patch. On the unchanged
base commit, apply only `baseline_completion.patch`. If the previous
`CPR_RCR_Completion.zip` baseline patch is already applied, apply only
`baseline_followup.patch`; it is checked against that exact previous artifact.
Do not apply both. The RCR companion is unchanged and need not be applied twice.
Do not reset an existing working tree with uncommitted work.

After attaching the delivery as a Kaggle dataset, adjust `PATCH_DIR` to its actual
folder. Run this Bash cell once:

```bash
%%bash
set -euo pipefail
PATCH_DIR=/kaggle/input/cpr-rcr-completion/CPR_RCR_Completion

git clone https://github.com/longvu2005/cpr_baseline_bench.git /kaggle/working/cpr_baseline_bench
cd /kaggle/working/cpr_baseline_bench
git checkout 65a7c373b98ece2fb632242d13f2ea0cdddd4de6
git apply --check "$PATCH_DIR/baseline_completion.patch"
git apply "$PATCH_DIR/baseline_completion.patch"

git clone https://github.com/longvu2005/referential-composition-retrieval.git /kaggle/working/rcr
git -C /kaggle/working/rcr checkout 2fe92766d5158f8dd87297d3bc4cb6280b8ed63b
git -C /kaggle/working/rcr apply --check "$PATCH_DIR/rcr_cache_provenance.patch"
git -C /kaggle/working/rcr apply "$PATCH_DIR/rcr_cache_provenance.patch"
```

The RCR patch adds build-time image digests; it changes no Proposed model,
loss, optimizer or retrieval formula. A legacy detection/scene index must first
be numerically verified against actual frozen model forwards or rebuilt. The
workflow below verifies all original tensors into a separate proven index when
needed, preserving original tensors and cache IDs. The original RCR caches,
FAFA shards, supervision and trained checkpoints are not edited by this verifier.
A verification mismatch stops the run; use the correct images/environment or
rebuild the affected cache and dependencies rather than adding hashes by hand.
The scene verification copy contains the globals index for baseline reuse;
keep using the original full scene cache (including feature shards) for Proposed.

## Required external assets

A combined checkpoint dataset should mirror this repo's `checkpoints/` paths.
`stage_checkpoint_mount.py` creates writable directories, symlinks weight files,
and copies tokenizer/config/receipt metadata so preparation never writes to input.
Native downloaders can fill missing public assets. Original author source/checkpoint
availability and dependency compatibility remain requirements.

Word4Per requires real reproduced Stage-2 weights, matching training config and
`word4per_cuhk_pedes_stage2_reproduction.json` containing the external image audit.
An old receipt without this audit is not sufficient. To reproduce instead, supply
`WORD4PER_STAGE1_CHECKPOINT` and `WORD4PER_REPRO_DATA_ROOT`: the original CUHK-PEDES
and ITCPR layout documented by `reproduce_stage2.py`. The original training recipe
uses CUHK-PEDES and external ITCPR selection, never RCR val/test. This can take many
hours and is not a missing-weight fallback. Retain the actual training records.

Imagine and Seek requires the exact giant BLIP2 and Qwen32 GPTQ snapshots mounted
read-only. Its configured native total GPU check is 23 GiB, but this is only a
prerequisite, not evidence that a real run fits. WISER also requires BAGEL, BLIP2,
Qwen and OpenCLIP weights, sufficient writable disk, its native environments and
an OpenAI refiner credential. The 80GiB BAGEL setting is an allocation budget,
not a verified minimum GPU requirement. A single T4/P100 run of all 29 methods
has not been demonstrated. Use the reported blockers and actual native runtime
requirements; the workflow deliberately does not downsize native methods.

For WISER, enable an `OPENAI_API_KEY` Kaggle Secret and run a Python cell before
launching the Bash workflow. Keep the key out of notebook output:

```python
import os
from kaggle_secrets import UserSecretsClient
os.environ["OPENAI_API_KEY"] = UserSecretsClient().get_secret("OPENAI_API_KEY")
```

WISER makes the native API requests and can incur API charges. Its requests,
resolved response model and exact response JSON are retained as experiment evidence.

## Complete execution

Replace only the dataset path values with the mounted folders actually present.
`RCR_IMAGE_ROOT` must contain the original `train/`, `val/`, `test/` subdirectories;
the verifier needs every image in the full original cache. `RCR_SCENE_CACHE` is
optional: omit its export to encode gallery scenes with the same frozen HF pipeline.

```bash
%%bash
set -euo pipefail
export CPR_RCR_REPO=/kaggle/working/rcr
export RCR_FINAL_DIR=/kaggle/input/rcr-final/dataset/data/final
export RCR_IMAGE_ROOT=/kaggle/input/rcr-images/images
export RCR_DETECTION_CACHE=/kaggle/input/rcr-proposed-cache/cache/detections
export RCR_SCENE_CACHE=/kaggle/input/rcr-proposed-cache/cache/scene
export CPR_CHECKPOINT_MOUNT=/kaggle/input/cpr-native-checkpoints/checkpoints
export WORD4PER_STAGE2_PROVENANCE=/kaggle/input/word4per-reproduced/word4per_cuhk_pedes_stage2_reproduction.json
export IPCIR_BLIP2_DIR=/kaggle/input/imagine-seek-models/blip2-opt-6.7b-coco
export IPCIR_QWEN32_DIR=/kaggle/input/imagine-seek-models/Qwen1.5-32B-Chat-GPTQ-Int4
cd /kaggle/working/cpr_baseline_bench
bash scripts/run_all_kaggle.sh
```

The script imports exact val/test manifests, validates images, prepares shared HF
CLIP, verifies legacy indexes when needed, then runs all 29 in dependency order.
S9 requires successful S5 and S8 in the batch; S11 requires S6. Failure/unavailability
is recorded and propagates to dependents; remaining methods still run. Exit status
is nonzero if any method fails or publication artifacts are incomplete. No test
accuracy chooses prompts, cap, checkpoints, penalty protocol or method subset.

Outputs include `runs/batch_status.json`, per-method `scores.npy`/`run.json`,
portable `runs/<method>/evidence/`, `outputs/<method>/metrics.json`, the two tables,
and `runs/publication_readiness.json`. A successful artifact gate establishes
recorded run consistency, not independent certification of native fidelity.

For prerequisite inspection before expensive work:

```bash
python -m pip install -r requirements-dev.txt
python run_baseline.py --preflight
```

For a later retry, restore the saved nonsecret paths first and retry missing
methods after resolving their actual blockers. Use `--skip-install` only with
already prepared environments. Rerun the full batch after resolving failures so its recorded status reflects
the final runs; do not remove failed status merely to pass artifact validation.

```bash
source runs/kaggle_environment.sh
python run_baseline.py instruct_reid_setmatch --skip-install
python run_baseline.py --all --skip-install
python scripts/check_publication.py
```

## Direct parity tests

The offline unit suite needs `requirements-dev.txt`. Real component checks also
need Torch/Transformers/OpenAI CLIP, both pinned CLIP assets, a prepared selector
tokenizer and an RCR checkout. Opt-in missing assets fail rather than becoming a pass:

```bash
export CPR_TEST_RCR_REPO=/kaggle/working/rcr
export CPR_TEST_PRETRAINED_CLIP=1
export CPR_TEST_FULL_EVALUATOR=1
export OMP_NUM_THREADS=2
export USE_TF=0
export USE_FLAX=0
.baseline_venvs/shared_clip/bin/python -m pytest -q -rs
```

These component tests use CPU fixture images. They must not be reported as native
FAFA/ReID/pseudo-token/editor/VLM inference or as final benchmark model accuracy.
