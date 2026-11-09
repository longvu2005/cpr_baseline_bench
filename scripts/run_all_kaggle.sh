#!/usr/bin/env bash
set -euo pipefail
: "${CPR_RCR_REPO:?Set the RCR checkout path}"
: "${RCR_FINAL_DIR:?Set the complete final export path}"
: "${RCR_IMAGE_ROOT:?Set the full RGB image root}"
: "${RCR_DETECTION_CACHE:?Set the original RCR detection directory/index}"
cd "$(dirname "$0")/.."
case "$PWD" in /kaggle/input|/kaggle/input/*) echo 'Use a checkout under /kaggle/working.' >&2; exit 1;; esac
export HF_HOME=/kaggle/working/hf-cache
export HF_HUB_CACHE="$HF_HOME/hub"
export TORCH_HOME=/kaggle/working/torch-cache
export XDG_CACHE_HOME=/kaggle/working/xdg-cache
export PYTHONDONTWRITEBYTECODE=1
export USE_TF=0
export USE_FLAX=0
export CPR_SHARED_CACHE_ROOT="$PWD/cache/shared"
export CPR_SCENE_CACHE_ROOT="$PWD/cache/scene_clip"
python -m pip install -r requirements-dev.txt
if [[ -n "${CPR_CHECKPOINT_MOUNT:-}" ]]; then
    python scripts/stage_checkpoint_mount.py --source "$CPR_CHECKPOINT_MOUNT" --destination checkpoints
fi
python prepare_data.py --rcr-final-dir "$RCR_FINAL_DIR"
python link_gallery.py --source "$RCR_IMAGE_ROOT"
python validate_data.py
python -c 'from cpr_baselines.scene_runtime import prepare_runtime;prepare_runtime()'
export CPR_SCENE_CLIP_PYTHON="$PWD/.baseline_venvs/shared_clip/bin/python"
"$CPR_SCENE_CLIP_PYTHON" -m cpr_baselines.scene_worker --prepare checkpoints/clip/hf-vit-base-patch32
has_provenance() {
    "$CPR_SCENE_CLIP_PYTHON" -c 'import sys,torch;from pathlib import Path;p=Path(sys.argv[1]);p=p/"index.pt" if p.is_dir() else p;x=torch.load(p,map_location="cpu",weights_only=True,mmap=True);raise SystemExit(0 if isinstance(x.get("image_content_sha256"),list) and len(x["image_content_sha256"])==len(x["image_ids"]) else 1)' "$1"
}
verify=()
export CPR_RCR_DETECTION_CACHE="$RCR_DETECTION_CACHE"
if ! has_provenance "$RCR_DETECTION_CACHE"; then
    verify+=(--detections "$RCR_DETECTION_CACHE")
    export CPR_RCR_DETECTION_CACHE="$PWD/cache/verified_rcr/detections"
fi
if [[ -n "${RCR_SCENE_CACHE:-}" ]]; then
    export CPR_RCR_SCENE_CACHE="$RCR_SCENE_CACHE"
    if ! has_provenance "$RCR_SCENE_CACHE"; then
        verify+=(--scene "$RCR_SCENE_CACHE")
        export CPR_RCR_SCENE_CACHE="$PWD/cache/verified_rcr/scene"
    fi
fi
if ((${#verify[@]})); then
    "$CPR_SCENE_CLIP_PYTHON" scripts/verify_rcr_cache_provenance.py --rcr-repo "$CPR_RCR_REPO" \
        --rcr-final-dir "$RCR_FINAL_DIR" --image-root "$RCR_IMAGE_ROOT" \
        --output "$PWD/cache/verified_rcr" --device cuda "${verify[@]}"
fi
python - <<'ENV_PY'
from pathlib import Path
import os,shlex
names=['CPR_RCR_DETECTION_CACHE','CPR_RCR_SCENE_CACHE','CPR_SCENE_CLIP_PYTHON','CPR_SCENE_CLIP_CHECKPOINT',
       'CPR_SHARED_CACHE_ROOT','CPR_SCENE_CACHE_ROOT','CPR_SHARED_CACHE_READ_ROOTS','CPR_SCENE_CACHE_READ_ROOTS',
       'HF_HOME','TORCH_HOME','XDG_CACHE_HOME']
names += ['HF_HUB_CACHE','WORD4PER_STAGE2_PROVENANCE','IPCIR_BLIP2_DIR','IPCIR_QWEN32_DIR']
path=Path('runs/kaggle_environment.sh');path.parent.mkdir(parents=True,exist_ok=True)
path.write_text(''.join(f'export {name}={shlex.quote(os.environ[name])}\n' for name in names if name in os.environ))
ENV_PY
status=0
python run_baseline.py --all || status=$?
python scripts/check_publication.py || status=$?
exit "$status"
