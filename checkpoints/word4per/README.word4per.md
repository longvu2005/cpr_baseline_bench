## Word4Per + SetMatch

Model:  
Word4Per / Word for Person: Zero-shot Composed Person Retrieval, with benchmark-level SetMatch adaptation for MULTI queries.

Official source:  
https://github.com/Delong-liu-bupt/Composed_Person_Retrieval

Official implementation:  
`old_project`

Paper:  
*Word for Person: Zero-shot Composed Person Retrieval*, arXiv:2311.16515.

Checkpoint availability:

- The official Word4Per repository does not provide a reliably accessible final Stage-2 checkpoint.
- The benchmark therefore reproduces Word4Per Stage-1 and Stage-2 from the official `old_project` implementation.
- Training uses CUHK-PEDES only.
- The CUHK-PEDES copy used here is downloaded from the OCDL repository release, which provides the original CUHK-PEDES dataset in the expected structure.
- Stage-2 initializes from the reproduced Stage-1 checkpoint.
- SetMatch is a benchmark-level adaptation for MULTI queries and is not part of the original Word4Per method.
- No training, fine-tuning, or hyperparameter tuning is performed on the CPR pilot.

Benchmark policy:  
Reproduce Word4Per Stage-1 and Stage-2 on CUHK-PEDES, then use the resulting Stage-2 checkpoint only for inference on the CPR benchmark.

Do not train or tune on the CPR pilot.

### Download and reproduce

Clone the official Word4Per source:

```bash
%%bash
set -e

cd /kaggle/working/cpr_baseline_bench

mkdir -p checkpoints/word4per

if [ ! -d checkpoints/word4per/Composed_Person_Retrieval ]; then
    git clone \
        --branch main \
        --single-branch \
        https://github.com/Delong-liu-bupt/Composed_Person_Retrieval.git \
        checkpoints/word4per/Composed_Person_Retrieval
fi

test -f \
    checkpoints/word4per/Composed_Person_Retrieval/old_project/train_stage1.py

echo "Word4Per source OK"
```

Download CUHK-PEDES from the OCDL dataset release:

```bash
%%bash
set -e

pip install -q gdown

cd /kaggle/working

gdown --fuzzy \
    "https://drive.google.com/file/d/1X7rmw0TmDjqa0b69qCn_EGSK3KCC-8zs/view?usp=drive_link" \
    -O tbpr_datasets.zip

ls -lh /kaggle/working/tbpr_datasets.zip
```

Extract the dataset:

```bash
%%bash
set -e

rm -rf /kaggle/working/tbpr_data_raw
mkdir -p /kaggle/working/tbpr_data_raw

unzip -q \
    /kaggle/working/tbpr_datasets.zip \
    -d /kaggle/working/tbpr_data_raw

find /kaggle/working/tbpr_data_raw \
    -type f \
    -name "reid_raw.json" \
    -print
```

Normalize the CUHK-PEDES path expected by Word4Per:

```bash
%%bash
set -e

ANN="$(
    find /kaggle/working/tbpr_data_raw \
        -type f \
        -name "reid_raw.json" \
        | head -n 1
)"

if [ -z "$ANN" ]; then
    echo "ERROR: reid_raw.json not found"
    exit 1
fi

CUHK_DIR="$(dirname "$ANN")"

if [ ! -d "$CUHK_DIR/imgs" ]; then
    echo "ERROR: imgs/ not found in $CUHK_DIR"
    exit 1
fi

rm -rf /kaggle/working/word4per_data
mkdir -p /kaggle/working/word4per_data

ln -s "$CUHK_DIR" \
    /kaggle/working/word4per_data/CUHK-PEDES

echo "CUHK-PEDES path:"
readlink -f /kaggle/working/word4per_data/CUHK-PEDES

ls -lah /kaggle/working/word4per_data/CUHK-PEDES
```

The resulting structure must be:

```text
/kaggle/working/word4per_data/
└── CUHK-PEDES/
    ├── imgs/
    └── reid_raw.json
```

Reproduce Word4Per Stage-1:

```bash
%%bash
set -e

WORD4PER=/kaggle/working/cpr_baseline_bench/checkpoints/word4per/Composed_Person_Retrieval/old_project

cd "$WORD4PER"

CUDA_VISIBLE_DEVICES=0 \
python3 train_stage1.py \
    --name word4per_stage1 \
    --root_dir /kaggle/working/word4per_data \
    --img_aug \
    --batch_size 64 \
    --MLM \
    --dataset_name CUHK-PEDES \
    --loss_names 'sdm+mlm+id' \
    --num_epoch 60
```

Install the reproduced Stage-1 checkpoint at the path required by Stage-2:

```bash
%%bash
set -e

WORD4PER=/kaggle/working/cpr_baseline_bench/checkpoints/word4per/Composed_Person_Retrieval/old_project

cd "$WORD4PER"

STAGE1="$(
    find logs/CUHK-PEDES \
        -type f \
        -name "best.pth" \
        -printf '%T@ %p\n' \
    | sort -n \
    | tail -n 1 \
    | cut -d' ' -f2-
)"

if [ -z "$STAGE1" ]; then
    echo "ERROR: Stage-1 best.pth not found"
    exit 1
fi

mkdir -p models

cp "$STAGE1" \
    models/stage1_model_vitb.pth

ls -lh models/stage1_model_vitb.pth
```

Reproduce Word4Per Stage-2:

```bash
%%bash
set -e

WORD4PER=/kaggle/working/cpr_baseline_bench/checkpoints/word4per/Composed_Person_Retrieval/old_project

cd "$WORD4PER"

CUDA_VISIBLE_DEVICES=0 \
python3 train_stage2.py \
    --name word4per_stage2 \
    --root_dir /kaggle/working/word4per_data \
    --img_aug \
    --batch_size 128 \
    --MLM \
    --lr 1e-4 \
    --optimizer AdamW \
    --dataset_name CUHK-PEDES \
    --loss_names 'sdm+id+mlm' \
    --toword_loss 'text' \
    --num_epoch 60
```

Install the reproduced Stage-2 checkpoint into the benchmark checkpoint directory:

```bash
%%bash
set -e

BENCH=/kaggle/working/cpr_baseline_bench
WORD4PER="$BENCH/checkpoints/word4per/Composed_Person_Retrieval/old_project"

cd "$WORD4PER"

STAGE2="$(
    find logs/CUHK-PEDES \
        -type f \
        -name "best.pth" \
        -printf '%T@ %p\n' \
    | sort -n \
    | tail -n 1 \
    | cut -d' ' -f2-
)"

if [ -z "$STAGE2" ]; then
    echo "ERROR: Stage-2 best.pth not found"
    exit 1
fi

STAGE2_DIR="$(dirname "$STAGE2")"

cp "$STAGE2" \
    "$BENCH/checkpoints/word4per/word4per_cuhk_pedes_stage2_best.pth"

cp "$STAGE2_DIR/configs.yaml" \
    "$BENCH/checkpoints/word4per/word4per_cuhk_pedes_stage2_configs.yaml"

ls -lh \
    "$BENCH/checkpoints/word4per/word4per_cuhk_pedes_stage2_best.pth" \
    "$BENCH/checkpoints/word4per/word4per_cuhk_pedes_stage2_configs.yaml"
```

Expected benchmark checkpoint files:

```text
checkpoints/word4per/word4per_cuhk_pedes_stage2_best.pth
checkpoints/word4per/word4per_cuhk_pedes_stage2_configs.yaml
```

Run Word4Per + SetMatch on the CPR benchmark:

```bash
%%bash
set -e

cd /kaggle/working/cpr_baseline_bench

python3 methods/published/01_word4per_setmatch/run.py

python3 evaluate.py \
    --method word4per_setmatch
```

The complete reproduction pipeline is:

```text
Official Word4Per source
        ↓
CUHK-PEDES
        ↓
Word4Per Stage-1
        ↓
stage1_model_vitb.pth
        ↓
Word4Per Stage-2
        ↓
word4per_cuhk_pedes_stage2_best.pth
        ↓
Word4Per + SetMatch inference
        ↓
CPR benchmark evaluation
```

Do not commit dataset or checkpoint files.