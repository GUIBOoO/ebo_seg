#!/bin/bash
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --output=/scratch/guibo/slurm_outputs/slurm-%j.out
#SBATCH --error=/scratch/guibo/slurm_outputs/slurm-%j.err
#SBATCH --account=rrg-josedolz
#SBATCH --partition=compute


source ~/.bash_profile
module load python/3.11.5
source /home/guibo/ebo-seg/bin/activate

set -euo pipefail


ROOT_DIR="${ROOT_DIR:-/home/guibo/links/projects/rrg-josedolz/guibo/ebo_seg}"

DATASET="${DATASET:-acdc}"
DATASET_LOWER=$(printf '%s' "$DATASET" | tr '[:upper:]' '[:lower:]')

# DATASET_ARG is what d_matrix.py's --dataset accepts; brats3 is the same loader
# as brats, only with 3 channels (T1ce/T2/FLAIR) and contiguous labels 0..3.
case "$DATASET_LOWER" in
  acdc)
    DATA_DIR="$SLURM_TMPDIR/dataset/ACDC/database"
    DATASET_ZIP="$SCRATCH/datasets/ACDC/ACDC.zip"
    UNZIP_TARGET="$SLURM_TMPDIR/dataset"
    DATASET_ARG=acdc
    ;;
  brats)
    DATA_DIR="$SCRATCH/datasets/Brats/data_slices"
    DATASET_ARG=brats
    ;;
  brats3)
    # Built by extract_data_brats3.py. The only BraTS variant TransUNet accepts.
    DATA_DIR="$SCRATCH/datasets/Brats/data_slices_3ch"
    DATASET_ARG=brats
    ;;
  *)
    echo "ERROR: unsupported dataset '$DATASET'. Expected 'acdc', 'brats' or 'brats3'."
    exit 1
    ;;
esac

if [ "$DATASET_LOWER" = "acdc" ]; then
    mkdir -p "$DATA_DIR"

    if [ ! -f "$DATASET_ZIP" ]; then
        echo "ERROR: ACDC zip not found"
        exit 1
    fi

    echo "Décompression ACDC images..."
    unzip -q "$DATASET_ZIP" -d "$UNZIP_TARGET"
else
    if [ ! -d "$DATA_DIR" ]; then
        echo "ERROR: prepared BraTS directory not found at $DATA_DIR"
        exit 1
    fi
fi

export PYTHON_DATA_DIR="$DATA_DIR"
DATASET_ROOT="${DATASET_ROOT:-$DATA_DIR}"

# MODEL only picks the default checkpoint/output paths. The architecture,
# image_size and in_channels are read back from the checkpoint by d_matrix.py,
# so a TransUNet checkpoint is rebuilt with the geometry it was trained on.
MODEL="${MODEL:-transunet}"
MODEL_LOWER=$(printf '%s' "$MODEL" | tr '[:upper:]' '[:lower:]')

CHECKPOINT=${CHECKPOINT:-"$SCRATCH/models/ebo_seg/$MODEL_LOWER/$DATASET_LOWER/best_ce_dice.pt"}
OUTPUT_DIR=${OUTPUT_DIR:-"$SCRATCH/d_matrices/$MODEL_LOWER/$DATASET_LOWER"}
# D must be estimated on a split you do NOT report on: val, not test.
SPLIT=${SPLIT:-"val"}
BATCH_SIZE=${BATCH_SIZE:-4}
DEVICE=${DEVICE:-"cuda"}
LAMBDA_WEIGHT=${LAMBDA_WEIGHT:-0.5}
SEED=${SEED:-42}

if [ "$MODEL_LOWER" = "transunet" ] && [ "$DATASET_LOWER" = "brats" ]; then
    echo "ERROR: TransUNet's pretrained stem takes 3 channels; raw BraTS has 4."
    echo "Use DATASET=brats3 (built by extract_data_brats3.py)."
    exit 1
fi

if [ -z "$DATASET_ROOT" ]; then
    echo "ERROR: DATASET_ROOT is empty. Set DATASET_ROOT=/path/to/dataset or PYTHON_DATA_DIR."
    exit 1
fi
if [ ! -f "$CHECKPOINT" ]; then
    echo "ERROR: checkpoint not found: $CHECKPOINT"
    echo "Set CHECKPOINT=/path/to/best_<loss>.pt"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"
cd "$ROOT_DIR"

echo "SLURM_JOB_ID=${SLURM_JOB_ID:-local}"
echo "Dataset      : $DATASET_LOWER (--dataset $DATASET_ARG)"
echo "Dataset root : $DATASET_ROOT"
echo "Model        : $MODEL_LOWER (architecture read from checkpoint)"
echo "Checkpoint   : $CHECKPOINT"
echo "Split        : $SPLIT"
echo "Lambda weight: $LAMBDA_WEIGHT"
echo "Output       : $OUTPUT_DIR"

python d_matrix.py \
    --checkpoint "$CHECKPOINT" \
    --dataset-root "$DATASET_ROOT" \
    --dataset "$DATASET_ARG" \
    --output-dir "$OUTPUT_DIR" \
    --split "$SPLIT" \
    --batch-size "$BATCH_SIZE" \
    --device "$DEVICE" \
    --lambda-weight "$LAMBDA_WEIGHT" \
    --seed "$SEED"

echo
echo "Done. Pass the resulting .npy to slurm_test_unet.sh via D_MATRIX=..."
