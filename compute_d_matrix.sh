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


DATASET="${DATASET:-brats}"
DATASET_LOWER=$(printf '%s' "$DATASET" | tr '[:upper:]' '[:lower:]')

case "$DATASET_LOWER" in
  acdc)
    DATA_DIR="$SLURM_TMPDIR/dataset/ACDC/database"
    DATASET_ZIP="$SCRATCH/datasets/ACDC/ACDC.zip"
    UNZIP_TARGET="$SLURM_TMPDIR/dataset"
    DEFAULT_MODEL_ROOT="/home/guibo/links/scratch/grid_search/1.005log3boundebo1margin"
    ;;
  brats)
    DATA_DIR="$SCRATCH/datasets/Brats/data_slices"
    DEFAULT_MODEL_ROOT="/home/guibo/links/scratch/grid_search/brats/ebo_ce"
    ;;
  *)
    echo "ERROR: unsupported dataset '$DATASET'. Expected 'acdc' or 'brats'."
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

CHECKPOINT=${CHECKPOINT:-"/home/guibo/links/scratch/models/ebo_seg/brats/ce_dice_baseline/best_ce_dice.pt"}
DATASET=${DATASET:-"brats"}
OUTPUT_DIR=${OUTPUT_DIR:-"/home/guibo/links/scratch/d_matrices/brats"}
SPLIT=${SPLIT:-"val"}
BATCH_SIZE=${BATCH_SIZE:-4}
DEVICE=${DEVICE:-"cuda"}
LAMBDA_WEIGHT=${LAMBDA_WEIGHT:-0.5}
SEED=${SEED:-42}

if [ -z "$DATASET_ROOT" ]; then
    echo "ERROR: DATASET_ROOT is empty. Set DATASET_ROOT=/path/to/dataset or PYTHON_DATA_DIR."
    exit 1
fi

python d_matrix.py\
    --checkpoint "$CHECKPOINT" \
    --dataset-root "$DATASET_ROOT" \
    --dataset "$DATASET" \
    --output-dir "$OUTPUT_DIR" \
    --split "$SPLIT" \
    --batch-size "$BATCH_SIZE" \
    --device "$DEVICE" \
    --lambda-weight "$LAMBDA_WEIGHT" \
    --seed "$SEED"
