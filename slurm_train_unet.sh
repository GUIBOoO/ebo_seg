#!/bin/bash
#SBATCH --time=10:00:00
#SBATCH --partition=compute_full_node
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
    DEFAULT_NUM_CLASSES=4
    ;;
  brats)
    DATA_DIR="$SCRATCH/datasets/Brats/data_slices"
    DEFAULT_NUM_CLASSES=5
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

echo "Structure finale :"
find "$DATA_DIR" -maxdepth 1 -type d

export PYTHON_DATA_DIR="$DATA_DIR"
DATASET_ROOT="${DATASET_ROOT:-$DATA_DIR}"

OUTPUT_DIR="${OUTPUT_DIR:-/home/guibo/links/scratch/models/ebo_seg/brats/hybridebo_17_5_5in}"
PYTHON_BIN="${PYTHON_BIN:-python}"
EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-8}"
LR="${LR:-1e-3}"
NUM_WORKERS="${NUM_WORKERS:-4}"
IMAGE_SIZE="${IMAGE_SIZE:-256}"
NUM_CLASSES="${NUM_CLASSES:-$DEFAULT_NUM_CLASSES}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-42}"
LOSS="${LOSS:-hybrid_ebo_ce}"
TRACK_LOSS_GRADIENTS="${TRACK_LOSS_GRADIENTS:-0}"

cd /home/guibo/links/projects/rrg-josedolz/guibo/ebo_seg

echo "SLURM_JOB_ID=${SLURM_JOB_ID:-local}"
echo "Dataset      : ${DATASET_LOWER}"
echo "Dataset root : ${DATASET_ROOT}"
echo "Output dir   : ${OUTPUT_DIR}"
echo "Track grads  : ${TRACK_LOSS_GRADIENTS}"

TRACK_LOSS_GRADIENTS_ARGS=()
if [ "${TRACK_LOSS_GRADIENTS}" = "1" ]; then
  TRACK_LOSS_GRADIENTS_ARGS+=(--track-loss-gradients)
fi

echo "About to launch training..."

"${PYTHON_BIN}" train_unet.py \
  --dataset "${DATASET_LOWER}" \
  --dataset-root "${DATASET_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --lr "${LR}" \
  --num-workers "${NUM_WORKERS}" \
  --image-size "${IMAGE_SIZE}" \
  --num-classes "${NUM_CLASSES}" \
  --device "${DEVICE}" \
  --seed "${SEED}" \
  --loss "${LOSS}" \
  "${TRACK_LOSS_GRADIENTS_ARGS[@]}"
