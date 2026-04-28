#!/bin/bash
#SBATCH --time=01:10:00
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
    ;;
  brats)
    DATA_DIR="$SCRATCH/datasets/Brats/data_slices"
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

PYTHON_BIN="${PYTHON_BIN:-python}"
if [ "$DATASET_LOWER" = "acdc" ]; then
    DEFAULT_CHECKPOINT="/home/guibo/links/scratch/models/ebo_seg/acdc/ebo_17_5_in0p3/best_ebo_ce.pt"
    DEFAULT_OUTPUT_DIR="/home/guibo/links/scratch/inference/acdc/inference_ebo_ce_17_5_in0p3"
else
    DEFAULT_CHECKPOINT="/home/guibo/links/scratch/models/ebo_seg/brats/ebo_17_5_0p3in/best_ebo_ce.pt"
    DEFAULT_OUTPUT_DIR="/home/guibo/links/scratch/inference/brats/inference_ebo_ce_17_5_0p3in"
fi

CHECKPOINT="${CHECKPOINT:-$DEFAULT_CHECKPOINT}"
OUTPUT_DIR="${OUTPUT_DIR:-$DEFAULT_OUTPUT_DIR}"
BATCH_SIZE="${BATCH_SIZE:-4}"
DEVICE="${DEVICE:-cuda}"
NUM_SAMPLES="${NUM_SAMPLES:-4}"
TEMPERATURE="${TEMPERATURE:-1.0}"
MAX_PIXELS_KDE="${MAX_PIXELS_KDE:-200000}"
ENERGY_THRESH="${ENERGY_THRESH:--21}"
MODES="${MODES:-all}"

cd /home/guibo/links/projects/rrg-josedolz/guibo/ebo_seg

echo "SLURM_JOB_ID=${SLURM_JOB_ID:-local}"
echo "Dataset      : ${DATASET_LOWER}"
echo "Dataset root : ${DATASET_ROOT}"
echo "Checkpoint   : ${CHECKPOINT}"
echo "Output dir   : ${OUTPUT_DIR}"
echo "Modes        : ${MODES}"

"${PYTHON_BIN}" inference.py ${MODES} \
  --checkpoint "${CHECKPOINT}" \
  --dataset "${DATASET_LOWER}" \
  --dataset-root "${DATASET_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --batch-size "${BATCH_SIZE}" \
  --device "${DEVICE}" \
  --num-samples "${NUM_SAMPLES}" \
  --temperature "${TEMPERATURE}" \
  --max-pixels-kde "${MAX_PIXELS_KDE}" \
  --energy-threshold "${ENERGY_THRESH}"
