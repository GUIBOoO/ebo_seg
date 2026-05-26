#!/bin/bash
#SBATCH --time=00:40:00
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

DATASET="${DATASET:-acdc}"
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

OUTPUT_DIR="${OUTPUT_DIR:-/home/guibo/links/scratch/models/ebo_seg/acdc/boundebo_10_5_outin5_cenin2_t1.02}"
PYTHON_BIN="${PYTHON_BIN:-python}"
EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-8}"
LR="${LR:-1e-3}"
NUM_WORKERS="${NUM_WORKERS:-4}"
IMAGE_SIZE="${IMAGE_SIZE:-256}"
NUM_CLASSES="${NUM_CLASSES:-$DEFAULT_NUM_CLASSES}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-42}"
LOSS="${LOSS:-bound_log_ebo}"
LAMBDA_EBO_IN="${LAMBDA_EBO_IN:-0.1}"
LAMBDA_EBO_CORR="${LAMBDA_EBO_CORR:-0.1}"
LAMBDA_EBO_CEN_IN="${LAMBDA_EBO_CEN_IN:-2}"
LAMBDA_EBO_OUT_IN="${LAMBDA_EBO_OUT_IN:-5}"
LAMBDA_EBO_CEN_CORR="${LAMBDA_EBO_CEN_CORR:-0.1}"
LAMBDA_EBO_OUT_CORR="${LAMBDA_EBO_OUT_CORR:-0.1}"
BOUNDARY_K="${BOUNDARY_K:-3}"
MARGIN_CORRECT="${MARGIN_CORRECT:--10}"
MARGIN_MISS="${MARGIN_MISS:--5}"
BARRIER_T="${BARRIER_T:-1.0}"
BARRIER_T_GROWTH="${BARRIER_T_GROWTH:-1.02}"
TRACK_LOSS_GRADIENTS="${TRACK_LOSS_GRADIENTS:-0}"

cd /home/guibo/links/projects/rrg-josedolz/guibo/ebo_seg

echo "SLURM_JOB_ID=${SLURM_JOB_ID:-local}"
echo "Dataset      : ${DATASET_LOWER}"
echo "Dataset root : ${DATASET_ROOT}"
echo "Output dir   : ${OUTPUT_DIR}"
echo "Loss         : ${LOSS}"
echo "Lambdas      : in=${LAMBDA_EBO_IN}, corr=${LAMBDA_EBO_CORR}"
echo "Bound lambdas: cen_in=${LAMBDA_EBO_CEN_IN}, out_in=${LAMBDA_EBO_OUT_IN}, cen_corr=${LAMBDA_EBO_CEN_CORR}, out_corr=${LAMBDA_EBO_OUT_CORR}"
echo "Boundary k   : ${BOUNDARY_K}"
echo "Margins      : correct=${MARGIN_CORRECT}, miss=${MARGIN_MISS}"
echo "Barrier t    : ${BARRIER_T} (growth=${BARRIER_T_GROWTH})"
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
  --lambda-ebo-in "${LAMBDA_EBO_IN}" \
  --lambda-ebo-corr "${LAMBDA_EBO_CORR}" \
  --lambda-ebo-cen-in "${LAMBDA_EBO_CEN_IN}" \
  --lambda-ebo-out-in "${LAMBDA_EBO_OUT_IN}" \
  --lambda-ebo-cen-corr "${LAMBDA_EBO_CEN_CORR}" \
  --lambda-ebo-out-corr "${LAMBDA_EBO_OUT_CORR}" \
  --boundary-k "${BOUNDARY_K}" \
  --margin-correct "${MARGIN_CORRECT}" \
  --margin-miss "${MARGIN_MISS}" \
  --barrier-t "${BARRIER_T}" \
  --barrier-t-growth "${BARRIER_T_GROWTH}" \
  "${TRACK_LOSS_GRADIENTS_ARGS[@]}"
