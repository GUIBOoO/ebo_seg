#!/bin/bash
#SBATCH --time=10:00:00
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

ROOT_DIR="/home/guibo/links/projects/rrg-josedolz/guibo/ebo_seg"
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
    DATA_DIR="$SLURM_TMPDIR/dataset"
    DATASET_ZIP="$SCRATCH/datasets/Brats/brats_train.zip"
    UNZIP_TARGET="$DATA_DIR"
    DEFAULT_NUM_CLASSES=3
    ;;
  *)
    echo "ERROR: unsupported dataset '$DATASET'. Expected 'acdc' or 'brats'."
    exit 1
    ;;
esac

mkdir -p "$DATA_DIR"

if [ ! -f "$DATASET_ZIP" ]; then
    echo "ERROR: dataset zip not found: $DATASET_ZIP"
    exit 1
fi

if [ "$DATASET_LOWER" = "acdc" ]; then
    echo "Decompression ACDC images..."
else
    echo "Decompression BraTS train..."
fi
unzip -q "$DATASET_ZIP" -d "$UNZIP_TARGET"

export PYTHON_DATA_DIR="$DATA_DIR"
OUTPUT_DIR="${OUTPUT_DIR:-$SCRATCH/grid_search}"
PYTHON_SCRIPT="${PYTHON_SCRIPT:-grid_search.py}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL="${MODEL:-unet}"
LOSS="${LOSS:-bound_log_ebo}"
EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-8}"
LR="${LR:-1e-3}"
NUM_WORKERS="${NUM_WORKERS:-8}"
IMAGE_SIZE="${IMAGE_SIZE:-256}"
NUM_CLASSES="${NUM_CLASSES:-$DEFAULT_NUM_CLASSES}"
SEED="${SEED:-42}"
DEVICE="${DEVICE:-cuda}"
METRIC="${METRIC:-fpr95}"
SELECTION_MODE="${SELECTION_MODE:-best}"

LAMBDA_EBO_IN_GRID="${LAMBDA_EBO_IN_GRID:-2 5}"
LAMBDA_EBO_CORR_GRID="${LAMBDA_EBO_CORR_GRID:-0.6 2}"
LAMBDA_EBO_CEN_IN_GRID="${LAMBDA_EBO_CEN_IN_GRID:-2 0.5}"
LAMBDA_EBO_OUT_IN_GRID="${LAMBDA_EBO_OUT_IN_GRID:-5 7}"
LAMBDA_EBO_CEN_CORR_GRID="${LAMBDA_EBO_CEN_CORR_GRID:-0.1}"
LAMBDA_EBO_OUT_CORR_GRID="${LAMBDA_EBO_OUT_CORR_GRID:-2 0.5}"
BOUNDARY_K_GRID="${BOUNDARY_K_GRID:-3}"
MARGIN_CORRECT_GRID="${MARGIN_CORRECT_GRID:--5}"
MARGIN_MISS_GRID="${MARGIN_MISS_GRID:--5}"
BARRIER_T_GRID="${BARRIER_T_GRID:-1.0}"
BARRIER_T_GROWTH_GRID="${BARRIER_T_GROWTH_GRID:-${BARRIER_T_GROWTH:-1.005 1.01}}"

read -r -a lambda_ebo_in_values <<< "$LAMBDA_EBO_IN_GRID"
read -r -a lambda_ebo_corr_values <<< "$LAMBDA_EBO_CORR_GRID"
read -r -a lambda_ebo_cen_in_values <<< "$LAMBDA_EBO_CEN_IN_GRID"
read -r -a lambda_ebo_out_in_values <<< "$LAMBDA_EBO_OUT_IN_GRID"
read -r -a lambda_ebo_cen_corr_values <<< "$LAMBDA_EBO_CEN_CORR_GRID"
read -r -a lambda_ebo_out_corr_values <<< "$LAMBDA_EBO_OUT_CORR_GRID"
read -r -a boundary_k_values <<< "$BOUNDARY_K_GRID"
read -r -a margin_correct_values <<< "$MARGIN_CORRECT_GRID"
read -r -a margin_miss_values <<< "$MARGIN_MISS_GRID"
read -r -a barrier_t_values <<< "$BARRIER_T_GRID"
read -r -a barrier_t_growth_values <<< "$BARRIER_T_GROWTH_GRID"
cd "$ROOT_DIR"

echo "SLURM_JOB_ID=${SLURM_JOB_ID:-local}"
echo "Dataset        : ${DATASET_LOWER}"
echo "Dataset root   : ${DATA_DIR}"
echo "Output dir     : ${OUTPUT_DIR}"
echo "Python script  : ${PYTHON_SCRIPT}"
echo "Loss           : ${LOSS}"

echo "Starting grid search..."

"$PYTHON_BIN" "$PYTHON_SCRIPT" \
  --dataset "$DATASET_LOWER" \
  --dataset-root "$DATA_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --model "$MODEL" \
  --loss "$LOSS" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --lr "$LR" \
  --num-workers "$NUM_WORKERS" \
  --image-size "$IMAGE_SIZE" \
  --num-classes "$NUM_CLASSES" \
  --seed "$SEED" \
  --device "$DEVICE" \
  --metric "$METRIC" \
  --selection-mode "$SELECTION_MODE" \
  --python-bin "$PYTHON_BIN" \
  --lambda-ebo-in-grid "${lambda_ebo_in_values[@]}" \
  --lambda-ebo-corr-grid "${lambda_ebo_corr_values[@]}" \
  --lambda-ebo-cen-in-grid "${lambda_ebo_cen_in_values[@]}" \
  --lambda-ebo-out-in-grid "${lambda_ebo_out_in_values[@]}" \
  --lambda-ebo-cen-corr-grid "${lambda_ebo_cen_corr_values[@]}" \
  --lambda-ebo-out-corr-grid "${lambda_ebo_out_corr_values[@]}" \
  --boundary-k-grid "${boundary_k_values[@]}" \
  --margin-correct-grid "${margin_correct_values[@]}" \
  --margin-miss-grid "${margin_miss_values[@]}" \
  --barrier-t-grid "${barrier_t_values[@]}" \
  --barrier-t-growth-grid "${barrier_t_growth_values[@]}"

echo "Grid search finished!"
