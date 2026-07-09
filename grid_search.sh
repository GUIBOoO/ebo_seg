#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --array=0-23
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
DATASET="${DATASET:-brats3}"
DATASET_LOWER=$(printf '%s' "$DATASET" | tr '[:upper:]' '[:lower:]')

# DATASET_ARG is what grid_search.py / train_unet.py accept; brats3 uses the same
# loader as brats, only with 3 channels and contiguous labels (0..3).
case "$DATASET_LOWER" in
  acdc)
    DATA_DIR="$SLURM_TMPDIR/dataset/ACDC/database"
    DATASET_ZIP="$SCRATCH/datasets/ACDC/ACDC.zip"
    UNZIP_TARGET="$SLURM_TMPDIR/dataset"
    DATASET_ARG=acdc
    DEFAULT_NUM_CLASSES=4
    DEFAULT_IN_CHANNELS=1
    ;;
  brats)
    DATA_DIR="$SCRATCH/datasets/Brats/data_slices"
    DATASET_ARG=brats
    DEFAULT_NUM_CLASSES=5
    DEFAULT_IN_CHANNELS=4
    ;;
  brats3)
    # Built by extract_data_brats3.py: T1ce/T2/FLAIR, labels remapped to 0..3.
    DATA_DIR="$SCRATCH/datasets/Brats/data_slices_3ch"
    DATASET_ARG=brats
    DEFAULT_NUM_CLASSES=4
    DEFAULT_IN_CHANNELS=3
    ;;
  *)
    echo "ERROR: unsupported dataset '$DATASET'. Expected 'acdc', 'brats' or 'brats3'."
    exit 1
    ;;
esac

if [ "$DATASET_LOWER" = "acdc" ]; then
    mkdir -p "$DATA_DIR"

    if [ ! -f "$DATASET_ZIP" ]; then
        echo "ERROR: dataset zip not found: $DATASET_ZIP"
        exit 1
    fi

    echo "Decompression ACDC images..."
    unzip -q "$DATASET_ZIP" -d "$UNZIP_TARGET"
else
    if [ ! -d "$DATA_DIR" ]; then
        echo "ERROR: prepared BraTS directory not found at $DATA_DIR"
        exit 1
    fi
fi

export PYTHON_DATA_DIR="$DATA_DIR"
PYTHON_SCRIPT="${PYTHON_SCRIPT:-grid_search.py}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL="${MODEL:-transunet}"
MODEL_LOWER=$(printf '%s' "$MODEL" | tr '[:upper:]' '[:lower:]')
LOSS="${LOSS:-bound_log_ebo}"
OUTPUT_DIR="${OUTPUT_DIR:-$SCRATCH/grid_search/$MODEL_LOWER/$DATASET_LOWER/$LOSS}"
EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-8}"
LR="${LR:-1e-3}"
NUM_WORKERS="${NUM_WORKERS:-8}"
IMAGE_SIZE="${IMAGE_SIZE:-256}"
NUM_CLASSES="${NUM_CLASSES:-$DEFAULT_NUM_CLASSES}"
IN_CHANNELS="${IN_CHANNELS:-$DEFAULT_IN_CHANNELS}"
# TransUNet's R50 stem is ImageNet-21k pretrained; other models ignore this.
if [ "$MODEL_LOWER" = "transunet" ]; then
    PRETRAINED_PATH="${PRETRAINED_PATH:-/home/guibo/links/scratch/pretrained/transunet/R50+ViT-B_16.npz}"
else
    PRETRAINED_PATH="${PRETRAINED_PATH:-}"
fi
SEED="${SEED:-42}"
DEVICE="${DEVICE:-cuda}"
METRIC="${METRIC:-fpr95}"
SELECTION_MODE="${SELECTION_MODE:-best}"
MODELS_PER_GPU="${MODELS_PER_GPU:-3}"
MAX_PARALLEL="${MAX_PARALLEL:-}"
GPU_IDS="${GPU_IDS:-}"

LAMBDA_EBO_IN_GRID="${LAMBDA_EBO_IN_GRID:-0.5 2 5}"
LAMBDA_EBO_CORR_GRID="${LAMBDA_EBO_CORR_GRID:-0.1}"
LAMBDA_EBO_CEN_IN_GRID="${LAMBDA_EBO_CEN_IN_GRID:-0.1 0.5 1}"
LAMBDA_EBO_OUT_IN_GRID="${LAMBDA_EBO_OUT_IN_GRID:-2 5}"
LAMBDA_EBO_CEN_CORR_GRID="${LAMBDA_EBO_CEN_CORR_GRID:-0.1 }"
LAMBDA_EBO_OUT_CORR_GRID="${LAMBDA_EBO_OUT_CORR_GRID:-0.1 0.5}"
BOUNDARY_K_GRID="${BOUNDARY_K_GRID:-3}"
MARGIN_CORRECT_GRID="${MARGIN_CORRECT_GRID:--17 -20}"
MARGIN_MISS_GRID="${MARGIN_MISS_GRID:--5}"
BARRIER_T_GRID="${BARRIER_T_GRID:-1.0}"
BARRIER_T_GROWTH_GRID="${BARRIER_T_GROWTH_GRID:-${BARRIER_T_GROWTH:-1.005}}"
RHO_GRID="${RHO_GRID:-1.0 0.1 0.5 4}"

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
read -r -a rho_values <<< "$RHO_GRID"

if [ "$MODEL_LOWER" = "transunet" ] && [ "$DATASET_LOWER" = "brats" ]; then
    echo "ERROR: TransUNet's pretrained stem takes 3 channels; raw BraTS has 4."
    echo "Build the 3-channel dataset with extract_data_brats3.py and use DATASET=brats3."
    exit 1
fi

pretrained_args=()
if [ -n "$PRETRAINED_PATH" ]; then
    if [ ! -f "$PRETRAINED_PATH" ]; then
        echo "ERROR: pretrained checkpoint not found: $PRETRAINED_PATH"
        exit 1
    fi
    pretrained_args+=(--pretrained-path "$PRETRAINED_PATH")
fi

gpu_args=(--models-per-gpu "$MODELS_PER_GPU")
if [ -n "$MAX_PARALLEL" ]; then
    gpu_args+=(--max-parallel "$MAX_PARALLEL")
fi
if [ -n "$GPU_IDS" ]; then
    gpu_args+=(--gpu-ids "$GPU_IDS")
fi

cd "$ROOT_DIR"

echo "SLURM_JOB_ID=${SLURM_JOB_ID:-local}"
echo "Dataset        : ${DATASET_LOWER} (--dataset ${DATASET_ARG})"
echo "Dataset root   : ${DATA_DIR}"
echo "Output dir     : ${OUTPUT_DIR}"
echo "Python script  : ${PYTHON_SCRIPT}"
echo "Model          : ${MODEL}"
echo "In channels    : ${IN_CHANNELS}"
echo "Num classes    : ${NUM_CLASSES}"
echo "Image size     : ${IMAGE_SIZE}"
echo "Pretrained     : ${PRETRAINED_PATH:-none}"
echo "Loss           : ${LOSS}"
echo "Models per GPU : ${MODELS_PER_GPU}"
if [ -n "$MAX_PARALLEL" ]; then
    echo "Max parallel   : ${MAX_PARALLEL}"
fi
if [ -n "$GPU_IDS" ]; then
    echo "GPU IDs        : ${GPU_IDS}"
fi

echo "Starting grid search..."

"$PYTHON_BIN" "$PYTHON_SCRIPT" \
  --dataset "$DATASET_ARG" \
  --dataset-root "$DATA_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --model "$MODEL" \
  --loss "$LOSS" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --lr "$LR" \
  --num-workers "$NUM_WORKERS" \
  --image-size "$IMAGE_SIZE" \
  --in-channels "$IN_CHANNELS" \
  --num-classes "$NUM_CLASSES" \
  --seed "$SEED" \
  --device "$DEVICE" \
  --metric "$METRIC" \
  --selection-mode "$SELECTION_MODE" \
  --python-bin "$PYTHON_BIN" \
  "${pretrained_args[@]}" \
  "${gpu_args[@]}" \
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
  --barrier-t-growth-grid "${barrier_t_growth_values[@]}" \
  --rho-grid "${rho_values[@]}"

echo "Grid search finished!"
