#!/bin/bash
#SBATCH --job-name=sens_lambda
#SBATCH --time=24:00:00
#SBATCH --array=0-99
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --output=/scratch/guibo/slurm_outputs/slurm-%A_%a.out
#SBATCH --error=/scratch/guibo/slurm_outputs/slurm-%A_%a.err
#SBATCH --account=rrg-josedolz
#SBATCH --partition=compute
#
# One-at-a-time sensitivity analysis of the four bound_ebo_ce lambdas on a UNet,
# on every dataset at once:
#   sbatch sensitivity_lambda.sh
#
# For each lambda, the three others stay frozen at their reference value and the
# tested one takes every value of SENS_VALUES: 4 lambdas x 5 values = 20 models
# per dataset, 5 datasets = 100 models, one per array task (--array=0-99):
#   task  0-19 -> acdc        task 60-79 -> chaos_mr
#   task 20-39 -> brats       task 80-99 -> refuge
#   task 40-59 -> cataracts   (order follows DATASETS, see below)
# Inside a dataset's block of 20:
#   +0-4   -> lambda_ebo_cen_in      +10-14 -> lambda_ebo_cen_corr
#   +5-9   -> lambda_ebo_out_in      +15-19 -> lambda_ebo_out_corr
# Each lambda gets its own grid_search.py output dir, so trial numbering, result
# shards and the dice/fpr95 selections stay per lambda:
#   $SCRATCH/grid_search/unet/<dataset>/sensitivity_lambda/<lambda>/trial_*
#
# Restricting to one dataset: DATASET=<name> runs only its 20 tasks, so submit
# with --array=0-19 (the sensitivity_lambda_<dataset>.sh wrappers do exactly that).
# Re-running one task of the full sweep: sbatch --array=<task id> sensitivity_lambda.sh

source ~/.bash_profile
module load python/3.11.5
source /home/guibo/ebo-seg/bin/activate

set -euo pipefail

ROOT_DIR="/home/guibo/links/projects/rrg-josedolz/guibo/ebo_seg"

# Dataset order fixes the task layout: block i of 20 tasks is DATASETS[i].
DATASETS="${DATASETS:-acdc brats cataracts chaos_mr refuge}"
if [ -n "${DATASET:-}" ]; then
    DATASETS="$DATASET"
fi
read -r -a dataset_list <<< "$DATASETS"

# Values tried for the lambda under test, in [0.1, 10].
SENS_VALUES="${SENS_VALUES:-0.1 0.5 1 5 10}"
LAMBDA_NAMES=(lambda_ebo_cen_in lambda_ebo_out_in lambda_ebo_cen_corr lambda_ebo_out_corr)
read -r -a sens_values <<< "$SENS_VALUES"
values_per_lambda="${#sens_values[@]}"
tasks_per_dataset=$(( ${#LAMBDA_NAMES[@]} * values_per_lambda ))
total_tasks=$(( ${#dataset_list[@]} * tasks_per_dataset ))

global_task_id="${SLURM_ARRAY_TASK_ID:-0}"
if [ "$global_task_id" -ge "$total_tasks" ]; then
    echo "Task $global_task_id outside the $total_tasks sensitivity runs. Nothing to do."
    exit 0
fi
DATASET="${dataset_list[$(( global_task_id / tasks_per_dataset ))]}"
task_id=$(( global_task_id % tasks_per_dataset ))
DATASET_LOWER=$(printf '%s' "$DATASET" | tr '[:upper:]' '[:lower:]')

# boundary_k frozen per dataset: dice_top_fpr95 winner of its ablation_k sweep.
# BOUNDARY_K, when set, overrides it for every dataset.
case "$DATASET_LOWER" in
  acdc)      DEFAULT_BOUNDARY_K=10 ;;
  brats)     DEFAULT_BOUNDARY_K=3 ;;
  cataracts) DEFAULT_BOUNDARY_K=1 ;;
  chaos_mr)  DEFAULT_BOUNDARY_K=3 ;;
  refuge)    DEFAULT_BOUNDARY_K=4 ;;
  *)         DEFAULT_BOUNDARY_K=3 ;;
esac

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
  refuge)
    DATA_DIR="$SCRATCH/datasets/refuge/data_slices"
    DATASET_ARG=refuge
    DEFAULT_NUM_CLASSES=3
    DEFAULT_IN_CHANNELS=3
    ;;
  cataracts)
    DATA_DIR="$SCRATCH/datasets/cataracts/data_slices"
    DATASET_ARG=cataracts
    DEFAULT_NUM_CLASSES=8
    DEFAULT_IN_CHANNELS=3
    ;;
  chaos_mr)
    DATA_DIR="$SCRATCH/datasets/chaos/data_slices_mr"
    DATASET_ARG=chaos
    DEFAULT_NUM_CLASSES=5
    DEFAULT_IN_CHANNELS=1
    ;;
  *)
    echo "ERROR: unsupported dataset '$DATASET'. Expected 'acdc', 'brats',"
    echo "       'refuge', 'cataracts' or 'chaos_mr'."
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
elif [ ! -d "$DATA_DIR" ]; then
    echo "ERROR: prepared dataset directory not found at $DATA_DIR"
    exit 1
fi

export PYTHON_DATA_DIR="$DATA_DIR"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL=unet
LOSS="${LOSS:-bound_ebo_ce}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$SCRATCH/grid_search/unet/$DATASET_LOWER/sensitivity_lambda}"
EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-8}"
LR="${LR:-1e-3}"
NUM_WORKERS="${NUM_WORKERS:-8}"
IMAGE_SIZE="${IMAGE_SIZE:-256}"
NUM_CLASSES="${NUM_CLASSES:-$DEFAULT_NUM_CLASSES}"
IN_CHANNELS="${IN_CHANNELS:-$DEFAULT_IN_CHANNELS}"
SEED="${SEED:-42}"
DEVICE="${DEVICE:-cuda}"
METRIC="${METRIC:-fpr95}"
SELECTION_MODE="${SELECTION_MODE:-best}"
TOP_FRAC="${TOP_FRAC:-0.2}"

# Reference (frozen) values, the ones every previous grid search used.
REF_CEN_IN="${REF_CEN_IN:-0.5}"
REF_OUT_IN="${REF_OUT_IN:-5}"
REF_CEN_CORR="${REF_CEN_CORR:-0.1}"
REF_OUT_CORR="${REF_OUT_CORR:-0.1}"
BOUNDARY_K="${BOUNDARY_K:-$DEFAULT_BOUNDARY_K}"
MARGIN_CORRECT="${MARGIN_CORRECT:--10}"
MARGIN_MISS="${MARGIN_MISS:--5}"

lambda_idx=$(( task_id / values_per_lambda ))
value_idx=$(( task_id % values_per_lambda ))
lambda_name="${LAMBDA_NAMES[$lambda_idx]}"

cen_in_grid="$REF_CEN_IN"
out_in_grid="$REF_OUT_IN"
cen_corr_grid="$REF_CEN_CORR"
out_corr_grid="$REF_OUT_CORR"
case "$lambda_name" in
  lambda_ebo_cen_in)   cen_in_grid="$SENS_VALUES" ;;
  lambda_ebo_out_in)   out_in_grid="$SENS_VALUES" ;;
  lambda_ebo_cen_corr) cen_corr_grid="$SENS_VALUES" ;;
  lambda_ebo_out_corr) out_corr_grid="$SENS_VALUES" ;;
esac
read -r -a cen_in_values <<< "$cen_in_grid"
read -r -a out_in_values <<< "$out_in_grid"
read -r -a cen_corr_values <<< "$cen_corr_grid"
read -r -a out_corr_values <<< "$out_corr_grid"

OUTPUT_DIR="$OUTPUT_ROOT/$lambda_name"
mkdir -p "$OUTPUT_DIR"

cd "$ROOT_DIR"

echo "SLURM_JOB_ID=${SLURM_JOB_ID:-local} task=$global_task_id (dataset task $task_id)"
echo "Dataset        : ${DATASET_LOWER} (--dataset ${DATASET_ARG})"
echo "Dataset root   : ${DATA_DIR}"
echo "Tested lambda  : ${lambda_name} = ${sens_values[$value_idx]} (value $((value_idx + 1))/$values_per_lambda)"
echo "Frozen         : cen_in=${REF_CEN_IN} out_in=${REF_OUT_IN} cen_corr=${REF_CEN_CORR} out_corr=${REF_OUT_CORR}"
echo "               : boundary_k=${BOUNDARY_K} margin_correct=${MARGIN_CORRECT} margin_miss=${MARGIN_MISS}"
echo "Output dir     : ${OUTPUT_DIR}"

# grid_search.py shards its grid with SLURM_ARRAY_TASK_ID/COUNT. Each lambda's
# grid holds exactly values_per_lambda configs, so re-index the task inside that
# lambda: shard value_idx of values_per_lambda trains exactly one model.
SLURM_ARRAY_TASK_ID="$value_idx" SLURM_ARRAY_TASK_COUNT="$values_per_lambda" \
"$PYTHON_BIN" grid_search.py \
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
  --top-frac "$TOP_FRAC" \
  --python-bin "$PYTHON_BIN" \
  --models-per-gpu 1 \
  --lambda-ebo-cen-in-grid "${cen_in_values[@]}" \
  --lambda-ebo-out-in-grid "${out_in_values[@]}" \
  --lambda-ebo-cen-corr-grid "${cen_corr_values[@]}" \
  --lambda-ebo-out-corr-grid "${out_corr_values[@]}" \
  --boundary-k-grid "$BOUNDARY_K" \
  --margin-correct-grid "$MARGIN_CORRECT" \
  --margin-miss-grid "$MARGIN_MISS"

echo "Sensitivity run finished: ${lambda_name} = ${sens_values[$value_idx]}"
