#!/bin/bash
#SBATCH --time=1:00:00
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --output=/scratch/guibo/slurm_outputs/slurm-%j.out
#SBATCH --error=/scratch/guibo/slurm_outputs/slurm-%j.err
#SBATCH --account=rrg-josedolz
#SBATCH --partition=compute

set -euo pipefail

module load python/3.10.13
source ~/nnunet/bin/activate

ROOT_DIR="${ROOT_DIR:-/home/guibo/links/projects/rrg-josedolz/guibo/ebo_seg}"
export nnUNet_extTrainer="$ROOT_DIR/nnunet_ext_trainers"
PYTHON_BIN="${PYTHON_BIN:-python}"

export nnUNet_raw="${nnUNet_raw:-/scratch/$USER/nnUNet_raw}"
export nnUNet_preprocessed="${nnUNet_preprocessed:-/scratch/$USER/nnUNet_preprocessed}"
NNUNET_RESULTS_BASE="${NNUNET_RESULTS_BASE:-/scratch/$USER/nnUNet_results}"
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"

# DATASET selects the nnU-Net dataset directory (and the default output dir).
DATASET="${DATASET:-chaos_mr}"
DATASET_LOWER=$(printf '%s' "$DATASET" | tr '[:upper:]' '[:lower:]')

case "$DATASET_LOWER" in
  acdc)      DEFAULT_DATASET_ID=1 ;;
  brats)     DEFAULT_DATASET_ID=2 ;;
  refuge)    DEFAULT_DATASET_ID=4 ;;
  cataracts) DEFAULT_DATASET_ID=5 ;;
  chaos_mr)  DEFAULT_DATASET_ID=6 ;;
  *)
    echo "ERROR: unsupported dataset '$DATASET'. Expected 'acdc', 'brats', 'refuge',"
    echo "       'cataracts' or 'chaos_mr'."
    exit 1
    ;;
esac
DATASET_ID="${DATASET_ID:-$DEFAULT_DATASET_ID}"

# MODE picks how the trained model is located:
#   baseline (default) - a plain nnUNetv2_train run, e.g. the CEDiceTrainer baseline
#   best               - the winner of the grid search, via nnunet_grid_search_best.json
#   folder             - whatever MODEL_FOLDER points at
MODE="${MODE:-baseline}"

# Baselines from train_nnunet.sh land in $nnUNet_results; the CEDice baselines
# trained through the grid-search harness land in $NNUNET_RESULTS_BASE/baselines.
BASELINE_ROOT="${BASELINE_ROOT:-$NNUNET_RESULTS_BASE/baselines}"
TRAINER="${TRAINER:-CEDiceTrainer}"
PLANS="${PLANS:-nnUNetPlans}"
CONFIGURATION="${CONFIGURATION:-2d}"

# MODE=best picks one of the three selections written by select_nnunet_grid_best.py
# (dice, dice_top_fpr95, fpr95), the same criteria as the U-Net pipeline. The D
# matrix is fitted on a single checkpoint, so one run is needed per criterion.
CRITERION="${CRITERION:-dice}"
case "$CRITERION" in
  dice|dice_top_fpr95|fpr95) ;;
  *)
    echo "ERROR: unsupported CRITERION '$CRITERION'. Expected 'dice', 'dice_top_fpr95' or 'fpr95'."
    exit 1
    ;;
esac
BEST_CONFIG_JSON="${BEST_CONFIG_JSON:-$NNUNET_RESULTS_BASE/nnunet_grid_search_best_${CRITERION}.json}"
MODEL_FOLDER="${MODEL_FOLDER:-}"
FOLD="${FOLD:-}"
CHECKPOINT_NAME="${CHECKPOINT_NAME:-checkpoint_best.pth}"
# D must be estimated on a split you do NOT report on: val, not test.
SPLIT="${SPLIT:-val}"
DEVICE="${DEVICE:-cuda}"
LAMBDA_WEIGHT="${LAMBDA_WEIGHT:-0.5}"
TILE_STEP_SIZE="${TILE_STEP_SIZE:-0.5}"
PERFORM_EVERYTHING_ON_DEVICE="${PERFORM_EVERYTHING_ON_DEVICE:-0}"
DISABLE_MIRRORING="${DISABLE_MIRRORING:-0}"

resolve_dataset_dir() {
    # nnU-Net names datasets DatasetXXX_<Name>; find it under $1 by id.
    local root="$1"
    local matches=("$root"/$(printf 'Dataset%03d' "$DATASET_ID")_*)
    if [ ! -d "${matches[0]}" ]; then
        echo "ERROR: no Dataset$(printf '%03d' "$DATASET_ID")_* directory under $root" >&2
        return 1
    fi
    printf '%s' "${matches[0]}"
}

model_args=()
case "$MODE" in
  folder)
    if [ -z "$MODEL_FOLDER" ]; then
        echo "ERROR: MODE=folder requires MODEL_FOLDER=/path/to/Trainer__Plans__config"
        exit 1
    fi
    ;;
  baseline)
    if [ -z "$MODEL_FOLDER" ]; then
        dataset_dir=$(resolve_dataset_dir "$BASELINE_ROOT") || exit 1
        MODEL_FOLDER="$dataset_dir/${TRAINER}__${PLANS}__${CONFIGURATION}"

        # nnU-Net names the folder after the trainer class, never after the
        # loss: a CE+Dice baseline trained through train_nnunet.sh keeps that
        # script's TRAINER default (BoundEBOLogBarrierTrainer with
        # NNUNET_EBO_LOSS=ce_dice). When the requested trainer is absent but the
        # dataset has exactly one model folder, take it instead of failing.
        if [ ! -d "$MODEL_FOLDER" ]; then
            candidates=("$dataset_dir"/*__"${PLANS}"__"${CONFIGURATION}")
            if [ "${#candidates[@]}" -eq 1 ] && [ -d "${candidates[0]}" ]; then
                echo "NOTE: ${TRAINER}__${PLANS}__${CONFIGURATION} not found; using the only"
                echo "      model folder of this dataset: $(basename "${candidates[0]}")"
                MODEL_FOLDER="${candidates[0]}"
            fi
        fi
    fi
    if [ ! -d "$MODEL_FOLDER" ]; then
        echo "ERROR: baseline model folder not found: $MODEL_FOLDER"
        echo "Available under $BASELINE_ROOT:"
        find "$BASELINE_ROOT" -maxdepth 2 -mindepth 2 -type d 2>/dev/null || echo "  (nothing)"
        echo "Set BASELINE_ROOT / TRAINER / PLANS / CONFIGURATION, or MODE=folder MODEL_FOLDER=..."
        exit 1
    fi
    FOLD="${FOLD:-0}"
    ;;
  best)
    if [ -z "$MODEL_FOLDER" ]; then
        if [ ! -f "$BEST_CONFIG_JSON" ]; then
            echo "ERROR: best config JSON not found: $BEST_CONFIG_JSON"
            echo "Produce it with: SELECT_BEST=1 NNUNET_RESULTS_BASE=$NNUNET_RESULTS_BASE bash nnunet_grid_search.sh"
            echo "(or pick another criterion with CRITERION=dice|dice_top_fpr95|fpr95)"
            exit 1
        fi
        model_args+=(--best-json "$BEST_CONFIG_JSON")
    fi
    ;;
  *)
    echo "ERROR: unsupported MODE '$MODE'. Expected 'baseline', 'best' or 'folder'."
    exit 1
    ;;
esac

if [ -n "$MODEL_FOLDER" ]; then
    model_args+=(--model-folder "$MODEL_FOLDER")
fi

# Keep baseline and grid-best D matrices apart: d_matrix_nnunet.py names its
# output from the dataset and split only, so a shared dir would overwrite.
if [ "$MODE" = "best" ]; then
    # One D matrix per criterion: the three selections can land on different trials.
    DEFAULT_OUTPUT_SUBDIR="grid_best_$CRITERION"
else
    DEFAULT_OUTPUT_SUBDIR="$(basename "$MODEL_FOLDER")"
fi
OUTPUT_DIR="${OUTPUT_DIR:-/scratch/$USER/d_matrices/nnunet/$DATASET_LOWER/$DEFAULT_OUTPUT_SUBDIR}"

if [ -n "$FOLD" ]; then
    model_args+=(--fold "$FOLD")
fi
if [ "$PERFORM_EVERYTHING_ON_DEVICE" = "1" ]; then
    model_args+=(--perform-everything-on-device)
fi
if [ "$DISABLE_MIRRORING" = "1" ]; then
    model_args+=(--disable-mirroring)
fi

mkdir -p "$OUTPUT_DIR"
cd "$ROOT_DIR"

echo "SLURM_JOB_ID=${SLURM_JOB_ID:-local}"
echo "Dataset        : $DATASET_LOWER (Dataset$(printf '%03d' "$DATASET_ID"))"
if [ "$MODE" = "best" ]; then
    echo "Mode           : $MODE (criterion $CRITERION)"
else
    echo "Mode           : $MODE"
fi
echo "Model source   : ${MODEL_FOLDER:-$BEST_CONFIG_JSON}"
echo "Fold           : ${FOLD:-from best-json (default 0)}"
echo "Checkpoint     : $CHECKPOINT_NAME"
echo "Preprocessed   : $nnUNet_preprocessed"
echo "Split          : $SPLIT"
echo "Lambda weight  : $LAMBDA_WEIGHT"
echo "Output         : $OUTPUT_DIR"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

"$PYTHON_BIN" d_matrix_nnunet.py \
    "${model_args[@]}" \
    --checkpoint-name "$CHECKPOINT_NAME" \
    --preprocessed-dir "$nnUNet_preprocessed" \
    --output-dir "$OUTPUT_DIR" \
    --split "$SPLIT" \
    --device "$DEVICE" \
    --tile-step-size "$TILE_STEP_SIZE" \
    --lambda-weight "$LAMBDA_WEIGHT"

echo
echo "Finished. Pass the resulting .npy to inference_nnunet.sh via D_MATRIX=..."
