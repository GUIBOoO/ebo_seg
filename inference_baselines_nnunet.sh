#!/bin/bash
#SBATCH --time=2:00:00
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --output=/scratch/guibo/slurm_outputs/slurm-%j.out
#SBATCH --error=/scratch/guibo/slurm_outputs/slurm-%j.err
#SBATCH --account=rrg-josedolz
#SBATCH --partition=compute

# Evaluate the nnU-Net *baselines* (no grid search) on the held-out test split,
# including the RELU score when a D matrix is available.
#
# Companion to compute_d_matrix_nnunet.sh: run that first (on --split val), then
# this (on --split test). Estimating D and reporting on the same split biases the
# RELU metrics, so this script refuses that combination.

set -euo pipefail

module load python/3.10.13
source ~/nnunet/bin/activate

ROOT_DIR="${ROOT_DIR:-/home/guibo/links/projects/rrg-josedolz/guibo/ebo_seg}"
export nnUNet_extTrainer="$ROOT_DIR/nnunet_ext_trainers"
PYTHON_BIN="${PYTHON_BIN:-python}"

export nnUNet_raw="${nnUNet_raw:-/scratch/$USER/nnUNet_raw}"
export nnUNet_preprocessed="${nnUNet_preprocessed:-/scratch/$USER/nnUNet_preprocessed}"
NNUNET_RESULTS_BASE="${NNUNET_RESULTS_BASE:-/scratch/$USER/nnUNet_results_grid_search}"
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"

# DATASET: acdc, brats, or all (both, in sequence).
DATASET="${DATASET:-all}"
DATASET_LOWER=$(printf '%s' "$DATASET" | tr '[:upper:]' '[:lower:]')

case "$DATASET_LOWER" in
  acdc|brats) DATASETS=("$DATASET_LOWER") ;;
  all)        DATASETS=(acdc brats) ;;
  *)
    echo "ERROR: unsupported dataset '$DATASET'. Expected 'acdc', 'brats' or 'all'."
    exit 1
    ;;
esac

BASELINE_ROOT="${BASELINE_ROOT:-$NNUNET_RESULTS_BASE/baselines}"
TRAINER="${TRAINER:-CEDiceTrainer}"
PLANS="${PLANS:-nnUNetPlans}"
CONFIGURATION="${CONFIGURATION:-2d}"
FOLD="${FOLD:-0}"
CHECKPOINT_NAME="${CHECKPOINT_NAME:-checkpoint_best.pth}"

# Report on the held-out test set; D was estimated on val.
SPLIT="${SPLIT:-test}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/scratch/$USER/nnUNet_inference/baselines}"

# Where compute_d_matrix_nnunet.sh wrote its output. Set D_MATRIX to override the
# lookup with an explicit file, or D_MATRIX=none to skip the RELU score entirely.
D_MATRIX_ROOT="${D_MATRIX_ROOT:-/scratch/$USER/d_matrices/nnunet}"
D_MATRIX="${D_MATRIX:-}"
LAMBDA_WEIGHT="${LAMBDA_WEIGHT:-0.5}"

DEVICE="${DEVICE:-cuda}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TILE_STEP_SIZE="${TILE_STEP_SIZE:-0.5}"
PERFORM_EVERYTHING_ON_DEVICE="${PERFORM_EVERYTHING_ON_DEVICE:-0}"
DISABLE_MIRRORING="${DISABLE_MIRRORING:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"

dataset_id_for() {
    case "$1" in
      acdc)  echo 1 ;;
      brats) echo 2 ;;
    esac
}

resolve_dataset_dir() {
    # nnU-Net names datasets DatasetXXX_<Name>; find it under $1 by id.
    local root="$1" dataset_id="$2"
    local matches=("$root"/$(printf 'Dataset%03d' "$dataset_id")_*)
    if [ ! -d "${matches[0]}" ]; then
        echo "ERROR: no Dataset$(printf '%03d' "$dataset_id")_* directory under $root" >&2
        return 1
    fi
    printf '%s' "${matches[0]}"
}

# Reads the "split" field the D matrix was estimated on, so we can refuse to
# report on that same split.
d_matrix_split() {
    local npy="$1"
    local json="${npy%.*}.json"
    [ -f "$json" ] || { printf 'unknown'; return 0; }
    "$PYTHON_BIN" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("split","unknown"))' "$json"
}

cd "$ROOT_DIR"

echo "SLURM_JOB_ID=${SLURM_JOB_ID:-local}"
echo "Datasets       : ${DATASETS[*]}"
echo "Baseline root  : $BASELINE_ROOT"
echo "Trainer        : ${TRAINER}__${PLANS}__${CONFIGURATION} (fold $FOLD)"
echo "Split          : $SPLIT"
echo "Preprocessed   : $nnUNet_preprocessed"
echo "Output root    : $OUTPUT_ROOT"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo

failures=0
for ds in "${DATASETS[@]}"; do
    echo "=================== $ds ==================="
    dataset_id=$(dataset_id_for "$ds")

    dataset_dir=$(resolve_dataset_dir "$BASELINE_ROOT" "$dataset_id") || { failures=$((failures + 1)); continue; }
    dataset_name=$(basename "$dataset_dir")
    model_folder="$dataset_dir/${TRAINER}__${PLANS}__${CONFIGURATION}"

    if [ ! -d "$model_folder" ]; then
        echo "ERROR: baseline model folder not found: $model_folder"
        echo "Available under $dataset_dir:"
        find "$dataset_dir" -maxdepth 1 -mindepth 1 -type d 2>/dev/null || echo "  (nothing)"
        failures=$((failures + 1))
        continue
    fi
    if [ ! -f "$model_folder/fold_${FOLD}/$CHECKPOINT_NAME" ]; then
        echo "ERROR: checkpoint not found: $model_folder/fold_${FOLD}/$CHECKPOINT_NAME"
        failures=$((failures + 1))
        continue
    fi

    subdir="${TRAINER}__${PLANS}__${CONFIGURATION}"
    output_dir="$OUTPUT_ROOT/$ds/$subdir"
    metrics_file="$output_dir/metrics.json"

    if [ "$SKIP_EXISTING" = "1" ] && [ -f "$metrics_file" ]; then
        echo "[SKIP] existing metrics: $metrics_file"
        continue
    fi

    # Locate the D matrix unless the user pinned or disabled it.
    d_matrix_args=()
    if [ "$D_MATRIX" = "none" ]; then
        echo "D matrix       : disabled (RELU score off)"
    else
        if [ -n "$D_MATRIX" ]; then
            d_matrix="$D_MATRIX"
        else
            d_matrix="$D_MATRIX_ROOT/$ds/$subdir/d_matrix_nnunet_${dataset_name}_val_lambda_${LAMBDA_WEIGHT}.npy"
        fi

        if [ ! -f "$d_matrix" ]; then
            echo "ERROR: D matrix not found: $d_matrix"
            echo "Produce it with: DATASET=$ds SPLIT=val bash compute_d_matrix_nnunet.sh"
            echo "Or set D_MATRIX=none to evaluate without the RELU score."
            failures=$((failures + 1))
            continue
        fi

        estimated_on=$(d_matrix_split "$d_matrix")
        if [ "$estimated_on" = "$SPLIT" ]; then
            echo "ERROR: D was estimated on '$estimated_on', which is the split you are reporting on."
            echo "That biases the RELU metrics. Recompute D with SPLIT=val, or report on SPLIT=test."
            failures=$((failures + 1))
            continue
        fi
        echo "D matrix       : $d_matrix (estimated on '$estimated_on')"
        d_matrix_args+=(--d-matrix "$d_matrix")
    fi

    extra_args=()
    if [ "$PERFORM_EVERYTHING_ON_DEVICE" = "1" ]; then
        extra_args+=(--perform-everything-on-device)
    fi
    if [ "$DISABLE_MIRRORING" = "1" ]; then
        extra_args+=(--disable-mirroring)
    fi

    mkdir -p "$output_dir"
    echo "Model folder   : $model_folder"
    echo "Output         : $output_dir"

    "$PYTHON_BIN" inference_nnunet.py \
        --model-folder "$model_folder" \
        --fold "$FOLD" \
        --checkpoint-name "$CHECKPOINT_NAME" \
        --preprocessed-dir "$nnUNet_preprocessed" \
        --output-dir "$output_dir" \
        --split "$SPLIT" \
        --device "$DEVICE" \
        --temperature "$TEMPERATURE" \
        --tile-step-size "$TILE_STEP_SIZE" \
        "${d_matrix_args[@]}" \
        "${extra_args[@]}"

    echo "Wrote $metrics_file"
    echo
done

if [ "$failures" -gt 0 ]; then
    echo "Finished with $failures dataset(s) in error."
    exit 1
fi

echo "Finished baseline inference for: ${DATASETS[*]}"
