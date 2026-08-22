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
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"

# DATASET picks the grid-search results and the default output directory.
#   acdc -> Dataset001_ACDC, brats -> Dataset002_BraTS2D,
#   refuge -> Dataset004_REFUGE, cataracts -> Dataset005_CaDIS,
#   chaos_mr -> Dataset006_CHAOSMR
DATASET="${DATASET:-refuge}"
DATASET_LOWER=$(printf '%s' "$DATASET" | tr '[:upper:]' '[:lower:]')
case "$DATASET_LOWER" in
  acdc|brats|refuge|cataracts|chaos_mr) ;;
  *)
    echo "ERROR: unsupported dataset '$DATASET'. Expected 'acdc', 'brats', 'refuge',"
    echo "       'cataracts' or 'chaos_mr'."
    exit 1
    ;;
esac

NNUNET_RESULTS_BASE="${NNUNET_RESULTS_BASE:-/scratch/$USER/nnUNet_results_grid_search/boundebo/$DATASET_LOWER}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/scratch/$USER/nnUNet_inference_grid_search/boundebo/$DATASET_LOWER}"
DEVICE="${DEVICE:-cuda}"
TILE_STEP_SIZE="${TILE_STEP_SIZE:-0.5}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
TRIAL_FILTER="${TRIAL_FILTER:-}"
PERFORM_EVERYTHING_ON_DEVICE="${PERFORM_EVERYTHING_ON_DEVICE:-0}"
DISABLE_MIRRORING="${DISABLE_MIRRORING:-0}"

# Three independent ways of picking the winner of the grid search, all written by
# select_nnunet_grid_best.py, identical to the U-Net side (slurm_test_grid_search.sh):
#   dice           - best validation Dice.
#   dice_top_fpr95 - best (lowest) energy FPR95 among the top TOP_FRAC by Dice.
#   fpr95          - best (lowest) energy FPR95 over every config, no Dice filter.
# SELECTION=both runs dice + dice_top_fpr95, SELECTION=all runs the three.
SELECTION="${SELECTION:-all}"
TOP_FRAC="${TOP_FRAC:-0.2}"
# Rebuild the selection JSONs from the trial dirs when they are missing.
AUTO_SELECT="${AUTO_SELECT:-1}"
SELECTION_MODE="${SELECTION_MODE:-best}"
# Passed to select_nnunet_grid_best.py: drops trials with a shorter history
# (still running / crashed). Empty means rank every trial that has a checkpoint.
EXPECTED_EPOCHS="${EXPECTED_EPOCHS:-}"
FOLD="${FOLD:-}"
LOSS="${LOSS:-}"

if [ "$SELECTION_MODE" = "last" ]; then
    CHECKPOINT_NAME="${CHECKPOINT_NAME:-checkpoint_final.pth}"
else
    CHECKPOINT_NAME="${CHECKPOINT_NAME:-checkpoint_best.pth}"
fi

case "$SELECTION" in
  all)   criteria=(dice dice_top_fpr95 fpr95) ;;
  both)  criteria=(dice dice_top_fpr95) ;;
  dice)  criteria=(dice) ;;
  dice_top_fpr95) criteria=(dice_top_fpr95) ;;
  fpr95) criteria=(fpr95) ;;
  *)
    echo "ERROR: unsupported SELECTION '$SELECTION'. Expected 'dice', 'dice_top_fpr95',"
    echo "       'fpr95', 'both' (dice + dice_top_fpr95) or 'all'."
    exit 1
    ;;
esac

# BEST_CONFIG_JSON only makes sense when a single criterion is requested.
BEST_CONFIG_JSON="${BEST_CONFIG_JSON:-}"
if [ -n "$BEST_CONFIG_JSON" ] && [ "${#criteria[@]}" -ne 1 ]; then
    echo "ERROR: BEST_CONFIG_JSON needs a single criterion. Set SELECTION=dice,"
    echo "       SELECTION=dice_top_fpr95 or SELECTION=fpr95."
    exit 1
fi

# The D matrix is fitted on one checkpoint, so each selection needs its own.
# D_MATRIX stays as a fallback for when several selections land on the same trial.
D_MATRIX="${D_MATRIX:-}"
D_MATRIX_DICE="${D_MATRIX_DICE:-$D_MATRIX}"
D_MATRIX_DICE_TOP_FPR95="${D_MATRIX_DICE_TOP_FPR95:-$D_MATRIX}"
D_MATRIX_FPR95="${D_MATRIX_FPR95:-$D_MATRIX}"

mkdir -p "$OUTPUT_ROOT"
cd "$ROOT_DIR"

echo "SLURM_JOB_ID=${SLURM_JOB_ID:-local}"
echo "Dataset        : $DATASET_LOWER"
echo "Results base   : $NNUNET_RESULTS_BASE"
echo "Output root    : $OUTPUT_ROOT"
echo "Checkpoint name: $CHECKPOINT_NAME"
echo "Selections     : ${criteria[*]} (top-frac $TOP_FRAC, mode $SELECTION_MODE)"
echo "Preprocessed   : $nnUNet_preprocessed"
echo "Trial filter   : ${TRIAL_FILTER:-<none>}"
echo "Device         : $DEVICE"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

# Rebuild every selection JSON when at least one is missing.
need_selection=0
for criterion in "${criteria[@]}"; do
    if [ -z "$BEST_CONFIG_JSON" ] && [ ! -f "$NNUNET_RESULTS_BASE/nnunet_grid_search_best_${criterion}.json" ]; then
        need_selection=1
    fi
done

if [ "$need_selection" = "1" ]; then
    if [ "$AUTO_SELECT" != "1" ]; then
        echo "ERROR: selection JSON missing under $NNUNET_RESULTS_BASE."
        echo "Run: SELECT_BEST=1 NNUNET_RESULTS_BASE=$NNUNET_RESULTS_BASE bash nnunet_grid_search.sh"
        exit 1
    fi
    echo
    echo "Selection JSON missing: running select_nnunet_grid_best.py on $NNUNET_RESULTS_BASE"
    select_args=()
    if [ -n "$EXPECTED_EPOCHS" ]; then
        select_args+=(--expected-epochs "$EXPECTED_EPOCHS")
    fi
    if [ -n "$FOLD" ]; then
        select_args+=(--fold "$FOLD")
    fi
    if [ -n "$LOSS" ]; then
        select_args+=(--loss "$LOSS")
    fi
    "$PYTHON_BIN" select_nnunet_grid_best.py \
        --results-base "$NNUNET_RESULTS_BASE" \
        --selection-mode "$SELECTION_MODE" \
        --top-frac "$TOP_FRAC" \
        "${select_args[@]}"
fi

read_json_field() {
    "$PYTHON_BIN" -c 'import json, sys; from pathlib import Path; data=json.loads(Path(sys.argv[1]).read_text()); trial=data.get("best_trial", data); print(trial.get(sys.argv[2], "") or "")' "$1" "$2"
}

if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi
fi

# Several selections can land on the same trial; inference then runs once and the
# other criteria only record a pointer to the shared results.
declare -A ran_trials=()
executed=()

for criterion in "${criteria[@]}"; do
    config_json="${BEST_CONFIG_JSON:-$NNUNET_RESULTS_BASE/nnunet_grid_search_best_${criterion}.json}"

    echo
    echo "============================================================"
    echo "Selection: $criterion"
    echo "Config   : $config_json"

    if [ ! -f "$config_json" ]; then
        echo "ERROR: best config JSON not found: $config_json"
        exit 1
    fi

    best_trial_name=$(read_json_field "$config_json" trial_name)
    best_checkpoint=$(read_json_field "$config_json" checkpoint)

    if [ -z "$best_trial_name" ]; then
        if [ -n "$best_checkpoint" ]; then
            best_trial_name=$("$PYTHON_BIN" -c 'import sys; from pathlib import Path; parts=Path(sys.argv[1]).parts; print(next((p for p in parts if p.startswith("trial_")), "best_trial"))' "$best_checkpoint")
        else
            best_trial_name="best_trial"
        fi
    fi

    echo "Trial    : $best_trial_name"

    if [ -n "$TRIAL_FILTER" ] && [[ "$best_trial_name" != *"$TRIAL_FILTER"* ]]; then
        echo "SKIP: best trial '$best_trial_name' does not match TRIAL_FILTER='$TRIAL_FILTER'"
        continue
    fi

    case "$criterion" in
      dice)           criterion_d_matrix="$D_MATRIX_DICE" ;;
      dice_top_fpr95) criterion_d_matrix="$D_MATRIX_DICE_TOP_FPR95" ;;
      fpr95)          criterion_d_matrix="$D_MATRIX_FPR95" ;;
    esac

    extra_args=()
    if [ "$PERFORM_EVERYTHING_ON_DEVICE" = "1" ]; then
        extra_args+=(--perform-everything-on-device)
    fi
    if [ "$DISABLE_MIRRORING" = "1" ]; then
        extra_args+=(--disable-mirroring)
    fi
    if [ -n "$criterion_d_matrix" ]; then
        if [ ! -f "$criterion_d_matrix" ]; then
            echo "ERROR: D matrix not found: $criterion_d_matrix"
            echo "Produce it for this trial's checkpoint with compute_d_matrix_nnunet.sh (SPLIT=val)."
            exit 1
        fi
        extra_args+=(--d-matrix "$criterion_d_matrix")
    fi

    output_dir="$OUTPUT_ROOT/$criterion/$best_trial_name"
    mkdir -p "$output_dir"
    cp "$config_json" "$output_dir/selection.json"

    if [ -n "${ran_trials[$best_trial_name]:-}" ]; then
        echo "[REUSE] $best_trial_name already tested for selection '${ran_trials[$best_trial_name]}'"
        echo "        results: $OUTPUT_ROOT/${ran_trials[$best_trial_name]}/$best_trial_name"
        executed+=("$criterion -> $best_trial_name (same model as ${ran_trials[$best_trial_name]})")
        continue
    fi

    metrics_file="$output_dir/metrics.json"
    if [ "$SKIP_EXISTING" = "1" ] && [ -f "$metrics_file" ]; then
        echo "[SKIP] $best_trial_name: existing metrics found at $metrics_file"
        ran_trials[$best_trial_name]="$criterion"
        executed+=("$criterion -> $best_trial_name (skipped, already tested)")
        continue
    fi

    echo "Checkpoint: ${best_checkpoint:-from JSON/model folder}"
    echo "Output    : $output_dir"
    echo "D matrix  : ${criterion_d_matrix:-none (RELU score disabled)}"
    echo "============================================================"

    "$PYTHON_BIN" inference_nnunet.py \
        --best-json "$config_json" \
        --checkpoint-name "$CHECKPOINT_NAME" \
        --preprocessed-dir "$nnUNet_preprocessed" \
        --output-dir "$output_dir" \
        --device "$DEVICE" \
        --tile-step-size "$TILE_STEP_SIZE" \
        "${extra_args[@]}"

    ran_trials[$best_trial_name]="$criterion"
    executed+=("$criterion -> $best_trial_name")
done

echo
echo "Finished nnU-Net best-grid inference."
for line in "${executed[@]}"; do
    echo "  $line"
done
