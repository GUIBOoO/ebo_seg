#!/bin/bash
#SBATCH --time=01:00:00
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

ROOT_DIR="${ROOT_DIR:-/home/guibo/links/projects/rrg-josedolz/guibo/ebo_seg}"
DATASET="${DATASET:-brats}"
DATASET_LOWER=$(printf '%s' "$DATASET" | tr '[:upper:]' '[:lower:]')

# DATASET_ARG is what inference.py's --dataset accepts; brats3 is the same
# loader as brats, only with 3 channels and contiguous labels 0..3.
case "$DATASET_LOWER" in
  acdc)
    DATA_DIR="$SLURM_TMPDIR/dataset/ACDC/database"
    DATASET_ZIP="$SCRATCH/datasets/ACDC/ACDC.zip"
    UNZIP_TARGET="$SLURM_TMPDIR/dataset"
    DATASET_ARG=acdc
    ;;
  brats)
    DATA_DIR="$SCRATCH/datasets/Brats/data_slices"
    DATASET_ARG=brats
    ;;
  brats3)
    # Built by extract_data_brats3.py. The only BraTS variant TransUNet accepts.
    DATA_DIR="$SCRATCH/datasets/Brats/data_slices_3ch"
    DATASET_ARG=brats
    ;;
  refuge)
    # Built by extract_data_refuge.py: RGB fundus, 0 bg / 1 disc / 2 cup.
    DATA_DIR="$SCRATCH/datasets/refuge/data_slices"
    DATASET_ARG=refuge
    ;;
  cataracts)
    # Built by extract_data_cataracts.py, CaDIS Experiment I (8 classes).
    # Frames are 512x288: set IMAGE_SIZE only for TransUNet, which needs square input.
    DATA_DIR="$SCRATCH/datasets/cataracts/data_slices"
    DATASET_ARG=cataracts
    ;;
  chaos_ct)
    # Built by extract_data_chaos_unet.py --modality ct: liver only.
    DATA_DIR="$SCRATCH/datasets/chaos/data_slices_ct"
    DATASET_ARG=chaos
    ;;
  chaos_mr)
    # Built by extract_data_chaos_unet.py --modality mr: liver, kidneys, spleen.
    DATA_DIR="$SCRATCH/datasets/chaos/data_slices_mr"
    DATASET_ARG=chaos
    ;;
  *)
    echo "ERROR: unsupported dataset '$DATASET'. Expected 'acdc', 'brats', 'brats3',"
    echo "       'refuge', 'cataracts', 'chaos_ct' or 'chaos_mr'."
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
        echo "ERROR: prepared dataset directory not found at $DATA_DIR"
        echo "       Build it with the matching extract_data_*.py (extract_data_brats3.py,"
        echo "       extract_data_refuge.py, extract_data_cataracts.py, extract_data_chaos_unet.py)."
        exit 1
    fi
fi

export PYTHON_DATA_DIR="$DATA_DIR"
DATASET_ROOT="${DATASET_ROOT:-$DATA_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"

# MODEL only picks the default paths. inference.py rebuilds the architecture from
# the checkpoint's own args (model, image_size, in_channels), so a TransUNet
# checkpoint is restored with the geometry it was trained on.
MODEL="${MODEL:-unet}"
MODEL_LOWER=$(printf '%s' "$MODEL" | tr '[:upper:]' '[:lower:]')
LOSS="${LOSS:-bound_ebo_ce}"

if [ "$MODEL_LOWER" = "transunet" ] && [ "$DATASET_LOWER" = "brats" ]; then
    echo "ERROR: TransUNet's pretrained stem takes 3 channels; raw BraTS has 4."
    echo "Use DATASET=brats3 (built by extract_data_brats3.py)."
    exit 1
fi

MODEL_ROOT="${MODEL_ROOT:-$SCRATCH/grid_search/$MODEL_LOWER/$DATASET_LOWER/$LOSS}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$SCRATCH/inference/$MODEL_LOWER/$DATASET_LOWER/grid_search_$LOSS}"

# Three independent ways of picking the winner of the grid search, all written by
# select_grid_best.py:
#   dice           - best validation Dice.
#   dice_top_fpr95 - best (lowest) energy FPR95 among the top TOP_FRAC by Dice.
#   fpr95          - best (lowest) energy FPR95 over every config, no Dice filter.
# SELECTION=both runs dice + dice_top_fpr95, SELECTION=all runs the three.
SELECTION="${SELECTION:-all}"
TOP_FRAC="${TOP_FRAC:-0.2}"
# Rebuild the selection JSONs from the trial dirs when they are missing (grid
# searches run before this selection existed, or an array task that died).
AUTO_SELECT="${AUTO_SELECT:-1}"
# Passed to select_grid_best.py: drops trials with a shorter history (still
# running / crashed). Empty means rank every trial that has a checkpoint.
EXPECTED_EPOCHS="${EXPECTED_EPOCHS:-}"
SELECTION_MODE="${SELECTION_MODE:-best}"

LOSS_NORM=$(PYTHONPATH="$ROOT_DIR" "$PYTHON_BIN" -c 'import sys; from losses import normalize_loss_name; print(normalize_loss_name(sys.argv[1]))' "$LOSS")
if [ "$SELECTION_MODE" = "last" ]; then
    CHECKPOINT_NAME="${CHECKPOINT_NAME:-last_${LOSS_NORM}.pt}"
else
    CHECKPOINT_NAME="${CHECKPOINT_NAME:-best_${LOSS_NORM}.pt}"
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
BATCH_SIZE="${BATCH_SIZE:-4}"
DEVICE="${DEVICE:-cuda}"
NUM_SAMPLES="${NUM_SAMPLES:-4}"
TEMPERATURE="${TEMPERATURE:-1.0}"
MAX_PIXELS_KDE="${MAX_PIXELS_KDE:-200000}"
ENERGY_THRESH="${ENERGY_THRESH:--5}"
MSP_THRESH="${MSP_THRESH:-0.999}"
MODES="${MODES:-all}"
TRIAL_FILTER="${TRIAL_FILTER:-}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

# The D matrix is fitted on one checkpoint, so each selection needs its own.
# D_MATRIX stays as a fallback for when several selections land on the same trial.
D_MATRIX="${D_MATRIX:-}"
D_MATRIX_DICE="${D_MATRIX_DICE:-$D_MATRIX}"
D_MATRIX_DICE_TOP_FPR95="${D_MATRIX_DICE_TOP_FPR95:-$D_MATRIX}"
D_MATRIX_FPR95="${D_MATRIX_FPR95:-$D_MATRIX}"

if [ ! -d "$MODEL_ROOT" ]; then
    echo "ERROR: model root not found: $MODEL_ROOT"
    echo "Set MODEL_ROOT, or check MODEL=$MODEL_LOWER DATASET=$DATASET_LOWER LOSS=$LOSS"
    exit 1
fi

mkdir -p "$OUTPUT_ROOT"

cd "$ROOT_DIR"

echo "SLURM_JOB_ID=${SLURM_JOB_ID:-local}"
echo "Dataset        : ${DATASET_LOWER} (--dataset ${DATASET_ARG})"
echo "Dataset root   : ${DATASET_ROOT}"
echo "Model          : ${MODEL_LOWER} (architecture read from checkpoint)"
echo "Loss           : ${LOSS}"
echo "Model root     : ${MODEL_ROOT}"
echo "Output root    : ${OUTPUT_ROOT}"
echo "Checkpoint name: ${CHECKPOINT_NAME}"
echo "Selections     : ${criteria[*]} (top-frac ${TOP_FRAC})"
echo "Modes          : ${MODES}"
echo "Trial filter   : ${TRIAL_FILTER:-<none>}"

# Rebuild both selection JSONs when at least one is missing.
need_selection=0
for criterion in "${criteria[@]}"; do
    if [ -z "$BEST_CONFIG_JSON" ] && [ ! -f "$MODEL_ROOT/grid_search_best_${criterion}.json" ]; then
        need_selection=1
    fi
done

if [ "$need_selection" = "1" ]; then
    if [ "$AUTO_SELECT" != "1" ]; then
        echo "ERROR: selection JSON missing under $MODEL_ROOT. Run select_grid_best.py, or set AUTO_SELECT=1."
        exit 1
    fi
    echo
    echo "Selection JSON missing: running select_grid_best.py on $MODEL_ROOT"
    expected_epochs_args=()
    if [ -n "$EXPECTED_EPOCHS" ]; then
        expected_epochs_args+=(--expected-epochs "$EXPECTED_EPOCHS")
    fi
    "$PYTHON_BIN" select_grid_best.py \
      --results-dir "$MODEL_ROOT" \
      --loss "$LOSS_NORM" \
      --selection-mode "$SELECTION_MODE" \
      --top-frac "$TOP_FRAC" \
      "${expected_epochs_args[@]}"
fi

read_json_field() {
    "$PYTHON_BIN" -c 'import json, sys; from pathlib import Path; print(json.loads(Path(sys.argv[1]).read_text()).get(sys.argv[2], "") or "")' "$1" "$2"
}

# Both selections can land on the same trial; inference then runs once and the
# second criterion only records a pointer to the shared results.
declare -A ran_trials=()
executed=()

for criterion in "${criteria[@]}"; do
    config_json="${BEST_CONFIG_JSON:-$MODEL_ROOT/grid_search_best_${criterion}.json}"

    echo
    echo "============================================================"
    echo "Selection: $criterion"
    echo "Config   : $config_json"

    if [ ! -f "$config_json" ]; then
        echo "ERROR: best config JSON not found: $config_json"
        exit 1
    fi

    best_trial_dir=$(read_json_field "$config_json" trial_dir)
    if [ -z "$best_trial_dir" ]; then
        echo "ERROR: trial_dir missing in $config_json"
        exit 1
    fi

    if [ ! -d "$best_trial_dir" ]; then
        # Grid search results moved since selection (e.g. copied off $SCRATCH).
        candidate_dir="$MODEL_ROOT/$(basename "$best_trial_dir")"
        if [ -d "$candidate_dir" ]; then
            best_trial_dir="$candidate_dir"
        else
            echo "ERROR: best trial directory not found: $best_trial_dir"
            exit 1
        fi
    fi

    trial_name=$(basename "$best_trial_dir")
    echo "Trial    : $trial_name"

    if [ -n "$TRIAL_FILTER" ] && [[ "$trial_name" != *"$TRIAL_FILTER"* ]]; then
        echo "SKIP: trial '$trial_name' does not match TRIAL_FILTER='$TRIAL_FILTER'"
        continue
    fi

    checkpoint_path=$(read_json_field "$config_json" checkpoint)
    if [ -z "$checkpoint_path" ] || [ ! -f "$checkpoint_path" ]; then
        checkpoint_path="$best_trial_dir/$CHECKPOINT_NAME"
    fi
    if [ ! -f "$checkpoint_path" ]; then
        echo "ERROR: checkpoint not found at $checkpoint_path"
        exit 1
    fi

    case "$criterion" in
      dice)           criterion_d_matrix="$D_MATRIX_DICE" ;;
      dice_top_fpr95) criterion_d_matrix="$D_MATRIX_DICE_TOP_FPR95" ;;
      fpr95)          criterion_d_matrix="$D_MATRIX_FPR95" ;;
    esac

    d_matrix_args=()
    if [ -n "$criterion_d_matrix" ]; then
        if [ ! -f "$criterion_d_matrix" ]; then
            echo "ERROR: D matrix not found: $criterion_d_matrix"
            echo "Produce it for this trial's checkpoint with compute_d_matrix.sh (SPLIT=val)."
            exit 1
        fi
        d_matrix_args+=(--d-matrix "$criterion_d_matrix")
    fi

    output_dir="$OUTPUT_ROOT/$criterion/$trial_name"
    mkdir -p "$output_dir"

    cp "$config_json" "$output_dir/selection.json"

    if [ -n "${ran_trials[$trial_name]:-}" ]; then
        echo "[REUSE] $trial_name already tested for selection '${ran_trials[$trial_name]}'"
        echo "        results: $OUTPUT_ROOT/${ran_trials[$trial_name]}/$trial_name"
        executed+=("$criterion -> $trial_name (same model as ${ran_trials[$trial_name]})")
        continue
    fi

    metrics_file="$output_dir/metrics.json"
    if [ "$SKIP_EXISTING" = "1" ] && [ -f "$metrics_file" ]; then
        echo "[SKIP] $trial_name: existing metrics found at $metrics_file"
        ran_trials[$trial_name]="$criterion"
        executed+=("$criterion -> $trial_name (skipped, already tested)")
        continue
    fi

    echo "Checkpoint: $checkpoint_path"
    echo "Output    : $output_dir"
    echo "D matrix  : ${criterion_d_matrix:-none (RELU score disabled)}"
    echo "============================================================"

    "$PYTHON_BIN" inference.py ${MODES} \
      --checkpoint "$checkpoint_path" \
      --dataset "$DATASET_ARG" \
      --dataset-root "$DATASET_ROOT" \
      --output-dir "$output_dir" \
      --batch-size "$BATCH_SIZE" \
      --device "$DEVICE" \
      --num-samples "$NUM_SAMPLES" \
      --temperature "$TEMPERATURE" \
      --max-pixels-kde "$MAX_PIXELS_KDE" \
      --energy-threshold "$ENERGY_THRESH" \
      --msp-threshold "$MSP_THRESH" \
      "${d_matrix_args[@]}"

    ran_trials[$trial_name]="$criterion"
    executed+=("$criterion -> $trial_name")
done

echo
echo "Finished best-config inference."
for line in "${executed[@]}"; do
    echo "  $line"
done
