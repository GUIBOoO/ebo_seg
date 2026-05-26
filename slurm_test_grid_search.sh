#!/bin/bash
#SBATCH --time=00:15:00
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

ROOT_DIR="/home/guibo/links/projects/rrg-josedolz/guibo/ebo_seg"
DATASET="${DATASET:-acdc}"
DATASET_LOWER=$(printf '%s' "$DATASET" | tr '[:upper:]' '[:lower:]')

case "$DATASET_LOWER" in
  acdc)
    DATA_DIR="$SLURM_TMPDIR/dataset/ACDC/database"
    DATASET_ZIP="$SCRATCH/datasets/ACDC/ACDC.zip"
    UNZIP_TARGET="$SLURM_TMPDIR/dataset"
    DEFAULT_MODEL_ROOT="/home/guibo/links/scratch/grid_search"
    ;;
  brats)
    DATA_DIR="$SCRATCH/datasets/Brats/data_slices"
    DEFAULT_MODEL_ROOT="/home/guibo/links/scratch/models/ebo_seg/brats/grid_search"
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
MODEL_ROOT="${MODEL_ROOT:-$DEFAULT_MODEL_ROOT}"
BEST_CONFIG_JSON="${BEST_CONFIG_JSON:-$MODEL_ROOT/grid_search_best.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/guibo/links/scratch/inference/grid_search}"
CHECKPOINT_NAME="${CHECKPOINT_NAME:-best_bound_log_ebo.pt}"
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

if [ ! -d "$MODEL_ROOT" ]; then
    echo "ERROR: model root not found: $MODEL_ROOT"
    exit 1
fi

mkdir -p "$OUTPUT_ROOT"

cd "$ROOT_DIR"

echo "SLURM_JOB_ID=${SLURM_JOB_ID:-local}"
echo "Dataset        : ${DATASET_LOWER}"
echo "Dataset root   : ${DATASET_ROOT}"
echo "Model root     : ${MODEL_ROOT}"
echo "Best config    : ${BEST_CONFIG_JSON}"
echo "Output root    : ${OUTPUT_ROOT}"
echo "Checkpoint name: ${CHECKPOINT_NAME}"
echo "Modes          : ${MODES}"
echo "Trial filter   : ${TRIAL_FILTER:-<none>}"

if [ ! -f "$BEST_CONFIG_JSON" ]; then
    echo "ERROR: best config JSON not found: $BEST_CONFIG_JSON"
    exit 1
fi

best_trial_dir=$("$PYTHON_BIN" -c 'import json, sys; from pathlib import Path; data = json.loads(Path(sys.argv[1]).read_text()); print(data.get("trial_dir", ""))' "$BEST_CONFIG_JSON")

if [ -z "$best_trial_dir" ]; then
    echo "ERROR: trial_dir missing in $BEST_CONFIG_JSON"
    exit 1
fi

if [ ! -d "$best_trial_dir" ]; then
    candidate_dir="$MODEL_ROOT/$(basename "$best_trial_dir")"
    if [ -d "$candidate_dir" ]; then
        best_trial_dir="$candidate_dir"
    else
        echo "ERROR: best trial directory not found: $best_trial_dir"
        exit 1
    fi
fi

trial_name=$(basename "$best_trial_dir")

if [ -n "$TRIAL_FILTER" ] && [[ "$trial_name" != *"$TRIAL_FILTER"* ]]; then
    echo "SKIP: best trial '$trial_name' does not match TRIAL_FILTER='$TRIAL_FILTER'"
    exit 0
fi

checkpoint_path="$best_trial_dir/$CHECKPOINT_NAME"
if [ ! -f "$checkpoint_path" ]; then
    echo "ERROR: checkpoint not found at $checkpoint_path"
    exit 1
fi

output_dir="$OUTPUT_ROOT/$trial_name"
metrics_file="$output_dir/metrics.json"

if [ "$SKIP_EXISTING" = "1" ] && [ -f "$metrics_file" ]; then
    echo "[SKIP] $trial_name: existing metrics found at $metrics_file"
    exit 0
fi

mkdir -p "$output_dir"

echo
echo "============================================================"
echo "Running inference for best config: $trial_name"
echo "Checkpoint: $checkpoint_path"
echo "Output    : $output_dir"
echo "============================================================"

"$PYTHON_BIN" inference.py ${MODES} \
  --checkpoint "$checkpoint_path" \
  --dataset "$DATASET_LOWER" \
  --dataset-root "$DATASET_ROOT" \
  --output-dir "$output_dir" \
  --batch-size "$BATCH_SIZE" \
  --device "$DEVICE" \
  --num-samples "$NUM_SAMPLES" \
  --temperature "$TEMPERATURE" \
  --max-pixels-kde "$MAX_PIXELS_KDE" \
  --energy-threshold "$ENERGY_THRESH" \
  --msp-threshold "$MSP_THRESH"

echo
echo "Finished best-config inference."
echo "Trial executed: $trial_name"
