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
NNUNET_RESULTS_BASE="${NNUNET_RESULTS_BASE:-/scratch/$USER/nnUNet_results_grid_search/BraTS}"
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"

BEST_CONFIG_JSON="${BEST_CONFIG_JSON:-$NNUNET_RESULTS_BASE/nnunet_grid_search_best.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/scratch/$USER/nnUNet_inference_grid_search/BraTS}"
CHECKPOINT_NAME="${CHECKPOINT_NAME:-checkpoint_best.pth}"
DEVICE="${DEVICE:-cuda}"
TILE_STEP_SIZE="${TILE_STEP_SIZE:-0.5}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
TRIAL_FILTER="${TRIAL_FILTER:-}"
PERFORM_EVERYTHING_ON_DEVICE="${PERFORM_EVERYTHING_ON_DEVICE:-0}"
DISABLE_MIRRORING="${DISABLE_MIRRORING:-0}"
# Optional: .npy/.pt produced by compute_d_matrix_nnunet.sh. Enables the RELU score.
D_MATRIX="${D_MATRIX:-}"

if [ ! -f "$BEST_CONFIG_JSON" ]; then
    echo "ERROR: best config JSON not found: $BEST_CONFIG_JSON"
    echo "Run: SELECT_BEST=1 NNUNET_RESULTS_BASE=$NNUNET_RESULTS_BASE bash nnunet_grid_search.sh"
    exit 1
fi

best_trial_name=$("$PYTHON_BIN" -c 'import json, sys; from pathlib import Path; data=json.loads(Path(sys.argv[1]).read_text()); trial=data.get("best_trial", data); print(trial.get("trial_name", ""))' "$BEST_CONFIG_JSON")
best_checkpoint=$("$PYTHON_BIN" -c 'import json, sys; from pathlib import Path; data=json.loads(Path(sys.argv[1]).read_text()); trial=data.get("best_trial", data); print(trial.get("checkpoint", ""))' "$BEST_CONFIG_JSON")

if [ -z "$best_trial_name" ]; then
    if [ -n "$best_checkpoint" ]; then
        best_trial_name=$("$PYTHON_BIN" -c 'import sys; from pathlib import Path; parts=Path(sys.argv[1]).parts; print(next((p for p in parts if p.startswith("trial_")), "best_trial"))' "$best_checkpoint")
    else
        best_trial_name="best_trial"
    fi
fi

if [ -n "$TRIAL_FILTER" ] && [[ "$best_trial_name" != *"$TRIAL_FILTER"* ]]; then
    echo "SKIP: best trial '$best_trial_name' does not match TRIAL_FILTER='$TRIAL_FILTER'"
    exit 0
fi

output_dir="$OUTPUT_ROOT/$best_trial_name"
metrics_file="$output_dir/metrics.json"

if [ "$SKIP_EXISTING" = "1" ] && [ -f "$metrics_file" ]; then
    echo "[SKIP] $best_trial_name: existing metrics found at $metrics_file"
    exit 0
fi

mkdir -p "$output_dir"
cd "$ROOT_DIR"

extra_args=()
if [ "$PERFORM_EVERYTHING_ON_DEVICE" = "1" ]; then
    extra_args+=(--perform-everything-on-device)
fi
if [ "$DISABLE_MIRRORING" = "1" ]; then
    extra_args+=(--disable-mirroring)
fi
if [ -n "$D_MATRIX" ]; then
    if [ ! -f "$D_MATRIX" ]; then
        echo "ERROR: D matrix not found: $D_MATRIX"
        echo "Produce it with: SPLIT=val bash compute_d_matrix_nnunet.sh"
        exit 1
    fi
    extra_args+=(--d-matrix "$D_MATRIX")
fi

echo "SLURM_JOB_ID=${SLURM_JOB_ID:-local}"
echo "Best config    : $BEST_CONFIG_JSON"
echo "Best trial     : $best_trial_name"
echo "Checkpoint     : ${best_checkpoint:-from JSON/model folder}"
echo "Preprocessed   : $nnUNet_preprocessed"
echo "Output         : $output_dir"
echo "D matrix       : ${D_MATRIX:-none (RELU score disabled)}"
echo "Device         : $DEVICE"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi
fi

"$PYTHON_BIN" inference_nnunet.py \
    --best-json "$BEST_CONFIG_JSON" \
    --checkpoint-name "$CHECKPOINT_NAME" \
    --preprocessed-dir "$nnUNet_preprocessed" \
    --output-dir "$output_dir" \
    --device "$DEVICE" \
    --tile-step-size "$TILE_STEP_SIZE" \
    "${extra_args[@]}"

echo
echo "Finished nnU-Net best-grid inference."
echo "Trial executed: $best_trial_name"
