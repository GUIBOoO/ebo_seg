#!/bin/bash
#SBATCH --time=04:00:00
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

# MODEL only picks the default checkpoint/output paths. inference.py rebuilds the
# architecture from the checkpoint's own args (model, image_size, in_channels),
# so a TransUNet checkpoint is restored with the geometry it was trained on.
MODEL="${MODEL:-unet}"
MODEL_LOWER=$(printf '%s' "$MODEL" | tr '[:upper:]' '[:lower:]')

if [ "$MODEL_LOWER" = "transunet" ] && [ "$DATASET_LOWER" = "brats" ]; then
    echo "ERROR: TransUNet's pretrained stem takes 3 channels; raw BraTS has 4."
    echo "Use DATASET=brats3 (built by extract_data_brats3.py)."
    exit 1
fi

DEFAULT_CHECKPOINT="$SCRATCH/models/ebo_seg/$MODEL_LOWER/$DATASET_ARG/best_ce_dice.pt"
DEFAULT_OUTPUT_DIR="$SCRATCH/inference/$MODEL_LOWER/$DATASET_ARG/baseline"

CHECKPOINT="${CHECKPOINT:-$DEFAULT_CHECKPOINT}"
OUTPUT_DIR="${OUTPUT_DIR:-$DEFAULT_OUTPUT_DIR}"
BATCH_SIZE="${BATCH_SIZE:-4}"
DEVICE="${DEVICE:-cuda}"
NUM_SAMPLES="${NUM_SAMPLES:-4}"
TEMPERATURE="${TEMPERATURE:-1.0}"
MAX_PIXELS_KDE="${MAX_PIXELS_KDE:-200000}"
ENERGY_THRESH="${ENERGY_THRESH:--5}"
MSP_THRESH="${MSP_THRESH:-0.999}"
MODES="${MODES:-all}"

LAMBDA_WEIGHT="${LAMBDA_WEIGHT:-0.5}"
LAMBDA_TAG=$(printf '%g' "$LAMBDA_WEIGHT")
D_MATRIX="${D_MATRIX:-$SCRATCH/d_matrices/$MODEL_LOWER/$DATASET_ARG/d_matrix_${DATASET_ARG}_val_lambda_${LAMBDA_TAG}.npy}"
if [ "$D_MATRIX" = "none" ]; then
    D_MATRIX=""
fi

cd /home/guibo/links/projects/rrg-josedolz/guibo/ebo_seg

if [ ! -f "$CHECKPOINT" ]; then
    echo "ERROR: checkpoint not found: $CHECKPOINT"
    echo "Set CHECKPOINT=/path/to/best_<loss>.pt"
    exit 1
fi

d_matrix_args=()
if [ -n "$D_MATRIX" ]; then
    if [ ! -f "$D_MATRIX" ]; then
        echo "ERROR: D matrix not found: $D_MATRIX"
        echo "Produce it with: DATASET=$DATASET_LOWER MODEL=$MODEL_LOWER SPLIT=val bash compute_d_matrix.sh"
        exit 1
    fi
    d_matrix_args+=(--d-matrix "$D_MATRIX")
fi

echo "SLURM_JOB_ID=${SLURM_JOB_ID:-local}"
echo "Dataset      : ${DATASET_LOWER} (--dataset ${DATASET_ARG})"
echo "Dataset root : ${DATASET_ROOT}"
echo "Model        : ${MODEL_LOWER} (architecture read from checkpoint)"
echo "Checkpoint   : ${CHECKPOINT}"
echo "Output dir   : ${OUTPUT_DIR}"
echo "Modes        : ${MODES}"
echo "Energy thr   : ${ENERGY_THRESH}"
echo "MSP thr      : ${MSP_THRESH}"
echo "D matrix     : ${D_MATRIX:-none (RELU score disabled)}"

"${PYTHON_BIN}" inference.py ${MODES} \
  --checkpoint "${CHECKPOINT}" \
  --dataset "${DATASET_ARG}" \
  --dataset-root "${DATASET_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --batch-size "${BATCH_SIZE}" \
  --device "${DEVICE}" \
  --num-samples "${NUM_SAMPLES}" \
  --temperature "${TEMPERATURE}" \
  --max-pixels-kde "${MAX_PIXELS_KDE}" \
  --energy-threshold "${ENERGY_THRESH}" \
  --msp-threshold "${MSP_THRESH}" \
  "${d_matrix_args[@]}"
