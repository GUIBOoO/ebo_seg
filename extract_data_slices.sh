#!/bin/bash
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/scratch/guibo/slurm_outputs/slurm-%j.out
#SBATCH --error=/scratch/guibo/slurm_outputs/slurm-%j.err
#SBATCH --account=rrg-josedolz
#SBATCH --partition=compute

# Convert REFUGE / CaDIS / CHAOS to the .npy slice layout the U-Net pipeline
# reads. No GPU needed: this is decoding + resizing only.
#
#   DATASET=refuge    bash extract_data_slices.sh
#   DATASET=cataracts bash extract_data_slices.sh     # CADIS_EXPERIMENT=1|2|3
#   DATASET=chaos     bash extract_data_slices.sh     # writes _ct and _mr
#   DATASET=all       bash extract_data_slices.sh
#
# The archives must be unzipped first (they ship one big zip each, and CHAOS
# nests Train_Sets.zip / Test_Sets.zip inside):
#   unzip refuge.zip    -d $SCRATCH/datasets/refuge/raw
#   unzip cataracts.zip -d $SCRATCH/datasets/cataracts/raw
#   unzip chaos.zip     -d $SCRATCH/datasets/chaos/raw
#   unzip $SCRATCH/datasets/chaos/raw/CHAOS_Original/Train_Sets.zip \
#         -d $SCRATCH/datasets/chaos/raw/CHAOS_Original

set -euo pipefail

source ~/.bash_profile
module load python/3.11.5
source /home/guibo/ebo-seg/bin/activate

ROOT_DIR="${ROOT_DIR:-/home/guibo/links/projects/rrg-josedolz/guibo/ebo_seg}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATASET="${DATASET:-all}"
DATASET_LOWER=$(printf '%s' "$DATASET" | tr '[:upper:]' '[:lower:]')

case "$DATASET_LOWER" in
  refuge|cataracts|chaos) DATASETS=("$DATASET_LOWER") ;;
  all)                    DATASETS=(refuge cataracts chaos) ;;
  *)
    echo "ERROR: unsupported dataset '$DATASET'. Expected 'refuge', 'cataracts', 'chaos' or 'all'."
    exit 1
    ;;
esac

# Passed through to the converters; see their module docstrings for the rest.
CADIS_EXPERIMENT="${CADIS_EXPERIMENT:-1}"
CHAOS_MODALITY="${CHAOS_MODALITY:-both}"

cd "$ROOT_DIR"

echo "SLURM_JOB_ID=${SLURM_JOB_ID:-local}"
echo "Datasets : ${DATASETS[*]}"
echo "Scratch  : ${SCRATCH:-unset}"

for dataset in "${DATASETS[@]}"; do
    echo
    echo "=================== $dataset ==================="
    case "$dataset" in
      refuge)
        "$PYTHON_BIN" extract_data_refuge.py
        ;;
      cataracts)
        "$PYTHON_BIN" extract_data_cataracts.py --experiment "$CADIS_EXPERIMENT"
        ;;
      chaos)
        "$PYTHON_BIN" extract_data_chaos_unet.py --modality "$CHAOS_MODALITY"
        ;;
    esac
done

echo
echo "Done. Train with, for example:"
echo "  DATASET=refuge    LOSS=bound_ebo_ce bash slurm_train_unet.sh"
echo "  DATASET=cataracts LOSS=bound_ebo_ce bash slurm_train_unet.sh"
echo "  DATASET=chaos_mr  LOSS=bound_ebo_ce bash slurm_train_unet.sh"
