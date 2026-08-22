#!/bin/bash
#SBATCH --time=04:00:00
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --output=/scratch/guibo/slurm_outputs/slurm-%j.out
#SBATCH --error=/scratch/guibo/slurm_outputs/slurm-%j.err
#SBATCH --account=rrg-josedolz
#SBATCH --partition=compute

# Fingerprint + planning + preprocessing of an nnU-Net dataset.
#
#   DATASET=refuge    sbatch preprocess_data_nnunet.sh
#   DATASET=cataracts sbatch preprocess_data_nnunet.sh
#   DATASET=chaos_mr  sbatch preprocess_data_nnunet.sh
#   DATASET=all       sbatch preprocess_data_nnunet.sh
#
# The raw dataset must exist first (extract_data_*_nnunet.py). This step is CPU
# only; NUM_PROCESSES defaults to the CPUs the job was given.

set -euo pipefail

module load python/3.10.13
source ~/nnunet/bin/activate

export nnUNet_raw="${nnUNet_raw:-/scratch/$USER/nnUNet_raw}"
export nnUNet_preprocessed="${nnUNet_preprocessed:-/scratch/$USER/nnUNet_preprocessed}"
export nnUNet_results="${nnUNet_results:-/scratch/$USER/nnUNet_results}"

DATASET="${DATASET:-cataracts}"
DATASET_LOWER=$(printf '%s' "$DATASET" | tr '[:upper:]' '[:lower:]')

case "$DATASET_LOWER" in
  acdc|brats|refuge|cataracts|chaos_mr) DATASETS=("$DATASET_LOWER") ;;
  all) DATASETS=(acdc brats refuge cataracts chaos_mr) ;;
  *)
    echo "ERROR: unsupported dataset '$DATASET'. Expected 'acdc', 'brats', 'refuge',"
    echo "       'cataracts', 'chaos_mr' or 'all'."
    exit 1
    ;;
esac

dataset_id_for() {
    case "$1" in
      acdc)      echo 1 ;;
      brats)     echo 2 ;;
      refuge)    echo 4 ;;
      cataracts) echo 5 ;;
      chaos_mr)  echo 6 ;;
    esac
}

# Which configurations to plan and preprocess. The grid search trains 2d only,
# and planning 3d_fullres on a stack of independent 2D images is meaningless, so
# the 2D datasets stop at 2d. Override with CONFIGURATIONS="2d 3d_fullres".
configurations_for() {
    case "$1" in
      acdc|brats) echo "" ;;        # keep nnU-Net's default (all configurations)
      *)          echo "2d" ;;
    esac
}

VERIFY="${VERIFY:-1}"
NUM_PROCESSES="${NUM_PROCESSES:-${SLURM_CPUS_PER_TASK:-4}}"

echo "SLURM_JOB_ID=${SLURM_JOB_ID:-local}"
echo "Datasets     : ${DATASETS[*]}"
echo "nnUNet_raw   : $nnUNet_raw"
echo "Preprocessed : $nnUNet_preprocessed"
echo "Processes    : $NUM_PROCESSES"

for dataset in "${DATASETS[@]}"; do
    dataset_id=$(dataset_id_for "$dataset")
    configurations="${CONFIGURATIONS:-$(configurations_for "$dataset")}"

    raw_dir=$(printf '%s/Dataset%03d_*' "$nnUNet_raw" "$dataset_id")
    if ! compgen -G "$raw_dir" > /dev/null; then
        echo "ERROR: no Dataset$(printf '%03d' "$dataset_id")_* under $nnUNet_raw for '$dataset'."
        echo "       Build it first with the matching extract_data_*_nnunet.py."
        exit 1
    fi

    cmd=(nnUNetv2_plan_and_preprocess -d "$dataset_id" -np "$NUM_PROCESSES")
    if [ -n "$configurations" ]; then
        # shellcheck disable=SC2206
        cmd+=(-c $configurations)
    fi
    if [ "$VERIFY" = "1" ]; then
        cmd+=(--verify_dataset_integrity)
    fi

    echo
    echo "=================== $dataset (Dataset$(printf '%03d' "$dataset_id")) ==================="
    echo "${cmd[@]}"
    "${cmd[@]}"
done

echo
echo "Done. Train with, for example:"
echo "  DATASET=$DATASET_LOWER LOSS=bound_ebo SELECT_BEST=0 sbatch nnunet_grid_search.sh"
