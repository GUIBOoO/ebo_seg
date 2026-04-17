#!/bin/bash
#SBATCH --time=00:40:00
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

DATA_DIR=$SLURM_TMPDIR/dataset/ACDC/database
mkdir -p $DATA_DIR

ACDC_ZIP=$SCRATCH/datasets/ACDC/ACDC.zip

if [ ! -f $ACDC_ZIP ]; then
    echo "ERROR: image zip not found"
    exit 1
fi

echo "Décompression ACDC images..."
unzip -q $ACDC_ZIP -d $SLURM_TMPDIR/dataset

echo "Structure finale :"
find $DATA_DIR -maxdepth 2 -type d

export PYTHON_DATA_DIR=$DATA_DIR
DATASET_ROOT="${DATASET_ROOT:-$DATA_DIR}"

OUTPUT_DIR="${OUTPUT_DIR:-/home/guibo/links/scratch/models/ebo_seg/ebo_seg_unet_25_5_2in_1cor}"
PYTHON_BIN="${PYTHON_BIN:-python}"
EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-8}"
LR="${LR:-1e-3}"
NUM_WORKERS="${NUM_WORKERS:-4}"
IMAGE_SIZE="${IMAGE_SIZE:-256}"
NUM_CLASSES="${NUM_CLASSES:-4}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-42}"
LOSS="${LOSS:-ebo_ce}"

cd /home/guibo/links/projects/rrg-josedolz/guibo/ebo_seg

echo "SLURM_JOB_ID=${SLURM_JOB_ID:-local}"
echo "Dataset root : ${DATASET_ROOT}"
echo "Output dir   : ${OUTPUT_DIR}"

"${PYTHON_BIN}" train_unet.py \
  --dataset-root "${DATASET_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --lr "${LR}" \
  --num-workers "${NUM_WORKERS}" \
  --image-size "${IMAGE_SIZE}" \
  --num-classes "${NUM_CLASSES}" \
  --device "${DEVICE}" \
  --seed "${SEED}"\
  --loss "${LOSS}"
