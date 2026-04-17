#!/bin/bash
#SBATCH --time=05:00:00
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


OUTPUT_DIR=$SCRATCH/grid_search

PYTHON_SCRIPT=grid_search.py
export PYTHON_DATA_DIR="$DATA_DIR"

echo "Starting grid search..."

python $PYTHON_SCRIPT \
  --dataset-root $DATA_DIR \
  --output-dir $OUTPUT_DIR \
  --model unet \
  --loss ebo_ce \
  --epochs 50 \
  --batch-size 8 \
  --lr 1e-3 \
  --num-workers 8 \
  --image-size 256 \
  --num-classes 4 \
  --device cuda \
  --metric loss \
  --selection-mode best \
  --lambda-ebo-in-grid 0.1 0.3 0.6\
  --lambda-ebo-corr-grid 0.1 0.3 0.6\
  --margin-correct-grid -17\
  --margin-miss-grid -5

echo "Grid search finished!"