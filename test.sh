#!/bin/bash
#SBATCH --time=00:05:00
#SBATCH --partition=compute_full_node
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --output=/scratch/guibo/slurm_outputs/slurm-%j.out
#SBATCH --error=/scratch/guibo/slurm_outputs/slurm-%j.err
#SBATCH --account=rrg-josedolz
#SBATCH --partition=compute


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