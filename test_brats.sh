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

echo "Job started on $(hostname)"

DATA_DIR=$SLURM_TMPDIR/dataset
mkdir -p $DATA_DIR

BRATS_ZIP=$SCRATCH/datasets/Brats/brats_train.zip

if [ ! -f $BRATS_ZIP ]; then
    echo "ERROR: brats_train.zip not found"
    exit 1
fi


echo "Unzipping BraTS"
unzip -q $BRATS_ZIP -d $DATA_DIR

echo "FULL DATA_DIR root:"
find $DATA_DIR -maxdepth 2 -type d | sort | head -n 50

echo ""
echo "Brats folders:"
find $DATA_DIR -type d -iname "*train*" | sort | head -n 50

echo ""
echo "SEG files sample:"
find $DATA_DIR -name "*seg.nii.gz" | head -n 10

export PYTHON_DATA_DIR=$DATA_DIR

echo "PYTHON_DATA_DIR set to $PYTHON_DATA_DIR"

python3 datasets.py