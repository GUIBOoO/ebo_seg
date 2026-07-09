#!/bin/bash
#SBATCH --time=02:00:00
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --output=/scratch/guibo/slurm_outputs/slurm-%j.out
#SBATCH --error=/scratch/guibo/slurm_outputs/slurm-%j.err
#SBATCH --account=rrg-josedolz
#SBATCH --partition=compute

source ~/.bash_profile 
module load python/3.11.5 
source /home/guibo/ebo-seg/bin/activate

python extract_data_brats.py 
