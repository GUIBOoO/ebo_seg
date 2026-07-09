#!/bin/bash
#SBATCH --time=04:00:00
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --output=/scratch/guibo/slurm_outputs/slurm-%j.out
#SBATCH --error=/scratch/guibo/slurm_outputs/slurm-%j.err
#SBATCH --account=rrg-josedolz
#SBATCH --partition=compute

module load python/3.10.13
source ~/nnunet/bin/activate

export nnUNet_raw="/scratch/$USER/nnUNet_raw"
export nnUNet_preprocessed="/scratch/$USER/nnUNet_preprocessed"
export nnUNet_results="/scratch/$USER/nnUNet_results"

# nnUNetv2_plan_and_preprocess -d 1 --verify_dataset_integrity   # ACDC
nnUNetv2_plan_and_preprocess -d 2 --verify_dataset_integrity   # BraTS2D
