#!/bin/bash
#SBATCH --job-name=sens_cataracts
#SBATCH --time=24:00:00
#SBATCH --array=0-19
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --output=/scratch/guibo/slurm_outputs/slurm-%A_%a.out
#SBATCH --error=/scratch/guibo/slurm_outputs/slurm-%A_%a.err
#SBATCH --account=rrg-josedolz
#SBATCH --partition=compute

# Lambda sensitivity analysis on cataracts (UNet, bound_ebo_ce): 4 lambdas x 5 values,
# one model per array task. See sensitivity_lambda.sh for the task layout.
# BOUNDARY_K is the dice_top_fpr95 winner of the cataracts ablation_k sweep.
export DATASET=cataracts
export BOUNDARY_K="${BOUNDARY_K:-3}"

exec bash /home/guibo/links/projects/rrg-josedolz/guibo/ebo_seg/sensitivity_lambda.sh
