#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --output=/scratch/guibo/slurm_outputs/slurm-%j.out
#SBATCH --error=/scratch/guibo/slurm_outputs/slurm-%j.err
#SBATCH --account=rrg-josedolz
#SBATCH --partition=compute

module load python/3.10.13
source ~/nnunet/bin/activate

ROOT_DIR="${ROOT_DIR:-/home/guibo/links/projects/rrg-josedolz/guibo/ebo_seg}"

export nnUNet_raw="${nnUNet_raw:-/scratch/$USER/nnUNet_raw}"
export nnUNet_preprocessed="${nnUNet_preprocessed:-/scratch/$USER/nnUNet_preprocessed}"
export nnUNet_results="${nnUNet_results:-/scratch/$USER/nnUNet_results}"
export nnUNet_compile="${nnUNet_compile:-False}"
export nnUNet_extTrainer="${nnUNet_extTrainer:-$ROOT_DIR/nnunet_ext_trainers}"
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"

DATASET_ID="${DATASET_ID:-2}"
CONFIGURATION="${CONFIGURATION:-2d}"
FOLD="${FOLD:-${SLURM_ARRAY_TASK_ID:-0}}"
TRAINER="${TRAINER:-BoundEBOLogBarrierTrainer}"
LOSS="${LOSS:-bound_log_ebo}"
PLANS="${PLANS:-nnUNetPlans}"
DEVICE="${DEVICE:-cuda}"

export NNUNET_EBO_LOSS="$LOSS"
export LAMBDA_EBO_IN="${LAMBDA_EBO_IN:-0.1}"
export LAMBDA_EBO_CORR="${LAMBDA_EBO_CORR:-0.1}"
export LAMBDA_EBO_CEN_IN="${LAMBDA_EBO_CEN_IN:-0.1}"
export LAMBDA_EBO_OUT_IN="${LAMBDA_EBO_OUT_IN:-0.2}"
export LAMBDA_EBO_CEN_CORR="${LAMBDA_EBO_CEN_CORR:-0.05}"
export LAMBDA_EBO_OUT_CORR="${LAMBDA_EBO_OUT_CORR:-0.1}"
export BOUNDARY_K="${BOUNDARY_K:-1}"
export MARGIN_CORRECT="${MARGIN_CORRECT:--25}"
export MARGIN_MISS="${MARGIN_MISS:--5}"
export BARRIER_T="${BARRIER_T:-1.0}"
export BARRIER_T_GROWTH="${BARRIER_T_GROWTH:-1.0}"
export RHO="${RHO:-1.0}"

echo "Trainer: $TRAINER"
echo "Loss: $LOSS"
echo "Fold: $FOLD"
echo "nnUNet_extTrainer: $nnUNet_extTrainer"

nnUNetv2_train \
"$DATASET_ID" \
"$CONFIGURATION" \
"$FOLD" \
-tr "$TRAINER" \
-p "$PLANS" \
-device "$DEVICE"
