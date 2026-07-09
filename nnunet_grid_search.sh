#!/bin/bash
#SBATCH --time=20:00:00
#SBATCH --array=0
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --output=/scratch/guibo/slurm_outputs/slurm-%j.out
#SBATCH --error=/scratch/guibo/slurm_outputs/slurm-%j.err
#SBATCH --account=rrg-josedolz
#SBATCH --partition=compute

set -euo pipefail

module load python/3.10.13
source ~/nnunet/bin/activate

ROOT_DIR="${ROOT_DIR:-/home/guibo/links/projects/rrg-josedolz/guibo/ebo_seg}"
DATASET_ID="${DATASET_ID:-2}"
CONFIGURATION="${CONFIGURATION:-2d}"
FOLD="${FOLD:-0}"
PLANS="${PLANS:-nnUNetPlans}"
LOSS="${LOSS:-cedice}"
CONTINUE="${CONTINUE:-0}"
DEVICE="${DEVICE:-cuda}"
NUM_GPUS="${NUM_GPUS:-}"
NNUNET_FPR95_MAX_PIXELS="${NNUNET_FPR95_MAX_PIXELS:-200000}"
NNUNET_FPR95_MAX_PIXELS_PER_BATCH="${NNUNET_FPR95_MAX_PIXELS_PER_BATCH:-50000}"

export nnUNet_raw="${nnUNet_raw:-/scratch/$USER/nnUNet_raw}"
export nnUNet_preprocessed="${nnUNet_preprocessed:-/scratch/$USER/nnUNet_preprocessed}"
NNUNET_RESULTS_BASE="${NNUNET_RESULTS_BASE:-/scratch/$USER/nnUNet_results_grid_search/BraTS}"
export nnUNet_compile="${nnUNet_compile:-False}"
export nnUNet_extTrainer="${nnUNet_extTrainer:-$ROOT_DIR/nnunet_ext_trainers}"
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"
export NNUNET_FPR95_MAX_PIXELS
export NNUNET_FPR95_MAX_PIXELS_PER_BATCH
SELECT_BEST="${SELECT_BEST:-1}"
BEST_OUTPUT_JSON="${BEST_OUTPUT_JSON:-$NNUNET_RESULTS_BASE/nnunet_grid_search_best.json}"

if [ "$SELECT_BEST" = "1" ]; then
    cd "$ROOT_DIR"
    python select_nnunet_grid_best.py \
        --results-base "$NNUNET_RESULTS_BASE" \
        --output-json "$BEST_OUTPUT_JSON" \
        --fold "$FOLD"
    exit 0
fi

if [[ "$FOLD" == "all" || "$FOLD" == *","* ]]; then
    echo "ERROR: nnunet_grid_search.sh trains exactly one fold. Set FOLD to a single integer, for example FOLD=0."
    exit 1
fi

LAMBDA_EBO_IN_GRID="${LAMBDA_EBO_IN_GRID:-0.5 2 5}"
LAMBDA_EBO_CORR_GRID="${LAMBDA_EBO_CORR_GRID:-0.1}"
LAMBDA_EBO_CEN_IN_GRID="${LAMBDA_EBO_CEN_IN_GRID:-0.1 0.5 1}"
LAMBDA_EBO_OUT_IN_GRID="${LAMBDA_EBO_OUT_IN_GRID:-2 5}"
LAMBDA_EBO_CEN_CORR_GRID="${LAMBDA_EBO_CEN_CORR_GRID:-0.1}"
LAMBDA_EBO_OUT_CORR_GRID="${LAMBDA_EBO_OUT_CORR_GRID:-0.1 0.5}"
BOUNDARY_K_GRID="${BOUNDARY_K_GRID:-3}"
MARGIN_CORRECT_GRID="${MARGIN_CORRECT_GRID:--17 -20}"
MARGIN_MISS_GRID="${MARGIN_MISS_GRID:--5 -3}"
BARRIER_T_GRID="${BARRIER_T_GRID:-1.0}"
BARRIER_T_GROWTH_GRID="${BARRIER_T_GROWTH_GRID:-${BARRIER_T_GROWTH:-1.005}}"
RHO_GRID="${RHO_GRID:-1.0 0.1 0.5 4}"

read -r -a lambda_ebo_in_values <<< "$LAMBDA_EBO_IN_GRID"
read -r -a lambda_ebo_corr_values <<< "$LAMBDA_EBO_CORR_GRID"
read -r -a lambda_ebo_cen_in_values <<< "$LAMBDA_EBO_CEN_IN_GRID"
read -r -a lambda_ebo_out_in_values <<< "$LAMBDA_EBO_OUT_IN_GRID"
read -r -a lambda_ebo_cen_corr_values <<< "$LAMBDA_EBO_CEN_CORR_GRID"
read -r -a lambda_ebo_out_corr_values <<< "$LAMBDA_EBO_OUT_CORR_GRID"
read -r -a boundary_k_values <<< "$BOUNDARY_K_GRID"
read -r -a margin_correct_values <<< "$MARGIN_CORRECT_GRID"
read -r -a margin_miss_values <<< "$MARGIN_MISS_GRID"
read -r -a barrier_t_values <<< "$BARRIER_T_GRID"
read -r -a barrier_t_growth_values <<< "$BARRIER_T_GROWTH_GRID"
read -r -a rho_values <<< "$RHO_GRID"

sanitize() {
    local value="$1"
    value="${value//- /m}"
    value="${value//-/m}"
    value="${value//./p}"
    echo "$value"
}

trainer_for_loss() {
    case "$1" in
        ebo_ce|ebo_cross_entropy) echo "EBOTrainer" ;;
        log_ebo) echo "EBOLossLogBarrierTrainer" ;;
        bound_ebo|bound_ebo_ce|bound_ebo_cross_entropy|boundary_ebo_ce) echo "BoundEBOTrainer" ;;
        bound_log_ebo|bound_ebo_log_barrier|boundary_log_ebo) echo "BoundEBOLogBarrierTrainer" ;;
        cedice|ce_dice) echo "CEDiceTrainer" ;;
        *) echo "ERROR: unsupported nnU-Net EBO loss '$1'" >&2; return 1 ;;
    esac
}

configs=()
case "$LOSS" in
    cedice|ce_dice)
        configs+=("|||||||||||")
        ;;
    ebo_ce|ebo_cross_entropy)
        for lin in "${lambda_ebo_in_values[@]}"; do
        for lcorr in "${lambda_ebo_corr_values[@]}"; do
        for mcorr in "${margin_correct_values[@]}"; do
        for mmiss in "${margin_miss_values[@]}"; do
            configs+=("$lin|$lcorr|||||$mcorr|$mmiss|||")
        done; done; done; done
        ;;
    log_ebo)
        for lin in "${lambda_ebo_in_values[@]}"; do
        for lcorr in "${lambda_ebo_corr_values[@]}"; do
        for mcorr in "${margin_correct_values[@]}"; do
        for mmiss in "${margin_miss_values[@]}"; do
        for bt in "${barrier_t_values[@]}"; do
        for btg in "${barrier_t_growth_values[@]}"; do
            configs+=("$lin|$lcorr|||||$mcorr|$mmiss|$bt|$btg|")
        done; done; done; done; done; done
        ;;
    bound_ebo|bound_ebo_ce|bound_ebo_cross_entropy|boundary_ebo_ce)
        for cenin in "${lambda_ebo_cen_in_values[@]}"; do
        for outin in "${lambda_ebo_out_in_values[@]}"; do
        for cencorr in "${lambda_ebo_cen_corr_values[@]}"; do
        for outcorr in "${lambda_ebo_out_corr_values[@]}"; do
        for bk in "${boundary_k_values[@]}"; do
        for mcorr in "${margin_correct_values[@]}"; do
        for mmiss in "${margin_miss_values[@]}"; do
            configs+=("||$cenin|$outin|$cencorr|$outcorr|$bk|$mcorr|$mmiss|||")
        done; done; done; done; done; done; done
        ;;
    bound_log_ebo|bound_ebo_log_barrier|boundary_log_ebo)
        for cenin in "${lambda_ebo_cen_in_values[@]}"; do
        for outin in "${lambda_ebo_out_in_values[@]}"; do
        for cencorr in "${lambda_ebo_cen_corr_values[@]}"; do
        for outcorr in "${lambda_ebo_out_corr_values[@]}"; do
        for bk in "${boundary_k_values[@]}"; do
        for mcorr in "${margin_correct_values[@]}"; do
        for mmiss in "${margin_miss_values[@]}"; do
        for bt in "${barrier_t_values[@]}"; do
        for btg in "${barrier_t_growth_values[@]}"; do
            configs+=("||$cenin|$outin|$cencorr|$outcorr|$bk|$mcorr|$mmiss|$bt|$btg|")
        done; done; done; done; done; done; done; done; done
        ;;
    *)
        echo "ERROR: unsupported nnU-Net EBO loss '$LOSS'."
        exit 1
        ;;
esac

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
TOTAL="${#configs[@]}"
if [ "$TASK_ID" -ge "$TOTAL" ]; then
    echo "Task $TASK_ID outside grid with $TOTAL combinations. Nothing to do."
    exit 0
fi

IFS='|' read -r LAMBDA_EBO_IN LAMBDA_EBO_CORR LAMBDA_EBO_CEN_IN LAMBDA_EBO_OUT_IN LAMBDA_EBO_CEN_CORR LAMBDA_EBO_OUT_CORR BOUNDARY_K MARGIN_CORRECT MARGIN_MISS BARRIER_T BARRIER_T_GROWTH RHO <<< "${configs[$TASK_ID]}"

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

TRAINER="${TRAINER:-$(trainer_for_loss "$LOSS")}"
TRIAL_NAME="trial_$(printf '%03d' "$TASK_ID")_fold_${FOLD}_${LOSS}_cenin_$(sanitize "$LAMBDA_EBO_CEN_IN")_outin_$(sanitize "$LAMBDA_EBO_OUT_IN")_cencorr_$(sanitize "$LAMBDA_EBO_CEN_CORR")_outcorr_$(sanitize "$LAMBDA_EBO_OUT_CORR")_lin_$(sanitize "$LAMBDA_EBO_IN")_lcorr_$(sanitize "$LAMBDA_EBO_CORR")_bk_${BOUNDARY_K}_mcorr_$(sanitize "$MARGIN_CORRECT")_mmiss_$(sanitize "$MARGIN_MISS")_t_$(sanitize "$BARRIER_T")_tg_$(sanitize "$BARRIER_T_GROWTH")"
export nnUNet_results="$NNUNET_RESULTS_BASE/$TRIAL_NAME"
mkdir -p "$nnUNet_results"

cd "$ROOT_DIR"

python -c "from nnunet_split_utils import sync_split_files_by_id; sync_split_files_by_id($DATASET_ID)"

echo "SLURM_JOB_ID=${SLURM_JOB_ID:-local}"
echo "Task           : $TASK_ID / $TOTAL"
echo "Dataset ID     : $DATASET_ID"
echo "Configuration  : $CONFIGURATION"
echo "Fold           : $FOLD"
echo "Fold mode      : single fold only"
echo "Trainer        : $TRAINER"
echo "Loss           : $LOSS"
echo "nnUNet_results : $nnUNet_results"
echo "FPR95 pixels   : max=$NNUNET_FPR95_MAX_PIXELS per_batch=$NNUNET_FPR95_MAX_PIXELS_PER_BATCH"
echo "Params         : lin=$LAMBDA_EBO_IN lcorr=$LAMBDA_EBO_CORR cenin=$LAMBDA_EBO_CEN_IN outin=$LAMBDA_EBO_OUT_IN cencorr=$LAMBDA_EBO_CEN_CORR outcorr=$LAMBDA_EBO_OUT_CORR bk=$BOUNDARY_K mcorr=$MARGIN_CORRECT mmiss=$MARGIN_MISS t=$BARRIER_T tg=$BARRIER_T_GROWTH rho=$RHO"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

cmd=(nnUNetv2_train "$DATASET_ID" "$CONFIGURATION" "$FOLD" -tr "$TRAINER" -p "$PLANS" -device "$DEVICE")
if [ -n "$NUM_GPUS" ]; then
    cmd+=(-num_gpus "$NUM_GPUS")
fi
if [ "$CONTINUE" = "1" ]; then
    cmd+=(--c)
fi

"${cmd[@]}"
