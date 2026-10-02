#!/bin/bash
#SBATCH --job-name=bir_real_sub
#SBATCH --account=aerofaul
#SBATCH --partition=medium
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/bir_real_sub-%j.out
#SBATCH --error=logs/bir_real_sub-%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
source ./alice_activate_conda.sh

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

PROFILE="${PROFILE:-pilot}"
DATASET="${DATASET:-mammography}"

echo "========================================"
echo "Beyond Imbalance Ratio: training rarity"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Host: $(hostname)"
echo "Started: $(date)"
echo "Profile: ${PROFILE}"
echo "Dataset: ${DATASET}"
echo "CPUs: ${SLURM_CPUS_PER_TASK}"
echo "========================================"

python3 run_real_subsample.py \
    --profile "${PROFILE}" \
    --datasets "${DATASET}" \
    --models logistic random_forest \
    --methods none ros smote borderline_smote adasyn class_weight

echo "========================================"
echo "Completed: $(date)"
echo "========================================"
