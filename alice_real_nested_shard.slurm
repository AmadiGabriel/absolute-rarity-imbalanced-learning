#!/bin/bash
#SBATCH --job-name=bir_ncv_shard
#SBATCH --account=aerofaul
#SBATCH --partition=medium
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/bir_ncv_shard-%A_%a.out
#SBATCH --error=logs/bir_ncv_shard-%A_%a.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
source ./alice_activate_conda.sh

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

DATASET="${DATASET:?DATASET must be supplied}"

echo "========================================"
echo "Nested-CV outer-fold shard"
echo "Array job: ${SLURM_ARRAY_JOB_ID}"
echo "Task: ${SLURM_ARRAY_TASK_ID}"
echo "Dataset: ${DATASET}"
echo "Host: $(hostname)"
echo "Started: $(date)"
echo "CPUs: ${SLURM_CPUS_PER_TASK}"
echo "========================================"

python3 -u run_real_nested_shard.py \
    --profile full \
    --dataset "${DATASET}" \
    --outer-index "${SLURM_ARRAY_TASK_ID}" \
    --models logistic random_forest \
    --methods none ros smote borderline_smote adasyn class_weight

echo "========================================"
echo "Completed: $(date)"
echo "========================================"
