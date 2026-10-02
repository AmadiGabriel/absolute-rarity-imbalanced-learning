#!/bin/bash
#SBATCH --job-name=bir_analyse
#SBATCH --account=aerofaul
#SBATCH --partition=medium
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/bir_analyse-%j.out
#SBATCH --error=logs/bir_analyse-%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

source ./alice_activate_conda.sh

echo "========================================"
echo "Beyond Imbalance Ratio: full analysis"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Host: $(hostname)"
echo "Started: $(date)"
echo "CPUs: ${SLURM_CPUS_PER_TASK}"
echo "========================================"

python3 analyze_results_v2.py --profile full

echo "========================================"
echo "Analysis completed: $(date)"
echo "========================================"
