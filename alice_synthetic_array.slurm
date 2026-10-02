#!/bin/bash
#SBATCH --job-name=bir_synth
#SBATCH --partition=medium
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=2-00:00:00
#SBATCH --array=1-30%8
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --export=NONE
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

# ALICE documents this activation form for non-interactive Conda jobs.
source ./alice_activate_conda.sh

# Keep numerical libraries from spawning hidden thread pools. Random Forest
# parallelism is controlled by SLURM_CPUS_PER_TASK in study_common_alice.py.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export PYTHONUNBUFFERED=1

python3 run_synthetic_alice.py --profile full --rep "$SLURM_ARRAY_TASK_ID"
