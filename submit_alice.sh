#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs results

source ./alice_activate_conda.sh
python3 prepare_synthetic_alice.py --profile full

ARRAY_JOB=$(sbatch --parsable alice_synthetic_array.slurm)
FINAL_JOB=$(sbatch --parsable --dependency=afterok:${ARRAY_JOB} alice_synthetic_finalize.slurm)

echo "Synthetic array submitted: ${ARRAY_JOB}"
echo "Finalizer submitted:       ${FINAL_JOB}"
echo
 echo "Monitor with: squeue -j ${ARRAY_JOB}"
echo "Cancel entire array with: scancel ${ARRAY_JOB}"
echo "Finalizer will run only if every array task finishes successfully."
