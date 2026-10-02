#!/bin/bash
# Source this file from ALICE scripts. It supports either Miniconda or Miniforge.
set -euo pipefail

if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniforge3/etc/profile.d/conda.sh"
elif command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
else
    echo "ERROR: Could not find Conda under ~/miniconda3 or ~/miniforge3, and 'conda' is not on PATH." >&2
    exit 2
fi

conda activate imbalance-rarity
