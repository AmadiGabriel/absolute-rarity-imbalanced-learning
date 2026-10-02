#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"

if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniforge3/etc/profile.d/conda.sh"
elif command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
else
    echo "ERROR: Conda not found. Expected ~/miniconda3 or ~/miniforge3." >&2
    exit 2
fi

if conda env list | awk '{print $1}' | grep -qx "imbalance-rarity"; then
    echo "Conda environment 'imbalance-rarity' already exists."
    echo "Updating it from environment-alice.yml..."
    conda env update -n imbalance-rarity -f environment-alice.yml --prune
else
    echo "Creating Conda environment 'imbalance-rarity'..."
    conda env create -f environment-alice.yml
fi

conda activate imbalance-rarity

echo
echo "Environment ready:"
python3 - <<'PY'
import sys, numpy, pandas, scipy, sklearn, imblearn
print("Python:", sys.version.split()[0])
print("NumPy:", numpy.__version__)
print("pandas:", pandas.__version__)
print("SciPy:", scipy.__version__)
print("scikit-learn:", sklearn.__version__)
print("imbalanced-learn:", imblearn.__version__)
PY
