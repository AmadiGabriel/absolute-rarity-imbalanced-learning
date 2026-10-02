# Absolute Rarity in Imbalanced Learning: Distinct Roles in Training and Validation

Source code for the experiments and analyses reported in the paper.

This repository intentionally excludes large generated result CSVs, temporary
checkpoints, logs, and third-party datasets. The exact study configuration,
Python environment specification, canonical experiment/analysis code, and
ALICE/SLURM execution scripts are retained.

## Main implementation

- `study_common.py` — shared modelling, metrics, deterministic seeding and utilities.
- `prepare_data_v3.py` — final real-data preparation.
- `run_synthetic_final.py` — final synthetic experiment.
- `run_real_nested.py` — native-distribution repeated nested cross-validation.
- `run_real_subsample.py` — controlled real-data training-rarity experiment.
- `run_real_validation_rarity.py` — real-data validation-rarity experiment.
- `analyze_results_v2.py` — synthetic statistical analysis.
- `make_synthetic_figures.py` — paper-figure generation.
- `analysis/analyse_real_nested_corrected.py` — corrected repeated-CV inference.

The manuscript's principal performance metrics are AUC and average precision
(AP). Some implementation code retains ancillary metrics generated during the
wider analysis; these are not principal manuscript outcomes.

## Local/workstation execution

Use `requirements.txt` or `environment-alice.yml`, prepare the public datasets
with `prepare_data_v3.py`, then run the relevant experiment scripts. Full study
reproduction is computationally demanding.

## ALICE/HPC execution

The `*_alice.py`, `.slurm`, and ALICE shell scripts are the execution layer used
for the HPC run. They are kept at repository root because the original SLURM
scripts call these filenames using root-relative paths. This preserves the
release code without rewriting the launchers.

## Results and data

Large generated result CSVs are intentionally not tracked in normal Git history.
If archival result files are distributed, they should be attached to the
versioned DOI release rather than committed as ordinary Git source files.

Public real-world datasets should be obtained from their original repositories;
they are not redistributed here.
