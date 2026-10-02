import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import t as student_t

# ============================================================
# Configuration
# ============================================================

INPUT = Path("results/real_nested_full_all_datasets.csv")

OUT_INFERENCE = Path(
    "analysis/real_nested_corrected_inference_full.csv"
)

OUT_MEANS = Path(
    "analysis/real_nested_method_means_full.csv"
)

OUT_DELTAS = Path(
    "analysis/real_nested_paired_deltas_full.csv"
)

METRICS = ["auc", "ap"]

METHODS = [
    "ros",
    "smote",
    "borderline_smote",
    "adasyn",
    "class_weight",
]

# 5-fold CV repeated 5 times
K = 5
R = 5
N = K * R

# Outer split = 80% training, 20% test
TEST_TRAIN_RATIO = 0.20 / 0.80

ALPHA = 0.05


# ============================================================
# Helpers
# ============================================================

def holm_adjust(pvalues):
    """
    Holm family-wise error correction.
    Returns adjusted p-values in original order.
    """
    pvalues = np.asarray(pvalues, dtype=float)
    m = len(pvalues)

    order = np.argsort(pvalues)
    ranked = pvalues[order]

    adjusted_ranked = np.empty(m, dtype=float)

    running_max = 0.0

    for i, p in enumerate(ranked):
        adj = (m - i) * p
        running_max = max(running_max, adj)
        adjusted_ranked[i] = min(running_max, 1.0)

    adjusted = np.empty(m, dtype=float)
    adjusted[order] = adjusted_ranked

    return adjusted


def corrected_repeated_cv_test(d):
    """
    Bouckaert & Frank corrected repeated k-fold CV test.

    d = paired method-minus-none differences across the 25
        outer test folds.

    corrected variance:
        (1/(k*r) + n_test/n_train) * s^2
    """
    d = np.asarray(d, dtype=float)

    if len(d) != N:
        raise ValueError(
            f"Expected {N} paired differences, got {len(d)}"
        )

    mean_d = np.mean(d)
    sd_d = np.std(d, ddof=1)

    correction = (1.0 / N) + TEST_TRAIN_RATIO

    corrected_se = np.sqrt(correction * (sd_d ** 2))

    df = N - 1

    if corrected_se == 0:
        if mean_d == 0:
            t_stat = 0.0
            p_value = 1.0
        else:
            t_stat = np.sign(mean_d) * np.inf
            p_value = 0.0
    else:
        t_stat = mean_d / corrected_se
        p_value = 2.0 * student_t.sf(
            np.abs(t_stat),
            df=df
        )

    critical = student_t.ppf(
        1.0 - ALPHA / 2.0,
        df=df
    )

    ci_low = mean_d - critical * corrected_se
    ci_high = mean_d + critical * corrected_se

    return {
        "n_pairs": len(d),
        "mean_delta": mean_d,
        "sd_delta": sd_d,
        "correction_factor": correction,
        "corrected_se": corrected_se,
        "df": df,
        "t_corrected": t_stat,
        "p_raw": p_value,
        "ci95_low": ci_low,
        "ci95_high": ci_high,
    }


# ============================================================
# Load and validate
# ============================================================

df = pd.read_csv(INPUT)

print("=== INPUT AUDIT ===")
print("rows:", len(df))
print("datasets:", sorted(df["dataset"].unique()))
print("models:", sorted(df["model"].unique()))
print("methods:", sorted(df["method"].unique()))
print("outer indices:", df["outer_index"].nunique())

assert len(df) == 1800
assert df["dataset"].nunique() == 6
assert df["outer_index"].nunique() == 25
assert df["auc"].isna().sum() == 0
assert df["ap"].isna().sum() == 0

key = [
    "dataset",
    "outer_index",
    "model",
    "method",
]

assert df.duplicated(key).sum() == 0


# ============================================================
# Raw method means
# ============================================================

mean_rows = []

for (dataset, model, method), g in df.groupby(
    ["dataset", "model", "method"],
    sort=True
):
    row = {
        "dataset": dataset,
        "model": model,
        "method": method,
        "n_outer": len(g),
    }

    for metric in METRICS:
        row[f"mean_{metric}"] = g[metric].mean()
        row[f"sd_{metric}"] = g[metric].std(ddof=1)

    mean_rows.append(row)

means = pd.DataFrame(mean_rows)

means.to_csv(
    OUT_MEANS,
    index=False
)


# ============================================================
# Exact paired method-minus-none differences
# ============================================================

delta_rows = []

for dataset in sorted(df["dataset"].unique()):

    for model in sorted(df["model"].unique()):

        base = df[
            (df["dataset"] == dataset) &
            (df["model"] == model) &
            (df["method"] == "none")
        ].copy()

        assert len(base) == N

        for method in METHODS:

            aug = df[
                (df["dataset"] == dataset) &
                (df["model"] == model) &
                (df["method"] == method)
            ].copy()

            assert len(aug) == N

            merged = base.merge(
                aug,
                on=[
                    "dataset",
                    "outer_index",
                    "model",
                ],
                suffixes=("_none", "_method"),
                validate="one_to_one",
            )

            assert len(merged) == N

            for metric in METRICS:

                delta = (
                    merged[f"{metric}_method"]
                    - merged[f"{metric}_none"]
                )

                for oi, value in zip(
                    merged["outer_index"],
                    delta
                ):
                    delta_rows.append({
                        "dataset": dataset,
                        "model": model,
                        "method": method,
                        "metric": metric,
                        "outer_index": oi,
                        "delta": value,
                    })

deltas = pd.DataFrame(delta_rows)

deltas.to_csv(
    OUT_DELTAS,
    index=False
)

print("\nPaired delta rows:", len(deltas))

# 6 datasets x 2 models x 5 methods x 2 metrics x 25 folds
assert len(deltas) == 6 * 2 * 5 * 2 * 25


# ============================================================
# Corrected repeated-CV inference
# ============================================================

result_rows = []

for (
    dataset,
    model,
    metric,
    method
), g in deltas.groupby(
    ["dataset", "model", "metric", "method"],
    sort=True
):

    stats = corrected_repeated_cv_test(
        g.sort_values("outer_index")["delta"].values
    )

    result_rows.append({
        "dataset": dataset,
        "model": model,
        "metric": metric,
        "method": method,
        **stats,
    })

results = pd.DataFrame(result_rows)

# ============================================================
# Holm correction
#
# Family:
# within each dataset x model x metric,
# correct the five augmentation-vs-none comparisons.
# ============================================================

results["p_holm"] = np.nan

for keys, idx in results.groupby(
    ["dataset", "model", "metric"]
).groups.items():

    idx = list(idx)

    results.loc[idx, "p_holm"] = holm_adjust(
        results.loc[idx, "p_raw"].values
    )

results["supported_holm_005"] = (
    results["p_holm"] < 0.05
)

results["ci_excludes_zero"] = (
    (results["ci95_low"] > 0) |
    (results["ci95_high"] < 0)
)

results = results.sort_values(
    ["dataset", "model", "metric", "method"]
).reset_index(drop=True)

results.to_csv(
    OUT_INFERENCE,
    index=False
)


# ============================================================
# Final audit / concise output
# ============================================================

print("\n=== CORRECTED REPEATED-CV ANALYSIS ===")
print("comparisons:", len(results), "/ 120")
print(
    "correction factor:",
    (1 / N) + TEST_TRAIN_RATIO
)
print("degrees of freedom:", N - 1)

assert len(results) == 120
assert results["p_holm"].isna().sum() == 0

print("\nHolm-supported comparisons:")
supported = results[
    results["supported_holm_005"]
].copy()

if len(supported) == 0:
    print("None")
else:
    print(
        supported[
            [
                "dataset",
                "model",
                "metric",
                "method",
                "mean_delta",
                "ci95_low",
                "ci95_high",
                "p_holm",
            ]
        ].to_string(index=False)
    )

print("\n=== SMOTE ONLY ===")

smote = results[
    results["method"] == "smote"
][
    [
        "dataset",
        "model",
        "metric",
        "mean_delta",
        "ci95_low",
        "ci95_high",
        "p_raw",
        "p_holm",
        "supported_holm_005",
    ]
]

print(smote.to_string(index=False))

print("\nSaved:")
print(OUT_DELTAS)
print(OUT_MEANS)
print(OUT_INFERENCE)

print("\nCOMPLETE: True")
