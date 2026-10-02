from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import font_manager
from scipy import stats


# ==========================================================
# PATHS
# ==========================================================

ROOT = Path(__file__).resolve().parent
ANALYSIS = ROOT / "analysis"
FIG = ROOT / "figures" / "synthetic_final"

FIG.mkdir(parents=True, exist_ok=True)


# ==========================================================
# FONT CONFIGURATION
# ==========================================================

def choose_serif_font():
    """
    Prefer Times New Roman.

    On ALICE, Times New Roman may not be installed, in which case
    use a close serif substitute. The actual font is printed when
    the script runs.
    """
    candidates = [
        "Times New Roman",
        "Times",
        "Nimbus Roman",
        "Liberation Serif",
        "DejaVu Serif",
    ]

    for name in candidates:
        try:
            font_manager.findfont(
                name,
                fallback_to_default=False
            )
            return name
        except ValueError:
            continue

    return "DejaVu Serif"


FIGURE_FONT = choose_serif_font()

print(f"Figure font: {FIGURE_FONT}")


plt.rcParams.update({
    "font.family": "serif",
    "font.serif": [FIGURE_FONT],
    "mathtext.fontset": "stix",

    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 10.5,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,

    "axes.linewidth": 0.8,

    "xtick.direction": "out",
    "ytick.direction": "out",

    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,

    "lines.linewidth": 1.5,
    "lines.markersize": 5,

    "figure.dpi": 150,
    "savefig.dpi": 300,

    # Embed editable TrueType fonts.
    "pdf.fonttype": 42,
    "ps.fonttype": 42,

    "savefig.facecolor": "white",
})


# ==========================================================
# DISPLAY LABELS AND STYLES
# ==========================================================

MODEL_LABEL = {
    "logistic": "Logistic regression",
    "random_forest": "Random forest",
}


MODEL_STYLE = {
    "logistic": {
        "marker": "o",
        "linestyle": "-",
    },
    "random_forest": {
        "marker": "s",
        "linestyle": "--",
    },
}


# ==========================================================
# GENERAL HELPERS
# ==========================================================

def save(fig, stem):
    """
    Save vector PDF and 300-dpi PNG.
    """
    fig.savefig(
        FIG / f"{stem}.pdf",
        bbox_inches="tight"
    )

    fig.savefig(
        FIG / f"{stem}.png",
        bbox_inches="tight",
        dpi=300
    )

    plt.close(fig)


def panel_label(ax, label):
    """
    Add journal-style panel label.
    """
    ax.text(
        -0.16,
        1.04,
        label,
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        va="top",
        ha="left",
        clip_on=False
    )


def summarise_across_repetitions(
    d,
    group_cols,
    value_col,
    lower_bound=None,
    upper_bound=None
):
    """
    Compute mean and 95% t confidence interval across independent
    repetitions.

    The input dataframe should already contain one value per
    repetition for each condition being summarised.
    """
    s = (
        d.groupby(group_cols)[value_col]
        .agg(
            mean="mean",
            std="std",
            count="count"
        )
        .reset_index()
    )

    s["se"] = (
        s["std"] /
        np.sqrt(s["count"])
    )

    s["crit"] = stats.t.ppf(
        0.975,
        s["count"] - 1
    )

    s["ci_half"] = (
        s["crit"] *
        s["se"]
    )

    s["ci_low"] = (
        s["mean"] -
        s["ci_half"]
    )

    s["ci_high"] = (
        s["mean"] +
        s["ci_half"]
    )

    if lower_bound is not None:
        s["ci_low"] = s["ci_low"].clip(
            lower=lower_bound
        )

    if upper_bound is not None:
        s["ci_high"] = s["ci_high"].clip(
            upper=upper_bound
        )

    s["yerr_low"] = (
        s["mean"] -
        s["ci_low"]
    )

    s["yerr_high"] = (
        s["ci_high"] -
        s["mean"]
    )

    return s


# ==========================================================
# TRAINING-RARITY SUMMARIES
# ==========================================================

def training_summary(metric, scenario, xvar):
    """
    Summarise SMOTE-vs-none oracle effects.

    Step 1:
        Average moderator-specific effects within each independent
        simulation repetition.

    Step 2:
        Calculate mean and 95% confidence interval across the
        30 independent repetitions.
    """
    d = pd.read_csv(
        ANALYSIS /
        f"synthetic_paired_delta_{metric}_full.csv"
    )

    d = d[
        (d["scenario"] == scenario) &
        (d["method"] == "smote")
    ].copy()

    # One value per repetition, model and rarity level.
    r = (
        d.groupby(
            ["rep", "model", xvar],
            as_index=False
        )["delta"]
        .mean()
    )

    return summarise_across_repetitions(
        r,
        ["model", xvar],
        "delta"
    )


# ==========================================================
# VALIDATION-RARITY SUMMARIES
# ==========================================================

def validation_repetition_summary(
    metric,
    scenario,
    xvar
):
    """
    Read row-level validation-vs-oracle comparisons and create
    repetition-level summaries for SMOTE.

    Returns:
        1. mean absolute estimation error
        2. practical sign-error rate among eligible oracle effects

    Each plotted point is then summarised across 30 independent
    repetitions.
    """
    d = pd.read_csv(
        ANALYSIS /
        f"validation_paired_detail_{metric}_full.csv"
    )

    d = d[
        (d["validation_scenario"] == scenario) &
        (d["method"] == "smote")
    ].copy()

    # ------------------------------------------------------
    # Mean absolute estimation error within each repetition
    # ------------------------------------------------------

    abs_rep = (
        d.groupby(
            ["rep", "model", xvar],
            as_index=False
        )["absolute_error"]
        .mean()
    )

    abs_summary = summarise_across_repetitions(
        abs_rep,
        ["model", xvar],
        "absolute_error",
        lower_bound=0
    )

    # ------------------------------------------------------
    # Practical sign-error rate within each repetition
    #
    # practical_sign_error is NaN for oracle effects that do
    # not exceed the configured practical-effect margin.
    # Pandas mean therefore evaluates the rate only among
    # eligible effects.
    # ------------------------------------------------------

    sign_rep = (
        d.groupby(
            ["rep", "model", xvar],
            as_index=False
        )["practical_sign_error"]
        .mean()
    )

    sign_rep = sign_rep.dropna(
        subset=["practical_sign_error"]
    )

    sign_summary = summarise_across_repetitions(
        sign_rep,
        ["model", xvar],
        "practical_sign_error",
        lower_bound=0,
        upper_bound=1
    )

    return abs_summary, sign_summary


# ==========================================================
# FIGURE 1
#
# A2:
# Fixed training prevalence = 1%
# Vary absolute minority training support
# ==========================================================

fig, axes = plt.subplots(
    1,
    2,
    figsize=(7.4, 3.25),
    constrained_layout=True
)

for j, (ax, metric, title) in enumerate(
    zip(
        axes,
        ["auc", "ap"],
        ["AUROC", "Average precision"]
    )
):
    s = training_summary(
        metric,
        "A2_fixed_ratio",
        "m_train"
    )

    for model in [
        "logistic",
        "random_forest"
    ]:
        g = (
            s[s["model"] == model]
            .sort_values("m_train")
        )

        style = MODEL_STYLE[model]

        yerr = np.vstack([
            g["yerr_low"].to_numpy(),
            g["yerr_high"].to_numpy()
        ])

        ax.errorbar(
            g["m_train"],
            g["mean"],
            yerr=yerr,
            marker=style["marker"],
            linestyle=style["linestyle"],
            capsize=3,
            linewidth=1.5,
            markersize=5,
            label=MODEL_LABEL[model]
        )

    ax.axhline(
        0,
        linewidth=0.8,
        linestyle=":",
        color="black",
        alpha=0.7,
        zorder=0
    )

    ax.set_xscale("log")

    ax.set_xticks(
        [10, 20, 50, 100, 200]
    )

    ax.set_xticklabels(
        ["10", "20", "50", "100", "200"]
    )

    ax.set_xlabel(
        r"Training minority count, $m_{\mathrm{train}}$"
    )

    ylabel = {
        "auc": r"SMOTE effect ($\Delta$AUROC)",
        "ap": r"SMOTE effect ($\Delta$AP)",
    }[metric]

    ax.set_ylabel(ylabel)

    ax.set_title(title)

    ax.grid(
        axis="y",
        alpha=0.20,
        linewidth=0.6
    )

    panel_label(
        ax,
        f"({chr(97 + j)})"
    )


axes[0].legend(
    frameon=False,
    loc="best"
)


save(
    fig,
    "figure1_A2_absolute_training_rarity"
)


# ==========================================================
# FIGURE 2
#
# A3:
# Fixed m_train = 50
# Vary relative class imbalance
# ==========================================================

fig, axes = plt.subplots(
    1,
    2,
    figsize=(7.4, 3.25),
    constrained_layout=True
)

for j, (ax, metric, title) in enumerate(
    zip(
        axes,
        ["auc", "ap"],
        ["AUROC", "Average precision"]
    )
):
    s = training_summary(
        metric,
        "A3_fixed_minority",
        "pi_train"
    )

    s["prevalence_pct"] = (
        100 *
        s["pi_train"]
    )

    for model in [
        "logistic",
        "random_forest"
    ]:
        g = (
            s[s["model"] == model]
            .sort_values("prevalence_pct")
        )

        style = MODEL_STYLE[model]

        yerr = np.vstack([
            g["yerr_low"].to_numpy(),
            g["yerr_high"].to_numpy()
        ])

        ax.errorbar(
            g["prevalence_pct"],
            g["mean"],
            yerr=yerr,
            marker=style["marker"],
            linestyle=style["linestyle"],
            capsize=3,
            linewidth=1.5,
            markersize=5,
            label=MODEL_LABEL[model]
        )

    ax.axhline(
        0,
        linewidth=0.8,
        linestyle=":",
        color="black",
        alpha=0.7,
        zorder=0
    )

    ax.set_xscale("log")

    ax.set_xticks(
        [0.5, 1, 2.5, 5, 10]
    )

    ax.set_xticklabels(
        ["0.5", "1", "2.5", "5", "10"]
    )

    ax.set_xlabel(
        "Training minority prevalence (%)"
    )

    ylabel = {
        "auc": r"SMOTE effect ($\Delta$AUROC)",
        "ap": r"SMOTE effect ($\Delta$AP)",
    }[metric]

    ax.set_ylabel(ylabel)

    ax.set_title(title)

    ax.grid(
        axis="y",
        alpha=0.20,
        linewidth=0.6
    )

    panel_label(
        ax,
        f"({chr(97 + j)})"
    )


axes[0].legend(
    frameon=False,
    loc="best"
)


save(
    fig,
    "figure2_A3_relative_training_imbalance"
)


# ==========================================================
# FIGURE 3
#
# V2:
# Fixed validation prevalence = 1%
# Vary absolute minority validation support
#
# 95% confidence intervals are calculated across
# independent simulation repetitions.
# ==========================================================

fig, axes = plt.subplots(
    2,
    2,
    figsize=(7.4, 5.7),
    constrained_layout=True
)

panel_letters = [
    ["(a)", "(b)"],
    ["(c)", "(d)"],
]


for row, metric, metric_title in [
    (0, "auc", "AUROC"),
    (1, "ap", "Average precision")
]:
    abs_s, sign_s = validation_repetition_summary(
        metric,
        "V2_fixed_ratio",
        "m_val"
    )

    for model in [
        "logistic",
        "random_forest"
    ]:
        style = MODEL_STYLE[model]

        # --------------------------------------------------
        # Absolute estimation error
        # --------------------------------------------------

        g = (
            abs_s[abs_s["model"] == model]
            .sort_values("m_val")
        )

        yerr = np.vstack([
            g["yerr_low"].to_numpy(),
            g["yerr_high"].to_numpy()
        ])

        axes[row, 0].errorbar(
            g["m_val"],
            g["mean"],
            yerr=yerr,
            marker=style["marker"],
            linestyle=style["linestyle"],
            capsize=3,
            linewidth=1.5,
            markersize=5,
            label=MODEL_LABEL[model]
        )

        # --------------------------------------------------
        # Practical sign error
        # --------------------------------------------------

        g = (
            sign_s[sign_s["model"] == model]
            .sort_values("m_val")
        )

        y = (
            100 *
            g["mean"].to_numpy()
        )

        yerr = np.vstack([
            100 * g["yerr_low"].to_numpy(),
            100 * g["yerr_high"].to_numpy()
        ])

        axes[row, 1].errorbar(
            g["m_val"],
            y,
            yerr=yerr,
            marker=style["marker"],
            linestyle=style["linestyle"],
            capsize=3,
            linewidth=1.5,
            markersize=5,
            label=MODEL_LABEL[model]
        )

    for col in [0, 1]:
        axes[row, col].set_xscale(
            "log"
        )

        axes[row, col].set_xticks(
            [5, 10, 20, 50, 100, 200]
        )

        axes[row, col].set_xticklabels(
            ["5", "10", "20", "50", "100", "200"]
        )

        axes[row, col].grid(
            axis="y",
            alpha=0.20,
            linewidth=0.6
        )

        panel_label(
            axes[row, col],
            panel_letters[row][col]
        )

    axes[row, 0].set_ylabel(
        f"{metric_title}\n"
        "mean absolute error"
    )

    axes[row, 1].set_ylabel(
        f"{metric_title}\n"
        "practical sign error (%)"
    )


axes[1, 0].set_xlabel(
    r"Validation minority count, $m_{\mathrm{val}}$"
)

axes[1, 1].set_xlabel(
    r"Validation minority count, $m_{\mathrm{val}}$"
)

axes[0, 0].set_title(
    "Estimation error"
)

axes[0, 1].set_title(
    "Practical sign error"
)

axes[0, 0].legend(
    frameon=False,
    loc="best"
)


save(
    fig,
    "figure3_V2_absolute_validation_rarity"
)


# ==========================================================
# FIGURE 4
#
# V3:
# Fixed m_val = 50
# Vary validation prevalence
#
# 95% confidence intervals are calculated across
# independent simulation repetitions.
# ==========================================================

fig, axes = plt.subplots(
    2,
    2,
    figsize=(7.4, 5.7),
    constrained_layout=True
)


for row, metric, metric_title in [
    (0, "auc", "AUROC"),
    (1, "ap", "Average precision")
]:
    abs_s, sign_s = validation_repetition_summary(
        metric,
        "V3_fixed_minority",
        "pi_val"
    )

    abs_s["prevalence_pct"] = (
        100 *
        abs_s["pi_val"]
    )

    sign_s["prevalence_pct"] = (
        100 *
        sign_s["pi_val"]
    )

    for model in [
        "logistic",
        "random_forest"
    ]:
        style = MODEL_STYLE[model]

        # --------------------------------------------------
        # Absolute estimation error
        # --------------------------------------------------

        g = (
            abs_s[abs_s["model"] == model]
            .sort_values("prevalence_pct")
        )

        yerr = np.vstack([
            g["yerr_low"].to_numpy(),
            g["yerr_high"].to_numpy()
        ])

        axes[row, 0].errorbar(
            g["prevalence_pct"],
            g["mean"],
            yerr=yerr,
            marker=style["marker"],
            linestyle=style["linestyle"],
            capsize=3,
            linewidth=1.5,
            markersize=5,
            label=MODEL_LABEL[model]
        )

        # --------------------------------------------------
        # Practical sign error
        # --------------------------------------------------

        g = (
            sign_s[sign_s["model"] == model]
            .sort_values("prevalence_pct")
        )

        y = (
            100 *
            g["mean"].to_numpy()
        )

        yerr = np.vstack([
            100 * g["yerr_low"].to_numpy(),
            100 * g["yerr_high"].to_numpy()
        ])

        axes[row, 1].errorbar(
            g["prevalence_pct"],
            y,
            yerr=yerr,
            marker=style["marker"],
            linestyle=style["linestyle"],
            capsize=3,
            linewidth=1.5,
            markersize=5,
            label=MODEL_LABEL[model]
        )

    for col in [0, 1]:
        axes[row, col].set_xscale(
            "log"
        )

        axes[row, col].set_xticks(
            [0.5, 1, 2.5, 5, 10]
        )

        axes[row, col].set_xticklabels(
            ["0.5", "1", "2.5", "5", "10"]
        )

        axes[row, col].grid(
            axis="y",
            alpha=0.20,
            linewidth=0.6
        )

        panel_label(
            axes[row, col],
            panel_letters[row][col]
        )

    axes[row, 0].set_ylabel(
        f"{metric_title}\n"
        "mean absolute error"
    )

    axes[row, 1].set_ylabel(
        f"{metric_title}\n"
        "practical sign error (%)"
    )


axes[1, 0].set_xlabel(
    "Validation minority prevalence (%)"
)

axes[1, 1].set_xlabel(
    "Validation minority prevalence (%)"
)

axes[0, 0].set_title(
    "Estimation error"
)

axes[0, 1].set_title(
    "Practical sign error"
)

axes[0, 0].legend(
    frameon=False,
    loc="best"
)


save(
    fig,
    "figure4_V3_relative_validation_imbalance"
)


# ==========================================================
# FINISHED
# ==========================================================

print("\nCreated publication figures:")

for f in sorted(FIG.glob("*")):
    print(" ", f)

print(
    "\nFigures 1-2: 95% CIs across 30 independent "
    "repetition-level mean effects."
)

print(
    "Figures 3-4: 95% CIs across 30 independent "
    "repetition-level validation summaries."
)

print(
    "Practical sign-error rates are conditional on oracle "
    "effects exceeding the configured practical margin "
    "(0.01 for AUROC; 0.02 for AP)."
)
