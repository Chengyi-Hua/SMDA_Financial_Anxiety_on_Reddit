# visualize_pragmatism_story_figures.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


# ============================================================
# 1. CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class VizConfig:
    project_dir: Path = Path(
        r"D:\Users\cheng\Documents\GitHub\SMDA_Financial_Anxiety_on_Reddit"
    )

    @property
    def prag_output_dir(self) -> Path:
        return self.project_dir / "pragmatism_analysis" / "outputs"

    @property
    def figure_dir(self) -> Path:
        return self.prag_output_dir / "figures"

    @property
    def figure_data_dir(self) -> Path:
        return self.figure_dir / "figure_data"


CFG = VizConfig()
CFG.figure_dir.mkdir(parents=True, exist_ok=True)
CFG.figure_data_dir.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. LABELS AND CONSTANTS
# ============================================================

COMMUNITY_ORDER = ["finanzen", "personalfinance"]
YEAR_ORDER = [2020, 2025]

COMMUNITY_LABELS = {
    "finanzen": "finanzen",
    "personalfinance": "personalfinance",
}

CATEGORY_ORDER = [
    "policy_regulation",
    "macro_economic_context",
    "financial_institutions_products",
    "investment_strategy",
    "planning_budgeting_control",
    "risk_uncertainty_management",
    "practical_problem_solving",
    "opportunity_action_orientation",
]

CATEGORY_LABELS = {
    "policy_regulation": "Policy / regulation",
    "macro_economic_context": "Macro-economic context",
    "financial_institutions_products": "Financial institutions / products",
    "investment_strategy": "Investment strategy",
    "planning_budgeting_control": "Planning / budgeting / control",
    "risk_uncertainty_management": "Risk / uncertainty management",
    "practical_problem_solving": "Practical problem-solving",
    "opportunity_action_orientation": "Opportunity / action orientation",
}

DOMINANT_CATEGORY_ORDER = [
    "financial_institutions_products",
    "investment_strategy",
    "macro_economic_context",
    "planning_budgeting_control",
    "policy_regulation",
    "risk_uncertainty_management",
    "practical_problem_solving",
    "opportunity_action_orientation",
    "none",
]

DOMINANT_CATEGORY_LABELS = {
    **CATEGORY_LABELS,
    "none": "No structural pragmatism",
}

plt.rcParams.update(
    {
        "figure.figsize": (10, 6),
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
    }
)


# ============================================================
# 3. HELPERS
# ============================================================

def require_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return path


def read_csv_required(filename: str) -> pd.DataFrame:
    path = require_file(CFG.prag_output_dir / filename)
    return pd.read_csv(path)


def require_columns(df: pd.DataFrame, columns: list[str], source_name: str) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(
            f"{source_name} is missing required columns: {missing}"
        )


def save_figure(fig: plt.Figure, filename: str) -> None:
    output_path = CFG.figure_dir / filename
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {output_path}")


def save_figure_data(df: pd.DataFrame, filename: str) -> None:
    output_path = CFG.figure_data_dir / filename
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"Saved figure data: {output_path}")


def percent_label(value: float, decimals: int = 1) -> str:
    return f"{value * 100:.{decimals}f}%"


def pp_label(value: float, decimals: int = 1) -> str:
    return f"{value * 100:+.{decimals}f} pp"


def add_bar_labels(ax, bars, decimals: int = 2, percent: bool = False) -> None:
    for bar in bars:
        height = bar.get_height()
        if np.isnan(height):
            continue

        label = (
            f"{height * 100:.1f}%"
            if percent
            else f"{height:.{decimals}f}"
        )

        ax.annotate(
            label,
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )


def clean_category_var_name(variable: str) -> str:
    return (
        variable
        .replace("prag_", "")
        .replace("_has", "")
        .replace("_rate_per_100_words", "")
    )


# ============================================================
# 4. LOAD ONLY THE INPUTS NEEDED FOR THE STORY FIGURES
# ============================================================

def load_story_inputs() -> dict[str, pd.DataFrame]:
    data = {}

    data["balanced_sample"] = read_csv_required(
        "pragmatism_balanced_sample.csv"
    )

    data["balanced_summary"] = read_csv_required(
        "pragmatism_group_summary_balanced.csv"
    )

    data["balanced_categorical_tests"] = read_csv_required(
        "pragmatism_categorical_statistical_tests_balanced_sample.csv"
    )

    return data


# ============================================================
# 5. SHARED CATEGORY TABLES
# ============================================================

def category_presence_table(sample: pd.DataFrame, year: int) -> pd.DataFrame:
    require_columns(
        sample,
        ["community", "year"],
        "pragmatism_balanced_sample.csv",
    )

    subset = sample.copy()
    subset["year"] = pd.to_numeric(subset["year"], errors="coerce").astype("Int64")
    subset = subset[subset["year"].eq(year)].copy()

    rows = []

    for category in CATEGORY_ORDER:
        col = f"prag_{category}_has"

        if col not in subset.columns:
            print(f"Warning: column not found and skipped: {col}")
            continue

        for community in COMMUNITY_ORDER:
            group = subset[subset["community"] == community]

            rows.append(
                {
                    "year": year,
                    "community": community,
                    "category": category,
                    "category_label": CATEGORY_LABELS[category],
                    "share": float(group[col].mean()) if len(group) else np.nan,
                    "n_posts": int(len(group)),
                }
            )

    out = pd.DataFrame(rows)

    pivot = (
        out.pivot_table(
            index=["year", "category", "category_label"],
            columns="community",
            values="share",
        )
        .reset_index()
    )

    for community in COMMUNITY_ORDER:
        if community not in pivot.columns:
            pivot[community] = np.nan

    pivot["difference_finanzen_minus_personalfinance"] = (
        pivot["finanzen"] - pivot["personalfinance"]
    )

    return pivot


def make_dominant_category_table(sample: pd.DataFrame) -> pd.DataFrame:
    require_columns(
        sample,
        ["community", "year", "dominant_prag_category"],
        "pragmatism_balanced_sample.csv",
    )

    df = sample.copy()
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["group"] = df["community"] + " " + df["year"].astype(str)

    counts = pd.crosstab(df["group"], df["dominant_prag_category"])

    for category in DOMINANT_CATEGORY_ORDER:
        if category not in counts.columns:
            counts[category] = 0

    group_order = [
        "finanzen 2020",
        "finanzen 2025",
        "personalfinance 2020",
        "personalfinance 2025",
    ]

    counts = counts.reindex(group_order, fill_value=0)
    counts = counts[DOMINANT_CATEGORY_ORDER]

    shares = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0)

    long = (
        shares.reset_index()
        .melt(
            id_vars="group",
            var_name="dominant_category",
            value_name="share",
        )
    )

    long["dominant_category_label"] = long["dominant_category"].map(
        DOMINANT_CATEGORY_LABELS
    )

    return long


# ============================================================
# 6. FIGURE 1: OVERALL STRUCTURAL PRAGMATISM INDEX
# ============================================================

def plot_overall_index(summary: pd.DataFrame) -> None:
    require_columns(
        summary,
        ["community", "year", "mean_structural_pragmatism_index"],
        "pragmatism_group_summary_balanced.csv",
    )

    df = summary.copy()
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")

    value_col = "mean_structural_pragmatism_index"

    fig, ax = plt.subplots(figsize=(9, 5.5))

    x = np.arange(len(YEAR_ORDER))
    width = 0.35

    for i, community in enumerate(COMMUNITY_ORDER):
        values = []

        for year in YEAR_ORDER:
            row = df[(df["community"] == community) & (df["year"] == year)]
            values.append(float(row[value_col].iloc[0]) if not row.empty else np.nan)

        offset = (i - 0.5) * width

        bars = ax.bar(
            x + offset,
            values,
            width,
            label=COMMUNITY_LABELS[community],
        )

        add_bar_labels(ax, bars, decimals=2, percent=False)

    ax.set_title("Overall structural pragmatism index")
    ax.set_ylabel("Mean structural pragmatism index")
    ax.set_xticks(x)
    ax.set_xticklabels([str(y) for y in YEAR_ORDER])
    ax.legend(title="Community")
    ax.grid(axis="y", alpha=0.25)

    data_cols = [
        "community",
        "year",
        "mean_structural_pragmatism_index",
        "median_structural_pragmatism_index",
        "structural_pragmatism_index_ci_low",
        "structural_pragmatism_index_ci_high",
    ]

    fig_data = df[[col for col in data_cols if col in df.columns]].copy()

    save_figure_data(fig_data, "01_overall_structural_pragmatism_index.csv")
    save_figure(fig, "01_overall_structural_pragmatism_index.png")


# ============================================================
# 7. FIGURE 3: CATEGORY PROFILE, 2025
# ============================================================

def plot_category_profile_2025(sample: pd.DataFrame) -> None:
    df = category_presence_table(sample, year=2025)

    df = df.sort_values(
        "difference_finanzen_minus_personalfinance",
        ascending=True,
    )

    fig, ax = plt.subplots(figsize=(11, 7))

    y = np.arange(len(df))
    height = 0.38

    ax.barh(
        y - height / 2,
        df["personalfinance"],
        height,
        label="personalfinance",
    )

    ax.barh(
        y + height / 2,
        df["finanzen"],
        height,
        label="finanzen",
    )

    ax.set_title("Category profile of structural pragmatism, 2025")
    ax.set_xlabel("Share of posts containing category")
    ax.set_yticks(y)
    ax.set_yticklabels(df["category_label"])
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.legend(title="Community")
    ax.grid(axis="x", alpha=0.25)

    max_share = np.nanmax(df[["finanzen", "personalfinance"]].to_numpy())
    ax.set_xlim(0, min(1.0, max_share + 0.15))

    for pos, row in enumerate(df.itertuples(index=False)):
        if not np.isnan(row.personalfinance):
            ax.text(
                row.personalfinance + 0.01,
                pos - height / 2,
                percent_label(row.personalfinance),
                va="center",
                fontsize=8,
            )

        if not np.isnan(row.finanzen):
            ax.text(
                row.finanzen + 0.01,
                pos + height / 2,
                percent_label(row.finanzen),
                va="center",
                fontsize=8,
            )

    save_figure_data(df, "03_category_profile_2025.csv")
    save_figure(fig, "03_category_profile_2025.png")


# ============================================================
# 8. FIGURE 4: CATEGORY DIFFERENCES, 2025
# ============================================================

def plot_category_difference_2025(sample: pd.DataFrame) -> None:
    df = category_presence_table(sample, year=2025)

    df = df.sort_values(
        "difference_finanzen_minus_personalfinance",
        ascending=True,
    )

    fig, ax = plt.subplots(figsize=(10, 6.5))

    y = np.arange(len(df))
    diff = df["difference_finanzen_minus_personalfinance"]

    bars = ax.barh(y, diff)

    ax.axvline(0, linewidth=1)
    ax.set_title("Category differences in structural pragmatism, 2025")
    ax.set_xlabel("Difference in share of posts: finanzen minus personalfinance")
    ax.set_yticks(y)
    ax.set_yticklabels(df["category_label"])
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.grid(axis="x", alpha=0.25)

    max_abs = max(float(np.nanmax(np.abs(diff))), 0.01)
    ax.set_xlim(-max_abs * 1.35, max_abs * 1.35)

    for bar, value in zip(bars, diff):
        if np.isnan(value):
            continue

        x = bar.get_width()
        label_x = x + 0.01 if x >= 0 else x - 0.01
        ha = "left" if x >= 0 else "right"

        ax.text(
            label_x,
            bar.get_y() + bar.get_height() / 2,
            pp_label(value),
            va="center",
            ha=ha,
            fontsize=9,
        )

    ax.text(
        0.99,
        0.02,
        "Positive = higher in finanzen\nNegative = higher in personalfinance",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        bbox=dict(boxstyle="round", alpha=0.08),
    )

    save_figure_data(df, "04_category_differences_2025.csv")
    save_figure(fig, "04_category_differences_2025.png")


# ============================================================
# 9. FIGURE 5: DOMINANT CATEGORY STACKED BARS
# ============================================================

def plot_dominant_category_stacked(sample: pd.DataFrame) -> None:
    long = make_dominant_category_table(sample)

    pivot = long.pivot_table(
        index="group",
        columns="dominant_category",
        values="share",
        fill_value=0,
    )

    group_order = [
        "finanzen 2020",
        "finanzen 2025",
        "personalfinance 2020",
        "personalfinance 2025",
    ]

    pivot = pivot.reindex(group_order)
    pivot = pivot[DOMINANT_CATEGORY_ORDER]

    fig, ax = plt.subplots(figsize=(12, 6.5))

    left = np.zeros(len(pivot))
    y = np.arange(len(pivot))

    cmap = plt.get_cmap("tab20")
    colors = [
        cmap(i)
        for i in np.linspace(0, 1, len(DOMINANT_CATEGORY_ORDER))
    ]

    for category, color in zip(DOMINANT_CATEGORY_ORDER, colors):
        values = pivot[category].to_numpy()

        ax.barh(
            y,
            values,
            left=left,
            label=DOMINANT_CATEGORY_LABELS[category],
            color=color,
        )

        left += values

    ax.set_title("Dominant structural pragmatism category")
    ax.set_xlabel("Share of posts")
    ax.set_yticks(y)
    ax.set_yticklabels(pivot.index)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.grid(axis="x", alpha=0.25)

    ax.legend(
        title="Dominant category",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        borderaxespad=0,
    )

    save_figure_data(long, "05_dominant_category_stacked.csv")
    save_figure(fig, "05_dominant_category_stacked.png")


# ============================================================
# 10. FIGURE 8: CATEGORY EFFECT SIZES, 2025
# ============================================================

def plot_category_effects_2025(categorical_tests: pd.DataFrame) -> None:
    require_columns(
        categorical_tests,
        [
            "comparison_type",
            "categorical_variable",
            "cramers_v",
            "share_difference_1_minus_2",
        ],
        "pragmatism_categorical_statistical_tests_balanced_sample.csv",
    )

    df = categorical_tests.copy()

    target = "between_communities_finanzen_minus_personalfinance_2025"

    keep_vars = [
        f"prag_{category}_has"
        for category in CATEGORY_ORDER
    ]

    df = df[
        (df["comparison_type"] == target)
        & (df["categorical_variable"].isin(keep_vars))
    ].copy()

    df["category"] = df["categorical_variable"].map(clean_category_var_name)
    df["category_label"] = df["category"].map(CATEGORY_LABELS)

    df = df.sort_values("cramers_v", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6.5))

    y = np.arange(len(df))
    values = df["cramers_v"]

    bars = ax.barh(y, values)

    ax.set_title("Strength of category differences, 2025")
    ax.set_xlabel("Cramer's V")
    ax.set_yticks(y)
    ax.set_yticklabels(df["category_label"])
    ax.grid(axis="x", alpha=0.25)

    max_v = max(float(np.nanmax(values)), 0.01)
    ax.set_xlim(0, max_v * 1.45)

    for bar, v, diff in zip(
        bars,
        df["cramers_v"],
        df["share_difference_1_minus_2"],
    ):
        ax.text(
            v + max_v * 0.03,
            bar.get_y() + bar.get_height() / 2,
            f"V={v:.3f}, diff={pp_label(diff)}",
            va="center",
            fontsize=9,
        )

    save_figure_data(df, "08_category_effects_2025.csv")
    save_figure(fig, "08_category_effects_2025.png")


# ============================================================
# 11. COMPACT STORY KEY-NUMBERS TABLE
# ============================================================

def make_story_key_number_table(
    summary: pd.DataFrame,
    sample: pd.DataFrame,
    categorical_tests: pd.DataFrame,
) -> None:
    rows = []

    summary = summary.copy()
    summary["year"] = pd.to_numeric(summary["year"], errors="coerce").astype("Int64")

    # Overall structural pragmatism index.
    for year in YEAR_ORDER:
        for community in COMMUNITY_ORDER:
            row = summary[
                (summary["community"] == community)
                & (summary["year"] == year)
            ]

            if row.empty:
                continue

            row = row.iloc[0]

            rows.append(
                {
                    "figure": "01_overall_structural_pragmatism_index",
                    "section": "overall_level",
                    "metric": "mean_structural_pragmatism_index",
                    "community": community,
                    "year": year,
                    "value": row.get("mean_structural_pragmatism_index", np.nan),
                    "interpretation": (
                        "Higher values indicate more intense and broader "
                        "structural pragmatism."
                    ),
                }
            )

    # Category profile and differences for 2025.
    cat_2025 = category_presence_table(sample, year=2025)

    for _, row in cat_2025.iterrows():
        rows.append(
            {
                "figure": "03_category_profile_2025",
                "section": "category_profile_2025",
                "metric": row["category_label"],
                "community": "finanzen",
                "year": 2025,
                "value": row["finanzen"],
                "interpretation": "Share of finanzen posts containing this category.",
            }
        )

        rows.append(
            {
                "figure": "03_category_profile_2025",
                "section": "category_profile_2025",
                "metric": row["category_label"],
                "community": "personalfinance",
                "year": 2025,
                "value": row["personalfinance"],
                "interpretation": (
                    "Share of personalfinance posts containing this category."
                ),
            }
        )

        rows.append(
            {
                "figure": "04_category_differences_2025",
                "section": "category_difference_2025",
                "metric": row["category_label"],
                "community": "finanzen_minus_personalfinance",
                "year": 2025,
                "value": row["difference_finanzen_minus_personalfinance"],
                "interpretation": (
                    "Positive means higher in finanzen; negative means higher "
                    "in personalfinance."
                ),
            }
        )

    # Dominant category distribution.
    dominant = make_dominant_category_table(sample)

    for _, row in dominant.iterrows():
        rows.append(
            {
                "figure": "05_dominant_category_stacked",
                "section": "dominant_category_distribution",
                "metric": row["dominant_category_label"],
                "community": row["group"],
                "year": np.nan,
                "value": row["share"],
                "interpretation": (
                    "Share of posts where this is the dominant structural "
                    "pragmatism category."
                ),
            }
        )

    # Category effect sizes.
    target = "between_communities_finanzen_minus_personalfinance_2025"

    keep_vars = [
        f"prag_{category}_has"
        for category in CATEGORY_ORDER
    ]

    effects = categorical_tests[
        (categorical_tests["comparison_type"] == target)
        & (categorical_tests["categorical_variable"].isin(keep_vars))
    ].copy()

    effects["category"] = effects["categorical_variable"].map(clean_category_var_name)
    effects["category_label"] = effects["category"].map(CATEGORY_LABELS)

    for _, row in effects.iterrows():
        rows.append(
            {
                "figure": "08_category_effects_2025",
                "section": "category_effect_size_2025",
                "metric": row["category_label"],
                "community": "finanzen_minus_personalfinance",
                "year": 2025,
                "value": row["cramers_v"],
                "interpretation": (
                    f"Cramer's V for category presence difference; "
                    f"share difference = {pp_label(row['share_difference_1_minus_2'])}."
                ),
            }
        )

    out = pd.DataFrame(rows)

    save_figure_data(out, "00_story_key_numbers.csv")


# ============================================================
# 12. MAIN
# ============================================================

def main() -> None:
    print("=" * 80)
    print("CREATING CORE STRUCTURAL PRAGMATISM STORY FIGURES")
    print("=" * 80)

    inputs = load_story_inputs()

    sample = inputs["balanced_sample"]
    summary = inputs["balanced_summary"]
    categorical_tests = inputs["balanced_categorical_tests"]

    for df in [sample, summary]:
        if "year" in df.columns:
            df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")

    make_story_key_number_table(
        summary=summary,
        sample=sample,
        categorical_tests=categorical_tests,
    )

    plot_overall_index(summary)
    plot_category_profile_2025(sample)
    plot_category_difference_2025(sample)
    plot_dominant_category_stacked(sample)
    plot_category_effects_2025(categorical_tests)

    print()
    print("Done.")
    print("=" * 80)
    print(f"Figures saved to: {CFG.figure_dir}")
    print(f"Figure data saved to: {CFG.figure_data_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()