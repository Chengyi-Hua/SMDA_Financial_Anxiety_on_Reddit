from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


@dataclass(frozen=True)
class VizConfig:
    project_dir: Path = Path(
        r"SMDA_Financial_Anxiety_on_Reddit"
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


COMMUNITY_ORDER = ["finanzen", "personalfinance"]
YEAR_ORDER = [2020, 2025]

COMMUNITY_LABELS = {
    "finanzen": "German",
    "personalfinance": "English",
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

COLOR_FINANZEN = "#1F77B4"
COLOR_PERSONALFINANCE = "#D62728"
COLOR_NEUTRAL = "#6E6E6E"
COLOR_GRID = "#D9D9D9"

plt.rcParams.update(
    {
        "figure.figsize": (13.33, 7.5),   # 16:9 slide format
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.size": 14,
        "axes.titlesize": 20,
        "axes.labelsize": 15,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 12,
        "legend.title_fontsize": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


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


def save_figure(
    fig: plt.Figure,
    filename: str,
    tight_layout: bool = True,
) -> None:
    output_path = CFG.figure_dir / filename

    if tight_layout:
        fig.tight_layout()

    fig.savefig(output_path, bbox_inches="tight")

    # Also save vector versions for presentation quality.
    stem = output_path.with_suffix("")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")

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


def load_story_inputs() -> dict[str, pd.DataFrame]:
    data = {}

    # Full sample for main-text figures
    data["sample"] = read_csv_required(
        "pragmatism_scored_posts_slim.csv"
    )

    data["summary"] = read_csv_required(
        "pragmatism_group_summary_full_sample.csv"
    )

    data["categorical_tests"] = read_csv_required(
        "pragmatism_categorical_statistical_tests_full_sample.csv"
    )

    return data



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


def dominant_category_profile_by_year(sample: pd.DataFrame, year: int) -> pd.DataFrame:
    require_columns(
        sample,
        ["community", "year", "dominant_prag_category"],
        "pragmatism_balanced_sample.csv",
    )

    df = sample.copy()
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df = df[df["year"].eq(year)].copy()

    counts = pd.crosstab(df["dominant_prag_category"], df["community"])

    for community in COMMUNITY_ORDER:
        if community not in counts.columns:
            counts[community] = 0

    for category in DOMINANT_CATEGORY_ORDER:
        if category not in counts.index:
            counts.loc[category] = 0

    counts = counts.loc[DOMINANT_CATEGORY_ORDER, COMMUNITY_ORDER]

    shares = counts.div(counts.sum(axis=0).replace(0, np.nan), axis=1)

    out = shares.reset_index().rename(
        columns={"dominant_prag_category": "category"}
    )

    out["category_label"] = out["category"].map(DOMINANT_CATEGORY_LABELS)

    out["difference_finanzen_minus_personalfinance"] = (
        out["finanzen"] - out["personalfinance"]
    )

    return out


def plot_category_profile_by_year(sample: pd.DataFrame, year: int) -> None:
    df = dominant_category_profile_by_year(sample, year=year)

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
        label=COMMUNITY_LABELS["personalfinance"],
        color=COLOR_PERSONALFINANCE,
    )

    ax.barh(
        y + height / 2,
        df["finanzen"],
        height,
        label=COMMUNITY_LABELS["finanzen"],
        color=COLOR_FINANZEN,
    )

    ax.set_title(f"Dominant category of structural pragmatism, {year}")
    ax.set_xlabel("Share of posts where this is the dominant category")
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

    save_figure_data(df, f"03_dominant_category_profile_{year}.csv")
    save_figure(fig, f"03_dominant_category_profile_{year}.png")



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
    label=COMMUNITY_LABELS["personalfinance"],
    color=COLOR_PERSONALFINANCE,
    )

    ax.barh(
        y + height / 2,
        df["finanzen"],
        height,
        label=COMMUNITY_LABELS["finanzen"],
        color=COLOR_FINANZEN,
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



def plot_category_difference_by_year(sample: pd.DataFrame, year: int) -> None:
    df = category_presence_table(sample, year=year)

    df = df.sort_values(
        "difference_finanzen_minus_personalfinance",
        ascending=True,
    )

    fig, ax = plt.subplots(figsize=(10, 6.5))

    y = np.arange(len(df))
    diff = df["difference_finanzen_minus_personalfinance"]

    bars = ax.barh(y, diff)

    ax.axvline(0, linewidth=1)
    ax.set_title(f"The communities differ in the type of pragmatism they express, {year}")
    ax.set_xlabel("Difference in share of posts: German minus English")
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
        "Positive = higher in German\nNegative = higher in English",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        bbox=dict(boxstyle="round", alpha=0.08),
    )

    save_figure_data(df, f"04_category_differences_{year}.csv")
    save_figure(fig, f"04_category_differences_{year}.png")


def plot_dominant_category_stacked_simplified(sample: pd.DataFrame) -> None:
    long = make_dominant_category_table(sample)

    keep = {
        "financial_institutions_products",
        "investment_strategy",
        "macro_economic_context",
        "planning_budgeting_control",
        "policy_regulation",
        "none",
    }

    long["dominant_category_simplified"] = np.where(
        long["dominant_category"].isin(keep),
        long["dominant_category"],
        "other",
    )

    label_map = {
        **DOMINANT_CATEGORY_LABELS,
        "other": "Other categories",
    }

    simplified = (
        long.groupby(["group", "dominant_category_simplified"], as_index=False)
        ["share"]
        .sum()
    )

    simplified["label"] = simplified["dominant_category_simplified"].map(label_map)

    order = [
        "financial_institutions_products",
        "investment_strategy",
        "macro_economic_context",
        "planning_budgeting_control",
        "policy_regulation",
        "other",
        "none",
    ]

    group_order = [
        "finanzen 2020",
        "finanzen 2025",
        "personalfinance 2020",
        "personalfinance 2025",
    ]

    pivot = simplified.pivot_table(
        index="group",
        columns="dominant_category_simplified",
        values="share",
        fill_value=0,
    ).reindex(group_order)

    for col in order:
        if col not in pivot.columns:
            pivot[col] = 0.0

    pivot = pivot[order]

    save_figure_data(
        simplified,
        "05_dominant_category_stacked_simplified.csv",
    )

    fig, ax = plt.subplots(figsize=(13.33, 7.5))

    colors = {
        "financial_institutions_products": "#D62728",
        "investment_strategy": "#1F77B4",
        "macro_economic_context": "#2CA02C",
        "planning_budgeting_control": "#FF7F0E",
        "policy_regulation": "#9467BD",
        "other": "#BDBDBD",
        "none": "#F0F0F0",
    }

    GROUP_LABELS = {
        "finanzen 2020": "German 2020",
        "finanzen 2025": "German 2025",
        "personalfinance 2020": "English 2020",
        "personalfinance 2025": "English 2025",
    }
    left = np.zeros(len(pivot))
    y = np.arange(len(pivot))

    for category in order:
        values = pivot[category].to_numpy()

        ax.barh(
            y,
            values,
            left=left,
            label=label_map[category],
            color=colors[category],
            edgecolor="white",
            linewidth=0.5,
        )

        left += values

    ax.set_title(
        "Different forms of pragmatism dominate each community",
        fontsize=20,
        weight="bold",
    )

    ax.set_xlabel("Share of posts")
    ax.set_yticks(y)
    ax.set_yticklabels([GROUP_LABELS[g] for g in pivot.index])
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.grid(axis="x", alpha=0.20, color=COLOR_GRID)

    ax.legend(
        title="Dominant category",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        borderaxespad=0,
    )

    save_figure(fig, "05_dominant_category_stacked_simplified_slide.png")



def plot_category_effects_by_year(
    categorical_tests: pd.DataFrame,
    year: int,
) -> None:
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

    target = f"between_communities_finanzen_minus_personalfinance_{year}"

    keep_vars = [
        f"prag_{category}_has"
        for category in CATEGORY_ORDER
    ]

    df = df[
        (df["comparison_type"] == target)
        & (df["categorical_variable"].isin(keep_vars))
    ].copy()

    if df.empty:
        print(f"Warning: no category effect-size rows found for {year}; skipped Figure 8 for this year.")
        return

    df["category"] = df["categorical_variable"].map(clean_category_var_name)
    df["category_label"] = df["category"].map(CATEGORY_LABELS)

    df = df.sort_values("cramers_v", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6.5))

    y = np.arange(len(df))
    values = df["cramers_v"]

    bars = ax.barh(y, values)

    ax.set_title(f"Strength of category differences, {year}")
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

    save_figure_data(df, f"08_category_effects_{year}.csv")
    save_figure(fig, f"08_category_effects_{year}.png")


def plot_category_difference_2020_2025(sample: pd.DataFrame) -> None:
    df_2020 = dominant_category_profile_by_year(sample, year=2020)
    df_2025 = dominant_category_profile_by_year(sample, year=2025)

    df = df_2020[
        ["category", "category_label", "difference_finanzen_minus_personalfinance"]
    ].rename(
        columns={
            "difference_finanzen_minus_personalfinance": "difference_2020"
        }
    ).merge(
        df_2025[
            ["category", "difference_finanzen_minus_personalfinance"]
        ].rename(
            columns={
                "difference_finanzen_minus_personalfinance": "difference_2025"
            }
        ),
        on="category",
        how="inner",
    )

    df["sort_value"] = df[["difference_2020", "difference_2025"]].mean(axis=1)
    df = df.sort_values("sort_value", ascending=True)

    save_figure_data(df, "04_category_differences_2020_2025.csv")

    fig, ax = plt.subplots(figsize=(13.33, 7.5))

    y = np.arange(len(df))
    height = 0.34

    ax.barh(
        y - height / 2,
        df["difference_2020"],
        height,
        label="2020",
        color="#9ECAE1",
    )

    ax.barh(
        y + height / 2,
        df["difference_2025"],
        height,
        label="2025",
        color="#08519C",
    )

    ax.axvline(0, color="black", linewidth=1)

    ax.set_title(
        "The community divide is stable across 2020 and 2025",
        fontsize=20,
        weight="bold",
    )

    ax.set_xlabel(
        "Difference in dominant-category share: German minus English",
        fontsize=15,
    )

    ax.set_yticks(y)
    ax.set_yticklabels(df["category_label"])
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.grid(axis="x", alpha=0.25, color=COLOR_GRID)

    max_abs = max(
        float(np.nanmax(np.abs(df[["difference_2020", "difference_2025"]].to_numpy()))),
        0.01,
    )
    ax.set_xlim(-max_abs * 1.35, max_abs * 1.35)

    ax.text(
        0.01,
        0.02,
        "Left = higher in English\nRight = higher in German",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=12,
        bbox=dict(boxstyle="round", alpha=0.08),
    )

    ax.legend(title="Year", loc="lower right")

    save_figure(fig, "04_category_differences_2020_2025_slide.png")



def category_development_table(sample: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for year in YEAR_ORDER:
        year_table = category_presence_table(sample, year=year)

        for _, row in year_table.iterrows():
            for community in COMMUNITY_ORDER:
                rows.append(
                    {
                        "year": year,
                        "community": community,
                        "category": row["category"],
                        "category_label": row["category_label"],
                        "share": row[community],
                    }
                )

    return pd.DataFrame(rows)

def plot_category_development_compact(sample: pd.DataFrame) -> None:
    from textwrap import fill

    df = category_development_table(sample)

    save_figure_data(
        df,
        "12_category_development_compact.csv",
    )

    max_share = max(float(np.nanmax(df["share"].to_numpy())), 0.01)
    y_max = min(1.0, max_share + 0.12)

    fig, axes = plt.subplots(
        2,
        4,
        figsize=(13.33, 7.5),
        sharex=True,
        sharey=True,
    )

    axes = axes.flatten()

    for ax, category in zip(axes, CATEGORY_ORDER):
        category_df = df[df["category"] == category].copy()

        pivot = (
            category_df.pivot_table(
                index="year",
                columns="community",
                values="share",
            )
            .reindex(YEAR_ORDER)
        )

        for community in COMMUNITY_ORDER:
            if community not in pivot.columns:
                pivot[community] = np.nan

        finanzen_2020 = pivot.loc[2020, "finanzen"]
        finanzen_2025 = pivot.loc[2025, "finanzen"]
        pf_2020 = pivot.loc[2020, "personalfinance"]
        pf_2025 = pivot.loc[2025, "personalfinance"]

        finanzen_change = finanzen_2025 - finanzen_2020
        pf_change = pf_2025 - pf_2020

        ax.plot(
            YEAR_ORDER,
            pivot["personalfinance"],
            marker="o",
            linewidth=2.4,
            color=COLOR_PERSONALFINANCE,
        )

        ax.plot(
            YEAR_ORDER,
            pivot["finanzen"],
            marker="o",
            linewidth=2.4,
            color=COLOR_FINANZEN,
        )

        ax.set_title(
            fill(CATEGORY_LABELS[category], width=24),
            fontsize=10.5,
            weight="bold",
            pad=8,
        )

        ax.set_xticks(YEAR_ORDER)
        ax.set_xlim(2019.75, 2025.65)
        ax.set_ylim(0, y_max)
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.grid(axis="y", alpha=0.20, color=COLOR_GRID)

        fin_label_y = finanzen_2025
        pf_label_y = pf_2025

        if (
            not np.isnan(finanzen_2025)
            and not np.isnan(pf_2025)
            and abs(finanzen_2025 - pf_2025) < 0.035
        ):
            if finanzen_2025 >= pf_2025:
                fin_label_y = finanzen_2025 + 0.018
                pf_label_y = pf_2025 - 0.018
            else:
                fin_label_y = finanzen_2025 - 0.018
                pf_label_y = pf_2025 + 0.018

        if not np.isnan(finanzen_2025):
            ax.text(
                2025.10,
                fin_label_y,
                percent_label(finanzen_2025),
                va="center",
                ha="left",
                fontsize=7.5,
                color=COLOR_FINANZEN,
            )

        if not np.isnan(pf_2025):
            ax.text(
                2025.10,
                pf_label_y,
                percent_label(pf_2025),
                va="center",
                ha="left",
                fontsize=7.5,
                color=COLOR_PERSONALFINANCE,
            )

        ax.text(
            0.03,
            0.04,
            f"German: {pp_label(finanzen_change)}\nEnglish: {pp_label(pf_change)}",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=7.5,
            bbox=dict(
                boxstyle="round,pad=0.25",
                facecolor="white",
                edgecolor=COLOR_GRID,
                alpha=0.90,
            ),
        )

    for ax in axes[len(CATEGORY_ORDER):]:
        ax.axis("off")

    fig.suptitle(
        "Development of structural pragmatism categories, 2020–2025",
        fontsize=18,
        weight="bold",
        y=0.965,
    )

    fig.text(
        0.5,
        0.925,
        "Red = English-speaking communities     Blue = German speaking communities",
        ha="center",
        va="center",
        fontsize=11,
    )

    fig.supxlabel("Year", fontsize=12, y=0.035)
    fig.supylabel("Share of posts containing category", fontsize=12, x=0.015)

    fig.subplots_adjust(
        top=0.84,
        left=0.08,
        right=0.94,
        bottom=0.10,
        hspace=0.42,
        wspace=0.25,
    )

    output_path = CFG.figure_dir / "12_category_development_compact_slide.png"
    fig.savefig(output_path, bbox_inches="tight")

    stem = output_path.with_suffix("")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")

    plt.close(fig)

    print(f"Saved figure: {output_path}")


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




def main() -> None:
    print("=" * 80)
    print("CREATING CORE STRUCTURAL PRAGMATISM STORY FIGURES")
    print("=" * 80)

    inputs = load_story_inputs()

    sample = inputs["sample"]
    summary = inputs["summary"]
    categorical_tests = inputs["categorical_tests"]
    
    for df in [sample, summary]:
        if "year" in df.columns:
            df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")

    make_story_key_number_table(
        summary=summary,
        sample=sample,
        categorical_tests=categorical_tests,
    )

    for year in YEAR_ORDER:
        plot_category_profile_by_year(sample, year)
        plot_category_difference_by_year(sample, year)
        plot_category_effects_by_year(categorical_tests, year)

    plot_dominant_category_stacked_simplified(sample)
    plot_category_difference_2020_2025(sample)

    # Professor-requested compact development figure.
    plot_category_development_compact(sample)

    print()
    print("Done.")
    print("=" * 80)
    print(f"Figures saved to: {CFG.figure_dir}")
    print(f"Figure data saved to: {CFG.figure_data_dir}")
    print("=" * 80)



if __name__ == "__main__":
    main()
