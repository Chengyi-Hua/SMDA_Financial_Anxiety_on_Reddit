from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import (
    mannwhitneyu,
    kruskal,
    ks_2samp,
    ttest_ind,
    chi2_contingency,
    shapiro,
    skew,
    kurtosis,
    normaltest,
)


DATA_DIR = Path(r"SMDA_Financial_Anxiety_on_Reddit\sentiment_analysis")
OUTPUT_DIR = DATA_DIR / "sentiment_outputs"
TEST_DIR = OUTPUT_DIR / "significance_tests"
FIG_DIR = TEST_DIR / "distribution_figures"

TEST_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

POSTS_FILE = OUTPUT_DIR / "sentiment_scored_posts_slim.csv"

MAIN_SCORE = "no_emoji_xlm_sentiment_score"
VADER_SCORE = "no_emoji_vader_compound"
EMOJI_SCORE = "esr_emoji_score_mean"

MAIN_LABEL = "no_emoji_xlm_label"
EMOJI_LABEL = "esr_emoji_label"
EMOJI_BINARY = "has_emoji"

RANDOM_SEED = 42
NORMALITY_SAMPLE_N = 5000
BOOTSTRAP_N = 1000
BOOTSTRAP_MAX_N = 20000

rng = np.random.default_rng(RANDOM_SEED)

def pretty_community(x):
    if x == "finanzen":
        return "r/Finanzen"
    if x == "personalfinance":
        return "r/personalfinance"
    return str(x)


def clean_numeric(series):
    return pd.to_numeric(series, errors="coerce").dropna().values


def rank_biserial_from_u(u_stat, n1, n2):
    """
    Positive value means group 1 tends to have higher values than group 2.
    """
    return (2 * u_stat) / (n1 * n2) - 1


def cramers_v(table):
    chi2, _, _, _ = chi2_contingency(table)
    n = table.to_numpy().sum()
    if n == 0:
        return np.nan
    r, k = table.shape
    return np.sqrt((chi2 / n) / max(1, min(k - 1, r - 1)))


def bootstrap_ci_diff(x, y, stat_func=np.mean, n_boot=BOOTSTRAP_N):
    """
    Bootstrap CI for difference: stat(x) - stat(y).
    To keep runtime reasonable, very large groups are capped.
    """
    x = np.asarray(x)
    y = np.asarray(y)

    if len(x) > BOOTSTRAP_MAX_N:
        x = rng.choice(x, size=BOOTSTRAP_MAX_N, replace=False)
    if len(y) > BOOTSTRAP_MAX_N:
        y = rng.choice(y, size=BOOTSTRAP_MAX_N, replace=False)

    diffs = np.empty(n_boot)

    for i in range(n_boot):
        xb = rng.choice(x, size=len(x), replace=True)
        yb = rng.choice(y, size=len(y), replace=True)
        diffs[i] = stat_func(xb) - stat_func(yb)

    return np.percentile(diffs, 2.5), np.percentile(diffs, 97.5)


data = pd.read_csv(POSTS_FILE, low_memory=False)

data["community_pretty"] = data["community"].map(pretty_community)
data["group"] = data["community"].astype(str) + "_" + data["year"].astype(str)

if EMOJI_BINARY in data.columns:
    data[EMOJI_BINARY] = data[EMOJI_BINARY].astype(str).str.lower().isin(["true", "1", "yes"])


# DISTRIBUTION DIAGNOSTICS
score_columns = [
    MAIN_SCORE,
    VADER_SCORE,
    EMOJI_SCORE,
]
score_columns = [c for c in score_columns if c in data.columns]

distribution_rows = []
normality_rows = []

for score in score_columns:
    for (community, year), group_df in data.groupby(["community", "year"]):
        values = clean_numeric(group_df[score])

        if len(values) == 0:
            continue

        distribution_rows.append({
            "score": score,
            "community": community,
            "year": year,
            "n": len(values),
            "mean": np.mean(values),
            "median": np.median(values),
            "std": np.std(values, ddof=1) if len(values) > 1 else np.nan,
            "min": np.min(values),
            "q25": np.percentile(values, 25),
            "q75": np.percentile(values, 75),
            "max": np.max(values),
            "skewness": skew(values) if len(values) > 2 else np.nan,
            "kurtosis": kurtosis(values) if len(values) > 3 else np.nan,
        })

        sample_values = values
        if len(sample_values) > NORMALITY_SAMPLE_N:
            sample_values = rng.choice(sample_values, size=NORMALITY_SAMPLE_N, replace=False)

        if len(sample_values) >= 8:
            shapiro_stat, shapiro_p = shapiro(sample_values)
        else:
            shapiro_stat, shapiro_p = np.nan, np.nan

        if len(sample_values) >= 20:
            normal_stat, normal_p = normaltest(sample_values)
        else:
            normal_stat, normal_p = np.nan, np.nan

        normality_rows.append({
            "score": score,
            "community": community,
            "year": year,
            "sample_n_used": len(sample_values),
            "shapiro_stat": shapiro_stat,
            "shapiro_p": shapiro_p,
            "dagostino_stat": normal_stat,
            "dagostino_p": normal_p,
            "interpretation": (
                "Likely non-normal / bounded / skewed"
                if (pd.notna(shapiro_p) and shapiro_p < 0.05)
                else "No strong rejection in sampled diagnostic"
            )
        })

        plt.figure(figsize=(7, 4))
        plt.hist(values, bins=50)
        plt.axvline(np.mean(values), linestyle="--", linewidth=1, label="Mean")
        plt.axvline(np.median(values), linestyle=":", linewidth=1, label="Median")
        plt.title(f"Distribution of {score}: {pretty_community(community)} {year}")
        plt.xlabel(score)
        plt.ylabel("Number of posts")
        plt.legend()
        plt.tight_layout()
        plt.savefig(FIG_DIR / f"hist_{score}_{community}_{year}.png", dpi=300)
        plt.close()

distribution_df = pd.DataFrame(distribution_rows)
normality_df = pd.DataFrame(normality_rows)

distribution_df.to_csv(TEST_DIR / "distribution_summary.csv", index=False, encoding="utf-8-sig")
normality_df.to_csv(TEST_DIR / "normality_diagnostics.csv", index=False, encoding="utf-8-sig")


#  SCORE TESTS: 2025 VS 2020 WITHIN EACH COMMUNITY
test_rows = []

for score in score_columns:
    # Within-community comparisons
    for community in sorted(data["community"].dropna().unique()):
        x = clean_numeric(data[(data["community"] == community) & (data["year"] == 2025)][score])
        y = clean_numeric(data[(data["community"] == community) & (data["year"] == 2020)][score])

        if len(x) < 2 or len(y) < 2:
            continue

        u_stat, mw_p = mannwhitneyu(x, y, alternative="two-sided", method="asymptotic")
        rb = rank_biserial_from_u(u_stat, len(x), len(y))

        ks_stat, ks_p = ks_2samp(x, y, alternative="two-sided", method="asymp")
        t_stat, t_p = ttest_ind(x, y, equal_var=False)

        mean_diff = np.mean(x) - np.mean(y)
        median_diff = np.median(x) - np.median(y)

        mean_ci_low, mean_ci_high = bootstrap_ci_diff(x, y, np.mean)
        median_ci_low, median_ci_high = bootstrap_ci_diff(x, y, np.median)

        test_rows.append({
            "comparison_type": "within_community_2025_minus_2020",
            "score": score,
            "community": community,
            "group_1": f"{community}_2025",
            "group_2": f"{community}_2020",
            "n_2025": len(x),
            "n_2020": len(y),
            "mean_2025": np.mean(x),
            "mean_2020": np.mean(y),
            "mean_difference_2025_minus_2020": mean_diff,
            "mean_diff_ci_low": mean_ci_low,
            "mean_diff_ci_high": mean_ci_high,
            "median_2025": np.median(x),
            "median_2020": np.median(y),
            "median_difference_2025_minus_2020": median_diff,
            "median_diff_ci_low": median_ci_low,
            "median_diff_ci_high": median_ci_high,
            "mannwhitney_u": u_stat,
            "mannwhitney_p": mw_p,
            "rank_biserial_effect": rb,
            "ks_statistic": ks_stat,
            "ks_p": ks_p,
            "welch_t": t_stat,
            "welch_p": t_p,
        })

    x = clean_numeric(data[data["year"] == 2025][score])
    y = clean_numeric(data[data["year"] == 2020][score])

    if len(x) >= 2 and len(y) >= 2:
        u_stat, mw_p = mannwhitneyu(x, y, alternative="two-sided", method="asymptotic")
        rb = rank_biserial_from_u(u_stat, len(x), len(y))
        ks_stat, ks_p = ks_2samp(x, y, alternative="two-sided", method="asymp")
        t_stat, t_p = ttest_ind(x, y, equal_var=False)

        mean_diff = np.mean(x) - np.mean(y)
        median_diff = np.median(x) - np.median(y)

        mean_ci_low, mean_ci_high = bootstrap_ci_diff(x, y, np.mean)
        median_ci_low, median_ci_high = bootstrap_ci_diff(x, y, np.median)

        test_rows.append({
            "comparison_type": "pooled_2025_minus_2020_composition_sensitive",
            "score": score,
            "community": "pooled",
            "group_1": "pooled_2025",
            "group_2": "pooled_2020",
            "n_2025": len(x),
            "n_2020": len(y),
            "mean_2025": np.mean(x),
            "mean_2020": np.mean(y),
            "mean_difference_2025_minus_2020": mean_diff,
            "mean_diff_ci_low": mean_ci_low,
            "mean_diff_ci_high": mean_ci_high,
            "median_2025": np.median(x),
            "median_2020": np.median(y),
            "median_difference_2025_minus_2020": median_diff,
            "median_diff_ci_low": median_ci_low,
            "median_diff_ci_high": median_ci_high,
            "mannwhitney_u": u_stat,
            "mannwhitney_p": mw_p,
            "rank_biserial_effect": rb,
            "ks_statistic": ks_stat,
            "ks_p": ks_p,
            "welch_t": t_stat,
            "welch_p": t_p,
        })

score_tests_df = pd.DataFrame(test_rows)
score_tests_df.to_csv(TEST_DIR / "year_score_tests_2025_vs_2020.csv", index=False, encoding="utf-8-sig")


# KRUSKAL-WALLIS ACROSS ALL FOUR GROUPS

kruskal_rows = []

for score in score_columns:
    group_values = []
    group_names = []

    for group_name, group_df in data.groupby("group"):
        values = clean_numeric(group_df[score])
        if len(values) >= 2:
            group_values.append(values)
            group_names.append(group_name)

    if len(group_values) >= 2:
        stat, p = kruskal(*group_values)
        kruskal_rows.append({
            "score": score,
            "groups": "; ".join(group_names),
            "kruskal_statistic": stat,
            "kruskal_p": p,
            "interpretation": "At least one community-year group differs" if p < 0.05 else "No significant group difference detected",
        })

kruskal_df = pd.DataFrame(kruskal_rows)
kruskal_df.to_csv(TEST_DIR / "kruskal_all_groups.csv", index=False, encoding="utf-8-sig")


# CATEGORICAL TESTS: XLM LABEL DISTRIBUTION BY YEAR

label_test_rows = []

if MAIN_LABEL in data.columns:
    for community in sorted(data["community"].dropna().unique()):
        subset = data[data["community"] == community].copy()
        table = pd.crosstab(subset["year"], subset[MAIN_LABEL])

        if table.shape[0] >= 2 and table.shape[1] >= 2:
            chi2, p, dof, expected = chi2_contingency(table)
            label_test_rows.append({
                "test": "sentiment_label_distribution_by_year",
                "community": community,
                "table": table.to_dict(),
                "chi2": chi2,
                "p_value": p,
                "dof": dof,
                "cramers_v": cramers_v(table),
            })

    # Pooled
    table = pd.crosstab(data["year"], data[MAIN_LABEL])
    if table.shape[0] >= 2 and table.shape[1] >= 2:
        chi2, p, dof, expected = chi2_contingency(table)
        label_test_rows.append({
            "test": "sentiment_label_distribution_by_year",
            "community": "pooled_composition_sensitive",
            "table": table.to_dict(),
            "chi2": chi2,
            "p_value": p,
            "dof": dof,
            "cramers_v": cramers_v(table),
        })

label_tests_df = pd.DataFrame(label_test_rows)
label_tests_df.to_csv(TEST_DIR / "sentiment_label_chi_square_tests.csv", index=False, encoding="utf-8-sig")


# EMOJI USAGE TESTS: HAS_EMOJI BY YEAR

emoji_presence_rows = []

if EMOJI_BINARY in data.columns:
    for community in sorted(data["community"].dropna().unique()):
        subset = data[data["community"] == community].copy()
        table = pd.crosstab(subset["year"], subset[EMOJI_BINARY])

        if table.shape[0] >= 2 and table.shape[1] >= 2:
            chi2, p, dof, expected = chi2_contingency(table)

            share_2025 = subset[subset["year"] == 2025][EMOJI_BINARY].mean()
            share_2020 = subset[subset["year"] == 2020][EMOJI_BINARY].mean()

            emoji_presence_rows.append({
                "test": "emoji_presence_by_year",
                "community": community,
                "emoji_share_2025": share_2025,
                "emoji_share_2020": share_2020,
                "difference_2025_minus_2020": share_2025 - share_2020,
                "chi2": chi2,
                "p_value": p,
                "dof": dof,
                "cramers_v": cramers_v(table),
            })

    # Pooled
    table = pd.crosstab(data["year"], data[EMOJI_BINARY])
    if table.shape[0] >= 2 and table.shape[1] >= 2:
        chi2, p, dof, expected = chi2_contingency(table)
        share_2025 = data[data["year"] == 2025][EMOJI_BINARY].mean()
        share_2020 = data[data["year"] == 2020][EMOJI_BINARY].mean()

        emoji_presence_rows.append({
            "test": "emoji_presence_by_year",
            "community": "pooled_composition_sensitive",
            "emoji_share_2025": share_2025,
            "emoji_share_2020": share_2020,
            "difference_2025_minus_2020": share_2025 - share_2020,
            "chi2": chi2,
            "p_value": p,
            "dof": dof,
            "cramers_v": cramers_v(table),
        })

emoji_presence_df = pd.DataFrame(emoji_presence_rows)
emoji_presence_df.to_csv(TEST_DIR / "emoji_presence_chi_square_tests.csv", index=False, encoding="utf-8-sig")


# EMOJI AFFECT LABEL TESTS AMONG EMOJI POSTS ONLY

emoji_affect_rows = []

if EMOJI_LABEL in data.columns:
    emoji_data = data[data[EMOJI_BINARY] == True].copy()

    for community in sorted(emoji_data["community"].dropna().unique()):
        subset = emoji_data[emoji_data["community"] == community].copy()
        subset = subset[subset[EMOJI_LABEL] != "no_emoji"]

        table = pd.crosstab(subset["year"], subset[EMOJI_LABEL])

        if table.shape[0] >= 2 and table.shape[1] >= 2:
            chi2, p, dof, expected = chi2_contingency(table)
            emoji_affect_rows.append({
                "test": "emoji_affect_label_distribution_by_year_among_emoji_posts",
                "community": community,
                "n_emoji_posts": len(subset),
                "table": table.to_dict(),
                "chi2": chi2,
                "p_value": p,
                "dof": dof,
                "cramers_v": cramers_v(table),
            })

    table = pd.crosstab(emoji_data["year"], emoji_data[EMOJI_LABEL])
    if table.shape[0] >= 2 and table.shape[1] >= 2:
        chi2, p, dof, expected = chi2_contingency(table)
        emoji_affect_rows.append({
            "test": "emoji_affect_label_distribution_by_year_among_emoji_posts",
            "community": "pooled_composition_sensitive",
            "n_emoji_posts": len(emoji_data),
            "table": table.to_dict(),
            "chi2": chi2,
            "p_value": p,
            "dof": dof,
            "cramers_v": cramers_v(table),
        })

emoji_affect_df = pd.DataFrame(emoji_affect_rows)
emoji_affect_df.to_csv(TEST_DIR / "emoji_affect_label_chi_square_tests.csv", index=False, encoding="utf-8-sig")


# SHORT TEXT 

summary_path = TEST_DIR / "significance_testing_readme.txt"

with open(summary_path, "w", encoding="utf-8") as f:
    f.write("Significance and distribution testing summary\n")
    f.write("============================================\n\n")
    f.write("Main sentiment variable:\n")
    f.write(f"- {MAIN_SCORE}: multilingual XLM-R text sentiment, emojis removed.\n\n")
    f.write("Distribution diagnostics:\n")
    f.write("- See distribution_summary.csv and normality_diagnostics.csv.\n")
    f.write("- Histograms are saved in distribution_figures/.\n")
    f.write("- Because sentiment scores are bounded and typically non-normal, non-parametric tests should be the main tests.\n\n")
    f.write("Main tests:\n")
    f.write("- Mann-Whitney U: 2025 vs 2020 within each community.\n")
    f.write("- Rank-biserial correlation: effect size for Mann-Whitney U.\n")
    f.write("- Bootstrap confidence intervals: mean and median differences.\n")
    f.write("- Kruskal-Wallis: overall test across the four community-year groups.\n")
    f.write("- Chi-square tests: sentiment label shares, emoji presence, emoji affect labels.\n\n")
    f.write("Interpretation warning:\n")
    f.write("- With large samples, p-values may be extremely small even for small effects.\n")
    f.write("- Report mean/median differences, confidence intervals, and effect sizes, not only p-values.\n")
    f.write("- Pooled comparisons are composition-sensitive because community sizes differ strongly across years.\n")

print("Done.")
print(f"Saved outputs to: {TEST_DIR}")
print(f"Distribution figures: {FIG_DIR}")