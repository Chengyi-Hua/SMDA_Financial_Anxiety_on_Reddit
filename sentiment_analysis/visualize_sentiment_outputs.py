# visualize_sentiment_outputs.py

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#
# CONFIG
#

DATA_DIR = Path(r"D:\Users\cheng\Documents\GitHub\SMDA_Financial_Anxiety_on_Reddit\sentiment_analysis")
OUTPUT_DIR = DATA_DIR / "sentiment_outputs"
FIG_DIR = OUTPUT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

GROUP_SUMMARY_FILE = OUTPUT_DIR / "sentiment_group_summary.csv"
BALANCED_SUMMARY_FILE = OUTPUT_DIR / "sentiment_group_summary_balanced.csv"
EMOJI_SUMMARY_FILE = OUTPUT_DIR / "emoji_summary.csv"
EMOJI_LABEL_SHARES_FILE = OUTPUT_DIR / "emoji_label_shares.csv"
POSTS_FILE = OUTPUT_DIR / "sentiment_scored_posts_slim.csv"

MAIN_SCORE = "no_emoji_xlm_sentiment_score"
MAIN_LABEL_PREFIX = "no_emoji_xlm_label"
VADER_SCORE = "no_emoji_vader_compound"
ESR_SCORE = "esr_emoji_score_mean"


#
# HELPERS
#

def pretty_community(x):
    if x == "finanzen":
        return "r/Finanzen"
    if x == "personalfinance":
        return "r/personalfinance"
    return str(x)


def add_group_label(df):
    df = df.copy()
    df["group_label"] = df["community"].map(pretty_community) + " " + df["year"].astype(str)
    return df


def savefig(name):
    path_png = FIG_DIR / f"{name}.png"
    path_svg = FIG_DIR / f"{name}.svg"
    plt.tight_layout()
    plt.savefig(path_png, dpi=300, bbox_inches="tight")
    plt.savefig(path_svg, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path_png}")


def ci(series):
    series = pd.to_numeric(series, errors="coerce").dropna()
    n = len(series)
    if n <= 1:
        return pd.Series({"mean": np.nan, "ci_low": np.nan, "ci_high": np.nan, "n": n})
    mean = series.mean()
    margin = 1.96 * series.std(ddof=1) / np.sqrt(n)
    return pd.Series({"mean": mean, "ci_low": mean - margin, "ci_high": mean + margin, "n": n})


def sorted_summary(df):
    return df.sort_values(["community", "year"]).reset_index(drop=True)


#
# LOAD DATA
#

group_summary = add_group_label(sorted_summary(pd.read_csv(GROUP_SUMMARY_FILE)))
balanced_summary = add_group_label(sorted_summary(pd.read_csv(BALANCED_SUMMARY_FILE)))
emoji_summary = add_group_label(sorted_summary(pd.read_csv(EMOJI_SUMMARY_FILE)))
emoji_label_shares = add_group_label(sorted_summary(pd.read_csv(EMOJI_LABEL_SHARES_FILE)))
posts = pd.read_csv(POSTS_FILE, low_memory=False)

posts["community_pretty"] = posts["community"].map(pretty_community)
posts["group_label"] = posts["community_pretty"] + " " + posts["year"].astype(str)


#
# 1. MAIN TEXT SENTIMENT BY COMMUNITY AND YEAR
#

def plot_main_sentiment(summary_df, suffix):
    mean_col = f"{MAIN_SCORE}_mean"
    low_col = f"{MAIN_SCORE}_ci_low"
    high_col = f"{MAIN_SCORE}_ci_high"

    x = np.arange(len(summary_df))
    means = summary_df[mean_col].values
    yerr = np.vstack([
        means - summary_df[low_col].values,
        summary_df[high_col].values - means
    ])

    plt.figure(figsize=(9, 5))
    plt.bar(x, means, yerr=yerr, capsize=4)
    plt.axhline(0, linewidth=1)
    plt.xticks(x, summary_df["group_label"], rotation=25, ha="right")
    plt.ylabel("Mean XLM-R text sentiment\nP(positive) - P(negative)")
    plt.title("Main text sentiment by community and year")
    plt.figtext(
        0.5, -0.05,
        "Text sentiment is computed with emojis removed. Lower values indicate more negative sentiment.",
        ha="center",
        fontsize=9
    )
    savefig(f"01_main_text_sentiment_{suffix}")


plot_main_sentiment(group_summary, "full_sample")
plot_main_sentiment(balanced_summary, "balanced_sample")


#
# 2. TEXT SENTIMENT LABEL SHARES
#

def plot_label_shares(summary_df, suffix):
    cols = [
        f"{MAIN_LABEL_PREFIX}_share_negative",
        f"{MAIN_LABEL_PREFIX}_share_neutral",
        f"{MAIN_LABEL_PREFIX}_share_positive",
    ]
    labels = ["Negative", "Neutral", "Positive"]

    x = np.arange(len(summary_df))
    bottom = np.zeros(len(summary_df))

    plt.figure(figsize=(9, 5))

    for col, label in zip(cols, labels):
        values = summary_df[col].fillna(0).values
        plt.bar(x, values, bottom=bottom, label=label)
        bottom += values

    plt.xticks(x, summary_df["group_label"], rotation=25, ha="right")
    plt.ylabel("Share of posts")
    plt.ylim(0, 1)
    plt.title("Distribution of XLM-R text sentiment labels")
    plt.legend()
    savefig(f"02_xlm_label_shares_{suffix}")


plot_label_shares(group_summary, "full_sample")
plot_label_shares(balanced_summary, "balanced_sample")


#
# 3. POOLED 2020 VS 2025 OVERVIEW
#

pooled_sentiment = (
    posts
    .groupby("year")[MAIN_SCORE]
    .apply(ci)
    .reset_index()
    .pivot_table(index="year", columns="level_1", values=MAIN_SCORE)
    .reset_index()
)

pooled_labels = (
    posts
    .groupby("year")["no_emoji_xlm_label"]
    .value_counts(normalize=True)
    .rename("share")
    .reset_index()
    .pivot_table(index="year", columns="no_emoji_xlm_label", values="share", fill_value=0)
    .reset_index()
)

pooled_emoji = (
    posts
    .groupby("year")
    .agg(
        n_posts=("id", "count"),
        emoji_share=("has_emoji", "mean"),
        mean_emoji_count=("emoji_count", "mean"),
    )
    .reset_index()
)

pooled_sentiment.to_csv(OUTPUT_DIR / "pooled_sentiment_by_year.csv", index=False, encoding="utf-8-sig")
pooled_labels.to_csv(OUTPUT_DIR / "pooled_label_shares_by_year.csv", index=False, encoding="utf-8-sig")
pooled_emoji.to_csv(OUTPUT_DIR / "pooled_emoji_by_year.csv", index=False, encoding="utf-8-sig")

x = np.arange(len(pooled_sentiment))
means = pooled_sentiment["mean"].values
yerr = np.vstack([
    means - pooled_sentiment["ci_low"].values,
    pooled_sentiment["ci_high"].values - means
])

plt.figure(figsize=(6, 4))
plt.bar(x, means, yerr=yerr, capsize=4)
plt.axhline(0, linewidth=1)
plt.xticks(x, pooled_sentiment["year"].astype(str))
plt.ylabel("Mean XLM-R text sentiment")
plt.title("Pooled text sentiment by year")
plt.figtext(
    0.5, -0.08,
    "Pooled result is composition-sensitive because community sizes differ strongly.",
    ha="center",
    fontsize=9
)
savefig("03_pooled_text_sentiment_by_year")


#
# 4. EMOJI USAGE BY COMMUNITY AND YEAR
#

x = np.arange(len(emoji_summary))

plt.figure(figsize=(9, 5))
plt.bar(x, emoji_summary["emoji_share"] * 100)
plt.xticks(x, emoji_summary["group_label"], rotation=25, ha="right")
plt.ylabel("Posts with emoji (%)")
plt.title("Emoji usage by community and year")
savefig("04_emoji_share_by_group")


#
# 5. EXPLICIT EMOJI AFFECT
#

emoji_affect = emoji_summary.copy()
emoji_affect["mapped_total"] = (
    emoji_affect["positive_emoji_count"]
    + emoji_affect["negative_emoji_count"]
    + emoji_affect["neutral_mixed_emoji_count"]
)

for col in ["positive_emoji_count", "negative_emoji_count", "neutral_mixed_emoji_count"]:
    emoji_affect[col + "_share"] = np.where(
        emoji_affect["mapped_total"] > 0,
        emoji_affect[col] / emoji_affect["mapped_total"],
        0
    )

x = np.arange(len(emoji_affect))
bottom = np.zeros(len(emoji_affect))

plt.figure(figsize=(9, 5))
for col, label in [
    ("positive_emoji_count_share", "Positive emojis"),
    ("negative_emoji_count_share", "Negative emojis"),
    ("neutral_mixed_emoji_count_share", "Neutral/mixed emojis"),
]:
    values = emoji_affect[col].fillna(0).values
    plt.bar(x, values, bottom=bottom, label=label)
    bottom += values

plt.xticks(x, emoji_affect["group_label"], rotation=25, ha="right")
plt.ylabel("Share of mapped emojis")
plt.ylim(0, 1)
plt.title("Explicit emoji affect among mapped emojis")
plt.legend()
savefig("05_emoji_affect_distribution")


plt.figure(figsize=(9, 5))
plt.bar(x, emoji_summary["mean_esr_emoji_score"])
plt.axhline(0, linewidth=1)
plt.xticks(x, emoji_summary["group_label"], rotation=25, ha="right")
plt.ylabel("Mean Emoji Sentiment Ranking score")
plt.title("Mean emoji sentiment by community and year")
plt.figtext(
    0.5, -0.06,
    "This score describes emojis only, not the full text. Positive values indicate more positive emoji affect.",
    ha="center",
    fontsize=9
)
savefig("06_mean_emoji_sentiment_score")


#
# 6. VADER VS XLM-R METHOD COMPARISON
#

method_df = group_summary.copy()

x = np.arange(len(method_df))
width = 0.38

plt.figure(figsize=(10, 5))
plt.bar(
    x - width / 2,
    method_df[f"{VADER_SCORE}_mean"],
    width,
    label="VADER text-only baseline"
)
plt.bar(
    x + width / 2,
    method_df[f"{MAIN_SCORE}_mean"],
    width,
    label="XLM-R multilingual text sentiment"
)
plt.axhline(0, linewidth=1)
plt.xticks(x, method_df["group_label"], rotation=25, ha="right")
plt.ylabel("Mean sentiment score")
plt.title("VADER baseline vs multilingual XLM-R sentiment")
plt.legend()
plt.figtext(
    0.5, -0.08,
    "VADER is included as a baseline. XLM-R is the main cross-language sentiment measure.",
    ha="center",
    fontsize=9
)
savefig("07_vader_vs_xlm_method_comparison")


#
# 7. FULL SAMPLE VS BALANCED SAMPLE ROBUSTNESS
#

robust = group_summary[["group_label", f"{MAIN_SCORE}_mean"]].rename(
    columns={f"{MAIN_SCORE}_mean": "full_sample"}
).merge(
    balanced_summary[["group_label", f"{MAIN_SCORE}_mean"]].rename(
        columns={f"{MAIN_SCORE}_mean": "balanced_sample"}
    ),
    on="group_label",
    how="inner"
)

x = np.arange(len(robust))
width = 0.38

plt.figure(figsize=(10, 5))
plt.bar(x - width / 2, robust["full_sample"], width, label="Full sample")
plt.bar(x + width / 2, robust["balanced_sample"], width, label="Balanced sample")
plt.axhline(0, linewidth=1)
plt.xticks(x, robust["group_label"], rotation=25, ha="right")
plt.ylabel("Mean XLM-R text sentiment")
plt.title("Robustness check: full vs balanced sample")
plt.legend()
savefig("08_full_vs_balanced_sentiment")


#
# 8. TEXT SENTIMENT VS EMOJI AFFECT
#

text_emoji = group_summary[["community", "year", "group_label", f"{MAIN_SCORE}_mean"]].merge(
    emoji_summary[["community", "year", "mean_esr_emoji_score"]],
    on=["community", "year"],
    how="left"
)

x = np.arange(len(text_emoji))
width = 0.38

plt.figure(figsize=(10, 5))
plt.bar(
    x - width / 2,
    text_emoji[f"{MAIN_SCORE}_mean"],
    width,
    label="Text sentiment, XLM-R"
)
plt.bar(
    x + width / 2,
    text_emoji["mean_esr_emoji_score"],
    width,
    label="Emoji affect, ESR"
)
plt.axhline(0, linewidth=1)
plt.xticks(x, text_emoji["group_label"], rotation=25, ha="right")
plt.ylabel("Mean score")
plt.title("Text sentiment and emoji affect are different signals")
plt.legend()
plt.figtext(
    0.5, -0.08,
    "Text sentiment is computed on emoji-removed text. Emoji affect is computed from emojis only.",
    ha="center",
    fontsize=9
)
savefig("09_text_sentiment_vs_emoji_affect")


#
# 9. VALIDATION SAMPLES FOR MANUAL INSPECTION
#

validation_cols = [
    "analysis_id",
    "community",
    "year",
    "title",
    "url",
    "text_for_sentiment",
    "text_for_sentiment_no_emoji",
    "no_emoji_xlm_sentiment_score",
    "no_emoji_xlm_label",
    "no_emoji_vader_compound",
    "no_emoji_vader_label",
    "emoji_count",
    "emojis_extracted",
    "esr_emoji_score_mean",
    "esr_emoji_label",
]

validation_cols = [c for c in validation_cols if c in posts.columns]

samples = []

for (community, year), group in posts.groupby(["community", "year"]):
    most_negative = group.sort_values(MAIN_SCORE).head(10).copy()
    most_negative["sample_type"] = "most_negative_xlm"

    most_positive = group.sort_values(MAIN_SCORE, ascending=False).head(10).copy()
    most_positive["sample_type"] = "most_positive_xlm"

    emoji_posts = group[group["has_emoji"] == True].copy()
    if len(emoji_posts) > 0:
        emoji_positive = emoji_posts.sort_values("esr_emoji_score_mean", ascending=False).head(10).copy()
        emoji_positive["sample_type"] = "most_positive_emoji"

        emoji_negative = emoji_posts.sort_values("esr_emoji_score_mean").head(10).copy()
        emoji_negative["sample_type"] = "most_negative_emoji"

        samples.extend([emoji_positive, emoji_negative])

    samples.extend([most_negative, most_positive])

validation_sample = pd.concat(samples, ignore_index=True)
validation_sample = validation_sample[["sample_type"] + validation_cols]

validation_path = OUTPUT_DIR / "sentiment_manual_validation_samples.csv"
validation_sample.to_csv(validation_path, index=False, encoding="utf-8-sig")
print(f"Saved validation samples: {validation_path}")


#
# 10. FIGURE MANIFEST
#

manifest = pd.DataFrame({
    "figure": [
        "01_main_text_sentiment_full_sample",
        "01_main_text_sentiment_balanced_sample",
        "02_xlm_label_shares_full_sample",
        "02_xlm_label_shares_balanced_sample",
        "03_pooled_text_sentiment_by_year",
        "04_emoji_share_by_group",
        "05_emoji_affect_distribution",
        "06_mean_emoji_sentiment_score",
        "07_vader_vs_xlm_method_comparison",
        "08_full_vs_balanced_sentiment",
        "09_text_sentiment_vs_emoji_affect",
    ],
    "purpose": [
        "Main community-year comparison of multilingual text sentiment.",
        "Balanced-sample robustness version of the main comparison.",
        "Shows negative/neutral/positive shares by community-year.",
        "Balanced-sample version of sentiment-label shares.",
        "Pooled year-level overview; should be interpreted cautiously.",
        "Shows how much emoji usage increased from 2020 to 2025.",
        "Shows whether mapped emojis are positive, negative, or neutral/mixed.",
        "Shows mean explicit emoji affect score.",
        "Shows difference between VADER baseline and XLM-R main method.",
        "Checks whether unequal group sizes drive the result.",
        "Shows text sentiment and emoji affect as separate signals.",
    ]
})

manifest_path = FIG_DIR / "figure_manifest.csv"
manifest.to_csv(manifest_path, index=False, encoding="utf-8-sig")
print(f"Saved figure manifest: {manifest_path}")
print("All visualizations complete.")