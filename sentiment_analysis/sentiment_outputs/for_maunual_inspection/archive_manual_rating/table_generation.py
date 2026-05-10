# summarize_manual_validation.py

from pathlib import Path
import pandas as pd
import numpy as np

#
# CONFIG
#

OUTPUT_DIR = Path(r"D:\Users\cheng\Documents\GitHub\SMDA_Financial_Anxiety_on_Reddit\sentiment_analysis\sentiment_outputs\for_maunual_inspection\archive_manual_rating")
# Change this if your rated file has another name
INPUT_FILE = Path(r"D:\Users\cheng\Documents\GitHub\SMDA_Financial_Anxiety_on_Reddit\sentiment_analysis\sentiment_outputs\for_maunual_inspection\archive_manual_rating\manual_inspection_compact_sample_rated.csv")
OUTPUT_XLSX = Path(r"D:\Users\cheng\Documents\GitHub\SMDA_Financial_Anxiety_on_Reddit\sentiment_analysis\sentiment_outputs\for_maunual_inspection\archive_manual_rating\manual_validation_summary.xlsx")
OUTPUT_OVERVIEW_CSV = Path(r"D:\Users\cheng\Documents\GitHub\SMDA_Financial_Anxiety_on_Reddit\sentiment_analysis\sentiment_outputs\for_maunual_inspection\archive_manual_rating\manual_validation_summary_overview.csv")
OUTPUT_DETAILED_CSV = Path(r"D:\Users\cheng\Documents\GitHub\SMDA_Financial_Anxiety_on_Reddit\sentiment_analysis\sentiment_outputs\for_maunual_inspection\archive_manual_rating\manual_validation_detailed_counts.csv")
OUTPUT_BY_GROUP_CSV = Path(r"D:\Users\cheng\Documents\GitHub\SMDA_Financial_Anxiety_on_Reddit\sentiment_analysis\sentiment_outputs\for_maunual_inspection\archive_manual_rating\manual_validation_by_group.csv")


#
# LOAD DATA
#

print(f"Loading: {INPUT_FILE}")

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Could not find input file:\n{INPUT_FILE}\n\n"
        "Check whether the filename is correct."
    )

data = pd.read_csv(INPUT_FILE, low_memory=False)

required_cols = [
    "manual_assessment",
    "xlm_assessment",
    "vader_assessment",
    "emoji_assessment",
    "sample_type",
    "community",
    "year",
]

missing = [col for col in required_cols if col not in data.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

for col in required_cols:
    data[col] = data[col].fillna("").astype(str).str.strip()

n_total = len(data)

if n_total == 0:
    raise ValueError("Input file is empty.")


#
# HELPER FUNCTIONS
#

def make_count_table(column_name, component_name):
    table = (
        data[column_name]
        .replace("", "missing")
        .value_counts(dropna=False)
        .rename_axis("assessment")
        .reset_index(name="count")
    )

    table["share"] = table["count"] / table["count"].sum()
    table["share_percent"] = table["share"] * 100
    table.insert(0, "component", component_name)

    return table


def classify_manual(value):
    value = str(value).lower().strip()

    if value == "plausible":
        return "plausible"
    if value == "plausible_with_emoji_caveat":
        return "plausible_with_caveat"
    if value == "questionable":
        return "questionable"
    if value == "":
        return "missing"

    return "other"


def classify_xlm(value):
    value = str(value).lower().strip()

    if value == "plausible":
        return "plausible"
    if "borderline" in value:
        return "plausible_with_caveat"
    if "questionable" in value:
        return "questionable"
    if value == "":
        return "missing"

    return "other"


def classify_vader(value):
    value = str(value).lower().strip()

    if value == "plausible":
        return "plausible"
    if "misleading" in value:
        return "misleading"
    if value == "":
        return "missing"

    return "other"


def classify_emoji(value):
    value = str(value).lower().strip()

    if value in ["", "no_emoji"]:
        return "no_emoji"

    symbolic_terms = [
        "symbolic",
        "noise",
        "promotional",
    ]

    affective_terms = [
        "meaningful",
        "softening",
        "gratitude",
        "supportive",
        "politeness",
        "literal_or_contextual",
        "contextual_positive",
    ]

    if any(term in value for term in symbolic_terms):
        return "symbolic_or_context_dependent"

    if any(term in value for term in affective_terms):
        return "affective_or_tone_relevant"

    return "other"


def compact_component_summary(component_col, component_name):
    out = (
        data[component_col]
        .value_counts()
        .rename_axis("summary_category")
        .reset_index(name="count")
    )

    out["share"] = out["count"] / out["count"].sum()
    out["share_percent"] = out["share"] * 100
    out.insert(0, "component", component_name)

    return out


#
# DETAILED COUNTS
#

detailed_counts = pd.concat(
    [
        make_count_table("manual_assessment", "Overall manual assessment"),
        make_count_table("xlm_assessment", "XLM-RoBERTa text sentiment"),
        make_count_table("vader_assessment", "VADER baseline"),
        make_count_table("emoji_assessment", "Emoji sentiment"),
    ],
    ignore_index=True,
)

detailed_counts["share_percent"] = detailed_counts["share_percent"].round(2)


#
# COMPACT INTERPRETED CATEGORIES
#

data["manual_summary"] = data["manual_assessment"].apply(classify_manual)
data["xlm_summary"] = data["xlm_assessment"].apply(classify_xlm)
data["vader_summary"] = data["vader_assessment"].apply(classify_vader)
data["emoji_summary"] = data["emoji_assessment"].apply(classify_emoji)

component_summary = pd.concat(
    [
        compact_component_summary("manual_summary", "Overall manual assessment"),
        compact_component_summary("xlm_summary", "XLM-RoBERTa text sentiment"),
        compact_component_summary("vader_summary", "VADER baseline"),
        compact_component_summary("emoji_summary", "Emoji sentiment"),
    ],
    ignore_index=True,
)

component_summary["share_percent"] = component_summary["share_percent"].round(2)


#
# THESIS-STYLE OVERVIEW TABLE
#

xlm_plausible = data["xlm_summary"].isin(
    ["plausible", "plausible_with_caveat"]
).sum()

vader_misleading = data["vader_summary"].eq("misleading").sum()

emoji_affective = data["emoji_summary"].eq("affective_or_tone_relevant").sum()
emoji_symbolic = data["emoji_summary"].eq("symbolic_or_context_dependent").sum()

overview = pd.DataFrame(
    [
        {
            "component_checked": "XLM-RoBERTa text sentiment",
            "main_count": xlm_plausible,
            "main_share_percent": 100 * xlm_plausible / n_total,
            "main_finding": "Mostly plausible",
            "interpretation": "Suitable as the primary sentiment measure.",
        },
        {
            "component_checked": "VADER baseline",
            "main_count": vader_misleading,
            "main_share_percent": 100 * vader_misleading / n_total,
            "main_finding": "Frequently misleading",
            "interpretation": "Useful as a baseline only, not as the main result.",
        },
        {
            "component_checked": "Emoji sentiment",
            "main_count": emoji_affective + emoji_symbolic,
            "main_share_percent": 100 * (emoji_affective + emoji_symbolic) / n_total,
            "main_finding": "Useful but context-dependent",
            "interpretation": "Adds tone information, but symbolic emojis require caution.",
        },
    ]
)

overview["main_share_percent"] = overview["main_share_percent"].round(2)


#
# GROUP-LEVEL SUMMARY
#

count_column = "analysis_id" if "analysis_id" in data.columns else "sample_type"

by_group = (
    data
    .groupby(["community", "year"])
    .agg(
        n_posts=(count_column, "count"),
        manual_plausible=("manual_summary", lambda x: (x == "plausible").sum()),
        manual_plausible_with_caveat=("manual_summary", lambda x: (x == "plausible_with_caveat").sum()),
        manual_questionable=("manual_summary", lambda x: (x == "questionable").sum()),
        xlm_plausible_or_caveat=("xlm_summary", lambda x: x.isin(["plausible", "plausible_with_caveat"]).sum()),
        xlm_questionable=("xlm_summary", lambda x: (x == "questionable").sum()),
        vader_plausible=("vader_summary", lambda x: (x == "plausible").sum()),
        vader_misleading=("vader_summary", lambda x: (x == "misleading").sum()),
        emoji_no_emoji=("emoji_summary", lambda x: (x == "no_emoji").sum()),
        emoji_affective_or_tone_relevant=("emoji_summary", lambda x: (x == "affective_or_tone_relevant").sum()),
        emoji_symbolic_or_context_dependent=("emoji_summary", lambda x: (x == "symbolic_or_context_dependent").sum()),
    )
    .reset_index()
)

share_cols = [
    "manual_plausible",
    "manual_plausible_with_caveat",
    "manual_questionable",
    "xlm_plausible_or_caveat",
    "xlm_questionable",
    "vader_plausible",
    "vader_misleading",
    "emoji_no_emoji",
    "emoji_affective_or_tone_relevant",
    "emoji_symbolic_or_context_dependent",
]

for col in share_cols:
    by_group[col + "_share_percent"] = (100 * by_group[col] / by_group["n_posts"]).round(2)


#
# SAMPLE-TYPE SUMMARY
#

by_sample_type = (
    data
    .groupby("sample_type")
    .agg(
        n_posts=("sample_type", "count"),
        manual_plausible=("manual_summary", lambda x: (x == "plausible").sum()),
        manual_plausible_with_caveat=("manual_summary", lambda x: (x == "plausible_with_caveat").sum()),
        manual_questionable=("manual_summary", lambda x: (x == "questionable").sum()),
        xlm_plausible_or_caveat=("xlm_summary", lambda x: x.isin(["plausible", "plausible_with_caveat"]).sum()),
        xlm_questionable=("xlm_summary", lambda x: (x == "questionable").sum()),
        vader_plausible=("vader_summary", lambda x: (x == "plausible").sum()),
        vader_misleading=("vader_summary", lambda x: (x == "misleading").sum()),
        emoji_no_emoji=("emoji_summary", lambda x: (x == "no_emoji").sum()),
        emoji_affective_or_tone_relevant=("emoji_summary", lambda x: (x == "affective_or_tone_relevant").sum()),
        emoji_symbolic_or_context_dependent=("emoji_summary", lambda x: (x == "symbolic_or_context_dependent").sum()),
    )
    .reset_index()
)

for col in share_cols:
    if col in by_sample_type.columns:
        by_sample_type[col + "_share_percent"] = (
            100 * by_sample_type[col] / by_sample_type["n_posts"]
        ).round(2)


#
# SAVE OUTPUTS
#

overview.to_csv(OUTPUT_OVERVIEW_CSV, index=False, encoding="utf-8-sig")
detailed_counts.to_csv(OUTPUT_DETAILED_CSV, index=False, encoding="utf-8-sig")
by_group.to_csv(OUTPUT_BY_GROUP_CSV, index=False, encoding="utf-8-sig")

try:
    with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as writer:
        overview.to_excel(writer, index=False, sheet_name="overview")
        component_summary.to_excel(writer, index=False, sheet_name="component_summary")
        detailed_counts.to_excel(writer, index=False, sheet_name="detailed_counts")
        by_group.to_excel(writer, index=False, sheet_name="by_group")
        by_sample_type.to_excel(writer, index=False, sheet_name="by_sample_type")

    print(f"Saved Excel file: {OUTPUT_XLSX}")

except Exception as e:
    print("Excel export failed. Install openpyxl if needed:")
    print("python -m pip install openpyxl")
    print(f"Reason: {e}")


#
# PRINT SUMMARY
#

print("\nDone.")
print("=" * 70)
print(f"Total inspected posts: {n_total}")
print()
print("Overview:")
print(overview)
print()
print("Saved files:")
print(f"- {OUTPUT_OVERVIEW_CSV}")
print(f"- {OUTPUT_DETAILED_CSV}")
print(f"- {OUTPUT_BY_GROUP_CSV}")
print(f"- {OUTPUT_XLSX}")
print("=" * 70)