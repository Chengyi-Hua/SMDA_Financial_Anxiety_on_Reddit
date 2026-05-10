# manual_inspection_sample.py

from pathlib import Path
import pandas as pd

#
# CONFIG


PROJECT_DIR = Path(r"D:\Users\cheng\Documents\GitHub\SMDA_Financial_Anxiety_on_Reddit\sentiment_analysis")

# Main sentiment output folder
OUTPUT_DIR = PROJECT_DIR  / "sentiment_outputs" 

INPUT_FILE = OUTPUT_DIR / "sentiment_scored_posts_slim.csv"

OUTPUT_CSV = OUTPUT_DIR / "manual_inspection_compact_sample.csv"
OUTPUT_XLSX = OUTPUT_DIR / "manual_inspection_compact_sample.xlsx"

RANDOM_SEED = 42

# Compact sample size per community-year group
N_RANDOM = 3
N_NEGATIVE = 3
N_NEUTRAL = 3
N_EMOJI = 2

MAIN_SCORE = "no_emoji_xlm_sentiment_score"
MAIN_LABEL = "no_emoji_xlm_label"

VADER_SCORE = "no_emoji_vader_compound"
VADER_LABEL = "no_emoji_vader_label"

EMOJI_SCORE = "esr_emoji_score_mean"
EMOJI_LABEL = "esr_emoji_label"

TEXT_COL = "text_for_sentiment"
TEXT_NO_EMOJI_COL = "text_for_sentiment_no_emoji"


#
# LOAD DATA
#

print(f"Loading: {INPUT_FILE}")
data = pd.read_csv(INPUT_FILE, low_memory=False)

data[MAIN_SCORE] = pd.to_numeric(data[MAIN_SCORE], errors="coerce")
data[VADER_SCORE] = pd.to_numeric(data[VADER_SCORE], errors="coerce")

if EMOJI_SCORE in data.columns:
    data[EMOJI_SCORE] = pd.to_numeric(data[EMOJI_SCORE], errors="coerce")

data["has_emoji"] = (
    data["has_emoji"]
    .astype(str)
    .str.lower()
    .isin(["true", "1", "yes"])
)


#
# HELPERS
#

samples = []


def add_sample(df, sample_type):
    if df.empty:
        return

    out = df.copy()
    out["sample_type"] = sample_type
    samples.append(out)


def make_short_text(text, max_chars=700):
    if not isinstance(text, str):
        return ""

    text = " ".join(text.split())

    if len(text) <= max_chars:
        return text

    return text[:max_chars] + "..."


#
# SAMPLE TYPES
#

for (community, year), group in data.groupby(["community", "year"]):

    # 1. Random posts
    add_sample(
        group.sample(n=min(N_RANDOM, len(group)), random_state=RANDOM_SEED),
        "random_posts"
    )

    # 2. Most negative XLM posts
    add_sample(
        group.sort_values(MAIN_SCORE, ascending=True).head(N_NEGATIVE),
        "most_negative_xlm"
    )

    # 3. Most neutral XLM posts: score closest to zero
    neutral = group.copy()
    neutral["abs_xlm_score"] = neutral[MAIN_SCORE].abs()

    add_sample(
        neutral.sort_values("abs_xlm_score", ascending=True).head(N_NEUTRAL),
        "most_neutral_xlm"
    )

    # 4. Emoji posts, if available
    emoji_posts = group[group["has_emoji"] == True].copy()

    if not emoji_posts.empty and EMOJI_SCORE in emoji_posts.columns:
        emoji_posts["abs_emoji_score"] = emoji_posts[EMOJI_SCORE].abs()

        # Emoji posts with strongest mapped emoji affect
        add_sample(
            emoji_posts.sort_values("abs_emoji_score", ascending=False).head(N_EMOJI),
            "strongest_emoji_affect"
        )

        # Text is negative, but emoji is positive/softening
        softening = emoji_posts[
            (emoji_posts[MAIN_SCORE] < -0.2) &
            (emoji_posts[EMOJI_SCORE] > 0.05)
        ].copy()

        add_sample(
            softening.sort_values(MAIN_SCORE, ascending=True).head(N_EMOJI),
            "negative_text_positive_emoji"
        )


#
# COMBINE, CLEAN, SAVE
#

if not samples:
    raise RuntimeError("No samples were created. Check input file and column names.")

manual_sample = pd.concat(samples, ignore_index=True)

# Remove helper columns if present
for helper_col in ["abs_xlm_score", "abs_emoji_score"]:
    if helper_col in manual_sample.columns:
        manual_sample = manual_sample.drop(columns=[helper_col])

# Remove exact duplicate posts within the same sample type
manual_sample = manual_sample.drop_duplicates(
    subset=["sample_type", "analysis_id"],
    keep="first"
)

# Add short text for easier manual inspection
manual_sample["short_text"] = manual_sample[TEXT_COL].apply(make_short_text)

# Add blank columns for manual coding
manual_sample["manual_assessment"] = ""
manual_sample["manual_notes"] = ""

# Keep only necessary columns
front_cols = [
    "sample_type",
    "manual_assessment",
    "manual_notes",
    "analysis_id",
    "community",
    "year",
    "title",
    "url",
    MAIN_SCORE,
    MAIN_LABEL,
    VADER_SCORE,
    VADER_LABEL,
    "has_emoji",
    "emoji_count",
    "emojis_extracted",
    EMOJI_SCORE,
    EMOJI_LABEL,
    "short_text",
    TEXT_COL,
    TEXT_NO_EMOJI_COL,
    "language_confidence",
    "word_count",
]

front_cols = [c for c in front_cols if c in manual_sample.columns]

manual_sample = manual_sample[front_cols]

# Save CSV
manual_sample.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

# Save Excel version
try:
    with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as writer:
        manual_sample.to_excel(writer, index=False, sheet_name="all_samples")

        for sample_type, subset in manual_sample.groupby("sample_type"):
            sheet_name = sample_type[:31]
            subset.to_excel(writer, index=False, sheet_name=sheet_name)

    print(f"Saved Excel file: {OUTPUT_XLSX}")

except Exception as e:
    print("Excel export skipped. Install openpyxl if needed:")
    print("python -m pip install openpyxl")
    print(f"Reason: {e}")

print(f"Saved CSV file: {OUTPUT_CSV}")
print()
print("Sample counts:")
print(manual_sample.groupby(["sample_type", "community", "year"]).size())
print()
print(f"Total rows: {len(manual_sample)}")