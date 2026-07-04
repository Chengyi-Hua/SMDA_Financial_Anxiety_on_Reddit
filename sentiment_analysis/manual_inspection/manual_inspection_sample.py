from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(r"SMDA_Financial_Anxiety_on_Reddit\sentiment_analysis")

OUTPUT_DIR = PROJECT_DIR  / "manual_inspection" 
INPUT_FILE = OUTPUT_DIR / "sentiment_scored_posts_slim.csv"

OUTPUT_CSV = OUTPUT_DIR / "manual_inspection_compact_sample.csv"
OUTPUT_XLSX = OUTPUT_DIR / "manual_inspection_compact_sample.xlsx"

BLINDED_CSV = OUTPUT_DIR / "manual_inspection_blinded.csv"
BLINDED_XLSX = OUTPUT_DIR / "manual_inspection_blinded.xlsx"

ANSWER_KEY_CSV = OUTPUT_DIR / "manual_inspection_answer_key.csv"
ANSWER_KEY_XLSX = OUTPUT_DIR / "manual_inspection_answer_key.xlsx"

FULL_INTERNAL_CSV = OUTPUT_DIR / "manual_inspection_internal_full.csv"


RANDOM_SEED = 42

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

# LOAD DATA
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

# HELPERS

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



# SAMPLE TYPES

for (community, year), group in data.groupby(["community", "year"]):

    # Random posts
    add_sample(
        group.sample(n=min(N_RANDOM, len(group)), random_state=RANDOM_SEED),
        "random_posts"
    )

    # Most negative XLM posts
    add_sample(
        group.sort_values(MAIN_SCORE, ascending=True).head(N_NEGATIVE),
        "most_negative_xlm"
    )

    # Most neutral XLM posts: score closest to zero
    neutral = group.copy()
    neutral["abs_xlm_score"] = neutral[MAIN_SCORE].abs()

    add_sample(
        neutral.sort_values("abs_xlm_score", ascending=True).head(N_NEUTRAL),
        "most_neutral_xlm"
    )

    # Emoji posts, if available
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


# COMBINE, CLEAN, SAVE

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

manual_sample["short_text"] = manual_sample[TEXT_COL].apply(make_short_text)

# CREATE BLINDED + ANSWER-KEY FILES
# Randomize row order so the annotator cannot infer sampling type from order
manual_sample = manual_sample.sample(
    frac=1,
    random_state=RANDOM_SEED
).reset_index(drop=True)

# Anonymous ID used to merge annotations later
manual_sample["inspection_id"] = [
    f"MI_{i:04d}" for i in range(1, len(manual_sample) + 1)
]

# Save full internal version for reproducibility only
manual_sample.to_csv(FULL_INTERNAL_CSV, index=False, encoding="utf-8-sig")



# BLINDED FILE FOR ANNOTATOR
blinded = manual_sample.copy()

blinded["manual_text_sentiment"] = ""
blinded["manual_text_sentiment_notes"] = ""
blinded["manual_emoji_function"] = ""
blinded["manual_emoji_notes"] = ""

blinded_cols = [
    "inspection_id",

    # Manual annotation fields
    "manual_text_sentiment",
    "manual_text_sentiment_notes",
    "manual_emoji_function",
    "manual_emoji_notes",

    # Text shown to annotator
    "short_text",
    TEXT_NO_EMOJI_COL,
    TEXT_COL,

    # Emoji context
    "has_emoji",
    "emoji_count",
    "emojis_extracted",
]

blinded_cols = [c for c in blinded_cols if c in blinded.columns]
blinded = blinded[blinded_cols]

blinded.to_csv(BLINDED_CSV, index=False, encoding="utf-8-sig")

try:
    with pd.ExcelWriter(BLINDED_XLSX, engine="openpyxl") as writer:
        blinded.to_excel(writer, index=False, sheet_name="annotation")

        codebook = pd.DataFrame({
            "field": [
                "manual_text_sentiment",
                "manual_emoji_function",
            ],
            "allowed_values": [
                "negative / neutral_or_mixed / positive / unclear",
                "positive_or_softening / negative / neutral / no_emoji / unclear",
            ],
            "instruction": [
                "Judge the affective tone of the post without seeing model outputs.",
                "Judge how emojis function in context, not whether the post itself is anxious.",
            ],
        })

        codebook.to_excel(writer, index=False, sheet_name="codebook")

    print(f"Saved blinded Excel file: {BLINDED_XLSX}")

except Exception as e:
    print("Blinded Excel export skipped. Install openpyxl if needed:")
    print("python -m pip install openpyxl")
    print(f"Reason: {e}")

print(f"Saved blinded CSV file: {BLINDED_CSV}")


# ANSWER KEY FOR LATER ANALYSIS

answer_key_cols = [
    "inspection_id",
    "analysis_id",
    "sample_type",
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

    "language_confidence",
    "word_count",
]

answer_key_cols = [c for c in answer_key_cols if c in manual_sample.columns]
answer_key = manual_sample[answer_key_cols]

answer_key.to_csv(ANSWER_KEY_CSV, index=False, encoding="utf-8-sig")

try:
    answer_key.to_excel(ANSWER_KEY_XLSX, index=False)
    print(f"Saved answer-key Excel file: {ANSWER_KEY_XLSX}")

except Exception as e:
    print("Answer-key Excel export skipped.")
    print(f"Reason: {e}")

print(f"Saved answer-key CSV file: {ANSWER_KEY_CSV}")

print()
print("Internal sample counts:")
print(manual_sample.groupby(["sample_type", "community", "year"]).size())
print()
print(f"Total rows: {len(manual_sample)}")