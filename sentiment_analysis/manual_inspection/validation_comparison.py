import pandas as pd
import numpy as np
from pathlib import Path

# =========================
# Input / output files
# =========================

MANUAL_FILE = Path(
    r"D:\Users\cheng\Documents\GitHub\SMDA_Financial_Anxiety_on_Reddit\sentiment_analysis\manual_inspection\manual_inspection_blinded_filled.csv"
)

ANSWER_KEY_FILE = Path(
    r"D:\Users\cheng\Documents\GitHub\SMDA_Financial_Anxiety_on_Reddit\sentiment_analysis\manual_inspection\manual_inspection_answer_key.csv"
)

# IMPORTANT:
# Do NOT overwrite validation_result.csv.
# That file was used earlier as your source of filled manual labels.
OUT_CSV = Path(
    r"D:\Users\cheng\Documents\GitHub\SMDA_Financial_Anxiety_on_Reddit\sentiment_analysis\manual_inspection\validation_comparison_result.csv"
)


# =========================
# Helper functions
# =========================

def read_manual_annotations_only(path: Path) -> pd.DataFrame:
    """
    Reads ONLY the first annotation columns from the manual file.

    Expected first columns:
    inspection_id,
    manual_text_sentiment,
    manual_text_sentiment_notes,
    manual_emoji_function,
    manual_emoji_notes,
    ...

    This avoids parsing the long Reddit text columns, which may contain commas,
    quotes, emojis, or other messy CSV content.
    """
    rows = []

    with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
        header = f.readline()

        if not header:
            raise ValueError(f"Manual file appears empty: {path}")

        for line_number, line in enumerate(f, start=2):
            line = line.rstrip("\r\n")

            if not line.strip():
                continue

            # Ignore continuation lines from any accidental embedded line breaks
            if not line.startswith("MI_"):
                continue

            # Split only the first 5 commas.
            # This gives:
            # 0 inspection_id
            # 1 manual_text_sentiment
            # 2 manual_text_sentiment_notes
            # 3 manual_emoji_function
            # 4 manual_emoji_notes
            # 5 rest of row, ignored
            parts = line.split(",", 5)

            if len(parts) < 4:
                print(f"WARNING: Skipping malformed manual line {line_number}: {line[:120]}")
                continue

            rows.append({
                "inspection_id": parts[0].strip(),
                "manual_text_sentiment": parts[1].strip(),
                "manual_emoji_function": parts[3].strip(),
            })

    df = pd.DataFrame(rows)

    if df.empty:
        raise ValueError(f"No manual annotation rows were read from: {path}")

    return df


def read_answer_key(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in [".xlsx", ".xls"]:
        return pd.read_excel(path)

    if path.suffix.lower() == ".csv":
        return pd.read_csv(
            path,
            encoding="utf-8-sig",
            engine="python",
        )

    raise ValueError(f"Unsupported file type: {path}")


def require_columns(df: pd.DataFrame, required_cols, file_name: str):
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"\nColumns found in {file_name}:")
        print(df.columns.tolist())
        raise ValueError(f"{file_name} is missing columns: {missing}")


def as_bool(x):
    if isinstance(x, bool):
        return x

    if pd.isna(x):
        return False

    return str(x).strip().lower() in ["true", "1", "yes", "y"]


def pct(num, den):
    return round(100 * num / den, 2) if den else np.nan


def fmt_pct(x):
    if pd.isna(x):
        return "NA"
    return f"{x:.2f}"


def add_row(
    rows,
    section,
    metric="",
    row_label="",
    column_label="",
    value="",
    numerator="",
    denominator="",
    percent="",
    inspection_id="",
    sample_type="",
    community="",
    year="",
    title="",
    manual_label="",
    model_label="",
    emojis_extracted="",
    model_score="",
    short_text="",
):
    rows.append({
        "section": section,
        "metric": metric,
        "row_label": row_label,
        "column_label": column_label,
        "value": value,
        "numerator": numerator,
        "denominator": denominator,
        "percent": percent,
        "inspection_id": inspection_id,
        "sample_type": sample_type,
        "community": community,
        "year": year,
        "title": title,
        "manual_label": manual_label,
        "model_label": model_label,
        "emojis_extracted": emojis_extracted,
        "model_score": model_score,
        "short_text": short_text,
    })


# =========================
# Load data
# =========================

manual = read_manual_annotations_only(MANUAL_FILE)
answer = read_answer_key(ANSWER_KEY_FILE)

require_columns(
    manual,
    [
        "inspection_id",
        "manual_text_sentiment",
        "manual_emoji_function",
    ],
    "Manual annotation file",
)

require_columns(
    answer,
    [
        "inspection_id",
        "no_emoji_xlm_label",
        "no_emoji_xlm_sentiment_score",
        "esr_emoji_label",
        "esr_emoji_score_mean",
        "has_emoji",
        "sample_type",
        "community",
        "year",
        "title",
    ],
    "Answer-key file",
)

manual["inspection_id"] = manual["inspection_id"].astype(str).str.strip()
answer["inspection_id"] = answer["inspection_id"].astype(str).str.strip()

if manual["inspection_id"].duplicated().any():
    dupes = manual.loc[
        manual["inspection_id"].duplicated(), "inspection_id"
    ].head(20).tolist()
    raise ValueError(f"Duplicate inspection_id values found in manual file: {dupes}")

if answer["inspection_id"].duplicated().any():
    dupes = answer.loc[
        answer["inspection_id"].duplicated(), "inspection_id"
    ].head(20).tolist()
    raise ValueError(f"Duplicate inspection_id values found in answer-key file: {dupes}")

missing_from_answer = sorted(set(manual["inspection_id"]) - set(answer["inspection_id"]))
missing_from_manual = sorted(set(answer["inspection_id"]) - set(manual["inspection_id"]))

if missing_from_answer:
    print(f"WARNING: Manual IDs missing from answer key: {missing_from_answer[:20]}")
    print(f"Total missing from answer key: {len(missing_from_answer)}")

if missing_from_manual:
    print(f"WARNING: Answer-key IDs missing from manual file: {missing_from_manual[:20]}")
    print(f"Total missing from manual file: {len(missing_from_manual)}")


# =========================
# Merge
# =========================

# This is the key fix:
# manual contains ONLY annotation columns.
# answer contains has_emoji, sample_type, community, model labels, etc.
# Therefore pandas will NOT create has_emoji_x / has_emoji_y.
df = manual.merge(
    answer,
    on="inspection_id",
    how="inner",
    validate="one_to_one",
)

if df.empty:
    raise ValueError("Merge produced 0 rows. Check whether inspection_id values match.")

for col in [
    "manual_text_sentiment",
    "manual_emoji_function",
    "no_emoji_xlm_label",
    "esr_emoji_label",
]:
    df[col] = df[col].astype(str).str.strip().str.lower()

df["has_emoji_bool"] = df["has_emoji"].apply(as_bool)

# The answer key does not necessarily contain short_text.
# Use title as a readable fallback in mismatch output.
if "short_text" not in df.columns:
    df["short_text"] = df["title"].fillna("").astype(str)


# =========================
# Text sentiment comparison
# =========================

text_map = {
    "negative": "negative",
    "neutral_or_mixed": "neutral",
    "neutral": "neutral",
    "positive": "positive",
    "unclear": np.nan,
    "": np.nan,
    "nan": np.nan,
}

df["manual_text_norm"] = df["manual_text_sentiment"].map(text_map)
df["xlm_label_norm"] = df["no_emoji_xlm_label"]

valid_text_labels = ["negative", "neutral", "positive"]

df["text_eligible"] = (
    df["manual_text_norm"].isin(valid_text_labels)
    & df["xlm_label_norm"].isin(valid_text_labels)
)

df["xlm_match"] = np.where(
    df["text_eligible"],
    df["manual_text_norm"] == df["xlm_label_norm"],
    np.nan,
)

text_n = int(df["text_eligible"].sum())
text_matches = int(df.loc[df["text_eligible"], "xlm_match"].sum())
text_percent = pct(text_matches, text_n)

text_confusion = pd.crosstab(
    df.loc[df["text_eligible"], "manual_text_sentiment"],
    df.loc[df["text_eligible"], "xlm_label_norm"],
)

text_confusion = text_confusion.reindex(
    index=["negative", "neutral_or_mixed", "positive"],
    columns=["negative", "neutral", "positive"],
    fill_value=0,
)

text_mismatches = df.loc[
    df["text_eligible"] & (df["xlm_match"] == False)
].copy()


# =========================
# Emoji comparison
# =========================

emoji_map = {
    "positive_or_softening": "positive",
    "positive": "positive",
    "negative": "negative",
    "neutral": "neutral",
    "no_emoji": "no_emoji",
    "unclear": np.nan,
    "": np.nan,
    "nan": np.nan,
}

df["manual_emoji_norm"] = df["manual_emoji_function"].map(emoji_map)
df["esr_emoji_norm"] = df["esr_emoji_label"]

valid_emoji_labels = ["positive", "negative", "neutral", "no_emoji"]

df["emoji_eligible"] = (
    df["manual_emoji_norm"].isin(valid_emoji_labels)
    & df["esr_emoji_norm"].isin(valid_emoji_labels)
)

df["emoji_match"] = np.where(
    df["emoji_eligible"],
    df["manual_emoji_norm"] == df["esr_emoji_norm"],
    np.nan,
)

emoji_n = int(df["emoji_eligible"].sum())
emoji_matches = int(df.loc[df["emoji_eligible"], "emoji_match"].sum())
emoji_percent = pct(emoji_matches, emoji_n)

emoji_only = df["emoji_eligible"] & df["has_emoji_bool"]
emoji_only_n = int(emoji_only.sum())
emoji_only_matches = int(df.loc[emoji_only, "emoji_match"].sum())
emoji_only_percent = pct(emoji_only_matches, emoji_only_n)

emoji_confusion = pd.crosstab(
    df.loc[df["emoji_eligible"], "manual_emoji_function"],
    df.loc[df["emoji_eligible"], "esr_emoji_norm"],
)

emoji_confusion = emoji_confusion.reindex(
    index=["no_emoji", "positive_or_softening", "negative", "neutral"],
    columns=["no_emoji", "positive", "negative", "neutral"],
    fill_value=0,
)

emoji_mismatches = df.loc[
    df["emoji_eligible"] & (df["emoji_match"] == False)
].copy()


# =========================
# Build output CSV
# =========================

rows = []

add_row(rows, "summary", "Rows in manual file", value=len(manual))
add_row(rows, "summary", "Rows in answer-key file", value=len(answer))
add_row(rows, "summary", "Rows after merge", value=len(df))

add_row(
    rows,
    "summary",
    "XLM-RoBERTa text agreement",
    value=f"{text_matches}/{text_n}",
    numerator=text_matches,
    denominator=text_n,
    percent=text_percent,
)

add_row(
    rows,
    "summary",
    "Emoji agreement overall",
    value=f"{emoji_matches}/{emoji_n}",
    numerator=emoji_matches,
    denominator=emoji_n,
    percent=emoji_percent,
)

add_row(
    rows,
    "summary",
    "Emoji agreement on emoji-containing posts only",
    value=f"{emoji_only_matches}/{emoji_only_n}",
    numerator=emoji_only_matches,
    denominator=emoji_only_n,
    percent=emoji_only_percent,
)

add_row(rows, "summary", "Text mismatches", value=len(text_mismatches))
add_row(rows, "summary", "Emoji mismatches", value=len(emoji_mismatches))

# Text confusion matrix
for manual_label in text_confusion.index:
    for model_label in text_confusion.columns:
        add_row(
            rows,
            section="text_confusion",
            metric="manual_text_sentiment vs no_emoji_xlm_label",
            row_label=manual_label,
            column_label=model_label,
            value=int(text_confusion.loc[manual_label, model_label]),
        )

# Emoji confusion matrix
for manual_label in emoji_confusion.index:
    for model_label in emoji_confusion.columns:
        add_row(
            rows,
            section="emoji_confusion",
            metric="manual_emoji_function vs esr_emoji_label",
            row_label=manual_label,
            column_label=model_label,
            value=int(emoji_confusion.loc[manual_label, model_label]),
        )

# Text mismatches
for _, r in text_mismatches.iterrows():
    add_row(
        rows,
        section="text_mismatch",
        metric="XLM-RoBERTa mismatch",
        inspection_id=r.get("inspection_id", ""),
        sample_type=r.get("sample_type", ""),
        community=r.get("community", ""),
        year=r.get("year", ""),
        title=r.get("title", ""),
        manual_label=r.get("manual_text_sentiment", ""),
        model_label=r.get("no_emoji_xlm_label", ""),
        model_score=r.get("no_emoji_xlm_sentiment_score", ""),
        short_text=r.get("short_text", ""),
    )

# Emoji mismatches
for _, r in emoji_mismatches.iterrows():
    add_row(
        rows,
        section="emoji_mismatch",
        metric="Emoji model mismatch",
        inspection_id=r.get("inspection_id", ""),
        sample_type=r.get("sample_type", ""),
        community=r.get("community", ""),
        year=r.get("year", ""),
        title=r.get("title", ""),
        manual_label=r.get("manual_emoji_function", ""),
        model_label=r.get("esr_emoji_label", ""),
        emojis_extracted=r.get("emojis_extracted", ""),
        model_score=r.get("esr_emoji_score_mean", ""),
        short_text=r.get("short_text", ""),
    )

paper_sentence = (
    f"In the blinded inspection sample, XLM-RoBERTa matched the reference "
    f"text-sentiment labels in {text_matches}/{text_n} cases "
    f"({fmt_pct(text_percent)}\\%). Emoji-sentiment labels matched the reference "
    f"emoji-function annotations in {emoji_matches}/{emoji_n} cases overall "
    f"({fmt_pct(emoji_percent)}\\%) and in {emoji_only_matches}/{emoji_only_n} "
    f"emoji-containing cases ({fmt_pct(emoji_only_percent)}\\%)."
)

add_row(
    rows,
    section="paper_sentence",
    metric="Suggested LaTeX sentence",
    value=paper_sentence,
)

results = pd.DataFrame(rows)
results.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")


# =========================
# Print results
# =========================

print("\n=== Manual inspection validation summary ===")
print(f"Rows in manual file: {len(manual)}")
print(f"Rows in answer-key file: {len(answer)}")
print(f"Rows after merge: {len(df)}")

print(f"\nXLM-RoBERTa text agreement: {text_matches}/{text_n} = {fmt_pct(text_percent)}%")
print(f"Emoji agreement overall: {emoji_matches}/{emoji_n} = {fmt_pct(emoji_percent)}%")
print(
    f"Emoji agreement on emoji-containing posts only: "
    f"{emoji_only_matches}/{emoji_only_n} = {fmt_pct(emoji_only_percent)}%"
)

print("\n=== Text confusion matrix ===")
print(text_confusion)

print("\n=== Emoji confusion matrix ===")
print(emoji_confusion)

print("\n=== Suggested LaTeX sentence ===")
print(paper_sentence)

print(f"\nSaved output CSV to: {OUT_CSV}")