# run_gervader_baseline.py

from pathlib import Path
import os
import sys
from contextlib import contextmanager

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT_DIR = Path(r"D:\Users\cheng\Documents\GitHub\SMDA_Financial_Anxiety_on_Reddit")

OUTPUT_DIR = PROJECT_DIR / "sentiment_analysis" / "sentiment_outputs" / "GerVader"
INPUT_FILE =  Path(r"D:\Users\cheng\Documents\GitHub\SMDA_Financial_Anxiety_on_Reddit\sentiment_analysis\sentiment_outputs\sentiment_scored_posts_slim.csv")

GERVADER_DIR = PROJECT_DIR / "sentiment_analysis" / "external" / "GerVADER"

TEXT_COL = "text_for_sentiment_no_emoji"

OUTPUT_SCORED = OUTPUT_DIR / "gervader_scored_finanzen_posts.csv"
OUTPUT_SUMMARY = OUTPUT_DIR / "gervader_finanzen_group_summary.csv"
OUTPUT_LABEL_SHARES = OUTPUT_DIR / "gervader_finanzen_label_shares.csv"
OUTPUT_COMPARISON = OUTPUT_DIR / "gervader_vs_old_vader_finanzen.csv"

OLD_VADER_SCORE = "no_emoji_vader_compound"
OLD_VADER_LABEL = "no_emoji_vader_label"


# ============================================================
# 2. HELPERS
# ============================================================

@contextmanager
def temporary_cwd(path: Path):
    old_cwd = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old_cwd)


def label_from_compound(score):
    if pd.isna(score):
        return np.nan
    if score >= 0.05:
        return "positive"
    if score <= -0.05:
        return "negative"
    return "neutral"


def check_gervader_files():
    required_files = [
        GERVADER_DIR / "vaderSentimentGER.py",
        GERVADER_DIR / "GERVaderLexicon.txt",
        GERVADER_DIR / "emoji_utf8_lexicon.txt",
    ]

    missing = [p for p in required_files if not p.exists()]

    if missing:
        print("GerVADER files are missing.")
        print()
        print("Run this from your project root:")
        print()
        print(r"mkdir sentiment_analysis\external")
        print(
            r"git clone https://github.com/KarstenAMF/GerVADER.git "
            r"sentiment_analysis\external\GerVADER"
        )
        print()
        print("Missing files:")
        for p in missing:
            print(p)
        raise FileNotFoundError("GerVADER setup incomplete.")


def load_gervader_analyzer():
    check_gervader_files()

    sys.path.insert(0, str(GERVADER_DIR))

    # GerVADER expects its lexicon files near the script.
    # Temporarily running from the GerVADER folder avoids path issues.
    with temporary_cwd(GERVADER_DIR):
        from vaderSentimentGER import SentimentIntensityAnalyzer
        analyzer = SentimentIntensityAnalyzer()

    return analyzer


def score_with_gervader(texts, analyzer):
    rows = []

    for text in tqdm(texts, desc="GerVADER finanzen"):
        if not isinstance(text, str):
            text = ""

        try:
            score = analyzer.polarity_scores(text)
        except Exception:
            score = {}

        rows.append({
            "gervader_negative": score.get("neg", np.nan),
            "gervader_neutral": score.get("neu", np.nan),
            "gervader_positive": score.get("pos", np.nan),
            "gervader_compound": score.get("compound", np.nan),
        })

    return pd.DataFrame(rows)


# ============================================================
# 3. MAIN
# ============================================================

def main():
    print("=" * 70)
    print("GERVADER BASELINE FOR GERMAN FINANZEN POSTS")
    print("=" * 70)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

    print(f"Loading: {INPUT_FILE}")
    data = pd.read_csv(INPUT_FILE, low_memory=False)

    required_cols = ["analysis_id", "community", "year", TEXT_COL]
    missing = [col for col in required_cols if col not in data.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    finanzen = data[data["community"].astype(str).str.lower() == "finanzen"].copy()

    if finanzen.empty:
        raise ValueError("No finanzen rows found.")

    finanzen[TEXT_COL] = finanzen[TEXT_COL].fillna("").astype(str)

    print()
    print("German finanzen rows:")
    print(finanzen.groupby(["community", "year"]).size())

    analyzer = load_gervader_analyzer()

    scores = score_with_gervader(
        texts=finanzen[TEXT_COL].tolist(),
        analyzer=analyzer
    )

    finanzen = finanzen.reset_index(drop=True)
    finanzen = pd.concat([finanzen, scores], axis=1)

    finanzen["gervader_label"] = finanzen["gervader_compound"].apply(label_from_compound)

    if OLD_VADER_SCORE in finanzen.columns:
        finanzen[OLD_VADER_SCORE] = pd.to_numeric(
            finanzen[OLD_VADER_SCORE],
            errors="coerce"
        )
        finanzen["gervader_minus_old_english_vader"] = (
            finanzen["gervader_compound"] - finanzen[OLD_VADER_SCORE]
        )

    # --------------------------------------------------------
    # Save row-level GerVADER output
    # --------------------------------------------------------

    print()
    print(f"Saving scored GerVADER file: {OUTPUT_SCORED}")
    finanzen.to_csv(OUTPUT_SCORED, index=False, encoding="utf-8-sig")

    # --------------------------------------------------------
    # Group summary
    # --------------------------------------------------------

    summary_agg = {
        "n_posts": ("analysis_id", "count"),
        "gervader_compound_mean": ("gervader_compound", "mean"),
        "gervader_compound_median": ("gervader_compound", "median"),
        "gervader_compound_std": ("gervader_compound", "std"),
        "gervader_compound_min": ("gervader_compound", "min"),
        "gervader_compound_max": ("gervader_compound", "max"),
        "gervader_negative_mean": ("gervader_negative", "mean"),
        "gervader_neutral_mean": ("gervader_neutral", "mean"),
        "gervader_positive_mean": ("gervader_positive", "mean"),
    }

    if OLD_VADER_SCORE in finanzen.columns:
        summary_agg.update({
            "old_english_vader_compound_mean": (OLD_VADER_SCORE, "mean"),
            "old_english_vader_compound_median": (OLD_VADER_SCORE, "median"),
            "gervader_minus_old_english_vader_mean": (
                "gervader_minus_old_english_vader",
                "mean"
            ),
            "gervader_minus_old_english_vader_median": (
                "gervader_minus_old_english_vader",
                "median"
            ),
        })

    group_summary = (
        finanzen
        .groupby(["community", "year"])
        .agg(**summary_agg)
        .reset_index()
    )

    print(f"Saving group summary: {OUTPUT_SUMMARY}")
    group_summary.to_csv(OUTPUT_SUMMARY, index=False, encoding="utf-8-sig")

    # --------------------------------------------------------
    # Label shares
    # --------------------------------------------------------

    label_shares = (
        finanzen
        .groupby(["community", "year"])["gervader_label"]
        .value_counts(normalize=True)
        .rename("share")
        .reset_index()
        .pivot_table(
            index=["community", "year"],
            columns="gervader_label",
            values="share",
            fill_value=0
        )
        .reset_index()
    )

    print(f"Saving label shares: {OUTPUT_LABEL_SHARES}")
    label_shares.to_csv(OUTPUT_LABEL_SHARES, index=False, encoding="utf-8-sig")

    # --------------------------------------------------------
    # 2025 vs 2020 comparison
    # --------------------------------------------------------

    comparison_rows = []

    if set(finanzen["year"].dropna().unique()) >= {2020, 2025}:
        y2020 = finanzen.loc[finanzen["year"] == 2020, "gervader_compound"].dropna()
        y2025 = finanzen.loc[finanzen["year"] == 2025, "gervader_compound"].dropna()

        comparison_rows.append({
            "community": "finanzen",
            "score": "gervader_compound",
            "n_2020": len(y2020),
            "n_2025": len(y2025),
            "mean_2020": y2020.mean(),
            "mean_2025": y2025.mean(),
            "mean_difference_2025_minus_2020": y2025.mean() - y2020.mean(),
            "median_2020": y2020.median(),
            "median_2025": y2025.median(),
            "median_difference_2025_minus_2020": y2025.median() - y2020.median(),
        })

    comparison = pd.DataFrame(comparison_rows)

    print(f"Saving year comparison: {OUTPUT_COMPARISON}")
    comparison.to_csv(OUTPUT_COMPARISON, index=False, encoding="utf-8-sig")

    print()
    print("Done.")
    print("=" * 70)
    print("Created:")
    print(f"1. {OUTPUT_SCORED}")
    print(f"2. {OUTPUT_SUMMARY}")
    print(f"3. {OUTPUT_LABEL_SHARES}")
    print(f"4. {OUTPUT_COMPARISON}")
    print("=" * 70)


if __name__ == "__main__":
    main()