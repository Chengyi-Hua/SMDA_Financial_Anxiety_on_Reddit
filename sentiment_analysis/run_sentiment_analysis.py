from pathlib import Path
import re
import itertools
import warnings
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import emoji
warnings.filterwarnings("ignore")

DATA_DIR = Path(r"D:\Users\cheng\Documents\GitHub\SMDA_Financial_Anxiety_on_Reddit\data")

OUTPUT_DIR = DATA_DIR / "sentiment_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FILES = [
    "finanzen_2020_final.csv",
    "finanzen_2025_final_with_emo.csv",
    "personalfinance_2020_final_with_emo.csv",
    "personalfinance_2025_final.csv",
]

TEXT_COL = "text_for_sentiment"
LANG_CONFIDENCE_MIN = 0.80

RANDOM_SEED = 42

RUN_XLM_TRANSFORMER = True
XLM_MODEL_NAME = "cardiffnlp/twitter-xlm-roberta-base-sentiment"

# This doubles transformer runtime. Keep False. as it is not main analysis. we have already emoji own analysis
RUN_XLM_EMOJI_included = False

# Transformer settings
MAX_LENGTH = 256
BATCH_SIZE_GPU = 128
BATCH_SIZE_CPU = 16

# Emoji Sentiment Ranking by Kralj Novak et al.
RUN_EMOJI_SENTIMENT_RANKING = True
ESR_URL = "https://kt.ijs.si/data/Emoji_sentiment_ranking/"

# Thresholds for classifying emoji sentiment scores
# ESR score = positive proportion - negative proportion
ESR_POS_THRESHOLD = 0.05
ESR_NEG_THRESHOLD = -0.05


def count_emojis(text):
    if not isinstance(text, str):
        return 0
    return len(emoji.emoji_list(text))


def remove_emojis(text):
    if not isinstance(text, str):
        return ""
    return emoji.replace_emoji(text, replace="")


def normalize_whitespace(text):
    if not isinstance(text, str):
        return ""
    return re.sub(r"\s+", " ", text).strip()


VARIATION_SELECTOR_16 = "\ufe0f"
SKIN_TONE_RE = re.compile("[\U0001F3FB-\U0001F3FF]")


def normalize_emoji_for_lookup(e):

    if not isinstance(e, str):
        return ""

    e = e.replace(VARIATION_SELECTOR_16, "")
    e = SKIN_TONE_RE.sub("", e)
    return e


def extract_emojis(text):
    if not isinstance(text, str):
        return []
    return [entry["emoji"] for entry in emoji.emoji_list(text)]


def classify_esr_score(score):
    if pd.isna(score):
        return "unmapped"
    if score > ESR_POS_THRESHOLD:
        return "positive"
    if score < ESR_NEG_THRESHOLD:
        return "negative"
    return "neutral_or_mixed"


def load_emoji_sentiment_ranking():
    print("Loading Emoji Sentiment Ranking...")

    tables = pd.read_html(ESR_URL)
    if not tables:
        raise RuntimeError("Could not load Emoji Sentiment Ranking table.")

    esr = tables[0].copy()
    esr.columns = [
        "_".join([str(x) for x in col if str(x) != "nan"]).strip()
        if isinstance(col, tuple)
        else str(col).strip()
        for col in esr.columns
    ]

    print("ESR columns detected:")
    print(esr.columns.tolist())
    rename_by_position = {
        esr.columns[0]: "emoji",
        esr.columns[2]: "unicode_codepoint",
        esr.columns[3]: "occurrences",
        esr.columns[4]: "position",
        esr.columns[5]: "neg",
        esr.columns[6]: "neut",
        esr.columns[7]: "pos",
        esr.columns[8]: "sentiment_score",
    }

    esr = esr.rename(columns=rename_by_position)
    possible_name_cols = [c for c in esr.columns if "Unicode name" in c or "name" in c.lower()]
    possible_block_cols = [c for c in esr.columns if "Unicode block" in c or "block" in c.lower()]

    if possible_name_cols:
        esr = esr.rename(columns={possible_name_cols[0]: "unicode_name"})
    else:
        esr["unicode_name"] = np.nan

    if possible_block_cols:
        esr = esr.rename(columns={possible_block_cols[0]: "unicode_block"})
    else:
        esr["unicode_block"] = np.nan

    keep_cols = [
        "emoji",
        "unicode_codepoint",
        "occurrences",
        "position",
        "neg",
        "neut",
        "pos",
        "sentiment_score",
        "unicode_name",
        "unicode_block",
    ]

    esr = esr[[c for c in keep_cols if c in esr.columns]].copy()

    for col in ["occurrences", "position", "neg", "neut", "pos", "sentiment_score"]:
        if col in esr.columns:
            esr[col] = pd.to_numeric(esr[col], errors="coerce")

    esr["emoji"] = esr["emoji"].astype(str)
    esr["emoji_norm"] = esr["emoji"].apply(normalize_emoji_for_lookup)

    esr = esr.dropna(subset=["sentiment_score"])
    esr = esr.drop_duplicates(subset=["emoji_norm"], keep="first")

    print(f"Loaded {len(esr)} emoji sentiment entries.")

    return esr


def build_esr_lookup(esr):
    lookup = {}

    for _, row in esr.iterrows():
        value = {
            "esr_occurrences": row.get("occurrences", np.nan),
            "esr_neg": row.get("neg", np.nan),
            "esr_neut": row.get("neut", np.nan),
            "esr_pos": row.get("pos", np.nan),
            "esr_sentiment_score": row.get("sentiment_score", np.nan),
            "esr_unicode_name": row.get("unicode_name", np.nan),
            "esr_unicode_block": row.get("unicode_block", np.nan),
        }

        lookup[row["emoji"]] = value
        lookup[row["emoji_norm"]] = value

    return lookup


def emoji_sentiment_ranking_features(text, lookup):
    emojis = extract_emojis(text)

    if len(emojis) == 0:
        return pd.Series({
            "esr_total_emoji_count": 0,
            "esr_mapped_emoji_count": 0,
            "esr_unmapped_emoji_count": 0,
            "esr_coverage": np.nan,
            "esr_emoji_score_sum": 0.0,
            "esr_emoji_score_mean": np.nan,
            "esr_positive_emoji_count": 0,
            "esr_negative_emoji_count": 0,
            "esr_neutral_mixed_emoji_count": 0,
            "esr_emoji_label": "no_emoji",
            "emojis_extracted": "",
            "esr_unmapped_emojis": "",
        })

    scores = []
    unmapped = []

    positive_count = 0
    negative_count = 0
    neutral_mixed_count = 0

    for e in emojis:
        e_norm = normalize_emoji_for_lookup(e)

        match = lookup.get(e) or lookup.get(e_norm)

        if match is None:
            unmapped.append(e)
            continue

        score = match["esr_sentiment_score"]
        scores.append(score)

        label = classify_esr_score(score)
        if label == "positive":
            positive_count += 1
        elif label == "negative":
            negative_count += 1
        else:
            neutral_mixed_count += 1

    total_count = len(emojis)
    mapped_count = len(scores)
    unmapped_count = len(unmapped)

    if mapped_count > 0:
        score_sum = float(np.sum(scores))
        score_mean = float(np.mean(scores))
        coverage = mapped_count / total_count

        if score_mean > ESR_POS_THRESHOLD:
            overall_label = "positive"
        elif score_mean < ESR_NEG_THRESHOLD:
            overall_label = "negative"
        else:
            overall_label = "neutral_or_mixed"
    else:
        score_sum = 0.0
        score_mean = np.nan
        coverage = 0.0
        overall_label = "unmapped_only"

    return pd.Series({
        "esr_total_emoji_count": total_count,
        "esr_mapped_emoji_count": mapped_count,
        "esr_unmapped_emoji_count": unmapped_count,
        "esr_coverage": coverage,
        "esr_emoji_score_sum": score_sum,
        "esr_emoji_score_mean": score_mean,
        "esr_positive_emoji_count": positive_count,
        "esr_negative_emoji_count": negative_count,
        "esr_neutral_mixed_emoji_count": neutral_mixed_count,
        "esr_emoji_label": overall_label,
        "emojis_extracted": " ".join(emojis),
        "esr_unmapped_emojis": " ".join(unmapped),
    })


def add_emoji_sentiment_ranking(data):
    print("Adding explicit emoji sentiment features using Emoji Sentiment Ranking...")

    esr = load_emoji_sentiment_ranking()
    lookup = build_esr_lookup(esr)

    features = data[TEXT_COL].fillna("").astype(str).apply(
        lambda text: emoji_sentiment_ranking_features(text, lookup)
    )

    data = pd.concat([data, features], axis=1)

    return data


def infer_metadata(filename):
    name = filename.lower()

    if "finanzen" in name:
        community = "finanzen"
        expected_language = "de"
    elif "personalfinance" in name:
        community = "personalfinance"
        expected_language = "en"
    else:
        community = "unknown"
        expected_language = "unknown"

    year_match = re.search(r"20\d{2}", name)
    year = int(year_match.group()) if year_match else np.nan

    has_emo_in_filename = "with_emo" in name

    return community, expected_language, year, has_emo_in_filename


def read_csv_safely(path):
    try:
        return pd.read_csv(path, low_memory=False)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="utf-8-sig", low_memory=False)


def load_all_data():
    dfs = []

    for filename in FILES:
        path = DATA_DIR / filename
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        print(f"Loading: {filename}")
        df = read_csv_safely(path)

        community, expected_language, file_year, has_emo_in_filename = infer_metadata(filename)

        df["source_file"] = filename
        df["community"] = community
        df["expected_language"] = expected_language
        df["file_year"] = file_year
        df["has_emo_in_filename"] = has_emo_in_filename

        dfs.append(df)

    data = pd.concat(dfs, ignore_index=True)

    required_cols = [
        "created_utc",
        "id",
        "title",
        "detected_language",
        "language_confidence",
        TEXT_COL,
    ]

    missing = [col for col in required_cols if col not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    data[TEXT_COL] = data[TEXT_COL].fillna("").astype(str).apply(normalize_whitespace)

    data["date"] = pd.to_datetime(data["created_utc"], unit="s", errors="coerce", utc=True)
    data["year_from_timestamp"] = data["date"].dt.year
    data["year"] = data["file_year"]

    data["analysis_id"] = (
        data["community"].astype(str)
        + "_"
        + data["year"].astype(str)
        + "_"
        + data["id"].astype(str)
    )

    return data



def add_basic_text_features(data):
    print("Adding basic text features...")

    data["char_count"] = data[TEXT_COL].str.len()
    data["word_count"] = data[TEXT_COL].str.split().str.len()

    data["emoji_count"] = data[TEXT_COL].apply(count_emojis)
    data["has_emoji"] = data["emoji_count"] > 0

    data["text_for_sentiment_no_emoji"] = (
        data[TEXT_COL]
        .apply(remove_emojis)
        .apply(normalize_whitespace)
    )

    return data



def add_vader_sentiment(data, text_col, prefix):
    print(f"Running VADER on: {text_col}")

    import nltk
    nltk.download("vader_lexicon", quiet=True)

    from nltk.sentiment import SentimentIntensityAnalyzer

    sia = SentimentIntensityAnalyzer()

    texts = data[text_col].fillna("").astype(str).tolist()

    scores = []
    for text in tqdm(texts, desc=f"VADER {prefix}"):
        scores.append(sia.polarity_scores(text))

    score_df = pd.DataFrame(scores)

    data[f"{prefix}_vader_negative"] = score_df["neg"].values
    data[f"{prefix}_vader_neutral"] = score_df["neu"].values
    data[f"{prefix}_vader_positive"] = score_df["pos"].values
    data[f"{prefix}_vader_compound"] = score_df["compound"].values

    def label_from_compound(x):
        if x >= 0.05:
            return "positive"
        elif x <= -0.05:
            return "negative"
        else:
            return "neutral"

    data[f"{prefix}_vader_label"] = data[f"{prefix}_vader_compound"].apply(label_from_compound)

    return data



def get_label_indices(model):
    id2label = getattr(model.config, "id2label", {}) or {}

    labels_by_index = []
    for i in range(model.config.num_labels):
        label = id2label.get(i, id2label.get(str(i), f"LABEL_{i}"))
        labels_by_index.append(str(label).lower())

    label_indices = {}

    for target in ["negative", "neutral", "positive"]:
        matches = [i for i, label in enumerate(labels_by_index) if target in label]
        if matches:
            label_indices[target] = matches[0]

    # Fallback for Cardiff-style LABEL_0, LABEL_1, LABEL_2
    if len(label_indices) < 3 and model.config.num_labels == 3:
        label_indices = {
            "negative": 0,
            "neutral": 1,
            "positive": 2,
        }

    if len(label_indices) < 3:
        raise ValueError(
            f"Could not infer negative/neutral/positive labels from model labels: {labels_by_index}"
        )

    return label_indices


def add_xlm_sentiment(data, text_col, prefix):
    print(f"Running multilingual transformer on: {text_col}")

    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = BATCH_SIZE_GPU if device.type == "cuda" else BATCH_SIZE_CPU

    print(f"Device: {device}")
    print(f"Batch size: {batch_size}")
    print(f"Model: {XLM_MODEL_NAME}")

   
    tokenizer = AutoTokenizer.from_pretrained(
        XLM_MODEL_NAME,
        use_fast=False
    )
    model = AutoModelForSequenceClassification.from_pretrained(XLM_MODEL_NAME)
    model.to(device)
    model.eval()

    label_indices = get_label_indices(model)

    texts = data[text_col].fillna("").astype(str).apply(normalize_whitespace).tolist()

    all_probs = []

    for start in tqdm(range(0, len(texts), batch_size), desc=f"XLM {prefix}"):
        batch_texts = texts[start:start + batch_size]

        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )

        encoded = {k: v.to(device) for k, v in encoded.items()}

        with torch.no_grad():
            outputs = model(**encoded)
            probs = torch.softmax(outputs.logits, dim=-1).detach().cpu().numpy()

        all_probs.append(probs)

    probs = np.vstack(all_probs)

    neg_idx = label_indices["negative"]
    neu_idx = label_indices["neutral"]
    pos_idx = label_indices["positive"]

    negative_prob = probs[:, neg_idx]
    neutral_prob = probs[:, neu_idx]
    positive_prob = probs[:, pos_idx]

    data[f"{prefix}_xlm_negative_prob"] = negative_prob
    data[f"{prefix}_xlm_neutral_prob"] = neutral_prob
    data[f"{prefix}_xlm_positive_prob"] = positive_prob

    # Continuous sentiment score, comparable in spirit to VADER compound
    data[f"{prefix}_xlm_sentiment_score"] = positive_prob - negative_prob

    label_array = np.array(["negative", "neutral", "positive"])
    ordered_probs = np.vstack([negative_prob, neutral_prob, positive_prob]).T
    data[f"{prefix}_xlm_label"] = label_array[np.argmax(ordered_probs, axis=1)]

    return data



def confidence_interval_95(series):
    series = series.dropna()
    n = len(series)
    if n <= 1:
        return pd.Series({"ci_low": np.nan, "ci_high": np.nan})

    mean = series.mean()
    sem = series.std(ddof=1) / np.sqrt(n)
    margin = 1.96 * sem

    return pd.Series({
        "ci_low": mean - margin,
        "ci_high": mean + margin,
    })


def make_group_summary(data, score_columns, label_columns, output_name):
    print(f"Creating summary: {output_name}")

    group_cols = ["community", "year"]

    base = (
        data
        .groupby(group_cols)
        .agg(
            n_posts=("id", "count"),
            date_min=("date", "min"),
            date_max=("date", "max"),
            mean_word_count=("word_count", "mean"),
            median_word_count=("word_count", "median"),
            posts_with_emoji=("has_emoji", "sum"),
            emoji_share=("has_emoji", "mean"),
            mean_emoji_count=("emoji_count", "mean"),
        )
        .reset_index()
    )

    for score_col in score_columns:
        if score_col not in data.columns:
            continue

        stats = (
            data
            .groupby(group_cols)[score_col]
            .agg(["mean", "median", "std", "min", "max"])
            .reset_index()
            .rename(columns={
                "mean": f"{score_col}_mean",
                "median": f"{score_col}_median",
                "std": f"{score_col}_std",
                "min": f"{score_col}_min",
                "max": f"{score_col}_max",
            })
        )

        ci = (
            data
            .groupby(group_cols)[score_col]
            .apply(confidence_interval_95)
            .reset_index()
            .pivot_table(
                index=group_cols,
                columns="level_2",
                values=score_col
            )
            .reset_index()
            .rename(columns={
                "ci_low": f"{score_col}_ci_low",
                "ci_high": f"{score_col}_ci_high",
            })
        )

        base = base.merge(stats, on=group_cols, how="left")
        base = base.merge(ci, on=group_cols, how="left")

    for label_col in label_columns:
        if label_col not in data.columns:
            continue

        label_shares = (
            data
            .groupby(group_cols)[label_col]
            .value_counts(normalize=True)
            .rename("share")
            .reset_index()
        )

        label_pivot = (
            label_shares
            .pivot_table(
                index=group_cols,
                columns=label_col,
                values="share",
                fill_value=0
            )
            .reset_index()
        )

        label_pivot = label_pivot.rename(
            columns={
                "negative": f"{label_col}_share_negative",
                "neutral": f"{label_col}_share_neutral",
                "positive": f"{label_col}_share_positive",
            }
        )

        base = base.merge(label_pivot, on=group_cols, how="left")

    output_path = OUTPUT_DIR / output_name
    base.to_csv(output_path, index=False, encoding="utf-8-sig")

    return base


def make_balanced_sample(data):
    print("Creating balanced sample...")

    group_cols = ["community", "year"]
    counts = data.groupby(group_cols).size()
    min_n = counts.min()

    print(f"Smallest group size: {min_n}")
    print(f"Balanced sample size: {min_n} posts per group")

    balanced = (
        data
        .groupby(group_cols, group_keys=False)
        .apply(lambda x: x.sample(n=min_n, random_state=RANDOM_SEED))
        .reset_index(drop=True)
    )

    return balanced


# STATISTICAL TESTS

def run_statistical_tests(data, score_columns, output_name):
    print(f"Running statistical tests: {output_name}")

    try:
        from scipy.stats import kruskal, mannwhitneyu
    except ImportError:
        print("scipy not installed. Skipping statistical tests.")
        return pd.DataFrame()

    results = []

    data = data.copy()
    data["group"] = data["community"].astype(str) + "_" + data["year"].astype(str)

    groups = sorted(data["group"].dropna().unique())

    for score_col in score_columns:
        if score_col not in data.columns:
            continue

        group_values = {
            group: data.loc[data["group"] == group, score_col].dropna().values
            for group in groups
        }

        valid_group_values = [v for v in group_values.values() if len(v) > 1]

        if len(valid_group_values) >= 2:
            stat, p = kruskal(*valid_group_values)

            results.append({
                "test_type": "kruskal_all_groups",
                "score": score_col,
                "group_1": "all",
                "group_2": "all",
                "n_1": np.nan,
                "n_2": np.nan,
                "mean_1": np.nan,
                "mean_2": np.nan,
                "mean_difference_1_minus_2": np.nan,
                "test_statistic": stat,
                "p_value": p,
                "effect_size_rank_biserial": np.nan,
            })

        for g1, g2 in itertools.combinations(groups, 2):
            x = group_values[g1]
            y = group_values[g2]

            if len(x) < 2 or len(y) < 2:
                continue

            stat, p = mannwhitneyu(x, y, alternative="two-sided", method="asymptotic")

            # Rank-biserial effect size based on U statistic.
            # Positive means group_1 tends to have higher values than group_2.
            rank_biserial = (2 * stat) / (len(x) * len(y)) - 1

            results.append({
                "test_type": "mann_whitney_pairwise",
                "score": score_col,
                "group_1": g1,
                "group_2": g2,
                "n_1": len(x),
                "n_2": len(y),
                "mean_1": np.mean(x),
                "mean_2": np.mean(y),
                "mean_difference_1_minus_2": np.mean(x) - np.mean(y),
                "test_statistic": stat,
                "p_value": p,
                "effect_size_rank_biserial": rank_biserial,
            })

    results_df = pd.DataFrame(results)

    if not results_df.empty:
        output_path = OUTPUT_DIR / output_name
        results_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    return results_df



def main():
    print("=" * 70)
    print("SENTIMENT ANALYSIS PIPELINE")
    print("=" * 70)

    data = load_all_data()

    print("\nRaw data shape:")
    print(data.shape)

    print("\nRaw group sizes:")
    print(data.groupby(["community", "year"]).size())

    data = add_basic_text_features(data)

    before = len(data)
    data = data[data["language_confidence"] >= LANG_CONFIDENCE_MIN].copy()
    after = len(data)

    if RUN_EMOJI_SENTIMENT_RANKING:
        data = add_emoji_sentiment_ranking(data)

    print(f"\nLanguage confidence filter >= {LANG_CONFIDENCE_MIN}")
    print(f"Rows before: {before}")
    print(f"Rows after:  {after}")
    print(f"Rows removed: {before - after}")

    print("\nGroup sizes after filtering:")
    print(data.groupby(["community", "year"]).size())

    print("\nEmoji usage by group:")

    emoji_summary_agg = {
        "n_posts": ("id", "count"),
        "posts_with_emoji": ("has_emoji", "sum"),
        "emoji_share": ("has_emoji", "mean"),
        "total_emoji_count": ("emoji_count", "sum"),
        "mean_emoji_count": ("emoji_count", "mean"),
    }

    if RUN_EMOJI_SENTIMENT_RANKING:
        emoji_summary_agg.update({
            "esr_total_emoji_count": ("esr_total_emoji_count", "sum"),
            "esr_mapped_emoji_count": ("esr_mapped_emoji_count", "sum"),
            "esr_unmapped_emoji_count": ("esr_unmapped_emoji_count", "sum"),
            "mean_esr_coverage": ("esr_coverage", "mean"),
            "positive_emoji_count": ("esr_positive_emoji_count", "sum"),
            "negative_emoji_count": ("esr_negative_emoji_count", "sum"),
            "neutral_mixed_emoji_count": ("esr_neutral_mixed_emoji_count", "sum"),
            "mean_esr_emoji_score": ("esr_emoji_score_mean", "mean"),
            "median_esr_emoji_score": ("esr_emoji_score_mean", "median"),
        })

    emoji_summary = (
        data
        .groupby(["community", "year"])
        .agg(**emoji_summary_agg)
        .reset_index()
    )

    print(emoji_summary)

    emoji_summary.to_csv(
        OUTPUT_DIR / "emoji_summary.csv",
        index=False,
        encoding="utf-8-sig"
    )

    if RUN_EMOJI_SENTIMENT_RANKING:
        emoji_label_shares = (
            data
            .groupby(["community", "year"])["esr_emoji_label"]
            .value_counts(normalize=True)
            .rename("share")
            .reset_index()
            .pivot_table(
                index=["community", "year"],
                columns="esr_emoji_label",
                values="share",
                fill_value=0
            )
            .reset_index()
        )

        emoji_label_shares.to_csv(
            OUTPUT_DIR / "emoji_label_shares.csv",
            index=False,
            encoding="utf-8-sig"
        )


    data = add_vader_sentiment(
        data=data,
        text_col=TEXT_COL,
        prefix="with_emoji"
    )

    data = add_vader_sentiment(
        data=data,
        text_col="text_for_sentiment_no_emoji",
        prefix="no_emoji"
    )

    # Multilingual transformer
    if RUN_XLM_TRANSFORMER:
        # Main multilingual text sentiment: emojis removed
        data = add_xlm_sentiment(
            data=data,
            text_col="text_for_sentiment_no_emoji",
            prefix="no_emoji"
        )
        # Optional robustness: full post including emojis, but not needed
        if RUN_XLM_EMOJI_included:
            data = add_xlm_sentiment(
                data=data,
                text_col=TEXT_COL,
                prefix="with_emoji"
            )

    print("\nSaving scored data.")

    full_output_path = OUTPUT_DIR / "sentiment_scored_posts_full.csv"
    data.to_csv(full_output_path, index=False, encoding="utf-8-sig")

    slim_cols = [
        "analysis_id",
        "source_file",
        "community",
        "expected_language",
        "year",
        "date",
        "id",
        "author_fullname",
        "detected_language",
        "language_confidence",
        "title",
        "url",
        TEXT_COL,
        "text_for_sentiment_no_emoji",
        "word_count",
        "char_count",
        "emoji_count",
        "has_emoji",
        "esr_total_emoji_count",
        "esr_mapped_emoji_count",
        "esr_unmapped_emoji_count",
        "esr_coverage",
        "esr_emoji_score_sum",
        "esr_emoji_score_mean",
        "esr_positive_emoji_count",
        "esr_negative_emoji_count",
        "esr_neutral_mixed_emoji_count",
        "esr_emoji_label",
        "emojis_extracted",
        "esr_unmapped_emojis",
        "with_emoji_vader_negative",
        "with_emoji_vader_neutral",
        "with_emoji_vader_positive",
        "with_emoji_vader_compound",
        "with_emoji_vader_label",
        "no_emoji_vader_compound",
        "no_emoji_vader_label",
        "with_emoji_xlm_negative_prob",
        "with_emoji_xlm_neutral_prob",
        "with_emoji_xlm_positive_prob",
        "with_emoji_xlm_sentiment_score",
        "with_emoji_xlm_label",
        "no_emoji_xlm_negative_prob",
        "no_emoji_xlm_neutral_prob",
        "no_emoji_xlm_positive_prob",
        "no_emoji_xlm_sentiment_score",
        "no_emoji_xlm_label",
    ]

    slim_cols = [col for col in slim_cols if col in data.columns]

    slim_output_path = OUTPUT_DIR / "sentiment_scored_posts_slim.csv"
    data[slim_cols].to_csv(slim_output_path, index=False, encoding="utf-8-sig")
    score_columns = [
        "with_emoji_vader_compound",
        "no_emoji_vader_compound",
        "with_emoji_xlm_sentiment_score",
        "no_emoji_xlm_sentiment_score",
        "esr_emoji_score_mean",
    ]
    score_columns = [col for col in score_columns if col in data.columns]

    label_columns = [
        "with_emoji_vader_label",
        "no_emoji_vader_label",
        "with_emoji_xlm_label",
        "no_emoji_xlm_label",
    ]
    label_columns = [col for col in label_columns if col in data.columns]

    group_summary = make_group_summary(
        data=data,
        score_columns=score_columns,
        label_columns=label_columns,
        output_name="sentiment_group_summary.csv"
    )

    balanced = make_balanced_sample(data)

    balanced_output_path = OUTPUT_DIR / "sentiment_balanced_sample.csv"
    balanced[slim_cols].to_csv(balanced_output_path, index=False, encoding="utf-8-sig")

    balanced_summary = make_group_summary(
        data=balanced,
        score_columns=score_columns,
        label_columns=label_columns,
        output_name="sentiment_group_summary_balanced.csv"
    )

    run_statistical_tests(
        data=data,
        score_columns=score_columns,
        output_name="sentiment_statistical_tests_full_sample.csv"
    )

    run_statistical_tests(
        data=balanced,
        score_columns=score_columns,
        output_name="sentiment_statistical_tests_balanced_sample.csv"
    )

    print("\nDone.")
    print("=" * 70)
    print("Main outputs:")
    print(f"Full scored data:       {full_output_path}")
    print(f"Slim scored data:       {slim_output_path}")
    print(f"Group summary:          {OUTPUT_DIR / 'sentiment_group_summary.csv'}")
    print(f"Balanced summary:       {OUTPUT_DIR / 'sentiment_group_summary_balanced.csv'}")
    print(f"Emoji summary:          {OUTPUT_DIR / 'emoji_summary.csv'}")
    print(f"Full-sample tests:      {OUTPUT_DIR / 'sentiment_statistical_tests_full_sample.csv'}")
    print(f"Balanced-sample tests:  {OUTPUT_DIR / 'sentiment_statistical_tests_balanced_sample.csv'}")
    print("=" * 70)


if __name__ == "__main__":
    main()