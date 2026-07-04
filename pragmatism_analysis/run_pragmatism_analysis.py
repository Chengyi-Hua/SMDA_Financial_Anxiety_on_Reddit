from __future__ import annotations
from dataclasses import dataclass
from keywords.keywords import PRAGMATISM_LEXICON
from keywords.keywords import EPU_TRIAD_LEXICON
from pathlib import Path
import json
import re
import warnings
import numpy as np
import pandas as pd
try:
    from tqdm.auto import tqdm
except ImportError: 
    def tqdm(iterable, **kwargs):
        return iterable
warnings.filterwarnings("ignore")


@dataclass(frozen=True)
class Config:
    project_dir: Path = Path(r"SMDA_Financial_Anxiety_on_Reddit")
    lang_confidence_min: float = 0.80
    random_seed: int = 42
    bootstrap_iterations: int = 1000
    rate_multiplier: int = 100
    # "fast" uses one compiled regex per category and is much faster.
    # "exact" preserves the old script's term-by-term counting more closely, but is slower.
    matching_mode: str = "exact"  # options: "fast", "exact"
    create_manual_sample: bool = True
    n_manual_per_type: int = 5
    text_col: str = "text_for_sentiment"
    xlm_score_col: str = "no_emoji_xlm_sentiment_score"
    xlm_label_col: str = "no_emoji_xlm_label"
    fallback_files: tuple[str, ...] = (
        "finanzen_2020_final.csv",
        "finanzen_2025_final.csv",
        "personalfinance_2020_final.csv",
        "personalfinance_2025_final.csv",
    )
    @property
    def data_dir(self) -> Path:
        return self.project_dir / "data"
    @property
    def output_dir(self) -> Path:
        return self.project_dir / "pragmatism_analysis" / "outputs"


CFG = Config()
CFG.output_dir.mkdir(parents=True, exist_ok=True)



# BAKER-BLOOM-DAVIS-INSPIRED EPU TRIAD
 
EPU_ORDER = list(EPU_TRIAD_LEXICON.keys())

CATEGORY_ORDER = list(PRAGMATISM_LEXICON.keys())
WORD_CHARS = r"\wäöüÄÖÜß"

 
# TEXT AND REGEX HELPERS
def normalize_whitespace(text: object) -> str:
    if not isinstance(text, str):
        return ""
    return re.sub(r"\s+", " ", text).strip()

def make_short_text(text: object, max_chars: int = 700) -> str:
    text = normalize_whitespace(text)
    return text if len(text) <= max_chars else text[:max_chars] + "..."


def term_to_regex(term: str) -> str:
    term = term.strip()
    if term.startswith("REGEX:"):
        return term.replace("REGEX:", "", 1)

    regex_parts = []
    for part in term.split():
        if part.endswith("*"):
            base = re.escape(part[:-1])
            regex_parts.append(base + rf"[{WORD_CHARS}\-]*")
        else:
            regex_parts.append(re.escape(part))

    body = r"[\s\-]+".join(regex_parts)
    return rf"(?<![{WORD_CHARS}])" + body + rf"(?![{WORD_CHARS}])"


def term_specificity(term: str) -> tuple[int, int, int]:
    clean = term.replace("*", "")
    return (len(term.split()), len(clean), int(not term.endswith("*")))


@dataclass(frozen=True)
class CompiledTerm:
    term: str
    pattern: re.Pattern


@dataclass(frozen=True)
class CompiledCategoryPattern:
    pattern: re.Pattern
    group_to_term: dict[str, str]


def compile_lexicon_exact(lexicon: dict) -> dict[str, list[CompiledTerm]]:
    compiled: dict[str, list[CompiledTerm]] = {}
    for category, info in lexicon.items():
        compiled_terms = []
        for term in info["terms"]:
            try:
                compiled_terms.append(
                    CompiledTerm(term=term, pattern=re.compile(term_to_regex(term), flags=re.IGNORECASE))
                )
            except re.error as exc:
                raise ValueError(f"Invalid regex for term '{term}' in category '{category}': {exc}") from exc
        compiled[category] = compiled_terms
    return compiled


def compile_lexicon_fast(lexicon: dict) -> dict[str, CompiledCategoryPattern]:
    """
    Compiles one regex per category.
    Faster than checking every term separately for every post.

    Note: because this uses regex alternation, overlapping terms that start at the exact
    same character position are counted once, preferring the longest phrase. This is usually
    desirable for cleaner counts, but use matching_mode='exact' for closer replication of
    the original script.
    """
    compiled: dict[str, CompiledCategoryPattern] = {}
    global_idx = 0

    for category, info in lexicon.items():
        terms = sorted(info["terms"], key=term_specificity, reverse=True)
        group_to_term = {}
        alternatives = []

        for term in terms:
            group_name = f"t{global_idx:04d}"
            global_idx += 1
            group_to_term[group_name] = term
            alternatives.append(f"(?P<{group_name}>{term_to_regex(term)})")

        pattern_text = "|".join(alternatives) if alternatives else r"a^"
        try:
            compiled[category] = CompiledCategoryPattern(
                pattern=re.compile(pattern_text, flags=re.IGNORECASE),
                group_to_term=group_to_term,
            )
        except re.error as exc:
            raise ValueError(f"Invalid combined regex for category '{category}': {exc}") from exc

    return compiled


COMPILED_EXACT = compile_lexicon_exact(PRAGMATISM_LEXICON)
COMPILED_FAST = compile_lexicon_fast(PRAGMATISM_LEXICON)
COMPILED_EPU_EXACT = compile_lexicon_exact(EPU_TRIAD_LEXICON)


def score_epu_triad_terms(text: object) -> dict:
    """
    Scores the Baker-Bloom-Davis-inspired EPU triad.
    This is separate from the broader structural pragmatism index.
    """
    text = normalize_whitespace(text)

    result = {}
    matched_terms = {}

    for category in EPU_ORDER:
        category_count = 0
        terms_found = set()

        for item in COMPILED_EPU_EXACT[category]:
            matches = item.pattern.findall(text)
            if matches:
                category_count += len(matches)
                terms_found.add(item.term)

        result[f"{category}_count"] = category_count
        result[f"{category}_has"] = category_count > 0
        matched_terms[category] = terms_found

    result["epu_paper_triad_flag"] = (
        result["epu_economy_count"] > 0
        and result["epu_uncertainty_count"] > 0
        and result["epu_policy_count"] > 0
    )

    # Keep the old column name so the rest of your script still works.
    result["epu_style_triad_flag"] = result["epu_paper_triad_flag"]

    compact_matches = []
    for category in EPU_ORDER:
        terms = sorted(matched_terms.get(category, set()))
        if terms:
            compact_matches.append(f"{category}: " + ", ".join(terms))

    result["epu_paper_matched_terms"] = " | ".join(compact_matches)

    return result


def add_epu_triad_features_to_record(result: dict, text: object) -> dict:
    result.update(score_epu_triad_terms(text))
    return result

 
# DATA LOADING
 
def infer_metadata(filename: str) -> tuple[str, str, int | float]:
    name = filename.lower()
    if "finanzen" in name:
        community, expected_language = "finanzen", "de"
    elif "personalfinance" in name:
        community, expected_language = "personalfinance", "en"
    else:
        community, expected_language = "unknown", "unknown"

    year_match = re.search(r"20\d{2}", name)
    year = int(year_match.group()) if year_match else np.nan
    return community, expected_language, year


def read_csv_safely(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, low_memory=False)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="utf-8-sig", low_memory=False)


def ensure_analysis_id(data: pd.DataFrame) -> pd.DataFrame:
    if "analysis_id" not in data.columns:
        if "id" in data.columns:
            id_part = data["id"].astype(str)
        else:
            id_part = data.index.astype(str)
        data["analysis_id"] = (
            data["community"].astype(str) + "_" + data["year"].astype(str) + "_" + id_part
        )
    return data




def load_from_original_files() -> pd.DataFrame:
    print("Loading cleaned input files from data/.")
    dfs = []

    for filename in CFG.fallback_files:
        path = CFG.data_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        print(f"Loading: {filename}")
        df = read_csv_safely(path)
        community, expected_language, file_year = infer_metadata(filename)
        df["source_file"] = filename
        df["community"] = community
        df["expected_language"] = expected_language
        df["year"] = file_year
        dfs.append(df)

    data = pd.concat(dfs, ignore_index=True)

    required_cols = ["id", CFG.text_col, "detected_language", "language_confidence"]
    missing = [col for col in required_cols if col not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    before = len(data)
    data["language_confidence"] = pd.to_numeric(data["language_confidence"], errors="coerce")
    data = data[data["language_confidence"] >= CFG.lang_confidence_min].copy()
    after = len(data)

    print(f"Language confidence filter >= {CFG.lang_confidence_min}")
    print(f"Rows before: {before}")
    print(f"Rows after: {after}")
    print(f"Rows removed: {before - after}")

    data[CFG.text_col] = data[CFG.text_col].fillna("").astype(str).map(normalize_whitespace)
    data["pragmatism_text"] = data[CFG.text_col]

    if "word_count" not in data.columns:
        data["word_count"] = data["pragmatism_text"].str.split().str.len()

    data = ensure_analysis_id(data)

    if "created_utc" in data.columns and "date" not in data.columns:
        data["date"] = pd.to_datetime(data["created_utc"], unit="s", errors="coerce", utc=True)

    return data


def load_data() -> pd.DataFrame:
    data = load_from_original_files()

    data["community"] = data["community"].astype(str)
    data["year"] = pd.to_numeric(data["year"], errors="coerce").astype("Int64")
    data["pragmatism_text"] = data["pragmatism_text"].fillna("").astype(str).map(normalize_whitespace)
    data["word_count"] = pd.to_numeric(data["word_count"], errors="coerce").fillna(0)
    data["word_count_for_rate"] = data["word_count"].replace(0, np.nan)
    return data

 
# PRAGMATISM SCORING
def empty_score_record() -> dict:
    result = {}
    for category in CATEGORY_ORDER:
        result[f"prag_{category}_count"] = 0
        result[f"prag_{category}_has"] = False
    result.update({
        "prag_total_count": 0,
        "prag_unique_term_count": 0,
        "prag_category_count": 0,
        "has_structural_pragmatism": False,
        "dominant_prag_category": "none",
        "epu_style_triad_flag": False,
        "matched_prag_terms": "",
    })
    return result


def finalize_score_record(result: dict, matched_by_category: dict[str, set[str]]) -> dict:
    total_count = sum(result[f"prag_{category}_count"] for category in CATEGORY_ORDER)
    active_categories = [category for category in CATEGORY_ORDER if result[f"prag_{category}_count"] > 0]
    unique_terms = {f"{category}:{term}" for category, terms in matched_by_category.items() for term in terms}

    result["prag_total_count"] = total_count
    result["prag_unique_term_count"] = len(unique_terms)
    result["prag_category_count"] = len(active_categories)
    result["has_structural_pragmatism"] = total_count > 0

    if total_count > 0:
        category_counts = {category: result[f"prag_{category}_count"] for category in CATEGORY_ORDER}
        max_count = max(category_counts.values())
        dominant = [category for category, count in category_counts.items() if count == max_count and count > 0]
        result["dominant_prag_category"] = dominant[0] if dominant else "none"
    else:
        result["dominant_prag_category"] = "none"

    # This is overwritten later by the stricter Baker-Bloom-Davis-inspired
    # economy + uncertainty + policy triad.
    result["epu_style_triad_flag"] = False

    compact_matches = []
    for category in CATEGORY_ORDER:
        terms = sorted(matched_by_category.get(category, set()))
        if terms:
            compact_matches.append(f"{category}: " + ", ".join(terms))
    result["matched_prag_terms"] = " | ".join(compact_matches)
    return result


def score_text_pragmatism_fast(text: object) -> dict:
    text = normalize_whitespace(text)
    result = empty_score_record()
    matched_by_category = {category: set() for category in CATEGORY_ORDER}

    if not text:
        result = finalize_score_record(result, matched_by_category)
        return add_epu_triad_features_to_record(result, text)

    for category in CATEGORY_ORDER:
        category_pattern = COMPILED_FAST[category]
        category_count = 0

        for match in category_pattern.pattern.finditer(text):
            group_name = match.lastgroup
            if group_name is None:
                continue
            term = category_pattern.group_to_term[group_name]
            matched_by_category[category].add(term)
            category_count += 1

        result[f"prag_{category}_count"] = category_count
        result[f"prag_{category}_has"] = category_count > 0

    result = finalize_score_record(result, matched_by_category)
    return add_epu_triad_features_to_record(result, text)


def score_text_pragmatism_exact(text: object) -> dict:
    text = normalize_whitespace(text)
    result = empty_score_record()
    matched_by_category = {category: set() for category in CATEGORY_ORDER}

    if not text:
        result = finalize_score_record(result, matched_by_category)
        return add_epu_triad_features_to_record(result, text)

    for category in CATEGORY_ORDER:
        category_count = 0
        for item in COMPILED_EXACT[category]:
            matches = item.pattern.findall(text)
            if matches:
                category_count += len(matches)
                matched_by_category[category].add(item.term)

        result[f"prag_{category}_count"] = category_count
        result[f"prag_{category}_has"] = category_count > 0

    result = finalize_score_record(result, matched_by_category)
    return add_epu_triad_features_to_record(result, text)


def score_text_pragmatism(text: object) -> dict:
    if CFG.matching_mode == "fast":
        return score_text_pragmatism_fast(text)
    if CFG.matching_mode == "exact":
        return score_text_pragmatism_exact(text)
    raise ValueError("CFG.matching_mode must be either 'fast' or 'exact'.")


def add_pragmatism_features(data: pd.DataFrame) -> pd.DataFrame:
    print(f"Scoring structural pragmatism using matching_mode='{CFG.matching_mode}'...")

    texts = data["pragmatism_text"].fillna("").astype(str).tolist()
    records = [score_text_pragmatism(text) for text in tqdm(texts, desc="Pragmatism lexicon scoring")]

    features = pd.DataFrame.from_records(records)
    data = pd.concat([data.reset_index(drop=True), features.reset_index(drop=True)], axis=1)

    for category in CATEGORY_ORDER:
        count_col = f"prag_{category}_count"
        rate_col = f"prag_{category}_rate_per_100_words"
        data[rate_col] = safe_rate(data[count_col], data["word_count_for_rate"])

    data["prag_total_rate_per_100_words"] = safe_rate(data["prag_total_count"], data["word_count_for_rate"])
    data["prag_unique_term_rate_per_100_words"] = safe_rate(data["prag_unique_term_count"], data["word_count_for_rate"])
    data["prag_category_diversity"] = data["prag_category_count"] / len(CATEGORY_ORDER)

    data["structural_pragmatism_index"] = (
        np.log1p(data["prag_total_rate_per_100_words"]) * (1 + data["prag_category_diversity"])
    )
    return data


def safe_rate(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return (
        pd.to_numeric(numerator, errors="coerce")
        .div(pd.to_numeric(denominator, errors="coerce"))
        .mul(CFG.rate_multiplier)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )

 
# SUMMARIES

def confidence_interval_95(series: pd.Series) -> tuple[float, float]:
    series = pd.to_numeric(series, errors="coerce").dropna()
    n = len(series)
    if n <= 1:
        return np.nan, np.nan
    sem = series.std(ddof=1) / np.sqrt(n)
    margin = 1.96 * sem
    return float(series.mean() - margin), float(series.mean() + margin)


def add_group_cis(
    data: pd.DataFrame,
    summary: pd.DataFrame,
    group_cols: list[str],
    score_cols: list[str],
) -> pd.DataFrame:
    rows = []
    for keys, group in data.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        for score_col in score_cols:
            ci_low, ci_high = confidence_interval_95(group[score_col])
            row[f"{score_col}_ci_low"] = ci_low
            row[f"{score_col}_ci_high"] = ci_high
        rows.append(row)

    ci_df = pd.DataFrame(rows)
    return summary.merge(ci_df, on=group_cols, how="left")


def make_group_summary(
    data: pd.DataFrame,
    output_name: str = "pragmatism_group_summary.csv"
) -> pd.DataFrame:
    print("Creating group summary...")
    group_cols = ["community", "year"]

    agg_dict = {
        "n_posts": ("analysis_id", "count"),
        "mean_word_count": ("word_count", "mean"),
        "median_word_count": ("word_count", "median"),
        "posts_with_structural_pragmatism": ("has_structural_pragmatism", "sum"),
        "structural_pragmatism_share": ("has_structural_pragmatism", "mean"),
        "epu_style_triad_posts": ("epu_style_triad_flag", "sum"),
        "epu_style_triad_share": ("epu_style_triad_flag", "mean"),

        # Explicit Baker-Bloom-Davis-inspired EPU triad fields
        "epu_paper_triad_posts": ("epu_paper_triad_flag", "sum"),
        "epu_paper_triad_share": ("epu_paper_triad_flag", "mean"),
        "mean_epu_economy_count": ("epu_economy_count", "mean"),
        "mean_epu_uncertainty_count": ("epu_uncertainty_count", "mean"),
        "mean_epu_policy_count": ("epu_policy_count", "mean"),
        "epu_economy_share": ("epu_economy_has", "mean"),
        "epu_uncertainty_share": ("epu_uncertainty_has", "mean"),
        "epu_policy_share": ("epu_policy_has", "mean"),

        "mean_prag_total_count": ("prag_total_count", "mean"),
        "median_prag_total_count": ("prag_total_count", "median"),
        "mean_prag_total_rate": ("prag_total_rate_per_100_words", "mean"),
        "median_prag_total_rate": ("prag_total_rate_per_100_words", "median"),
        "mean_structural_pragmatism_index": ("structural_pragmatism_index", "mean"),
        "median_structural_pragmatism_index": ("structural_pragmatism_index", "median"),
        "mean_prag_category_count": ("prag_category_count", "mean"),
        "median_prag_category_count": ("prag_category_count", "median"),
        "mean_prag_unique_term_count": ("prag_unique_term_count", "mean"),
    }

    for category in CATEGORY_ORDER:
        agg_dict[f"{category}_count_sum"] = (f"prag_{category}_count", "sum")
        agg_dict[f"{category}_has_share"] = (f"prag_{category}_has", "mean")
        agg_dict[f"{category}_rate_mean"] = (f"prag_{category}_rate_per_100_words", "mean")
        agg_dict[f"{category}_rate_median"] = (f"prag_{category}_rate_per_100_words", "median")

    summary = data.groupby(group_cols, dropna=False).agg(**agg_dict).reset_index()
    summary = add_group_cis(
        data,
        summary,
        group_cols,
        [
            "prag_total_rate_per_100_words",
            "structural_pragmatism_index",
            "prag_category_count",
        ],
    )

    output_path = CFG.output_dir / output_name
    summary.to_csv(output_path, index=False, encoding="utf-8-sig")
    return summary


def make_category_summary(data: pd.DataFrame) -> pd.DataFrame:
    print("Creating category summary...")
    rows = []

    for (community, year), group in data.groupby(["community", "year"], dropna=False):
        for category in CATEGORY_ORDER:
            rows.append({
                "community": community,
                "year": year,
                "category": category,
                "description": PRAGMATISM_LEXICON[category]["description"],
                "n_posts": len(group),
                "posts_with_category": int(group[f"prag_{category}_has"].sum()),
                "category_share": float(group[f"prag_{category}_has"].mean()),
                "total_category_count": int(group[f"prag_{category}_count"].sum()),
                "mean_category_count": float(group[f"prag_{category}_count"].mean()),
                "median_category_count": float(group[f"prag_{category}_count"].median()),
                "mean_category_rate_per_100_words": float(group[f"prag_{category}_rate_per_100_words"].mean()),
                "median_category_rate_per_100_words": float(group[f"prag_{category}_rate_per_100_words"].median()),
            })

    category_summary = pd.DataFrame(rows)
    output_path = CFG.output_dir / "pragmatism_category_summary.csv"
    category_summary.to_csv(output_path, index=False, encoding="utf-8-sig")
    return category_summary


def make_dominant_category_shares(data: pd.DataFrame) -> pd.DataFrame:
    print("Creating dominant category shares...")
    dominant = (
        data.groupby(["community", "year"], dropna=False)["dominant_prag_category"]
        .value_counts(normalize=True)
        .rename("share")
        .reset_index()
    )

    dominant_pivot = (
        dominant.pivot_table(
            index=["community", "year"],
            columns="dominant_prag_category",
            values="share",
            fill_value=0,
        )
        .reset_index()
    )

    output_path = CFG.output_dir / "pragmatism_dominant_category_shares.csv"
    dominant_pivot.to_csv(output_path, index=False, encoding="utf-8-sig")
    return dominant_pivot

 
# STATISTICAL TESTS
 
def bootstrap_difference_ci(
    x: pd.Series,
    y: pd.Series,
    statistic: str = "mean",
    n_iter: int = CFG.bootstrap_iterations,
    random_seed: int = CFG.random_seed,
) -> tuple[float, float]:
    rng = np.random.default_rng(random_seed)
    x_arr = pd.to_numeric(pd.Series(x), errors="coerce").dropna().to_numpy()
    y_arr = pd.to_numeric(pd.Series(y), errors="coerce").dropna().to_numpy()

    if len(x_arr) < 2 or len(y_arr) < 2:
        return np.nan, np.nan

    diffs = np.empty(n_iter, dtype=float)
    for i in range(n_iter):
        xb = rng.choice(x_arr, size=len(x_arr), replace=True)
        yb = rng.choice(y_arr, size=len(y_arr), replace=True)
        if statistic == "mean":
            diffs[i] = xb.mean() - yb.mean()
        elif statistic == "median":
            diffs[i] = np.median(xb) - np.median(yb)
        else:
            raise ValueError("statistic must be 'mean' or 'median'")

    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def benjamini_hochberg(p_values: pd.Series) -> pd.Series:
    """Returns Benjamini-Hochberg adjusted p-values."""
    p = pd.to_numeric(p_values, errors="coerce")
    adjusted = pd.Series(np.nan, index=p.index, dtype=float)
    valid = p.dropna()
    if valid.empty:
        return adjusted

    order = valid.sort_values().index
    ranked = valid.loc[order].to_numpy()
    m = len(ranked)
    bh = ranked * m / np.arange(1, m + 1)
    bh = np.minimum.accumulate(bh[::-1])[::-1]
    adjusted.loc[order] = np.clip(bh, 0, 1)
    return adjusted


def year_values_available(data: pd.DataFrame, years: set[int]) -> bool:
    available = set(pd.to_numeric(data["year"], errors="coerce").dropna().astype(int).unique())
    return years.issubset(available)


def get_continuous_score_columns(data: pd.DataFrame) -> list[str]:
    """
    Continuous/count/rate/index outcomes used for pragmatism testing.
    These are expected to be zero-inflated and right-skewed, so
    Mann-Whitney U is the main comparison test.
    """
    score_columns = [
        "prag_total_rate_per_100_words",
        "structural_pragmatism_index",
        "prag_total_count",
        "prag_unique_term_rate_per_100_words",
        "prag_category_count",

        # Baker-Bloom-Davis-inspired EPU component counts
        "epu_economy_count",
        "epu_uncertainty_count",
        "epu_policy_count",
    ] + [f"prag_{category}_rate_per_100_words" for category in CATEGORY_ORDER]

    return [col for col in score_columns if col in data.columns]


def make_distribution_diagnostics(
    data: pd.DataFrame,
    output_name: str = "pragmatism_distribution_diagnostics.csv",
) -> pd.DataFrame:
    """
    Saves distribution diagnostics for pragmatism outcomes.

    This is used to justify the statistical strategy:
    pragmatism variables are count/rate/index variables, often zero-inflated
    and right-skewed, so non-parametric tests are used as the main tests.
    """
    print(f"Creating distribution diagnostics: {output_name}")

    score_columns = get_continuous_score_columns(data)
    rows = []

    for (community, year), group in data.groupby(["community", "year"], dropna=False):
        for score_col in score_columns:
            values = pd.to_numeric(group[score_col], errors="coerce")
            valid = values.dropna()

            if valid.empty:
                continue

            zero_count = int((valid == 0).sum())
            zero_share = float((valid == 0).mean())

            rows.append({
                "community": community,
                "year": year,
                "score": score_col,
                "n": int(valid.shape[0]),
                "missing": int(values.isna().sum()),
                "zero_count": zero_count,
                "zero_share": zero_share,
                "mean": float(valid.mean()),
                "median": float(valid.median()),
                "std": float(valid.std(ddof=1)) if len(valid) > 1 else np.nan,
                "min": float(valid.min()),
                "p25": float(valid.quantile(0.25)),
                "p75": float(valid.quantile(0.75)),
                "p90": float(valid.quantile(0.90)),
                "p95": float(valid.quantile(0.95)),
                "p99": float(valid.quantile(0.99)),
                "max": float(valid.max()),
                "skewness": float(valid.skew()) if len(valid) > 2 else np.nan,
                "zero_inflated_flag": zero_share >= 0.25,
                "recommended_main_test": "Mann-Whitney U with rank-biserial effect size",
            })

    diagnostics = pd.DataFrame(rows)
    output_path = CFG.output_dir / output_name
    diagnostics.to_csv(output_path, index=False, encoding="utf-8-sig")

    return diagnostics


def build_comparison_specs(data: pd.DataFrame) -> list[dict]:
    """
    Builds the comparisons needed for the pragmatism research question.

    Important:
    - within-community comparisons test change over time: 2025 vs 2020
    - between-community comparisons test the main hypothesis:
      finanzen vs personalfinance
    """
    comparison_specs = []

    # Within-community time comparison: 2025 minus 2020
    for community in sorted(data["community"].dropna().unique()):
        subset = data[data["community"] == community]

        if year_values_available(subset, {2020, 2025}):
            comparison_specs.append({
                "comparison_type": "within_community_2025_minus_2020",
                "group_1_label": f"{community}_2025",
                "group_2_label": f"{community}_2020",
                "group_1": subset[subset["year"].eq(2025)],
                "group_2": subset[subset["year"].eq(2020)],
            })

    # Between-community comparison within each year:
    #    finanzen minus personalfinance
    communities_available = set(data["community"].dropna().astype(str).unique())
    if {"finanzen", "personalfinance"}.issubset(communities_available):
        years = sorted(pd.to_numeric(data["year"], errors="coerce").dropna().astype(int).unique())

        for year in years:
            g_finanzen = data[(data["community"].eq("finanzen")) & (data["year"].eq(year))]
            g_personalfinance = data[(data["community"].eq("personalfinance")) & (data["year"].eq(year))]

            if len(g_finanzen) >= 2 and len(g_personalfinance) >= 2:
                comparison_specs.append({
                    "comparison_type": f"between_communities_finanzen_minus_personalfinance_{year}",
                    "group_1_label": f"finanzen_{year}",
                    "group_2_label": f"personalfinance_{year}",
                    "group_1": g_finanzen,
                    "group_2": g_personalfinance,
                })

        # Pooled between-community comparison.
        #    This is composition-sensitive in the full sample, but useful alongside balanced results.
        g_finanzen_all = data[data["community"].eq("finanzen")]
        g_personalfinance_all = data[data["community"].eq("personalfinance")]

        if len(g_finanzen_all) >= 2 and len(g_personalfinance_all) >= 2:
            comparison_specs.append({
                "comparison_type": "between_communities_finanzen_minus_personalfinance_pooled_composition_sensitive",
                "group_1_label": "finanzen_all",
                "group_2_label": "personalfinance_all",
                "group_1": g_finanzen_all,
                "group_2": g_personalfinance_all,
            })

    # Pooled year comparison, kept as secondary/descriptive.
    if year_values_available(data, {2020, 2025}):
        comparison_specs.append({
            "comparison_type": "pooled_2025_minus_2020_composition_sensitive",
            "group_1_label": "all_2025",
            "group_2_label": "all_2020",
            "group_1": data[data["year"].eq(2025)],
            "group_2": data[data["year"].eq(2020)],
        })

    return comparison_specs

def run_continuous_tests(
    data: pd.DataFrame,
    output_name: str = "pragmatism_continuous_statistical_tests.csv"
) -> pd.DataFrame:
    print("Running continuous statistical tests...")

    try:
        from scipy.stats import ks_2samp, mannwhitneyu, ttest_ind
    except ImportError:
        print("scipy not installed. Skipping continuous statistical tests.")
        return pd.DataFrame()

    score_columns = get_continuous_score_columns(data)
    comparison_specs = build_comparison_specs(data)

    rows = []

    for spec in comparison_specs:
        g1 = spec["group_1"]
        g2 = spec["group_2"]

        for score_col in score_columns:
            x = pd.to_numeric(g1[score_col], errors="coerce").dropna()
            y = pd.to_numeric(g2[score_col], errors="coerce").dropna()

            if len(x) < 2 or len(y) < 2:
                continue

            try:
                mw = mannwhitneyu(x, y, alternative="two-sided", method="asymptotic")
                mw_u = float(mw.statistic)
                mw_p = float(mw.pvalue)

                # Positive rank-biserial means group_1 tends to have higher values than group_2.
                rank_biserial = float((2 * mw_u) / (len(x) * len(y)) - 1)
            except Exception:
                mw_u = mw_p = rank_biserial = np.nan

            try:
                ks = ks_2samp(x, y)
                ks_stat = float(ks.statistic)
                ks_p = float(ks.pvalue)
            except Exception:
                ks_stat = ks_p = np.nan

            try:
                welch = ttest_ind(x, y, equal_var=False, nan_policy="omit")
                welch_t = float(welch.statistic)
                welch_p = float(welch.pvalue)
            except Exception:
                welch_t = welch_p = np.nan

            mean_ci_low, mean_ci_high = bootstrap_difference_ci(x, y, statistic="mean")
            median_ci_low, median_ci_high = bootstrap_difference_ci(x, y, statistic="median")

            rows.append({
                "comparison_type": spec["comparison_type"],
                "score": score_col,
                "group_1": spec["group_1_label"],
                "group_2": spec["group_2_label"],
                "n_1": len(x),
                "n_2": len(y),
                "mean_1": float(x.mean()),
                "mean_2": float(y.mean()),
                "mean_difference_1_minus_2": float(x.mean() - y.mean()),
                "mean_diff_ci_low": mean_ci_low,
                "mean_diff_ci_high": mean_ci_high,
                "median_1": float(x.median()),
                "median_2": float(y.median()),
                "median_difference_1_minus_2": float(x.median() - y.median()),
                "median_diff_ci_low": median_ci_low,
                "median_diff_ci_high": median_ci_high,
                "mannwhitney_u": mw_u,
                "mannwhitney_p": mw_p,
                "mannwhitney_p_bh": np.nan,
                "rank_biserial_effect": rank_biserial,
                "ks_statistic": ks_stat,
                "ks_p": ks_p,
                "ks_p_bh": np.nan,
                "welch_t": welch_t,
                "welch_p": welch_p,
                "welch_p_bh": np.nan,
            })

    tests = pd.DataFrame(rows)

    if not tests.empty:
        for raw_col, adjusted_col in [
            ("mannwhitney_p", "mannwhitney_p_bh"),
            ("ks_p", "ks_p_bh"),
            ("welch_p", "welch_p_bh"),
        ]:
            tests[adjusted_col] = (
                tests
                .groupby("comparison_type", group_keys=False)[raw_col]
                .apply(benjamini_hochberg)
            )

    output_path = CFG.output_dir / output_name
    tests.to_csv(output_path, index=False, encoding="utf-8-sig")

    return tests


def cramers_v_from_chi2(chi2: float, table: pd.DataFrame) -> float:
    n = table.to_numpy().sum()
    if n == 0:
        return np.nan
    r, k = table.shape
    denom = n * min(r - 1, k - 1)
    if denom == 0:
        return np.nan
    return float(np.sqrt(chi2 / denom))

def run_categorical_tests(
    data: pd.DataFrame,
    output_name: str = "pragmatism_categorical_statistical_tests.csv"
) -> pd.DataFrame:
    print("Running categorical statistical tests...")

    try:
        from scipy.stats import chi2_contingency
    except ImportError:
        print("scipy not installed. Skipping categorical statistical tests.")
        return pd.DataFrame()

    categorical_columns = [
        "has_structural_pragmatism",
        "epu_paper_triad_flag",
        "epu_economy_has",
        "epu_uncertainty_has",
        "epu_policy_has",
        "dominant_prag_category",
    ] + [f"prag_{category}_has" for category in CATEGORY_ORDER]

    categorical_columns = [col for col in categorical_columns if col in data.columns]
    comparison_specs = build_comparison_specs(data)

    rows = []

    for spec in comparison_specs:
        g1 = spec["group_1"].copy()
        g2 = spec["group_2"].copy()

        g1["comparison_group"] = spec["group_1_label"]
        g2["comparison_group"] = spec["group_2_label"]

        subset = pd.concat([g1, g2], ignore_index=True)

        for col in categorical_columns:
            table = pd.crosstab(subset["comparison_group"], subset[col])

            if table.shape[0] < 2 or table.shape[1] < 2:
                continue

            try:
                chi2, p_value, dof, expected = chi2_contingency(table)
                cramer_v = cramers_v_from_chi2(chi2, table)
                min_expected_count = float(np.min(expected))
            except Exception:
                chi2 = p_value = dof = cramer_v = min_expected_count = np.nan

            share_1 = np.nan
            share_2 = np.nan
            share_difference = np.nan

            # For boolean variables, report the True-share difference.
            if subset[col].dropna().isin([True, False]).all():
                share_1 = float(g1[col].mean())
                share_2 = float(g2[col].mean())
                share_difference = share_1 - share_2

            rows.append({
                "comparison_type": spec["comparison_type"],
                "categorical_variable": col,
                "group_1": spec["group_1_label"],
                "group_2": spec["group_2_label"],
                "n_1": len(g1),
                "n_2": len(g2),
                "share_1": share_1,
                "share_2": share_2,
                "share_difference_1_minus_2": share_difference,
                "chi2": float(chi2) if pd.notna(chi2) else np.nan,
                "p_value": float(p_value) if pd.notna(p_value) else np.nan,
                "p_value_bh": np.nan,
                "degrees_of_freedom": int(dof) if pd.notna(dof) else np.nan,
                "cramers_v": cramer_v,
                "min_expected_count": min_expected_count,
                "table_json": table.to_json(),
            })

    tests = pd.DataFrame(rows)

    if not tests.empty:
        tests["p_value_bh"] = (
            tests
            .groupby("comparison_type", group_keys=False)["p_value"]
            .apply(benjamini_hochberg)
        )

    output_path = CFG.output_dir / output_name
    tests.to_csv(output_path, index=False, encoding="utf-8-sig")

    return tests
 
# RELATIONSHIP WITH SENTIMENT
 

def analyze_sentiment_pragmatism_relationship(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if CFG.xlm_score_col not in data.columns:
        print("No XLM sentiment score found. Skipping sentiment-pragmatism relationship analysis.")
        return pd.DataFrame(), pd.DataFrame()

    print("Analyzing relationship between sentiment and structural pragmatism...")
    try:
        from scipy.stats import pearsonr, spearmanr
    except ImportError:
        print("scipy not installed. Skipping correlation tests.")
        return pd.DataFrame(), pd.DataFrame()

    df = data.copy()
    df[CFG.xlm_score_col] = pd.to_numeric(df[CFG.xlm_score_col], errors="coerce")

    score_cols = [
        "prag_total_rate_per_100_words",
        "structural_pragmatism_index",
        "prag_category_count",
    ] + [f"prag_{category}_rate_per_100_words" for category in CATEGORY_ORDER]

    rows = []
    for (community, year), group in df.groupby(["community", "year"], dropna=False):
        for score_col in score_cols:
            if score_col not in group.columns:
                continue
            valid = group[[CFG.xlm_score_col, score_col]].dropna()
            if len(valid) < 3:
                continue

            try:
                spearman = spearmanr(valid[CFG.xlm_score_col], valid[score_col])
                spearman_r = float(spearman.correlation)
                spearman_p = float(spearman.pvalue)
            except Exception:
                spearman_r = spearman_p = np.nan

            try:
                pearson = pearsonr(valid[CFG.xlm_score_col], valid[score_col])
                pearson_r = float(pearson.statistic)
                pearson_p = float(pearson.pvalue)
            except Exception:
                pearson_r = pearson_p = np.nan

            rows.append({
                "community": community,
                "year": year,
                "sentiment_score": CFG.xlm_score_col,
                "pragmatism_score": score_col,
                "n": len(valid),
                "spearman_r": spearman_r,
                "spearman_p": spearman_p,
                "spearman_p_bh": np.nan,
                "pearson_r": pearson_r,
                "pearson_p": pearson_p,
                "pearson_p_bh": np.nan,
            })

    correlations = pd.DataFrame(rows)
    if not correlations.empty:
        correlations["spearman_p_bh"] = correlations.groupby(["community", "year"], group_keys=False)["spearman_p"].apply(benjamini_hochberg)
        correlations["pearson_p_bh"] = correlations.groupby(["community", "year"], group_keys=False)["pearson_p"].apply(benjamini_hochberg)

    output_path = CFG.output_dir / "pragmatism_sentiment_correlations.csv"
    correlations.to_csv(output_path, index=False, encoding="utf-8-sig")

    if CFG.xlm_label_col in df.columns:
        label_summary = (
            df.groupby(["community", "year", CFG.xlm_label_col], dropna=False)
            .agg(
                n_posts=("analysis_id", "count"),
                mean_prag_total_rate=("prag_total_rate_per_100_words", "mean"),
                median_prag_total_rate=("prag_total_rate_per_100_words", "median"),
                mean_structural_pragmatism_index=("structural_pragmatism_index", "mean"),
                median_structural_pragmatism_index=("structural_pragmatism_index", "median"),
                structural_pragmatism_share=("has_structural_pragmatism", "mean"),
                epu_paper_triad_share=("epu_paper_triad_flag", "mean"),
            )
            .reset_index()
        )
    else:
        label_summary = pd.DataFrame()

    label_output_path = CFG.output_dir / "pragmatism_by_xlm_sentiment_label.csv"
    label_summary.to_csv(label_output_path, index=False, encoding="utf-8-sig")
    return correlations, label_summary

 
# MANUAL INSPECTION SAMPLE
 

def add_manual_sample(samples: list[pd.DataFrame], df: pd.DataFrame, sample_type: str, n: int | None = None) -> None:
    if df.empty:
        return
    n = CFG.n_manual_per_type if n is None else n
    out = df.head(n).copy()
    out["sample_type"] = sample_type
    samples.append(out)


def create_manual_inspection_sample(data: pd.DataFrame) -> pd.DataFrame:
    print("Creating pragmatism manual inspection sample...")
    samples: list[pd.DataFrame] = []

    for (community, year), group in data.groupby(["community", "year"], dropna=False):
        random_sample = group.sample(
            n=min(CFG.n_manual_per_type, len(group)),
            random_state=CFG.random_seed,
        )
        add_manual_sample(samples, random_sample, "random_posts")

        add_manual_sample(
            samples,
            group.sort_values("structural_pragmatism_index", ascending=False),
            "highest_structural_pragmatism",
        )
        add_manual_sample(
            samples,
            group.sort_values("structural_pragmatism_index", ascending=True),
            "lowest_structural_pragmatism",
        )

        epu_posts = group[group["epu_style_triad_flag"]]
        add_manual_sample(
            samples,
            epu_posts.sort_values("structural_pragmatism_index", ascending=False),
            "epu_style_triad_posts",
        )

        for category in CATEGORY_ORDER:
            add_manual_sample(
                samples,
                group.sort_values(f"prag_{category}_rate_per_100_words", ascending=False),
                f"highest_{category}",
            )

        if CFG.xlm_score_col in group.columns:
            tmp = group.copy()
            tmp[CFG.xlm_score_col] = pd.to_numeric(tmp[CFG.xlm_score_col], errors="coerce")
            negative_high_prag = tmp[
                (tmp[CFG.xlm_score_col] < -0.2)
                & (tmp["structural_pragmatism_index"] > tmp["structural_pragmatism_index"].median())
            ]
            add_manual_sample(
                samples,
                negative_high_prag.sort_values("structural_pragmatism_index", ascending=False),
                "negative_sentiment_high_pragmatism",
            )

    if not samples:
        return pd.DataFrame()

    manual = pd.concat(samples, ignore_index=True)
    manual = manual.drop_duplicates(subset=["sample_type", "analysis_id"], keep="first")
    manual["short_text"] = manual["pragmatism_text"].map(make_short_text)
    manual["manual_pragmatism_assessment"] = ""
    manual["manual_notes"] = ""

    front_cols = [
        "sample_type",
        "manual_pragmatism_assessment",
        "manual_notes",
        "analysis_id",
        "community",
        "year",
        "title",
        "url",
        "word_count",
        CFG.xlm_score_col,
        CFG.xlm_label_col,
        "prag_total_count",
        "prag_total_rate_per_100_words",
        "structural_pragmatism_index",
        "prag_category_count",
        "prag_unique_term_count",
        "dominant_prag_category",
        "has_structural_pragmatism",

        # Baker-Bloom-Davis-inspired EPU triad
        "epu_style_triad_flag",
        "epu_paper_triad_flag",
        "epu_economy_count",
        "epu_economy_has",
        "epu_uncertainty_count",
        "epu_uncertainty_has",
        "epu_policy_count",
        "epu_policy_has",
        "epu_paper_matched_terms",

        "matched_prag_terms",
        "short_text",
    ]

    for category in CATEGORY_ORDER:
        front_cols.extend([
            f"prag_{category}_count",
            f"prag_{category}_rate_per_100_words",
            f"prag_{category}_has",
        ])

    front_cols = [col for col in front_cols if col in manual.columns]
    manual = manual[front_cols]

    output_csv = CFG.output_dir / "pragmatism_manual_inspection_sample.csv"
    manual.to_csv(output_csv, index=False, encoding="utf-8-sig")

    output_xlsx = CFG.output_dir / "pragmatism_manual_inspection_sample.xlsx"
    try:
        with pd.ExcelWriter(output_xlsx, engine="openpyxl") as writer:
            manual.to_excel(writer, index=False, sheet_name="all_samples")
            for sample_type, subset in manual.groupby("sample_type"):
                sheet_name = re.sub(r"[^A-Za-z0-9_ -]", "_", str(sample_type))[:31]
                subset.to_excel(writer, index=False, sheet_name=sheet_name)
        print(f"Saved manual inspection Excel file: {output_xlsx}")
    except Exception as exc:
        print("Excel export skipped. Install openpyxl if needed:")
        print("python -m pip install openpyxl")
        print(f"Reason: {exc}")

    print(f"Saved manual inspection CSV file: {output_csv}")
    return manual

 
# SAVE SCORED DATA
 

def save_scored_data(data: pd.DataFrame) -> tuple[Path, Path]:
    print("Saving scored pragmatism data...")

    full_output = CFG.output_dir / "pragmatism_scored_posts_full.csv"
    data.to_csv(full_output, index=False, encoding="utf-8-sig")

    slim_cols = [
        "analysis_id",
        "source_file",
        "community",
        "expected_language",
        "year",
        "date",
        "id",
        "title",
        "url",
        "word_count",
        CFG.xlm_score_col,
        CFG.xlm_label_col,
        "prag_total_count",
        "prag_total_rate_per_100_words",
        "prag_unique_term_count",
        "prag_unique_term_rate_per_100_words",
        "prag_category_count",
        "prag_category_diversity",
        "structural_pragmatism_index",
        "has_structural_pragmatism",
        "dominant_prag_category",

        # Baker-Bloom-Davis-inspired EPU triad
        "epu_style_triad_flag",
        "epu_paper_triad_flag",
        "epu_economy_count",
        "epu_economy_has",
        "epu_uncertainty_count",
        "epu_uncertainty_has",
        "epu_policy_count",
        "epu_policy_has",
        "epu_paper_matched_terms",

        "matched_prag_terms",
    ]

    for category in CATEGORY_ORDER:
        slim_cols.extend([
            f"prag_{category}_count",
            f"prag_{category}_rate_per_100_words",
            f"prag_{category}_has",
        ])

    slim_cols = [col for col in slim_cols if col in data.columns]
    slim_output = CFG.output_dir / "pragmatism_scored_posts_slim.csv"
    data[slim_cols].to_csv(slim_output, index=False, encoding="utf-8-sig")

    return full_output, slim_output

 

def print_group_sizes(data: pd.DataFrame) -> None:
    print("\nLoaded data shape:")
    print(data.shape)
    print("\nGroup sizes:")
    print(data.groupby(["community", "year"], dropna=False).size())


def print_quick_summary(data: pd.DataFrame) -> None:
    print("\nStructural pragmatism usage by group:")
    quick_summary = (
        data.groupby(["community", "year"], dropna=False)
        .agg(
            n_posts=("analysis_id", "count"),
            posts_with_structural_pragmatism=("has_structural_pragmatism", "sum"),
            structural_pragmatism_share=("has_structural_pragmatism", "mean"),
            mean_prag_total_rate=("prag_total_rate_per_100_words", "mean"),
            median_prag_total_rate=("prag_total_rate_per_100_words", "median"),
            mean_structural_pragmatism_index=("structural_pragmatism_index", "mean"),
            median_structural_pragmatism_index=("structural_pragmatism_index", "median"),
            epu_style_triad_share=("epu_style_triad_flag", "mean"),
        )
        .reset_index()
    )
    print(quick_summary)


def make_balanced_sample(data: pd.DataFrame) -> pd.DataFrame:
    print("Creating balanced sample...")

    group_cols = ["community", "year"]
    counts = data.groupby(group_cols).size()
    min_n = counts.min()

    print(f"Smallest group size: {min_n}")
    print(f"Balanced sample size: {min_n} posts per group")

    balanced = (
        data
        .groupby(group_cols, group_keys=False)
        .apply(lambda x: x.sample(n=min_n, random_state=CFG.random_seed))
        .reset_index(drop=True)
    )

    output_path = CFG.output_dir / "pragmatism_balanced_sample.csv"
    balanced.to_csv(output_path, index=False, encoding="utf-8-sig")

    return balanced


def main() -> None:
    print("=" * 80)
    print("OPTIMIZED STRUCTURAL PRAGMATISM ANALYSIS PIPELINE")
    print("=" * 80)

    data = load_data()
    print_group_sizes(data)

    data = add_pragmatism_features(data)
    print_quick_summary(data)

    full_output, slim_output = save_scored_data(data)
    make_distribution_diagnostics(
        data,
        output_name="pragmatism_distribution_diagnostics_full_sample.csv"
    )
    
    # Full-sample summaries/tests
      
    make_group_summary(
        data,
        output_name="pragmatism_group_summary_full_sample.csv"
    )

    make_category_summary(data)
    make_dominant_category_shares(data)

    run_continuous_tests(
        data,
        output_name="pragmatism_continuous_statistical_tests_full_sample.csv"
    )

    run_categorical_tests(
        data,
        output_name="pragmatism_categorical_statistical_tests_full_sample.csv"
    )

      
    # Balanced-sample summaries/tests
      
    balanced = make_balanced_sample(data)
    make_distribution_diagnostics(
        balanced,
        output_name="pragmatism_distribution_diagnostics_balanced_sample.csv"
    )
    make_group_summary(
        balanced,
        output_name="pragmatism_group_summary_balanced.csv"
    )

    run_continuous_tests(
        balanced,
        output_name="pragmatism_continuous_statistical_tests_balanced_sample.csv"
    )

    run_categorical_tests(
        balanced,
        output_name="pragmatism_categorical_statistical_tests_balanced_sample.csv"
    )

    analyze_sentiment_pragmatism_relationship(data)

    if CFG.create_manual_sample:
        create_manual_inspection_sample(data)

    print("\nDone.")
    print("=" * 80)
    print("Main outputs:")
    print(f"Full scored data:              {full_output}")
    print(f"Slim scored data:              {slim_output}")
    print(f"Category summary:              {CFG.output_dir / 'pragmatism_category_summary.csv'}")
    print(f"Dominant category shares:      {CFG.output_dir / 'pragmatism_dominant_category_shares.csv'}")
    print(f"Full-sample summary:           {CFG.output_dir / 'pragmatism_group_summary_full_sample.csv'}")
    print(f"Balanced sample:               {CFG.output_dir / 'pragmatism_balanced_sample.csv'}")
    print(f"Balanced summary:              {CFG.output_dir / 'pragmatism_group_summary_balanced.csv'}")
    print(f"Full continuous tests:         {CFG.output_dir / 'pragmatism_continuous_statistical_tests_full_sample.csv'}")
    print(f"Balanced continuous tests:     {CFG.output_dir / 'pragmatism_continuous_statistical_tests_balanced_sample.csv'}")
    print(f"Full categorical tests:        {CFG.output_dir / 'pragmatism_categorical_statistical_tests_full_sample.csv'}")
    print(f"Balanced categorical tests:    {CFG.output_dir / 'pragmatism_categorical_statistical_tests_balanced_sample.csv'}")
    print(f"Sentiment correlations:        {CFG.output_dir / 'pragmatism_sentiment_correlations.csv'}")
    print(f"Pragmatism by sentiment label: {CFG.output_dir / 'pragmatism_by_xlm_sentiment_label.csv'}")
    print(f"Full distribution diagnostics: {CFG.output_dir / 'pragmatism_distribution_diagnostics_full_sample.csv'}")
    print(f"Balanced distribution diagnostics: {CFG.output_dir / 'pragmatism_distribution_diagnostics_balanced_sample.csv'}")

    if CFG.create_manual_sample:
        print(f"Manual inspection sample CSV:  {CFG.output_dir / 'pragmatism_manual_inspection_sample.csv'}")
        print(f"Manual inspection sample XLSX: {CFG.output_dir / 'pragmatism_manual_inspection_sample.xlsx'}")

    print("=" * 80)


if __name__ == "__main__":
    main()
