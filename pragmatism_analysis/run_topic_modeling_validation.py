from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import re
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics import silhouette_score
from scipy.stats import chi2_contingency
warnings.filterwarnings("ignore")

@dataclass(frozen=True)
class Config:
    project_dir: Path = Path(
        r"SMDA_Financial_Anxiety_on_Reddit"
    )

    input_filename: str = "pragmatism_balanced_sample.csv"

    random_seed: int = 42

    # Try 25–40. For your sample size, 30 is a good starting point.
    n_topics: int = 30

    min_chars: int = 25
    max_features: int = 80_000
    min_df: int = 5
    max_df: float = 0.75
    ngram_range: tuple[int, int] = (1, 2)

    # Set to e.g. 2000 for testing. Keep None for final run.
    max_docs_per_group: int | None = None

    use_sentence_transformers_if_available: bool = True
    embedding_model_name: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

    @property
    def prag_output_dir(self) -> Path:
        return self.project_dir / "pragmatism_analysis" / "outputs"

    @property
    def input_path(self) -> Path:
        return self.prag_output_dir / self.input_filename

    @property
    def output_dir(self) -> Path:
        return self.project_dir / "topic_modeling_validation" / "outputs"

    @property
    def table_dir(self) -> Path:
        return self.output_dir / "tables"

    @property
    def figure_dir(self) -> Path:
        return self.output_dir / "figures"


CFG = Config()

CFG.output_dir.mkdir(parents=True, exist_ok=True)
CFG.table_dir.mkdir(parents=True, exist_ok=True)
CFG.figure_dir.mkdir(parents=True, exist_ok=True)



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
    "unmapped_or_general": "Unmapped / general topic",
}

COMMUNITY_ORDER = ["finanzen", "personalfinance"]
YEAR_ORDER = [2020, 2025]


  
# MAPPING SEED TERMS

SEED_LEXICON = {
    "policy_regulation": [
        "tax", "taxes", "irs", "deduction", "law", "legal", "regulation",
        "policy", "government", "steuer", "steuern", "finanzamt", "gesetz",
        "bafin", "ezb", "regelung", "rente",
    ],
    "macro_economic_context": [
        "inflation", "recession", "economy", "interest rate", "rates",
        "crisis", "unemployment", "labor market", "market crash",
        "inflation", "rezession", "wirtschaft", "zinsen", "leitzins",
        "krise", "arbeitsmarkt", "ezb",
    ],
    "financial_institutions_products": [
        "bank", "account", "broker", "credit card", "loan", "mortgage",
        "insurance", "401k", "ira", "hsa", "debt", "bank", "konto",
        "girokonto", "depot", "broker", "kreditkarte", "kredit",
        "versicherung", "schufa", "sparkasse", "dkb", "ing",
    ],
    "investment_strategy": [
        "invest", "investment", "stock", "stocks", "etf", "index fund",
        "portfolio", "allocation", "diversification", "dividend",
        "compound", "vanguard", "fidelity", "aktie", "aktien", "etf",
        "msci", "portfolio", "sparplan", "rendite", "dividende",
        "a1jx52", "a2pkxg", "ftse",
    ],
    "planning_budgeting_control": [
        "budget", "planning", "plan", "track", "expense", "expenses",
        "income", "saving", "savings", "emergency fund", "rent",
        "monthly", "spreadsheet", "budget", "planung", "ausgaben",
        "einnahmen", "einkommen", "sparen", "sparrate", "miete",
        "haushaltsbuch", "notgroschen",
    ],
    "risk_uncertainty_management": [
        "risk", "uncertainty", "volatile", "volatility", "liquidity",
        "safe", "safety", "buffer", "emergency", "loss", "risiko",
        "unsicherheit", "volatilität", "liquidität", "sicherheit",
        "puffer", "verlust", "crash",
    ],
    "practical_problem_solving": [
        "problem", "solution", "fix", "reduce", "switch", "repay",
        "pay off", "fee", "cancel", "negotiate", "refinance",
        "problem", "lösung", "wechseln", "kündigen", "kosten senken",
        "tilgen", "schulden", "gebühr", "umschulden",
    ],
    "opportunity_action_orientation": [
        "opportunity", "chance", "improve", "salary", "raise",
        "promotion", "career", "wealth", "net worth", "fire",
        "financial freedom", "chance", "verbessern", "gehalt",
        "karriere", "vermögen", "finanzielle freiheit",
    ],
}



  
# 3A. EXTRA SEED TERMS FOR TOPIC-TO-CATEGORY MAPPING
# These extra terms are only used to map discovered topics back to
# structural-pragmatism categories. They do not change the main
# pragmatism lexicon analysis.

SEED_LEXICON["financial_institutions_products"].extend([
    "paypal", "bankkonto", "c24", "scalable", "scalable capital",
    "trade republic", "flatex", "comdirect",
    "car loan", "auto loan", "vehicle loan",
    "payment", "payments", "balance",
    "brokerage", "brokerage account",
    "mortgage", "home loan",
])

SEED_LEXICON["planning_budgeting_control"].extend([
    "house", "home", "rent", "apartment", "property", "buying",
    "haus", "wohnung", "immobilie", "miete", "kaufen",
    "eigenkapital", "rate", "raten", "monatlich",
    "salary", "wage", "gehalt", "netto", "income", "expenses",
])

SEED_LEXICON["practical_problem_solving"].extend([
    "pay", "paying", "payment", "payments",
    "refinance", "consolidate", "transfer",
    "wechseln", "übertragen", "uebertragen",
    "kündigen", "kuendigen",
])



GERMAN_STOPWORDS = {
    "aber", "alle", "alles", "als", "also", "am", "an", "auch", "auf", "aus",
    "bei", "bin", "bis", "bist", "da", "damit", "dann", "das", "dass", "dein",
    "deine", "dem", "den", "der", "des", "die", "diese", "dieser", "doch",
    "du", "durch", "ein", "eine", "einem", "einen", "einer", "eines", "er",
    "es", "etwas", "für", "gegen", "habe", "haben", "hat", "hatte", "hier",
    "ich", "ihm", "ihn", "ihr", "ihre", "im", "in", "ist", "ja", "kein",
    "keine", "kann", "man", "meine", "mit", "muss", "nach", "nicht", "noch",
    "nur", "oder", "sein", "seine", "sich", "sie", "sind", "so", "über",
    "um", "und", "uns", "unter", "viel", "vom", "von", "vor", "war", "was",
    "weil", "wenn", "werde", "werden", "wie", "wir", "wird", "wo", "zu",
    "zum", "zur",
}

REDDIT_STOPWORDS = {
    "reddit", "subreddit", "post", "comment", "comments", "thread", "edit",
    "update", "thanks", "thank", "hi", "hello", "hey", "frage", "danke",
    "hallo", "deleted", "removed", "https", "http", "www", "com", "de",
    "finanzen", "personalfinance",
}

  
# 3B. EXTRA TOPIC-MODELING NOISE STOPWORDS
# These are NOT financial-theory stopwords.
# They are only used for topic modeling so that generic Reddit/German filler
# words do not become artificial topics.

TOPIC_NOISE_STOPWORDS = {
    # ----------------------------
    # German pronouns / filler / modal words seen in noisy topics
    # ----------------------------
    "mir", "mich", "mein", "meine", "meiner", "meinem", "meinen",
    "dir", "dich", "dein", "deine", "deiner", "deinem", "deinen",
    "jemand", "niemand", "man", "mal", "halt", "eben", "eh",
    "schon", "jetzt", "heute", "morgen", "gestern", "immer",
    "wieder", "mehr", "weniger", "viel", "viele", "vielen",
    "gibt", "geben", "geht", "gehen", "kommt", "kommen",
    "macht", "machen", "würde", "wuerde", "würden", "wuerden",
    "möchte", "moechte", "möchten", "moechten", "könnte", "koennte",
    "sollte", "sollten", "müsste", "muesste",
    "nun", "warum", "vielleicht", "eigentlich", "ziemlich",
    "einfach", "wirklich", "aktuell", "aktuell", "aktuelle",
    "dazu", "ohne", "dank", "zeit", "thema", "themen",

    # ----------------------------
    # German community/discussion artifacts
    # ----------------------------
    "euch", "euer", "eure", "eurem", "euren", "eurer",
    "habt", "hast", "hätte", "haette", "hatte", "hatten",
    "finanzdiskussion", "womit", "woche", "wochen",
    "fortschritte", "probleme", "aufgekommen", "themenverwandte",
    "diskutieren", "gewählten", "gewaehlten", "ziel", "ziele",
    "beschäftigt", "beschaeftigt", "erfahrungen", "erfahrung",
    "frage", "fragen", "antwort", "antworten", "thread", "kommentar",
    "kommentare", "beitrag", "beiträge", "beitraege",

    # ----------------------------
    # URL / scraping / formatting artifacts
    # ----------------------------
    "url", "urls", "link", "links", "http", "https", "www",
    "amp", "nbsp", "com", "org", "net", "html", "pdf",
    "deleted", "removed", "gelöscht", "geloescht",

    # ----------------------------
    # English generic Reddit / conversational filler
    # ----------------------------
    "just", "really", "like", "want", "wants", "wanted",
    "know", "knows", "think", "thinking", "thought",
    "got", "get", "gets", "getting", "said", "called",
    "received", "use", "using", "used", "make", "makes",
    "making", "going", "trying", "looking", "advice",
    "question", "thanks", "thank", "hello", "hi", "hey",
    "people", "person", "time", "thing", "things",
    "way", "probably", "maybe", "basically", "actually",

    # ----------------------------
    # Domain-general finance words that are too broad for topic modeling
    # Keep these ONLY in topic modeling stopwords, not in pragmatism scoring.
    # ----------------------------
    "money", "geld", "euro", "eur", "dollar", "usd",
    "financial", "finance", "personal", "personalfinance",
    "finanzen",

    # ----------------------------
    # Generic time words that made broad topics less interpretable
    # ----------------------------
    "year", "years", "month", "months", "day", "days",
    "week", "weeks", "jahr", "jahre", "monat", "monate",
    "tag", "tage",

    # ----------------------------
    # Extra German filler terms 
    # ----------------------------
    "zusammen", "welche", "welcher", "welches", "welchem", "welchen",
    "gerne", "hab", "gerade", "sehr", "wäre", "waere",
    "gut", "paar", "beim", "weiß", "weiss", "würdet", "wuerdet",
    "jahren", "seit", "soll", "sollen", "wollen", "wollt",
    "dort", "wurde", "wurden", "dieses", "diese", "dieser",
    "zahlen", "gezahlt", "monatlich", "kosten", "netto",

    # ----------------------------
    # Extra English filler terms still dominating broad topics
    # ----------------------------
    "don", "does", "did", "need", "help", "good", "best", "better",
    "new", "current", "currently",
}

STOPWORDS = sorted(
    set(ENGLISH_STOP_WORDS)
    | GERMAN_STOPWORDS
    | REDDIT_STOPWORDS
    | TOPIC_NOISE_STOPWORDS
)
  

def save_table(df: pd.DataFrame, filename: str) -> None:
    path = CFG.table_dir / filename
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"Saved table: {path}")


def save_figure(fig: plt.Figure, filename: str) -> None:
    path = CFG.figure_dir / filename
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {path}")


def clean_text(text: object) -> str:
    if pd.isna(text):
        return ""

    text = str(text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\[deleted\]|\[removed\]", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\br/[A-Za-z0-9_]+\b", " ", text)
    text = re.sub(r"\bu/[A-Za-z0-9_]+\b", " ", text)
    text = re.sub(r"[*_>#~|`]+", " ", text)
    text = re.sub(r"[^A-Za-zÄÖÜäöüß0-9€$%.,!?;:\-\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def normalize_for_matching(text: str) -> str:
    text = str(text).lower()
    text = text.replace("ä", "ae").replace("ö", "oe").replace("ü", "ue").replace("ß", "ss")
    text = re.sub(r"[^a-z0-9\s_\-]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_text_column(df: pd.DataFrame) -> pd.Series:
    for col in ["pragmatism_text", "text_for_sentiment", "clean_text", "combined_text", "text", "body"]:
        if col in df.columns:
            return df[col].fillna("").astype(str)

    if "title" in df.columns and "selftext" in df.columns:
        return (
            df["title"].fillna("").astype(str)
            + ". "
            + df["selftext"].fillna("").astype(str)
        )

    raise ValueError(
        "No usable text column found. Expected pragmatism_text, text_for_sentiment, "
        "clean_text, combined_text, text, body, or title/selftext."
    )


def make_topic_label(topic_terms: list[str], topic_id: int, n: int = 5) -> str:
    return f"T{topic_id}: " + ", ".join(topic_terms[:n])

def short_topic_terms(topic_label: str, n_terms: int = 3) -> str:
    """
    Extracts the first n topic terms from labels like:
    'T18: tax, taxes, file, income, return'
    """
    if pd.isna(topic_label):
        return ""

    label = str(topic_label)

    if ":" in label:
        label = label.split(":", 1)[1]

    terms = [term.strip() for term in label.split(",") if term.strip()]

    return ", ".join(terms[:n_terms])


def bh_adjust(p_values: pd.Series) -> pd.Series:
    p = pd.to_numeric(p_values, errors="coerce").to_numpy(dtype=float)
    adjusted = np.full_like(p, np.nan, dtype=float)

    valid = ~np.isnan(p)
    p_valid = p[valid]

    if len(p_valid) == 0:
        return pd.Series(adjusted, index=p_values.index)

    order = np.argsort(p_valid)
    ranked = p_valid[order]
    n = len(ranked)

    q = ranked * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)

    adjusted_valid = np.empty_like(q)
    adjusted_valid[order] = q
    adjusted[valid] = adjusted_valid

    return pd.Series(adjusted, index=p_values.index)


  
# LOAD DATA
  

def load_data() -> pd.DataFrame:
    if not CFG.input_path.exists():
        raise FileNotFoundError(f"Input file not found: {CFG.input_path}")

    df = pd.read_csv(CFG.input_path, low_memory=False)

    if "community" not in df.columns or "year" not in df.columns:
        raise ValueError("Input file must contain community and year columns.")

    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["community"] = df["community"].astype(str)

    if "analysis_id" in df.columns:
        df["document_id"] = df["analysis_id"].astype(str)
    elif "id" in df.columns:
        df["document_id"] = df["id"].astype(str)
    else:
        df["document_id"] = [f"doc_{i:08d}" for i in range(len(df))]

    df["topic_model_text"] = build_text_column(df).map(clean_text)

    before = len(df)
    df = df[df["topic_model_text"].str.len() >= CFG.min_chars].copy()
    after = len(df)

    print(f"Loaded rows: {before:,}")
    print(f"Rows after text-length filter: {after:,}")

    if CFG.max_docs_per_group is not None:
        pieces = []
        for _, group in df.groupby(["community", "year"], dropna=False):
            n = min(len(group), CFG.max_docs_per_group)
            pieces.append(group.sample(n=n, random_state=CFG.random_seed))
        df = pd.concat(pieces, ignore_index=True)
        print(f"Trial sample rows: {len(df):,}")

    return df.reset_index(drop=True)


  
# VECTORIZE AND CLUSTER
  

def build_tfidf_matrix(texts: list[str]):
    vectorizer = TfidfVectorizer(
        stop_words=STOPWORDS,
        lowercase=True,
        min_df=CFG.min_df,
        max_df=CFG.max_df,
        max_features=CFG.max_features,
        ngram_range=CFG.ngram_range,
        token_pattern=r"(?u)\b[a-zA-ZÄÖÜäöüß][a-zA-ZÄÖÜäöüß0-9_\-]{2,}\b",
    )

    matrix = vectorizer.fit_transform(texts)

    return vectorizer, matrix


def get_clustering_matrix(tfidf_matrix):
    if not CFG.use_sentence_transformers_if_available:
        print("Using TF-IDF matrix for KMeans clustering.")
        return tfidf_matrix, "tfidf"

    try:
        from sentence_transformers import SentenceTransformer

        print(f"Trying sentence-transformer embeddings: {CFG.embedding_model_name}")
        model = SentenceTransformer(CFG.embedding_model_name)

        embeddings = model.encode(
            texts_global,
            batch_size=64,
            show_progress_bar=True,
            normalize_embeddings=True,
        )

        print("Using multilingual sentence-transformer embeddings for KMeans clustering.")
        return embeddings, "sentence_transformers"

    except Exception as exc:
        print("Sentence-transformers unavailable or failed.")
        print(f"Reason: {exc}")
        print("Falling back to TF-IDF KMeans.")
        return tfidf_matrix, "tfidf"


def fit_kmeans_topic_model(df: pd.DataFrame):
    global texts_global

    texts_global = df["topic_model_text"].tolist()

    print("Building matrix...")
    vectorizer, tfidf_matrix = build_tfidf_matrix(texts_global)

    clustering_matrix, clustering_source = get_clustering_matrix(tfidf_matrix)

    print(f"Fitting KMeans with n_topics={CFG.n_topics}...")
    kmeans = KMeans(
        n_clusters=CFG.n_topics,
        random_state=CFG.random_seed,
        n_init=10,
    )

    topics = kmeans.fit_predict(clustering_matrix)

    doc_topics = df.copy()
    doc_topics["topic"] = topics
    doc_topics["clustering_source"] = clustering_source

    try:
        sil = silhouette_score(
            clustering_matrix,
            topics,
            sample_size=min(5000, len(topics)),
            random_state=CFG.random_seed,
        )
    except Exception:
        sil = np.nan

    return vectorizer, tfidf_matrix, kmeans, doc_topics, sil


  
# TOPIC TERMS
  

def extract_topic_terms(
    vectorizer: TfidfVectorizer,
    tfidf_matrix,
    doc_topics: pd.DataFrame,
    top_n: int = 20,
) -> tuple[pd.DataFrame, dict[int, list[str]]]:
    feature_names = np.array(vectorizer.get_feature_names_out())

    rows = []
    topic_to_terms = {}

    for topic in sorted(doc_topics["topic"].unique()):
        idx = np.where(doc_topics["topic"].to_numpy() == topic)[0]

        if len(idx) == 0:
            continue

        topic_tfidf_mean = np.asarray(tfidf_matrix[idx].mean(axis=0)).ravel()
        top_idx = topic_tfidf_mean.argsort()[::-1][:top_n]

        terms = feature_names[top_idx].tolist()
        scores = topic_tfidf_mean[top_idx].tolist()

        topic_to_terms[int(topic)] = terms

        for rank, (term, score) in enumerate(zip(terms, scores), start=1):
            rows.append(
                {
                    "topic": int(topic),
                    "rank": rank,
                    "term": term,
                    "score": float(score),
                }
            )

    return pd.DataFrame(rows), topic_to_terms


def make_topic_info(
    doc_topics: pd.DataFrame,
    topic_to_terms: dict[int, list[str]],
    silhouette: float,
) -> pd.DataFrame:
    rows = []

    for topic, group in doc_topics.groupby("topic"):
        terms = topic_to_terms[int(topic)]
        rows.append(
            {
                "topic": int(topic),
                "n_posts": int(len(group)),
                "share": float(len(group) / len(doc_topics)),
                "topic_label": make_topic_label(terms, int(topic)),
                "top_terms": ", ".join(terms[:15]),
                "silhouette_score_overall": silhouette,
            }
        )

    return pd.DataFrame(rows).sort_values("n_posts", ascending=False)


def add_topic_labels(
    doc_topics: pd.DataFrame,
    topic_to_terms: dict[int, list[str]],
) -> pd.DataFrame:
    out = doc_topics.copy()
    out["topic_label"] = out["topic"].map(
        lambda t: make_topic_label(topic_to_terms[int(t)], int(t))
    )
    return out


  
# MAP TOPICS TO STRUCTURAL PRAGMATISM CATEGORIES
  

def seed_overlap(topic_terms: list[str], seed_terms: list[str]) -> tuple[int, str]:
    matched = set()

    normalized_topic_terms = [normalize_for_matching(t) for t in topic_terms]
    normalized_seed_terms = [normalize_for_matching(s) for s in seed_terms]

    for topic_term in normalized_topic_terms:
        for seed, seed_norm in zip(seed_terms, normalized_seed_terms):
            if not seed_norm:
                continue
            if topic_term == seed_norm or seed_norm in topic_term or topic_term in seed_norm:
                matched.add(seed)

    return len(matched), "; ".join(sorted(matched))


def positive_minmax(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce").fillna(0).clip(lower=0)
    max_value = s.max()
    if max_value <= 0:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return s / max_value


def map_topics_to_categories(
    doc_topics: pd.DataFrame,
    topic_to_terms: dict[int, list[str]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []

    for topic, group in doc_topics.groupby("topic"):
        topic = int(topic)
        topic_terms = topic_to_terms[topic]

        for category in CATEGORY_ORDER:
            has_col = f"prag_{category}_has"
            rate_col = f"prag_{category}_rate_per_100_words"
            count_col = f"prag_{category}_count"

            if has_col in doc_topics.columns:
                global_presence = pd.to_numeric(
                    doc_topics[has_col], errors="coerce"
                ).fillna(0).mean()
                topic_presence = pd.to_numeric(
                    group[has_col], errors="coerce"
                ).fillna(0).mean()
                presence_lift = topic_presence - global_presence
            else:
                global_presence = np.nan
                topic_presence = np.nan
                presence_lift = np.nan

            if rate_col in doc_topics.columns:
                global_rate = pd.to_numeric(
                    doc_topics[rate_col], errors="coerce"
                ).fillna(0).mean()
                topic_rate = pd.to_numeric(
                    group[rate_col], errors="coerce"
                ).fillna(0).mean()
                rate_lift = topic_rate - global_rate
            elif count_col in doc_topics.columns:
                global_rate = pd.to_numeric(
                    doc_topics[count_col], errors="coerce"
                ).fillna(0).mean()
                topic_rate = pd.to_numeric(
                    group[count_col], errors="coerce"
                ).fillna(0).mean()
                rate_lift = topic_rate - global_rate
            else:
                global_rate = np.nan
                topic_rate = np.nan
                rate_lift = np.nan

            overlap_count, matched_terms = seed_overlap(
                topic_terms,
                SEED_LEXICON[category],
            )

            rows.append(
                {
                    "topic": topic,
                    "topic_label": group["topic_label"].iloc[0],
                    "topic_n_posts": int(len(group)),
                    "category": category,
                    "category_label": CATEGORY_LABELS[category],
                    "topic_category_presence": topic_presence,
                    "global_category_presence": global_presence,
                    "presence_lift": presence_lift,
                    "topic_category_rate_or_count_mean": topic_rate,
                    "global_category_rate_or_count_mean": global_rate,
                    "rate_lift": rate_lift,
                    "keyword_overlap_count": overlap_count,
                    "matched_seed_terms": matched_terms,
                }
            )

    alignment = pd.DataFrame(rows)

    alignment["presence_component"] = (
        alignment.groupby("topic")["presence_lift"].transform(positive_minmax)
    )
    alignment["rate_component"] = (
        alignment.groupby("topic")["rate_lift"].transform(positive_minmax)
    )
    alignment["keyword_component"] = (
        alignment.groupby("topic")["keyword_overlap_count"].transform(positive_minmax)
    )

    alignment["combined_alignment_score"] = (
        0.50 * alignment["presence_component"]
        + 0.30 * alignment["rate_component"]
        + 0.20 * alignment["keyword_component"]
    )

    # Prefer categories where the topic actually shares seed terms.
    # This prevents generic topics from being mapped only because their posts
    # have high lexicon scores in one category.
    alignment["has_seed_overlap"] = (
        pd.to_numeric(alignment["keyword_overlap_count"], errors="coerce")
        .fillna(0)
        .gt(0)
    )

    best = (
        alignment.sort_values(
            ["topic", "has_seed_overlap", "combined_alignment_score"],
            ascending=[True, False, False],
        )
        .groupby("topic")
        .head(1)
        .copy()
    )

    best["mapped_structural_category"] = np.where(
        best["has_seed_overlap"] & (best["combined_alignment_score"] >= 0.20),
        best["category"],
        "unmapped_or_general",
    )

    best["mapped_structural_category_label"] = best["mapped_structural_category"].map(
        CATEGORY_LABELS
    )

    mapping = best[
        [
            "topic",
            "topic_label",
            "topic_n_posts",
            "mapped_structural_category",
            "mapped_structural_category_label",
            "combined_alignment_score",
            "matched_seed_terms",
        ]
    ].sort_values("topic_n_posts", ascending=False)

    return alignment, mapping


def add_mapping_to_documents(
    doc_topics: pd.DataFrame,
    mapping: pd.DataFrame,
) -> pd.DataFrame:
    category_map = mapping.set_index("topic")["mapped_structural_category"].to_dict()
    label_map = mapping.set_index("topic")["mapped_structural_category_label"].to_dict()

    out = doc_topics.copy()
    out["mapped_structural_category"] = out["topic"].map(category_map)
    out["mapped_structural_category_label"] = out["topic"].map(label_map)

    return out


  
# PREVALENCE AND TESTS
  

def topic_prevalence_by_group(doc_topics: pd.DataFrame) -> pd.DataFrame:
    group_n = (
        doc_topics.groupby(["community", "year"])
        .size()
        .rename("group_n")
        .reset_index()
    )

    counts = (
        doc_topics.groupby(["community", "year", "topic", "topic_label"])
        .size()
        .rename("n_posts")
        .reset_index()
    )

    out = counts.merge(group_n, on=["community", "year"], how="left")
    out["share"] = out["n_posts"] / out["group_n"]

    return out


def mapped_category_prevalence_by_group(doc_topics: pd.DataFrame) -> pd.DataFrame:
    group_n = (
        doc_topics.groupby(["community", "year"])
        .size()
        .rename("group_n")
        .reset_index()
    )

    counts = (
        doc_topics.groupby(
            [
                "community",
                "year",
                "mapped_structural_category",
                "mapped_structural_category_label",
            ]
        )
        .size()
        .rename("n_posts")
        .reset_index()
    )

    out = counts.merge(group_n, on=["community", "year"], how="left")
    out["share"] = out["n_posts"] / out["group_n"]

    return out


def topic_chisquare_tests(doc_topics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    topics = sorted(doc_topics["topic"].unique())

    comparisons = [
        (
            "between_communities_finanzen_minus_personalfinance_2025",
            doc_topics["community"].eq("finanzen") & doc_topics["year"].eq(2025),
            doc_topics["community"].eq("personalfinance") & doc_topics["year"].eq(2025),
            "finanzen_2025",
            "personalfinance_2025",
        ),
        (
            "between_communities_finanzen_minus_personalfinance_2020",
            doc_topics["community"].eq("finanzen") & doc_topics["year"].eq(2020),
            doc_topics["community"].eq("personalfinance") & doc_topics["year"].eq(2020),
            "finanzen_2020",
            "personalfinance_2020",
        ),
    ]

    topic_label_map = doc_topics.drop_duplicates("topic").set_index("topic")["topic_label"].to_dict()
    mapped_map = doc_topics.drop_duplicates("topic").set_index("topic")["mapped_structural_category"].to_dict()

    for comparison_type, g1_mask, g2_mask, g1_label, g2_label in comparisons:
        g1 = doc_topics[g1_mask]
        g2 = doc_topics[g2_mask]

        for topic in topics:
            g1_present = int((g1["topic"] == topic).sum())
            g2_present = int((g2["topic"] == topic).sum())

            table = np.array(
                [
                    [g1_present, len(g1) - g1_present],
                    [g2_present, len(g2) - g2_present],
                ]
            )

            try:
                chi2, p_value, _, _ = chi2_contingency(table, correction=False)
                n = table.sum()
                cramers_v = np.sqrt(chi2 / n)
            except Exception:
                chi2, p_value, cramers_v = np.nan, np.nan, np.nan

            share_1 = g1_present / len(g1) if len(g1) else np.nan
            share_2 = g2_present / len(g2) if len(g2) else np.nan

            rows.append(
                {
                    "comparison_type": comparison_type,
                    "topic": int(topic),
                    "topic_label": topic_label_map.get(topic),
                    "mapped_structural_category": mapped_map.get(topic),
                    "group_1": g1_label,
                    "group_2": g2_label,
                    "group_1_share": share_1,
                    "group_2_share": share_2,
                    "share_difference_1_minus_2": share_1 - share_2,
                    "chi2": chi2,
                    "p_value": p_value,
                    "cramers_v": cramers_v,
                }
            )

    tests = pd.DataFrame(rows)

    if not tests.empty:
        tests["p_value_bh"] = tests.groupby("comparison_type")["p_value"].transform(bh_adjust)

    return tests


  

def plot_topic_differences_by_year(
    prevalence: pd.DataFrame,
    mapping: pd.DataFrame,
    year: int,
    filename_prefix: str,
) -> None:
    df = prevalence[prevalence["year"].eq(year)].copy()

    pivot = (
        df.pivot_table(
            index=["topic", "topic_label"],
            columns="community",
            values="share",
            fill_value=0,
        )
        .reset_index()
    )

    for community in COMMUNITY_ORDER:
        if community not in pivot.columns:
            pivot[community] = 0.0

    pivot["difference_finanzen_minus_personalfinance"] = (
        pivot["finanzen"] - pivot["personalfinance"]
    )

    pivot = pivot.merge(
        mapping[
            [
                "topic",
                "mapped_structural_category",
                "mapped_structural_category_label",
            ]
        ],
        on="topic",
        how="left",
    )

    pivot = (
        pivot.reindex(
            pivot["difference_finanzen_minus_personalfinance"]
            .abs()
            .sort_values(ascending=False)
            .index
        )
        .head(15)
        .sort_values("difference_finanzen_minus_personalfinance")
    )

    save_table(pivot, f"{filename_prefix}_topic_differences_{year}_data.csv")

    pivot["topic_display"] = "T" + pivot["topic"].astype(str)
    fig, ax = plt.subplots(figsize=(15, 9))
    y = np.arange(len(pivot))
    values = pivot["difference_finanzen_minus_personalfinance"]
    bars = ax.barh(y, values)
    ax.axvline(0, linewidth=1)
    year_label = "2020 crisis year" if year == 2020 else str(year)
    ax.set_title(
        f"Largest topic differences: finanzen vs personalfinance, {year_label}",
        fontsize=18,
    )
    ax.set_xlabel(
        "Difference in topic share: finanzen minus personalfinance",
        fontsize=14,
    )
    ax.set_ylabel("Topic", fontsize=14)

    ax.set_yticks(y)
    ax.set_yticklabels(pivot["topic_display"], fontsize=13)

    ax.tick_params(axis="x", labelsize=12)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.grid(axis="x", alpha=0.25)

    max_abs = max(float(np.nanmax(np.abs(values))), 0.01)
    ax.set_xlim(-max_abs * 1.55, max_abs * 1.75)

    for bar, value, mapped_label in zip(
        bars,
        values,
        pivot["mapped_structural_category_label"],
    ):
        x = bar.get_width()
        label_x = x + 0.008 if x >= 0 else x - 0.008
        ha = "left" if x >= 0 else "right"

        ax.text(
            label_x,
            bar.get_y() + bar.get_height() / 2,
            f"{value * 100:+.1f} pp | {mapped_label}",
            va="center",
            ha=ha,
            fontsize=11,
        )

    ax.text(
        0.99,
        0.02,
        "Positive = higher in finanzen\nNegative = higher in personalfinance",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=11,
        bbox=dict(boxstyle="round", alpha=0.08),
    )

    fig.subplots_adjust(left=0.18, right=0.96)

    save_figure(fig, f"{filename_prefix}_topic_differences_{year}.png")

def plot_mapped_category_profile_by_year(
    mapped_prev: pd.DataFrame,
    year: int,
    filename_prefix: str,
) -> None:
    df = mapped_prev[mapped_prev["year"].eq(year)].copy()
    df = df[df["mapped_structural_category"] != "unmapped_or_general"].copy()

    pivot = (
        df.pivot_table(
            index=[
                "mapped_structural_category",
                "mapped_structural_category_label",
            ],
            columns="community",
            values="share",
            fill_value=0,
        )
        .reset_index()
    )

    for community in COMMUNITY_ORDER:
        if community not in pivot.columns:
            pivot[community] = 0.0

    order = CATEGORY_ORDER + ["unmapped_or_general"]
    pivot["order"] = pivot["mapped_structural_category"].map(
        {cat: i for i, cat in enumerate(order)}
    )
    pivot = pivot.sort_values("order")

    save_table(
        pivot,
        f"{filename_prefix}_topic_mapped_category_profile_{year}_data.csv",
    )

    fig, ax = plt.subplots(figsize=(11, 7))

    y = np.arange(len(pivot))
    height = 0.38

    ax.barh(
        y - height / 2,
        pivot["personalfinance"],
        height,
        label="personalfinance",
    )

    ax.barh(
        y + height / 2,
        pivot["finanzen"],
        height,
        label="finanzen",
    )

    year_label = "2020 crisis year" if year == 2020 else str(year)

    ax.set_title(f"Topic-derived structural category profile, {year_label} (mapped categories only)")
    ax.set_xlabel("Share of posts assigned to mapped topic category")
    ax.set_yticks(y)
    ax.set_yticklabels(pivot["mapped_structural_category_label"])
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.legend(title="Community")
    ax.grid(axis="x", alpha=0.25)

    save_figure(fig, f"{filename_prefix}_topic_mapped_category_profile_{year}.png")

def plot_topic_driver_bubble_chart(
    prevalence: pd.DataFrame,
    mapping: pd.DataFrame,
    year: int,
    filename_prefix: str,
    top_n: int = 16,
) -> None:

    df = prevalence[prevalence["year"].eq(year)].copy()

    pivot = (
        df.pivot_table(
            index=["topic", "topic_label"],
            columns="community",
            values="share",
            fill_value=0,
        )
        .reset_index()
    )

    for community in COMMUNITY_ORDER:
        if community not in pivot.columns:
            pivot[community] = 0.0

    pivot["difference_finanzen_minus_personalfinance"] = (
        pivot["finanzen"] - pivot["personalfinance"]
    )

    pivot["mean_topic_share"] = (
        pivot["finanzen"] + pivot["personalfinance"]
    ) / 2

    pivot = pivot.merge(
        mapping[
            [
                "topic",
                "mapped_structural_category",
                "mapped_structural_category_label",
                "topic_n_posts",
                "matched_seed_terms",
            ]
        ],
        on="topic",
        how="left",
    )
    pivot = pivot[
        pivot["mapped_structural_category"].notna()
        & (pivot["mapped_structural_category"] != "unmapped_or_general")
    ].copy()
    pivot = (
        pivot.reindex(
            pivot["difference_finanzen_minus_personalfinance"]
            .abs()
            .sort_values(ascending=False)
            .index
        )
        .head(top_n)
        .copy()
    )
    category_order = [
        category
        for category in CATEGORY_ORDER
        if category in pivot["mapped_structural_category"].unique()
    ]

    category_to_y = {
        category: i
        for i, category in enumerate(category_order)
    }

    pivot["y_base"] = pivot["mapped_structural_category"].map(category_to_y)
    pivot = pivot.sort_values(
        ["mapped_structural_category", "difference_finanzen_minus_personalfinance"]
    ).copy()

    offsets = []
    for _, group in pivot.groupby("mapped_structural_category", sort=False):
        n = len(group)
        if n == 1:
            offsets.extend([0.0])
        else:
            offsets.extend(np.linspace(-0.18, 0.18, n))

    pivot["y"] = pivot["y_base"] + offsets
    max_share = max(float(pivot["mean_topic_share"].max()), 0.001)
    pivot["bubble_size"] = 350 + 2200 * (pivot["mean_topic_share"] / max_share)

    pivot["topic_display"] = "T" + pivot["topic"].astype(int).astype(str)

    save_table(
        pivot[
            [
                "topic",
                "topic_label",
                "mapped_structural_category",
                "mapped_structural_category_label",
                "finanzen",
                "personalfinance",
                "difference_finanzen_minus_personalfinance",
                "mean_topic_share",
                "topic_n_posts",
                "matched_seed_terms",
            ]
        ],
        f"{filename_prefix}_topic_driver_bubble_{year}_data.csv",
    )

    fig, ax = plt.subplots(figsize=(14, 8))

    cmap = plt.get_cmap("tab20")

    for i, category in enumerate(category_order):
        subset = pivot[pivot["mapped_structural_category"] == category]

        ax.scatter(
            subset["difference_finanzen_minus_personalfinance"],
            subset["y"],
            s=subset["bubble_size"],
            alpha=0.72,
            color=cmap(i),
            edgecolor="black",
            linewidth=0.7,
            label=CATEGORY_LABELS.get(category, category),
        )

        for _, row in subset.iterrows():
            ax.text(
                row["difference_finanzen_minus_personalfinance"],
                row["y"],
                row["topic_display"],
                ha="center",
                va="center",
                fontsize=10,
                weight="bold",
            )

    year_label = "2020 crisis year" if year == 2020 else str(year)

    ax.axvline(0, color="black", linewidth=1)

    ax.set_title(
        f"Topic drivers of community differences, {year_label}",
        fontsize=18,
    )

    ax.set_xlabel(
        "Difference in topic share: finanzen minus personalfinance",
        fontsize=14,
    )

    ax.set_ylabel("Mapped structural category", fontsize=14)

    ax.set_yticks(range(len(category_order)))
    ax.set_yticklabels(
        [CATEGORY_LABELS.get(category, category) for category in category_order],
        fontsize=12,
    )

    ax.tick_params(axis="x", labelsize=12)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.grid(axis="x", alpha=0.25)
    ax.grid(axis="y", alpha=0.10)

    max_abs = max(
        float(np.nanmax(np.abs(pivot["difference_finanzen_minus_personalfinance"]))),
        0.01,
    )

    ax.set_xlim(-max_abs * 1.35, max_abs * 1.35)

    ax.text(
        0.01,
        0.02,
        "Left = higher in personalfinance\nRight = higher in finanzen\nBubble size = average topic prevalence",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10,
        bbox=dict(boxstyle="round", alpha=0.08),
    )

    save_figure(fig, f"{filename_prefix}_topic_driver_bubble_{year}.png")



def make_topic_driver_label_table(
    prevalence: pd.DataFrame,
    mapping: pd.DataFrame,
    year: int,
    filename_prefix: str,
    top_n: int = 12,
    n_terms: int = 3,
) -> None:
 
    df = prevalence[prevalence["year"].eq(year)].copy()

    pivot = (
        df.pivot_table(
            index=["topic", "topic_label"],
            columns="community",
            values="share",
            fill_value=0,
        )
        .reset_index()
    )

    for community in COMMUNITY_ORDER:
        if community not in pivot.columns:
            pivot[community] = 0.0

    pivot["difference_finanzen_minus_personalfinance"] = (
        pivot["finanzen"] - pivot["personalfinance"]
    )

    pivot["mean_topic_share"] = (
        pivot["finanzen"] + pivot["personalfinance"]
    ) / 2

    pivot = pivot.merge(
        mapping[
            [
                "topic",
                "mapped_structural_category",
                "mapped_structural_category_label",
            ]
        ],
        on="topic",
        how="left",
    )
    pivot = pivot[
        pivot["mapped_structural_category"].notna()
        & (pivot["mapped_structural_category"] != "unmapped_or_general")
    ].copy()
    pivot = (
        pivot.reindex(
            pivot["difference_finanzen_minus_personalfinance"]
            .abs()
            .sort_values(ascending=False)
            .index
        )
        .head(top_n)
        .copy()
    )
    pivot["topic_id"] = "T" + pivot["topic"].astype(int).astype(str)
    pivot["top_terms"] = pivot["topic_label"].map(
        lambda x: short_topic_terms(x, n_terms=n_terms)
    )
    pivot["higher_in"] = np.where(
        pivot["difference_finanzen_minus_personalfinance"] > 0,
        "finanzen",
        "personalfinance",
    )
    pivot["difference_pp"] = (
        pivot["difference_finanzen_minus_personalfinance"] * 100
    ).round(1)

    out = pivot[
        [
            "topic_id",
            "top_terms",
            "mapped_structural_category_label",
            "higher_in",
            "difference_pp",
            "finanzen",
            "personalfinance",
            "topic_label",
        ]
    ].copy()

    out = out.rename(
        columns={
            "topic_id": "Topic",
            "top_terms": "Top terms",
            "mapped_structural_category_label": "Mapped category",
            "higher_in": "Higher in",
            "difference_pp": "Difference, pp",
            "finanzen": "finanzen share",
            "personalfinance": "personalfinance share",
            "topic_label": "Full topic label",
        }
    )

    save_table(
        out,
        f"{filename_prefix}_topic_driver_key_{year}.csv",
    )

    display = out[
        [
            "Topic",
            "Top terms",
            "Mapped category",
            "Higher in",
            "Difference, pp",
        ]
    ].copy()

    display["Difference, pp"] = display["Difference, pp"].map(
        lambda x: f"{x:+.1f}"
    )

    year_label = "2020 crisis year" if year == 2020 else str(year)

    fig_height = 0.55 + 0.45 * len(display)
    fig, ax = plt.subplots(figsize=(14, fig_height))
    ax.axis("off")

    table = ax.table(
        cellText=display.values,
        colLabels=display.columns,
        cellLoc="left",
        colLoc="left",
        loc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.35)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold")
            cell.set_height(cell.get_height() * 1.15)
        if col == 0:
            cell.set_text_props(weight="bold")

    ax.set_title(
        f"Topic key for topic-driver plot, {year_label}",
        fontsize=16,
        weight="bold",
        pad=14,
    )

    save_figure(fig, f"{filename_prefix}_topic_driver_key_{year}.png")


  

def main() -> None:
    print("=" * 80)
    print("TOPIC MODELING VALIDATION")
    print("=" * 80)

    df = load_data()

    vectorizer, tfidf_matrix, kmeans, doc_topics, silhouette = fit_kmeans_topic_model(df)

    terms, topic_to_terms = extract_topic_terms(
        vectorizer=vectorizer,
        tfidf_matrix=tfidf_matrix,
        doc_topics=doc_topics,
        top_n=20,
    )

    doc_topics = add_topic_labels(doc_topics, topic_to_terms)

    topic_info = make_topic_info(doc_topics, topic_to_terms, silhouette)

    alignment, mapping = map_topics_to_categories(doc_topics, topic_to_terms)

    doc_topics = add_mapping_to_documents(doc_topics, mapping)

    prevalence = topic_prevalence_by_group(doc_topics)
    mapped_prevalence = mapped_category_prevalence_by_group(doc_topics)
    topic_tests = topic_chisquare_tests(doc_topics)

    save_table(topic_info, "topic_model_topic_info.csv")
    save_table(terms, "topic_model_topic_terms.csv")
    save_table(alignment, "topic_model_topic_category_alignment_long.csv")
    save_table(mapping, "topic_model_topic_category_mapping.csv")
    save_table(doc_topics, "topic_model_document_topics.csv")
    save_table(prevalence, "topic_model_topic_prevalence_by_group.csv")
    save_table(mapped_prevalence, "topic_model_mapped_category_prevalence_by_group.csv")
    save_table(topic_tests, "topic_model_topic_chisquare_tests.csv")

    plot_topic_differences_by_year(
        prevalence=prevalence,
        mapping=mapping,
        year=2020,
        filename_prefix="01",
    )

    plot_topic_differences_by_year(
        prevalence=prevalence,
        mapping=mapping,
        year=2025,
        filename_prefix="02",
    )

    plot_mapped_category_profile_by_year(
    mapped_prev=mapped_prevalence,
    year=2020,
    filename_prefix="03",
    )

    plot_mapped_category_profile_by_year(
        mapped_prev=mapped_prevalence,
        year=2025,
        filename_prefix="04",
    )

    plot_topic_driver_bubble_chart(
        prevalence=prevalence,
        mapping=mapping,
        year=2020,
        filename_prefix="05",
        top_n=16,
    )

    plot_topic_driver_bubble_chart(
        prevalence=prevalence,
        mapping=mapping,
        year=2025,
        filename_prefix="06",
        top_n=16,
    )

    make_topic_driver_label_table(
        prevalence=prevalence,
        mapping=mapping,
        year=2020,
        filename_prefix="07",
        top_n=12,
        n_terms=3,
    )

    make_topic_driver_label_table(
        prevalence=prevalence,
        mapping=mapping,
        year=2025,
        filename_prefix="08",
        top_n=12,
        n_terms=3,
    )

    summary_path = CFG.output_dir / "topic_modeling_validation_summary.md"

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("# No-HDBSCAN Topic Modeling Validation Summary\n\n")
        f.write(f"- Input file: `{CFG.input_path}`\n")
        f.write(f"- Documents used: `{len(doc_topics):,}`\n")
        f.write(f"- Number of topics: `{CFG.n_topics}`\n")
        f.write(f"- Clustering source: `{doc_topics['clustering_source'].iloc[0]}`\n")
        f.write(f"- Overall silhouette score: `{silhouette}`\n\n")
        f.write("## Purpose\n\n")
        f.write(
            "This model is used as an inductive validation layer for the structural "
            "pragmatism categories. When available, it clusters posts using multilingual "
            "sentence-transformer embeddings to reduce German-English vocabulary confounding. "
            "TF-IDF is retained for topic-term labeling. It avoids BERTopic/HDBSCAN and "
            "therefore does not require Microsoft C++ Build Tools.\n\n"
        )
        f.write("## Most important files\n\n")
        f.write("- `tables/topic_model_topic_info.csv`\n")
        f.write("- `tables/topic_model_topic_category_mapping.csv`\n")
        f.write("- `tables/topic_model_mapped_category_prevalence_by_group.csv`\n")
        f.write("- `figures/01_topic_differences_2020.png`\n")
        f.write("- `figures/02_topic_differences_2025.png`\n")
        f.write("- `figures/03_topic_mapped_category_profile_2020.png`\n")
        f.write("- `figures/04_topic_mapped_category_profile_2025.png`\n")
        f.write("- `figures/05_topic_driver_bubble_2020.png`\n")
        f.write("- `figures/06_topic_driver_bubble_2025.png`\n")
        f.write("- `figures/07_topic_driver_key_2020.png`\n")
        f.write("- `figures/08_topic_driver_key_2025.png`\n")

    print(f"Saved summary: {summary_path}")

    print()
    print("=" * 80)
    print("DONE")
    print("=" * 80)
    print(f"Tables saved to:  {CFG.table_dir}")
    print(f"Figures saved to: {CFG.figure_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()