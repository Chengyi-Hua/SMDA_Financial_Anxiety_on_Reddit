# Financial Anxiety on Reddit: A Cross-Lingual Distress Analysis

This repository contains the analysis code for the seminar paper *"Financial Anxiety on Reddit: A Cross-Lingual Distress Analysis of English and German Communities"*, examining LIWC-based financial distress signals in r/personalfinance (English) and r/Finanzen (German) between 2020 and 2025.

---

## Project Structure

```
.
├── asset/                          # Input CSVs and output figures
│   ├── personalfinance_2020_final.csv
│   ├── personalfinance_2025_final.csv
│   ├── finanzen_2020_final.csv
│   ├── finanzen_2025_final.csv
│   ├── fdi.csv                     # FDI data (UNCTAD)
│   ├── post_counts.csv             # Annual post count per subreddit
│   └── distress_*_with_scores.csv  # Output of distress.ipynb
├── keyword/
│   ├── distress_en.json            # LIWC keyword dictionary (English)
│   └── distress_de.json            # LIWC keyword dictionary (German)
├── utils/
│   ├── constants.py                # Env-loaded config constants
│   ├── ner.py                      # spaCy NER helpers
│   └── post_count.py               # Reddit post count fetcher
├── eda.ipynb                       # Exploratory Data Analysis
├── distress.ipynb                  # LIWC distress scoring & visualisation
├── validate_emotions.py            # Transformer-based LIWC validation
├── load_model.py                   # Hugging Face model loader helper
└── .env                            # Environment variables (see below)
```

---

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

Key packages include: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scipy`, `scikit-learn`, `spacy`, `bertopic`, `transformers`, `torch`, `python-dotenv`.

### 2. Configure Environment Variables

Copy `.env` and fill in your paths and keys:

```bash
cp .env .env.local   # or edit .env directly
```

The `.env` file controls all file paths and model names via `utils/constants.py`:

| Variable              | Description                                |
|-----------------------|--------------------------------------------|
| `ASSET_DIR`           | Path to the asset folder (`./asset`)       |
| `KEYWORD_DIR`         | Path to the keyword JSON folder            |
| `CSV_EN_2020`         | Filename for the English 2020 dataset      |
| `CSV_EN_2025`         | Filename for the English 2025 dataset      |
| `CSV_DE_2020`         | Filename for the German 2020 dataset       |
| `CSV_DE_2025`         | Filename for the German 2025 dataset       |
| `CSV_POST_COUNT`      | Filename for annual post count CSV         |
| `CSV_FDI`             | Filename for the FDI data CSV              |
| `JSON_DISTRESS_EN`    | Filename for English LIWC keyword JSON     |
| `JSON_DISTRESS_DE`    | Filename for German LIWC keyword JSON      |
| `EN_MODEL`            | spaCy English model (`en_core_web_lg`)     |
| `DE_MODEL`            | spaCy German model (`de_core_news_lg`)     |
| `HUGGING_FACE_API_KEY`| Your Hugging Face token (for model downloads) |
| `EMBEDDING_MODEL_NAME`| Sentence embedding model for BERTopic     |

### 3. Download spaCy Models

```bash
python -m spacy download en_core_web_lg
python -m spacy download de_core_news_lg
```

---

## Running the Notebooks

### `distress.ipynb` — LIWC Distress Scoring

**Purpose:** Applies a LIWC keyword-matching approach to compute per-post distress scores across three affective dimensions (Anxiety, Anger, Sadness) for all four corpora (English/German × 2020/2025). Produces the scored CSVs used by all downstream analysis.

**Prerequisites:** The four raw CSVs (`*_final.csv`) must exist in `asset/`, along with the keyword JSON files in `keyword/`.

**Run order (top to bottom):**

| Cell | What it does |
|------|--------------|
| 0    | Loads all four raw CSVs into `en_data` and `de_data` dicts |
| 1    | Loads LIWC keyword dictionaries from JSON; scores each post with `distress_score_Anxiety`, `distress_score_Anger`, `distress_score_Sad`, composite `distress_score`, and `dominant_distress`; saves `distress_*_with_scores.csv` to `asset/` |
| 2    | Descriptive statistics table (mean, median, N) per corpus |
| 3    | Normality check — plots histograms and runs Shapiro-Wilk tests |
| 4    | Standalone loader (reloads the scored CSVs if starting from this cell) |
| 5    | **Statistical tests** — Mann-Whitney U tests (2020 vs 2025 within each language; English vs German within each year) and Cohen's *d* effect sizes |
| 6    | **Violin plot** — overall distress score distribution across all corpora; saved to `asset/` |
| 7    | Bar chart of `dominant_distress` label share across all four groups |
| 8    | **BERTopic topic modelling** — runs on distress-flagged posts, saves UMAP projections to `asset/` |

**Output files:**
```
asset/distress_english_2020_with_scores.csv
asset/distress_english_2025_with_scores.csv
asset/distress_german_2020_with_scores.csv
asset/distress_german_2025_with_scores.csv
asset/distress_violin_plot_*.png
asset/*_bertopic_merged_umap2d.png
```

---

### `eda.ipynb` — Exploratory Data Analysis

**Purpose:** Provides a broad exploratory overview of the four corpora, covering post volume trends, FDI correlation, named-entity distributions, and TF-IDF keyword evolution. **Run `distress.ipynb` first** to generate the scored CSVs that some cells depend on.

**Run order (top to bottom):**

| Cell | What it does |
|------|--------------|
| 0    | Loads all four raw CSVs; imports constants from `.env` |
| 1    | **Dataset overview table** — post count, average/median text length per corpus |
| 2 (markdown) | Section separator: "post_count vs fdi" |
| 3    | Generates `asset/post_counts.csv` via the Reddit Arctic Shift API (skips if file already exists) |
| 4    | **FDI correlation analysis** — Pearson correlation between annual post counts and Germany/US FDI; plots dual-axis time series (post count + FDI per year); saves `asset/correlation_plot.png` |
| 5 (markdown) | Section separator: "NER via Spacy" |
| 6    | Loads spaCy NER models for English and German |
| 7    | **Named Entity Recognition** — extracts entity distributions per year per language; runs chi-square tests |
| 8    | Plots saved NER heatmaps; prints chi-square results from `asset/eda/ner_chi_square_results.txt` |
| 9    | **TF-IDF n-gram analysis** — computes unigram+bigram and bigram+trigram TF-IDF rankings per corpus |
| 10   | Plots keyword evolution heatmaps and faceted bar charts |

**Output files:**
```
asset/post_counts.csv
asset/correlation_plot.png
asset/result_1205/eda/ner_chi_square_results.txt
asset/eda/*.png   (NER heatmaps, TF-IDF plots)
```

> **Note:** Cell 6–7 (NER extraction) can be slow on large corpora. The `SAMPLE_SIZE` variable in `.env` controls the number of posts sampled for NER analysis.

---

## `validate_emotions.py` — Transformer-Based LIWC Validation

**Purpose:** Cross-validates the LIWC-based distress labels against transformer emotion classifiers to measure how well keyword-matching agrees with contextual model predictions.

For each `distress_*_with_scores.csv` in `asset/`, the script:

1. **Detects language** from the filename and selects the appropriate model:
   - English → `Amirhossein75/multi-label-emotion-classification-reddit-comments-roberta` (RoBERTa, GoEmotions 28-class)
   - German → `ChrisLalk/German-Emotions` (XLM-RoBERTa, GoEmotions 28-class)
2. **Filters** to only rows with at least one LIWC distress flag > 0.05.
3. **Runs predictions** on `text_for_sentiment` in batches.
4. **Evaluates agreement** between LIWC flags (as *predicted*) and model output (as *ground truth*) for three emotion-to-dimension mappings:
   - `distress_score_Anxiety` ↔ `nervousness`
   - `distress_score_Anger` ↔ `anger`
   - `distress_score_Sad` ↔ `sadness`
5. **Reports Precision, Recall, F1, and Macro F1** for each file.
6. **Saves** per-file detailed CSVs and a global `asset/validation_summary.csv`.

**Run:**
```bash
python validate_emotions.py
```

**Output files:**
```
asset/validation_results_distress_*_with_scores.csv   # per-post predictions
asset/validation_summary.csv                          # aggregated metrics
```

**Interpreting results:** LIWC consistently achieves high Recall (≥ 0.84 for anger and sadness in English) but low Precision (0.12–0.36), confirming that the lexicon casts a wider semantic net than the contextual transformer model. This is expected behaviour and does not invalidate LIWC; instead, it confirms that LIWC and transformer classifiers measure complementary aspects of emotional expression.
