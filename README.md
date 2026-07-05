# Financial Anxiety on Reddit: A Cross-Lingual Study

Team Project — University of Mannheim, Social and Media Data Analysis (Spring 2026).
A cross-lingual, multi-method analysis of financial anxiety expressed in Reddit communities **r/personalfinance** (English) and **r/Finanzen** (German) across two time windows (2020 and 2025).

---

## Research Questions

- **RQ1.** How did distress-related language, sentiment, emoji tone, and structural pragmatism change between 2020 and 2025?
- **RQ2.** How do English- and German-language Reddit finance communities differ in their emotional and pragmatic framing of financial problems?

---

## Key Findings

- **Temporal Changes (2020 vs. 2025):** The temporal changes over the 5-year period are statistically detectable but substantively small. Across all metrics—Distress, Sentiment, Emoji usage, and Pragmatism—the observed effect sizes remain minimal.

- **Cross-Community Differences:** Community differences are far more pronounced and structurally distinct than temporal shifts.
  - **r/personalfinance (English):** Exhibits significantly higher financial distress (driven by *Anger* over short-term debt) and consistently higher shares of negative-label posts. Emojis are used less frequently. In terms of structural pragmatism, discussions focus heavily on financial institutions/products, household budgeting, and practical problem-solving.
  - **r/Finanzen (German):** Exhibits distress levels clustering near zero (driven by *Anxiety* over long-term planning). While its mean sentiment score dropped below the English community in 2025, it features notably higher emoji usage, which predominantly functions as positive or softening tone markers. Pragmatically, discussions are strongly oriented around investment strategies (e.g., ETFs) and macroeconomic contexts.
- **Conclusion:** Financial anxiety on Reddit is not merely an expression of raw negative affect; it manifests as community-specific practical reasoning. The American context is anchored in short-term liquidity and debt management, whereas the German context is anchored in long-term asset-allocation and market uncertainty.
---

## Analysis Pipeline
| Step | Location | Description |
|---|---|---|
| **1. Data Collection & Cleaning** | `data_collection_and_cleaning.ipynb` | Raw data collection via Arctic Shift, cleaning & anonymisation. |
| **2. EDA** | `eda_distress_analysis/eda.ipynb` | Broad exploratory overview (post-volume trends, FDI correlation, NER, TF-IDF evolution). |
| **3. Distress Analysis (LIWC)** | `eda_distress_analysis/distress.ipynb` `eda_distress_analysis/validate_emotions.py` | LIWC keyword scoring (Anxiety, Anger, Sadness) and BERTopic modelling. Optional RoBERTa validation. |
| **4. Sentiment Analysis** | `sentiment_analysis/` | Main XLM-RoBERTa pipeline, emoji affect, statistical tests, and visualisations. |
| **5. Pragmatism Analysis** | `pragmatism_analysis/` | Keyword-based pragmatism classification, topic modelling validation, statistical tests. |

---

## Dataset

| Corpus | Subreddit | Year | Language |
|---|---|---|---|
| EN-2020 | r/personalfinance | 2020 | English |
| EN-2025 | r/personalfinance | 2025 | English |
| DE-2020 | r/Finanzen | 2020 | German |
| DE-2025 | r/Finanzen | 2025 | German |

Raw posts were collected via the Reddit API / Arctic Shift, then cleaned and anonymised (usernames removed). Cleaned files are stored in `Cleaned and Anonymized Data/`.

---

## Authors

- Chengyi Hua (2117289)
- Bahri Selçuk Eşkil (2117150)
- Ching-Yun Cheng (2112322)

## Supervisor

Chair of Data Science in the Economic and Social Sciences, University of Mannheim.
