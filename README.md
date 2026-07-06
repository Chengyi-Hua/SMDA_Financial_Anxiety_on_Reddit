# Financial Anxiety on Reddit: A Cross-Community and Cross-Lingual Study

Team Project — University of Mannheim, Social and Media Data Analysis (Spring 2026).
A cross-lingual, multi-method analysis of financial anxiety expressed in Reddit communities **r/personalfinance** (English) and **r/Finanzen** (German) across two time windows (2020 and 2025).

---

## Research Questions

- **RQ1.** Did distress-related language, sentiment, and structural pragmatism change between 2020 and 2025? 
- **RQ2.** Do English- and German-language Reddit finance communities differ in their emotional and pragmatic framing of financial problems?

with corresponding:
- **H1:** Distress-related language, sentiment, and structural pragmatism differ between 2020 and 2025. 
- **H2:** Distress-related language, sentiment, and structural pragmatism differ between \textit{r/personalfinance} and \textit{r/Finanzen}.

---

## Key Findings
The findings should be interpreted as a subreddit-level communities pattern, not as absolute evidence about cultural differences.

- **Temporal Changes (2020 vs. 2025):** Temporal changes are statistically detectable but substantively small. Distress-related vocabulary increases modestly, but this does not translate into a uniform rise in negative sentiment. `r/Finanzen` becomes slightly more negative, while `r/personalfinance` becomes slightly less negative. Emoji use increases in both communities, especially in `r/Finanzen`.

- **Cross-Community Differences:** Cross-community differences are clearer than temporal changes.
  - **r/personalfinance (English):** Shows higher distress-related language and consistently higher shares of negative-label posts. Structurally, discussions are more strongly oriented toward financial institutions/products, household budgeting, and practical problem-solving.
  - **r/Finanzen (German):** Shows lower distress-related language overall, but a more investment- and macroeconomic-oriented form of structural pragmatism. Emoji use is more frequent in 2025 and emojis mostly function as positive or softening tone markers.

- **Conclusion:** Financial anxiety on Reddit is not only expressed through negative affect. It also appears as community-specific practical reasoning: `r/personalfinance` (English-speaking community) is more household-, budgeting-, and institution-oriented, while `r/Finanzen` (German-speaking community) is more investment-, portfolio-, and macroeconomic-oriented. The results show modest temporal changes but clearer cross-community differences. These findings highlight the importance of analyzing financial anxiety not only through sentiment, but also through distress-language and practical financial problem framing.

- **Limitation:** The lexicon-based measures applied within this project are linguistic proxies, not direct measures of financial anxiety, and may miss context, irony, or negation. Cross-language comparability is limited by chosen subreddits and by the fact that related financial discussions may occur elsewhere. Reddit users are not representative of the general population.
  
---

## Analysis Pipeline
| Step | Location | Description |
|---|---|---|
| **1. Data Collection & Cleaning** | `data_collection_and_cleaning/` | Raw data collection via Arctic Shift, cleaning & anonymisation. |
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

Raw posts were collected via the Reddit API / Arctic Shift, then cleaned and anonymised (usernames removed). Cleaned files are stored in `data_collection_and_cleaning/cleaned_and_anonymized_data/`.

---

## Authors

- Chengyi Hua (2117289)
- Bahri Selçuk Eşkil (2117150)
- Ching-Yun Cheng (2112322)

## Supervisor

Chair of Data Science in the Economic and Social Sciences, University of Mannheim.
