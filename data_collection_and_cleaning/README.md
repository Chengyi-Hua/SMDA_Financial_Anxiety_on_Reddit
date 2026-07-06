# Data Collection and Cleaning

This directory contains the complete data collection and cleaning pipeline used in our study.

## Repository Contents

```
data_collection_and_cleaning/
├── data_collection_and_cleaning.ipynb
├── cleaned_and_anonymized_data/
│   ├── personalfinance_2020_sample.csv
│   ├── personalfinance_2025_sample.csv
│   ├── finanzen_2020_sample.csv
│   └── finanzen_2025_sample.csv
└── README.md
```

### `data_collection_and_cleaning.ipynb`

The notebook contains the complete pipeline, including:

- Reddit data collection using the Arctic Shift API
- Data cleaning and preprocessing
- Final quality checks

The notebook is extensively documented with explanations, comments, methodological justifications, and intermediate outputs for every processing step.

---

## Dataset Availability

The complete cleaned datasets are **not included** in this repository because their size exceeds GitHub's file size limitations.

Instead, this repository provides four **random samples (4,000 posts each)** from the final cleaned and anonymized datasets:

- `personalfinance_2020_sample.csv`
- `personalfinance_2025_sample.csv`
- `finanzen_2020_sample.csv`
- `finanzen_2025_sample.csv`

These samples allow readers to inspect the final dataset structure and preprocessing results.

---

## Reproducing the Full Datasets

The full datasets can be recreated by running the notebook from beginning to end, assuming no changes to the data returned by the Arctic Shift API for the specified collection period and queries, and no API-related issues during execution.

The notebook performs the entire workflow, including:

1. Data collection from Reddit via the Arctic Shift API
2. Saving the raw datasets
3. Applying every cleaning and preprocessing step
4. Producing the final cleaned and anonymized datasets

Executing the notebook therefore reproduces both the raw datasets (before cleaning) and the final processed datasets.
> **Note:** Arctic Shift API rate limits may vary over time. If you encounter rate limit errors while running the notebook, consider increasing the request delay (currently `0.25` seconds via `time.sleep()`) until the requests complete successfully.

### Requirements

Before running the notebook, install the required Python packages (`pandas`, `requests`, `matplotlib`, and `fastText`) and download the fastText language identification model (`lid.176.bin`).

Place the downloaded model file in the same directory as the notebook.

The model can be downloaded from the official fastText documentation:

https://fasttext.cc/docs/en/language-identification.html
---

## Cleaning Log

A detailed cleaning log documenting every preprocessing step, its rationale, and the number of rows before and after each operation is available:

- inside the notebook (`data_collection_and_cleaning.ipynb`)
- below for quick reference.

<img width="1088" height="781" alt="image" src="https://github.com/user-attachments/assets/6ac1f8ad-b96c-4911-83e7-e0ce2a7ef0a0" />
