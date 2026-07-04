# Pragmatism Analysis

This folder contains the scripts and supporting files for the structural-pragmatism analysis for this project.


## Folder Structure

```text
pragmatism_analysis/
├── analysis_outcome_files/
├── keywords/
│   └── keywords.py
├── topic_modeling_validation_outcome_files/
├── README.md
|__ result_readme.md
├── run_pragmatism_analysis.py
├── run_topic_modeling_validation.py
└── visualize_pragmatism_outputs.py
```

## Main Components

Run the scripts in this order.

### 1. Run the main pragmatism analysis

```bash
python pragmatism_analysis/run_pragmatism_analysis.py
```

This creates the main scored datasets, summary tables, statistical tests, diagnostics, sentiment-pragmatism tables, and the balanced sample.

### 2. Run the topic-modeling validation

```bash
python pragmatism_analysis/run_topic_modeling_validation.py
```

This requires the file:

```text
pragmatism_balanced_sample.csv
```

which is created by the main pragmatism analysis script.

### 3. Generate the main figures

```bash
python pragmatism_analysis/visualize_pragmatism_outputs.py
```

This reads the generated pragmatism outputs and saves figures plus figure-data tables.

## Configuration Notes

Each script contains a configuration class near the top of the file.

Before running the scripts, check that the project directory is set correctly:

```python
project_dir: Path = Path(r"SMDA_Financial_Anxiety_on_Reddit")
```

If the scripts are run from a different working directory, this path may need to be adjusted.

The main pragmatism script also contains key settings such as:

```python
lang_confidence_min = 0.80
random_seed = 42
bootstrap_iterations = 1000
matching_mode = "exact"
create_manual_sample = True
```

The topic-modeling validation script contains settings such as:

```python
n_topics = 30
min_chars = 25
max_features = 80000
min_df = 5
max_df = 0.75
ngram_range = (1, 2)
```

## Python Dependencies

The scripts use common Python data-analysis packages, including:

```text
pandas
numpy
matplotlib
scikit-learn
scipy
tqdm
openpyxl
```

Optional dependency for topic modeling:

```text
sentence-transformers
```

If `sentence-transformers` is unavailable, the topic-modeling script falls back to TF-IDF-based KMeans clustering.

## Notes on Outputs

The output files are analysis artifacts. They are not raw data.

The most important output types are:

* post-level scored pragmatism files
* manual-inspection samples
* topic-modeling validation outputs


## Reproducibility

The scripts use fixed random seeds where sampling or modeling randomness is involved.

The main balanced sample is created with:

```python
random_seed = 42
```

The topic model also uses:

```python
random_seed = 42
```

This helps keep the analysis outputs stable across reruns, assuming the input data and package versions remain unchanged.

## Purpose in the Project

This folder is the analysis of structural pragmatism in financial-anxiety discussions on Reddit.
