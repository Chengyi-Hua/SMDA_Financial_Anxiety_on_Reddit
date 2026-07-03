# No-HDBSCAN Topic Modeling Validation Summary

- Input file: `D:\Users\cheng\Documents\GitHub\SMDA_Financial_Anxiety_on_Reddit\pragmatism_analysis\outputs\pragmatism_balanced_sample.csv`
- Documents used: `18,616`
- Number of topics: `30`
- Clustering source: `sentence_transformers`
- Overall silhouette score: `0.032567400485277176`

## Purpose

This model is used as an inductive validation layer for the structural pragmatism categories. When available, it clusters posts using multilingual sentence-transformer embeddings to reduce German-English vocabulary confounding. TF-IDF is retained for topic-term labeling. It avoids BERTopic/HDBSCAN and therefore does not require Microsoft C++ Build Tools.

## Most important files

- `tables/topic_model_topic_info.csv`
- `tables/topic_model_topic_category_mapping.csv`
- `tables/topic_model_mapped_category_prevalence_by_group.csv`
- `figures/01_topic_differences_2020.png`
- `figures/02_topic_differences_2025.png`
- `figures/03_topic_mapped_category_profile_2020.png`
- `figures/04_topic_mapped_category_profile_2025.png`
- `figures/05_topic_driver_bubble_2020.png`
- `figures/06_topic_driver_bubble_2025.png`
- `figures/07_topic_driver_key_2020.png`
- `figures/08_topic_driver_key_2025.png`
