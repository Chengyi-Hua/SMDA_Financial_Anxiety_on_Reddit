SLIDE 1 — Data Collection
Reddit data was collected using the Arctic Shift API, chosen over PullPush because it can return up to 1000 posts per call depending on server capacity (requiring fewer API calls for pagination), and it allows selecting specific fields to reduce response time and size.
Data was collected from two subreddits across two time periods:

r/personalfinance (English) — 2020 and 2025
r/Finanzen (German) — 2020 and 2025

This results in four datasets: EN–2020, EN–2025, DE–2020, DE–2025.

SLIDE 2 — Data Cleaning Overview
All cleaning steps from the course were applied in a way suited to our context. The pipeline is:
Missing Values & Short Posts → Deduplication → Bot & Spam Detection → Language Filtering → Text Normalization → Anonymisation → Final Quality Checks
DatasetPosts (raw)Posts (after cleaning)r/personalfinance 2020~182,000~103,000r/personalfinance 2025~109,000~72,700r/Finanzen 2020~8,000~4,650r/Finanzen 2025~33,000~22,700

SLIDE 3 — Missing Values, Deduplication, Bot Detection & Language Filtering
Missing Values & Short Posts

Rows where selftext is NaN or contains [deleted] / [removed] were removed.
Posts below 60 characters (title + selftext combined) were removed after manually inspecting posts at different thresholds.

Deduplication

Title and selftext were merged into a full_text column.
Duplicate rows based on full_text were removed, keeping the first occurrence.

Bot & Spam Detection

Content repetition bots were handled via deduplication.
Link-only and image-only bot posts were removed via the short post filter.
A post velocity filter (threshold: 5 posts/hour) was applied — no suspicious accounts were detected, suggesting bot activity is negligible.

Language Filtering

Language detection was performed using fastText with a confidence threshold of 0.90.
Only English posts were kept in r/personalfinance datasets; only German posts in r/Finanzen datasets.


SLIDE 4 — Text Normalization
Two separate normalized text columns were created to support different downstream analyses:
text_for_sentiment — suitable for sentiment analysis (e.g. VADER):

HTML entity decoding
URL → [URL] token
Reddit username → [USER] token, subreddit → [SUBREDDIT] token
Whitespace normalization
Emojis, punctuation, and casing are preserved (they carry sentiment signals)

text_for_keywords — suitable for keyword frequency analysis:

All steps from text_for_sentiment, plus lowercasing and punctuation/special character removal
Punctuation is replaced with a space (not removed) to avoid incorrect word merging (e.g. "I did this.He did that")

Tokens were used instead of full removal to preserve sentence context, which may also benefit potential LLM-based analysis later.

SLIDE 5 — Anonymisation & Final Quality Checks
Anonymisation & Column Cleanup

The author column (visible Reddit usernames) was removed to reduce privacy risks.
The temporary combined_length column was dropped.

Final Quality Checks

Distribution check: Post volume over time was plotted for all four datasets. No abrupt spikes or gaps were observed, suggesting stable and reliable data collection.
Top-account audit: The top 10 most active accounts per dataset were reviewed. No single user disproportionately dominates any dataset.
Text sample review: Random text samples from both final columns were manually inspected to verify normalization steps were applied correctly.
