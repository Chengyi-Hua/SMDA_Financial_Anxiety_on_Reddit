Data Collection
Reddit data was collected using the Arctic Shift API, chosen over PullPush because it can return up to 1000 posts per call depending on server capacity (requiring fewer API calls for pagination), and it allows selecting specific fields to reduce response time and size.
Data was collected from two subreddits across two time periods:

r/personalfinance (English) — 2020 and 2025
r/Finanzen (German) — 2020 and 2025

This results in four datasets: EN–2020, EN–2025, DE–2020, DE–2025.

Data Cleaning
All cleaning steps from the course were applied in a way suited to our context. The pipeline is:
Missing Values & Short Posts → Deduplication → Bot & Spam Detection → Language Filtering → Text Normalization → Anonymisation → Final Quality Checks

Missing Values & Short Posts

Rows where selftext is missing (NaN) or contains placeholder values ([deleted], [removed]) were removed, as these contain no meaningful textual content.
A combined_length column (title + selftext character count) was computed. Posts below 60 characters were removed after manually inspecting posts at different thresholds — posts below this threshold were consistently very short, low-effort, or already removed by moderators.


Deduplication

Title and selftext were merged into a single full_text column, since in some cases the main question is expressed in the title while the selftext is very short.
Duplicate rows based on full_text were removed, keeping the first occurrence.


Bot & Spam Detection
Several bot filtering steps were already addressed in earlier stages:

Deduplication handled content repetition bots.
The short post filter removed link-only and image-only bot posts.

As an additional measure, a post velocity filter was applied: accounts posting more than 5 posts per hour were flagged as suspicious. No suspicious accounts were detected across any of the four datasets, suggesting bot activity is negligible — likely due to active subreddit moderation.

Language Filtering
Language detection was performed using fastText (Meta's multilingual model), which is fast, accurate, and deterministic. A confidence threshold of 0.90 was applied:

Only English posts were kept in r/personalfinance datasets.
Only German posts were kept in r/Finanzen datasets.


Text Normalization
Two separate normalized text columns were created to support different downstream analyses:
text_for_sentiment — suitable for sentiment analysis (e.g. VADER):

HTML entity decoding
URL replacement with [URL] token
Reddit username & subreddit replacement with [USER] / [SUBREDDIT] tokens
Whitespace normalization
Emojis, punctuation, and casing are preserved (they carry sentiment signals)

text_for_keywords — suitable for keyword frequency analysis:

All steps from text_for_sentiment, plus:
Lowercasing
Punctuation & special character removal (replaced with space to avoid incorrect word merging)
Whitespace normalization

Tokens (e.g. [URL], [USER]) were used instead of full removal to preserve sentence context, which may also benefit potential LLM-based analysis later.

Anonymisation & Column Cleanup
The author column (containing visible Reddit usernames) was removed to reduce privacy risks. The temporary combined_length column was also dropped.

Final Quality Checks

Distribution check: Post volume over time was plotted for all four datasets. No abrupt spikes or gaps were observed, suggesting stable and reliable data collection.
Top-account audit: The top 10 most active accounts per dataset were reviewed. No single user disproportionately dominates any dataset, indicating balanced user activity.
Text sample review: Random text samples from text_for_sentiment and text_for_keywords were manually inspected to verify that normalization steps were applied correctly.
