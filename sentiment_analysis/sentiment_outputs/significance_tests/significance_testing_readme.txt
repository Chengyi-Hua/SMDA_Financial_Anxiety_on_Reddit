Significance and distribution testing summary
============================================

Main sentiment variable:
- no_emoji_xlm_sentiment_score: multilingual XLM-R text sentiment, emojis removed.

Distribution diagnostics:
- See distribution_summary.csv and normality_diagnostics.csv.
- Histograms are saved in distribution_figures/.
- Because sentiment scores are bounded and typically non-normal, non-parametric tests should be the main tests.

Main tests:
- Mann-Whitney U: 2025 vs 2020 within each community.
- Rank-biserial correlation: effect size for Mann-Whitney U.
- Bootstrap confidence intervals: mean and median differences.
- Kruskal-Wallis: overall test across the four community-year groups.
- Chi-square tests: sentiment label shares, emoji presence, emoji affect labels.

Interpretation warning:
- With large samples, p-values may be extremely small even for small effects.
- Report mean/median differences, confidence intervals, and effect sizes, not only p-values.
- Pooled comparisons are composition-sensitive because community sizes differ strongly across years.
