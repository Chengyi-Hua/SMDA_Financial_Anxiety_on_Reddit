# Structural Pragmatism Analysis README

## 1. What This Analysis Does

This analysis measures **structural pragmatism** in Reddit financial discourse.

The goal is to compare how the communities `finanzen` and `personalfinance` talk about practical financial decision-making in 2020 and 2025.

The main research question is:

> Do `finanzen` and `personalfinance` differ in the intensity, breadth, and thematic form of pragmatic financial discourse, and how do these patterns change from 2020 to 2025?

The answer is:

> Yes. Both communities are highly pragmatic, but `personalfinance` shows higher overall and broader structural pragmatism, while `finanzen` shows a more investment- and macro-oriented form of pragmatism.

In simple terms:

```text
personalfinance = product-, institution-, budgeting-, and problem-solving pragmatism

finanzen = investment-, market-, and macro-context pragmatism
```

---

## 2. Definition of Structural Pragmatism

In this project, **structural pragmatism** means financial discourse that frames personal finance in practical, actionable, institutional, or strategic terms.

A post is structurally pragmatic when it contains language about:

- financial products or institutions,
- investment strategy,
- budgeting and planning,
- tax, law, policy, or regulation,
- macroeconomic conditions,
- risk and uncertainty,
- concrete problem-solving,
- opportunity-seeking or financial agency.

Structural pragmatism is different from sentiment.

Sentiment asks:

> Is the post positive, negative, or neutral?

Structural pragmatism asks:

> Does the post contain practical financial reasoning?

Example:

```text
"I am worried about inflation."
```

This is mainly emotional or macroeconomic concern.

```text
"I am worried about inflation, so I am increasing my emergency fund and changing my ETF allocation."
```

This is structurally pragmatic because it connects concern to concrete financial action.


Important: structural pragmatism is measured as a lexicon-based indicator. This means the analysis captures the presence of practical, institutional, strategic, or action-oriented financial language. It does not prove that a post is objectively rational, useful, or high-quality advice. Instead, it measures how strongly a post uses language associated with pragmatic financial reasoning.
---

## 3. How Structural Pragmatism Is Measured

Structural pragmatism is measured with a bilingual English-German lexicon.

The lexicon has eight categories:

| Category | Meaning |
|---|---|
| `policy_regulation` | tax, law, regulation, government, institutional rules |
| `macro_economic_context` | inflation, recession, interest rates, crisis, labor market |
| `financial_institutions_products` | banks, brokers, accounts, loans, credit cards, insurance |
| `investment_strategy` | ETFs, stocks, portfolios, allocation, diversification |
| `planning_budgeting_control` | budgeting, planning, tracking, comparing, optimizing |
| `risk_uncertainty_management` | risk, volatility, liquidity, safety buffers |
| `practical_problem_solving` | reducing costs, switching providers, repaying debt |
| `opportunity_action_orientation` | wealth-building, financial opportunity, improving income or credit |


For each post, the script calculates:

| Variable | Meaning |
|---|---|
| `prag_total_count` | total number of pragmatism terms in the post |
| `prag_total_rate_per_100_words` | pragmatism terms per 100 words |
| `prag_unique_term_count` | number of unique pragmatism terms |
| `prag_category_count` | number of pragmatism categories present |
| `prag_category_diversity` | category count divided by 8 |
| `has_structural_pragmatism` | whether the post contains at least one pragmatism term |
| `dominant_prag_category` | the category with the highest count in the post |
| `structural_pragmatism_index` | combined intensity and diversity score |

The main index is:

```text
structural_pragmatism_index =
log1p(prag_total_rate_per_100_words) * (1 + prag_category_diversity)
```

This index combines two things:

1. **Intensity**: how many pragmatism terms appear per 100 words.
2. **Breadth**: how many different pragmatism categories appear.

The log transformation reduces the influence of extremely long or term-heavy posts.


### 3.1 Literature Grounding of the Operational Categories

The eight structural-pragmatism categories are not adopted from a single standardized taxonomy. They are operational coding categories constructed for this study. The category labels should therefore be understood as analytical labels, not as ready-made constructs taken directly from prior literature.

The framework is grounded in adjacent literatures on economic policy uncertainty, financial capability, household finance, investment behavior, mental accounting, financial literacy, risk perception, and financial well-being. These literatures support the conceptual dimensions behind the categories, while the exact category names are study-specific.

The papers listed below should therefore be read as **conceptual grounding**, not as direct sources of the exact category labels. The usable papers are within our google drive folder

| Operational category | Status | Conceptual grounding | How this supports the category |
|---|---|---|---|
| `policy_regulation` | Literature-informed conceptual grounding | Baker, Bloom, and Davis (2016) | Economic policy uncertainty research supports treating tax, law, government, fiscal policy, regulation, and institutional rules as meaningful dimensions of financial discourse. |
| `macro_economic_context` | Literature-informed conceptual grounding | Baker, Bloom, and Davis (2016) | Economic policy uncertainty research also supports the relevance of macroeconomic and contextual conditions such as recession, inflation, interest rates, unemployment, crises, and broader economic uncertainty. |
| `financial_institutions_products` | Literature-informed conceptual grounding | Johnson and Sherraden (2007) | Financial capability research emphasizes not only knowledge, but also access to and use of financial instruments and institutions. This supports coding banks, accounts, brokers, loans, credit cards, insurance, pensions, and retirement products. |
| `investment_strategy` | Study-specific operational category | Informed by household-finance and investment-behavior logic | This category operationalizes investment-oriented Reddit discourse, including ETFs, stocks, portfolios, allocation, diversification, returns, brokers, and long-term investment decisions. |
| `planning_budgeting_control` | Literature-informed conceptual grounding | Thaler (1999) | Mental accounting research supports coding language about budgeting, planning, tracking, organizing, categorizing, controlling spending, and managing financial activities. |
| `risk_uncertainty_management` | Study-specific operational extension | Informed by risk, uncertainty, and financial-security logic | This category operationalizes risk-related financial reasoning in Reddit posts, including volatility, uncertainty, liquidity, safety, buffers, emergency funds, and protective financial behavior. |
| `practical_problem_solving` | Study-specific operational extension | Informed by financial capability as applied behavior | This category captures concrete action-oriented financial reasoning, such as switching providers, reducing costs, repaying debt, avoiding fees, cancelling contracts, refinancing, and solving household financial problems. |
| `opportunity_action_orientation` | Study-specific operational extension | Informed by financial capability and agency logic | This category captures opportunity-seeking and agency-oriented financial reasoning, such as improving income, taking advantage of opportunities, building wealth, pursuing financial freedom, and turning financial knowledge into action. |

The purpose of this mapping is not to claim that prior research already defines these exact eight categories. Rather, the categories translate concepts from adjacent research traditions into observable discourse categories suitable for Reddit financial discussion.

In other words, the papers support the conceptual foundations of the framework, while the exact category labels and lexicon terms are part of this study’s operationalization.

In short:

```text
The labels are study-specific.
Some underlying dimensions are grounded in prior research.
The lexicon operationalizes those dimensions for Reddit financial discourse.

```


---

## 4. Why These Statistical Tests Are Used

Distribution diagnostics show that pragmatism variables are generally right-skewed and, especially at the category level, often zero-inflated.

They are mostly:

- count variables,
- rate variables,
- index variables,
- binary indicators,
- categorical labels.

Many category-level variables are zero-inflated and right-skewed.

For example, in the balanced 2025 sample:

| Variable | finanzen zero share | personalfinance zero share |
|---|---:|---:|
| `prag_policy_regulation_rate_per_100_words` | 73.7% | 61.3% |
| `prag_macro_economic_context_rate_per_100_words` | 62.6% | 81.6% |
| `prag_financial_institutions_products_rate_per_100_words` | 48.3% | 18.7% |
| `prag_planning_budgeting_control_rate_per_100_words` | 65.6% | 45.6% |
| `prag_practical_problem_solving_rate_per_100_words` | 85.5% | 77.8% |

This means that many posts have zero values for specific pragmatism categories, while a smaller number of posts contain many terms.

Therefore, the main testing strategy is non-parametric.

| Outcome type | Main test | Effect size / interpretation |
|---|---|---|
| Counts, rates, index scores | Mann-Whitney U test | rank-biserial correlation |
| Distributional differences | Kolmogorov-Smirnov test | KS statistic |
| Mean robustness check | Welch t-test | mean difference |
| Binary category presence | Chi-square test | Cramer’s V |
| Dominant category | Chi-square test | Cramer’s V |

P-values are adjusted using Benjamini-Hochberg correction.

Because the dataset is large, interpretation should focus on:

```text
direction + effect size + practical meaning
```

not p-values alone.

The balanced sample is the main inferential sample because each community-year group has the same size:

| Community | Year | Posts |
|---|---:|---:|
| finanzen | 2020 | 4,654 |
| finanzen | 2025 | 4,654 |
| personalfinance | 2020 | 4,654 |
| personalfinance | 2025 | 4,654 |

The full sample is used as a robustness check.

---

## 5. Main Descriptive Results

### 5.1 Share of Posts With Structural Pragmatism

In the balanced sample:

| Community | Year | Share with structural pragmatism |
|---|---:|---:|
| finanzen | 2020 | 93.1% |
| finanzen | 2025 | 90.5% |
| personalfinance | 2020 | 96.9% |
| personalfinance | 2025 | 96.8% |

Interpretation:

> Both communities are highly pragmatic. The main difference is not whether pragmatism exists, but what type of pragmatism dominates.

---

### 5.2 Structural Pragmatism Index

In the balanced sample:

| Community | Year | Mean index | Median index |
|---|---:|---:|---:|
| finanzen | 2020 | 2.57 | 2.70 |
| finanzen | 2025 | 2.42 | 2.53 |
| personalfinance | 2020 | 2.79 | 2.84 |
| personalfinance | 2025 | 2.82 | 2.86 |

Interpretation:

> `personalfinance` has a higher structural pragmatism index than `finanzen` in both years.

The difference becomes especially clear in 2025.

---

## 6. Main Statistical Results: Community Differences

The key comparison is:

```text
finanzen - personalfinance
```

Negative values mean `personalfinance` is higher.

Positive values mean `finanzen` is higher.

### 6.1 Overall Pragmatism, 2025

Balanced sample, 2025:

| Measure | finanzen | personalfinance | Difference | Rank-biserial effect |
|---|---:|---:|---:|---:|
| structural pragmatism index | 2.42 | 2.82 | -0.40 | -0.177 |
| total pragmatism rate per 100 words | 7.03 | 8.06 | -1.03 | -0.138 |
| total pragmatism count | 8.38 | 12.66 | -4.28 | -0.283 |
| pragmatism category count | 2.42 | 3.00 | -0.58 | -0.197 |

Statistical interpretation:

- The Mann-Whitney tests are statistically significant after Benjamini-Hochberg correction.
- However, because of the sample size, the interpretation focuses mainly on direction and effect size.
- The effect sizes are small to moderate.
- The practical direction is clear: `personalfinance` has more overall structural pragmatism, especially more total terms and broader category coverage.

Substantive interpretation:

> In 2025, `personalfinance` posts contain more pragmatic financial vocabulary overall and cover more types of pragmatic reasoning per post.

---

### 6.2 Overall Pragmatism, 2020

Balanced sample, 2020:

| Measure | finanzen | personalfinance | Difference | Rank-biserial effect |
|---|---:|---:|---:|---:|
| structural pragmatism index | 2.57 | 2.79 | -0.22 | -0.088 |
| total pragmatism rate per 100 words | 7.86 | 8.20 | -0.34 | -0.064 |
| total pragmatism count | 10.36 | 12.62 | -2.26 | -0.168 |
| pragmatism category count | 2.54 | 2.88 | -0.33 | -0.116 |

Interpretation:

> `personalfinance` was already more structurally pragmatic than `finanzen` in 2020, but the difference is smaller than in 2025.

---

## 7. Category-Level Results

The category-level results are the most important part of the story.

### 7.1 Category Presence, 2025

Balanced sample, 2025:

| Category | finanzen | personalfinance | Difference | Interpretation |
|---|---:|---:|---:|---|
| financial institutions/products | 51.7% | 81.3% | -29.6 pp | much higher in personalfinance |
| planning/budgeting/control | 34.4% | 54.4% | -20.1 pp | higher in personalfinance |
| policy/regulation | 26.3% | 38.7% | -12.4 pp | higher in personalfinance |
| practical problem-solving | 14.5% | 22.2% | -7.8 pp | higher in personalfinance |
| risk/uncertainty management | 14.5% | 20.6% | -6.0 pp | higher in personalfinance |
| opportunity/action orientation | 13.3% | 18.1% | -4.8 pp | higher in personalfinance |
| macro-economic context | 37.4% | 18.4% | +19.1 pp | higher in finanzen |
| investment strategy | 50.1% | 46.5% | +3.6 pp | slightly higher in finanzen |

Interpretation:

>`personalfinance` is more focused on institutions, products, accounts, budgeting, planning, and problem-solving.

>`finanzen` is more focused on macroeconomic context and investment strategy. The investment difference is modest when measured as simple category presence, but it becomes much clearer in the dominant-category results and in the investment-strategy rate.

---

### 7.2 Strongest Category Differences, 2025

Balanced sample, 2025:

| Category | Difference | Cramer’s V | Interpretation |
|---|---:|---:|---|
| financial institutions/products | -29.6 pp | 0.313 | strongest difference; much higher in personalfinance |
| macro-economic context | +19.1 pp | 0.212 | much higher in finanzen |
| planning/budgeting/control | -20.1 pp | 0.202 | higher in personalfinance |
| policy/regulation | -12.4 pp | 0.133 | higher in personalfinance |
| practical problem-solving | -7.8 pp | 0.100 | higher in personalfinance |
| investment strategy | +3.6 pp | 0.036 | slightly higher in finanzen |

Interpretation:

> The largest contrast is between product/institution pragmatism and macro/investment pragmatism.

---

## 8. Dominant Category Results

The dominant-category variable identifies which pragmatism category is strongest in each post.

### 8.1 Dominant Category, 2025

Balanced sample, 2025:

| Dominant category | finanzen count | personalfinance count |
|---|---:|---:|
| investment strategy | 1,492 | 580 |
| financial institutions/products | 1,236 | 2,792 |
| macro-economic context | 468 | 52 |
| planning/budgeting/control | 433 | 633 |
| policy/regulation | 423 | 383 |
| none | 441 | 151 |
| risk/uncertainty management | 57 | 32 |
| opportunity/action orientation | 56 | 18 |
| practical problem-solving | 48 | 13 |


In percentage terms, the dominant-category contrast is very clear:

| Dominant category | finanzen 2025 | personalfinance 2025 |
|---|---:|---:|
| investment strategy | 32.1% | 12.5% |
| financial institutions/products | 26.6% | 60.0% |
| macro-economic context | 10.1% | 1.1% |
| planning/budgeting/control | 9.3% | 13.6% |

This shows that `finanzen` is much more investment- and macro-centered, while `personalfinance` is much more product- and institution-centered.

Statistical result:

```text
Chi-square test for dominant category, 2025:
Cramer's V = 0.410
```

Interpretation:

> In 2025, `finanzen` is most often dominated by investment strategy, while `personalfinance` is overwhelmingly dominated by financial institutions and products.

This is one of the strongest findings in the analysis.

---

### 8.2 Dominant Category, 2020

Balanced sample, 2020:

| Dominant category | finanzen count | personalfinance count |
|---|---:|---:|
| investment strategy | 1,788 | 503 |
| financial institutions/products | 1,072 | 2,762 |
| policy/regulation | 500 | 575 |
| planning/budgeting/control | 435 | 457 |
| macro-economic context | 415 | 133 |
| none | 322 | 146 |
| opportunity/action orientation | 52 | 20 |
| risk/uncertainty management | 40 | 36 |
| practical problem-solving | 30 | 22 |

In percentage terms:

| Dominant category | finanzen 2020 | personalfinance 2020 |
|---|---:|---:|
| investment strategy | 38.4% | 10.8% |
| financial institutions/products | 23.0% | 59.3% |
| macro-economic context | 8.9% | 2.9% |
| planning/budgeting/control | 9.3% | 9.8% |

This shows that the community difference was already present in 2020.

Statistical result:

```text
Chi-square test for dominant category, 2020:
Cramer's V = 0.427
```

Interpretation:

> The same community difference already existed in 2020. `finanzen` was more investment-centered, while `personalfinance` was more institution/product-centered.

---

## 9. Change Over Time

### 9.1 finanzen: 2025 Compared With 2020

Balanced sample:

| Measure | 2020 | 2025 | Change |
|---|---:|---:|---:|
| structural pragmatism index | 2.57 | 2.42 | -0.16 |
| total pragmatism rate per 100 words | 7.86 | 7.03 | -0.82 |
| total pragmatism count | 10.36 | 8.38 | -1.98 |
| investment strategy rate | 3.72 | 2.86 | -0.85 |
| macro-economic context rate | 0.52 | 0.63 | +0.12 |
| planning/budgeting/control rate | 0.77 | 0.70 | -0.07 |

Interpretation:

> `finanzen` shows a decline in overall structural pragmatism from 2020 to 2025.  
> The decline is mainly driven by less investment-strategy language.  
> At the same time, macroeconomic context becomes slightly more prominent.

---

### 9.2 personalfinance: 2025 Compared With 2020

Balanced sample:

| Measure | 2020 | 2025 | Change |
|---|---:|---:|---:|
| structural pragmatism index | 2.79 | 2.82 | +0.03 |
| total pragmatism rate per 100 words | 8.20 | 8.06 | -0.14 |
| total pragmatism count | 12.62 | 12.66 | +0.04 |
| pragmatism category count | 2.88 | 3.00 | +0.13 |
| planning/budgeting/control rate | 1.06 | 1.34 | +0.28 |
| investment strategy rate | 1.24 | 1.28 | +0.04 |

Interpretation:

> `personalfinance` remains stable in overall pragmatism, but its pragmatism becomes slightly broader and more planning/budgeting-oriented.

---

## 10. EPU-Style Submeasure

The analysis also includes a stricter EPU-style submeasure based on the triad:

```text
economy + uncertainty + policy
```

A post receives the EPU triad flag only if all three elements appear.

Balanced sample:

| Community | Year | EPU triad share |
|---|---:|---:|
| finanzen | 2020 | 0.77% |
| finanzen | 2025 | 0.64% |
| personalfinance | 2020 | 0.49% |
| personalfinance | 2025 | 0.52% |

Interpretation:

> EPU-style discourse is rare in both communities.  
> Most pragmatic financial discourse on Reddit is not full economy-policy-uncertainty discourse.  
> Instead, it is mostly practical reasoning around products, accounts, investments, budgeting, and decisions.

This means the EPU measure should be treated as a secondary finding, not the main pragmatism measure.

---

## 11. Topic-Modeling Validation

To validate the structural-pragmatism categories inductively, a KMeans-based topic model was estimated on the balanced sample. The topic model is not the main inferential method. Instead, it is used as an exploratory validation layer to check whether data-driven topic clusters broadly reproduce the same community-level contrast found in the lexicon-based analysis.

The topic model was fitted on the full balanced corpus:

```text
finanzen 2020
finanzen 2025
personalfinance 2020
personalfinance 2025
```

This keeps all posts in one shared topic space and then compares topic prevalence between `finanzen` and `personalfinance` separately for 2020 and 2025.

The method is:

```text
TF-IDF representation + KMeans clustering + topic-to-category mapping
```

The resulting topics were interpreted through their highest-weighted TF-IDF terms and then mapped back to the predefined structural-pragmatism categories. This means the topic model does not create new theoretical categories. It asks whether the topics that emerge from the text can be meaningfully aligned with the eight structural-pragmatism categories used in the main lexicon analysis.

Because HDBSCAN-based BERTopic could not be run reliably in the local Windows environment, the validation uses this no-HDBSCAN topic-modeling approach.

---

### 11.1 Topic-Derived Category Profile, 2020 Crisis Year

In the 2020 crisis-year comparison, the topic model reproduces the main community contrast.

| Topic-derived category | finanzen | personalfinance | Interpretation |
|---|---:|---:|---|
| policy/regulation | 43.9% | 8.5% | higher in finanzen |
| macro-economic context | 12.6% | 0.0% | higher in finanzen |
| financial institutions/products | 8.9% | 54.9% | much higher in personalfinance |
| investment strategy | 28.2% | 7.4% | higher in finanzen |
| planning/budgeting/control | 5.2% | 29.2% | higher in personalfinance |

Interpretation:

> In the 2020 crisis year, `finanzen` topic clusters are more concentrated around policy/regulation, macro-context, and investment strategy, while `personalfinance` topic clusters are more concentrated around financial products, institutions, accounts, credit, retirement, and planning/budgeting.

This supports the main lexicon-based finding that the community difference was already present in 2020.

---

### 11.2 Topic-Derived Category Profile, 2025

The same broad contrast is visible in 2025.

| Topic-derived category | finanzen | personalfinance | Interpretation |
|---|---:|---:|---|
| policy/regulation | 48.0% | 5.0% | much higher in finanzen |
| macro-economic context | 14.1% | 0.0% | higher in finanzen |
| financial institutions/products | 9.6% | 52.9% | much higher in personalfinance |
| investment strategy | 21.6% | 8.1% | higher in finanzen |
| planning/budgeting/control | 6.7% | 34.0% | higher in personalfinance |

Interpretation:

> In 2025, the topic model again shows a strong community-level divide. `personalfinance` remains centered on product-, institution-, account-, credit/debt-, retirement-, and planning-related topics, while `finanzen` remains more strongly associated with policy/regulation, macro-context, and investment topics.

---

### 11.3 Representative Topics

The topic model also produced interpretable topic examples that match the category-level story.

| Structural-pragmatism category | Representative topic examples | Interpretation |
|---|---|---|
| policy/regulation | `T20: tax, taxes, file, income, return`; `T16: steuererklärung, steuer, arbeitgeber, bav, steuern`; `T14: deutschland, schweiz, steuern, usa, ausland` | tax filing, German tax context, cross-border policy/regulatory context |
| macro-economic context | `T6: zinsen, kredit, scalable, scalable capital, capital`; `T15: auto, gehalt, studium, ausbildung, arbeiten` | interest-rate and broader economic/labor-market context |
| financial institutions/products | `T13: account, savings, bank, savings account, checking`; `T7: retirement, ira, roth, employer, match`; `T11: card, credit, credit card, pay, balance`; `T21: konto, bank, kreditkarte, c24, girokonto` | accounts, banks, credit cards, retirement accounts, German banking products |
| investment strategy | `T29: etfs, investieren, etf, sparplan, fonds`; `T27: invest, investing, fund, funds, stock`; `T26: etf, sparplan, a1jx52, world, investieren`; `T17: aktien, aktie, kaufen, verkaufen, depot`; `T0: msci, world, msci world, etf, ishares` | ETFs, stocks, MSCI World, Sparplan, portfolios, brokers |
| planning/budgeting/control | `T2: job, work, pay, company, rent`; `T22: haus, wohnung, eltern, immobilie, miete`; `T23: car, miles, loan, buy, payment` | rent, work, income, housing, car purchase, household planning |

These topic examples show that the topic model broadly recovers the same substantive structure as the lexicon analysis. `personalfinance` topics cluster around products, accounts, credit, debt, retirement, and household planning, while `finanzen` topics cluster around ETFs, stocks, brokers, German tax, interest rates, and macro-financial context.

---

### 11.4 Interpretation of the Topic-Modeling Validation

The topic model supports the main finding:

```text
personalfinance = product-, institution-, budgeting-, credit/debt-, retirement-, and household-planning pragmatism

finanzen = investment-, portfolio-, broker/depot-, tax-, policy-, and macro-context pragmatism
```

The validation is useful because it shows that the community contrast is not only visible in lexicon counts. It also appears in the corpus structure when posts are grouped into data-driven topics.

In other words:

> The lexicon analysis measures the category differences directly, while the topic model shows that similar thematic clusters emerge inductively from the text.

---

### 11.5 Limitations of the Topic Model

The topic-modeling results should be interpreted as exploratory.

Some topics are semantically mixed. For example, broader German-language discussion topics can combine general discussion terms with policy, tax, insurance, or macroeconomic vocabulary. Therefore, the topic-to-category mapping is less precise than the lexicon-based category scoring.

The overall silhouette score is also low, which is common for noisy high-dimensional Reddit text clustered with TF-IDF and KMeans. This means the topic model should not be treated as the main statistical proof.

Instead, the topic model is best understood as a robustness and triangulation step:

```text
Main evidence = lexicon-based structural-pragmatism scores and statistical tests

Supporting evidence = KMeans topic-modeling validation
```

The topic model broadly confirms the main community contrast, but the final interpretation should rely primarily on the lexicon-based results.





## 12. Final Interpretation and Conclusion

The pragmatism analysis shows that both `finanzen` and `personalfinance` are highly pragmatic financial communities. In the balanced sample, more than 90% of posts in every community-year group contain at least one structural-pragmatic element.

However, the communities differ clearly in the form of pragmatism they emphasize.

`personalfinance` shows higher overall structural pragmatism. In 2025, the balanced-sample structural pragmatism index is 2.42 for `finanzen` and 2.82 for `personalfinance`. The difference is statistically significant after Benjamini-Hochberg correction, with a rank-biserial effect of -0.177. This means that `personalfinance` posts tend to contain more and broader structural-pragmatic language.

The category-level results explain this difference. `personalfinance` is much more strongly associated with financial institutions and products, budgeting, planning, risk management, and practical problem-solving. In contrast, `finanzen` is more strongly associated with investment strategy and macroeconomic context.

The strongest category contrast is the dominant-category result. In 2025, investment strategy is the dominant pragmatism category in 32.1% of `finanzen` posts but only 12.5% of `personalfinance` posts. By contrast, financial institutions and products dominate 60.0% of `personalfinance` posts but only 26.6% of `finanzen` posts.

The main conclusion is therefore:

```text
Both communities are pragmatic, but they are pragmatic in different ways.

personalfinance = broader household-practical, product-, institution-, budgeting-, and problem-solving pragmatism

finanzen = investment-, portfolio-, market-, and macro-context pragmatism


```
