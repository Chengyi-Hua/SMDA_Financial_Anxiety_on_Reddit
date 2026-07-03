# Sentiment Analysis Summary

## Aim

This analysis examines whether the sentiment of financial Reddit posts changed between **2020** and **2025** in two communities:

- `r/Finanzen` — German-language financial discussion
- `r/personalfinance` — English-language financial discussion

The analysis separates three layers:

1. **Main text sentiment** using multilingual XLM-RoBERTa
2. **Rule-based baseline sentiment** using VADER and GerVADER
3. **Emoji sentiment** using the Emoji Sentiment Ranking by Kralj Novak et al.

The main interpretation is based on XLM-RoBERTa. VADER, GerVADER, emoji sentiment, and manual validation are used as supporting analyses.

---

## Data

Four cleaned Reddit datasets were analyzed:

```text
finanzen_2020_final.csv
finanzen_2025_final_with_emo.csv
personalfinance_2020_final_with_emo.csv
personalfinance_2025_final.csv
```

After language-confidence filtering, the final sample sizes were:

| Community | Year | Posts |
|---|---:|---:|
| finanzen | 2020 | 4,654 |
| finanzen | 2025 | 22,669 |
| personalfinance | 2020 | 103,192 |
| personalfinance | 2025 | 72,726 |

---

## Main Sentiment Model

The primary sentiment model was:

```text
cardiffnlp/twitter-xlm-roberta-base-sentiment
```

This model was chosen because it is multilingual and can process both German and English posts.

For the main text sentiment analysis, emojis were removed before scoring. Emoji sentiment was analyzed separately.

The main score was calculated as:

```text
no_emoji_xlm_sentiment_score = positive_probability - negative_probability
```

Interpretation:

| Score direction | Meaning |
|---|---|
| Positive | More positive sentiment |
| Around 0 | Neutral or mixed sentiment |
| Negative | More negative sentiment |

---

## Main XLM-RoBERTa Results

| Community | Mean 2020 | Mean 2025 | Change 2025 - 2020 | Interpretation |
|---|---:|---:|---:|---|
| finanzen | -0.288 | -0.317 | -0.029 | More negative in 2025 |
| personalfinance | -0.313 | -0.304 | +0.010 | Slightly less negative in 2025 |
| pooled | -0.312 | -0.307 | +0.005 | Slightly less negative overall, but composition-sensitive |

The within-community comparisons are more important than the pooled comparison because the two communities differ strongly in size, language, and discussion style.

### Interpretation

The main XLM-RoBERTa results suggest that:

- `r/Finanzen` became slightly more negative from 2020 to 2025.
- `r/personalfinance` became slightly less negative from 2020 to 2025.
- The changes are statistically significant, but the effect sizes are small.

This indicates modest but systematic sentiment changes rather than a dramatic emotional shift.

---

## Statistical Testing

Because sentiment scores are bounded and non-normally distributed, non-parametric tests were used as the main statistical approach.

The main tests were:

| Outcome | Test |
|---|---|
| Continuous sentiment scores | Mann-Whitney U |
| Practical effect size | Rank-biserial correlation |
| Distributional differences | Kolmogorov-Smirnov test |
| Mean robustness check | Welch t-test |
| Label distributions | Chi-square test |
| Emoji-use proportions | Chi-square test / proportion comparison |

Because the dataset is large, interpretation focuses on:

```text
direction + effect size + practical meaning
```

rather than p-values alone.

---

## Key Statistical Results

### XLM-RoBERTa: finanzen

```text
mean 2020 = -0.288
mean 2025 = -0.317
difference = -0.029
Mann-Whitney p ≈ 1.89e-14
rank-biserial effect ≈ -0.071
KS p ≈ 4.32e-22
Welch p ≈ 2.41e-13
```

Interpretation:

> `r/Finanzen` became significantly more negative in 2025, but the effect size is small.

### XLM-RoBERTa: personalfinance

```text
mean 2020 = -0.313
mean 2025 = -0.304
difference = +0.010
Mann-Whitney p ≈ 4.56e-19
rank-biserial effect ≈ +0.025
KS p ≈ 2.21e-36
Welch p ≈ 4.00e-15
```

Interpretation:

> `r/personalfinance` became significantly less negative in 2025, but the effect size is very small.

---

## XLM-RoBERTa Label Shares

### finanzen

| Year | Negative | Neutral | Positive |
|---:|---:|---:|---:|
| 2020 | 31.37% | 65.99% | 2.64% |
| 2025 | 38.11% | 58.30% | 3.59% |

Interpretation:

> `r/Finanzen` shows a higher share of negative posts in 2025.

### personalfinance

| Year | Negative | Neutral | Positive |
|---:|---:|---:|---:|
| 2020 | 61.13% | 36.93% | 1.94% |
| 2025 | 59.56% | 38.31% | 2.14% |

Interpretation:

> `r/personalfinance` shows a slight decrease in negative posts and a slight increase in neutral posts in 2025.

---

## VADER Baseline

Standard VADER was included as a rule-based baseline. However, it was not used as the main result.

Standard VADER was problematic because:

1. It is mainly designed for English and is not suitable for German `r/Finanzen` posts.
2. It often misread financial advice language.
3. It sometimes classified practical questions as strongly positive or strongly negative because of isolated words.

### Standard VADER Results

| Community | Mean 2020 | Mean 2025 | Change 2025 - 2020 |
|---|---:|---:|---:|
| finanzen | -0.646 | -0.611 | +0.036 |
| personalfinance | 0.454 | 0.448 | -0.006 |

Manual validation showed that VADER was frequently misleading:

```text
VADER plausible = 42.31%
VADER misleading = 57.69%
```

Therefore, VADER is treated only as a baseline and not as the main sentiment measure.

---

## GerVADER Robustness Check

Because standard VADER is not suitable for German, GerVADER was added as a German-specific baseline for `r/Finanzen`.

| Community | Year | GerVADER Mean | GerVADER Median |
|---|---:|---:|---:|
| finanzen | 2020 | 0.673 | 0.868 |
| finanzen | 2025 | 0.637 | 0.848 |

GerVADER label shares:

| Year | Negative | Neutral | Positive |
|---:|---:|---:|---:|
| 2020 | 9.15% | 2.21% | 88.63% |
| 2025 | 10.65% | 2.83% | 86.52% |

Interpretation:

> GerVADER gives much more positive absolute scores than XLM-RoBERTa, but it supports the same direction for `r/Finanzen`: 2025 is less positive / more negative than 2020.

This strengthens the interpretation that the German community became slightly more negative in 2025.

---

## Emoji Usage

Emoji use increased strongly in both communities.

| Community | Year | Posts with emoji | Emoji share | Total emoji count |
|---|---:|---:|---:|---:|
| finanzen | 2020 | 51 | 1.10% | 68 |
| finanzen | 2025 | 1,535 | 6.77% | 2,580 |
| personalfinance | 2020 | 685 | 0.66% | 952 |
| personalfinance | 2025 | 1,919 | 2.64% | 2,699 |

Change in emoji use:

| Community | Emoji share 2020 | Emoji share 2025 | Change |
|---|---:|---:|---:|
| finanzen | 1.10% | 6.77% | +5.68 percentage points |
| personalfinance | 0.66% | 2.64% | +1.97 percentage points |

Interpretation:

> Emoji use became much more common between 2020 and 2025, especially in `r/Finanzen`.

---

## Emoji Sentiment

Emoji sentiment was analyzed using the Emoji Sentiment Ranking by Kralj Novak et al.

The emoji score is:

```text
ESR score = positive emoji proportion - negative emoji proportion
```

Mean emoji sentiment:

| Community | Year | Mean emoji sentiment | Median emoji sentiment |
|---|---:|---:|---:|
| finanzen | 2020 | 0.384 | 0.421 |
| finanzen | 2025 | 0.371 | 0.417 |
| personalfinance | 2020 | 0.289 | 0.279 |
| personalfinance | 2025 | 0.260 | 0.298 |

Interpretation:

- Emojis were mostly positive in both communities.
- Emoji sentiment stayed broadly similar in `r/Finanzen`.
- Emoji sentiment became slightly lower in `r/personalfinance`.
- Emojis should not be interpreted as simple positive/negative replacements for text sentiment.

Manual inspection showed that emojis often functioned as:

| Emoji type | Function |
|---|---|
| 😅 | Softening embarrassment, uncertainty, or anxiety |
| ❤️ | Gratitude, warmth, support |
| 🙏 | Politeness or thanks |
| 🎁 | Literal, symbolic, promotional, or positive cue |
| 🏧 / 📷 | Often symbolic rather than emotional |

Therefore:

> Emoji sentiment is best interpreted as an additional tone layer, not as the main sentiment result.

---

## Manual Validation

A compact manual inspection sample was created to check whether model outputs were plausible.

The sample included:

- random posts
- most negative XLM-RoBERTa posts
- most neutral XLM-RoBERTa posts
- strongest emoji-affect posts
- negative-text / positive-emoji cases

### Manual Validation Summary

| Component | Main finding | Interpretation |
|---|---|---|
| XLM-RoBERTa | 98.08% mostly plausible | Suitable as the primary sentiment measure |
| VADER | 57.69% frequently misleading | Baseline only |
| Emoji sentiment | 32.69% useful but context-dependent | Adds tone information |

Detailed validation:

| Component | Assessment | Count | Share |
|---|---|---:|---:|
| Overall manual assessment | plausible | 47 | 90.38% |
| Overall manual assessment | plausible with emoji caveat | 3 | 5.77% |
| Overall manual assessment | questionable | 2 | 3.85% |
| XLM-RoBERTa | plausible | 50 | 96.15% |
| XLM-RoBERTa | questionable / borderline | 2 | 3.85% |
| VADER | plausible | 22 | 42.31% |
| VADER | misleading | 30 | 57.69% |

Interpretation:

> Manual validation supports XLM-RoBERTa as the main sentiment model. VADER is much less reliable, especially in this multilingual financial-advice context. Emoji sentiment is useful but must be interpreted contextually.

---

## Overall Findings

The sentiment analysis suggests the following:

1. **`r/Finanzen` became slightly more negative from 2020 to 2025.**  
   This appears in XLM-RoBERTa and is directionally supported by GerVADER.

2. **`r/personalfinance` became slightly less negative from 2020 to 2025.**  
   The change is statistically significant but very small.

3. **Emoji use increased substantially in both communities.**  
   The increase was especially strong in `r/Finanzen`.

4. **Emojis were mostly positive or softening in tone.**  
   However, emoji meaning was context-dependent and sometimes symbolic.

5. **Standard VADER was not reliable enough as a main measure.**  
   It was frequently misleading in manual validation and is especially unsuitable for German posts.

6. **GerVADER improved the German baseline but was still only a robustness check.**  
   It supported the direction of the `r/Finanzen` trend but produced much more positive absolute scores than XLM-RoBERTa.

7. **The overall sentiment changes are statistically detectable but modest.**  
   The interpretation should emphasize direction and effect size rather than p-values alone.

---

## Final Interpretation

The sentiment analysis shows small but systematic shifts in financial Reddit discourse between 2020 and 2025.

Using multilingual XLM-RoBERTa as the primary sentiment model, `r/Finanzen` became slightly more negative, while `r/personalfinance` became slightly less negative. These changes were statistically significant but small in practical magnitude.

Emoji use increased substantially in both communities, especially in `r/Finanzen`. Emojis were mostly positive or softening in tone, often adding friendliness, politeness, gratitude, or playful self-consciousness to otherwise neutral or negative financial posts.

Manual validation supported XLM-RoBERTa as the main sentiment measure. Standard VADER was frequently misleading, while GerVADER provided a useful German-specific robustness check that confirmed the directional trend for `r/Finanzen`.

Overall, the results suggest not a dramatic sentiment transformation, but a modest change in affective tone combined with increased use of positive or softening emojis in financial discussions.

---

## Short Report Wording

The sentiment analysis indicates small but systematic changes in financial Reddit discourse between 2020 and 2025. Multilingual XLM-RoBERTa was used as the primary model because it supports both German and English text. Emojis were removed from the main text sentiment analysis and analyzed separately using the Emoji Sentiment Ranking. The XLM-RoBERTa results show that `r/Finanzen` became slightly more negative in 2025, whereas `r/personalfinance` became slightly less negative. These differences were statistically significant, but effect sizes were small. Emoji use increased substantially in both communities, particularly in `r/Finanzen`, and emojis were mostly positive or softening in tone. Manual validation supported XLM-RoBERTa as the main sentiment measure, while standard VADER was frequently misleading. A German-specific GerVADER robustness check supported the directional trend for `r/Finanzen`, although its absolute scores were much more positive than XLM-RoBERTa. Overall, the results suggest modest changes in affective tone rather than a dramatic sentiment shift.

---

## Limitations

1. **Sentiment is not the same as financial anxiety.**  
   Sentiment measures general emotional tone. It does not directly measure financial anxiety unless combined with more specific anxiety indicators.

2. **Large samples make small differences statistically significant.**  
   The practical interpretation should focus on effect size and direction.

3. **VADER is not reliable as a main model.**  
   It is especially problematic for German posts and domain-specific financial advice language.

4. **GerVADER is useful but not definitive.**  
   It improves the German baseline but produces much more positive absolute scores than XLM-RoBERTa.

5. **Emoji sentiment is context-dependent.**  
   Emojis can express emotion, but they can also be symbolic, literal, promotional, or used for politeness.

6. **Pooled comparisons are composition-sensitive.**  
   The within-community comparisons are more meaningful than the pooled comparison.

---

## Main Conclusion

The strongest conclusion is:

> Financial Reddit sentiment did not radically change between 2020 and 2025, but the emotional tone shifted modestly. `r/Finanzen` became slightly more negative, while `r/personalfinance` became slightly less negative. Emoji use increased substantially and mostly added positive, polite, or softening tone to financial discussions.