# Write-Up — Trader Performance vs Market Sentiment

---

## Methodology

**Data:** Two datasets were merged on daily date — 211,000+ Hyperliquid trades across 32 accounts, and 479 days of Bitcoin Fear/Greed index values (April 2023 – April 2025). After deduplication and an inner join, the working dataset contained 2,342 trader-day records.

**Feature Engineering:** For each trader-day, computed: total PnL, win rate, trade count, average trade size (USD), long/short ratio, and a leverage proxy (mean Size USD ÷ |Start Position|, capped at 99th percentile to remove outliers). A 5-day rolling PnL volatility was added per account for the ML model.

**Segmentation:** Accounts were labelled using quantile thresholds on lifetime statistics — Whale (top 25% trade size), Degen (top 25% leverage), Grinder (low leverage + high trade count), Retail (all others). K-means (k=3, StandardScaler) provided a parallel data-driven clustering independent of manual labels.

**Statistical Testing:** Mann-Whitney U test applied to validate whether Fear vs Greed PnL differences were statistically significant beyond outlier effects.

**ML Models:** Two Random Forest classifiers trained with a time-based 80/20 split (no data leakage) — one predicting next-day profitability (binary, ROC-AUC = 0.606) and one predicting next-day PnL volatility bucket (Low / Medium / High).

---

## Key Insights

**Insight 1 — Whales profit more on Fear days (counter-intuitive)**
Chart 4 shows Whales average **$10,861/day on Fear vs $2,632 on Greed** — a 4× gap. Large capital enables contrarian positioning: they absorb panic-selling and exit profitably during dislocations while smaller traders flee.

**Insight 2 — Leverage rises on Fear days across all segments**
Chart 2: median leverage proxy is **26.03 on Fear vs 17.60 on Greed**. Chart 5 confirms this across all four segments. Traders do not de-risk during market stress — they increase leverage while shrinking position sizes, a counter-intuitive but consistent pattern.

**Insight 3 — Grinders are the most regime-resilient segment**
Chart 4: Grinders earn **$7,071 on Fear vs $5,755 on Greed** — the smallest swing among all segments. High-frequency, low-leverage trading captures consistent returns regardless of market direction.

**Insight 4 — Degens lose money on Fear days**
Chart 4: Degen avg PnL on Fear = **−$23.8** vs +$1,820 on Greed. Their amplified leverage turns Fear-driven volatility into net losses, making them the most sentiment-sensitive segment.

---

## Strategy Recommendations

### Strategy 1 — "Fear Brake" for Degen Traders
> When Fear/Greed index < 40: cap leverage at 3×, cut position size by 30%, avoid longs when long_ratio > 0.6

Degens average a net loss on Fear days. Their high leverage amplifies drawdowns during volatile, downward markets. A hard leverage cap during Fear periods limits losses and preserves capital for the Greed-phase recovery where their strategy works well.

### Strategy 2 — "Greed Filter" for Grinder Traders
> During Fear: maintain or increase trade frequency. On Extreme Greed (index > 75): reduce frequency ~20%

Grinders thrive on mean-reversion setups, which are abundant during Fear due to elevated volatility. Extreme Greed markets are trend-driven and punish their scalping edge with false breakouts. Throttling frequency during Extreme Greed protects their win rate.

---

*Tools: Python · pandas · scikit-learn · scipy · matplotlib · seaborn · streamlit*
