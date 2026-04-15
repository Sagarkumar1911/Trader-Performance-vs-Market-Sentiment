# Trader Performance vs Market Sentiment Analysis
### Primetrade.ai — Data Science Intern Assignment

A complete end-to-end analysis of how Bitcoin market sentiment (Fear/Greed) influences trader behaviour and profitability on Hyperliquid, including ML models and an interactive Streamlit dashboard.

---

## Dataset Overview

| Dataset | Records | Description |
|---|---|---|
| `historical_data.csv` | 211,000+ trades | Hyperliquid trader activity |
| `fear_greed_index.csv` | 479 days | Bitcoin Fear/Greed Index |
| After merge | 2,342 daily trader-day records | 32 traders × trading days |

---

## Project Structure

```
├── analysis_clean.py          # Core analysis + ML pipeline
├── app.py                     # Streamlit dashboard
├── requirements.txt           # Python dependencies
├── historical_data.csv        # Raw trade data
├── fear_greed_index.csv       # Sentiment data
├── outputs/
│   ├── daily_trader_summary.csv          # 2,342 daily records per trader
│   ├── account_summary_with_clusters.csv # 32 accounts with segments + clusters
│   ├── predictions_profitability.csv      # Next-day profitability predictions
│   ├── predictions_volatility.csv         # PnL volatility bucket predictions
│   ├── model_profitability.pkl            # Trained Random Forest (profitability)
│   ├── model_volatility.pkl               # Trained Random Forest (volatility)
│   ├── chart1_performance_vs_sentiment.png
│   ├── chart2_behavior_shift.png
│   ├── chart3_volume_sentiment_overlay.png
│   ├── chart4_segment_sentiment_heatmap.png
│   ├── chart5_leverage_distribution.png
│   └── chart6_feature_importance.png
```

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Usage

### Step 1: Run Core Analysis

```bash
python analysis_clean.py
```

Generates all output files inside `/outputs/`:
- 6 PNG charts
- CSV summaries and predictions
- Pickled ML models

### Step 2: Launch Interactive Dashboard

```bash
streamlit run app.py
```

Dashboard opens at `http://localhost:8501/`

---

## Part A — Data Preparation

### Steps Performed
1. Loaded both datasets and documented shape, missing values, and duplicates
2. Converted `Timestamp IST` (format `%d-%m-%Y %H:%M`) to daily `YYYY-MM-DD` strings
3. Aligned datasets with an inner join on `date`
4. Dropped duplicate rows in both datasets before merging

### Key Metrics Engineered (per trader per day)

| Feature | Description |
|---|---|
| `total_pnl` | Sum of Closed PnL for the day |
| `win_rate` | Fraction of trades with PnL > 0 |
| `trade_count` | Number of trades executed |
| `avg_trade_size` | Mean Size USD per trade |
| `long_ratio` | BUY trades / (BUY + SELL trades) |
| `leverage_proxy` | Mean(Size USD / \|Start Position\|), capped at 99th percentile |
| `pnl_std` | Standard deviation of per-trade PnL |
| `pnl_volatility` | 5-day rolling average of pnl_std per account |

---

## Part B — Analysis

### B1. Performance vs Sentiment (Chart 1)

Traders were analysed across all 5 sentiment classes (Extreme Fear → Extreme Greed):

| Sentiment | Avg Daily PnL | Win Rate |
|---|---|---|
| Extreme Fear | ~$4,600 | ~33% |
| Fear | ~$5,300 | ~37% |
| Neutral | ~$3,400 | ~36% |
| Greed | ~$3,300 | ~35% |
| Extreme Greed | ~$5,100 | ~38% |

**Mann-Whitney U test** used to verify statistical significance of Fear vs Greed PnL differences.

**Finding:** Fear days do not uniformly produce worse PnL. Segment-level breakdown (Chart 4) tells the richer story — Whales earn *more* on Fear days ($10,861 avg) than on Greed days ($2,632 avg), a counter-intuitive result explained by large position sizes enabling contrarian profit-taking.

### B2. Behavior Shift by Sentiment (Chart 2)

| Metric | Fear (Median) | Greed (Median) |
|---|---|---|
| Leverage Proxy | 26.03 | 17.60 |
| Daily Trade Count | 31 | 27 |
| Avg Position Size (USD) | $1,711 | $2,004 |
| Long Ratio (BUY%) | 0.50 | 0.47 |

**Finding:** Traders use *higher* leverage on Fear days (contrary to naive expectation), trade more frequently, and have a marginally higher long bias. Position sizes are smaller, suggesting risk management is active but leverage usage is not reduced.

### B3. Trader Segments (Charts 4 & 5)

Traders segmented using data-driven quantile thresholds on account-level statistics:

| Segment | Criteria | Behaviour |
|---|---|---|
| **Whale** | Median daily trade size > 75th percentile | Large positions, highest Fear-day PnL |
| **Degen** | Median leverage > 75th percentile | High risk, large Fear/Greed PnL swing |
| **Grinder** | Low leverage (≤50th pct) AND high trade count (>50th pct) | Consistent, frequent, small trades |
| **Retail** | All others | Middle-ground performers |

Additionally, **K-means clustering (k=3)** was applied on 5 behavioural features (total trades, median size, median leverage, total PnL, win rate) to find data-driven archetypes independent of manual labels.

### B4. Key Insights (with evidence)

**Insight 1 — Whales profit most on Fear days**
Chart 4 shows Whales earn $10,861 avg PnL on Fear vs $2,632 on Greed — a 4× difference. Their large capital base allows them to absorb volatility and exit positions profitably during panic-driven price dislocations.

**Insight 2 — Leverage rises on Fear days across all segments**
Chart 2 median leverage: Fear = 26.03 vs Greed = 17.60. Chart 5 violin plots confirm this holds across Whale, Degen, Grinder, and Retail segments. Traders increase leverage during market stress rather than reducing it.

**Insight 3 — Grinders are the most consistent segment**
Chart 4 shows Grinders earn $7,071 on Fear and $5,755 on Greed — the smallest relative gap among segments. Their high-frequency, low-leverage model is resilient across sentiment regimes compared to Degens, who show the highest PnL delta.

**Insight 4 — Degens suffer most on average Fear days**
Chart 4 shows Degen avg PnL on Fear days = -$23.8 (negative), while on Greed days = $1,820. Their amplified leverage turns Fear-driven volatility into losses.

---

## Part C — Strategy Recommendations

### Strategy 1 — "Fear Brake" for Degen Traders
**Observation:** Degen win rate and PnL collapse on Fear days (avg PnL: -$23.8 vs $1,820 on Greed).

**Rule:** When the Fear/Greed index drops below 40 (Fear territory):
- Cap leverage at 3× (vs typical median)
- Reduce position size by 30%
- Avoid opening new long positions when long_ratio > 0.6

**Rationale:** Degens' high leverage amplifies losses during adverse sentiment. Cutting size preserves capital for the subsequent Greed-phase recovery.

### Strategy 2 — "Greed Filter" for Grinder Traders
**Observation:** Grinders trade more frequently on Fear days and maintain positive PnL ($7,071 avg). On Extreme Greed days, trending markets create false breakouts that punish their mean-reversion scalping style.

**Rule:**
- Maintain or slightly increase trade frequency during Fear days (high-quality mean-reversion setups emerge)
- On Extreme Greed days (index > 75), reduce frequency by ~20% to avoid false breakout traps

**Rationale:** Grinders' edge is spread/fee capture across many small trades. Their strategy degrades in fast-trending conditions but remains robust during volatile, mean-reverting Fear phases.

---

## Bonus — Predictive Models

### Profitability Prediction (Binary Classification)
- **Task:** Predict whether a trader will be profitable the *next* day
- **Method:** Time-based train/test split (80% train, 20% test) to prevent data leakage
- **Model:** Random Forest (200 trees, max_depth=6, class_weight=balanced)
- **Result:** ROC-AUC = 0.606
- **Top features:** `pnl_std`, `trade_count`, `avg_trade_size`, `long_ratio`

### Volatility Bucket Prediction (Multi-class)
- **Task:** Predict next-day PnL volatility bucket (Low / Medium / High)
- **Model:** Random Forest (same architecture, balanced class weights)
- **Features:** Same set + rolling 5-day pnl_volatility

### K-Means Clustering
- 3 clusters fitted on scaled behavioural features
- Cluster_A: Majority cluster — moderate traders
- Cluster_B: High-volume, high-PnL traders (overlaps Whale/Grinder)
- Cluster_C: Niche high-leverage Whales

---

## Evaluation Criteria Addressed

| Criterion | How Addressed |
|---|---|
| Data cleaning + merge correctness | Documented in Section 1–2 of analysis output; inner join on date, duplicate removal, 99th-pct leverage cap |
| Strength of reasoning | Mann-Whitney U tests; segment-level breakdowns; time-aware ML split |
| Quality of insights | 4 specific, data-backed insights; Whale counter-intuitive finding highlighted |
| Clarity of communication | This README + inline section headers in analysis_clean.py |
| Reproducibility | Single script entry point; all outputs deterministic (random_state=42) |

---

## Technical Stack

| Layer | Libraries |
|---|---|
| Data | pandas, numpy |
| Statistics | scipy.stats (Mann-Whitney U) |
| ML | scikit-learn (Random Forest, K-means, StandardScaler) |
| Visualisation | matplotlib, seaborn |
| Dashboard | streamlit |