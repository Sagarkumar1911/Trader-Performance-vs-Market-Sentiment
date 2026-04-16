# Trader Performance vs Market Sentiment


A complete end-to-end analysis of how Bitcoin market sentiment (Fear/Greed) influences trader behaviour and profitability on Hyperliquid. Includes data cleaning, feature engineering, segment analysis, ML models, and an interactive Streamlit dashboard.

---

## Repository Structure

```
├── analysis_clean.py                         
├── app.py                                  
├── requirements.txt                          
├── historical_data.csv                      
├── fear_greed_index.csv                      
├── WRITEUP.md                               
└── outputs/
    ├── daily_trader_summary.csv              # 2,342 daily trader-day records
    ├── account_summary_with_clusters.csv     # 32 accounts — segments + K-means clusters
    ├── predictions_profitability.csv         # Next-day profitability predictions (test set)
    ├── predictions_volatility.csv            # PnL volatility bucket predictions (test set)
    ├── model_profitability.pkl               # Trained Random Forest — profitability
    ├── model_volatility.pkl                  # Trained Random Forest — volatility
    ├── chart1_performance_vs_sentiment.png
    ├── chart2_behavior_shift.png
    ├── chart3_volume_sentiment_overlay.png
    ├── chart4_segment_sentiment_heatmap.png
    ├── chart5_leverage_distribution.png
    └── chart6_feature_importance.png
```

---

## Setup

**Requirements:** Python 3.9+

```bash
pip install -r requirements.txt
```

---

## How to Run

### Step 1 — Core Analysis

```bash
python analysis_clean.py
```

Runs end-to-end and saves everything to `/outputs/`:
- Prints full data audit (rows, columns, missing values, duplicates)
- Engineers daily features per trader
- Segments traders into Whale / Degen / Grinder / Retail
- Runs K-means clustering (k=3)
- Produces 6 charts
- Trains and saves both ML models
- Saves all CSVs and pickled models

### Step 2 — Interactive Dashboard

```bash
streamlit run app.py
```

Opens at **`http://localhost:8501/`**

>

---

## Datasets

| File | Rows | Key Columns |
|---|---|---|
| `historical_data.csv` | 211,000+ trades | Account, Coin, Size USD, Side, Timestamp IST, Start Position, Closed PnL, Fee |
| `fear_greed_index.csv` | 479 days | date, value, classification |

---

## Key Numbers

| Metric | Value |
|---|---|
| Total trades analyzed | 211,000+ |
| Unique traders | 32 |
| Trading days (after merge) | 479 |
| Daily trader-day records | 2,342 |
| Top earner (lifetime PnL) | $2,143,382 |
| Highest account win rate | 71% |
| ML test set size | 455 records |
| Profitability model ROC-AUC | 0.606 |

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

Traders analysed across all 5 sentiment classes (Extreme Fear → Extreme Greed):

| Sentiment | Avg Daily PnL | Win Rate |
|---|---|---|
| Extreme Fear | ~$4,600 | ~33% |
| Fear | ~$5,300 | ~37% |
| Neutral | ~$3,400 | ~36% |
| Greed | ~$3,300 | ~35% |
| Extreme Greed | ~$5,100 | ~38% |

*

### B2. Behavior Shift by Sentiment (Chart 2)

| Metric | Fear (Median) | Greed (Median) |
|---|---|---|
| Leverage Proxy | 26.03 | 17.60 |
| Daily Trade Count | 31 | 27 |
| Avg Position Size (USD) | $1,711 | $2,004 |
| Long Ratio (BUY%) | 0.50 | 0.47 |

**Finding:** Traders use *higher* leverage on Fear days (contrary to naive expectation), trade more frequently, and maintain a marginally higher long bias — while reducing position sizes. Risk management is active but leverage is not reduced during stress.

### B3. Trader Segments (Charts 4 & 5)

Segments assigned using quantile-based thresholds on account-level lifetime statistics:

| Segment | Criteria | Behaviour |
|---|---|---|
| **Whale** | Median daily trade size > 75th percentile | Large positions, highest Fear-day PnL |
| **Degen** | Median leverage > 75th percentile | High risk, largest Fear/Greed PnL swing |
| **Grinder** | Leverage ≤ 50th pct AND total trades > 50th pct | Consistent, frequent, small trades |
| **Retail** | All others | Middle-ground performers |

**K-means clustering (k=3)** also applied on 5 scaled behavioural features to find data-driven archetypes independent of manual labels:

| Cluster | Profile |
|---|---|
| Cluster_A | Majority group — moderate volume, mixed leverage |
| Cluster_B | High-volume, high-PnL (overlaps Whale / Grinder) |
| Cluster_C | Niche high-leverage Whales |

### B4. Key Insights (with evidence)

**Insight 1 — Whales profit most on Fear days**
Chart 4 shows Whales earn $10,861 avg PnL on Fear vs $2,632 on Greed — a 4× difference. Their large capital base allows contrarian positioning during panic-driven price dislocations.

**Insight 2 — Leverage rises on Fear days across all segments**
Chart 2: median leverage Fear = 26.03 vs Greed = 17.60. Chart 5 violin plots confirm this holds across all four segments. Traders increase leverage under stress rather than reducing it.

**Insight 3 — Grinders are the most regime-resilient segment**
Chart 4: Grinders earn $7,071 on Fear and $5,755 on Greed — the smallest relative swing among all segments. Their high-frequency, low-leverage model is robust across both sentiment regimes.

**Insight 4 — Degens are punished hardest on Fear days**
Chart 4: Degen avg PnL on Fear = **−$23.8** (negative) vs $1,820 on Greed. High leverage amplifies losses when sentiment turns negative.

---

## Part C — Strategy Recommendations

### Strategy 1 — "Fear Brake" for Degen Traders

**Observation:** Degen PnL collapses on Fear days (avg −$23.8 vs +$1,820 on Greed).

**Rule:** When Fear/Greed index < 40:
- Cap leverage at 3× (vs their typical high median)
- Reduce position size by 30%
- Avoid opening new long positions when long_ratio > 0.6

**Rationale:** Degens' high leverage amplifies losses during adverse sentiment. Cutting size during Fear preserves capital for the Greed-phase recovery where their strategy excels.

### Strategy 2 — "Greed Filter" for Grinder Traders

**Observation:** Grinders earn $7,071 avg on Fear days and trade more frequently. On Extreme Greed days their mean-reversion style is punished by trending markets.

**Rule:**
- Maintain or slightly increase trade frequency during Fear days — mean-reversion setups are abundant
- On Extreme Greed (index > 75), reduce trade frequency by ~20% to avoid false breakout traps

**Rationale:** Grinders' edge is spread/fee capture across many small trades. Their strategy degrades in fast-trending Extreme Greed conditions but remains strong during volatile, mean-reverting Fear phases.

---



### Profitability Prediction (Binary Classification)
- **Task:** Predict whether a trader will be profitable the next day
- **Split:** Time-based 80% train / 20% test — no data leakage
- **Model:** Random Forest (200 trees, max_depth=6, class_weight=balanced)
- **Features:** `fg_score`, `sentiment_encoded`, `leverage_proxy`, `trade_count`, `long_ratio`, `avg_trade_size`, `pnl_std`
- **Result:** ROC-AUC = 0.606
- **Top features by importance:** `pnl_std`, `trade_count`, `avg_trade_size`, `long_ratio`

### Volatility Bucket Prediction (Multi-class)
- **Task:** Predict next-day PnL volatility bucket (Low / Medium / High)
- **Model:** Random Forest (same architecture, balanced class weights)
- **Features:** Same set + rolling 5-day `pnl_volatility`

---

## Output Files

| File | Description |
|---|---|
| `daily_trader_summary.csv` | One row per trader per day — all features + segment + cluster |
| `account_summary_with_clusters.csv` | One row per account — lifetime stats, segment, K-means cluster |
| `predictions_profitability.csv` | Actual vs Predicted next-day profitability + probability score |
| `predictions_volatility.csv` | Actual vs Predicted PnL volatility bucket (Low / Medium / High) |

---

## Charts

| Chart | What It Shows |
|---|---|
| `chart1_performance_vs_sentiment` | Avg daily PnL and win rate across all 5 sentiment classes |
| `chart2_behavior_shift` | Boxplots — leverage, trade count, position size, long ratio: Fear vs Greed |
| `chart3_volume_sentiment_overlay` | Daily trade volume over time with Fear/Greed background shading |
| `chart4_segment_sentiment_heatmap` | Heatmap + bar chart — PnL and win rate by segment × sentiment |
| `chart5_leverage_distribution` | Violin plots — leverage distribution by segment and sentiment |
| `chart6_feature_importance` | Feature importance for profitability Random Forest (ROC-AUC = 0.606) |

---

## Technical Stack

| Layer | Library |
|---|---|
| Data wrangling | pandas, numpy |
| Statistics | scipy.stats (Mann-Whitney U) |
| Machine learning | scikit-learn (RandomForest, KMeans, StandardScaler) |
| Visualisation | matplotlib, seaborn |
| Dashboard | streamlit |
