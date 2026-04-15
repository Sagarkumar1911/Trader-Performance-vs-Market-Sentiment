# Trader Performance vs Market Sentiment Analysis

A comprehensive analysis framework for understanding how traders respond to market sentiment and predicting trader profitability.

## Features

### 1. **Core Analysis**
- Data loading and audit (data quality checks)
- Timestamp conversion and merge with sentiment data
- Feature engineering (PnL, win rate, leverage, trade patterns)
- Trader segmentation (Whale, Degen, Grinder, Retail)
- Performance vs sentiment analysis
- Behavior shift analysis

### 2. **Machine Learning Models**
- **Profitability Prediction**: Binary classification (Random Forest) to predict if a trader will be profitable tomorrow
- **Volatility Prediction**: Multi-class classification to predict PnL volatility bucket (Low/Medium/High)
- **K-means Clustering**: 3-cluster behavioral archetype segmentation alongside traditional segments

### 3. **Visualizations**
- 6 automated charts covering:
  - Performance by sentiment
  - Behavior shifts (Fear vs Greed)
  - Trade volume sentiment overlay
  - Segment x Sentiment heatmaps
  - Leverage distributions
  - Feature importance

### 4. **Streamlit Dashboard**
- Interactive web-based exploration of results
- Real-time filtering by trader account
- Predictive model performance review
- Cluster composition analysis

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Step 1: Run the Core Analysis

```bash
python analysis_clean.py
```

This generates:
- 6 PNG charts in `/outputs/`
- CSV files with predictions and summaries
- Pickle files with trained ML models

### Step 2: Launch the Interactive Dashboard

```bash
streamlit run dashboard.py
```

The dashboard will open at `http://localhost:8501/` with the following tabs:

- **Overview**: Executive summary and key metrics
- **Segmentation & Clusters**: Trader archetypes and K-means results
- **Sentiment Analysis**: Fear vs Greed impact on performance
- **Predictions**: Model outputs and accuracy metrics
- **Individual Traders**: Per-account deep dive analysis

## Output Files

### Data Files
- `daily_trader_summary.csv` - Daily performance metrics per trader
- `account_summary_with_clusters.csv` - Account-level stats + cluster assignments
- `predictions_profitability.csv` - Next-day profitability predictions
- `predictions_volatility.csv` - PnL volatility bucket predictions

### Models
- `model_profitability.pkl` - Trained Random Forest for profitability
- `model_volatility.pkl` - Trained Random Forest for volatility

### Charts
- `chart1_performance_vs_sentiment.png` - Avg PnL & win rate by sentiment
- `chart2_behavior_shift.png` - Behavioral metrics (Fear vs Greed)
- `chart3_volume_sentiment_overlay.png` - Trading volume with sentiment background
- `chart4_segment_sentiment_heatmap.png` - Segment x Sentiment performance
- `chart5_leverage_distribution.png` - Leverage by segment and sentiment
- `chart6_feature_importance.png` - ML model feature importance

## Key Insights

### Insight 1: Sentiment-Based Performance
- Traders earn differently based on market sentiment regimes
- Fear days tend to show different PnL distribution than Greed days
- Statistical significance tested with Mann-Whitney U test

### Insight 2: Behavioral Adaptation
- Leverage usage changes significantly between Fear and Greed periods
- Traders exhibit different trade frequencies in different sentiment regimes
- Position sizing reflects sentiment-driven risk appetite

### Insight 3: Segment-Specific Strategies
- **Grinders**: Most consistent across regimes (suitable for mean-reversion in Fear)
- **Degens**: Highest PnL delta between regimes (leverage amplifies sentiment effects)
- **Whales**: Stable due to large position sizes
- **Retail**: Middle performers with moderate consistency

## Predictive Models

### Profitability Bucket
- **Task**: Binary classification (Profitable/Not Profitable next day)
- **Features**: Sentiment score, leverage, trade count, position size, PnL std
- **Architecture**: Random Forest (200 trees, max_depth=6)
- **Metric**: ROC-AUC

### Volatility Bucket
- **Task**: Multi-class classification (Low/Medium/High volatility)
- **Features**: Same as profitability, plus rolling volatility
- **Architecture**: Random Forest with balanced class weights
- **Metric**: Accuracy

## Data Requirements

The analysis expects two CSV files in the same directory:

1. **historical_data.csv** - Trade-level data with columns:
   - Account, Coin, Execution Price, Size Tokens, Size USD
   - Side (BUY/SELL), Timestamp IST, Start Position
   - Direction, Closed PnL, Transaction Hash
   - Order ID, Crossed, Fee, Trade ID, Timestamp

2. **fear_greed_index.csv** - Daily sentiment data with columns:
   - timestamp, value, classification, date

## Performance Notes

- Full analysis typically takes 2-3 minutes on standard hardware
- Dashboard loads instantly from cached data
- 211K+ trades, 32 traders, 479 trading days analyzed
- K-means clustering on 5 behavioral features scales efficiently

## Customization

Edit the following parameters in `analysis_clean.py`:

```python
OUTPUTS_DIR = "outputs"              # Output folder
RandomForest n_estimators = 200     # Number of trees
KMeans n_clusters = 3               # Number of clusters
Test split = 0.80                   # Train/test ratio
Leverage cap = 0.99                 # Percentile cap for outliers
```

## Next Steps

1. Run `python analysis_clean.py` to generate analysis
2. Launch `streamlit run dashboard.py` for interactive exploration
3. Review CSV predictions for integration with trading systems
4. Load pickled models for real-time next-day predictions

## Technical Stack

- **Data**: pandas, numpy
- **ML**: scikit-learn (Random Forest, K-means)
- **Visualization**: matplotlib, seaborn
- **Dashboard**: streamlit
- **Statistics**: scipy.stats
