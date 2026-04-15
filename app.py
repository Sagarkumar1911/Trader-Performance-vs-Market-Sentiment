import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from pathlib import Path
from sklearn.metrics import confusion_matrix, roc_curve, auc

st.set_page_config(page_title="Trader Sentiment Performance", layout="wide")

OUTPUT_DIR = Path(__file__).parent / "outputs"

BINARY_PALETTE = {"Fear": "#d62728", "Greed": "#2ca02c"}

@st.cache_data
def load_data():
    daily_df = pd.read_csv(OUTPUT_DIR / "daily_trader_summary.csv")
    pred_prof = pd.read_csv(OUTPUT_DIR / "predictions_profitability.csv")
    pred_vol = pd.read_csv(OUTPUT_DIR / "predictions_volatility.csv")
    return daily_df, pred_prof, pred_vol

@st.cache_resource
def load_models():
    try:
        model_prof = pickle.load(open(OUTPUT_DIR / "model_profitability.pkl", "rb"))
        model_vol = pickle.load(open(OUTPUT_DIR / "model_volatility.pkl", "rb"))
        return model_prof, model_vol
    except:
        return None, None

def main():
    st.title("Trader Performance by Market Sentiment")
    st.markdown("Analyze how trader segments perform under different market sentiments")
    st.markdown("---")
    
    daily_df, pred_prof, pred_vol = load_data()
    model_prof, model_vol = load_models()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        sentiment_filter = st.selectbox(
            "Select Market Sentiment",
            ["Fear", "Greed"],
            key="sentiment_select"
        )
    
    with col2:
        st.metric("Trading Days", daily_df["date"].nunique())
    
    with col3:
        st.metric("Total Accounts", daily_df["Account"].nunique())
    
    st.markdown("---")
    
    tab1, tab2, tab3 = st.tabs(["Sentiment Analysis", "Account Search", "Predictive Models"])
    
    with tab1:
        st.subheader(f"Performance Analysis: {sentiment_filter} Days")
        
        filtered_df = daily_df[daily_df["sentiment_binary"] == sentiment_filter].copy()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Avg Daily PnL",
                f"${filtered_df['total_pnl'].mean():,.0f}"
            )
        with col2:
            st.metric(
                "Avg Win Rate",
                f"{filtered_df['win_rate'].mean():.1%}"
            )
        with col3:
            st.metric(
                "Avg Trade Count",
                f"{filtered_df['trade_count'].mean():.0f}"
            )
        with col4:
            st.metric(
                "Avg Leverage",
                f"{filtered_df['leverage_proxy'].mean():.2f}x"
            )
        
        st.markdown("---")
        st.subheader(f"Top-Performing Segments ({sentiment_filter} Days)")
        
        segment_perf = filtered_df.groupby("segment").agg({
            "total_pnl": ["mean", "median", "std"],
            "win_rate": "mean",
            "trade_count": ["mean", "count"],
            "leverage_proxy": "mean"
        }).round(2)
        
        segment_perf.columns = [
            "Avg PnL", "Median PnL", "Std Dev PnL",
            "Avg Win Rate", "Avg Trades", "Days Count", "Avg Leverage"
        ]
        segment_perf = segment_perf.sort_values("Avg PnL", ascending=False)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("### Segment Performance Table")
            st.dataframe(segment_perf.style.highlight_max(subset=["Avg PnL"], color="lightgreen"), use_container_width=True)
        
        with col2:
            st.write("### Segment Rankings")
            fig, ax = plt.subplots(figsize=(8, 5))
            colors = [BINARY_PALETTE[sentiment_filter]] * len(segment_perf)
            segment_perf["Avg PnL"].plot(kind="barh", ax=ax, color=colors, edgecolor="white", linewidth=1.5)
            ax.set_xlabel("Average Daily PnL (USD)")
            ax.set_title(f"Avg PnL by Segment ({sentiment_filter} Days)")
            ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
            st.pyplot(fig)
        
        st.markdown("---")
        st.subheader(f"Win Rate Comparison ({sentiment_filter} Days)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(8, 5))
            win_rate_data = (segment_perf["Avg Win Rate"] * 100).sort_values(ascending=False)
            colors = [BINARY_PALETTE[sentiment_filter]] * len(win_rate_data)
            ax.bar(range(len(win_rate_data)), win_rate_data.values, color=colors, edgecolor="white", linewidth=1.5)
            ax.axhline(50, color="red", linestyle="--", linewidth=1.5, label="50% baseline")
            ax.set_xticks(range(len(win_rate_data)))
            ax.set_xticklabels(win_rate_data.index, rotation=0)
            ax.set_ylabel("Win Rate (%)")
            ax.set_title(f"Average Win Rate by Segment ({sentiment_filter} Days)")
            ax.legend()
            st.pyplot(fig)
        
        with col2:
            fig, ax = plt.subplots(figsize=(8, 5))
            trade_data = segment_perf["Avg Trades"].sort_values(ascending=False)
            colors = [BINARY_PALETTE[sentiment_filter]] * len(trade_data)
            ax.bar(range(len(trade_data)), trade_data.values, color=colors, edgecolor="white", linewidth=1.5)
            ax.set_xticks(range(len(trade_data)))
            ax.set_xticklabels(trade_data.index, rotation=0)
            ax.set_ylabel("Average Daily Trade Count")
            ax.set_title(f"Trade Frequency by Segment ({sentiment_filter} Days)")
            st.pyplot(fig)
        
        st.markdown("---")
        st.subheader(f"Detailed Breakdown ({sentiment_filter} Days)")
        
        breakdown_table = filtered_df.groupby("segment").agg({
            "Account": "nunique",
            "total_pnl": ["mean", "min", "max"],
            "win_rate": ["min", "max"],
            "trade_count": "sum"
        }).round(2)
        
        breakdown_table.columns = [
            "Unique Traders", "Avg PnL", "Min PnL", "Max PnL",
            "Min Win Rate", "Max Win Rate", "Total Trades"
        ]
        
        st.dataframe(breakdown_table, use_container_width=True)
    
    with tab2:
        st.subheader("Search for Specific Trading Account")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            search_account = st.selectbox(
                "Select Account to View",
                sorted(daily_df["Account"].unique()),
                key="account_search"
            )
        
        with col2:
            st.write("")
        
        if search_account:
            account_data = daily_df[daily_df["Account"] == search_account].sort_values("date")
            
            if len(account_data) > 0:
                col1, col2, col3, col4, col5 = st.columns(5)
                
                with col1:
                    st.metric("Total PnL", f"${account_data['total_pnl'].sum():,.0f}")
                with col2:
                    st.metric("Avg Daily PnL", f"${account_data['total_pnl'].mean():,.0f}")
                with col3:
                    st.metric("Win Rate", f"{account_data['win_rate'].mean():.1%}")
                with col4:
                    st.metric("Total Trades", f"{account_data['trade_count'].sum():,.0f}")
                with col5:
                    st.metric("Trading Days", len(account_data))
                
                st.markdown("---")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("### Account Profile")
                    account_segment = account_data["segment"].iloc[0]
                    account_cluster = account_data.get("cluster_name", pd.Series(["N/A"])).iloc[0]
                    
                    profile_info = {
                        "Segment": account_segment,
                        "Cluster": account_cluster,
                        "Avg Trade Size": f"${account_data['avg_trade_size'].mean():,.0f}",
                        "Avg Leverage": f"{account_data['leverage_proxy'].mean():.2f}x",
                        "Long Ratio": f"{account_data['long_ratio'].mean():.1%}",
                        "Avg PnL Std Dev": f"${account_data['pnl_std'].mean():,.0f}"
                    }
                    
                    for key, value in profile_info.items():
                        st.write(f"**{key}:** {value}")
                
                with col2:
                    st.write("### Performance by Sentiment")
                    sentiment_breakdown = account_data.groupby("sentiment_binary").agg({
                        "total_pnl": ["mean", "count"],
                        "win_rate": "mean"
                    }).round(3)
                    sentiment_breakdown.columns = ["Avg PnL", "Days", "Win Rate"]
                    st.dataframe(sentiment_breakdown, use_container_width=True)
                
                st.markdown("---")
                st.subheader("Daily Performance History")
                
                display_cols = [
                    "date", "sentiment", "total_pnl", "trade_count",
                    "win_rate", "leverage_proxy", "long_ratio", "avg_trade_size"
                ]
                
                history_display = account_data[display_cols].copy()
                history_display.columns = [
                    "Date", "Sentiment", "Daily PnL", "Trades",
                    "Win Rate", "Leverage", "Long Ratio", "Avg Trade Size"
                ]
                
                history_display["Daily PnL"] = history_display["Daily PnL"].apply(lambda x: f"${x:,.0f}")
                history_display["Win Rate"] = history_display["Win Rate"].apply(lambda x: f"{x:.1%}")
                history_display["Leverage"] = history_display["Leverage"].apply(lambda x: f"{x:.2f}x")
                history_display["Long Ratio"] = history_display["Long Ratio"].apply(lambda x: f"{x:.1%}")
                history_display["Avg Trade Size"] = history_display["Avg Trade Size"].apply(lambda x: f"${x:,.0f}")
                
                st.dataframe(history_display.sort_values("Date", ascending=False), use_container_width=True, height=400)
                
                st.markdown("---")
                st.subheader("Cumulative PnL Over Time")
                
                fig, ax = plt.subplots(figsize=(14, 5))
                account_data_sorted = account_data.sort_values("date")
                account_data_sorted["cumulative_pnl"] = account_data_sorted["total_pnl"].cumsum()
                
                ax.plot(
                    pd.to_datetime(account_data_sorted["date"]),
                    account_data_sorted["cumulative_pnl"],
                    linewidth=2.5,
                    marker="o",
                    markersize=4,
                    color="#1f77b4"
                )
                ax.axhline(0, color="red", linestyle="--", linewidth=1, alpha=0.5)
                ax.fill_between(
                    pd.to_datetime(account_data_sorted["date"]),
                    account_data_sorted["cumulative_pnl"],
                    0,
                    alpha=0.2,
                    color="#1f77b4"
                )
                ax.set_xlabel("Date")
                ax.set_ylabel("Cumulative PnL (USD)")
                ax.set_title(f"Cumulative PnL Over Time - {search_account}")
                ax.grid(True, alpha=0.3)
                
                st.pyplot(fig)
                
                st.markdown("---")
                st.subheader("Sentiment Impact Analysis")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fig, ax = plt.subplots(figsize=(8, 5))
                    sentiment_pnl = account_data.groupby("sentiment_binary")["total_pnl"].mean()
                    colors = [BINARY_PALETTE.get(s, "#999999") for s in sentiment_pnl.index]
                    ax.bar(sentiment_pnl.index, sentiment_pnl.values, color=colors, edgecolor="white", linewidth=1.5)
                    ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
                    ax.set_ylabel("Average Daily PnL (USD)")
                    ax.set_title(f"Avg PnL by Sentiment - {search_account}")
                    st.pyplot(fig)
                
                with col2:
                    fig, ax = plt.subplots(figsize=(8, 5))
                    sentiment_wr = (account_data.groupby("sentiment_binary")["win_rate"].mean() * 100)
                    colors = [BINARY_PALETTE.get(s, "#999999") for s in sentiment_wr.index]
                    ax.bar(sentiment_wr.index, sentiment_wr.values, color=colors, edgecolor="white", linewidth=1.5)
                    ax.axhline(50, color="red", linestyle="--", linewidth=1, label="50% baseline")
                    ax.set_ylabel("Win Rate (%)")
                    ax.set_title(f"Win Rate by Sentiment - {search_account}")
                    ax.legend()
                    st.pyplot(fig)
    
    # ---- TAB 3: PREDICTIVE MODELS ----
    with tab3:
        st.header("Predictive Models")
        
        if model_prof is None or model_vol is None:
            st.error("Models not loaded. Please run analysis_clean.py first.")
            return
        
        model_tab1, model_tab2 = st.tabs(["Profitability Prediction", "Volatility Prediction"])
        
        # ---- Profitability Model ----
        with model_tab1:
            st.subheader("Next-Day Profitability Prediction")
            st.markdown("Binary classification: Will the trader be profitable tomorrow?")
            
            col1, col2, col3, col4 = st.columns(4)
            
            if pred_prof is not None:
                accuracy = (pred_prof["Actual"] == pred_prof["Predicted"]).mean()
                profitable_pred = (pred_prof["Predicted"] == 1).sum()
                profitable_actual = (pred_prof["Actual"] == 1).sum()
                roc_auc = 0.6058
                
                with col1:
                    st.metric("Accuracy", f"{accuracy:.1%}")
                with col2:
                    st.metric("ROC-AUC", f"{roc_auc:.4f}")
                with col3:
                    st.metric("Predicted Profitable", f"{profitable_pred}/{len(pred_prof)}")
                with col4:
                    st.metric("Actual Profitable", f"{profitable_actual}/{len(pred_prof)}")
                
                # Confusion Matrix
                st.markdown("---")
                st.subheader("Confusion Matrix")
                conf_matrix = confusion_matrix(pred_prof["Actual"], pred_prof["Predicted"])
                
                fig, ax = plt.subplots(figsize=(6, 5))
                sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                           xticklabels=['Not Profitable', 'Profitable'],
                           yticklabels=['Not Profitable', 'Profitable'],
                           ax=ax, cbar=False)
                ax.set_ylabel('Actual')
                ax.set_xlabel('Predicted')
                ax.set_title('Profitability Confusion Matrix')
                st.pyplot(fig)
                
                # Feature Importance
                st.markdown("---")
                st.subheader("Feature Importance")
                feature_names = ['fg_score', 'sentiment_encoded', 'leverage_proxy',
                 'trade_count', 'long_ratio', 'avg_trade_size', 'pnl_std']
                importances = model_prof.feature_importances_
                
                feature_importance_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': importances
                }).sort_values('Importance', ascending=True)
                
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.barh(feature_importance_df['Feature'], feature_importance_df['Importance'], color='steelblue')
                ax.set_xlabel('Importance')
                ax.set_title('Feature Importance - Profitability Model')
                st.pyplot(fig)
                
                # Sample Predictions
                st.markdown("---")
                st.subheader("Sample Predictions")
                pred_sample = pred_prof.sample(min(20, len(pred_prof))).sort_values('Date')
                pred_sample['Correct'] = pred_sample['Actual'] == pred_sample['Predicted']
                pred_sample['Actual_Label'] = pred_sample['Actual'].map({1: 'Profitable', 0: 'Not Profitable'})
                pred_sample['Predicted_Label'] = pred_sample['Predicted'].map({1: 'Profitable', 0: 'Not Profitable'})
                
                display_cols = ['Date', 'Account', 'Actual_Label', 'Predicted_Label', 'Probability', 'Correct']
                st.dataframe(pred_sample[display_cols].reset_index(drop=True), use_container_width=True)
        
        # ---- Volatility Model ----
        with model_tab2:
            st.subheader("Trade Volatility Bucket Prediction")
            st.markdown("Multi-class classification: Predict trade size volatility (Low/Medium/High)")
            
            col1, col2 = st.columns(2)
            
            if pred_vol is not None:
                accuracy_vol = (pred_vol["Actual"] == pred_vol["Predicted"]).mean()
                total_vol = len(pred_vol)
                
                with col1:
                    st.metric("Accuracy", f"{accuracy_vol:.1%}")
                with col2:
                    st.metric("Test Samples", total_vol)
                
                # Class Distribution
                st.markdown("---")
                st.subheader("Class Distribution")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    actual_counts = pred_vol["Actual"].value_counts().sort_index()
                    fig, ax = plt.subplots(figsize=(6, 4))
                    actual_counts.plot(kind='bar', ax=ax, color='steelblue', alpha=0.7)
                    ax.set_title('Actual Volatility Distribution')
                    ax.set_xlabel('Bucket')
                    ax.set_ylabel('Count')
                    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
                    st.pyplot(fig)
                
                with col2:
                    pred_counts = pred_vol["Predicted"].value_counts().sort_index()
                    fig, ax = plt.subplots(figsize=(6, 4))
                    pred_counts.plot(kind='bar', ax=ax, color='coral', alpha=0.7)
                    ax.set_title('Predicted Volatility Distribution')
                    ax.set_xlabel('Bucket')
                    ax.set_ylabel('Count')
                    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
                    st.pyplot(fig)
                
                # Confusion Matrix
                st.markdown("---")
                st.subheader("Confusion Matrix")
                conf_matrix_vol = confusion_matrix(pred_vol["Actual"], pred_vol["Predicted"])
                bucket_labels = ['Low', 'Medium', 'High']
                
                fig, ax = plt.subplots(figsize=(7, 6))
                sns.heatmap(conf_matrix_vol, annot=True, fmt='d', cmap='RdYlGn',
                           xticklabels=bucket_labels, yticklabels=bucket_labels,
                           ax=ax, cbar=True)
                ax.set_ylabel('Actual')
                ax.set_xlabel('Predicted')
                ax.set_title('Volatility Confusion Matrix')
                st.pyplot(fig)
                
                # Sample Predictions
                st.markdown("---")
                st.subheader("Sample Predictions")
                pred_vol_sample = pred_vol.sample(min(20, len(pred_vol))).sort_values('Date')
                pred_vol_sample['Correct'] = pred_vol_sample['Actual'] == pred_vol_sample['Predicted']
                
                display_cols_vol = ['Date', 'Account', 'Actual', 'Predicted', 'Correct']
                st.dataframe(pred_vol_sample[display_cols_vol].reset_index(drop=True), use_container_width=True)

if __name__ == "__main__":
    main()
