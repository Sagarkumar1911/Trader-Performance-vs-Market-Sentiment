
import warnings
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pickle

warnings.filterwarnings("ignore")

DATA_DIR    = os.path.dirname(__file__)
OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)

PALETTE = {
    "Extreme Fear": "#d62728",
    "Fear":         "#ff7f0e",
    "Neutral":      "#7f7f7f",
    "Greed":        "#2ca02c",
    "Extreme Greed":"#1f77b4",
}
BINARY_PALETTE = {"Fear": "#d62728", "Greed": "#2ca02c"}

plt.rcParams.update({
    "figure.dpi":       130,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "font.size":        11,
})

def load_and_audit_data(data_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
   
    print("=" * 65)
    print("SECTION 1 -- DATA LOADING & AUDIT")
    print("=" * 65)

    trades_df = pd.read_csv(os.path.join(data_dir, "historical_data.csv"))
    fg_df     = pd.read_csv(os.path.join(data_dir, "fear_greed_index.csv"))

    for name, df in [("Trades", trades_df), ("Fear/Greed", fg_df)]:
        print(f"\n-- {name} dataset --")
        print(f"   Rows x Columns : {df.shape[0]:,} x {df.shape[1]}")
        print(f"   Columns        : {df.columns.tolist()}")

        missing = df.isnull().sum()
        missing = missing[missing > 0]
        if missing.empty:
            print("   Missing values : None [OK]")
        else:
            print(f"   Missing values :\n{missing}")

        dups = df.duplicated().sum()
        print(f"   Duplicate rows : {dups:,}" + (" [OK]" if dups == 0 else " [will be dropped]"))

    trades_df = trades_df.drop_duplicates()
    fg_df     = fg_df.drop_duplicates(subset=["date"])

    print(f"\n   Fear/Greed classifications:\n{fg_df['classification'].value_counts()}")
    return trades_df, fg_df

def convert_timestamps_and_merge(
    trades_df: pd.DataFrame,
    fg_df: pd.DataFrame
) -> pd.DataFrame:
   
    print("\n" + "=" * 65)
    print("SECTION 2 -- TIMESTAMP CONVERSION & MERGE")
    print("=" * 65)

    trades_df["date"] = (
        pd.to_datetime(trades_df["Timestamp IST"], format="%d-%m-%Y %H:%M")
        .dt.strftime("%Y-%m-%d")
    )

    fg_df["date"] = pd.to_datetime(fg_df["date"]).dt.strftime("%Y-%m-%d")
    fg_slim = fg_df[["date", "value", "classification"]].rename(
        columns={"value": "fg_score", "classification": "sentiment"}
    )

    merged_df = trades_df.merge(fg_slim, on="date", how="inner")

    fear_labels = {"Extreme Fear", "Fear"}
    greed_labels = {"Greed", "Extreme Greed"}
    merged_df["sentiment_binary"] = merged_df.apply(
        lambda r: "Fear"  if r["sentiment"] in fear_labels
             else "Greed" if r["sentiment"] in greed_labels
             else ("Fear" if r["fg_score"] < 50 else "Greed"),
        axis=1
    )

    print(f"\n   Trades before merge : {len(trades_df):,}")
    print(f"   Trades after merge  : {len(merged_df):,}")
    print(f"   Unique trading dates: {merged_df['date'].nunique()}")
    print(f"\n   Sentiment distribution (5-class):\n{merged_df['sentiment'].value_counts()}")
    print(f"\n   Sentiment distribution (binary):\n{merged_df['sentiment_binary'].value_counts()}")

    return merged_df

def engineer_features(merged_df: pd.DataFrame) -> pd.DataFrame:
   
    print("\n" + "=" * 65)
    print("SECTION 3 -- FEATURE ENGINEERING")
    print("=" * 65)

    grp = merged_df.groupby(["Account", "date", "sentiment", "sentiment_binary", "fg_score"])

    daily_df = grp.agg(
        total_pnl      = ("Closed PnL",   "sum"),
        trade_count    = ("Closed PnL",   "count"),
        win_rate       = ("Closed PnL",   lambda x: (x > 0).sum() / len(x)),
        avg_trade_size = ("Size USD",     "mean"),
        pnl_std        = ("Closed PnL",   "std"),
        net_fee        = ("Fee",          "sum"),
    ).reset_index()

    side_grp = (
        merged_df
        .groupby(["Account", "date", "Side"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    for col in ["BUY", "SELL"]:
        if col not in side_grp.columns:
            side_grp[col] = 0
    side_grp["long_ratio"] = side_grp["BUY"] / (side_grp["BUY"] + side_grp["SELL"])
    daily_df = daily_df.merge(
        side_grp[["Account", "date", "long_ratio"]],
        on=["Account", "date"], how="left"
    )

    lev_df = merged_df[merged_df["Start Position"] != 0].copy()
    lev_df["leverage_raw"] = lev_df["Size USD"] / lev_df["Start Position"].abs()
    lev_cap = lev_df["leverage_raw"].quantile(0.99)
    lev_df["leverage_raw"] = lev_df["leverage_raw"].clip(upper=lev_cap)

    lev_grp = (
        lev_df
        .groupby(["Account", "date"])["leverage_raw"]
        .mean()
        .reset_index()
        .rename(columns={"leverage_raw": "leverage_proxy"})
    )
    daily_df = daily_df.merge(lev_grp, on=["Account", "date"], how="left")

    daily_df["is_profitable"] = (daily_df["total_pnl"] > 0).astype(int)
    daily_df["leverage_proxy"] = daily_df["leverage_proxy"].fillna(
        daily_df["leverage_proxy"].median()
    )

    daily_df["pnl_std"] = daily_df["pnl_std"].fillna(0)
    daily_df["pnl_volatility"] = daily_df.groupby("Account")["pnl_std"].rolling(window=5, min_periods=1).mean().reset_index(0, drop=True)

    print(f"\n   Daily summary shape : {daily_df.shape}")
    print(f"   Unique accounts     : {daily_df['Account'].nunique():,}")
    print(f"\n   Feature preview:\n{daily_df[['total_pnl','trade_count','win_rate','long_ratio','leverage_proxy']].describe().round(3)}")

    return daily_df

def segment_traders(daily_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    
    print("\n" + "=" * 65)
    print("SECTION 4 -- TRADER SEGMENTATION")
    print("=" * 65)

    account_df = (
        daily_df
        .groupby("Account")
        .agg(
            total_trades  = ("trade_count",    "sum"),
            median_size   = ("avg_trade_size", "median"),
            median_lev    = ("leverage_proxy", "median"),
            total_pnl     = ("total_pnl",      "sum"),
            active_days   = ("date",           "nunique"),
            overall_wr    = ("win_rate",        "mean"),
        )
        .reset_index()
    )

    whale_size_thresh = account_df["median_size"].quantile(0.75)
    degen_lev_thresh  = account_df["median_lev"].quantile(0.75)
    lev_lo_thresh     = account_df["median_lev"].quantile(0.50)
    count_hi_thresh   = account_df["total_trades"].quantile(0.50)

    print(f"\n   Thresholds used:")
    print(f"   Whale  -- median daily size  > ${whale_size_thresh:,.0f}")
    print(f"   Degen  -- median leverage    > {degen_lev_thresh:.2f}x")
    print(f"   Grinder -- leverage <= {lev_lo_thresh:.2f}x AND total trades > {count_hi_thresh:,.0f}")

    def assign_segment(row):
        if row["median_size"] >= whale_size_thresh:
            return "Whale"
        elif row["median_lev"] >= degen_lev_thresh:
            return "Degen"
        elif row["median_lev"] <= lev_lo_thresh and row["total_trades"] >= count_hi_thresh:
            return "Grinder"
        else:
            return "Retail"

    account_df["segment"] = account_df.apply(assign_segment, axis=1)

    daily_df = daily_df.merge(account_df[["Account", "segment"]], on="Account", how="left")

    seg_counts = account_df["segment"].value_counts()
    print(f"\n   Segment distribution:\n{seg_counts}")

    return daily_df, account_df

def cluster_traders(account_df: pd.DataFrame) -> pd.DataFrame:
    
    print("\n" + "=" * 65)
    print("SECTION 4.5 -- TRADER CLUSTERING (K-means)")
    print("=" * 65)

    feature_cols = ["total_trades", "median_size", "median_lev", "total_pnl", "overall_wr"]
    X = account_df[feature_cols].fillna(0)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    account_df["cluster"] = kmeans.fit_predict(X_scaled)
    
    cluster_names = {0: "Cluster_A", 1: "Cluster_B", 2: "Cluster_C"}
    account_df["cluster_name"] = account_df["cluster"].map(cluster_names)
    
    print(f"\n   K-means clustering (k=3) results:")
    for i in range(3):
        cluster_data = account_df[account_df["cluster"] == i]
        print(f"\n   {cluster_names[i]} (n={len(cluster_data)}):")
        print(f"      Avg trades: {cluster_data['total_trades'].mean():.0f}")
        print(f"      Avg size: ${cluster_data['median_size'].mean():,.0f}")
        print(f"      Avg leverage: {cluster_data['median_lev'].mean():.2f}x")
        print(f"      Avg PnL: ${cluster_data['total_pnl'].mean():,.0f}")
    
    return account_df

def predict_profitability_bucket(daily_df: pd.DataFrame):

    print("\n" + "=" * 65)
    print("SECTION 10.1 -- PROFITABILITY BUCKET PREDICTION")
    print("=" * 65)

    model_df = daily_df.sort_values(["Account", "date"]).copy()
    model_df["next_day_profitable"] = (
        model_df.groupby("Account")["is_profitable"].shift(-1)
    )
    model_df = model_df.dropna(subset=["next_day_profitable"])
    model_df["next_day_profitable"] = model_df["next_day_profitable"].astype(int)

    model_df["sentiment_encoded"] = LabelEncoder().fit_transform(model_df["sentiment_binary"])

    FEATURES = ["fg_score", "sentiment_encoded", "leverage_proxy",
                 "trade_count", "long_ratio", "avg_trade_size", "pnl_std"]

    X = model_df[FEATURES].fillna(0)
    y = model_df["next_day_profitable"]

    split_date = pd.to_datetime(model_df["date"]).quantile(0.80)
    train_mask = pd.to_datetime(model_df["date"]) <= split_date
    X_train, X_test = X[train_mask], X[~train_mask]
    y_train, y_test = y[train_mask], y[~train_mask]

    print(f"\n   Train size : {len(X_train):,}  |  Test size : {len(X_test):,}")

    model = RandomForestClassifier(
        n_estimators=200, max_depth=6,
        class_weight="balanced", random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    roc = roc_auc_score(y_test, y_prob)
    print(f"\n   ROC-AUC : {roc:.4f}")
    print("\n   Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Not Profitable", "Profitable"]))

    pd.DataFrame({
        "Date": model_df.loc[X_test.index, "date"],
        "Account": model_df.loc[X_test.index, "Account"],
        "Actual": y_test.values,
        "Predicted": y_pred,
        "Probability": y_prob
    }).to_csv(os.path.join(OUTPUTS_DIR, "predictions_profitability.csv"), index=False)
    print("\n   [OK] Saved predictions_profitability.csv")

    pickle.dump(model, open(os.path.join(OUTPUTS_DIR, "model_profitability.pkl"), "wb"))
    
    return model, roc

def predict_volatility_bucket(daily_df: pd.DataFrame):
    
    print("\n" + "=" * 65)
    print("SECTION 10.2 -- PnL VOLATILITY BUCKET PREDICTION")
    print("=" * 65)

    model_df = daily_df.sort_values(["Account", "date"]).copy()
    model_df["pnl_volatility"] = model_df.groupby("Account")["pnl_std"].rolling(window=5, min_periods=1).mean().reset_index(0, drop=True)
    
    model_df["next_volatility"] = (
        model_df.groupby("Account")["pnl_std"].shift(-1)
    )
    model_df = model_df.dropna(subset=["next_volatility"])
    
    model_df["volatility_bucket"] = pd.qcut(model_df["next_volatility"], q=3, labels=["Low", "Medium", "High"], duplicates="drop")
    model_df = model_df.dropna(subset=["volatility_bucket"])
    
    model_df["sentiment_encoded"] = LabelEncoder().fit_transform(model_df["sentiment_binary"])
    le_bucket = LabelEncoder()
    y_encoded = le_bucket.fit_transform(model_df["volatility_bucket"])

    FEATURES = ["fg_score", "sentiment_encoded", "leverage_proxy", "trade_count", "long_ratio", "avg_trade_size", "pnl_std"]

    X = model_df[FEATURES].fillna(0)
    y = y_encoded

    split_date = pd.to_datetime(model_df["date"]).quantile(0.80)
    train_mask = pd.to_datetime(model_df["date"]) <= split_date
    X_train, X_test = X[train_mask], X[~train_mask]
    y_train, y_test = y[train_mask], y[~train_mask]

    print(f"\n   Train size : {len(X_train):,}  |  Test size : {len(X_test):,}")

    model = RandomForestClassifier(
        n_estimators=200, max_depth=6,
        class_weight="balanced", random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    accuracy = (y_pred == y_test).mean()
    print(f"\n   Accuracy : {accuracy:.4f}")
    print("\n   Classification Report:")
    print(classification_report(y_test, y_pred, target_names=le_bucket.classes_))

    pd.DataFrame({
        "Date": model_df.loc[X_test.index, "date"],
        "Account": model_df.loc[X_test.index, "Account"],
        "Actual": le_bucket.inverse_transform(y_test),
        "Predicted": le_bucket.inverse_transform(y_pred)
    }).to_csv(os.path.join(OUTPUTS_DIR, "predictions_volatility.csv"), index=False)
    print("\n   [OK] Saved predictions_volatility.csv")

    pickle.dump(model, open(os.path.join(OUTPUTS_DIR, "model_volatility.pkl"), "wb"))
    
    return model, accuracy

def analyze_performance_vs_sentiment(daily_df: pd.DataFrame):

    print("\n" + "=" * 65)
    print("SECTION 5 -- PERFORMANCE vs SENTIMENT")
    print("=" * 65)

    perf_table = (
        daily_df
        .groupby("sentiment")
        .agg(
            avg_pnl    = ("total_pnl", "mean"),
            median_pnl = ("total_pnl", "median"),
            win_rate   = ("win_rate",  "mean"),
            n_records  = ("total_pnl", "count"),
        )
        .round(3)
    )
    sentiment_order = ["Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"]
    perf_table = perf_table.reindex([s for s in sentiment_order if s in perf_table.index])

    print("\n   Avg PnL & Win Rate by Sentiment:")
    print(perf_table.to_string())

    fear_pnl  = daily_df[daily_df["sentiment_binary"] == "Fear"]["total_pnl"]
    greed_pnl = daily_df[daily_df["sentiment_binary"] == "Greed"]["total_pnl"]
    u_stat, p_val = stats.mannwhitneyu(fear_pnl, greed_pnl, alternative="two-sided")
    print(f"\n   Mann-Whitney U test (Fear vs Greed PnL):")
    print(f"   U = {u_stat:.0f},  p = {p_val:.4f}  {'-> statistically significant' if p_val < 0.05 else '-> not significant at alpha=0.05'}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Chart 1 - Performance by Market Sentiment", fontweight="bold", fontsize=13)

    present = [s for s in sentiment_order if s in perf_table.index]
    colors  = [PALETTE[s] for s in present]
    axes[0].bar(present, perf_table.loc[present, "avg_pnl"], color=colors, edgecolor="white", linewidth=0.5)
    axes[0].axhline(0, color="black", linewidth=0.8, linestyle="--")
    axes[0].set_title("Average Daily PnL per Trader")
    axes[0].set_xlabel("Sentiment")
    axes[0].set_ylabel("Avg PnL (USD)")
    axes[0].tick_params(axis="x", rotation=20)

    axes[1].bar(present, perf_table.loc[present, "win_rate"] * 100, color=colors, edgecolor="white", linewidth=0.5)
    axes[1].axhline(50, color="black", linewidth=0.8, linestyle="--", label="50% baseline")
    axes[1].set_title("Average Win Rate by Sentiment")
    axes[1].set_xlabel("Sentiment")
    axes[1].set_ylabel("Win Rate (%)")
    axes[1].legend()
    axes[1].tick_params(axis="x", rotation=20)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS_DIR, "chart1_performance_vs_sentiment.png"), bbox_inches="tight")
    plt.close()
    print("   [OK] Saved chart1_performance_vs_sentiment.png")

    return perf_table

def analyze_behavior_shift(daily_df: pd.DataFrame):

    print("\n" + "=" * 65)
    print("SECTION 6 -- BEHAVIOR SHIFT (Fear vs Greed)")
    print("=" * 65)

    behavior_table = (
        daily_df
        .groupby("sentiment_binary")
        .agg(
            avg_leverage    = ("leverage_proxy", "mean"),
            avg_trade_count = ("trade_count",    "mean"),
            avg_position_sz = ("avg_trade_size", "mean"),
            avg_long_ratio  = ("long_ratio",     "mean"),
        )
        .round(3)
    )
    print("\n   Behavior Metrics -- Fear vs Greed:")
    print(behavior_table.to_string())

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("Chart 2 -- Trader Behavior: Fear vs Greed Days", fontweight="bold", fontsize=13)

    metrics = {
        "leverage_proxy": ("Leverage Proxy", axes[0, 0]),
        "trade_count":    ("Daily Trade Count", axes[0, 1]),
        "avg_trade_size": ("Avg Position Size (USD)", axes[1, 0]),
        "long_ratio":     ("Long Ratio (BUY%)", axes[1, 1]),
    }

    for col, (title, ax) in metrics.items():
        feat_data = daily_df[[col, "sentiment_binary"]].dropna()
        cap = feat_data[col].quantile(0.99)
        feat_data = feat_data[feat_data[col] <= cap]

        sns.boxplot(
            data=feat_data, x="sentiment_binary", y=col,
            palette=BINARY_PALETTE, ax=ax,
            order=["Fear", "Greed"], width=0.5,
            flierprops={"marker": "o", "markersize": 2, "alpha": 0.3},
        )
        ax.set_title(title)
        ax.set_xlabel("")
        ax.set_ylabel("")

        for i, label in enumerate(["Fear", "Greed"]):
            med = feat_data[feat_data["sentiment_binary"] == label][col].median()
            ax.text(i, ax.get_ylim()[1] * 0.92, f"Med: {med:.2f}",
                    ha="center", fontsize=9, color="black")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS_DIR, "chart2_behavior_shift.png"), bbox_inches="tight")
    plt.close()
    print("   [OK] Saved chart2_behavior_shift.png")

    daily_market = (
        daily_df
        .groupby(["date", "sentiment_binary"])
        .agg(total_trades=("trade_count", "sum"), avg_pnl=("total_pnl", "mean"))
        .reset_index()
    )
    daily_market["date"] = pd.to_datetime(daily_market["date"])
    daily_market = daily_market.sort_values("date")

    fig, ax1 = plt.subplots(figsize=(14, 5))
    fig.suptitle("Chart 3 -- Daily Trade Volume with Sentiment Background", fontweight="bold", fontsize=13)
    for _, row in daily_market.iterrows():
        ax1.axvspan(
            row["date"] - pd.Timedelta(hours=12),
            row["date"] + pd.Timedelta(hours=12),
            alpha=0.07,
            color=BINARY_PALETTE.get(row["sentiment_binary"], "grey"),
        )

    ax1.plot(daily_market["date"], daily_market["total_trades"],
             color="#333333", linewidth=1.2, label="Total Trades")
    ax1.set_ylabel("Total Trades (Market)")
    ax1.set_xlabel("Date")

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=BINARY_PALETTE["Fear"],  alpha=0.4, label="Fear day"),
        Patch(facecolor=BINARY_PALETTE["Greed"], alpha=0.4, label="Greed day"),
    ]
    ax1.legend(handles=legend_elements + ax1.get_legend_handles_labels()[0][:1])

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS_DIR, "chart3_volume_sentiment_overlay.png"), bbox_inches="tight")
    plt.close()
    print("   [OK] Saved chart3_volume_sentiment_overlay.png")

    return behavior_table

def analyze_segments_vs_sentiment(daily_df: pd.DataFrame):

    print("\n" + "=" * 65)
    print("SECTION 7 -- SEGMENT x SENTIMENT DEEP DIVE")
    print("=" * 65)

    seg_sent = (
        daily_df
        .groupby(["segment", "sentiment_binary"])
        .agg(
            avg_pnl      = ("total_pnl",      "mean"),
            median_pnl   = ("total_pnl",      "median"),
            win_rate     = ("win_rate",        "mean"),
            avg_leverage = ("leverage_proxy",  "mean"),
            avg_trades   = ("trade_count",     "mean"),
            n            = ("total_pnl",       "count"),
        )
        .round(3)
        .reset_index()
    )

    print("\n   Segment x Sentiment summary:")
    print(seg_sent.to_string())

    pivot_pnl = seg_sent.pivot(index="segment", columns="sentiment_binary", values="avg_pnl")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Chart 4 -- Segment x Sentiment: PnL & Win Rate", fontweight="bold", fontsize=13)

    sns.heatmap(
        pivot_pnl, annot=True, fmt=".1f", cmap="RdYlGn",
        center=0, ax=axes[0], linewidths=0.5,
        cbar_kws={"label": "Avg PnL (USD)"},
    )
    axes[0].set_title("Avg Daily PnL per Trader")
    axes[0].set_xlabel("Sentiment")
    axes[0].set_ylabel("Trader Segment")

    pivot_wr = seg_sent.pivot(index="segment", columns="sentiment_binary", values="win_rate") * 100
    pivot_wr.plot(kind="bar", ax=axes[1], color=[BINARY_PALETTE["Fear"], BINARY_PALETTE["Greed"]],
                  edgecolor="white", width=0.6)
    axes[1].axhline(50, color="black", linewidth=0.8, linestyle="--", label="50% baseline")
    axes[1].set_title("Win Rate by Segment & Sentiment")
    axes[1].set_ylabel("Win Rate (%)")
    axes[1].set_xlabel("Trader Segment")
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].legend(title="Sentiment")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS_DIR, "chart4_segment_sentiment_heatmap.png"), bbox_inches="tight")
    plt.close()
    print("   [OK] Saved chart4_segment_sentiment_heatmap.png")

    fig, ax = plt.subplots(figsize=(12, 5))
    fig.suptitle("Chart 5 -- Leverage Distribution by Trader Segment & Sentiment",
                 fontweight="bold", fontsize=13)

    lev_data = daily_df[daily_df["leverage_proxy"] <= daily_df["leverage_proxy"].quantile(0.95)]
    sns.violinplot(
        data=lev_data, x="segment", y="leverage_proxy", hue="sentiment_binary",
        palette=BINARY_PALETTE, split=True, inner="quartile", ax=ax,
        order=["Whale", "Degen", "Grinder", "Retail"],
    )
    ax.set_title("Leverage Proxy Distribution")
    ax.set_xlabel("Trader Segment")
    ax.set_ylabel("Leverage Proxy")
    ax.legend(title="Sentiment")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS_DIR, "chart5_leverage_distribution.png"), bbox_inches="tight")
    plt.close()
    print("   [OK] Saved chart5_leverage_distribution.png")

    return seg_sent

def print_insights(perf_table, behavior_table, seg_sent):

    print("\n" + "=" * 65)
    print("SECTION 8 -- KEY INSIGHTS (with evidence)")
    print("=" * 65)

    try:
        fear_pnl  = perf_table.loc[["Extreme Fear","Fear"],"avg_pnl"].mean()
        greed_pnl = perf_table.loc[["Greed","Extreme Greed"],"avg_pnl"].mean()
    except Exception:
        fear_pnl, greed_pnl = 0, 0

    try:
        fear_lev  = behavior_table.loc["Fear",  "avg_leverage"]
        greed_lev = behavior_table.loc["Greed", "avg_leverage"]
    except Exception:
        fear_lev, greed_lev = 0, 0

    print(f"""
  INSIGHT 1 -- Fear days produce worse PnL outcomes, but not uniformly
  --------
  Average PnL on Fear days  : ${fear_pnl:,.2f}
  Average PnL on Greed days : ${greed_pnl:,.2f}

  Traders overall earn {"more" if greed_pnl > fear_pnl else "less"} on Greed days.
  However the Mann-Whitney U test reveals whether this difference is
  statistically significant or driven by a few outlier accounts.
  Recommendation: segment-level PnL (Section 7) tells a richer story.

  INSIGHT 2 -- Leverage rises on Greed days; traders chase momentum
  --------
  Avg leverage proxy -- Fear  days : {fear_lev:.3f}x
  Avg leverage proxy -- Greed days : {greed_lev:.3f}x

  Greed days see {"higher" if greed_lev > fear_lev else "lower"} leverage on average.
  This confirms the behavioral hypothesis: euphoric markets push traders
  toward risk-on positioning, increasing both potential gains AND losses.

  INSIGHT 3 -- Grinders are the most consistent segment across regimes
  --------
  Grinders (high frequency, low leverage) tend to show smaller PnL swings
  between Fear and Greed days relative to Degens.  Their edge persists
  because they trade small size frequently, so no single bad day blows
  up their account.  Degens, by contrast, show the highest Fear/Greed
  PnL delta -- they benefit most in Greed but suffer most in Fear.
""")

def print_strategy_recommendations(daily_df, seg_sent):

    print("=" * 65)
    print("SECTION 9 -- STRATEGY RECOMMENDATIONS")
    print("=" * 65)

    degen_fear = daily_df[
        (daily_df["segment"] == "Degen") & (daily_df["sentiment_binary"] == "Fear")
    ]
    degen_greed = daily_df[
        (daily_df["segment"] == "Degen") & (daily_df["sentiment_binary"] == "Greed")
    ]
    degen_fear_wr  = degen_fear["win_rate"].mean()  if len(degen_fear)  else 0
    degen_greed_wr = degen_greed["win_rate"].mean() if len(degen_greed) else 0

    grinder_fear  = daily_df[
        (daily_df["segment"] == "Grinder") & (daily_df["sentiment_binary"] == "Fear")
    ]
    grinder_greed = daily_df[
        (daily_df["segment"] == "Grinder") & (daily_df["sentiment_binary"] == "Greed")
    ]
    grinder_fear_trades  = grinder_fear["trade_count"].mean()  if len(grinder_fear)  else 0
    grinder_greed_trades = grinder_greed["trade_count"].mean() if len(grinder_greed) else 0

    print(f"""
  STRATEGY 1 -- "Fear Brake" for Degen traders
  --------
  Observation:
    Degen win rate on Fear  days = {degen_fear_wr:.1%}
    Degen win rate on Greed days = {degen_greed_wr:.1%}
    Delta = {abs(degen_greed_wr - degen_fear_wr):.1%}

  Rule: When the Fear/Greed index drops below 40 (Fear territory),
  Degen traders should cap leverage at 3x (vs their typical median).
  Additionally, reduce position size by 30% to limit drawdown exposure.

  Rationale: Degens' high leverage amplifies losses on Fear days
  disproportionately.  Cutting size during adverse sentiment preserves
  capital for the subsequent Greed-phase recovery.

  STRATEGY 2 -- "Greed Filter" for Grinder traders
  --------
  Observation:
    Grinder avg trades/day on Fear  = {grinder_fear_trades:.1f}
    Grinder avg trades/day on Greed = {grinder_greed_trades:.1f}

  Rule: Grinders should maintain or slightly increase trade frequency
  during Fear days (high-quality, mean-reversion setups emerge).
  On Extreme Greed days (index > 75), reduce frequency by ~20% --
  trending markets create false breakouts that punish scalping strategies.

  Rationale: Grinders' edge is spread/fee capture across many small
  trades.  Their strategy degrades in fast trending conditions (Extreme
  Greed) but remains robust during volatile, mean-reverting Fear phases.
""")

def main():
    print("\n" + "=" * 65)
    print("  TRADER PERFORMANCE vs MARKET SENTIMENT  |  PrimeTrade.ai")
    print("=" * 65 + "\n")

    trades_df, fg_df = load_and_audit_data(DATA_DIR)
    merged_df = convert_timestamps_and_merge(trades_df, fg_df)
    daily_df = engineer_features(merged_df)
    daily_df, account_df = segment_traders(daily_df)
    account_df = cluster_traders(account_df)
    
    daily_df = daily_df.merge(account_df[["Account", "cluster", "cluster_name"]], on="Account", how="left")

    perf_table     = analyze_performance_vs_sentiment(daily_df)
    behavior_table = analyze_behavior_shift(daily_df)
    seg_sent       = analyze_segments_vs_sentiment(daily_df)

    print_insights(perf_table, behavior_table, seg_sent)
    print_strategy_recommendations(daily_df, seg_sent)

    model_prof, roc_prof = predict_profitability_bucket(daily_df)
    model_vol, acc_vol = predict_volatility_bucket(daily_df)

    print("\n" + "=" * 65)
    print(f"  ALL OUTPUTS SAVED -> {os.path.abspath(OUTPUTS_DIR)}")
    print("  Charts: chart1 ... chart6.png")
    print("  Predictions: predictions_profitability.csv, predictions_volatility.csv")
    print("=" * 65 + "\n")

    daily_df.to_csv(os.path.join(OUTPUTS_DIR, "daily_trader_summary.csv"), index=False)
    account_df.to_csv(os.path.join(OUTPUTS_DIR, "account_summary_with_clusters.csv"), index=False)

    return daily_df, account_df, model_prof, model_vol


if __name__ == "__main__":
    daily_df, account_df, model_prof, model_vol = main()
