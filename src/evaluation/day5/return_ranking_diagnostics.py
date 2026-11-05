from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.models.day5.time_split import create_time_split
from src.utils.mvp_config import modeling_dir, models_dir


DATASET_PATH = modeling_dir() / "day5" / "nse100_technical_model_dataset.parquet"

MODEL_PATH = models_dir() / "xgboost_return_5d_xgb_v1.joblib"

TARGET = "target_return_5d"

FEATURES = [
    "return_1d",
    "return_3d",
    "return_5d",
    "return_10d",
    "return_20d",
    "rsi_14",
    "roc_10",
    "roc_20",
    "volatility_5d",
    "volatility_20d",
    "atr_pct_14",
    "volume_change_1d",
    "volume_ratio_20d",
    "close_vs_sma20",
    "close_vs_sma50",
    "close_vs_ema20",
    "nifty50_return_1d",
    "nifty50_return_5d",
    "nifty50_return_20d",
    "stock_vs_nifty50_5d",
    "stock_vs_nifty50_20d",
    "rolling_beta_20d",
    "nifty50_volatility_20d",
    "relative_volatility_20d",
]


def main():
    print("=" * 70)
    print("XGBoost 5D Return Ranking Diagnostics")
    print("=" * 70)

    # ------------------------------------------------------------
    # Load data and model
    # ------------------------------------------------------------

    df = pd.read_parquet(DATASET_PATH)
    model = joblib.load(MODEL_PATH)

    print(f"\nDataset rows: {len(df):,}")

    _, _, test_df = create_time_split(df)

    # Only rows with known targets.
    test_df = test_df[
        test_df[TARGET].notna()
    ].copy()

    X_test = test_df[FEATURES]

    test_df["predicted_return_5d"] = model.predict(X_test)

    print(f"Test rows: {len(test_df):,}")

    # ------------------------------------------------------------
    # Daily cross-sectional ranking
    # ------------------------------------------------------------

    daily_results = []

    for trade_date, day in test_df.groupby("trade_date"):

        # Need enough stocks to make ranking meaningful.
        if len(day) < 5:
            continue

        actual = day[TARGET].to_numpy(dtype=float)
        predicted = day["predicted_return_5d"].to_numpy(dtype=float)

        # Spearman rank correlation.
        spearman = spearmanr(
            predicted,
            actual,
        ).statistic

        # Ranking.
        ranked = day.sort_values(
            "predicted_return_5d",
            ascending=False,
        )

        universe_return = ranked[TARGET].mean()

        top5_return = ranked.head(5)[TARGET].mean()
        top10_return = ranked.head(10)[TARGET].mean()
        top20_return = ranked.head(20)[TARGET].mean()

        daily_results.append(
            {
                "trade_date": trade_date,
                "stock_count": len(day),
                "spearman": spearman,
                "universe_return": universe_return,
                "top5_return": top5_return,
                "top10_return": top10_return,
                "top20_return": top20_return,
            }
        )

    daily = pd.DataFrame(daily_results)

    if daily.empty:
        raise RuntimeError(
            "No valid daily cross-sectional ranking results."
        )

    # ------------------------------------------------------------
    # Overall ranking statistics
    # ------------------------------------------------------------

    print("\nDaily Cross-Sectional Ranking")
    print("-" * 70)

    print(f"Trading days: {len(daily):,}")

    print(
        f"Average Spearman correlation : "
        f"{daily['spearman'].mean():.6f}"
    )

    print(
        f"Median Spearman correlation  : "
        f"{daily['spearman'].median():.6f}"
    )

    print(
        f"Spearman > 0 days            : "
        f"{(daily['spearman'] > 0).mean():.2%}"
    )

    # ------------------------------------------------------------
    # Top-N performance
    # ------------------------------------------------------------

    print("\nTop-N Actual 5D Returns")
    print("-" * 70)

    universe_avg = daily["universe_return"].mean()
    top5_avg = daily["top5_return"].mean()
    top10_avg = daily["top10_return"].mean()
    top20_avg = daily["top20_return"].mean()

    print(f"Universe average : {universe_avg:.6%}")
    print(f"Top 5 average    : {top5_avg:.6%}")
    print(f"Top 10 average   : {top10_avg:.6%}")
    print(f"Top 20 average   : {top20_avg:.6%}")

    # ------------------------------------------------------------
    # Top-N spread versus universe
    # ------------------------------------------------------------

    print("\nTop-N Spread vs Universe")
    print("-" * 70)

    print(
        f"Top 5 spread  : "
        f"{top5_avg - universe_avg:.6%}"
    )

    print(
        f"Top 10 spread : "
        f"{top10_avg - universe_avg:.6%}"
    )

    print(
        f"Top 20 spread : "
        f"{top20_avg - universe_avg:.6%}"
    )

    # ------------------------------------------------------------
    # Hit rate
    # ------------------------------------------------------------

    print("\nTop-N Positive Return Hit Rate")
    print("-" * 70)

    # Calculate daily percentage of selected stocks
    # whose actual 5D return was positive.

    hit_results = []

    for trade_date, day in test_df.groupby("trade_date"):

        if len(day) < 5:
            continue

        ranked = day.sort_values(
            "predicted_return_5d",
            ascending=False,
        )

        hit_results.append(
            {
                "top5_hit_rate":
                    (ranked.head(5)[TARGET] > 0).mean(),

                "top10_hit_rate":
                    (ranked.head(10)[TARGET] > 0).mean(),

                "top20_hit_rate":
                    (ranked.head(20)[TARGET] > 0).mean(),

                "universe_hit_rate":
                    (day[TARGET] > 0).mean(),
            }
        )

    hit_df = pd.DataFrame(hit_results)

    print(
        f"Universe : "
        f"{hit_df['universe_hit_rate'].mean():.2%}"
    )

    print(
        f"Top 5    : "
        f"{hit_df['top5_hit_rate'].mean():.2%}"
    )

    print(
        f"Top 10   : "
        f"{hit_df['top10_hit_rate'].mean():.2%}"
    )

    print(
        f"Top 20   : "
        f"{hit_df['top20_hit_rate'].mean():.2%}"
    )

    # ------------------------------------------------------------
    # Save diagnostics
    # ------------------------------------------------------------

    output_path = Path(
        "data/predictions/"
        "xgboost_return_5d_ranking_diagnostics.parquet"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    daily.to_parquet(
        output_path,
        index=False,
    )

    print("\nSaved:")
    print(output_path)

    print("\n" + "=" * 70)
    print("RANKING DIAGNOSTICS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()