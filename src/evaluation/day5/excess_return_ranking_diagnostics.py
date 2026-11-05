"""
XGBoost 5D Excess Return Ranking Diagnostics

Purpose
-------
Evaluate whether the excess-return XGBoost model can rank stocks
cross-sectionally by their future 5-day excess return.

This is different from ordinary regression metrics.

For each trading day:
    1. Predict excess return for every available stock.
    2. Rank stocks by predicted excess return.
    3. Compare the ranking with actual 5-day excess return.
    4. Evaluate Top-5 / Top-10 / Top-20 portfolios.

No training is performed here.
The previously trained model is loaded and evaluated on the
strict out-of-sample TEST period.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.utils.mvp_config import modeling_dir, models_dir, predictions_dir


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

DATASET_PATH = (
    modeling_dir()
    / "day5"
    / "nse100_technical_model_dataset.parquet"
)

MODEL_PATH = models_dir() / "xgboost_excess_return_5d_xgb_v1.joblib"

OUTPUT_PATH = predictions_dir() / "xgboost_excess_return_5d_ranking_diagnostics.parquet"


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

TARGET = "target_excess_return_5d"

TOP_N_VALUES = [5, 10, 20]


# ---------------------------------------------------------------------
# Time split
# ---------------------------------------------------------------------

def create_test_split(df: pd.DataFrame) -> pd.DataFrame:
    """
    Recreate the same time split used during model training.

    Test period:
        2026-01-01 onward

    The model is evaluated only on this period.
    """

    df = df.copy()

    df["trade_date"] = pd.to_datetime(df["trade_date"])

    test = df[
        df["trade_date"] >= pd.Timestamp("2026-01-01")
    ].copy()

    return test


# ---------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------

def validate_dataset(df: pd.DataFrame) -> None:

    missing_features = [
        feature
        for feature in FEATURES
        if feature not in df.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing model features: {missing_features}"
        )

    if TARGET not in df.columns:
        raise ValueError(
            f"Missing target column: {TARGET}"
        )

    required = [
        "symbol",
        "trade_date",
        TARGET,
    ]

    missing_required = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing_required:
        raise ValueError(
            f"Missing required columns: {missing_required}"
        )


# ---------------------------------------------------------------------
# Ranking helper
# ---------------------------------------------------------------------

def calculate_daily_ranking(
    day_df: pd.DataFrame,
) -> dict:

    day_df = day_df.copy()

    day_df = day_df.dropna(
        subset=["prediction", TARGET]
    )

    if len(day_df) < 2:
        return {
            "rows": len(day_df),
            "spearman": np.nan,
            "top_5_actual": np.nan,
            "top_10_actual": np.nan,
            "top_20_actual": np.nan,
            "universe_actual": np.nan,
            "top_5_hit_rate": np.nan,
            "top_10_hit_rate": np.nan,
            "top_20_hit_rate": np.nan,
        }

    # -------------------------------------------------------------
    # Spearman rank correlation
    # -------------------------------------------------------------

    correlation = spearmanr(
        day_df["prediction"],
        day_df[TARGET],
    ).statistic

    # -------------------------------------------------------------
    # Universe actual excess return
    # -------------------------------------------------------------

    universe_actual = day_df[TARGET].mean()

    results = {
        "rows": len(day_df),
        "spearman": correlation,
        "universe_actual": universe_actual,
    }

    # -------------------------------------------------------------
    # Top-N analysis
    # -------------------------------------------------------------

    for top_n in TOP_N_VALUES:

        n = min(top_n, len(day_df))

        top = (
            day_df
            .sort_values(
                "prediction",
                ascending=False,
            )
            .head(n)
        )

        actual_return = top[TARGET].mean()

        positive_hit_rate = (
            top[TARGET] > 0
        ).mean()

        results[f"top_{top_n}_actual"] = actual_return

        results[
            f"top_{top_n}_hit_rate"
        ] = positive_hit_rate

    return results


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    print("=" * 70)
    print("XGBoost 5D Excess Return Ranking Diagnostics")
    print("=" * 70)

    # -------------------------------------------------------------
    # Load dataset
    # -------------------------------------------------------------

    print()
    print("Loading dataset...")

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}"
        )

    df = pd.read_parquet(DATASET_PATH)

    print(f"Dataset rows: {len(df):,}")
    print(
        f"Dataset stocks: "
        f"{df['symbol'].nunique():,}"
    )

    validate_dataset(df)

    # -------------------------------------------------------------
    # Load model
    # -------------------------------------------------------------

    print()
    print("Loading trained model...")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    model = joblib.load(MODEL_PATH)

    # -------------------------------------------------------------
    # Test split
    # -------------------------------------------------------------

    test = create_test_split(df)

    print()
    print("TEST DATA")
    print("-" * 70)

    print(f"Rows:   {len(test):,}")
    print(
        f"Stocks: "
        f"{test['symbol'].nunique():,}"
    )
    print(
        f"Dates:  "
        f"{test['trade_date'].min().date()} "
        f"-> "
        f"{test['trade_date'].max().date()}"
    )

    # -------------------------------------------------------------
    # Remove rows with missing target/features
    # -------------------------------------------------------------

    before = len(test)

    test = test.dropna(
        subset=FEATURES + [TARGET]
    ).copy()

    removed = before - len(test)

    print()
    print("USABLE TEST ROWS")
    print("-" * 70)
    print(f"Original rows : {before:,}")
    print(f"Removed rows  : {removed:,}")
    print(f"Usable rows   : {len(test):,}")

    if test.empty:
        raise ValueError(
            "No usable test rows after filtering."
        )

    # -------------------------------------------------------------
    # Build feature matrix
    # -------------------------------------------------------------

    X_test = test[FEATURES]

    # Make sure all features are numeric.
    non_numeric = [
        column
        for column in FEATURES
        if not pd.api.types.is_numeric_dtype(
            X_test[column]
        )
    ]

    if non_numeric:
        raise TypeError(
            f"Non-numeric model features: {non_numeric}"
        )

    # -------------------------------------------------------------
    # Generate predictions
    # -------------------------------------------------------------

    print()
    print("Generating predictions...")

    predictions = model.predict(X_test)

    predictions = np.asarray(
        predictions,
        dtype=float,
    )

    if not np.isfinite(predictions).all():
        raise ValueError(
            "Model generated NaN or infinite predictions."
        )

    test["prediction"] = predictions

    # -------------------------------------------------------------
    # Prediction statistics
    # -------------------------------------------------------------

    print()
    print("PREDICTION DISTRIBUTION")
    print("-" * 70)

    print(
        pd.Series(
            predictions
        ).describe(
            percentiles=[
                0.01,
                0.05,
                0.25,
                0.50,
                0.75,
                0.95,
                0.99,
            ]
        )
    )

    # -------------------------------------------------------------
    # Daily cross-sectional ranking
    # -------------------------------------------------------------

    print()
    print("Daily Cross-Sectional Ranking")
    print("-" * 70)

    daily_results = []

    for trade_date, day_df in test.groupby(
        "trade_date",
        sort=True,
    ):

        result = calculate_daily_ranking(
            day_df
        )

        result["trade_date"] = trade_date

        daily_results.append(result)

    daily = pd.DataFrame(
        daily_results
    )

    # -------------------------------------------------------------
    # Spearman statistics
    # -------------------------------------------------------------

    spearman_values = (
        daily["spearman"]
        .dropna()
    )

    print(
        f"Trading days: "
        f"{len(daily):,}"
    )

    print(
        f"Average Spearman correlation : "
        f"{spearman_values.mean():.6f}"
    )

    print(
        f"Median Spearman correlation  : "
        f"{spearman_values.median():.6f}"
    )

    print(
        f"Spearman > 0 days            : "
        f"{(spearman_values > 0).mean():.2%}"
    )

    # -------------------------------------------------------------
    # Top-N actual returns
    # -------------------------------------------------------------

    print()
    print("Top-N Actual 5D Excess Returns")
    print("-" * 70)

    universe_mean = (
        daily["universe_actual"]
        .mean()
    )

    print(
        f"Universe average : "
        f"{universe_mean:.6%}"
    )

    for top_n in TOP_N_VALUES:

        value = (
            daily[f"top_{top_n}_actual"]
            .mean()
        )

        print(
            f"Top {top_n:<2} average    : "
            f"{value:.6%}"
        )

    # -------------------------------------------------------------
    # Top-N spread
    # -------------------------------------------------------------

    print()
    print("Top-N Spread vs Universe")
    print("-" * 70)

    for top_n in TOP_N_VALUES:

        top_value = (
            daily[f"top_{top_n}_actual"]
            .mean()
        )

        spread = (
            top_value
            - universe_mean
        )

        print(
            f"Top {top_n:<2} spread  : "
            f"{spread:.6%}"
        )

    # -------------------------------------------------------------
    # Positive-return hit rate
    # -------------------------------------------------------------

    print()
    print("Top-N Positive Excess Return Hit Rate")
    print("-" * 70)

    universe_hit_rate = (
        test[TARGET] > 0
    ).mean()

    print(
        f"Universe : "
        f"{universe_hit_rate:.2%}"
    )

    for top_n in TOP_N_VALUES:

        value = (
            daily[f"top_{top_n}_hit_rate"]
            .mean()
        )

        print(
            f"Top {top_n:<2}    : "
            f"{value:.2%}"
        )

    # -------------------------------------------------------------
    # Ranking diagnostics by year
    # -------------------------------------------------------------

    print()
    print("PERFORMANCE BY YEAR")
    print("-" * 70)

    test["year"] = (
        test["trade_date"]
        .dt.year
    )

    yearly_rows = []

    for year, year_df in test.groupby(
        "year",
        sort=True,
    ):

        year_daily = []

        for trade_date, day_df in year_df.groupby(
            "trade_date",
            sort=True,
        ):

            result = calculate_daily_ranking(
                day_df
            )

            year_daily.append(result)

        year_daily = pd.DataFrame(
            year_daily
        )

        yearly_rows.append(
            {
                "year": year,
                "rows": len(year_df),
                "trading_days": len(year_daily),
                "spearman": (
                    year_daily["spearman"]
                    .mean()
                ),
                "top_5_actual": (
                    year_daily["top_5_actual"]
                    .mean()
                ),
                "top_10_actual": (
                    year_daily["top_10_actual"]
                    .mean()
                ),
                "top_20_actual": (
                    year_daily["top_20_actual"]
                    .mean()
                ),
                "universe_actual": (
                    year_daily["universe_actual"]
                    .mean()
                ),
            }
        )

    yearly = pd.DataFrame(
        yearly_rows
    )

    print(
        yearly.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    # -------------------------------------------------------------
    # Save detailed predictions
    # -------------------------------------------------------------

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_columns = [
        "symbol",
        "trade_date",
        TARGET,
        "prediction",
    ]

    output = test[
        output_columns
    ].copy()

    output.to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    print()
    print("Saved:")
    print(OUTPUT_PATH)

    # -------------------------------------------------------------
    # Final
    # -------------------------------------------------------------

    print()
    print("=" * 70)
    print("EXCESS RETURN RANKING DIAGNOSTICS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()