from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

from src.models.time_split import create_time_split
from src.utils.mvp_config import modeling_dir, models_dir


DATASET_PATH = modeling_dir() / "day5" / "nse100_technical_model_dataset.parquet"

MODEL_PATH = models_dir() / "xgboost_direction_5d_xgb_v1.joblib"

TARGET = "target_direction_5d"

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


def evaluate_predictions(
    name: str,
    y_true: pd.Series,
    probabilities: np.ndarray,
) -> None:

    predictions = (probabilities >= 0.5).astype(int)

    print(f"\n{name}")
    print("-" * 60)

    print(f"Rows:     {len(y_true):,}")
    print(f"Positive: {y_true.mean():.4f}")

    print(f"Accuracy:  {accuracy_score(y_true, predictions):.6f}")
    print(f"ROC-AUC:   {roc_auc_score(y_true, probabilities):.6f}")
    print(f"PR-AUC:    {average_precision_score(y_true, probabilities):.6f}")
    print(f"Log Loss:  {log_loss(y_true, probabilities):.6f}")
    print(f"Brier:     {brier_score_loss(y_true, probabilities):.6f}")


def evaluate_by_year(
    df: pd.DataFrame,
    probabilities: np.ndarray,
) -> None:

    result = df[
        ["trade_date", "symbol", TARGET]
    ].copy()

    result["probability"] = probabilities
    result["year"] = result["trade_date"].dt.year

    print("\nPERFORMANCE BY YEAR")
    print("-" * 80)

    rows = []

    for year, group in result.groupby("year"):

        y_true = group[TARGET]
        probabilities_year = group["probability"]

        if y_true.nunique() < 2:
            continue

        predictions = (probabilities_year >= 0.5).astype(int)

        rows.append(
            {
                "year": year,
                "rows": len(group),
                "positive_rate": y_true.mean(),
                "accuracy": accuracy_score(
                    y_true,
                    predictions,
                ),
                "roc_auc": roc_auc_score(
                    y_true,
                    probabilities_year,
                ),
                "pr_auc": average_precision_score(
                    y_true,
                    probabilities_year,
                ),
                "log_loss": log_loss(
                    y_true,
                    probabilities_year,
                ),
            }
        )

    yearly = pd.DataFrame(rows)

    print(
        yearly.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )


def evaluate_probability_buckets(
    df: pd.DataFrame,
    probabilities: np.ndarray,
) -> None:

    result = df[
        ["symbol", TARGET]
    ].copy()

    result["probability"] = probabilities

    result["probability_bucket"] = pd.cut(
        result["probability"],
        bins=[
            0.0,
            0.40,
            0.45,
            0.50,
            0.55,
            0.60,
            1.00,
        ],
        include_lowest=True,
    )

    bucket = (
        result.groupby(
            "probability_bucket",
            observed=False,
        )
        .agg(
            rows=(TARGET, "size"),
            actual_positive_rate=(TARGET, "mean"),
            average_probability=("probability", "mean"),
        )
        .reset_index()
    )

    print("\nPROBABILITY BUCKETS")
    print("-" * 80)

    print(
        bucket.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )


def evaluate_cross_sectional_ranking(
    df: pd.DataFrame,
    probabilities: np.ndarray,
) -> None:

    result = df[
        ["trade_date", "symbol", TARGET]
    ].copy()

    result["probability"] = probabilities

    print("\nCROSS-SECTIONAL RANKING")
    print("-" * 80)

    daily_rows = []

    for date, group in result.groupby("trade_date"):

        group = group.sort_values(
            "probability",
            ascending=False,
        )

        available = len(group)

        if available < 5:
            continue

        for top_n in [5, 10, 20]:

            n = min(top_n, available)

            top_group = group.head(n)

            daily_rows.append(
                {
                    "trade_date": date,
                    "top_n": n,
                    "actual_positive_rate": top_group[TARGET].mean(),
                    "average_probability": top_group["probability"].mean(),
                }
            )

    ranking = pd.DataFrame(daily_rows)

    summary = (
        ranking.groupby("top_n")
        .agg(
            trading_days=("trade_date", "nunique"),
            average_actual_positive_rate=(
                "actual_positive_rate",
                "mean",
            ),
            average_predicted_probability=(
                "average_probability",
                "mean",
            ),
        )
        .reset_index()
    )

    print(
        summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )


def evaluate_feature_importance(
    model,
) -> None:

    importance = (
        pd.Series(
            model.feature_importances_,
            index=FEATURES,
        )
        .sort_values(
            ascending=False,
        )
    )

    print("\nFEATURE IMPORTANCE")
    print("-" * 80)

    print(
        importance.to_string(
            float_format=lambda x: f"{x:.6f}",
        )
    )


def main() -> None:

    print("=" * 70)
    print("XGBoost Baseline Diagnostics")
    print("=" * 70)

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}"
        )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    print("\nLoading dataset...")
    df = pd.read_parquet(DATASET_PATH)

    print(f"Rows:   {len(df):,}")
    print(f"Stocks: {df['symbol'].nunique()}")

    print("\nLoading trained model...")
    model = joblib.load(MODEL_PATH)

    train_df, validation_df, test_df = create_time_split(df)

    print("\nGenerating predictions...")

    validation_probabilities = model.predict_proba(
        validation_df[FEATURES]
    )[:, 1]

    test_probabilities = model.predict_proba(
        test_df[FEATURES]
    )[:, 1]

    # ---------------------------------------------------------
    # VALIDATION
    # ---------------------------------------------------------

    evaluate_predictions(
        "VALIDATION",
        validation_df[TARGET],
        validation_probabilities,
    )

    # ---------------------------------------------------------
    # TEST
    # ---------------------------------------------------------

    evaluate_predictions(
        "TEST",
        test_df[TARGET],
        test_probabilities,
    )

    # ---------------------------------------------------------
    # YEARLY PERFORMANCE
    # ---------------------------------------------------------

    evaluate_by_year(
        test_df,
        test_probabilities,
    )

    # ---------------------------------------------------------
    # PROBABILITY DISTRIBUTION
    # ---------------------------------------------------------

    print("\nTEST PROBABILITY DISTRIBUTION")
    print("-" * 80)

    print(
        pd.Series(test_probabilities).describe(
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

    # ---------------------------------------------------------
    # PROBABILITY BUCKETS
    # ---------------------------------------------------------

    evaluate_probability_buckets(
        test_df,
        test_probabilities,
    )

    # ---------------------------------------------------------
    # CROSS-SECTIONAL RANKING
    # ---------------------------------------------------------

    evaluate_cross_sectional_ranking(
        test_df,
        test_probabilities,
    )

    # ---------------------------------------------------------
    # FEATURE IMPORTANCE
    # ---------------------------------------------------------

    evaluate_feature_importance(model)

    print("\nDIAGNOSTICS COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()