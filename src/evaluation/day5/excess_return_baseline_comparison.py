"""
XGBoost 5D Excess Return Baseline Comparison

Purpose
-------
Compare the trained XGBoost 5-day excess-return model against simple
non-ML baselines on the untouched test period.

Target
------
target_excess_return_5d

Baselines
---------
1. Zero baseline
   Predict 0% excess return for every stock.

2. Historical-mean baseline
   Predict the mean excess return observed in the training set.

3. Cross-sectional daily mean baseline
   For each test trading day, predict the mean excess return of the
   test universe for that day.

The ML model is evaluated on exactly the same usable test rows.

Important
---------
The time split is imported from src.models.time_split and is expected
to return:

    train_df, validation_df, test_df

Run
---
python -m src.evaluation.excess_return_baseline_comparison
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.models.day5.time_split import create_time_split
from src.utils.mvp_config import modeling_dir, models_dir, predictions_dir


# ======================================================================
# CONFIGURATION
# ======================================================================

DATASET_PATH = (
    modeling_dir()
    / "day5"
    / "nse100_technical_model_dataset.parquet"
)

MODEL_PATH = models_dir() / "xgboost_excess_return_5d_xgb_v1.joblib"

OUTPUT_PATH = predictions_dir() / "xgboost_excess_return_5d_baseline_comparison.parquet"

TARGET_COLUMN = "target_excess_return_5d"


# ======================================================================
# HELPERS
# ======================================================================

def safe_float(value) -> float:
    """Convert a value to float safely."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict:
    """
    Calculate standard regression metrics.

    Returns
    -------
    dict
        MAE, RMSE, R2 and Pearson correlation.
    """

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mask = (
        np.isfinite(y_true)
        & np.isfinite(y_pred)
    )

    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if len(y_true) == 0:
        return {
            "rows": 0,
            "mae": np.nan,
            "rmse": np.nan,
            "r2": np.nan,
            "correlation": np.nan,
        }

    mae = mean_absolute_error(y_true, y_pred)

    rmse = np.sqrt(
        mean_squared_error(y_true, y_pred)
    )

    if len(y_true) > 1 and np.std(y_true) > 0:
        r2 = r2_score(y_true, y_pred)
        correlation = np.corrcoef(
            y_true,
            y_pred,
        )[0, 1]
    else:
        r2 = np.nan
        correlation = np.nan

    return {
        "rows": int(len(y_true)),
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
        "correlation": float(correlation),
    }


def print_metrics(
    name: str,
    metrics: dict,
) -> None:
    """Print regression metrics."""

    print()
    print(name)
    print("-" * 60)
    print(f"Rows        : {metrics['rows']:,}")
    print(f"MAE         : {metrics['mae']:.6f}")
    print(f"RMSE        : {metrics['rmse']:.6f}")
    print(f"R2          : {metrics['r2']:.6f}")
    print(f"Correlation : {metrics['correlation']:.6f}")


def validate_required_columns(
    df: pd.DataFrame,
    required_columns: list[str],
    name: str,
) -> None:
    """Validate that required columns exist."""

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{name} is missing required columns: {missing}"
        )


# ======================================================================
# MAIN
# ======================================================================

def main() -> None:

    print("=" * 70)
    print("XGBoost 5D Excess Return Baseline Comparison")
    print("=" * 70)

    # ------------------------------------------------------------------
    # LOAD DATASET
    # ------------------------------------------------------------------

    print()
    print("Loading dataset...")
    print(DATASET_PATH)

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}"
        )

    df = pd.read_parquet(DATASET_PATH)

    print(f"Dataset rows:   {len(df):,}")
    print(
        f"Dataset stocks: "
        f"{df['symbol'].nunique():,}"
    )

    validate_required_columns(
        df,
        [
            "symbol",
            "trade_date",
            TARGET_COLUMN,
        ],
        "Dataset",
    )

    # Ensure deterministic ordering.
    df = (
        df
        .sort_values(
            ["symbol", "trade_date"]
        )
        .reset_index(drop=True)
    )

    # ------------------------------------------------------------------
    # TIME SPLIT
    # ------------------------------------------------------------------

    #
    # IMPORTANT:
    #
    # create_time_split() returns a tuple:
    #
    #     train_df, validation_df, test_df
    #
    # Do NOT use:
    #
    #     splits["train"]
    #
    # because that causes:
    #
    # TypeError:
    # tuple indices must be integers or slices, not str
    #
    # ------------------------------------------------------------------

    train_df, validation_df, test_df = create_time_split(df)

    print()
    print("TIME SPLIT VALIDATION: PASS")

    print()
    print("Train")
    print(
        f"  Rows:   {len(train_df):,}"
    )
    print(
        f"  Stocks: {train_df['symbol'].nunique():,}"
    )
    print(
        f"  Dates:  "
        f"{train_df['trade_date'].min().date()} -> "
        f"{train_df['trade_date'].max().date()}"
    )

    print()
    print("Validation")
    print(
        f"  Rows:   {len(validation_df):,}"
    )
    print(
        f"  Stocks: {validation_df['symbol'].nunique():,}"
    )
    print(
        f"  Dates:  "
        f"{validation_df['trade_date'].min().date()} -> "
        f"{validation_df['trade_date'].max().date()}"
    )

    print()
    print("Test")
    print(
        f"  Rows:   {len(test_df):,}"
    )
    print(
        f"  Stocks: {test_df['symbol'].nunique():,}"
    )
    print(
        f"  Dates:  "
        f"{test_df['trade_date'].min().date()} -> "
        f"{test_df['trade_date'].max().date()}"
    )

    # ------------------------------------------------------------------
    # TARGET VALIDATION
    # ------------------------------------------------------------------

    train_df = train_df.copy()
    validation_df = validation_df.copy()
    test_df = test_df.copy()

    train_df[TARGET_COLUMN] = pd.to_numeric(
        train_df[TARGET_COLUMN],
        errors="coerce",
    )

    validation_df[TARGET_COLUMN] = pd.to_numeric(
        validation_df[TARGET_COLUMN],
        errors="coerce",
    )

    test_df[TARGET_COLUMN] = pd.to_numeric(
        test_df[TARGET_COLUMN],
        errors="coerce",
    )

    train_valid = (
        train_df[TARGET_COLUMN]
        .notna()
        & np.isfinite(
            train_df[TARGET_COLUMN].to_numpy(
                dtype=float
            )
        )
    )

    validation_valid = (
        validation_df[TARGET_COLUMN]
        .notna()
        & np.isfinite(
            validation_df[TARGET_COLUMN].to_numpy(
                dtype=float
            )
        )
    )

    test_valid = (
        test_df[TARGET_COLUMN]
        .notna()
        & np.isfinite(
            test_df[TARGET_COLUMN].to_numpy(
                dtype=float
            )
        )
    )

    train_model = train_df.loc[
        train_valid
    ].copy()

    validation_model = validation_df.loc[
        validation_valid
    ].copy()

    test_model = test_df.loc[
        test_valid
    ].copy()

    print()
    print("USABLE TARGET ROWS")
    print("-" * 70)
    print(
        f"Train      : {len(train_model):,}"
    )
    print(
        f"Validation : {len(validation_model):,}"
    )
    print(
        f"Test       : {len(test_model):,}"
    )

    # ------------------------------------------------------------------
    # TRAINING TARGET STATISTICS
    # ------------------------------------------------------------------

    print()
    print("TRAINING TARGET STATISTICS")
    print("-" * 70)

    print(
        train_model[TARGET_COLUMN]
        .describe()
        .to_string()
    )

    # ------------------------------------------------------------------
    # LOAD MODEL
    # ------------------------------------------------------------------

    print()
    print("Loading trained model...")
    print(MODEL_PATH)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    model = joblib.load(MODEL_PATH)

    # ------------------------------------------------------------------
    # FEATURE DISCOVERY
    # ------------------------------------------------------------------

    if hasattr(model, "feature_names_in_"):
        feature_columns = list(
            model.feature_names_in_
        )
    elif hasattr(model, "get_booster"):
        feature_columns = (
            model.get_booster()
            .feature_names
        )
    else:
        raise ValueError(
            "Unable to determine model feature names."
        )

    if not feature_columns:
        raise ValueError(
            "Model contains no feature names."
        )

    print()
    print("MODEL FEATURES")
    print("-" * 70)
    print(
        f"Feature count: "
        f"{len(feature_columns)}"
    )

    missing_features = [
        column
        for column in feature_columns
        if column not in test_model.columns
    ]

    if missing_features:
        raise ValueError(
            "Test dataset is missing model "
            f"features: {missing_features}"
        )

    # ------------------------------------------------------------------
    # PREPARE TEST FEATURES
    # ------------------------------------------------------------------

    X_test = test_model[
        feature_columns
    ].copy()

    # Make sure feature values are numeric.
    for column in feature_columns:
        X_test[column] = pd.to_numeric(
            X_test[column],
            errors="coerce",
        )

    # XGBoost can handle NaN, but infinite values
    # should not be passed through.
    X_test = X_test.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    # ------------------------------------------------------------------
    # GENERATE MODEL PREDICTIONS
    # ------------------------------------------------------------------

    print()
    print("Generating XGBoost predictions...")

    model_predictions = model.predict(
        X_test
    )

    model_predictions = np.asarray(
        model_predictions,
        dtype=float,
    )

    # ------------------------------------------------------------------
    # BUILD EVALUATION DATAFRAME
    # ------------------------------------------------------------------

    evaluation_df = test_model[
        [
            "symbol",
            "trade_date",
            TARGET_COLUMN,
        ]
    ].copy()

    evaluation_df[
        "model_prediction"
    ] = model_predictions

    # ------------------------------------------------------------------
    # BASELINE 1: ZERO
    # ------------------------------------------------------------------

    #
    # Predict:
    #
    #     excess return = 0
    #
    # This means:
    #
    # "I expect the stock to perform exactly
    #  like NIFTY over the next five days."
    #
    evaluation_df[
        "zero_baseline_prediction"
    ] = 0.0

    # ------------------------------------------------------------------
    # BASELINE 2: TRAINING MEAN
    # ------------------------------------------------------------------

    training_mean = float(
        train_model[TARGET_COLUMN].mean()
    )

    print()
    print("BASELINE PARAMETERS")
    print("-" * 70)
    print(
        f"Training mean excess return : "
        f"{training_mean:.8f}"
    )

    evaluation_df[
        "historical_mean_prediction"
    ] = training_mean

    # ------------------------------------------------------------------
    # BASELINE 3: DAILY CROSS-SECTIONAL MEAN
    # ------------------------------------------------------------------

    #
    # For every test date:
    #
    #     prediction =
    #     average actual excess return
    #     across all stocks on that date
    #
    # This is primarily a descriptive benchmark.
    #
    daily_mean = (
        evaluation_df
        .groupby("trade_date")[
            TARGET_COLUMN
        ]
        .transform("mean")
    )

    evaluation_df[
        "daily_mean_baseline_prediction"
    ] = daily_mean

    # ------------------------------------------------------------------
    # METRICS
    # ------------------------------------------------------------------

    y_true = (
        evaluation_df[TARGET_COLUMN]
        .to_numpy(dtype=float)
    )

    y_model = (
        evaluation_df["model_prediction"]
        .to_numpy(dtype=float)
    )

    y_zero = (
        evaluation_df[
            "zero_baseline_prediction"
        ]
        .to_numpy(dtype=float)
    )

    y_historical_mean = (
        evaluation_df[
            "historical_mean_prediction"
        ]
        .to_numpy(dtype=float)
    )

    y_daily_mean = (
        evaluation_df[
            "daily_mean_baseline_prediction"
        ]
        .to_numpy(dtype=float)
    )

    model_metrics = regression_metrics(
        y_true,
        y_model,
    )

    zero_metrics = regression_metrics(
        y_true,
        y_zero,
    )

    historical_mean_metrics = regression_metrics(
        y_true,
        y_historical_mean,
    )

    daily_mean_metrics = regression_metrics(
        y_true,
        y_daily_mean,
    )

    # ------------------------------------------------------------------
    # PRINT RESULTS
    # ------------------------------------------------------------------

    print()
    print("=" * 70)
    print("TEST PERFORMANCE")
    print("=" * 70)

    print_metrics(
        "XGBoost",
        model_metrics,
    )

    print_metrics(
        "Zero Baseline",
        zero_metrics,
    )

    print_metrics(
        "Historical Mean Baseline",
        historical_mean_metrics,
    )

    print_metrics(
        "Daily Cross-Sectional Mean Baseline",
        daily_mean_metrics,
    )

    # ------------------------------------------------------------------
    # COMPARISON TABLE
    # ------------------------------------------------------------------

    comparison = pd.DataFrame(
        [
            {
                "model": "XGBoost",
                **model_metrics,
            },
            {
                "model": "Zero Baseline",
                **zero_metrics,
            },
            {
                "model": "Historical Mean",
                **historical_mean_metrics,
            },
            {
                "model": "Daily Cross-Sectional Mean",
                **daily_mean_metrics,
            },
        ]
    )

    print()
    print("=" * 70)
    print("BASELINE COMPARISON")
    print("=" * 70)

    print(
        comparison.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    # ------------------------------------------------------------------
    # MODEL IMPROVEMENT VS ZERO BASELINE
    # ------------------------------------------------------------------

    print()
    print("=" * 70)
    print("XGBOOST VS ZERO BASELINE")
    print("=" * 70)

    mae_improvement = (
        zero_metrics["mae"]
        - model_metrics["mae"]
    )

    rmse_improvement = (
        zero_metrics["rmse"]
        - model_metrics["rmse"]
    )

    print(
        f"MAE improvement  : "
        f"{mae_improvement:.6f}"
    )

    print(
        f"RMSE improvement : "
        f"{rmse_improvement:.6f}"
    )

    if zero_metrics["mae"] != 0:
        mae_improvement_pct = (
            mae_improvement
            / zero_metrics["mae"]
            * 100.0
        )
    else:
        mae_improvement_pct = np.nan

    if zero_metrics["rmse"] != 0:
        rmse_improvement_pct = (
            rmse_improvement
            / zero_metrics["rmse"]
            * 100.0
        )
    else:
        rmse_improvement_pct = np.nan

    print(
        f"MAE improvement %  : "
        f"{mae_improvement_pct:.2f}%"
    )

    print(
        f"RMSE improvement % : "
        f"{rmse_improvement_pct:.2f}%"
    )

    # ------------------------------------------------------------------
    # CROSS-SECTIONAL RANKING CHECK
    # ------------------------------------------------------------------

    print()
    print("=" * 70)
    print("CROSS-SECTIONAL RANKING CHECK")
    print("=" * 70)

    ranking_records = []

    for trade_date, group in (
        evaluation_df
        .groupby("trade_date")
    ):

        group = group.dropna(
            subset=[
                TARGET_COLUMN,
                "model_prediction",
            ]
        )

        if len(group) < 2:
            continue

        actual = group[
            TARGET_COLUMN
        ].to_numpy(dtype=float)

        predicted = group[
            "model_prediction"
        ].to_numpy(dtype=float)

        if (
            np.std(actual) == 0
            or np.std(predicted) == 0
        ):
            continue

        actual_rank = pd.Series(
            actual
        ).rank(method="average")

        predicted_rank = pd.Series(
            predicted
        ).rank(method="average")

        correlation = np.corrcoef(
            actual_rank,
            predicted_rank,
        )[0, 1]

        ranking_records.append(
            {
                "trade_date": trade_date,
                "spearman": correlation,
            }
        )

    ranking_df = pd.DataFrame(
        ranking_records
    )

    if not ranking_df.empty:

        print(
            f"Trading days: "
            f"{len(ranking_df):,}"
        )

        print(
            f"Average Spearman correlation : "
            f"{ranking_df['spearman'].mean():.6f}"
        )

        print(
            f"Median Spearman correlation  : "
            f"{ranking_df['spearman'].median():.6f}"
        )

        print(
            f"Spearman > 0 days            : "
            f"{(
                ranking_df['spearman'] > 0
            ).mean() * 100:.2f}%"
        )

    else:

        print(
            "Unable to calculate cross-sectional "
            "ranking statistics."
        )

    # ------------------------------------------------------------------
    # TOP-N ANALYSIS
    # ------------------------------------------------------------------

    print()
    print("=" * 70)
    print("TOP-N CROSS-SECTIONAL PERFORMANCE")
    print("=" * 70)

    top_n_results = []

    for trade_date, group in (
        evaluation_df
        .groupby("trade_date")
    ):

        group = group.dropna(
            subset=[
                TARGET_COLUMN,
                "model_prediction",
            ]
        )

        if len(group) < 5:
            continue

        group = group.sort_values(
            "model_prediction",
            ascending=False,
        )

        universe_return = group[
            TARGET_COLUMN
        ].mean()

        row = {
            "trade_date": trade_date,
            "universe_actual": universe_return,
        }

        for top_n in [5, 10, 20]:

            if len(group) < top_n:
                row[
                    f"top_{top_n}_actual"
                ] = np.nan
                continue

            top_group = group.head(
                top_n
            )

            row[
                f"top_{top_n}_actual"
            ] = top_group[
                TARGET_COLUMN
            ].mean()

        top_n_results.append(row)

    top_n_df = pd.DataFrame(
        top_n_results
    )

    if not top_n_df.empty:

        for top_n in [5, 10, 20]:

            column = (
                f"top_{top_n}_actual"
            )

            if column not in top_n_df:
                continue

            top_value = (
                top_n_df[column]
                .mean()
            )

            universe_value = (
                top_n_df[
                    "universe_actual"
                ].mean()
            )

            spread = (
                top_value
                - universe_value
            )

            print()
            print(
                f"Top {top_n}"
            )
            print(
                f"  Actual excess return : "
                f"{top_value * 100:.6f}%"
            )
            print(
                f"  Universe             : "
                f"{universe_value * 100:.6f}%"
            )
            print(
                f"  Spread               : "
                f"{spread * 100:.6f}%"
            )

    # ------------------------------------------------------------------
    # SAVE DETAILED PREDICTIONS
    # ------------------------------------------------------------------

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    evaluation_df.to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    print()
    print("=" * 70)
    print("SAVED")
    print("=" * 70)
    print(
        f"Predictions and baselines:\n"
        f"{OUTPUT_PATH}"
    )

    print()
    print("=" * 70)
    print("BASELINE COMPARISON COMPLETE")
    print("=" * 70)


# ======================================================================
# ENTRY POINT
# ======================================================================

if __name__ == "__main__":
    main()