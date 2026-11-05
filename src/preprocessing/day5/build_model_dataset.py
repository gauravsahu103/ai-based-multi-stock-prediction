from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.mvp_config import (
    features_technical_dir,
    modeling_dir,
    targets_day_dir,
)


# ============================================================
# Configuration
# ============================================================

FEATURE_DIR = features_technical_dir()
TARGET_DIR = targets_day_dir(5)
OUTPUT_DIR = modeling_dir() / "day5"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

HORIZON = 5

TARGET_RETURN_COLUMN = f"target_return_{HORIZON}d"
TARGET_DIRECTION_COLUMN = f"target_direction_{HORIZON}d"
TARGET_EXCESS_RETURN_COLUMN = f"target_excess_return_{HORIZON}d"

OUTPUT_FILE = (
    OUTPUT_DIR / "nse100_technical_model_dataset.parquet"
)


# ============================================================
# Build one stock dataset
# ============================================================

def build_stock_dataset(
    feature_file: Path,
    target_file: Path,
) -> pd.DataFrame:

    features = pd.read_parquet(feature_file)
    targets = pd.read_parquet(target_file)

    symbol = feature_file.stem.replace(
        "_features",
        "",
    )

    # --------------------------------------------------------
    # Validate required columns
    # --------------------------------------------------------

    required_features = {
        "symbol",
        "trade_date",
        "close",
    }

    required_targets = {
        "symbol",
        "trade_date",
        TARGET_RETURN_COLUMN,
        TARGET_DIRECTION_COLUMN,
        TARGET_EXCESS_RETURN_COLUMN,
    }

    missing_features = (
        required_features - set(features.columns)
    )

    missing_targets = (
        required_targets - set(targets.columns)
    )

    if missing_features:
        raise ValueError(
            f"{symbol}: missing feature columns: "
            f"{sorted(missing_features)}"
        )

    if missing_targets:
        raise ValueError(
            f"{symbol}: missing target columns: "
            f"{sorted(missing_targets)}"
        )

    # --------------------------------------------------------
    # Date normalization
    # --------------------------------------------------------

    features["trade_date"] = pd.to_datetime(
        features["trade_date"]
    )

    targets["trade_date"] = pd.to_datetime(
        targets["trade_date"]
    )

    # --------------------------------------------------------
    # Validate feature dates
    # --------------------------------------------------------

    if features["trade_date"].duplicated().any():
        raise ValueError(
            f"{symbol}: duplicate feature dates"
        )

    if targets["trade_date"].duplicated().any():
        raise ValueError(
            f"{symbol}: duplicate target dates"
        )

    # --------------------------------------------------------
    # Validate symbols
    # --------------------------------------------------------

    if features["symbol"].nunique() != 1:
        raise ValueError(
            f"{symbol}: feature file contains multiple symbols"
        )

    if targets["symbol"].nunique() != 1:
        raise ValueError(
            f"{symbol}: target file contains multiple symbols"
        )

    if features["symbol"].iloc[0] != symbol:
        raise ValueError(
            f"{symbol}: feature symbol mismatch"
        )

    if targets["symbol"].iloc[0] != symbol:
        raise ValueError(
            f"{symbol}: target symbol mismatch"
        )

    # --------------------------------------------------------
    # Select target columns
    # --------------------------------------------------------

    targets = targets[
        [
            "symbol",
            "trade_date",
            TARGET_RETURN_COLUMN,
            TARGET_DIRECTION_COLUMN,
            TARGET_EXCESS_RETURN_COLUMN,
        ]
    ].copy()

    # --------------------------------------------------------
    # Join on stock + date
    # --------------------------------------------------------

    df = features.merge(
        targets,
        on=["symbol", "trade_date"],
        how="left",
        validate="one_to_one",
    )

    # --------------------------------------------------------
    # Sort
    # --------------------------------------------------------

    df = (
        df.sort_values(
            ["symbol", "trade_date"]
        )
        .reset_index(drop=True)
    )

    return df


# ============================================================
# Validate finite numeric values
# ============================================================

def validate_numeric_values(
    dataset: pd.DataFrame,
) -> None:
    """
    Check numeric columns for +/- infinity.

    NaN values are allowed because technical indicators
    naturally contain warm-up NaNs at the beginning of
    a stock's history.
    """

    numeric_columns = dataset.select_dtypes(
        include=[np.number]
    ).columns

    infinite_values = {}

    for column in numeric_columns:

        series = dataset[column]

        # Convert only this individual column to a numeric
        # NumPy representation.
        values = pd.to_numeric(
            series,
            errors="coerce",
        ).to_numpy(
            dtype="float64",
            na_value=np.nan,
        )

        finite_mask = np.isfinite(values)

        # NaN is intentionally allowed.
        invalid_mask = (
            ~finite_mask
            & ~np.isnan(values)
        )

        invalid_count = int(
            invalid_mask.sum()
        )

        if invalid_count > 0:
            infinite_values[column] = invalid_count

    total_invalid = sum(
        infinite_values.values()
    )

    print(
        f"Infinite numeric values : "
        f"{total_invalid}"
    )

    if total_invalid > 0:
        raise ValueError(
            "Infinite numeric values found: "
            f"{infinite_values}"
        )


# ============================================================
# Main
# ============================================================

def main() -> None:

    print("=" * 70)
    print("Build Modeling Dataset")
    print("=" * 70)

    # --------------------------------------------------------
    # Directory validation
    # --------------------------------------------------------

    if not FEATURE_DIR.exists():
        raise FileNotFoundError(
            f"Feature directory not found: {FEATURE_DIR}"
        )

    if not TARGET_DIR.exists():
        raise FileNotFoundError(
            f"Target directory not found: {TARGET_DIR}"
        )

    # --------------------------------------------------------
    # Find feature files
    # --------------------------------------------------------

    feature_files = sorted(
        FEATURE_DIR.glob("*_features.parquet")
    )

    print(
        f"Feature files found: {len(feature_files)}"
    )

    if len(feature_files) != 100:
        raise ValueError(
            f"Expected 100 feature files, "
            f"found {len(feature_files)}"
        )

    all_data = []

    successful = 0
    failed = 0

    # --------------------------------------------------------
    # Process stocks
    # --------------------------------------------------------

    for feature_file in feature_files:

        symbol = feature_file.stem.replace(
            "_features",
            "",
        )

        target_file = (
            TARGET_DIR / f"{symbol}.parquet"
        )

        if not target_file.exists():

            print(
                f"[FAILED] {symbol:<15} "
                f"target file not found"
            )

            failed += 1
            continue

        try:

            df = build_stock_dataset(
                feature_file,
                target_file,
            )

            all_data.append(df)

            successful += 1

            print(
                f"[OK] {symbol:<15} "
                f"{len(df):>6,} rows"
            )

        except Exception as exc:

            failed += 1

            print(
                f"[FAILED] {symbol:<15} "
                f"{type(exc).__name__}: {exc}"
            )

    # --------------------------------------------------------
    # Stop on failures
    # --------------------------------------------------------

    if failed > 0:
        raise RuntimeError(
            f"Dataset construction failed for "
            f"{failed} stock(s)"
        )

    # --------------------------------------------------------
    # Combine all stocks
    # --------------------------------------------------------

    dataset = pd.concat(
        all_data,
        ignore_index=True,
    )

    dataset = (
        dataset
        .sort_values(
            ["trade_date", "symbol"]
        )
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Global uniqueness
    # --------------------------------------------------------

    duplicate_keys = dataset.duplicated(
        subset=["symbol", "trade_date"]
    )

    duplicate_count = int(
        duplicate_keys.sum()
    )

    if duplicate_count > 0:
        raise ValueError(
            f"Duplicate stock/date rows: "
            f"{duplicate_count}"
        )

    # --------------------------------------------------------
    # Target validation
    # --------------------------------------------------------

    valid_target = (
        dataset[TARGET_RETURN_COLUMN].notna()
        & dataset[TARGET_DIRECTION_COLUMN].notna()
        & dataset[TARGET_EXCESS_RETURN_COLUMN].notna()
    )

    valid_rows = int(
        valid_target.sum()
    )

    missing_target_rows = (
        len(dataset) - valid_rows
    )

    print()
    print("=" * 70)
    print("Dataset Validation")
    print("=" * 70)

    print(
        f"Total rows              : {len(dataset):,}"
    )

    print(
        f"Valid target rows       : {valid_rows:,}"
    )

    print(
        f"Missing target rows     : "
        f"{missing_target_rows:,}"
    )

    print(
        f"Duplicate stock/date    : "
        f"{duplicate_count}"
    )

    # --------------------------------------------------------
    # Target consistency
    # --------------------------------------------------------

    target_return = dataset[
        TARGET_RETURN_COLUMN
    ]

    target_direction = dataset[
        TARGET_DIRECTION_COLUMN
    ]

    # Only validate rows where both target values exist.
    # The final 5 rows of every stock intentionally have
    # missing future targets.

    direction_check = (
        target_return.notna()
        & target_direction.notna()
    )

    expected_direction = (
        target_return.loc[direction_check] > 0
    ).astype("int64")

    actual_direction = (
        target_direction.loc[direction_check]
        .astype("int64")
    )

    inconsistent_direction = (
        actual_direction != expected_direction
    )

    inconsistent_count = int(
        inconsistent_direction.sum()
    )

    print(
        f"Target direction errors : "
        f"{inconsistent_count}"
    )

    if inconsistent_count > 0:
        raise ValueError(
            "Target direction is inconsistent "
            "with target return"
        )

    # --------------------------------------------------------
    # Validate target return values
    # --------------------------------------------------------

    valid_returns = target_return.loc[
        direction_check
    ]

    valid_return_values = pd.to_numeric(
        valid_returns,
        errors="coerce",
    ).to_numpy(
        dtype="float64",
        na_value=np.nan,
    )

    invalid_return_mask = (
        ~np.isfinite(valid_return_values)
    )

    invalid_return_count = int(
        invalid_return_mask.sum()
    )

    print(
        f"Invalid target returns : "
        f"{invalid_return_count}"
    )

    if invalid_return_count > 0:
        raise ValueError(
            "Target return contains "
            "invalid numeric values"
        )

    # --------------------------------------------------------
    # Validate target direction values
    # --------------------------------------------------------

    valid_directions = target_direction.loc[
        direction_check
    ]

    invalid_direction_values = (
        ~valid_directions.isin([0, 1])
    )

    invalid_direction_count = int(
        invalid_direction_values.sum()
    )

    print(
        f"Invalid direction values: "
        f"{invalid_direction_count}"
    )

    if invalid_direction_count > 0:
        raise ValueError(
            "Target direction contains values "
            "other than 0 or 1"
        )

    # --------------------------------------------------------
    # Date validation
    # --------------------------------------------------------

    print(
        f"Date range               : "
        f"{dataset['trade_date'].min().date()} "
        f"-> "
        f"{dataset['trade_date'].max().date()}"
    )

    # --------------------------------------------------------
    # Stock count
    # --------------------------------------------------------

    stock_count = dataset[
        "symbol"
    ].nunique()

    print(
        f"Unique stocks            : {stock_count}"
    )

    if stock_count != 100:
        raise ValueError(
            f"Expected 100 stocks, "
            f"found {stock_count}"
        )

    # --------------------------------------------------------
    # Leakage checks
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("Leakage Checks")
    print("=" * 70)

    target_columns = {
        TARGET_RETURN_COLUMN,
        TARGET_DIRECTION_COLUMN,
        TARGET_EXCESS_RETURN_COLUMN,
    }

    feature_columns = [
        column
        for column in dataset.columns
        if column not in target_columns
    ]

    accidental_target_features = (
        set(feature_columns)
        & target_columns
    )

    print(
        f"Target columns excluded : "
        f"{len(accidental_target_features) == 0}"
    )

    if accidental_target_features:
        raise ValueError(
            "Target columns found in feature set: "
            f"{sorted(accidental_target_features)}"
        )

    # --------------------------------------------------------
    # Feature version
    # --------------------------------------------------------

    if "feature_version" not in dataset.columns:
        raise ValueError(
            "feature_version column missing"
        )

    versions = (
        dataset["feature_version"]
        .dropna()
        .unique()
        .tolist()
    )

    print(
        f"Feature versions        : {versions}"
    )

    # --------------------------------------------------------
    # Infinite-value validation
    # --------------------------------------------------------

    validate_numeric_values(
        dataset
    )

    # --------------------------------------------------------
    # Feature NaN report
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("Feature Missing-Value Report")
    print("=" * 70)

    feature_nan_counts = (
        dataset[feature_columns]
        .isna()
        .sum()
    )

    feature_nan_counts = (
        feature_nan_counts[
            feature_nan_counts > 0
        ]
        .sort_values(
            ascending=False
        )
    )

    if feature_nan_counts.empty:

        print(
            "Feature NaN values       : 0"
        )

    else:

        print(
            "Features containing NaN:"
        )

        for column, count in (
            feature_nan_counts.items()
        ):

            percentage = (
                count / len(dataset)
            ) * 100

            print(
                f"  {column:<30} "
                f"{count:>7,} "
                f"({percentage:>6.2f}%)"
            )

    # --------------------------------------------------------
    # Remove rows without target
    # --------------------------------------------------------
    #
    # These are the final 5 trading days for each stock.
    # They cannot be used for supervised training because
    # T+5 is not available yet.
    #
    # --------------------------------------------------------

    model_dataset = dataset[
        valid_target
    ].copy()

    model_dataset = (
        model_dataset
        .sort_values(
            ["trade_date", "symbol"]
        )
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Final validation
    # --------------------------------------------------------

    if model_dataset.empty:
        raise ValueError(
            "Model dataset is empty"
        )

    if model_dataset[
        ["symbol", "trade_date"]
    ].duplicated().any():

        raise ValueError(
            "Duplicate rows in final model dataset"
        )

    # --------------------------------------------------------
    # Validate final target completeness
    # --------------------------------------------------------

    if model_dataset[
        TARGET_RETURN_COLUMN
    ].isna().any():

        raise ValueError(
            f"Missing {TARGET_RETURN_COLUMN} "
            "in final model dataset"
        )

    if model_dataset[
        TARGET_DIRECTION_COLUMN
    ].isna().any():

        raise ValueError(
            f"Missing {TARGET_DIRECTION_COLUMN} "
            "in final model dataset"
        )

    if model_dataset[
        TARGET_EXCESS_RETURN_COLUMN
    ].isna().any():

        raise ValueError(
            f"Missing {TARGET_EXCESS_RETURN_COLUMN} "
            "in final model dataset"
        )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    model_dataset.to_parquet(
        OUTPUT_FILE,
        index=False,
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("Modeling Dataset Summary")
    print("=" * 70)

    print(
        f"Training-ready rows      : "
        f"{len(model_dataset):,}"
    )

    print(
        f"Stocks                   : "
        f"{model_dataset['symbol'].nunique()}"
    )

    print(
        f"Date range               : "
        f"{model_dataset['trade_date'].min().date()} "
        f"-> "
        f"{model_dataset['trade_date'].max().date()}"
    )

    print(
        f"Feature columns          : "
        f"{len(feature_columns)}"
    )

    print(
        f"Output                   : "
        f"{OUTPUT_FILE}"
    )

    print()
    print(
        "Dataset build completed successfully."
    )


if __name__ == "__main__":
    main()