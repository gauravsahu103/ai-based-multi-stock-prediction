from pathlib import Path

import pandas as pd

from src.utils.mvp_config import modeling_dir


# ============================================================
# Project Paths
# ============================================================

DATASET_FILE = (
    modeling_dir()
    / "day1"
    / "nse100_technical_model_dataset.parquet"
)


# ============================================================
# Day-1 Target Columns
# ============================================================

TARGET_RETURN_COLUMN = "target_return_1d"
TARGET_DIRECTION_COLUMN = "target_direction_1d"

TARGET_COLUMNS = {
    TARGET_RETURN_COLUMN,
    TARGET_DIRECTION_COLUMN,
}


# ============================================================
# Columns That Must NOT Be Model Features
# ============================================================

NON_FEATURE_COLUMNS = {
    "symbol",
    "trade_date",
    "feature_version",
    *TARGET_COLUMNS,
}


# ============================================================
# Validation
# ============================================================

def validate_required_columns(
    df: pd.DataFrame,
) -> None:

    required_columns = {
        "symbol",
        "trade_date",
        *TARGET_COLUMNS,
    }

    missing_columns = (
        required_columns
        - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            "Required columns are missing from "
            f"the Day-1 modeling dataset: "
            f"{sorted(missing_columns)}"
        )


# ============================================================
# Main
# ============================================================

def main() -> None:

    print("=" * 70)
    print("Inspect Day-1 Modeling Dataset")
    print("=" * 70)

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    if not DATASET_FILE.exists():
        raise FileNotFoundError(
            f"Day-1 modeling dataset not found:\n"
            f"{DATASET_FILE}"
        )

    df = pd.read_parquet(
        DATASET_FILE
    )

    if df.empty:
        raise ValueError(
            "Day-1 modeling dataset is empty."
        )

    validate_required_columns(df)

    # --------------------------------------------------------
    # Dataset Overview
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("Dataset Overview")
    print("=" * 70)

    print(
        f"Rows                    : "
        f"{len(df):,}"
    )

    print(
        f"Columns                 : "
        f"{len(df.columns)}"
    )

    print(
        f"Stocks                  : "
        f"{df['symbol'].nunique():,}"
    )

    print(
        f"Date range              : "
        f"{df['trade_date'].min().date()} "
        f"-> "
        f"{df['trade_date'].max().date()}"
    )

    print(
        f"Dataset file            : "
        f"{DATASET_FILE}"
    )

    # --------------------------------------------------------
    # All Columns
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("All Columns")
    print("=" * 70)

    for index, column in enumerate(
        df.columns,
        start=1,
    ):
        print(
            f"{index:>2}. "
            f"{column:<40} "
            f"{df[column].dtype}"
        )

    # --------------------------------------------------------
    # Candidate Model Features
    # --------------------------------------------------------

    model_features = [
        column
        for column in df.columns
        if column not in NON_FEATURE_COLUMNS
    ]

    print()
    print("=" * 70)
    print("Candidate Model Features")
    print("=" * 70)

    print(
        f"Candidate feature count : "
        f"{len(model_features)}"
    )

    for index, column in enumerate(
        model_features,
        start=1,
    ):
        print(
            f"{index:>2}. "
            f"{column:<40} "
            f"{df[column].dtype}"
        )

    # --------------------------------------------------------
    # Excluded Columns
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("Excluded Columns")
    print("=" * 70)

    excluded_present = [
        column
        for column in df.columns
        if column in NON_FEATURE_COLUMNS
    ]

    for column in excluded_present:
        print(
            f"- {column:<40} "
            f"{df[column].dtype}"
        )

    # --------------------------------------------------------
    # Target Leakage Check
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("Target Leakage Check")
    print("=" * 70)

    leaked_targets = [
        column
        for column in TARGET_COLUMNS
        if column in model_features
    ]

    if leaked_targets:
        raise ValueError(
            "TARGET LEAKAGE DETECTED. "
            "The following target columns are "
            f"present as model features: "
            f"{sorted(leaked_targets)}"
        )

    print(
        "PASS - No Day-1 target columns are "
        "included as model features."
    )

    # --------------------------------------------------------
    # Missing Feature Values
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("Missing Feature Values")
    print("=" * 70)

    feature_missing = (
        df[model_features]
        .isna()
        .sum()
        .sort_values(
            ascending=False
        )
    )

    feature_missing = feature_missing[
        feature_missing > 0
    ]

    if feature_missing.empty:

        print(
            "No missing values in "
            "candidate model features."
        )

    else:

        print(
            f"{'Feature':<40}"
            f"{'Missing':>12}"
            f"{'Percent':>12}"
        )

        print("-" * 64)

        for column, count in (
            feature_missing.items()
        ):

            percentage = (
                count
                / len(df)
                * 100
            )

            print(
                f"{column:<40}"
                f"{count:>12,}"
                f"{percentage:>11.2f}%"
            )

    # --------------------------------------------------------
    # Target Missing Values
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("Target Missing Values")
    print("=" * 70)

    target_missing = (
        df[list(TARGET_COLUMNS)]
        .isna()
        .sum()
    )

    print(
        target_missing.to_string()
    )

    # --------------------------------------------------------
    # Candidate Feature Data Types
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("Candidate Feature Data Types")
    print("=" * 70)

    non_numeric_features = [
        column
        for column in model_features
        if not pd.api.types.is_numeric_dtype(
            df[column]
        )
    ]

    if not non_numeric_features:

        print(
            "All candidate model features "
            "are numeric."
        )

    else:

        print(
            "Non-numeric candidate features:"
        )

        for column in non_numeric_features:
            print(
                f"- {column}: "
                f"{df[column].dtype}"
            )

    # --------------------------------------------------------
    # Direction Distribution
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("1-Day Direction Target Distribution")
    print("=" * 70)

    direction_counts = (
        df[TARGET_DIRECTION_COLUMN]
        .dropna()
        .value_counts()
        .sort_index()
    )

    total_targets = (
        direction_counts.sum()
    )

    if total_targets == 0:

        print(
            "No valid Day-1 direction targets."
        )

    else:

        for direction, count in (
            direction_counts.items()
        ):

            percentage = (
                count
                / total_targets
                * 100
            )

            if direction == 0:
                label = "DOWN / NOT POSITIVE"
            elif direction == 1:
                label = "UP / POSITIVE"
            else:
                label = "UNKNOWN"

            print(
                f"{int(direction)} = "
                f"{label:<20} "
                f"{count:>10,} "
                f"({percentage:>6.2f}%)"
            )

    # --------------------------------------------------------
    # 1-Day Return Statistics
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("1-Day Target Return Statistics")
    print("=" * 70)

    print(
        df[
            TARGET_RETURN_COLUMN
        ].describe()
    )

    # --------------------------------------------------------
    # Candidate Feature Statistics
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("Candidate Feature Statistics")
    print("=" * 70)

    if model_features:

        statistics = (
            df[model_features]
            .describe()
            .T
        )

        statistics["missing"] = (
            df[model_features]
            .isna()
            .sum()
        )

        statistics["missing_pct"] = (
            statistics["missing"]
            / len(df)
            * 100
        )

        print(
            statistics[
                [
                    "count",
                    "mean",
                    "std",
                    "min",
                    "max",
                    "missing",
                    "missing_pct",
                ]
            ].to_string()
        )

    else:

        print(
            "No candidate model features found."
        )

    # --------------------------------------------------------
    # Final Summary
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("Inspection Summary")
    print("=" * 70)

    print(
        f"Total dataset rows       : "
        f"{len(df):,}"
    )

    print(
        f"Total columns            : "
        f"{len(df.columns)}"
    )

    print(
        f"Candidate model features : "
        f"{len(model_features)}"
    )

    print(
        f"Non-numeric features     : "
        f"{len(non_numeric_features)}"
    )

    print(
        f"Features with NaN        : "
        f"{len(feature_missing)}"
    )

    print(
        f"Target columns           : "
        f"{len(TARGET_COLUMNS)}"
    )

    print()
    print(
        "Day-1 modeling dataset "
        "inspection completed successfully."
    )


if __name__ == "__main__":
    main()