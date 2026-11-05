from pathlib import Path

import pandas as pd

from src.utils.mvp_config import modeling_dir


# ============================================================
# Configuration
# ============================================================

DATASET_FILE = (
    modeling_dir()
    / "day5"
    / "nse100_technical_model_dataset.parquet"
)


# ============================================================
# Target Columns
# ============================================================

TARGET_RETURN_COLUMN = "target_return_5d"
TARGET_DIRECTION_COLUMN = "target_direction_5d"
TARGET_EXCESS_RETURN_COLUMN = "target_excess_return_5d"


REQUIRED_TARGET_COLUMNS = {
    TARGET_RETURN_COLUMN,
    TARGET_DIRECTION_COLUMN,
    TARGET_EXCESS_RETURN_COLUMN,
}


# ============================================================
# Columns that identify a row or represent targets.
#
# These columns must NOT be used as model features.
# ============================================================

NON_FEATURE_COLUMNS = {
    "symbol",
    "trade_date",
    "feature_version",
    TARGET_RETURN_COLUMN,
    TARGET_DIRECTION_COLUMN,
    TARGET_EXCESS_RETURN_COLUMN,
}


# ============================================================
# Main
# ============================================================

def main() -> None:

    print("=" * 70)
    print("Inspect 5-Day Modeling Dataset")
    print("=" * 70)

    # --------------------------------------------------------
    # Validate dataset path
    # --------------------------------------------------------

    print()
    print(
        f"Dataset: {DATASET_FILE}"
    )

    if not DATASET_FILE.exists():

        raise FileNotFoundError(
            f"Dataset not found: {DATASET_FILE}"
        )

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    df = pd.read_parquet(
        DATASET_FILE
    )

    if df.empty:

        raise ValueError(
            "Modeling dataset is empty."
        )

    # --------------------------------------------------------
    # Basic dataset information
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
        f"{df['symbol'].nunique()}"
    )

    print(
        f"Date range              : "
        f"{df['trade_date'].min().date()} "
        f"-> "
        f"{df['trade_date'].max().date()}"
    )

    # --------------------------------------------------------
    # Validate required target columns
    # --------------------------------------------------------

    missing_target_columns = (
        REQUIRED_TARGET_COLUMNS
        - set(df.columns)
    )

    if missing_target_columns:

        raise ValueError(
            "Missing required target columns: "
            f"{sorted(missing_target_columns)}"
        )

    print()
    print(
        "Required 5-day target columns: OK"
    )

    # --------------------------------------------------------
    # Column list
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("All Columns")
    print("=" * 70)

    for i, column in enumerate(
        df.columns,
        start=1,
    ):

        dtype = str(
            df[column].dtype
        )

        print(
            f"{i:>2}. "
            f"{column:<35} "
            f"{dtype}"
        )

    # --------------------------------------------------------
    # Candidate model features
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

    for i, column in enumerate(
        model_features,
        start=1,
    ):

        print(
            f"{i:>2}. "
            f"{column:<35} "
            f"{df[column].dtype}"
        )

    # --------------------------------------------------------
    # Non-feature columns
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
            f"- {column:<35} "
            f"{df[column].dtype}"
        )

    # --------------------------------------------------------
    # Missing values in candidate features
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("Missing Values - Candidate Features")
    print("=" * 70)

    missing = (
        df[model_features]
        .isna()
        .sum()
        .sort_values(
            ascending=False
        )
    )

    missing = missing[
        missing > 0
    ]

    if missing.empty:

        print(
            "No missing values in "
            "candidate model features."
        )

    else:

        print(
            f"{'Feature':<35} "
            f"{'Missing':>10} "
            f"{'Percent':>10}"
        )

        print("-" * 58)

        for column, count in missing.items():

            percentage = (
                count
                / len(df)
            ) * 100

            print(
                f"{column:<35} "
                f"{count:>10,} "
                f"{percentage:>9.2f}%"
            )

    # --------------------------------------------------------
    # Data types
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("Candidate Feature Data Types")
    print("=" * 70)

    non_numeric_features = []

    for column in model_features:

        if not pd.api.types.is_numeric_dtype(
            df[column]
        ):

            non_numeric_features.append(
                column
            )

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
    # Target missing values
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("Target Missing Values")
    print("=" * 70)

    target_columns = [
        TARGET_RETURN_COLUMN,
        TARGET_DIRECTION_COLUMN,
        TARGET_EXCESS_RETURN_COLUMN,
    ]

    target_missing = (
        df[target_columns]
        .isna()
        .sum()
    )

    for column, count in (
        target_missing.items()
    ):

        percentage = (
            count
            / len(df)
        ) * 100

        print(
            f"{column:<35} "
            f"{count:>10,} "
            f"({percentage:>6.2f}%)"
        )

    # --------------------------------------------------------
    # Target distribution
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("5-Day Target Direction Distribution")
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
            "No valid direction targets found."
        )

    else:

        for direction, count in (
            direction_counts.items()
        ):

            percentage = (
                count
                / total_targets
            ) * 100

            if direction == 0:

                label = (
                    "DOWN / NOT POSITIVE"
                )

            elif direction == 1:

                label = (
                    "UP / POSITIVE"
                )

            else:

                label = "UNKNOWN"

            print(
                f"{int(direction)} = "
                f"{label:<20} "
                f"{count:>10,} "
                f"({percentage:>6.2f}%)"
            )

    # --------------------------------------------------------
    # Target return statistics
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("5-Day Target Return Statistics")
    print("=" * 70)

    print(
        df[TARGET_RETURN_COLUMN]
        .describe()
    )

    # --------------------------------------------------------
    # Excess return statistics
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("5-Day Excess Return Statistics")
    print("=" * 70)

    print(
        df[TARGET_EXCESS_RETURN_COLUMN]
        .describe()
    )

    # --------------------------------------------------------
    # Candidate feature statistics
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("Candidate Feature Statistics")
    print("=" * 70)

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

    # --------------------------------------------------------
    # Target leakage protection check
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("Target Leakage Check")
    print("=" * 70)

    leaked_targets = [
        column
        for column in target_columns
        if column in model_features
    ]

    if leaked_targets:

        print(
            "WARNING: Target columns detected "
            "as model features:"
        )

        for column in leaked_targets:

            print(
                f"- {column}"
            )

        raise ValueError(
            "Target leakage detected."
        )

    print(
        "No target columns are included "
        "in candidate model features."
    )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("Inspection Summary")
    print("=" * 70)

    print(
        f"Prediction horizon       : "
        f"5 trading days"
    )

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
        f"{len(missing)}"
    )

    print(
        f"Target columns           : "
        f"{len(target_columns)}"
    )

    print()
    print(
        "Inspection completed successfully."
    )


if __name__ == "__main__":
    main()