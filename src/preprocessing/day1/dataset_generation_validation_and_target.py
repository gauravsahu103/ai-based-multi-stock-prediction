"""
Dataset Validation and Modeling Dataset Assembly
=================================================

Purpose
-------
Create ONE reproducible Day-1 modeling dataset pipeline.

Pipeline
--------
1. Load Phase-1 technical feature files.
2. Load already-generated Day-1 target files.
3. Merge features + targets.
4. Validate structure.
5. Validate target direction.
6. Validate T -> T+1 target semantics.
7. Generate additional point-in-time input features.
8. Validate generated features.
9. Validate numeric values.
10. Run structural leakage checks.
11. Validate cross-sectional rank features.
12. Replace +/- infinity with NaN.
13. Keep rows with valid Day-1 targets.
14. Create the COMPLETE modeling dataset.
15. Split chronologically into:
       Train      70%
       Validation 15%
       Test       15%
16. Save all artifacts.

Important
---------
Additional features MUST be generated before splitting.

Therefore:

    final dataset
          |
          +---- train
          +---- validation
          +---- test

All three splits contain the same feature schema.

This script does NOT:
    - regenerate raw technical indicators
    - train a model
    - generate predictions
    - generate prediction rankings
    - generate future prediction outputs

Model outputs such as:

    predicted_return
    predicted_excess_return
    predicted_return_rank
    excess_return_rank

are generated later by the modeling/prediction pipeline.
"""

from __future__ import annotations

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

HORIZON = 1

EXPECTED_STOCK_COUNT = 95

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15


# ============================================================
# Directories
# ============================================================

FEATURE_DIR = (
    features_technical_dir(HORIZON)
    / "technical_v2"
)

TARGET_DIR = targets_day_dir(HORIZON)

OUTPUT_DIR = modeling_dir(HORIZON)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Output files
# ============================================================

OUTPUT_FILE = (
    OUTPUT_DIR
    / "nse100_technical_model_dataset.parquet"
)

TRAIN_FILE = (
    OUTPUT_DIR
    / "train.parquet"
)

VALIDATION_FILE = (
    OUTPUT_DIR
    / "validation.parquet"
)

TEST_FILE = (
    OUTPUT_DIR
    / "test.parquet"
)

DISTRIBUTION_FILE = (
    OUTPUT_DIR
    / "feature_distribution.parquet"
)

REPORT_FILE = (
    OUTPUT_DIR
    / "validation_report.txt"
)


# ============================================================
# Target columns
# ============================================================

TARGET_RETURN_COLUMN = (
    f"target_return_{HORIZON}d"
)

TARGET_DIRECTION_COLUMN = (
    f"target_direction_{HORIZON}d"
)


# ============================================================
# Baseline Phase-1 features
#
# These are validation expectations only.
#
# We DO NOT restrict Phase-2 modeling to these 24 columns.
# All valid numeric Phase-1 features remain available.
# ============================================================

REQUIRED_BASELINE_FEATURES = [
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


# ============================================================
# Additional input features
#
# These are generated BEFORE the dataset split.
# ============================================================

GENERATED_FEATURES = [
    "market_relative_return_1d",
    "market_relative_return_5d",
    "market_relative_return_20d",
    "stock_return_vol_adjusted",
    "return_acceleration_1d_5d",
    "trend_strength_score",
    "momentum_consistency_score",
    "market_alignment_score",
]


# ============================================================
# Identifier / metadata columns
# ============================================================

IDENTIFIER_COLUMNS = {
    "symbol",
    "trade_date",
    "feature_version",
    "sector",
}


# ============================================================
# Numeric helper
# ============================================================

def numeric(
    df: pd.DataFrame,
    column: str,
) -> pd.Series:
    """Return a numeric representation of a column."""

    return pd.to_numeric(
        df[column],
        errors="coerce",
    )


# ============================================================
# Additional feature validation
# ============================================================

ADDITIONAL_FEATURE_INPUTS = [
    "return_1d",
    "return_3d",
    "return_5d",
    "return_10d",
    "return_20d",
    "volatility_20d",
    "rsi_14",
    "roc_10",
    "roc_20",
    "adx_14",
    "close_vs_sma20",
    "close_vs_sma50",
    "sma20_vs_sma50",
    "nifty50_return_1d",
    "nifty50_return_5d",
    "nifty50_return_20d",
    "nifty50_volatility_20d",
    "stock_vs_nifty50_5d",
    "stock_vs_nifty50_20d",
    "market_trend_regime",
    "market_momentum_regime",
]


def validate_additional_feature_inputs(
    df: pd.DataFrame,
) -> None:
    """
    Validate the Phase-1 columns required to generate
    the additional point-in-time features.
    """

    missing = [
        column
        for column in ADDITIONAL_FEATURE_INPUTS
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "Required columns for additional feature "
            "generation are missing:\n"
            + "\n".join(
                f"  - {column}"
                for column in missing
            )
        )


# ============================================================
# 1. Market-relative return features
# ============================================================

def add_market_relative_features(
    df: pd.DataFrame,
) -> None:
    """
    Stock return minus NIFTY return.

    These use only information available at T.
    """

    df["market_relative_return_1d"] = (
        numeric(df, "return_1d")
        - numeric(df, "nifty50_return_1d")
    )

    df["market_relative_return_5d"] = (
        numeric(df, "return_5d")
        - numeric(df, "nifty50_return_5d")
    )

    df["market_relative_return_20d"] = (
        numeric(df, "return_20d")
        - numeric(df, "nifty50_return_20d")
    )


# ============================================================
# 2. Volatility-adjusted return
# ============================================================

def add_volatility_adjusted_return(
    df: pd.DataFrame,
) -> None:
    """
    Formula:

        return_1d / volatility_20d

    Zero volatility becomes NaN.
    """

    volatility = numeric(
        df,
        "volatility_20d",
    )

    volatility = volatility.replace(
        0,
        np.nan,
    )

    df["stock_return_vol_adjusted"] = (
        numeric(df, "return_1d")
        / volatility
    )


# ============================================================
# 3. Return acceleration
# ============================================================

def add_return_acceleration(
    df: pd.DataFrame,
) -> None:
    """
    Formula:

        return_1d - (return_5d / 5)
    """

    return_1d = numeric(
        df,
        "return_1d",
    )

    return_5d = numeric(
        df,
        "return_5d",
    )

    average_daily_5d_return = (
        return_5d / 5.0
    )

    df["return_acceleration_1d_5d"] = (
        return_1d
        - average_daily_5d_return
    )


# ============================================================
# 4. Trend strength
# ============================================================

def add_trend_strength_score(
    df: pd.DataFrame,
) -> None:
    """
    Components:

        close_vs_sma20
        close_vs_sma50
        sma20_vs_sma50
        adx_14
    """

    close_vs_sma20 = numeric(
        df,
        "close_vs_sma20",
    )

    close_vs_sma50 = numeric(
        df,
        "close_vs_sma50",
    )

    sma20_vs_sma50 = numeric(
        df,
        "sma20_vs_sma50",
    )

    adx = numeric(
        df,
        "adx_14",
    )

    adx_signal = (
        adx.clip(
            lower=0,
            upper=50,
        )
        / 50.0
    )

    df["trend_strength_score"] = (
        (
            close_vs_sma20
            + close_vs_sma50
            + sma20_vs_sma50
        )
        / 3.0
        * (
            0.5
            + 0.5 * adx_signal
        )
    )


# ============================================================
# 5. Momentum consistency
# ============================================================

def add_momentum_consistency_score(
    df: pd.DataFrame,
) -> None:
    """
    Measures directional agreement across:

        return_1d
        return_3d
        return_5d
        return_10d
        return_20d
        roc_10
        roc_20

    Approximate range:

        -1 -> consistently negative
         0 -> mixed
        +1 -> consistently positive
    """

    columns = [
        "return_1d",
        "return_3d",
        "return_5d",
        "return_10d",
        "return_20d",
        "roc_10",
        "roc_20",
    ]

    signals = []

    for column in columns:

        series = numeric(
            df,
            column,
        )

        signals.append(
            np.sign(series)
        )

    signal_frame = pd.concat(
        signals,
        axis=1,
    )

    df["momentum_consistency_score"] = (
        signal_frame.mean(axis=1)
    )


# ============================================================
# 6. Market regime encoding
# ============================================================

def encode_regime(
    series: pd.Series,
) -> pd.Series:
    """
    Convert regime values into numeric signals.

    Numeric:
        retained.

    Positive:
        +1

    Negative:
        -1

    Neutral:
        0

    Unknown:
        NaN
    """

    numeric_values = pd.to_numeric(
        series,
        errors="coerce",
    )

    result = numeric_values.copy()

    string_values = (
        series
        .astype(str)
        .str.strip()
        .str.lower()
    )

    positive = string_values.isin(
        {
            "bullish",
            "positive",
            "up",
            "strong_up",
            "strong bullish",
            "bull",
        }
    )

    negative = string_values.isin(
        {
            "bearish",
            "negative",
            "down",
            "strong_down",
            "strong bearish",
            "bear",
        }
    )

    neutral = string_values.isin(
        {
            "neutral",
            "sideways",
            "flat",
        }
    )

    result.loc[positive] = 1.0
    result.loc[negative] = -1.0
    result.loc[neutral] = 0.0

    return result


# ============================================================
# 7. Market alignment
# ============================================================

def add_market_alignment_score(
    df: pd.DataFrame,
) -> None:
    """
    Compare stock momentum with the market environment.

    Stock signals:
        stock_vs_nifty50_5d
        stock_vs_nifty50_20d

    Market signals:
        market_trend_regime
        market_momentum_regime
    """

    stock_5d = numeric(
        df,
        "stock_vs_nifty50_5d",
    )

    stock_20d = numeric(
        df,
        "stock_vs_nifty50_20d",
    )

    trend_regime = encode_regime(
        df["market_trend_regime"]
    )

    momentum_regime = encode_regime(
        df["market_momentum_regime"]
    )

    stock_signal = (
        np.sign(stock_5d)
        + np.sign(stock_20d)
    ) / 2.0

    market_signal = (
        trend_regime
        + momentum_regime
    ) / 2.0

    df["market_alignment_score"] = (
        stock_signal
        * market_signal
    )


# ============================================================
# 8. Generate all additional features
# ============================================================

def generate_additional_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Generate all additional input features.

    IMPORTANT:
    This function must run BEFORE train/validation/test split.
    """

    validate_additional_feature_inputs(
        df
    )

    add_market_relative_features(
        df
    )

    add_volatility_adjusted_return(
        df
    )

    add_return_acceleration(
        df
    )

    add_trend_strength_score(
        df
    )

    add_momentum_consistency_score(
        df
    )

    add_market_alignment_score(
        df
    )

    # Replace only generated infinities.
    # No imputation here.
    for column in GENERATED_FEATURES:

        values = pd.to_numeric(
            df[column],
            errors="coerce",
        )

        df[column] = values.replace(
            [np.inf, -np.inf],
            np.nan,
        )

    return df


# ============================================================
# Validate generated features
# ============================================================

def validate_generated_features(
    df: pd.DataFrame,
) -> list[str]:

    problems = []

    missing = [
        column
        for column in GENERATED_FEATURES
        if column not in df.columns
    ]

    if missing:
        problems.append(
            "Generated features missing: "
            + ", ".join(missing)
        )

        return problems

    for column in GENERATED_FEATURES:

        values = pd.to_numeric(
            df[column],
            errors="coerce",
        )

        infinite_count = int(
            np.isinf(
                values.to_numpy(
                    dtype=float,
                    na_value=np.nan,
                )
            ).sum()
        )

        if infinite_count > 0:
            problems.append(
                f"{column}: "
                f"{infinite_count:,} infinite values."
            )

    return problems


# ============================================================
# Load feature files
# ============================================================

def load_stock_files() -> list[Path]:

    if not FEATURE_DIR.exists():
        raise FileNotFoundError(
            f"Feature directory not found: "
            f"{FEATURE_DIR}"
        )

    if not TARGET_DIR.exists():
        raise FileNotFoundError(
            f"Target directory not found: "
            f"{TARGET_DIR}"
        )

    feature_files = sorted(
        FEATURE_DIR.glob(
            "*_features.parquet"
        )
    )

    if not feature_files:
        raise FileNotFoundError(
            f"No feature files found in "
            f"{FEATURE_DIR}"
        )

    return feature_files


# ============================================================
# Target file
# ============================================================

def get_target_file(
    symbol: str,
) -> Path:

    return (
        TARGET_DIR
        / f"{symbol}.parquet"
    )


# ============================================================
# Build one stock dataset
# ============================================================

def build_stock_dataset(
    feature_file: Path,
    target_file: Path,
) -> pd.DataFrame:

    features = pd.read_parquet(
        feature_file
    )

    targets = pd.read_parquet(
        target_file
    )

    symbol = feature_file.stem.replace(
        "_features",
        "",
    )

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
    }

    missing_features = (
        required_features
        - set(features.columns)
    )

    missing_targets = (
        required_targets
        - set(targets.columns)
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

    features["trade_date"] = pd.to_datetime(
        features["trade_date"]
    )

    targets["trade_date"] = pd.to_datetime(
        targets["trade_date"]
    )

    if features["trade_date"].duplicated().any():
        raise ValueError(
            f"{symbol}: duplicate feature dates"
        )

    if targets["trade_date"].duplicated().any():
        raise ValueError(
            f"{symbol}: duplicate target dates"
        )

    if features["symbol"].nunique() != 1:
        raise ValueError(
            f"{symbol}: feature file contains "
            "multiple symbols"
        )

    if targets["symbol"].nunique() != 1:
        raise ValueError(
            f"{symbol}: target file contains "
            "multiple symbols"
        )

    if features["symbol"].iloc[0] != symbol:
        raise ValueError(
            f"{symbol}: feature symbol mismatch"
        )

    if targets["symbol"].iloc[0] != symbol:
        raise ValueError(
            f"{symbol}: target symbol mismatch"
        )

    targets = targets[
        [
            "symbol",
            "trade_date",
            TARGET_RETURN_COLUMN,
            TARGET_DIRECTION_COLUMN,
        ]
    ].copy()

    df = features.merge(
        targets,
        on=[
            "symbol",
            "trade_date",
        ],
        how="left",
        validate="one_to_one",
    )

    return (
        df
        .sort_values(
            [
                "symbol",
                "trade_date",
            ]
        )
        .reset_index(drop=True)
    )


# ============================================================
# Baseline feature validation
# ============================================================

def validate_baseline_features(
    dataset: pd.DataFrame,
) -> list[str]:

    missing = [
        column
        for column in REQUIRED_BASELINE_FEATURES
        if column not in dataset.columns
    ]

    if missing:
        return [
            "Missing baseline features: "
            + ", ".join(missing)
        ]

    return []


# ============================================================
# Identify candidate feature columns
#
# IMPORTANT:
# We do NOT restrict this to 24 features.
#
# Every numeric Phase-1 / additional feature is available
# unless explicitly excluded.
# ============================================================

def identify_feature_columns(
    dataset: pd.DataFrame,
) -> list[str]:

    excluded = (
        IDENTIFIER_COLUMNS
        | {
            TARGET_RETURN_COLUMN,
            TARGET_DIRECTION_COLUMN,

            # Raw price fields are not model candidates.
            "open",
            "high",
            "low",
            "close",
            "volume",

            # Target-generation helper columns.
            "next_close",
            "next_trade_date",
            "target_gap_days",
        }
    )

    return sorted(
        column
        for column in dataset.columns
        if column not in excluded
        and pd.api.types.is_numeric_dtype(
            dataset[column]
        )
    )


# ============================================================
# Numeric validation
# ============================================================

def validate_numeric_values(
    dataset: pd.DataFrame,
) -> list[str]:

    problems = []

    numeric_columns = (
        dataset
        .select_dtypes(
            include=[np.number]
        )
        .columns
    )

    for column in numeric_columns:

        values = pd.to_numeric(
            dataset[column],
            errors="coerce",
        ).to_numpy(
            dtype="float64",
            na_value=np.nan,
        )

        finite_invalid = (
            ~np.isfinite(values)
        ) & ~np.isnan(values)

        invalid_count = int(
            finite_invalid.sum()
        )

        if invalid_count > 0:
            problems.append(
                f"{column}: "
                f"{invalid_count:,} "
                "infinite numeric values"
            )

    return problems


# ============================================================
# Target validation
# ============================================================

def validate_targets(
    dataset: pd.DataFrame,
) -> list[str]:

    problems = []

    target_return = dataset[
        TARGET_RETURN_COLUMN
    ]

    target_direction = dataset[
        TARGET_DIRECTION_COLUMN
    ]

    valid_target = (
        target_return.notna()
        & target_direction.notna()
    )

    if int(valid_target.sum()) == 0:
        problems.append(
            "No valid Day-1 target rows found."
        )

        return problems

    expected_direction = (
        target_return.loc[valid_target] > 0
    ).astype("int64")

    actual_direction = pd.to_numeric(
        target_direction.loc[valid_target],
        errors="coerce",
    )

    invalid_direction_values = (
        ~actual_direction.isin([0, 1])
    )

    invalid_direction_count = int(
        invalid_direction_values.sum()
    )

    if invalid_direction_count > 0:
        problems.append(
            f"{invalid_direction_count:,} "
            "target direction values are "
            "not 0 or 1."
        )

    comparable = (
        actual_direction.notna()
        & expected_direction.notna()
    )

    inconsistent_count = int(
        (
            actual_direction.loc[comparable]
            != expected_direction.loc[comparable]
        ).sum()
    )

    if inconsistent_count > 0:
        problems.append(
            f"{inconsistent_count:,} target "
            "direction values are inconsistent "
            "with target return."
        )

    target_values = pd.to_numeric(
        target_return.loc[valid_target],
        errors="coerce",
    ).to_numpy(
        dtype="float64",
        na_value=np.nan,
    )

    invalid_return_count = int(
        (~np.isfinite(target_values)).sum()
    )

    if invalid_return_count > 0:
        problems.append(
            f"{invalid_return_count:,} invalid "
            "target return values found."
        )

    return problems


# ============================================================
# T+1 semantics
# ============================================================

def validate_t_plus_1_semantics(
    dataset: pd.DataFrame,
) -> list[str]:

    problems = []

    check = dataset[
        [
            "symbol",
            "trade_date",
            "close",
            TARGET_RETURN_COLUMN,
        ]
    ].copy()

    check = check.sort_values(
        [
            "symbol",
            "trade_date",
        ]
    )

    check["next_close_from_features"] = (
        check
        .groupby("symbol")["close"]
        .shift(-1)
    )

    check["expected_target_return"] = (
        check["next_close_from_features"]
        / check["close"]
        - 1.0
    )

    valid = (
        check[
            "next_close_from_features"
        ].notna()
        & check[
            TARGET_RETURN_COLUMN
        ].notna()
        & check[
            "expected_target_return"
        ].notna()
    )

    if not valid.any():
        problems.append(
            "Unable to validate T+1 target semantics."
        )

        return problems

    actual = pd.to_numeric(
        check.loc[
            valid,
            TARGET_RETURN_COLUMN,
        ],
        errors="coerce",
    )

    expected = pd.to_numeric(
        check.loc[
            valid,
            "expected_target_return",
        ],
        errors="coerce",
    )

    valid_numeric = (
        actual.notna()
        & expected.notna()
    )

    mismatch = ~np.isclose(
        actual.loc[valid_numeric].to_numpy(
            dtype=float
        ),
        expected.loc[valid_numeric].to_numpy(
            dtype=float
        ),
        rtol=1e-8,
        atol=1e-10,
    )

    mismatch_count = int(
        mismatch.sum()
    )

    if mismatch_count > 0:
        problems.append(
            f"{mismatch_count:,} rows do not "
            "match Close(T+1) / Close(T) - 1."
        )

    return problems


# ============================================================
# Leakage scan
# ============================================================

def leakage_scan(
    dataset: pd.DataFrame,
    feature_columns: list[str],
) -> list[str]:

    problems = []

    suspicious_words = [
        "target",
        "future",
        "next_close",
        "t_plus_1",
        "t+1",
    ]

    suspicious_features = [
        column
        for column in feature_columns
        if any(
            word in column.lower()
            for word in suspicious_words
        )
    ]

    if suspicious_features:
        problems.append(
            "Suspicious feature names detected: "
            + ", ".join(
                suspicious_features
            )
        )

    target = pd.to_numeric(
        dataset[TARGET_RETURN_COLUMN],
        errors="coerce",
    )

    for column in feature_columns:

        values = pd.to_numeric(
            dataset[column],
            errors="coerce",
        )

        pair = pd.concat(
            [
                values,
                target,
            ],
            axis=1,
        ).dropna()

        if len(pair) < 100:
            continue

        if np.allclose(
            pair.iloc[:, 0].to_numpy(),
            pair.iloc[:, 1].to_numpy(),
        ):
            problems.append(
                f"Feature '{column}' appears "
                "identical to target return."
            )

    return problems


# ============================================================
# Cross-sectional rank validation
# ============================================================

def validate_cross_sectional_ranks(
    dataset: pd.DataFrame,
) -> list[str]:

    problems = []

    rank_columns = [
        "return_rank_1d",
        "return_rank_5d",
        "return_rank_20d",
        "relative_strength_rank",
        "volume_rank",
    ]

    for column in rank_columns:

        if column not in dataset.columns:
            continue

        values = pd.to_numeric(
            dataset[column],
            errors="coerce",
        ).dropna()

        if values.empty:
            problems.append(
                f"{column}: no valid values."
            )

            continue

        if (
            (values < 0).any()
            or (values > 1).any()
        ):
            problems.append(
                f"{column}: values outside [0, 1]."
            )

    return problems


# ============================================================
# Distribution report
# ============================================================

def distribution_summary(
    dataset: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:

    rows = []

    for column in feature_columns:

        series = pd.to_numeric(
            dataset[column],
            errors="coerce",
        )

        finite = series.replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )

        rows.append(
            {
                "feature": column,
                "non_null": int(
                    finite.notna().sum()
                ),
                "null_pct": float(
                    finite.isna().mean()
                    * 100
                ),
                "min": finite.min(),
                "p01": finite.quantile(0.01),
                "median": finite.median(),
                "p99": finite.quantile(0.99),
                "max": finite.max(),
                "inf_count": int(
                    np.isinf(
                        series.to_numpy(
                            dtype=float,
                            na_value=np.nan,
                        )
                    ).sum()
                ),
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# Chronological split
# ============================================================

def create_chronological_splits(
    model_dataset: pd.DataFrame,
):

    dates = np.array(
        sorted(
            model_dataset[
                "trade_date"
            ]
            .dropna()
            .unique()
        )
    )

    if len(dates) < 3:
        raise ValueError(
            "Not enough unique dates for "
            "chronological split."
        )

    train_end = int(
        len(dates)
        * TRAIN_RATIO
    )

    validation_end = int(
        len(dates)
        * (
            TRAIN_RATIO
            + VALIDATION_RATIO
        )
    )

    train_end = max(
        1,
        min(
            train_end,
            len(dates) - 2,
        ),
    )

    validation_end = max(
        train_end + 1,
        min(
            validation_end,
            len(dates) - 1,
        ),
    )

    train_dates = dates[
        :train_end
    ]

    validation_dates = dates[
        train_end:validation_end
    ]

    test_dates = dates[
        validation_end:
    ]

    train = model_dataset[
        model_dataset["trade_date"].isin(
            train_dates
        )
    ].copy()

    validation = model_dataset[
        model_dataset["trade_date"].isin(
            validation_dates
        )
    ].copy()

    test = model_dataset[
        model_dataset["trade_date"].isin(
            test_dates
        )
    ].copy()

    return (
        train,
        validation,
        test,
    )


# ============================================================
# Split validation
# ============================================================

def validate_split_boundaries(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
) -> list[str]:

    problems = []

    if train.empty:
        problems.append(
            "Train split is empty."
        )

    if validation.empty:
        problems.append(
            "Validation split is empty."
        )

    if test.empty:
        problems.append(
            "Test split is empty."
        )

    if (
        not train.empty
        and not validation.empty
    ):
        if (
            train["trade_date"].max()
            >= validation["trade_date"].min()
        ):
            problems.append(
                "Train and validation dates "
                "overlap or are out of order."
            )

    if (
        not validation.empty
        and not test.empty
    ):
        if (
            validation["trade_date"].max()
            >= test["trade_date"].min()
        ):
            problems.append(
                "Validation and test dates "
                "overlap or are out of order."
            )

    return problems


# ============================================================
# Validate schema consistency across splits
# ============================================================

def validate_split_schema(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
) -> list[str]:

    problems = []

    train_columns = list(
        train.columns
    )

    validation_columns = list(
        validation.columns
    )

    test_columns = list(
        test.columns
    )

    if train_columns != validation_columns:
        problems.append(
            "Train and validation schemas differ."
        )

    if train_columns != test_columns:
        problems.append(
            "Train and test schemas differ."
        )

    return problems


# ============================================================
# Report
# ============================================================

def write_report(
    dataset: pd.DataFrame,
    model_dataset: pd.DataFrame,
    feature_columns: list[str],
    baseline_problems: list[str],
    additional_feature_problems: list[str],
    target_problems: list[str],
    t_plus_1_problems: list[str],
    numeric_problems: list[str],
    leakage_problems: list[str],
    rank_problems: list[str],
    split_problems: list[str],
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    distribution: pd.DataFrame,
) -> None:

    lines = []

    lines.append("=" * 90)
    lines.append(
        "DAY-1 MODELING DATASET VALIDATION REPORT"
    )
    lines.append("=" * 90)
    lines.append("")

    lines.append(
        "Pipeline:"
    )

    lines.append(
        "Phase-1 features + generated targets "
        "-> additional point-in-time features "
        "-> validation "
        "-> chronological split"
    )

    lines.append("")

    lines.append("PATHS")
    lines.append("-" * 90)

    lines.append(
        f"Feature directory : {FEATURE_DIR}"
    )

    lines.append(
        f"Target directory  : {TARGET_DIR}"
    )

    lines.append(
        f"Modeling directory: {OUTPUT_DIR}"
    )

    lines.append("")

    lines.append("DATASET")
    lines.append("-" * 90)

    lines.append(
        f"Rows assembled       : "
        f"{len(dataset):,}"
    )

    lines.append(
        f"Training-ready rows  : "
        f"{len(model_dataset):,}"
    )

    lines.append(
        f"Stocks               : "
        f"{dataset['symbol'].nunique():,}"
    )

    lines.append(
        f"Total model features : "
        f"{len(feature_columns):,}"
    )

    lines.append(
        f"Additional features  : "
        f"{len(GENERATED_FEATURES):,}"
    )

    lines.append(
        f"Date range            : "
        f"{dataset['trade_date'].min().date()} "
        f"-> "
        f"{dataset['trade_date'].max().date()}"
    )

    lines.append("")

    def add_check(
        title: str,
        problems: list[str],
    ):

        lines.append(title)
        lines.append("-" * 90)

        if problems:

            for problem in problems:
                lines.append(
                    f"[FAIL] {problem}"
                )

        else:
            lines.append(
                "[PASS]"
            )

        lines.append("")

    add_check(
        "BASELINE FEATURE CHECK",
        baseline_problems,
    )

    add_check(
        "ADDITIONAL FEATURE CHECK",
        additional_feature_problems,
    )

    add_check(
        "TARGET VALIDATION",
        target_problems,
    )

    add_check(
        "T -> T+1 TARGET SEMANTICS",
        t_plus_1_problems,
    )

    add_check(
        "NUMERIC VALUE CHECK",
        numeric_problems,
    )

    add_check(
        "LEAKAGE CHECK",
        leakage_problems,
    )

    add_check(
        "CROSS-SECTIONAL RANK CHECK",
        rank_problems,
    )

    add_check(
        "CHRONOLOGICAL SPLIT CHECK",
        split_problems,
    )

    lines.append(
        "ADDITIONAL FEATURES GENERATED"
    )

    lines.append("-" * 90)

    for feature in GENERATED_FEATURES:
        lines.append(
            f"  {feature}"
        )

    lines.append("")

    lines.append(
        "SPLIT SUMMARY"
    )

    lines.append("-" * 90)

    for name, frame in [
        ("TRAIN", train),
        ("VALIDATION", validation),
        ("TEST", test),
    ]:

        if frame.empty:

            lines.append(
                f"{name}: EMPTY"
            )

        else:

            lines.append(
                f"{name}: "
                f"rows={len(frame):,}, "
                f"stocks="
                f"{frame['symbol'].nunique():,}, "
                f"dates="
                f"{frame['trade_date'].min().date()} "
                f"-> "
                f"{frame['trade_date'].max().date()}"
            )

    lines.append("")

    lines.append(
        "FEATURE NULL / DISTRIBUTION REVIEW"
    )

    lines.append("-" * 90)

    high_null = distribution[
        distribution["null_pct"] > 30
    ].sort_values(
        "null_pct",
        ascending=False,
    )

    infinite_features = distribution[
        distribution["inf_count"] > 0
    ]

    lines.append(
        f"Features with >30% null : "
        f"{len(high_null)}"
    )

    lines.append(
        f"Features with infinity  : "
        f"{len(infinite_features)}"
    )

    if not high_null.empty:

        lines.append("")
        lines.append(
            "High-null features:"
        )

        for _, row in high_null.iterrows():

            lines.append(
                f"  {row['feature']}: "
                f"{row['null_pct']:.2f}%"
            )

    lines.append("")

    lines.append(
        "TARGET SUMMARY"
    )

    lines.append("-" * 90)

    valid_target = dataset[
        TARGET_RETURN_COLUMN
    ].notna()

    if valid_target.any():

        returns = pd.to_numeric(
            dataset.loc[
                valid_target,
                TARGET_RETURN_COLUMN,
            ],
            errors="coerce",
        )

        directions = pd.to_numeric(
            dataset.loc[
                valid_target,
                TARGET_DIRECTION_COLUMN,
            ],
            errors="coerce",
        )

        lines.append(
            f"Valid target rows       : "
            f"{int(valid_target.sum()):,}"
        )

        lines.append(
            f"Mean target return      : "
            f"{returns.mean():.8f}"
        )

        lines.append(
            f"Median target return    : "
            f"{returns.median():.8f}"
        )

        lines.append(
            f"Positive direction rate : "
            f"{directions.mean():.4%}"
        )

    lines.append("")

    all_problems = (
        baseline_problems
        + additional_feature_problems
        + target_problems
        + t_plus_1_problems
        + numeric_problems
        + leakage_problems
        + rank_problems
        + split_problems
    )

    lines.append(
        "OVERALL STATUS"
    )

    lines.append("-" * 90)

    if all_problems:

        lines.append(
            "[REVIEW REQUIRED]"
        )

        lines.append(
            "Validation findings exist. "
            "Review this report before model training."
        )

    else:

        lines.append(
            "[PASS]"
        )

        lines.append(
            "All automated structural, target, "
            "T+1, additional-feature, leakage "
            "and split checks passed."
        )

        lines.append(
            "Feature null/distribution statistics "
            "should still be reviewed."
        )

    REPORT_FILE.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


# ============================================================
# Main
# ============================================================

def main() -> None:

    print("=" * 80)
    print(
        "DAY-1 MODELING DATASET VALIDATION"
    )
    print("=" * 80)

    print()
    print(
        "Feature directory:"
    )
    print(FEATURE_DIR)

    print(
        "Target directory:"
    )
    print(TARGET_DIR)

    print(
        "Modeling directory:"
    )
    print(OUTPUT_DIR)

    # --------------------------------------------------------
    # Load stock feature files
    # --------------------------------------------------------

    feature_files = load_stock_files()

    print()
    print(
        f"Feature files found: "
        f"{len(feature_files)}"
    )

    if (
        len(feature_files)
        != EXPECTED_STOCK_COUNT
    ):
        raise ValueError(
            f"Expected "
            f"{EXPECTED_STOCK_COUNT} "
            f"feature files, found "
            f"{len(feature_files)}"
        )

    # --------------------------------------------------------
    # Build feature + target dataset
    # --------------------------------------------------------

    all_data = []

    successful = 0
    failed = 0

    print()
    print(
        "Building stock datasets..."
    )

    print("-" * 80)

    for feature_file in feature_files:

        symbol = feature_file.stem.replace(
            "_features",
            "",
        )

        target_file = get_target_file(
            symbol
        )

        if not target_file.exists():

            print(
                f"[FAILED] "
                f"{symbol:<15} "
                "target file not found"
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
                f"[OK] "
                f"{symbol:<15} "
                f"{len(df):>7,} rows"
            )

        except Exception as exc:

            failed += 1

            print(
                f"[FAILED] "
                f"{symbol:<15} "
                f"{type(exc).__name__}: "
                f"{exc}"
            )

    if failed > 0:

        raise RuntimeError(
            f"Dataset construction failed "
            f"for {failed} stock(s)"
        )

    if (
        successful
        != EXPECTED_STOCK_COUNT
    ):

        raise ValueError(
            f"Expected "
            f"{EXPECTED_STOCK_COUNT} "
            f"successful stocks, got "
            f"{successful}"
        )

    dataset = pd.concat(
        all_data,
        ignore_index=True,
    )

    dataset = (
        dataset
        .sort_values(
            [
                "trade_date",
                "symbol",
            ]
        )
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Duplicate validation
    # --------------------------------------------------------

    duplicate_count = int(
        dataset.duplicated(
            subset=[
                "symbol",
                "trade_date",
            ]
        ).sum()
    )

    if duplicate_count:

        raise ValueError(
            f"Duplicate stock/date rows: "
            f"{duplicate_count}"
        )

    print()
    print(
        "Running dataset validation..."
    )

    print("-" * 80)

    # --------------------------------------------------------
    # Baseline features
    # --------------------------------------------------------

    baseline_problems = (
        validate_baseline_features(
            dataset
        )
    )

    # --------------------------------------------------------
    # Targets
    # --------------------------------------------------------

    target_problems = (
        validate_targets(
            dataset
        )
    )

    # --------------------------------------------------------
    # T+1 target semantics
    # --------------------------------------------------------

    t_plus_1_problems = (
        validate_t_plus_1_semantics(
            dataset
        )
    )

    # --------------------------------------------------------
    # Generate additional features
    #
    # THIS IS BEFORE THE SPLIT.
    # --------------------------------------------------------

    print()
    print(
        "Generating additional input features..."
    )

    dataset = generate_additional_features(
        dataset
    )

    print(
        "Additional features generated:"
    )

    for feature in GENERATED_FEATURES:
        print(
            f"  + {feature}"
        )

    # --------------------------------------------------------
    # Validate generated features
    # --------------------------------------------------------

    additional_feature_problems = (
        validate_generated_features(
            dataset
        )
    )

    # --------------------------------------------------------
    # Identify ALL numeric model features
    #
    # Not limited to 24.
    # --------------------------------------------------------

    feature_columns = (
        identify_feature_columns(
            dataset
        )
    )

    print()
    print(
        f"Total candidate model features: "
        f"{len(feature_columns)}"
    )

    # --------------------------------------------------------
    # Numeric validation
    # --------------------------------------------------------

    numeric_problems = (
        validate_numeric_values(
            dataset
        )
    )

    # --------------------------------------------------------
    # Leakage
    # --------------------------------------------------------

    leakage_problems = leakage_scan(
        dataset,
        feature_columns,
    )

    # --------------------------------------------------------
    # Cross-sectional ranks
    # --------------------------------------------------------

    rank_problems = (
        validate_cross_sectional_ranks(
            dataset
        )
    )

    # --------------------------------------------------------
    # Replace infinities with NaN
    #
    # DO NOT impute here.
    #
    # Imputation, if required, belongs inside the model
    # training pipeline and must be fitted using TRAIN only.
    # --------------------------------------------------------

    numeric_columns = (
        dataset
        .select_dtypes(
            include=[np.number]
        )
        .columns
    )

    dataset[numeric_columns] = (
        dataset[numeric_columns]
        .replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )
    )

    # --------------------------------------------------------
    # Keep only rows with valid targets
    # --------------------------------------------------------

    valid_target = (
        dataset[
            TARGET_RETURN_COLUMN
        ].notna()
        & dataset[
            TARGET_DIRECTION_COLUMN
        ].notna()
    )

    model_dataset = (
        dataset.loc[
            valid_target
        ]
        .copy()
        .sort_values(
            [
                "trade_date",
                "symbol",
            ]
        )
        .reset_index(drop=True)
    )

    if model_dataset.empty:

        raise ValueError(
            "Final Day-1 modeling dataset "
            "is empty."
        )

    # --------------------------------------------------------
    # Chronological split
    #
    # IMPORTANT:
    # The complete feature dataset is already built.
    # --------------------------------------------------------

    print()
    print(
        "Creating chronological "
        "train/validation/test splits..."
    )

    (
        train,
        validation,
        test,
    ) = create_chronological_splits(
        model_dataset
    )

    split_problems = (
        validate_split_boundaries(
            train,
            validation,
            test,
        )
    )

    schema_problems = (
        validate_split_schema(
            train,
            validation,
            test,
        )
    )

    split_problems.extend(
        schema_problems
    )

    # --------------------------------------------------------
    # Distribution report
    # --------------------------------------------------------

    distribution = (
        distribution_summary(
            model_dataset,
            feature_columns,
        )
    )

    # --------------------------------------------------------
    # Save COMPLETE dataset
    # --------------------------------------------------------

    model_dataset.to_parquet(
        OUTPUT_FILE,
        index=False,
    )

    train.to_parquet(
        TRAIN_FILE,
        index=False,
    )

    validation.to_parquet(
        VALIDATION_FILE,
        index=False,
    )

    test.to_parquet(
        TEST_FILE,
        index=False,
    )

    distribution.to_parquet(
        DISTRIBUTION_FILE,
        index=False,
    )

    # --------------------------------------------------------
    # Write report
    # --------------------------------------------------------

    write_report(
        dataset=dataset,
        model_dataset=model_dataset,
        feature_columns=feature_columns,
        baseline_problems=baseline_problems,
        additional_feature_problems=(
            additional_feature_problems
        ),
        target_problems=target_problems,
        t_plus_1_problems=t_plus_1_problems,
        numeric_problems=numeric_problems,
        leakage_problems=leakage_problems,
        rank_problems=rank_problems,
        split_problems=split_problems,
        train=train,
        validation=validation,
        test=test,
        distribution=distribution,
    )

    # --------------------------------------------------------
    # Final console output
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print(
        "DATASET PIPELINE COMPLETE"
    )
    print("=" * 80)

    print(
        f"All assembled rows    : "
        f"{len(dataset):,}"
    )

    print(
        f"Modeling rows         : "
        f"{len(model_dataset):,}"
    )

    print(
        f"Stocks                : "
        f"{model_dataset['symbol'].nunique()}"
    )

    print(
        f"Candidate features    : "
        f"{len(feature_columns)}"
    )

    print(
        f"Additional features   : "
        f"{len(GENERATED_FEATURES)}"
    )

    print()

    print(
        "Additional features:"
    )

    for feature in GENERATED_FEATURES:
        print(
            f"  {feature}"
        )

    print()

    print(
        f"Model dataset : "
        f"{OUTPUT_FILE}"
    )

    print(
        f"Train         : "
        f"{TRAIN_FILE}"
    )

    print(
        f"Validation    : "
        f"{VALIDATION_FILE}"
    )

    print(
        f"Test          : "
        f"{TEST_FILE}"
    )

    print(
        f"Distribution  : "
        f"{DISTRIBUTION_FILE}"
    )

    print(
        f"Report        : "
        f"{REPORT_FILE}"
    )

    print()

    all_problems = (
        baseline_problems
        + additional_feature_problems
        + target_problems
        + t_plus_1_problems
        + numeric_problems
        + leakage_problems
        + rank_problems
        + split_problems
    )

    if all_problems:

        print(
            "STATUS: REVIEW REQUIRED"
        )

        print(
            "Open validation_report.txt "
            "before model training."
        )

    else:

        print(
            "STATUS: ALL AUTOMATED CHECKS PASSED"
        )

    print("=" * 80)


if __name__ == "__main__":
    main()