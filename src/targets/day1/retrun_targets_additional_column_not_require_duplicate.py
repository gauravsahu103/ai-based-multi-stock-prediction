"""
Additional Feature Engineering
=========================================

Purpose
-------
Generate ONLY the additional derived features required.

Input
-----
modeling dataset:

    nse100_technical_model_dataset.parquet

This file already contains all  generated technical,
market, sector and cross-sectional features.

This script DOES NOT:
    - recreate  technical indicators
    - select/remove  features
    - train a model
    - generate predictions
    - generate prediction rankings
    - generate future targets

It ONLY adds additional derived INPUT FEATURES.

Phase-2 objectives supported by these features:
    1. Next-day return prediction
    2. Next-day direction prediction
    3. Stock / excess-NIFTY ranking

Important
---------
All features generated here use information available at T only.

No future close or T+1 information is used.

Generated feature groups
------------------------
Market-relative features:
    market_relative_return_1d
    market_relative_return_5d
    market_relative_return_20d

Risk-adjusted features:
    stock_return_vol_adjusted

Momentum / trend features:
    return_acceleration_1d_5d
    trend_strength_score
    momentum_consistency_score

Market alignment:
    market_alignment_score

The following are NOT generated here because they are model outputs:
    predicted_return
    predicted_excess_return
    predicted_return_rank
    excess_return_rank

The future excess-return target should also be generated separately.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.mvp_config import modeling_dir


# ============================================================
# Configuration
# ============================================================

HORIZON = 1

INPUT_FILE = (
    modeling_dir(HORIZON)
    / "nse100_technical_model_dataset.parquet"
)

OUTPUT_FILE = (
    modeling_dir(HORIZON)
    / "nse100_feature_dataset_addtional_feature.parquet"
)


# ============================================================
# Required  columns
# ============================================================

REQUIRED_COLUMNS = [
    # --------------------------------------------------------
    # Stock returns
    # --------------------------------------------------------
    "return_1d",
    "return_3d",
    "return_5d",
    "return_10d",
    "return_20d",

    # --------------------------------------------------------
    # Volatility
    # --------------------------------------------------------
    "volatility_20d",

    # --------------------------------------------------------
    # Technical trend / momentum
    # --------------------------------------------------------
    "rsi_14",
    "roc_10",
    "roc_20",
    "adx_14",

    # --------------------------------------------------------
    # Moving averages
    # --------------------------------------------------------
    "close_vs_sma20",
    "close_vs_sma50",
    "sma20_vs_sma50",

    # --------------------------------------------------------
    # Market
    # --------------------------------------------------------
    "nifty50_return_1d",
    "nifty50_return_5d",
    "nifty50_return_20d",
    "nifty50_volatility_20d",

    # --------------------------------------------------------
    # Stock vs market
    # --------------------------------------------------------
    "stock_vs_nifty50_5d",
    "stock_vs_nifty50_20d",

    # --------------------------------------------------------
    # Market regime
    # --------------------------------------------------------
    "market_trend_regime",
    "market_momentum_regime",
]


# ============================================================
# Validation
# ============================================================

def validate_input_columns(
    df: pd.DataFrame,
) -> None:
    """Validate that all required  columns exist."""

    missing = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "Required  columns are missing:\n"
            + "\n".join(
                f"  - {column}"
                for column in missing
            )
        )


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
# 1. Market-relative return features
# ============================================================

def add_market_relative_features(
    df: pd.DataFrame,
) -> None:
    """
    Calculate stock return relative to NIFTY return.

    Formula:

        stock return - NIFTY return

    These are equivalent in concept to excess performance
    over the market for the historical lookback period.

    They use information available at T only.
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
    Calculate current 1-day stock return adjusted for
    recent stock volatility.

    Formula:

        return_1d / volatility_20d

    This provides a simple risk-adjusted momentum signal.

    Zero volatility is converted to NaN rather than producing
    infinity.
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
    Measure whether short-term momentum is strengthening
    or weakening.

    Formula:

        1-day return - average daily return implied by
        the 5-day return

    5-day return is converted to an approximate daily rate
    before comparison.

    Positive:
        short-term momentum stronger than recent average.

    Negative:
        short-term momentum weaker than recent average.
    """

    return_1d = numeric(
        df,
        "return_1d",
    )

    return_5d = numeric(
        df,
        "return_5d",
    )

    # Approximate average daily return over 5 days.
    average_daily_5d_return = (
        return_5d / 5.0
    )

    df["return_acceleration_1d_5d"] = (
        return_1d
        - average_daily_5d_return
    )


# ============================================================
# 4. Trend strength score
# ============================================================

def add_trend_strength_score(
    df: pd.DataFrame,
) -> None:
    """
    Create a deterministic trend-strength feature.

    Components:
        close_vs_sma20
        close_vs_sma50
        sma20_vs_sma50
        ADX

    The components are first transformed into comparable
    bounded signals.

    This is a FEATURE, not a prediction.
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

    # ADX is approximately transformed into 0-1 range.
    # Values above 50 are clipped.
    adx_signal = (
        adx.clip(
            lower=0,
            upper=50,
        )
        / 50.0
    )

    # Average trend components.
    df["trend_strength_score"] = (
        close_vs_sma20
        + close_vs_sma50
        + sma20_vs_sma50
    ) / 3.0 * (
        0.5 + 0.5 * adx_signal
    )


# ============================================================
# 5. Momentum consistency
# ============================================================

def add_momentum_consistency_score(
    df: pd.DataFrame,
) -> None:
    """
    Measure consistency of momentum across multiple horizons.

    Inputs:
        return_1d
        return_3d
        return_5d
        return_10d
        return_20d
        roc_10
        roc_20

    The feature measures directional agreement rather than
    magnitude.

    Range is approximately [-1, +1].

        +1 -> strongly consistent positive momentum
        -1 -> strongly consistent negative momentum
         0 -> mixed momentum
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
# 6. Market alignment
# ============================================================

def add_market_alignment_score(
    df: pd.DataFrame,
) -> None:
    """
    Measure whether stock momentum agrees with the current
    market trend/momentum environment.

    Stock signals:
        stock_vs_nifty50_5d
        stock_vs_nifty50_20d

    Market signals:
        market_trend_regime
        market_momentum_regime

    The regime columns are expected to contain numeric regime
    representations generated .

    If the regime columns are categorical strings, they are
    encoded deterministically.
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
# Regime encoding
# ============================================================

def encode_regime(
    series: pd.Series,
) -> pd.Series:
    """
    Convert market regime values to numeric signals.

    Numeric regimes:
        retained as-is.

    String regimes:
        bullish / positive / up       -> +1
        bearish / negative / down     -> -1
        neutral / sideways            -> 0

    Unknown values become NaN.
    """

    numeric_values = pd.to_numeric(
        series,
        errors="coerce",
    )

    result = numeric_values.copy()

    string_values = (
        series.astype(str)
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
# Validation of generated features
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


def validate_generated_features(
    df: pd.DataFrame,
) -> None:
    """Validate generated Phase-2 feature columns."""

    missing = [
        column
        for column in GENERATED_FEATURES
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "Phase-2 feature generation failed. "
            "Missing columns:\n"
            + "\n".join(
                f"  - {column}"
                for column in missing
            )
        )

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
            raise ValueError(
                f"{column}: "
                f"{infinite_count:,} infinite values generated."
            )


# ============================================================
# Main feature-engineering pipeline
# ============================================================

def generate_phase2_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Generate all additional Phase-2 input features.

     columns are preserved unchanged.
    """

    validate_input_columns(df)

    # --------------------------------------------------------
    # Market-relative performance
    # --------------------------------------------------------

    add_market_relative_features(df)

    # --------------------------------------------------------
    # Risk-adjusted performance
    # --------------------------------------------------------

    add_volatility_adjusted_return(df)

    # --------------------------------------------------------
    # Momentum acceleration
    # --------------------------------------------------------

    add_return_acceleration(df)

    # --------------------------------------------------------
    # Trend strength
    # --------------------------------------------------------

    add_trend_strength_score(df)

    # --------------------------------------------------------
    # Momentum consistency
    # --------------------------------------------------------

    add_momentum_consistency_score(df)

    # --------------------------------------------------------
    # Market alignment
    # --------------------------------------------------------

    add_market_alignment_score(df)

    # --------------------------------------------------------
    # Replace generated infinity values.
    #
    # No imputation here.
    # Imputation, if required, belongs inside the model
    # training pipeline and must be fitted on TRAIN only.
    # --------------------------------------------------------

    numeric_columns = df.select_dtypes(
        include=[np.number]
    ).columns

    df[numeric_columns] = (
        df[numeric_columns]
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
    )

    validate_generated_features(df)

    return df


# ============================================================
# Main
# ============================================================

def main() -> None:

    print("=" * 80)
    print("ADDITIONAL FEATURE GENERATION")
    print("=" * 80)

    print()
    print(f"Input : {INPUT_FILE}")
    print(f"Output: {OUTPUT_FILE}")

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f" modeling dataset not found:\n"
            f"{INPUT_FILE}"
        )

    # --------------------------------------------------------
    # Load  dataset
    # --------------------------------------------------------

    df = pd.read_parquet(
        INPUT_FILE
    )

    print()
    print(
        f" rows    : {len(df):,}"
    )

    print(
        f" columns : {len(df.columns):,}"
    )

    # --------------------------------------------------------
    # Generate additional features
    # --------------------------------------------------------

    enriched = generate_phase2_features(
        df.copy()
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    enriched.to_parquet(
        OUTPUT_FILE,
        index=False,
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()
    print("-" * 80)
    print("GENERATED PHASE-2 FEATURES")
    print("-" * 80)

    for column in GENERATED_FEATURES:
        print(f"  {column}")

    print()
    print(
        f" columns        : "
        f"{len(df.columns):,}"
    )

    print(
        f"Additional columns     : "
        f"{len(GENERATED_FEATURES):,}"
    )

    print(
        f"Final Phase-2 columns  : "
        f"{len(enriched.columns):,}"
    )

    print(
        f"Rows                   : "
        f"{len(enriched):,}"
    )

    print()
    print(
        f"Saved dataset:\n{OUTPUT_FILE}"
    )

    print()
    print(
        "STATUS: PHASE-2 ADDITIONAL FEATURES GENERATED"
    )

    print("=" * 80)


if __name__ == "__main__":
    main()