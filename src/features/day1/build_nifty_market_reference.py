"""
Build canonical NIFTY-50 market reference features.

Purpose
-------
The stock feature generator must NOT calculate NIFTY returns/rolling
statistics on stock-specific rows.

This script calculates all market-reference features once from the
standalone NIFTY-50 trading series, then saves the result as a
date-indexed reference table.

Input
-----
data/processed/index/nifty50_daily.parquet

Expected columns
----------------
trade_date
close

Output
------
data/processed/index/nifty50_market_features.parquet

No stock data is modified by this script.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.mvp_config import storage_path


HORIZON = 1

INDEX_FILE = storage_path("processed") / "index" / "nifty50_daily.parquet"
OUTPUT_FILE = (
    storage_path("processed")
    / "index"
    / "nifty50_market_features.parquet"
)


def safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    return a / b.replace(0, np.nan)


def load_nifty() -> pd.DataFrame:
    if not INDEX_FILE.exists():
        raise FileNotFoundError(f"NIFTY file not found: {INDEX_FILE}")

    df = pd.read_parquet(INDEX_FILE)

    required = {"trade_date", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"NIFTY file missing columns: {sorted(missing)}")

    df = df[["trade_date", "close"]].copy()
    df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")

    if df["trade_date"].isna().any():
        raise ValueError("NIFTY contains invalid trade dates.")

    if df["close"].isna().any() or (df["close"] <= 0).any():
        raise ValueError("NIFTY contains invalid/non-positive close values.")

    if df["trade_date"].duplicated().any():
        raise ValueError("Duplicate NIFTY trade dates found.")

    return df.sort_values("trade_date").reset_index(drop=True)


def build_market_features(nifty: pd.DataFrame) -> pd.DataFrame:
    df = nifty.copy()

    # Historical market returns calculated ONLY on NIFTY observations.
    df["nifty50_return_1d"] = df["close"].pct_change(1)
    df["nifty50_return_3d"] = df["close"].pct_change(3)
    df["nifty50_return_5d"] = df["close"].pct_change(5)
    df["nifty50_return_20d"] = df["close"].pct_change(20)

    df["nifty50_volatility_5d"] = (
        df["nifty50_return_1d"].rolling(5, min_periods=5).std()
    )
    df["nifty50_volatility_10d"] = (
        df["nifty50_return_1d"].rolling(10, min_periods=10).std()
    )
    df["nifty50_volatility_20d"] = (
        df["nifty50_return_1d"].rolling(20, min_periods=20).std()
    )

    # Rolling beta reference statistics require stock returns and remain
    # in the stock branch. This table intentionally contains only NIFTY
    # features, not stock/NIFTY covariance.

    nifty_sma50 = df["close"].rolling(50, min_periods=50).mean()
    nifty_vol60 = (
        df["nifty50_return_1d"].rolling(60, min_periods=60).std()
    )

    # Preserve the same definitions as the MVP, but calculate them here
    # on the canonical NIFTY series.
    df["market_trend_regime"] = safe_div(df["close"], nifty_sma50) - 1.0
    df["market_volatility_regime"] = safe_div(
        df["nifty50_volatility_20d"], nifty_vol60
    )
    df["market_momentum_regime"] = df["nifty50_return_20d"]

    output_columns = [
        "trade_date",
        "close",
        "nifty50_return_1d",
        "nifty50_return_3d",
        "nifty50_return_5d",
        "nifty50_return_20d",
        "nifty50_volatility_5d",
        "nifty50_volatility_10d",
        "nifty50_volatility_20d",
        "market_trend_regime",
        "market_volatility_regime",
        "market_momentum_regime",
    ]

    return df[output_columns].copy()


def validate(df: pd.DataFrame) -> None:
    if df.empty:
        raise ValueError("Market feature table is empty.")

    if df["trade_date"].duplicated().any():
        raise ValueError("Duplicate dates in market feature table.")

    numeric_columns = [
        c for c in df.columns if c != "trade_date"
    ]

    for column in numeric_columns:
        values = pd.to_numeric(df[column], errors="coerce")
        if np.isinf(values.to_numpy(dtype=float, na_value=np.nan)).any():
            raise ValueError(f"Infinite values found in {column}.")


def main() -> None:
    nifty = load_nifty()
    market = build_market_features(nifty)
    validate(market)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    market.to_parquet(OUTPUT_FILE, index=False)

    print("=" * 80)
    print("NIFTY MARKET REFERENCE BUILD")
    print("=" * 80)
    print(f"Input  : {INDEX_FILE}")
    print(f"Output : {OUTPUT_FILE}")
    print(f"Rows   : {len(market):,}")
    print(f"Dates  : {market['trade_date'].min().date()} -> "
          f"{market['trade_date'].max().date()}")
    print("STATUS : PASSED")


if __name__ == "__main__":
    main()
