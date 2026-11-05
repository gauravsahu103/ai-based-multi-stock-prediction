from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# Configuration
# ============================================================

PRICE_DIR = Path("data/processed/prices")
INDEX_FILE = Path("data/processed/index/nifty50_daily.parquet")
OUTPUT_DIR = Path("data/features/technical")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Helper functions
# ============================================================

def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate Relative Strength Index (RSI).
    """

    delta = series.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(period, min_periods=period).mean()
    avg_loss = loss.rolling(period, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    rsi = 100 - (100 / (1 + rs))

    return rsi


def calculate_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """
    Calculate Average True Range (ATR).
    """

    previous_close = close.shift(1)

    tr1 = high - low
    tr2 = (high - previous_close).abs()
    tr3 = (low - previous_close).abs()

    true_range = pd.concat(
        [tr1, tr2, tr3],
        axis=1,
    ).max(axis=1)

    atr = true_range.rolling(
        period,
        min_periods=period,
    ).mean()

    return atr


def calculate_features(
    stock_df: pd.DataFrame,
    market_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate technical and market-relative features
    for a single stock.
    """

    df = stock_df.copy()

    # --------------------------------------------------------
    # Basic validation
    # --------------------------------------------------------

    required_stock_columns = {
        "symbol",
        "trade_date",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "traded_value",
    }

    missing = required_stock_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing stock columns: {sorted(missing)}"
        )

    required_market_columns = {
        "trade_date",
        "close",
    }

    missing_market = required_market_columns - set(
        market_df.columns
    )

    if missing_market:
        raise ValueError(
            f"Missing market columns: {sorted(missing_market)}"
        )

    # --------------------------------------------------------
    # Sort
    # --------------------------------------------------------

    df["trade_date"] = pd.to_datetime(df["trade_date"])
    market_df["trade_date"] = pd.to_datetime(
        market_df["trade_date"]
    )

    df = df.sort_values("trade_date").reset_index(drop=True)

    market_df = (
        market_df
        .sort_values("trade_date")
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Stock returns
    # --------------------------------------------------------

    df["return_1d"] = df["close"].pct_change(1)

    df["return_3d"] = df["close"].pct_change(3)

    df["return_5d"] = df["close"].pct_change(5)

    df["return_10d"] = df["close"].pct_change(10)

    df["return_20d"] = df["close"].pct_change(20)

    # --------------------------------------------------------
    # Moving averages
    # --------------------------------------------------------

    df["sma_5"] = (
        df["close"]
        .rolling(5, min_periods=5)
        .mean()
    )

    df["sma_20"] = (
        df["close"]
        .rolling(20, min_periods=20)
        .mean()
    )

    df["sma_50"] = (
        df["close"]
        .rolling(50, min_periods=50)
        .mean()
    )

    # --------------------------------------------------------
    # EMA
    # --------------------------------------------------------

    df["ema_20"] = (
        df["close"]
        .ewm(
            span=20,
            adjust=False,
            min_periods=20,
        )
        .mean()
    )

    # --------------------------------------------------------
    # RSI
    # --------------------------------------------------------

    df["rsi_14"] = calculate_rsi(
        df["close"],
        period=14,
    )

    # --------------------------------------------------------
    # Rate of Change / Momentum
    # --------------------------------------------------------

    df["roc_10"] = (
        df["close"]
        .pct_change(10)
    )

    df["roc_20"] = (
        df["close"]
        .pct_change(20)
    )

    # --------------------------------------------------------
    # Volatility
    # --------------------------------------------------------

    df["volatility_5d"] = (
        df["return_1d"]
        .rolling(5, min_periods=5)
        .std()
    )

    df["volatility_20d"] = (
        df["return_1d"]
        .rolling(20, min_periods=20)
        .std()
    )

    # --------------------------------------------------------
    # ATR
    # --------------------------------------------------------

    df["atr_14"] = calculate_atr(
        df["high"],
        df["low"],
        df["close"],
        period=14,
    )

    # ATR normalized by price
    df["atr_pct_14"] = (
        df["atr_14"] / df["close"]
    )

    # --------------------------------------------------------
    # Volume features
    # --------------------------------------------------------

    df["volume_change_1d"] = (
        df["volume"]
        .pct_change(1)
    )

    df["volume_sma_20"] = (
        df["volume"]
        .rolling(20, min_periods=20)
        .mean()
    )

    df["volume_ratio_20d"] = (
        df["volume"]
        / df["volume_sma_20"]
    )

    # --------------------------------------------------------
    # Price relative to moving averages
    # --------------------------------------------------------

    df["close_vs_sma20"] = (
        df["close"] / df["sma_20"] - 1
    )

    df["close_vs_sma50"] = (
        df["close"] / df["sma_50"] - 1
    )

    df["close_vs_ema20"] = (
        df["close"] / df["ema_20"] - 1
    )

    # --------------------------------------------------------
    # Market / NIFTY 50 merge
    # --------------------------------------------------------

    market = market_df[
        ["trade_date", "close"]
    ].copy()

    market = market.rename(
        columns={
            "close": "nifty50_close"
        }
    )

    df = df.merge(
        market,
        on="trade_date",
        how="left",
        validate="one_to_one",
    )

    # --------------------------------------------------------
    # NIFTY 50 returns
    # --------------------------------------------------------

    df["nifty50_return_1d"] = (
        df["nifty50_close"]
        .pct_change(1)
    )

    df["nifty50_return_5d"] = (
        df["nifty50_close"]
        .pct_change(5)
    )

    df["nifty50_return_20d"] = (
        df["nifty50_close"]
        .pct_change(20)
    )

    # --------------------------------------------------------
    # Stock relative to NIFTY 50
    # --------------------------------------------------------

    df["stock_vs_nifty50_5d"] = (
        df["return_5d"]
        - df["nifty50_return_5d"]
    )

    df["stock_vs_nifty50_20d"] = (
        df["return_20d"]
        - df["nifty50_return_20d"]
    )

    # --------------------------------------------------------
    # Market beta-like rolling relationship
    # --------------------------------------------------------

    covariance_20 = (
        df["return_1d"]
        .rolling(20, min_periods=20)
        .cov(df["nifty50_return_1d"])
    )

    market_variance_20 = (
        df["nifty50_return_1d"]
        .rolling(20, min_periods=20)
        .var()
    )

    df["rolling_beta_20d"] = (
        covariance_20
        / market_variance_20.replace(0, np.nan)
    )

    # --------------------------------------------------------
    # Stock volatility relative to market
    # --------------------------------------------------------

    df["nifty50_volatility_20d"] = (
        df["nifty50_return_1d"]
        .rolling(20, min_periods=20)
        .std()
    )

    df["relative_volatility_20d"] = (
        df["volatility_20d"]
        / df["nifty50_volatility_20d"].replace(
            0,
            np.nan,
        )
    )

    # --------------------------------------------------------
    # Feature metadata
    # --------------------------------------------------------

    df["feature_version"] = "technical_v1"

    return df


# ============================================================
# Process one stock
# ============================================================

def process_stock(
    price_file: Path,
    market_df: pd.DataFrame,
) -> tuple[str, int]:

    stock_df = pd.read_parquet(price_file)

    symbol = price_file.stem

    features = calculate_features(
        stock_df,
        market_df,
    )

    # --------------------------------------------------------
    # Basic validation
    # --------------------------------------------------------

    if features["trade_date"].duplicated().any():
        raise ValueError(
            f"{symbol}: duplicate trade dates found"
        )

    if not features["trade_date"].is_monotonic_increasing:
        raise ValueError(
            f"{symbol}: dates are not sorted"
        )

    if features["symbol"].nunique() != 1:
        raise ValueError(
            f"{symbol}: multiple symbols found"
        )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output_file = OUTPUT_DIR / f"{symbol}_features.parquet"

    features.to_parquet(
        output_file,
        index=False,
    )

    return symbol, len(features)


# ============================================================
# Main
# ============================================================

def main() -> None:

    print("=" * 70)
    print("Technical Feature Engineering")
    print("=" * 70)

    if not PRICE_DIR.exists():
        raise FileNotFoundError(
            f"Price directory not found: {PRICE_DIR}"
        )

    if not INDEX_FILE.exists():
        raise FileNotFoundError(
            f"NIFTY 50 file not found: {INDEX_FILE}"
        )

    market_df = pd.read_parquet(INDEX_FILE)

    price_files = sorted(
        PRICE_DIR.glob("*.parquet")
    )

    print(f"Price files found: {len(price_files)}")
    print(f"Market rows: {len(market_df):,}")
    print()

    successful = 0
    failed = 0
    total_rows = 0

    for price_file in price_files:

        symbol = price_file.stem

        try:

            processed_symbol, rows = process_stock(
                price_file,
                market_df,
            )

            successful += 1
            total_rows += rows

            print(
                f"[OK] {processed_symbol:<15} "
                f"{rows:>6,} rows"
            )

        except Exception as exc:

            failed += 1

            print(
                f"[FAILED] {symbol:<15} "
                f"{type(exc).__name__}: {exc}"
            )

    print()
    print("=" * 70)
    print("Feature Engineering Summary")
    print("=" * 70)

    print(f"Files processed : {len(price_files)}")
    print(f"Successful      : {successful}")
    print(f"Failed          : {failed}")
    print(f"Total rows      : {total_rows:,}")
    print(f"Output directory: {OUTPUT_DIR}")

    if failed > 0:
        raise RuntimeError(
            f"Feature engineering failed for {failed} stock(s)"
        )


if __name__ == "__main__":
    main()