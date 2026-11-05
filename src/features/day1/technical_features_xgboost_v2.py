"""
Day-1 Technical Feature Generation - Canonical Version

Purpose
-------
Generate the Day-1 structured technical feature set for the canonical
eligible stock universe.

Important architecture
----------------------
Stock-specific features are calculated from each stock's canonical price
series.

NIFTY market-reference features are NOT calculated inside each stock
dataframe.

Instead, they are read from the standalone canonical market-reference
table:

    data/processed/index/nifty50_market_features.parquet

The stock and market data are joined ONLY by trade_date.

Stock-vs-NIFTY relationship features such as relative returns and rolling
beta are calculated after the market reference has been joined.

Input
-----
Canonical stock prices:
    data/processed/prices_canonical/*.parquet

Canonical NIFTY market reference:
    data/processed/index/nifty50_market_features.parquet

Optional sector mapping:
    data/reference/stock_sector_mapping.parquet

Output
------
data/features/day1/technical_v2/<SYMBOL>_features.parquet

Day-1 universe
--------------
Only stocks present in prices_canonical are processed.

Therefore the current Day-1 universe is 95 stocks.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.mvp_config import storage_path


# =============================================================================
# Paths
# =============================================================================

PRICE_DIR = (
    storage_path("processed")
    / "prices_canonical"
)

MARKET_REFERENCE_FILE = (
    storage_path("processed")
    / "index"
    / "nifty50_market_features.parquet"
)

SECTOR_FILE = (
    storage_path("reference")
    / "stock_sector_mapping.parquet"
)

OUTPUT_DIR = (
    storage_path("features")
    / "day1"
    / "technical_v2"
)


# =============================================================================
# Helpers
# =============================================================================

def div(
    numerator: pd.Series,
    denominator: pd.Series,
) -> pd.Series:
    """
    Safe division.

    Zero denominators are converted to NaN.
    """
    denominator = denominator.replace(0, np.nan)
    return numerator / denominator


def rsi(
    series: pd.Series,
    period: int,
) -> pd.Series:
    """
    Simple rolling RSI.
    """

    delta = series.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(
        period,
        min_periods=period,
    ).mean()

    avg_loss = loss.rolling(
        period,
        min_periods=period,
    ).mean()

    rs = avg_gain / avg_loss.replace(
        0,
        np.nan,
    )

    return 100 - 100 / (1 + rs)


def atr(
    df: pd.DataFrame,
    period: int,
) -> pd.Series:
    """
    Average True Range.
    """

    previous_close = df["close"].shift(1)

    true_range = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - previous_close).abs(),
            (df["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    return true_range.rolling(
        period,
        min_periods=period,
    ).mean()


# =============================================================================
# Market reference
# =============================================================================

MARKET_FEATURE_COLUMNS = [
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


def load_market_reference() -> pd.DataFrame:
    """
    Load the independently calculated canonical NIFTY market-reference
    feature table.
    """

    if not MARKET_REFERENCE_FILE.exists():
        raise FileNotFoundError(
            f"NIFTY market reference not found: "
            f"{MARKET_REFERENCE_FILE}"
        )

    market = pd.read_parquet(
        MARKET_REFERENCE_FILE
    )

    required = {
        "trade_date",
        *MARKET_FEATURE_COLUMNS,
    }

    missing = required - set(market.columns)

    if missing:
        raise ValueError(
            "NIFTY market reference missing columns: "
            f"{sorted(missing)}"
        )

    market = market[
        [
            "trade_date",
            *MARKET_FEATURE_COLUMNS,
        ]
    ].copy()

    market["trade_date"] = pd.to_datetime(
        market["trade_date"],
        errors="coerce",
    )

    if market["trade_date"].isna().any():
        raise ValueError(
            "NIFTY market reference contains invalid trade dates."
        )

    if market["trade_date"].duplicated().any():
        raise ValueError(
            "Duplicate trade dates found in NIFTY market reference."
        )

    market = (
        market
        .sort_values("trade_date")
        .reset_index(drop=True)
    )

    return market


# =============================================================================
# Stock feature generation
# =============================================================================

def add_features(
    df: pd.DataFrame,
    market: pd.DataFrame,
) -> pd.DataFrame:
    """
    Generate stock technical features and merge canonical market-reference
    features.
    """

    df = df.copy()

    required = {
        "symbol",
        "trade_date",
        "open",
        "high",
        "low",
        "close",
        "volume",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Stock data missing required columns: {sorted(missing)}"
        )

    # -------------------------------------------------------------------------
    # Normalize types
    # -------------------------------------------------------------------------

    df["trade_date"] = pd.to_datetime(
        df["trade_date"],
        errors="coerce",
    )

    if df["trade_date"].isna().any():
        raise ValueError(
            "Invalid stock trade dates found."
        )

    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    if df["trade_date"].duplicated().any():
        raise ValueError(
            "Duplicate stock trade dates found."
        )

    df = (
        df
        .sort_values("trade_date")
        .reset_index(drop=True)
    )

    # =========================================================================
    # Existing 24 Day-1 baseline features
    # =========================================================================

    for period in [1, 3, 5, 10, 20]:
        df[f"return_{period}d"] = (
            df["close"].pct_change(period)
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

    df["ema_20"] = (
        df["close"]
        .ewm(
            span=20,
            adjust=False,
            min_periods=20,
        )
        .mean()
    )

    df["rsi_14"] = rsi(
        df["close"],
        14,
    )

    df["roc_10"] = (
        df["close"].pct_change(10)
    )

    df["roc_20"] = (
        df["close"].pct_change(20)
    )

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

    df["atr_pct_14"] = div(
        atr(df, 14),
        df["close"],
    )

    df["volume_change_1d"] = (
        df["volume"].pct_change()
    )

    volume20 = (
        df["volume"]
        .rolling(20, min_periods=20)
        .mean()
    )

    df["volume_ratio_20d"] = div(
        df["volume"],
        volume20,
    )

    df["close_vs_sma20"] = (
        div(df["close"], df["sma_20"]) - 1
    )

    df["close_vs_sma50"] = (
        div(df["close"], df["sma_50"]) - 1
    )

    df["close_vs_ema20"] = (
        div(df["close"], df["ema_20"]) - 1
    )

    # =========================================================================
    # Merge canonical NIFTY market-reference features
    # =========================================================================
    #
    # IMPORTANT:
    # No pct_change(), rolling volatility, or market regime calculation is
    # performed on the stock dataframe for NIFTY features.
    #
    # All market-reference values come from the independently calculated
    # canonical market table.
    # =========================================================================

    df = df.merge(
        market,
        on="trade_date",
        how="left",
        validate="one_to_one",
    )

    # =========================================================================
    # Stock-vs-NIFTY relationship features
    # =========================================================================

    df["stock_vs_nifty50_1d"] = (
        df["return_1d"]
        - df["nifty50_return_1d"]
    )

    df["stock_vs_nifty50_5d"] = (
        df["return_5d"]
        - df["nifty50_return_5d"]
    )

    df["stock_vs_nifty50_20d"] = (
        df["return_20d"]
        - df["nifty50_return_20d"]
    )

    # -------------------------------------------------------------------------
    # Rolling beta - 20 trading observations
    # -------------------------------------------------------------------------

    cov20 = (
        df["return_1d"]
        .rolling(20, min_periods=20)
        .cov(df["nifty50_return_1d"])
    )

    var20 = (
        df["nifty50_return_1d"]
        .rolling(20, min_periods=20)
        .var()
    )

    df["rolling_beta_20d"] = div(
        cov20,
        var20,
    )

    # -------------------------------------------------------------------------
    # Rolling beta - 60 trading observations
    # -------------------------------------------------------------------------

    cov60 = (
        df["return_1d"]
        .rolling(60, min_periods=60)
        .cov(df["nifty50_return_1d"])
    )

    var60 = (
        df["nifty50_return_1d"]
        .rolling(60, min_periods=60)
        .var()
    )

    df["rolling_beta_60d"] = div(
        cov60,
        var60,
    )

    # -------------------------------------------------------------------------
    # Relative volatility
    # -------------------------------------------------------------------------

    df["relative_volatility_20d"] = div(
        df["volatility_20d"],
        df["nifty50_volatility_20d"],
    )

    # =========================================================================
    # Extra 1: price action
    # =========================================================================

    previous_close = df["close"].shift(1)

    trading_range = (
        df["high"] - df["low"]
    )

    df["gap_pct"] = div(
        df["open"] - previous_close,
        previous_close,
    )

    df["intraday_return"] = div(
        df["close"] - df["open"],
        df["open"],
    )

    df["high_low_range_pct"] = div(
        df["high"] - df["low"],
        previous_close,
    )

    df["close_position_in_range"] = div(
        df["close"] - df["low"],
        trading_range,
    )

    df["body_pct"] = div(
        (df["close"] - df["open"]).abs(),
        df["open"],
    )

    df["upper_wick_pct"] = div(
        df["high"]
        - df[["open", "close"]].max(axis=1),
        df["open"],
    )

    df["lower_wick_pct"] = div(
        df[["open", "close"]].min(axis=1)
        - df["low"],
        df["open"],
    )

    # =========================================================================
    # Extra 2: momentum
    # =========================================================================

    df["rsi_7"] = rsi(
        df["close"],
        7,
    )

    df["rsi_21"] = rsi(
        df["close"],
        21,
    )

    ema12 = (
        df["close"]
        .ewm(
            span=12,
            adjust=False,
            min_periods=12,
        )
        .mean()
    )

    ema26 = (
        df["close"]
        .ewm(
            span=26,
            adjust=False,
            min_periods=26,
        )
        .mean()
    )

    df["macd"] = (
        ema12 - ema26
    )

    df["macd_signal"] = (
        df["macd"]
        .ewm(
            span=9,
            adjust=False,
            min_periods=9,
        )
        .mean()
    )

    df["macd_histogram"] = (
        df["macd"]
        - df["macd_signal"]
    )

    lowest_14 = (
        df["low"]
        .rolling(14, min_periods=14)
        .min()
    )

    highest_14 = (
        df["high"]
        .rolling(14, min_periods=14)
        .max()
    )

    df["stochastic_k"] = (
        100
        * div(
            df["close"] - lowest_14,
            highest_14 - lowest_14,
        )
    )

    df["stochastic_d"] = (
        df["stochastic_k"]
        .rolling(3, min_periods=3)
        .mean()
    )

    # -------------------------------------------------------------------------
    # ADX
    # -------------------------------------------------------------------------

    up_move = df["high"].diff()
    down_move = -df["low"].diff()

    plus_dm = up_move.where(
        (up_move > down_move) & (up_move > 0),
        0.0,
    )

    minus_dm = down_move.where(
        (down_move > up_move) & (down_move > 0),
        0.0,
    )

    atr14 = atr(
        df,
        14,
    )

    plus_di = (
        100
        * div(
            plus_dm.rolling(
                14,
                min_periods=14,
            ).mean(),
            atr14,
        )
    )

    minus_di = (
        100
        * div(
            minus_dm.rolling(
                14,
                min_periods=14,
            ).mean(),
            atr14,
        )
    )

    dx = (
        100
        * div(
            (plus_di - minus_di).abs(),
            plus_di + minus_di,
        )
    )

    df["adx_14"] = (
        dx
        .rolling(14, min_periods=14)
        .mean()
    )

    # =========================================================================
    # Extra 3: trend
    # =========================================================================

    sma10 = (
        df["close"]
        .rolling(10, min_periods=10)
        .mean()
    )

    sma100 = (
        df["close"]
        .rolling(100, min_periods=100)
        .mean()
    )

    sma200 = (
        df["close"]
        .rolling(200, min_periods=200)
        .mean()
    )

    ema50 = (
        df["close"]
        .ewm(
            span=50,
            adjust=False,
            min_periods=50,
        )
        .mean()
    )

    df["close_vs_sma10"] = (
        div(df["close"], sma10) - 1
    )

    df["close_vs_sma100"] = (
        div(df["close"], sma100) - 1
    )

    df["close_vs_sma200"] = (
        div(df["close"], sma200) - 1
    )

    df["sma20_vs_sma50"] = (
        div(df["sma_20"], df["sma_50"]) - 1
    )

    df["sma50_vs_sma200"] = (
        div(df["sma_50"], sma200) - 1
    )

    df["sma20_slope"] = (
        div(
            df["sma_20"],
            df["sma_20"].shift(5),
        ) - 1
    )

    df["sma50_slope"] = (
        div(
            df["sma_50"],
            df["sma_50"].shift(5),
        ) - 1
    )

    df["ema20_slope"] = (
        div(
            df["ema_20"],
            df["ema_20"].shift(5),
        ) - 1
    )

    df["ema50_slope"] = (
        div(
            ema50,
            ema50.shift(5),
        ) - 1
    )

    # =========================================================================
    # Extra 4: volatility
    # =========================================================================

    df["volatility_10d"] = (
        df["return_1d"]
        .rolling(10, min_periods=10)
        .std()
    )

    df["volatility_60d"] = (
        df["return_1d"]
        .rolling(60, min_periods=60)
        .std()
    )

    df["atr_pct_7"] = div(
        atr(df, 7),
        df["close"],
    )

    df["atr_pct_21"] = div(
        atr(df, 21),
        df["close"],
    )

    df["volatility_ratio_5_20"] = div(
        df["volatility_5d"],
        df["volatility_20d"],
    )

    df["volatility_ratio_20_60"] = div(
        df["volatility_20d"],
        df["volatility_60d"],
    )

    # =========================================================================
    # Extra 5: volume
    # =========================================================================

    for period in [5, 10, 60]:

        volume_mean = (
            df["volume"]
            .rolling(
                period,
                min_periods=period,
            )
            .mean()
        )

        df[f"volume_ratio_{period}d"] = div(
            df["volume"],
            volume_mean,
        )

    df["dollar_volume"] = (
        df["close"] * df["volume"]
    )

    df["price_volume_corr_20d"] = (
        df["return_1d"]
        .rolling(20, min_periods=20)
        .corr(df["volume"].pct_change())
    )

    return df


# =============================================================================
# Cleanup
# =============================================================================

def cleanup_stale_feature_files(
    active_symbols: set[str],
) -> None:
    """
    Remove feature files belonging to stocks that are no longer in the
    current canonical Day-1 universe.
    """

    if not OUTPUT_DIR.exists():
        return

    removed = 0

    for path in OUTPUT_DIR.glob("*_features.parquet"):

        symbol = path.name.removesuffix(
            "_features.parquet"
        ).upper()

        if symbol not in active_symbols:

            path.unlink()
            removed += 1

            print(
                f"[REMOVE] stale feature file: "
                f"{path.name}"
            )

    if removed:
        print()
        print(
            f"Removed stale feature files: {removed}"
        )
        print()


# =============================================================================
# Main
# =============================================================================

def main() -> None:

    if not PRICE_DIR.exists():
        raise FileNotFoundError(
            f"Canonical price directory not found: {PRICE_DIR}"
        )

    market = load_market_reference()

    files = sorted(
        PRICE_DIR.glob("*.parquet")
    )

    if not files:
        raise FileNotFoundError(
            f"No canonical price files found in: {PRICE_DIR}"
        )

    active_symbols = {
        path.stem.upper()
        for path in files
    }

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    cleanup_stale_feature_files(
        active_symbols
    )

    print("=" * 80)
    print("DAY-1 TECHNICAL FEATURE GENERATION")
    print("=" * 80)

    print(
        f"Canonical price directory : {PRICE_DIR}"
    )

    print(
        f"NIFTY market reference    : {MARKET_REFERENCE_FILE}"
    )

    print(
        f"Stock files                : {len(files)}"
    )

    print(
        f"Market reference rows     : {len(market):,}"
    )

    print(
        f"Market dates              : "
        f"{market['trade_date'].min().date()} -> "
        f"{market['trade_date'].max().date()}"
    )

    print()

    frames: list[pd.DataFrame] = []

    # -------------------------------------------------------------------------
    # Build stock-level features
    # -------------------------------------------------------------------------

    for path in files:

        df = pd.read_parquet(
            path
        )

        symbol_values = (
            df["symbol"]
            .astype(str)
            .str.upper()
            .unique()
        )

        if len(symbol_values) != 1:
            raise ValueError(
                f"Expected exactly one symbol in "
                f"{path.name}, found {symbol_values.tolist()}"
            )

        symbol = symbol_values[0]

        out = add_features(
            df,
            market,
        )

        out["symbol"] = (
            out["symbol"]
            .astype(str)
            .str.upper()
        )

        frames.append(
            out
        )

        print(
            f"[OK] {symbol:<15} "
            f"rows={len(out):>7,}"
        )

    if not frames:
        raise RuntimeError(
            "No stock feature frames were generated."
        )

    combined = pd.concat(
        frames,
        ignore_index=True,
    )

    # =========================================================================
    # Cross-sectional ranking
    #
    # Rankings are calculated only among stocks existing on date T.
    # =========================================================================

    grouped_by_date = (
        combined.groupby(
            "trade_date",
            sort=False,
        )
    )

    combined["return_rank_1d"] = (
        grouped_by_date["return_1d"]
        .rank(pct=True)
    )

    combined["return_rank_5d"] = (
        grouped_by_date["return_5d"]
        .rank(pct=True)
    )

    combined["return_rank_20d"] = (
        grouped_by_date["return_20d"]
        .rank(pct=True)
    )

    combined["relative_strength_rank"] = (
        grouped_by_date["stock_vs_nifty50_20d"]
        .rank(pct=True)
    )

    combined["volume_rank"] = (
        grouped_by_date["volume_ratio_20d"]
        .rank(pct=True)
    )

    # =========================================================================
    # Optional sector-relative features
    # =========================================================================

    if SECTOR_FILE.exists():

        sectors = pd.read_parquet(
            SECTOR_FILE
        )

        required_sector_columns = {
            "symbol",
            "sector",
        }

        missing_sector_columns = (
            required_sector_columns
            - set(sectors.columns)
        )

        if missing_sector_columns:
            raise ValueError(
                "Sector mapping missing columns: "
                f"{sorted(missing_sector_columns)}"
            )

        sectors = (
            sectors[
                ["symbol", "sector"]
            ]
            .copy()
        )

        sectors["symbol"] = (
            sectors["symbol"]
            .astype(str)
            .str.upper()
        )

        if sectors["symbol"].duplicated().any():
            raise ValueError(
                "Duplicate symbols found in sector mapping."
            )

        combined = combined.merge(
            sectors,
            on="symbol",
            how="left",
            validate="many_to_one",
        )

        sector_group = combined.groupby(
            [
                "trade_date",
                "sector",
            ],
            dropna=False,
            sort=False,
        )

        combined["sector_return_1d"] = (
            sector_group["return_1d"]
            .transform("mean")
        )

        combined["sector_return_5d"] = (
            sector_group["return_5d"]
            .transform("mean")
        )

        combined["sector_return_20d"] = (
            sector_group["return_20d"]
            .transform("mean")
        )

        combined["sector_volatility_20d"] = (
            sector_group["volatility_20d"]
            .transform("mean")
        )

        combined["stock_vs_sector_1d"] = (
            combined["return_1d"]
            - combined["sector_return_1d"]
        )

        combined["stock_vs_sector_5d"] = (
            combined["return_5d"]
            - combined["sector_return_5d"]
        )

        combined["stock_vs_sector_20d"] = (
            combined["return_20d"]
            - combined["sector_return_20d"]
        )

        combined["sector_relative_strength"] = (
            combined["stock_vs_sector_20d"]
        )

    # =========================================================================
    # Structural validation
    # =========================================================================

    baseline_features = [
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

    extra_features = [
        "gap_pct",
        "intraday_return",
        "high_low_range_pct",
        "close_position_in_range",
        "body_pct",
        "upper_wick_pct",
        "lower_wick_pct",
        "rsi_7",
        "rsi_21",
        "macd",
        "macd_signal",
        "macd_histogram",
        "stochastic_k",
        "stochastic_d",
        "adx_14",
        "close_vs_sma10",
        "close_vs_sma100",
        "close_vs_sma200",
        "sma20_vs_sma50",
        "sma50_vs_sma200",
        "sma20_slope",
        "sma50_slope",
        "ema20_slope",
        "ema50_slope",
        "volatility_10d",
        "volatility_60d",
        "atr_pct_7",
        "atr_pct_21",
        "volatility_ratio_5_20",
        "volatility_ratio_20_60",
        "volume_ratio_5d",
        "volume_ratio_10d",
        "volume_ratio_60d",
        "dollar_volume",
        "price_volume_corr_20d",
        "nifty50_return_3d",
        "nifty50_volatility_5d",
        "nifty50_volatility_10d",
        "stock_vs_nifty50_1d",
        "rolling_beta_60d",
        "return_rank_1d",
        "return_rank_5d",
        "return_rank_20d",
        "relative_strength_rank",
        "volume_rank",
        "market_trend_regime",
        "market_volatility_regime",
        "market_momentum_regime",
    ]

    if SECTOR_FILE.exists():
        extra_features.extend(
            [
                "sector_return_1d",
                "sector_return_5d",
                "sector_return_20d",
                "stock_vs_sector_1d",
                "stock_vs_sector_5d",
                "stock_vs_sector_20d",
                "sector_volatility_20d",
                "sector_relative_strength",
            ]
        )

    required_features = (
        baseline_features
        + extra_features
    )

    missing_features = [
        feature
        for feature in required_features
        if feature not in combined.columns
    ]

    if missing_features:
        raise RuntimeError(
            "Required features missing: "
            f"{missing_features}"
        )

    # -------------------------------------------------------------------------
    # Duplicate stock/date check
    # -------------------------------------------------------------------------

    duplicate_stock_dates = (
        combined
        .duplicated(
            subset=["symbol", "trade_date"],
            keep=False,
        )
    )

    if duplicate_stock_dates.any():
        raise RuntimeError(
            "Duplicate symbol/trade_date rows found in combined "
            "feature dataset."
        )

    # -------------------------------------------------------------------------
    # Infinite-value check
    # -------------------------------------------------------------------------

    numeric_columns = combined.select_dtypes(
        include=[np.number]
    ).columns

    infinite_columns = []

    for column in numeric_columns:

        values = combined[column].to_numpy(
            dtype=float,
            na_value=np.nan,
        )

        if np.isinf(values).any():
            infinite_columns.append(
                column
            )

    if infinite_columns:
        raise RuntimeError(
            "Infinite values found in feature columns: "
            f"{infinite_columns}"
        )

    # =========================================================================
    # Save one feature file per stock
    # =========================================================================

    for symbol, stock in combined.groupby(
        "symbol",
        sort=True,
    ):

        stock = (
            stock
            .sort_values("trade_date")
            .reset_index(drop=True)
        )

        stock["feature_version"] = (
            "technical_v2_canonical"
        )

        output = (
            OUTPUT_DIR
            / f"{symbol}_features.parquet"
        )

        stock.to_parquet(
            output,
            index=False,
        )

        print(
            f"[SAVED] {output}"
        )

    # =========================================================================
    # Final summary
    # =========================================================================

    stock_count = (
        combined["symbol"]
        .nunique()
    )

    row_count = len(
        combined
    )

    market_missing_rows = int(
        combined[
            "nifty50_return_1d"
        ].isna().sum()
    )

    print()
    print("=" * 80)
    print("DAY-1 FEATURE SUMMARY")
    print("=" * 80)

    print(
        f"Stocks                 : {stock_count}"
    )

    print(
        f"Rows                   : {row_count:,}"
    )

    print(
        f"Baseline 24            : "
        f"{len(baseline_features)}/24"
    )

    print(
        f"Required feature cols  : "
        f"{len(required_features)}"
    )

    print(
        f"Market-missing rows    : "
        f"{market_missing_rows:,}"
    )

    print(
        f"Output                 : {OUTPUT_DIR}"
    )

    print()

    if stock_count != len(files):
        raise RuntimeError(
            "Output stock count does not match canonical price file count."
        )

    print(
        "NIFTY features source  : canonical market reference"
    )

    print(
        "NIFTY features are NOT recalculated per stock."
    )

    print(
        "Sector features are generated only when the optional "
        f"mapping exists: {SECTOR_FILE}"
    )

    print(
        "Fundamentals/events are NOT fabricated by this script."
    )

    print(
        "STATUS                 : PASSED"
    )


if __name__ == "__main__":
    main()