"""
Calendar-aware Day-1 target generation.

Purpose
-------
Generate target_return_1d using the canonical exchange trading calendar.

Important
---------
The old target generator used shift(-1) on whatever rows happened to
exist for a stock. That can turn a "1-day" target into a many-calendar-day
return when a stock has missing observations.

This version uses the canonical trading calendar:

    T
    ↓
next canonical NSE trading date
    ↓
stock close on that exact date
    ↓
target_return_1d

If the stock has no close on the exact next canonical trading date,
the target is NA.

It does NOT jump to a later stock observation.

Input
-----
Canonical stock prices:
    data/processed/prices_canonical/*.parquet

Canonical trading calendar:
    data/processed/calendar/nse_trading_calendar.parquet

Calendar columns
----------------
Required:
    trade_date

Optional:
    is_trading_day

Output
------
data/targets/day1/<SYMBOL>.parquet

Important
---------
Only stocks present in prices_canonical are processed.

This prevents previously generated target files for excluded stocks from
remaining active in the Day-1 pipeline.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils.mvp_config import storage_path, targets_day_dir


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PRICE_DIR = (
    storage_path("processed")
    / "prices_canonical"
)

CALENDAR_DIR = (
    storage_path("processed")
    / "calendar"
)

TARGET_DIR = targets_day_dir(1)


# ---------------------------------------------------------------------------
# Calendar
# ---------------------------------------------------------------------------

def find_calendar() -> Path:
    """
    Locate the canonical NSE trading calendar.
    """

    preferred = [
        CALENDAR_DIR / "nse_trading_calendar.parquet",
        CALENDAR_DIR / "trading_calendar.parquet",
    ]

    for path in preferred:
        if path.exists():
            return path

    candidates = sorted(CALENDAR_DIR.glob("*.parquet"))

    if len(candidates) == 1:
        return candidates[0]

    raise FileNotFoundError(
        "Could not uniquely locate the canonical NSE trading calendar "
        f"in {CALENDAR_DIR}. Expected one of: "
        "nse_trading_calendar.parquet or trading_calendar.parquet."
    )


def load_calendar(path: Path) -> pd.DatetimeIndex:
    """
    Load and validate the canonical exchange trading calendar.
    """

    cal = pd.read_parquet(path)

    if "trade_date" not in cal.columns:
        raise ValueError(
            f"Calendar missing trade_date: {path}"
        )

    cal["trade_date"] = pd.to_datetime(
        cal["trade_date"],
        errors="coerce",
    )

    cal = cal.dropna(
        subset=["trade_date"]
    ).copy()

    if "is_trading_day" in cal.columns:
        cal = cal[
            cal["is_trading_day"].astype(bool)
        ].copy()

    if cal["trade_date"].duplicated().any():
        raise ValueError(
            f"Duplicate trade dates found in calendar: {path}"
        )

    dates = (
        cal["trade_date"]
        .sort_values()
        .reset_index(drop=True)
    )

    if len(dates) < 2:
        raise ValueError(
            "Calendar contains fewer than two trading dates."
        )

    return pd.DatetimeIndex(dates)


# ---------------------------------------------------------------------------
# Target construction
# ---------------------------------------------------------------------------

def build_target(
    df: pd.DataFrame,
    calendar: pd.DatetimeIndex,
) -> pd.DataFrame:
    """
    Build exact T -> next canonical trading date target for one stock.
    """

    required = {
        "symbol",
        "trade_date",
        "close",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Price data missing columns: {sorted(missing)}"
        )

    result = df.copy()

    # -----------------------------------------------------------------------
    # Normalize types
    # -----------------------------------------------------------------------

    result["trade_date"] = pd.to_datetime(
        result["trade_date"],
        errors="coerce",
    )

    result["close"] = pd.to_numeric(
        result["close"],
        errors="coerce",
    )

    if result["trade_date"].isna().any():
        raise ValueError(
            "Invalid price dates found."
        )

    if result["close"].isna().any():
        raise ValueError(
            "Missing close values found in canonical price data."
        )

    # -----------------------------------------------------------------------
    # One stock row per trading date
    # -----------------------------------------------------------------------

    if result["trade_date"].duplicated().any():
        duplicates = (
            result.loc[
                result["trade_date"].duplicated(keep=False),
                "trade_date",
            ]
            .dt.strftime("%Y-%m-%d")
            .unique()
            .tolist()
        )

        raise ValueError(
            "Duplicate stock price dates found. "
            f"Dates: {duplicates[:10]}"
        )

    result = (
        result
        .sort_values("trade_date")
        .reset_index(drop=True)
    )

    # -----------------------------------------------------------------------
    # Build exact calendar successor mapping
    #
    # Example:
    #
    # 2026-09-16 -> 2026-09-17
    # 2026-09-17 -> NaT
    # -----------------------------------------------------------------------

    calendar_df = pd.DataFrame({
        "trade_date": calendar,
    })

    calendar_df["next_trade_date"] = (
        calendar_df["trade_date"].shift(-1)
    )

    # -----------------------------------------------------------------------
    # Attach the canonical next trading date to the stock row.
    #
    # one stock row -> one calendar date
    # -----------------------------------------------------------------------

    result = result.merge(
        calendar_df,
        on="trade_date",
        how="left",
        validate="one_to_one",
    )

    # -----------------------------------------------------------------------
    # Build exact-date future close lookup.
    #
    # IMPORTANT:
    # The right-hand side contains every actual stock trade date.
    # We do not use shift(-1), and we do not generate the future date from
    # the stock observations.
    # -----------------------------------------------------------------------

    future = (
        result[["trade_date", "close"]]
        .rename(
            columns={
                "trade_date": "next_trade_date",
                "close": "future_close_1d",
            }
        )
    )

    # -----------------------------------------------------------------------
    # Merge future close.
    #
    # A stock may have multiple source rows mapping to the same
    # next_trade_date only in the sense that multiple current rows can
    # reference dates; therefore the LEFT side is many-to-one.
    #
    # The RIGHT side is unique because trade_date was validated unique.
    # -----------------------------------------------------------------------

    result = result.merge(
        future,
        on="next_trade_date",
        how="left",
        validate="many_to_one",
    )

    # -----------------------------------------------------------------------
    # Calculate target
    # -----------------------------------------------------------------------

    result["target_return_1d"] = (
        result["future_close_1d"]
        / result["close"]
        - 1.0
    )

    result["target_direction_1d"] = (
        result["target_return_1d"] > 0
    ).astype("Int64")

    # -----------------------------------------------------------------------
    # Missing exact next trading session:
    #
    # - no next calendar date
    # - stock has no observation on next calendar date
    #
    # Both cases must produce NA.
    # -----------------------------------------------------------------------

    missing_future = (
        result["future_close_1d"].isna()
    )

    result.loc[
        missing_future,
        "target_return_1d",
    ] = pd.NA

    result.loc[
        missing_future,
        "target_direction_1d",
    ] = pd.NA

    # -----------------------------------------------------------------------
    # Calendar gap in calendar days.
    # -----------------------------------------------------------------------

    result["target_gap_days"] = (
        result["next_trade_date"]
        - result["trade_date"]
    ).dt.days

    return result


# ---------------------------------------------------------------------------
# Output cleanup
# ---------------------------------------------------------------------------

def cleanup_stale_target_files(
    active_symbols: set[str],
) -> None:
    """
    Remove stale target parquet files that are not part of the current
    canonical Day-1 stock universe.

    This is important because five stocks were intentionally excluded from
    the Day-1 canonical price pipeline.
    """

    if not TARGET_DIR.exists():
        return

    removed = 0

    for path in TARGET_DIR.glob("*.parquet"):
        symbol = path.stem.upper()

        if symbol not in active_symbols:
            path.unlink()
            removed += 1
            print(
                f"[REMOVE] stale target file: {path.name}"
            )

    if removed:
        print(
            f"Removed stale target files: {removed}"
        )
        print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:

    # -----------------------------------------------------------------------
    # Validate paths
    # -----------------------------------------------------------------------

    if not PRICE_DIR.exists():
        raise FileNotFoundError(
            f"Canonical price directory not found: {PRICE_DIR}"
        )

    calendar_path = find_calendar()

    calendar = load_calendar(
        calendar_path
    )

    files = sorted(
        PRICE_DIR.glob("*.parquet")
    )

    if not files:
        raise FileNotFoundError(
            f"No canonical price files found in {PRICE_DIR}"
        )

    # -----------------------------------------------------------------------
    # Determine active Day-1 universe
    # -----------------------------------------------------------------------

    active_symbols = {
        path.stem.upper()
        for path in files
    }

    # -----------------------------------------------------------------------
    # Prepare target directory
    # -----------------------------------------------------------------------

    TARGET_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Remove target files from stocks excluded from the canonical pipeline.
    cleanup_stale_target_files(
        active_symbols
    )

    # -----------------------------------------------------------------------
    # Header
    # -----------------------------------------------------------------------

    print("=" * 80)
    print(
        "CALENDAR-AWARE DAY-1 TARGET GENERATION"
    )
    print("=" * 80)

    print(
        f"Canonical price directory : {PRICE_DIR}"
    )

    print(
        f"Price files               : {len(files)}"
    )

    print(
        f"Calendar                  : {calendar_path}"
    )

    print(
        f"Calendar days             : {len(calendar):,}"
    )

    print()

    # -----------------------------------------------------------------------
    # Process stocks
    # -----------------------------------------------------------------------

    total_rows = 0
    total_valid = 0
    total_missing = 0

    successful_symbols: list[str] = []

    for path in files:

        df = pd.read_parquet(path)

        result = build_target(
            df,
            calendar,
        )

        # -------------------------------------------------------------------
        # Validate symbol
        # -------------------------------------------------------------------

        symbols = (
            result["symbol"]
            .astype(str)
            .str.upper()
            .unique()
        )

        if len(symbols) != 1:
            raise ValueError(
                f"Expected exactly one symbol in {path.name}, "
                f"found: {symbols.tolist()}"
            )

        symbol = symbols[0]

        # -------------------------------------------------------------------
        # Output
        # -------------------------------------------------------------------

        output = (
            TARGET_DIR
            / f"{symbol}.parquet"
        )

        result.to_parquet(
            output,
            index=False,
        )

        valid = int(
            result["target_return_1d"]
            .notna()
            .sum()
        )

        missing = int(
            result["target_return_1d"]
            .isna()
            .sum()
        )

        total_rows += len(result)
        total_valid += valid
        total_missing += missing

        successful_symbols.append(
            symbol
        )

        print(
            f"[OK] {symbol:<15}"
            f"rows={len(result):>7,} "
            f"valid={valid:>7,} "
            f"missing={missing:>4,}"
        )

    # -----------------------------------------------------------------------
    # Final validation
    # -----------------------------------------------------------------------

    successful_symbol_set = set(
        successful_symbols
    )

    if successful_symbol_set != active_symbols:
        missing_outputs = sorted(
            active_symbols
            - successful_symbol_set
        )

        unexpected_outputs = sorted(
            successful_symbol_set
            - active_symbols
        )

        raise RuntimeError(
            "Target output universe mismatch. "
            f"Missing outputs: {missing_outputs}; "
            f"Unexpected outputs: {unexpected_outputs}"
        )

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------

    print()
    print("-" * 80)
    print(
        f"Day-1 universe  : {len(successful_symbol_set)}"
    )
    print(
        f"Total rows      : {total_rows:,}"
    )
    print(
        f"Valid targets   : {total_valid:,}"
    )
    print(
        f"Missing targets : {total_missing:,}"
    )
    print(
        f"Output          : {TARGET_DIR}"
    )
    print(
        "STATUS          : PASSED"
    )


if __name__ == "__main__":
    main()