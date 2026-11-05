from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils.mvp_config import storage_path


INDEX_FILE = (
    storage_path("processed")
    / "index"
    / "nifty50_daily.parquet"
)

OUTPUT_FILE = (
    storage_path("processed")
    / "calendar"
    / "nse_trading_calendar.parquet"
)


def build_calendar() -> pd.DataFrame:
    if not INDEX_FILE.exists():
        raise FileNotFoundError(f"NIFTY file not found: {INDEX_FILE}")

    df = pd.read_parquet(INDEX_FILE)

    if "trade_date" not in df.columns:
        raise ValueError(
            f"NIFTY file missing trade_date: {INDEX_FILE}"
        )

    dates = pd.to_datetime(
        df["trade_date"],
        errors="coerce",
    )

    if dates.isna().any():
        raise ValueError("NIFTY contains invalid trade dates.")

    dates = (
        dates
        .drop_duplicates()
        .sort_values()
        .reset_index(drop=True)
    )

    if len(dates) < 2:
        raise ValueError("Calendar contains fewer than two dates.")

    calendar = pd.DataFrame({
        "trade_date": dates,
        "is_trading_day": True,
    })

    return calendar


def validate(calendar: pd.DataFrame) -> None:
    if calendar.empty:
        raise ValueError("Calendar is empty.")

    if calendar["trade_date"].duplicated().any():
        raise ValueError("Duplicate calendar dates found.")

    if not calendar["trade_date"].is_monotonic_increasing:
        raise ValueError("Calendar dates are not sorted.")

    if not calendar["is_trading_day"].eq(True).all():
        raise ValueError(
            "Calendar contains non-trading-day rows unexpectedly."
        )


def main() -> None:
    calendar = build_calendar()
    validate(calendar)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    calendar.to_parquet(OUTPUT_FILE, index=False)

    print("=" * 80)
    print("NSE TRADING CALENDAR BUILD")
    print("=" * 80)
    print(f"Input   : {INDEX_FILE}")
    print(f"Output  : {OUTPUT_FILE}")
    print(f"Rows    : {len(calendar):,}")
    print(
        f"Dates   : {calendar['trade_date'].min().date()} -> "
        f"{calendar['trade_date'].max().date()}"
    )
    print("STATUS  : PASSED")


if __name__ == "__main__":
    main()