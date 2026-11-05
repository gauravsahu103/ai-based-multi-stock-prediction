from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = (
    PROJECT_ROOT
    / "data/raw/nse/prices"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data/processed/prices"
)


COLUMN_MAPPING = {
    "Symbol": "symbol",
    "Series": "series",
    "Date": "trade_date",
    "Prev Close": "prev_close",
    "Open Price": "open",
    "High Price": "high",
    "Low Price": "low",
    "Last Price": "last",
    "Close Price": "close",
    "Average Price": "vwap",
    "Total Traded Quantity": "volume",
    "Turnover ₹": "traded_value",
    "No. of Trades": "num_trades",
    "Deliverable Qty": "deliverable_qty",
    "% Dly Qt to Traded Qty": "deliverable_pct",
}


NUMERIC_COLUMNS = [
    "prev_close",
    "open",
    "high",
    "low",
    "last",
    "close",
    "vwap",
    "volume",
    "traded_value",
    "num_trades",
    "deliverable_qty",
    "deliverable_pct",
]


FINAL_COLUMNS = [
    "symbol",
    "series",
    "trade_date",
    "prev_close",
    "open",
    "high",
    "low",
    "last",
    "close",
    "vwap",
    "volume",
    "traded_value",
    "num_trades",
    "deliverable_qty",
    "deliverable_pct",
]


def get_requested_symbol(input_file: Path) -> str:
    """
    Extract the current/requested NSE symbol from the
    raw filename.

    Example:
        SBIN_2021_2026.csv
        -> SBIN
    """

    suffix = "_2021_2026.csv"

    if not input_file.name.endswith(suffix):
        raise ValueError(
            f"Unexpected input filename: "
            f"{input_file.name}"
        )

    return input_file.name[:-len(suffix)]


def parse_price_csv(
    input_file: Path,
) -> pd.DataFrame:

    # ---------------------------------------------------------
    # Read raw CSV
    # ---------------------------------------------------------

    df = pd.read_csv(
        input_file
    )

    if df.empty:
        raise ValueError(
            "CSV contains no rows"
        )

    # ---------------------------------------------------------
    # Clean column names
    # ---------------------------------------------------------

    df.columns = [
        column.strip()
        for column in df.columns
    ]

    # ---------------------------------------------------------
    # Validate expected columns
    # ---------------------------------------------------------

    missing_columns = (
        set(COLUMN_MAPPING.keys())
        - set(df.columns)
    )

    if missing_columns:

        raise ValueError(
            f"Missing columns: "
            f"{sorted(missing_columns)}"
        )

    # ---------------------------------------------------------
    # Rename columns
    # ---------------------------------------------------------

    df = df.rename(
        columns=COLUMN_MAPPING
    )

    # ---------------------------------------------------------
    # Clean symbol / series
    # ---------------------------------------------------------

    df["symbol"] = (
        df["symbol"]
        .astype(str)
        .str.strip()
    )

    df["series"] = (
        df["series"]
        .astype(str)
        .str.strip()
    )

    # ---------------------------------------------------------
    # Historical symbol handling
    # ---------------------------------------------------------
    #
    # NSE may return multiple symbols in a historical
    # report when a company/security has changed symbols.
    #
    # Examples:
    #
    #   ETERNAL + ZOMATO
    #   MOTHERSON + MOTHERSUMI
    #   ADANIENSOL + ADANITRANS
    #
    # Raw CSV is NOT modified.
    #
    # For the normalized current-universe dataset,
    # retain only rows belonging to the requested/current
    # symbol.
    # ---------------------------------------------------------

    requested_symbol = get_requested_symbol(
        input_file
    )

    symbols = (
        df["symbol"]
        .dropna()
        .unique()
    )

    if requested_symbol not in symbols:

        raise ValueError(
            f"Requested symbol "
            f"'{requested_symbol}' not found "
            f"in CSV. Found: "
            f"{list(symbols)}"
        )

    if len(symbols) > 1:

        print(
            f"  Historical aliases detected: "
            f"{list(symbols)}"
        )

        df = df[
            df["symbol"] == requested_symbol
        ].copy()

    # ---------------------------------------------------------
    # Validate final symbol set
    # ---------------------------------------------------------

    symbols = (
        df["symbol"]
        .dropna()
        .unique()
    )

    if len(symbols) != 1:

        raise ValueError(
            f"Expected one symbol after filtering, "
            f"found {list(symbols)}"
        )

    if symbols[0] != requested_symbol:

        raise ValueError(
            f"Expected symbol "
            f"'{requested_symbol}', "
            f"found '{symbols[0]}'"
        )

    # ---------------------------------------------------------
    # Validate series
    # ---------------------------------------------------------

    series_values = (
        df["series"]
        .dropna()
        .unique()
    )

    if not (
        len(series_values) == 1
        and series_values[0] == "EQ"
    ):

        raise ValueError(
            f"{requested_symbol}: "
            f"unexpected series "
            f"{list(series_values)}"
        )

    # ---------------------------------------------------------
    # Parse date
    # ---------------------------------------------------------

    df["trade_date"] = pd.to_datetime(
        df["trade_date"],
        format="%d-%b-%Y",
        errors="coerce",
    )

    if df["trade_date"].isna().any():

        invalid_dates = (
            df.loc[
                df["trade_date"].isna(),
                "trade_date",
            ]
        )

        raise ValueError(
            f"{requested_symbol}: "
            f"invalid trade_date values found"
        )

    # ---------------------------------------------------------
    # Convert numeric columns
    # ---------------------------------------------------------

    for column in NUMERIC_COLUMNS:

        df[column] = (
            df[column]
            .astype(str)
            .str.replace(
                ",",
                "",
                regex=False,
            )
            .str.strip()
        )

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    # ---------------------------------------------------------
    # Validate numeric conversion
    # ---------------------------------------------------------

    for column in NUMERIC_COLUMNS:

        invalid_count = (
            df[column]
            .isna()
            .sum()
        )

        if invalid_count > 0:

            raise ValueError(
                f"{requested_symbol}: "
                f"column '{column}' contains "
                f"{invalid_count} invalid values"
            )

    # ---------------------------------------------------------
    # Duplicate date validation
    # ---------------------------------------------------------

    duplicate_dates = (
        df["trade_date"]
        .duplicated(
            keep=False
        )
    )

    if duplicate_dates.any():

        duplicate_rows = df.loc[
            duplicate_dates,
            [
                "symbol",
                "trade_date",
            ],
        ]

        raise ValueError(
            f"{requested_symbol}: "
            f"duplicate trade dates found:\n"
            f"{duplicate_rows.to_string(index=False)}"
        )

    # ---------------------------------------------------------
    # OHLC validation
    # ---------------------------------------------------------

    invalid_ohlc = df[
        (df["high"] < df["open"])
        | (df["high"] < df["close"])
        | (df["high"] < df["low"])
        | (df["low"] > df["open"])
        | (df["low"] > df["close"])
    ]

    if not invalid_ohlc.empty:

        raise ValueError(
            f"{requested_symbol}: "
            f"invalid OHLC rows: "
            f"{len(invalid_ohlc)}"
        )

    # ---------------------------------------------------------
    # Volume validation
    # ---------------------------------------------------------

    if (df["volume"] < 0).any():

        raise ValueError(
            f"{requested_symbol}: "
            f"negative volume found"
        )

    # ---------------------------------------------------------
    # Traded value validation
    # ---------------------------------------------------------

    if (df["traded_value"] < 0).any():

        raise ValueError(
            f"{requested_symbol}: "
            f"negative traded_value found"
        )

    # ---------------------------------------------------------
    # Trade count validation
    # ---------------------------------------------------------

    if (df["num_trades"] < 0).any():

        raise ValueError(
            f"{requested_symbol}: "
            f"negative num_trades found"
        )

    # ---------------------------------------------------------
    # Deliverable quantity validation
    # ---------------------------------------------------------

    if (df["deliverable_qty"] < 0).any():

        raise ValueError(
            f"{requested_symbol}: "
            f"negative deliverable_qty found"
        )

    # ---------------------------------------------------------
    # Deliverable percentage validation
    # ---------------------------------------------------------

    invalid_deliverable_pct = df[
        (df["deliverable_pct"] < 0)
        | (df["deliverable_pct"] > 100)
    ]

    if not invalid_deliverable_pct.empty:

        raise ValueError(
            f"{requested_symbol}: "
            f"invalid deliverable_pct rows: "
            f"{len(invalid_deliverable_pct)}"
        )

    # ---------------------------------------------------------
    # Sort chronologically
    # ---------------------------------------------------------

    df = (
        df.sort_values(
            "trade_date"
        )
        .reset_index(
            drop=True
        )
    )

    # ---------------------------------------------------------
    # Final schema
    # ---------------------------------------------------------

    df = df[
        FINAL_COLUMNS
    ]

    return df


def process_file(
    input_file: Path,
) -> dict:

    symbol = get_requested_symbol(
        input_file
    )

    output_file = (
        OUTPUT_DIR
        / f"{symbol}.parquet"
    )

    try:

        df = parse_price_csv(
            input_file
        )

        df.to_parquet(
            output_file,
            index=False,
        )

        return {
            "symbol": symbol,
            "status": "SUCCESS",
            "rows": len(df),
            "min_date": (
                df["trade_date"]
                .min()
                .date()
            ),
            "max_date": (
                df["trade_date"]
                .max()
                .date()
            ),
            "output": str(
                output_file
            ),
        }

    except Exception as exc:

        return {
            "symbol": symbol,
            "status": "FAILED",
            "rows": 0,
            "min_date": None,
            "max_date": None,
            "output": None,
            "error": str(exc),
        }


def main() -> None:

    print("=" * 60)
    print("NSE Historical Price Parser")
    print("=" * 60)

    if not RAW_DIR.exists():

        raise FileNotFoundError(
            f"Raw directory not found: "
            f"{RAW_DIR}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    files = sorted(
        RAW_DIR.glob(
            "*_2021_2026.csv"
        )
    )

    print()
    print(
        f"Raw CSV files found: "
        f"{len(files)}"
    )

    if not files:

        raise ValueError(
            "No NSE historical CSV files found"
        )

    results = []

    print()
    print("=" * 60)
    print("Processing")
    print("=" * 60)

    for index, input_file in enumerate(
        files,
        start=1,
    ):

        print()
        print(
            f"[{index:03d}/{len(files):03d}] "
            f"{input_file.name}"
        )

        result = process_file(
            input_file
        )

        results.append(
            result
        )

        if result["status"] == "SUCCESS":

            print(
                f"  SUCCESS | "
                f"Rows: {result['rows']:,} | "
                f"{result['min_date']} → "
                f"{result['max_date']}"
            )

            print(
                f"  Parquet: "
                f"{Path(result['output']).name}"
            )

        else:

            print(
                f"  FAILED | "
                f"{result['error']}"
            )

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    success = [
        result
        for result in results
        if result["status"] == "SUCCESS"
    ]

    failed = [
        result
        for result in results
        if result["status"] == "FAILED"
    ]

    total_rows = sum(
        result["rows"]
        for result in success
    )

    print()
    print("=" * 60)
    print("Parser Summary")
    print("=" * 60)

    print(
        f"Files processed : {len(results)}"
    )

    print(
        f"Successful      : {len(success)}"
    )

    print(
        f"Failed          : {len(failed)}"
    )

    print(
        f"Total rows      : {total_rows:,}"
    )

    if failed:

        print()
        print("Failed symbols:")

        for result in failed:

            print(
                f"  {result['symbol']}: "
                f"{result['error']}"
            )

    print()
    print("=" * 60)
    print(
        "NSE Historical Price Parser Completed"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()