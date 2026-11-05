from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.mvp_config import storage_path, targets_day_dir


# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

PRICE_DIR = storage_path("processed_prices")

INDEX_PATH = (
    storage_path("processed_prices").parent
    / "index"
    / "nifty50_daily.parquet"
)

TARGET_DIR = targets_day_dir(5)

TARGET_HORIZON_DAYS = 5


# ---------------------------------------------------------------------------
# Load NIFTY 50 future return
# ---------------------------------------------------------------------------

def load_nifty50_returns() -> pd.DataFrame:
    """
    Load NIFTY 50 daily prices and calculate the future
    5-trading-day NIFTY 50 return.

    nifty50_return_5d:
        NIFTY50 Close[T+5] / Close[T] - 1
    """

    if not INDEX_PATH.exists():
        raise FileNotFoundError(
            f"NIFTY 50 dataset not found: {INDEX_PATH}"
        )

    nifty = pd.read_parquet(INDEX_PATH)

    required_columns = {
        "trade_date",
        "close",
    }

    missing_columns = (
        required_columns
        - set(nifty.columns)
    )

    if missing_columns:
        raise ValueError(
            "NIFTY dataset missing columns: "
            f"{sorted(missing_columns)}"
        )

    nifty = nifty[
        [
            "trade_date",
            "close",
        ]
    ].copy()

    nifty["trade_date"] = pd.to_datetime(
        nifty["trade_date"]
    )

    nifty = (
        nifty
        .sort_values("trade_date")
        .reset_index(drop=True)
    )

    # -----------------------------------------------------------------------
    # Validate duplicate dates
    # -----------------------------------------------------------------------

    if nifty["trade_date"].duplicated().any():
        raise ValueError(
            "Duplicate trade dates found in NIFTY 50 dataset"
        )

    # -----------------------------------------------------------------------
    # Future NIFTY 50 return
    # -----------------------------------------------------------------------

    nifty["nifty50_return_5d"] = (
        nifty["close"].shift(-TARGET_HORIZON_DAYS)
        / nifty["close"]
        - 1.0
    )

    return nifty[
        [
            "trade_date",
            "nifty50_return_5d",
        ]
    ]


# ---------------------------------------------------------------------------
# Generate Day-5 stock targets
# ---------------------------------------------------------------------------

def generate_targets(
    df: pd.DataFrame,
    nifty_returns: pd.DataFrame,
) -> pd.DataFrame:
    """
    Generate all Day-5 targets for a single stock.

    Normal return
    -------------
    target_return_5d:
        Close[T+5] / Close[T] - 1

    Direction
    ---------
    target_direction_5d:
        1 if target_return_5d > 0
        0 otherwise

    Excess return
    -------------
    target_excess_return_5d:
        stock_return_5d - nifty50_return_5d

    T+5 means the fifth subsequent trading observation,
    not five calendar days.
    """

    required_columns = {
        "symbol",
        "trade_date",
        "close",
    }

    missing_columns = (
        required_columns
        - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Missing required columns: "
            f"{sorted(missing_columns)}"
        )

    if df.empty:
        raise ValueError(
            "Input dataframe is empty"
        )

    # -----------------------------------------------------------------------
    # Prepare stock data
    # -----------------------------------------------------------------------

    result = (
        df[
            [
                "symbol",
                "trade_date",
                "close",
            ]
        ]
        .copy()
    )

    result["trade_date"] = pd.to_datetime(
        result["trade_date"]
    )

    result = (
        result
        .sort_values("trade_date")
        .reset_index(drop=True)
    )

    # -----------------------------------------------------------------------
    # Validate duplicate stock dates
    # -----------------------------------------------------------------------

    if result["trade_date"].duplicated().any():
        duplicate_dates = (
            result.loc[
                result["trade_date"].duplicated(
                    keep=False
                ),
                "trade_date",
            ]
            .astype(str)
            .unique()
            .tolist()
        )

        raise ValueError(
            "Duplicate trade dates found: "
            f"{duplicate_dates[:10]}"
        )

    # -----------------------------------------------------------------------
    # Future stock close
    # -----------------------------------------------------------------------

    result["future_close_5d"] = (
        result["close"]
        .shift(-TARGET_HORIZON_DAYS)
    )

    # -----------------------------------------------------------------------
    # Normal Day-5 stock return
    # -----------------------------------------------------------------------

    result["target_return_5d"] = (
        result["future_close_5d"]
        / result["close"]
        - 1.0
    )

    # Keep the stock return explicitly as it is also
    # required for excess-return calculation.
    result["stock_return_5d"] = (
        result["target_return_5d"]
    )

    # -----------------------------------------------------------------------
    # Day-5 direction
    #
    # Positive return -> 1
    # Zero / negative -> 0
    # -----------------------------------------------------------------------

    result["target_direction_5d"] = (
        result["target_return_5d"] > 0
    ).astype("Int64")

    # -----------------------------------------------------------------------
    # Join NIFTY 50 future return
    # -----------------------------------------------------------------------

    result = result.merge(
        nifty_returns,
        on="trade_date",
        how="left",
        validate="one_to_one",
    )

    # -----------------------------------------------------------------------
    # Day-5 excess return
    # -----------------------------------------------------------------------

    result["target_excess_return_5d"] = (
        result["stock_return_5d"]
        - result["nifty50_return_5d"]
    )

    # -----------------------------------------------------------------------
    # Rows without future stock data
    #
    # Final 5 stock observations do not have a future
    # stock price and therefore cannot have a target.
    # -----------------------------------------------------------------------

    no_future_stock_data = (
        result["future_close_5d"].isna()
    )

    result.loc[
        no_future_stock_data,
        [
            "target_return_5d",
            "target_direction_5d",
            "stock_return_5d",
            "target_excess_return_5d",
        ],
    ] = pd.NA

    return result


# ---------------------------------------------------------------------------
# Process one stock
# ---------------------------------------------------------------------------

def process_file(
    input_file: Path,
    nifty_returns: pd.DataFrame,
) -> dict:
    """
    Process one stock price Parquet file.
    """

    symbol = input_file.stem

    output_file = (
        TARGET_DIR
        / f"{symbol}.parquet"
    )

    try:
        # -------------------------------------------------------------------
        # Load stock price data
        # -------------------------------------------------------------------

        df = pd.read_parquet(
            input_file
        )

        original_rows = len(df)

        # -------------------------------------------------------------------
        # Generate all Day-5 targets
        # -------------------------------------------------------------------

        result_df = generate_targets(
            df,
            nifty_returns,
        )

        # -------------------------------------------------------------------
        # Statistics
        # -------------------------------------------------------------------

        valid_return_rows = (
            result_df["target_return_5d"]
            .notna()
            .sum()
        )

        valid_direction_rows = (
            result_df["target_direction_5d"]
            .notna()
            .sum()
        )

        valid_excess_rows = (
            result_df["target_excess_return_5d"]
            .notna()
            .sum()
        )

        missing_return_rows = (
            result_df["target_return_5d"]
            .isna()
            .sum()
        )

        # -------------------------------------------------------------------
        # Validate finite excess-return values
        # -------------------------------------------------------------------

        valid_excess_values = (
            result_df.loc[
                result_df["target_excess_return_5d"].notna(),
                "target_excess_return_5d",
            ]
            .to_numpy(dtype=float)
        )

        if not np.isfinite(
            valid_excess_values
        ).all():
            raise ValueError(
                "Non-finite Day-5 excess return target found"
            )

        # -------------------------------------------------------------------
        # Save
        # -------------------------------------------------------------------

        TARGET_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        result_df.to_parquet(
            output_file,
            index=False,
        )

        return {
            "symbol": symbol,
            "status": "SUCCESS",
            "rows": original_rows,
            "valid_return_targets": int(
                valid_return_rows
            ),
            "valid_direction_targets": int(
                valid_direction_rows
            ),
            "valid_excess_targets": int(
                valid_excess_rows
            ),
            "missing_return_targets": int(
                missing_return_rows
            ),
            "min_date": (
                result_df["trade_date"]
                .min()
                .date()
            ),
            "max_date": (
                result_df["trade_date"]
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
            "valid_return_targets": 0,
            "valid_direction_targets": 0,
            "valid_excess_targets": 0,
            "missing_return_targets": 0,
            "min_date": None,
            "max_date": None,
            "output": None,
            "error": str(exc),
        }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:

    print("=" * 70)
    print("NSE Day-5 Return Target Generator")
    print("=" * 70)

    # -----------------------------------------------------------------------
    # Validate directories/files
    # -----------------------------------------------------------------------

    if not PRICE_DIR.exists():
        raise FileNotFoundError(
            f"Price directory not found: {PRICE_DIR}"
        )

    if not INDEX_PATH.exists():
        raise FileNotFoundError(
            f"NIFTY 50 dataset not found: {INDEX_PATH}"
        )

    TARGET_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -----------------------------------------------------------------------
    # Load NIFTY 50 future returns once
    # -----------------------------------------------------------------------

    nifty_returns = load_nifty50_returns()

    print()
    print(
        f"NIFTY 50 rows: "
        f"{len(nifty_returns):,}"
    )

    # -----------------------------------------------------------------------
    # Find stock price files
    # -----------------------------------------------------------------------

    files = sorted(
        PRICE_DIR.glob("*.parquet")
    )

    print(
        f"Price files found: "
        f"{len(files):,}"
    )

    print(
        f"Target horizon: "
        f"{TARGET_HORIZON_DAYS} trading days"
    )

    print(
        f"Output directory: "
        f"{TARGET_DIR}"
    )

    if not files:
        raise ValueError(
            f"No price Parquet files found in {PRICE_DIR}"
        )

    # -----------------------------------------------------------------------
    # Process stocks
    # -----------------------------------------------------------------------

    results = []

    print()
    print("=" * 70)
    print("Generating Day-5 Targets")
    print("=" * 70)

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
            input_file,
            nifty_returns,
        )

        results.append(result)

        if result["status"] == "SUCCESS":

            print(
                f"  SUCCESS | "
                f"Rows: "
                f"{result['rows']:,}"
            )

            print(
                f"  Return targets: "
                f"{result['valid_return_targets']:,}"
            )

            print(
                f"  Direction targets: "
                f"{result['valid_direction_targets']:,}"
            )

            print(
                f"  Excess targets: "
                f"{result['valid_excess_targets']:,}"
            )

            print(
                f"  Missing return targets: "
                f"{result['missing_return_targets']:,}"
            )

            print(
                f"  Date range: "
                f"{result['min_date']} → "
                f"{result['max_date']}"
            )

        else:

            print(
                f"  FAILED | "
                f"{result['error']}"
            )

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------

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

    total_return_targets = sum(
        result["valid_return_targets"]
        for result in success
    )

    total_direction_targets = sum(
        result["valid_direction_targets"]
        for result in success
    )

    total_excess_targets = sum(
        result["valid_excess_targets"]
        for result in success
    )

    total_missing_targets = sum(
        result["missing_return_targets"]
        for result in success
    )

    print()
    print("=" * 70)
    print("Day-5 Target Generation Summary")
    print("=" * 70)

    print(
        f"Files processed          : "
        f"{len(results)}"
    )

    print(
        f"Successful               : "
        f"{len(success)}"
    )

    print(
        f"Failed                   : "
        f"{len(failed)}"
    )

    print(
        f"Total rows               : "
        f"{total_rows:,}"
    )

    print(
        f"Valid return targets     : "
        f"{total_return_targets:,}"
    )

    print(
        f"Valid direction targets  : "
        f"{total_direction_targets:,}"
    )

    print(
        f"Valid excess targets     : "
        f"{total_excess_targets:,}"
    )

    print(
        f"Missing return targets   : "
        f"{total_missing_targets:,}"
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
    print(
        f"Output directory: "
        f"{TARGET_DIR}"
    )

    print()
    print("=" * 70)
    print(
        "NSE Day-5 Target Generation Completed"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()