from pathlib import Path

import pandas as pd

from src.utils.mvp_config import storage_path, targets_day_dir


# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

PRICE_DIR = storage_path("processed_prices")
TARGET_DIR = targets_day_dir(1)

TARGET_HORIZON_DAYS = 1


# ---------------------------------------------------------------------------
# Target generation
# ---------------------------------------------------------------------------

def generate_targets(
    df: pd.DataFrame,
    horizon: int = TARGET_HORIZON_DAYS,
) -> pd.DataFrame:
    """
    Generate Day-1 future return and direction targets
    for a single stock.

    Target definition
    -----------------
    target_return_1d:
        Close[T+1] / Close[T] - 1

    target_direction_1d:
        1 if target_return_1d > 0
        0 otherwise

    T+1 means the next available trading observation
    for the same stock, not the next calendar day.

    The final row cannot have a target because its
    future closing price is not available.
    """

    required_columns = {
        "symbol",
        "trade_date",
        "close",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    if horizon <= 0:
        raise ValueError("Horizon must be greater than zero")

    if df.empty:
        raise ValueError("Input dataframe is empty")

    # -----------------------------------------------------------------------
    # Prepare data
    # -----------------------------------------------------------------------

    result = (
        df.copy()
        .sort_values("trade_date")
        .reset_index(drop=True)
    )

    # -----------------------------------------------------------------------
    # Validate duplicate dates
    # -----------------------------------------------------------------------

    if result["trade_date"].duplicated().any():
        duplicate_dates = (
            result.loc[
                result["trade_date"].duplicated(keep=False),
                "trade_date",
            ]
            .astype(str)
            .unique()
            .tolist()
        )

        raise ValueError(
            f"Duplicate trade dates found: {duplicate_dates[:10]}"
        )

    # -----------------------------------------------------------------------
    # Future close
    # -----------------------------------------------------------------------

    future_close = result["close"].shift(-horizon)

    # -----------------------------------------------------------------------
    # Target columns
    # -----------------------------------------------------------------------

    target_return_column = f"target_return_{horizon}d"
    target_direction_column = f"target_direction_{horizon}d"
    future_close_column = f"future_close_{horizon}d"

    # Keep future close explicitly for auditability/debugging.
    result[future_close_column] = future_close

    # -----------------------------------------------------------------------
    # Future return
    # -----------------------------------------------------------------------

    result[target_return_column] = (
        future_close / result["close"] - 1.0
    )

    # -----------------------------------------------------------------------
    # Direction
    #
    # Positive return -> 1
    # Zero / negative -> 0
    # -----------------------------------------------------------------------

    result[target_direction_column] = (
        result[target_return_column] > 0
    ).astype("Int64")

    # -----------------------------------------------------------------------
    # Rows without future data
    #
    # For Day-1 this is normally the final row.
    # -----------------------------------------------------------------------

    no_future_data = future_close.isna()

    result.loc[
        no_future_data,
        target_return_column,
    ] = pd.NA

    result.loc[
        no_future_data,
        target_direction_column,
    ] = pd.NA

    return result


# ---------------------------------------------------------------------------
# Process one stock
# ---------------------------------------------------------------------------

def process_file(input_file: Path) -> dict:
    """
    Process one stock price Parquet file and generate
    its Day-1 return targets.
    """

    symbol = input_file.stem

    output_file = TARGET_DIR / f"{symbol}.parquet"

    try:
        # -------------------------------------------------------------------
        # Load price data
        # -------------------------------------------------------------------

        df = pd.read_parquet(input_file)

        original_rows = len(df)

        # -------------------------------------------------------------------
        # Generate Day-1 targets
        # -------------------------------------------------------------------

        result_df = generate_targets(
            df,
            horizon=TARGET_HORIZON_DAYS,
        )

        target_column = f"target_return_{TARGET_HORIZON_DAYS}d"
        direction_column = f"target_direction_{TARGET_HORIZON_DAYS}d"
        future_close_column = f"future_close_{TARGET_HORIZON_DAYS}d"

        # -------------------------------------------------------------------
        # Target statistics
        # -------------------------------------------------------------------

        valid_target_rows = result_df[target_column].notna().sum()

        missing_target_rows = result_df[target_column].isna().sum()

        valid_direction_rows = (
            result_df[direction_column].notna().sum()
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
            "valid_targets": int(valid_target_rows),
            "missing_targets": int(missing_target_rows),
            "valid_direction_targets": int(valid_direction_rows),
            "min_date": result_df["trade_date"].min().date(),
            "max_date": result_df["trade_date"].max().date(),
            "output": str(output_file),
        }

    except Exception as exc:
        return {
            "symbol": symbol,
            "status": "FAILED",
            "rows": 0,
            "valid_targets": 0,
            "missing_targets": 0,
            "valid_direction_targets": 0,
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
    print("NSE Day-1 Return Target Generator")
    print("=" * 70)

    # -----------------------------------------------------------------------
    # Validate input directory
    # -----------------------------------------------------------------------

    if not PRICE_DIR.exists():
        raise FileNotFoundError(
            f"Price directory not found: {PRICE_DIR}"
        )

    TARGET_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -----------------------------------------------------------------------
    # Find stock price files
    # -----------------------------------------------------------------------

    files = sorted(
        PRICE_DIR.glob("*.parquet")
    )

    print()
    print(f"Price files found : {len(files)}")
    print(
        f"Target horizon    : "
        f"{TARGET_HORIZON_DAYS} trading day"
    )
    print(f"Output directory  : {TARGET_DIR}")

    if not files:
        raise ValueError(
            f"No price Parquet files found in {PRICE_DIR}"
        )

    # -----------------------------------------------------------------------
    # Process files
    # -----------------------------------------------------------------------

    results = []

    print()
    print("=" * 70)
    print("Generating Day-1 Return Targets")
    print("=" * 70)

    for index, input_file in enumerate(files, start=1):

        print()
        print(
            f"[{index:03d}/{len(files):03d}] "
            f"{input_file.name}"
        )

        result = process_file(input_file)

        results.append(result)

        if result["status"] == "SUCCESS":
            print(
                f"  SUCCESS | "
                f"Rows: {result['rows']:,} | "
                f"Valid targets: {result['valid_targets']:,} | "
                f"Missing: {result['missing_targets']:,}"
            )

            print(
                f"  Date range: "
                f"{result['min_date']} -> "
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

    total_valid_targets = sum(
        result["valid_targets"]
        for result in success
    )

    total_missing_targets = sum(
        result["missing_targets"]
        for result in success
    )

    total_valid_direction_targets = sum(
        result["valid_direction_targets"]
        for result in success
    )

    print()
    print("=" * 70)
    print("Day-1 Target Generation Summary")
    print("=" * 70)

    print(f"Files processed         : {len(results)}")
    print(f"Successful              : {len(success)}")
    print(f"Failed                  : {len(failed)}")
    print(f"Total rows              : {total_rows:,}")
    print(f"Valid return targets    : {total_valid_targets:,}")
    print(
        f"Valid direction targets : "
        f"{total_valid_direction_targets:,}"
    )
    print(f"Missing targets         : {total_missing_targets:,}")

    if failed:
        print()
        print("Failed symbols:")

        for result in failed:
            print(
                f"  {result['symbol']}: "
                f"{result['error']}"
            )

    print()
    print(f"Output directory: {TARGET_DIR}")

    print()
    print("=" * 70)
    print("NSE Day-1 Return Target Generation Completed")
    print("=" * 70)


if __name__ == "__main__":
    main()