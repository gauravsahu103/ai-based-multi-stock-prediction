from pathlib import Path
from datetime import datetime, timedelta
import json
import time

import pandas as pd
import requests


# ============================================================
# Configuration
# ============================================================

INDEX_NAME = "NIFTY 50"

START_DATE = datetime(2021, 9, 18)
END_DATE = datetime(2026, 9, 18)

CHUNK_DAYS = 365

RAW_DIR = Path("data/raw/nse/index")
PROCESSED_DIR = Path("data/processed/index")

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

API_URL = "https://www.nseindia.com/api/historicalOR/indicesHistory"

NSE_HOME_URL = "https://www.nseindia.com/"


# ============================================================
# HTTP Headers
# ============================================================

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/153.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json,text/plain,*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/",
    "Connection": "keep-alive",
}


# ============================================================
# Date helpers
# ============================================================

def format_date(value: datetime) -> str:
    return value.strftime("%d-%m-%Y")


def create_date_chunks(
    start_date: datetime,
    end_date: datetime,
):
    """
    Split the requested period into chunks of <= 365 days.
    """

    chunks = []

    current_start = start_date

    while current_start <= end_date:

        current_end = min(
            current_start + timedelta(days=CHUNK_DAYS),
            end_date,
        )

        chunks.append(
            (
                current_start,
                current_end,
            )
        )

        current_start = current_end + timedelta(days=1)

    return chunks


# ============================================================
# NSE Session
# ============================================================

def create_nse_session():

    print("Establishing NSE session...")

    session = requests.Session()

    session.headers.update(HEADERS)

    response = session.get(
        NSE_HOME_URL,
        timeout=30,
    )

    print(
        f"NSE homepage status: {response.status_code}"
    )

    response.raise_for_status()

    return session


# ============================================================
# Download
# ============================================================

def download_chunk(
    session,
    start_date,
    end_date,
    chunk_number,
    total_chunks,
    max_retries=3,
):
    """
    Download one <=365-day chunk from NSE.

    NSE endpoint returns JSON:
        {
            "data": [...]
        }
    """

    from_date = format_date(start_date)
    to_date = format_date(end_date)

    params = {
        "indexType": INDEX_NAME,
        "from": from_date,
        "to": to_date,
        "csv": "true",
    }

    output_file = (
        RAW_DIR
        / f"nifty50_{start_date:%Y%m%d}_{end_date:%Y%m%d}.json"
    )

    for attempt in range(1, max_retries + 1):

        try:

            print(
                f"[{chunk_number}/{total_chunks}] "
                f"Downloading "
                f"{from_date} -> {to_date}"
            )

            response = session.get(
                API_URL,
                params=params,
                timeout=60,
            )

            print(
                f"    Attempt {attempt}: "
                f"HTTP {response.status_code}"
            )

            response.raise_for_status()

            content = response.content

            print(
                f"    Downloaded bytes: "
                f"{len(content):,}"
            )

            if len(content) < 100:

                raise ValueError(
                    "NSE returned an unexpectedly "
                    f"small response: {len(content)} bytes"
                )

            # Verify that the response is valid JSON.
            payload = response.json()

            if not isinstance(payload, dict):

                raise ValueError(
                    "NSE response is not a JSON object"
                )

            if "data" not in payload:

                raise ValueError(
                    "NSE JSON response does not contain "
                    "'data' field"
                )

            if not payload["data"]:

                raise ValueError(
                    "NSE JSON response contains no data"
                )

            # Save the original API response.
            output_file.write_bytes(content)

            print(
                f"    Saved: {output_file}"
            )

            print(
                f"    Records returned: "
                f"{len(payload['data']):,}"
            )

            return output_file

        except Exception as exc:

            print(
                f"    Attempt {attempt} failed: {exc}"
            )

            if attempt < max_retries:

                sleep_seconds = 2 ** attempt

                print(
                    f"    Retrying in "
                    f"{sleep_seconds} seconds..."
                )

                time.sleep(sleep_seconds)

            else:

                raise


# ============================================================
# Parse NSE JSON
# ============================================================

def parse_index_json(
    file_path: Path,
) -> pd.DataFrame:

    with file_path.open(
        "r",
        encoding="utf-8",
    ) as file:

        payload = json.load(file)

    if not isinstance(payload, dict):

        raise ValueError(
            f"Invalid JSON structure in {file_path}"
        )

    records = payload.get("data")

    if not records:

        raise ValueError(
            f"No data records found in {file_path}"
        )

    df = pd.DataFrame(records)

    if df.empty:

        raise ValueError(
            f"Empty dataframe generated from {file_path}"
        )

    print(
        f"    Raw records: {len(df):,}"
    )

    print(
        f"    Raw columns: {list(df.columns)}"
    )

    # ========================================================
    # Rename NSE fields
    # ========================================================

    rename_map = {
        "EOD_INDEX_NAME": "index_name",
        "EOD_OPEN_INDEX_VAL": "open",
        "EOD_HIGH_INDEX_VAL": "high",
        "EOD_LOW_INDEX_VAL": "low",
        "EOD_CLOSE_INDEX_VAL": "close",
        "HIT_TURN_OVER": "turnover_cr",
        "HIT_TRADED_QTY": "volume",
        "EOD_TIMESTAMP": "trade_date",
    }

    df = df.rename(
        columns=rename_map
    )

    required_columns = {
        "index_name",
        "trade_date",
        "open",
        "high",
        "low",
        "close",
    }

    missing_columns = (
        required_columns
        - set(df.columns)
    )

    if missing_columns:

        raise ValueError(
            f"Missing required columns "
            f"{missing_columns} "
            f"in {file_path}"
        )

    # ========================================================
    # Validate index name
    # ========================================================

    unique_index_names = (
        df["index_name"]
        .dropna()
        .astype(str)
        .str.strip()
        .unique()
    )

    if len(unique_index_names) != 1:

        raise ValueError(
            f"Unexpected index names "
            f"in {file_path}: "
            f"{unique_index_names}"
        )

    if unique_index_names[0] != INDEX_NAME:

        raise ValueError(
            f"Expected index '{INDEX_NAME}' "
            f"but received "
            f"'{unique_index_names[0]}'"
        )

    # ========================================================
    # Date parsing
    # ========================================================

    df["trade_date"] = pd.to_datetime(
        df["trade_date"],
        format="%d-%b-%Y",
        errors="coerce",
    )

    if df["trade_date"].isna().any():

        invalid_count = int(
            df["trade_date"].isna().sum()
        )

        raise ValueError(
            f"{invalid_count} invalid dates "
            f"found in {file_path}"
        )

    # ========================================================
    # Numeric conversion
    # ========================================================

    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "turnover_cr",
    ]

    for column in numeric_columns:

        if column not in df.columns:
            continue

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    # ========================================================
    # Numeric validation
    # ========================================================

    price_columns = [
        "open",
        "high",
        "low",
        "close",
    ]

    for column in price_columns:

        if df[column].isna().any():

            raise ValueError(
                f"Null/invalid values found in "
                f"{column} in {file_path}"
            )

        if (df[column] <= 0).any():

            raise ValueError(
                f"Non-positive values found in "
                f"{column} in {file_path}"
            )

    # High must be >= Low.
    if (df["high"] < df["low"]).any():

        raise ValueError(
            f"High < Low detected in {file_path}"
        )

    # Open must be within the day's range.
    if (
        (df["open"] > df["high"])
        | (df["open"] < df["low"])
    ).any():

        raise ValueError(
            f"Open outside High/Low range "
            f"detected in {file_path}"
        )

    # Close must be within the day's range.
    if (
        (df["close"] > df["high"])
        | (df["close"] < df["low"])
    ).any():

        raise ValueError(
            f"Close outside High/Low range "
            f"detected in {file_path}"
        )

    # ========================================================
    # Volume validation
    # ========================================================

    if "volume" in df.columns:

        if df["volume"].isna().any():

            raise ValueError(
                f"Invalid volume values "
                f"found in {file_path}"
            )

        if (df["volume"] < 0).any():

            raise ValueError(
                f"Negative volume detected "
                f"in {file_path}"
            )

    # ========================================================
    # Turnover validation
    # ========================================================

    if "turnover_cr" in df.columns:

        if df["turnover_cr"].isna().any():

            raise ValueError(
                f"Invalid turnover values "
                f"found in {file_path}"
            )

        if (df["turnover_cr"] < 0).any():

            raise ValueError(
                f"Negative turnover detected "
                f"in {file_path}"
            )

    # ========================================================
    # Select normalized columns
    # ========================================================

    output_columns = [
        "index_name",
        "trade_date",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "turnover_cr",
    ]

    output_columns = [
        column
        for column in output_columns
        if column in df.columns
    ]

    df = df[output_columns].copy()

    # ========================================================
    # Duplicate validation within chunk
    # ========================================================

    duplicate_mask = df["trade_date"].duplicated(
        keep=False
    )

    if duplicate_mask.any():

        duplicate_count = int(
            duplicate_mask.sum()
        )

        raise ValueError(
            f"{duplicate_count} duplicate "
            f"trade-date rows found in "
            f"{file_path}"
        )

    # ========================================================
    # Sort ascending
    # ========================================================

    df = (
        df
        .sort_values("trade_date")
        .reset_index(drop=True)
    )

    return df


# ============================================================
# Combine chunks
# ============================================================

def combine_chunks(
    raw_files,
) -> pd.DataFrame:

    frames = []

    print()
    print("=" * 70)
    print("PARSING")
    print("=" * 70)

    for file_path in raw_files:

        print(
            f"Parsing: {file_path}"
        )

        try:

            df = parse_index_json(
                file_path
            )

            print(
                f"    Rows: {len(df):,}"
            )

            print(
                f"    Date range: "
                f"{df['trade_date'].min():%Y-%m-%d}"
                f" -> "
                f"{df['trade_date'].max():%Y-%m-%d}"
            )

            frames.append(df)

        except Exception as exc:

            print(
                f"    ERROR: {exc}"
            )

    if not frames:

        raise RuntimeError(
            "No valid NIFTY 50 data "
            "was parsed."
        )

    print()
    print("=" * 70)
    print("COMBINING")
    print("=" * 70)

    combined = pd.concat(
        frames,
        ignore_index=True,
    )

    rows_before = len(combined)

    combined = (
        combined
        .drop_duplicates(
            subset=["trade_date"],
            keep="first",
        )
        .sort_values("trade_date")
        .reset_index(drop=True)
    )

    duplicates_removed = (
        rows_before - len(combined)
    )

    print(
        f"Rows before deduplication : "
        f"{rows_before:,}"
    )

    print(
        f"Duplicates removed        : "
        f"{duplicates_removed:,}"
    )

    print(
        f"Final rows                : "
        f"{len(combined):,}"
    )

    print(
        f"Final date range          : "
        f"{combined['trade_date'].min():%Y-%m-%d}"
        f" -> "
        f"{combined['trade_date'].max():%Y-%m-%d}"
    )

    return combined


# ============================================================
# Final validation
# ============================================================

def validate_final_dataset(
    df: pd.DataFrame,
):

    print()
    print("=" * 70)
    print("FINAL VALIDATION")
    print("=" * 70)

    if df.empty:

        raise ValueError(
            "Final NIFTY 50 dataset is empty."
        )

    if df["trade_date"].duplicated().any():

        raise ValueError(
            "Duplicate dates remain "
            "after deduplication."
        )

    if not df["trade_date"].is_monotonic_increasing:

        raise ValueError(
            "Trade dates are not sorted "
            "ascending."
        )

    if df["trade_date"].min().date() > START_DATE.date():

        print(
            "WARNING: earliest available "
            f"date is {df['trade_date'].min():%Y-%m-%d}"
        )

    if df["trade_date"].max().date() < END_DATE.date():

        print(
            "WARNING: latest available "
            f"date is {df['trade_date'].max():%Y-%m-%d}"
        )

    print("Duplicate dates : 0")
    print("Date ordering   : OK")
    print("OHLC validation : OK")

    print()
    print("Dataset summary:")
    print(df.describe(include="all"))


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("NSE NIFTY 50 HISTORICAL DATA INGESTION")
    print("=" * 70)

    print(
        f"Index      : {INDEX_NAME}"
    )

    print(
        f"Start date : {START_DATE:%d-%m-%Y}"
    )

    print(
        f"End date   : {END_DATE:%d-%m-%Y}"
    )

    print(
        f"Chunk size : {CHUNK_DAYS} days"
    )

    print()

    # --------------------------------------------------------
    # Create date chunks
    # --------------------------------------------------------

    chunks = create_date_chunks(
        START_DATE,
        END_DATE,
    )

    print(
        f"Total chunks: {len(chunks)}"
    )

    print()

    # --------------------------------------------------------
    # Create NSE session
    # --------------------------------------------------------

    session = create_nse_session()

    raw_files = []

    # --------------------------------------------------------
    # Download chunks
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("DOWNLOADING")
    print("=" * 70)

    for number, (
        chunk_start,
        chunk_end,
    ) in enumerate(
        chunks,
        start=1,
    ):

        try:

            file_path = download_chunk(
                session=session,
                start_date=chunk_start,
                end_date=chunk_end,
                chunk_number=number,
                total_chunks=len(chunks),
            )

            raw_files.append(
                file_path
            )

        except Exception as exc:

            print(
                f"    FAILED: "
                f"{chunk_start:%d-%m-%Y}"
                f" -> "
                f"{chunk_end:%d-%m-%Y}"
            )

            print(
                f"    Reason: {exc}"
            )

        # Be polite to NSE.
        time.sleep(1)

    if not raw_files:

        raise RuntimeError(
            "No NIFTY 50 data files "
            "were downloaded."
        )

    # --------------------------------------------------------
    # Parse and combine
    # --------------------------------------------------------

    combined = combine_chunks(
        raw_files
    )

    # --------------------------------------------------------
    # Final validation
    # --------------------------------------------------------

    validate_final_dataset(
        combined
    )

    # --------------------------------------------------------
    # Save processed Parquet
    # --------------------------------------------------------

    output_file = (
        PROCESSED_DIR
        / "nifty50_daily.parquet"
    )

    combined.to_parquet(
        output_file,
        index=False,
    )

    # --------------------------------------------------------
    # Success
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("SUCCESS")
    print("=" * 70)

    print(
        f"Output : {output_file}"
    )

    print(
        f"Rows   : {len(combined):,}"
    )

    print(
        f"From   : "
        f"{combined['trade_date'].min():%Y-%m-%d}"
    )

    print(
        f"To     : "
        f"{combined['trade_date'].max():%Y-%m-%d}"
    )

    print()
    print("First 5 rows:")
    print(
        combined.head().to_string(
            index=False
        )
    )

    print()
    print("Last 5 rows:")
    print(
        combined.tail().to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()