from pathlib import Path
import time
from io import BytesIO

import pandas as pd
import requests


PROJECT_ROOT = Path(__file__).resolve().parents[2]

UNIVERSE_FILE = (
    PROJECT_ROOT
    / "data/processed/universe/nifty100.parquet"
)

RAW_DIR = (
    PROJECT_ROOT
    / "data/raw/nse/prices"
)

PROCESSED_DIR = (
    PROJECT_ROOT
    / "data/processed/prices"
)

FROM_DATE = "18-09-2021"
TO_DATE = "18-09-2026"

BASE_URL = (
    "https://www.nseindia.com/"
    "api/historicalOR/generateSecurityWiseHistoricalData"
)

REQUEST_DELAY_SECONDS = 1.0
MAX_RETRIES = 3


def create_session() -> requests.Session:

    session = requests.Session()

    headers = {
        "User-Agent": (
            "Mozilla/5.0 "
            "(Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 "
            "(KHTML, like Gecko) "
            "Chrome/153.0.0.0 Safari/537.36"
        ),
        "Accept": (
            "text/csv,"
            "application/csv,"
            "application/octet-stream,"
            "*/*"
        ),
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": (
            "https://www.nseindia.com/"
            "report-detail/eq_security"
        ),
        "Connection": "keep-alive",
    }

    session.headers.update(headers)

    print("Establishing NSE session...")

    response = session.get(
        "https://www.nseindia.com/",
        timeout=(15, 30),
    )

    response.raise_for_status()

    print(
        f"NSE homepage status: {response.status_code}"
    )

    return session


def download_symbol(
    session: requests.Session,
    symbol: str,
) -> bytes:

    params = {
        "from": FROM_DATE,
        "to": TO_DATE,
        "symbol": symbol,
        "type": "priceVolumeDeliverable",
        "series": "EQ",
        "csv": "true",
    }

    for attempt in range(1, MAX_RETRIES + 1):

        try:

            response = session.get(
                BASE_URL,
                params=params,
                timeout=(15, 120),
            )

            print(
                f"  Attempt {attempt}: "
                f"HTTP {response.status_code}"
            )

            if response.status_code == 200:

                content_type = (
                    response.headers
                    .get("Content-Type", "")
                    .lower()
                )

                if "text/csv" not in content_type:

                    raise ValueError(
                        f"Unexpected Content-Type: "
                        f"{content_type}"
                    )

                if not response.content:

                    raise ValueError(
                        "Empty response received"
                    )

                return response.content

            if response.status_code in {
                429,
                500,
                502,
                503,
                504,
            }:

                wait_seconds = 2 ** attempt

                print(
                    f"  Retrying in "
                    f"{wait_seconds} seconds..."
                )

                time.sleep(wait_seconds)
                continue

            response.raise_for_status()

        except (
            requests.RequestException,
            ValueError,
        ) as exc:

            print(
                f"  Error: {exc}"
            )

            if attempt < MAX_RETRIES:

                wait_seconds = 2 ** attempt

                print(
                    f"  Retrying in "
                    f"{wait_seconds} seconds..."
                )

                time.sleep(wait_seconds)

            else:
                raise

    raise RuntimeError(
        f"Failed to download {symbol}"
    )


def validate_csv_content(
    content: bytes,
    symbol: str,
) -> pd.DataFrame:

    if not content:

        raise ValueError(
            f"{symbol}: empty CSV"
        )

    try:

        df = pd.read_csv(
            BytesIO(content)
        )

    except Exception as exc:

        raise ValueError(
            f"{symbol}: unable to parse CSV: {exc}"
        ) from exc

    if df.empty:

        raise ValueError(
            f"{symbol}: CSV contains no rows"
        )

    # ---------------------------------------------------------
    # Clean column names
    # ---------------------------------------------------------

    df.columns = [
        column.strip()
        for column in df.columns
    ]

    # ---------------------------------------------------------
    # Validate required columns
    # ---------------------------------------------------------

    required_columns = {
        "Symbol",
        "Series",
        "Date",
    }

    missing = (
        required_columns
        - set(df.columns)
    )

    if missing:

        raise ValueError(
            f"{symbol}: missing columns "
            f"{sorted(missing)}"
        )

    # ---------------------------------------------------------
    # Normalize symbol values
    # ---------------------------------------------------------

    df["Symbol"] = (
        df["Symbol"]
        .astype(str)
        .str.strip()
    )

    # ---------------------------------------------------------
    # NSE can return historical aliases
    #
    # Example:
    #
    # ETERNAL -> ETERNAL + ZOMATO
    # MOTHERSON -> MOTHERSON + MOTHERSUMI
    #
    # We keep the raw CSV unchanged.
    #
    # For this downloader, retain only rows belonging
    # to the requested/current symbol.
    #
    # Historical alias mapping will be handled later
    # by the entity-resolution layer.
    # ---------------------------------------------------------

    csv_symbols = (
        df["Symbol"]
        .dropna()
        .unique()
    )

    if symbol not in csv_symbols:

        raise ValueError(
            f"{symbol}: requested symbol not present "
            f"in response. Returned: "
            f"{list(csv_symbols)}"
        )

    if len(csv_symbols) > 1:

        print(
            f"  Historical aliases detected: "
            f"{list(csv_symbols)}"
        )

        df = df[
            df["Symbol"] == symbol
        ].copy()

    # ---------------------------------------------------------
    # Validate series
    # ---------------------------------------------------------

    df["Series"] = (
        df["Series"]
        .astype(str)
        .str.strip()
    )

    series_values = (
        df["Series"]
        .dropna()
        .unique()
    )

    if not (
        len(series_values) == 1
        and series_values[0] == "EQ"
    ):

        raise ValueError(
            f"{symbol}: unexpected series "
            f"{list(series_values)}"
        )

    # ---------------------------------------------------------
    # Final validation
    # ---------------------------------------------------------

    if df.empty:

        raise ValueError(
            f"{symbol}: no rows remaining "
            f"after symbol filtering"
        )

    return df


def load_universe() -> list[str]:

    if not UNIVERSE_FILE.exists():

        raise FileNotFoundError(
            f"NIFTY100 universe not found: "
            f"{UNIVERSE_FILE}"
        )

    universe = pd.read_parquet(
        UNIVERSE_FILE
    )

    if len(universe) != 100:

        raise ValueError(
            f"Expected 100 stocks, "
            f"found {len(universe)}"
        )

    if "symbol" not in universe.columns:

        raise ValueError(
            "Universe file does not contain "
            "'symbol' column"
        )

    symbols = (
        universe["symbol"]
        .astype(str)
        .str.strip()
        .tolist()
    )

    if len(set(symbols)) != 100:

        raise ValueError(
            "Universe contains duplicate symbols"
        )

    return symbols


def main() -> None:

    print("=" * 60)
    print("NSE Historical Price Downloader")
    print("=" * 60)

    RAW_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    symbols = load_universe()

    print()
    print("Universe : NIFTY 100")
    print(f"Stocks   : {len(symbols)}")
    print(f"From     : {FROM_DATE}")
    print(f"To       : {TO_DATE}")

    session = create_session()

    success = []
    failed = []

    print()
    print("=" * 60)
    print("Starting downloads")
    print("=" * 60)

    for index, symbol in enumerate(
        symbols,
        start=1,
    ):

        print()
        print(
            f"[{index:03d}/{len(symbols):03d}] "
            f"{symbol}"
        )

        raw_file = (
            RAW_DIR
            / f"{symbol}_2021_2026.csv"
        )

        try:

            content = download_symbol(
                session,
                symbol,
            )

            # Validate before writing
            df = validate_csv_content(
                content,
                symbol,
            )

            # -------------------------------------------------
            # Save raw CSV exactly as received from NSE
            # -------------------------------------------------

            raw_file.write_bytes(
                content
            )

            print(
                f"  Rows: {len(df):,}"
            )

            print(
                f"  Raw saved: {raw_file.name}"
            )

            success.append(
                {
                    "symbol": symbol,
                    "rows": len(df),
                    "status": "SUCCESS",
                }
            )

        except Exception as exc:

            print(
                f"  FAILED: {exc}"
            )

            failed.append(
                {
                    "symbol": symbol,
                    "rows": 0,
                    "status": "FAILED",
                    "error": str(exc),
                }
            )

        if index < len(symbols):

            time.sleep(
                REQUEST_DELAY_SECONDS
            )

    print()
    print("=" * 60)
    print("Download Summary")
    print("=" * 60)

    print(
        f"Successful : {len(success)}"
    )

    print(
        f"Failed     : {len(failed)}"
    )

    total_rows = sum(
        item["rows"]
        for item in success
    )

    print(
        f"Total rows : {total_rows:,}"
    )

    if failed:

        print()
        print("Failed symbols:")

        for item in failed:

            print(
                f"  {item['symbol']}: "
                f"{item['error']}"
            )

    print()
    print("=" * 60)
    print("NSE Historical Price Download Completed")
    print("=" * 60)


if __name__ == "__main__":
    main()