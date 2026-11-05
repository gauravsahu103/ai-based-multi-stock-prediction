"""
Download NSE reference datasets needed for the Phase-1 correction pass.

Sources
-------
- NSE symbol change archive:
  https://nsearchives.nseindia.com/content/equities/symbolchange.csv
- NSE corporate-actions public API:
  https://www.nseindia.com/api/corporates-corporateActions

The corporate-action API requires an NSE homepage session/cookies.

Outputs
-------
data/raw/nse/reference/symbolchange.csv
data/raw/nse/reference/corporate_actions_<from>_<to>.csv

This script does not alter price data.
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
import re
import time

import pandas as pd
import requests


PROJECT_ROOT = Path(__file__).resolve().parents[2]

REFERENCE_DIR = PROJECT_ROOT / "data/raw/nse/reference"

SYMBOL_CHANGE_URL = (
    "https://nsearchives.nseindia.com/content/equities/symbolchange.csv"
)

CORPORATE_ACTION_URL = (
    "https://www.nseindia.com/api/corporates-corporateActions"
)

FROM_DATE = "01-01-2021"
TO_DATE = "31-12-2026"

# Large ranges may be limited by the NSE endpoint. Keep this configurable.
CHUNK_DAYS = 90


HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/153.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json,text/csv,*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/",
    "Connection": "keep-alive",
}


def create_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(HEADERS)

    response = session.get(
        "https://www.nseindia.com/",
        timeout=(15, 30),
    )
    response.raise_for_status()

    return session


def download_symbol_changes() -> Path:
    response = requests.get(
        SYMBOL_CHANGE_URL,
        headers=HEADERS,
        timeout=(15, 60),
    )
    response.raise_for_status()

    output = REFERENCE_DIR / "symbolchange.csv"
    output.write_bytes(response.content)

    print(f"[OK] Symbol changes: {output}")
    return output


def date_chunks() -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    start = pd.to_datetime(FROM_DATE, dayfirst=True)
    end = pd.to_datetime(TO_DATE, dayfirst=True)

    chunks = []
    current = start

    while current <= end:
        chunk_end = min(
            current + pd.Timedelta(days=CHUNK_DAYS - 1),
            end,
        )
        chunks.append((current, chunk_end))
        current = chunk_end + pd.Timedelta(days=1)

    return chunks


def fetch_corporate_actions(
    session: requests.Session,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    params = {
        "index": "equities",
        "from_date": start.strftime("%d-%m-%Y"),
        "to_date": end.strftime("%d-%m-%Y"),
    }

    response = session.get(
        CORPORATE_ACTION_URL,
        params=params,
        timeout=(15, 120),
    )
    response.raise_for_status()

    content_type = response.headers.get("Content-Type", "").lower()

    # NSE can return CSV or JSON depending on endpoint behavior.
    if "json" in content_type:
        payload = response.json()

        # NSE may return either:
        #   1) a top-level list of action records, or
        #   2) an object containing the records under ``data``.
        if isinstance(payload, list):
            records = payload
        elif isinstance(payload, dict):
            records = payload.get("data", payload)
        else:
            raise ValueError(
                f"Unexpected JSON corporate-action response: "
                f"{type(payload).__name__}"
            )

        if not isinstance(records, list):
            raise ValueError(
                f"Unexpected corporate-action records type: "
                f"{type(records).__name__}"
            )

        return pd.DataFrame(records)

    return pd.read_csv(BytesIO(response.content))


def _normalise_column_name(value: object) -> str:
    """Convert NSE column names to a predictable snake_case form."""
    text = str(value).strip()
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", text)
    text = re.sub(r"[^0-9A-Za-z]+", "_", text)
    return text.strip("_").lower()


# NSE's current corporate-actions API uses camelCase/short field names such as
# exDate, recDate, comp, faceVal and subject. The CSV/table representation may
# use labels such as EX-DATE or COMPANY NAME. Accept both forms, but emit one
# canonical schema for downstream preprocessing.
CORPORATE_ACTION_COLUMN_ALIASES = {
    "symbol": "symbol",
    "comp": "company_name",
    "company": "company_name",
    "company_name": "company_name",
    "series": "series",
    "face_val": "face_value",
    "face_value": "face_value",
    "subject": "purpose",
    "purpose": "purpose",
    "ex_date": "ex_date",
    "rec_date": "record_date",
    "record_date": "record_date",
    "bc_start_date": "book_closure_start_date",
    "book_closure_start_date": "book_closure_start_date",
    "bc_end_date": "book_closure_end_date",
    "book_closure_end_date": "book_closure_end_date",
    "pay_date": "payment_date",
    "payment_date": "payment_date",
    "remarks": "remarks",
}


def normalize_corporate_actions(df: pd.DataFrame) -> pd.DataFrame:
    canonical_columns = [
        "symbol",
        "company_name",
        "series",
        "purpose",
        "face_value",
        "ex_date",
        "record_date",
        "book_closure_start_date",
        "book_closure_end_date",
        "payment_date",
        "remarks",
    ]

    if df.empty:
        return pd.DataFrame(columns=canonical_columns)

    df = df.copy()
    df.columns = [_normalise_column_name(c) for c in df.columns]

    rename = {
        column: CORPORATE_ACTION_COLUMN_ALIASES[column]
        for column in df.columns
        if column in CORPORATE_ACTION_COLUMN_ALIASES
    }
    df = df.rename(columns=rename)

    required = {"symbol", "purpose", "ex_date"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            "Corporate-action response missing columns: "
            f"{sorted(missing)}. Received columns: {list(df.columns)}"
        )

    # Ensure optional canonical columns exist so concatenation has one stable schema.
    for column in canonical_columns:
        if column not in df.columns:
            df[column] = pd.NA

    df["symbol"] = (
        df["symbol"].astype("string").str.strip().str.upper()
    )
    df["purpose"] = df["purpose"].astype("string").str.strip()
    df["company_name"] = df["company_name"].astype("string").str.strip()
    df["series"] = df["series"].astype("string").str.strip().str.upper()

    for column in [
        "ex_date",
        "record_date",
        "book_closure_start_date",
        "book_closure_end_date",
        "payment_date",
    ]:
        df[column] = pd.to_datetime(
            df[column], errors="coerce", dayfirst=True
        )

    return df[canonical_columns]


def main() -> None:
    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("NSE REFERENCE DATA DOWNLOAD")
    print("=" * 80)

    download_symbol_changes()

    session = create_session()
    all_frames: list[pd.DataFrame] = []

    chunks = date_chunks()
    print(f"Corporate-action date chunks: {len(chunks)}")

    for index, (start, end) in enumerate(chunks, start=1):
        print(
            f"[{index:03d}/{len(chunks):03d}] "
            f"{start.date()} -> {end.date()}"
        )

        try:
            frame = fetch_corporate_actions(session, start, end)
            frame = normalize_corporate_actions(frame)

            if not frame.empty:
                all_frames.append(frame)
                print(f"  rows={len(frame):,}")
            else:
                print("  rows=0")

        except Exception as exc:
            print(f"  FAILED: {exc}")
            raise

        time.sleep(0.5)

    if not all_frames:
        raise RuntimeError(
            "No corporate actions were returned for the requested period."
        )

    corporate_actions = pd.concat(
        all_frames,
        ignore_index=True,
    )

    dedupe_columns = [
        c for c in [
            "symbol",
            "company_name",
            "series",
            "purpose",
            "face_value",
            "ex_date",
            "record_date",
            "book_closure_start_date",
            "book_closure_end_date",
        ]
        if c in corporate_actions.columns
    ]

    corporate_actions = (
        corporate_actions
        .drop_duplicates(subset=dedupe_columns)
        .sort_values(["symbol", "ex_date", "purpose"])
        .reset_index(drop=True)
    )

    output = (
        REFERENCE_DIR
        / f"corporate_actions_2021_2026.csv"
    )
    corporate_actions.to_csv(output, index=False)

    print()
    print(f"[OK] Corporate actions: {output}")
    print(f"Rows: {len(corporate_actions):,}")
    print("STATUS: PASSED")


if __name__ == "__main__":
    main()
