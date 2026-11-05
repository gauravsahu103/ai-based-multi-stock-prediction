from pathlib import Path

import requests


PROJECT_ROOT = Path(__file__).resolve().parents[2]

OUTPUT_DIR = PROJECT_ROOT / "data/raw/nse/prices"
OUTPUT_FILE = OUTPUT_DIR / "SBIN_test.csv"

URL = "https://www.nseindia.com/api/historical/cm/equity"

PARAMS = {
    "symbol": "SBIN",
    "series": '["EQ"]',
    "from": "01-01-2025",
    "to": "31-01-2025",
}


def main() -> None:

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    session = requests.Session()

    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json,text/plain,*/*",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.nseindia.com/",
    }

    print("=" * 60)
    print("NSE Historical Price Test")
    print("=" * 60)

    print("Symbol : SBIN")
    print("Period : 01-01-2025 to 31-01-2025")
    print(f"URL    : {URL}")

    # Establish NSE session first.
    session.get(
        "https://www.nseindia.com/",
        headers=headers,
        timeout=(15, 30),
    )

    response = session.get(
        URL,
        params=PARAMS,
        headers=headers,
        timeout=(15, 60),
    )

    print("HTTP status:", response.status_code)

    response.raise_for_status()

    data = response.json()

    print("Response type:", type(data).__name__)

    if isinstance(data, dict):
        print("Response keys:", list(data.keys()))

    print()
    print("First part of response:")
    print(str(data)[:2000])

    OUTPUT_FILE.write_text(
        response.text,
        encoding="utf-8",
    )

    print()
    print(f"Raw response saved: {OUTPUT_FILE}")
    print("=" * 60)


if __name__ == "__main__":
    main()