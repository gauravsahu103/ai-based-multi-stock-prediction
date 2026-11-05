from pathlib import Path

import requests


# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

OUTPUT_DIR = PROJECT_ROOT / "data/raw/nse/prices"

OUTPUT_FILE = OUTPUT_DIR / "SBIN_2021_2026.csv"

BASE_URL = (
    "https://www.nseindia.com/"
    "api/historicalOR/generateSecurityWiseHistoricalData"
)


# ============================================================
# Download CSV
# ============================================================

def main() -> None:

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

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

    params = {
        "from": "18-09-2021",
        "to": "18-09-2026",
        "symbol": "SBIN",
        "type": "priceVolumeDeliverable",
        "series": "EQ",
        "csv": "true",
    }

    print("=" * 60)
    print("NSE Historical CSV Test")
    print("=" * 60)

    print(f"Symbol : {params['symbol']}")
    print(f"From   : {params['from']}")
    print(f"To     : {params['to']}")
    print("Format : CSV")

    # --------------------------------------------------------
    # Establish NSE session
    # --------------------------------------------------------

    print()
    print("Establishing NSE session...")

    home_response = session.get(
        "https://www.nseindia.com/",
        headers=headers,
        timeout=(15, 30),
    )

    print(
        "NSE homepage status:",
        home_response.status_code,
    )

    # --------------------------------------------------------
    # Request CSV
    # --------------------------------------------------------

    print()
    print("Requesting historical CSV...")

    response = session.get(
        BASE_URL,
        params=params,
        headers=headers,
        timeout=(15, 120),
    )

    print(
        "Historical API status:",
        response.status_code,
    )

    print(
        "Content-Type:",
        response.headers.get("Content-Type"),
    )

    response.raise_for_status()

    # --------------------------------------------------------
    # Save raw CSV
    # --------------------------------------------------------

    OUTPUT_FILE.write_bytes(response.content)

    print()
    print(
        f"Downloaded bytes: "
        f"{len(response.content):,}"
    )

    print(
        f"Raw CSV saved: "
        f"{OUTPUT_FILE}"
    )

    print()
    print("First 500 characters:")
    print(
        response.content[:500]
        .decode("utf-8", errors="replace")
    )

    print()
    print("=" * 60)
    print("NSE Historical CSV Test Completed")
    print("=" * 60)


if __name__ == "__main__":
    main()