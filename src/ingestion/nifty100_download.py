from pathlib import Path

import requests


# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

OUTPUT_DIR = PROJECT_ROOT / "data/raw/nse/universe"
OUTPUT_FILE = OUTPUT_DIR / "NIFTY100.csv"

NIFTY100_URL = (
    "https://nsearchives.nseindia.com/"
    "content/indices/ind_nifty100list.csv"
)


# ============================================================
# Download
# ============================================================

def download_nifty100() -> None:
    """Download the official NSE NIFTY 100 constituent list."""

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "text/csv,*/*",
        "Referer": "https://www.nseindia.com/",
    }

    print("=" * 60)
    print("NIFTY 100 Universe Download")
    print("=" * 60)

    print(f"Source: {NIFTY100_URL}")
    print(f"Output: {OUTPUT_FILE}")

    response = requests.get(
        NIFTY100_URL,
        headers=headers,
        timeout=(15, 60),
    )

    response.raise_for_status()

    OUTPUT_FILE.write_bytes(response.content)

    print(f"Downloaded bytes: {len(response.content):,}")
    print("Download completed successfully.")
    print("=" * 60)


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    download_nifty100()