from pathlib import Path
import requests

URL = "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"
OUTPUT = Path("data/raw/nse/EQUITY_L.csv")

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

headers = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/153.0.0.0 Safari/537.36"
    ),
    "Accept": "text/csv,text/plain,*/*",
    "Referer": "https://www.nseindia.com/",
}

print(f"Downloading: {URL}")

with requests.Session() as session:
    response = session.get(
        URL,
        headers=headers,
        timeout=(15, 60),
    )

    print(f"HTTP status: {response.status_code}")
    response.raise_for_status()

    OUTPUT.write_bytes(response.content)

print(f"Saved: {OUTPUT}")
print(f"Size: {OUTPUT.stat().st_size:,} bytes")
