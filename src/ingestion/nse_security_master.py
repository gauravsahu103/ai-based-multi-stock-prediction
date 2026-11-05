from pathlib import Path

import pandas as pd


# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_FILE = PROJECT_ROOT / "data/raw/nse/EQUITY_L.csv"

OUTPUT_DIR = PROJECT_ROOT / "data/processed/stock_master"
OUTPUT_FILE = OUTPUT_DIR / "nse_equity_stock_master.parquet"


# ============================================================
# Expected NSE columns
# ============================================================

EXPECTED_COLUMNS = {
    "SYMBOL",
    "NAME OF COMPANY",
    "SERIES",
    "DATE OF LISTING",
    "PAID UP VALUE",
    "MARKET LOT",
    "ISIN NUMBER",
    "FACE VALUE",
}


# ============================================================
# Read raw NSE security master
# ============================================================

def load_raw_security_master() -> pd.DataFrame:
    """Load the raw NSE equity security master CSV."""

    if not RAW_FILE.exists():
        raise FileNotFoundError(
            f"NSE security master not found: {RAW_FILE}"
        )

    df = pd.read_csv(RAW_FILE)

    # NSE files may contain whitespace around column names.
    df.columns = df.columns.str.strip()

    return df


# ============================================================
# Validate source schema
# ============================================================

def validate_source_schema(df: pd.DataFrame) -> None:
    """Validate that the NSE file contains the expected columns."""

    actual_columns = set(df.columns)

    missing_columns = EXPECTED_COLUMNS - actual_columns

    if missing_columns:
        raise ValueError(
            f"Missing expected NSE columns: {sorted(missing_columns)}"
        )


# ============================================================
# Normalize security master
# ============================================================

def normalize_security_master(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and normalize NSE security master data."""

    df = df.copy()

    # --------------------------------------------------------
    # Keep regular equity series only
    # --------------------------------------------------------

    df["SERIES"] = df["SERIES"].astype("string").str.strip().str.upper()

    df = df[df["SERIES"] == "EQ"].copy()

    # --------------------------------------------------------
    # Normalize text fields
    # --------------------------------------------------------

    df["SYMBOL"] = (
        df["SYMBOL"]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    df["NAME OF COMPANY"] = (
        df["NAME OF COMPANY"]
        .astype("string")
        .str.strip()
    )

    df["ISIN NUMBER"] = (
        df["ISIN NUMBER"]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    # --------------------------------------------------------
    # Parse listing date
    # --------------------------------------------------------

    df["DATE OF LISTING"] = pd.to_datetime(
        df["DATE OF LISTING"],
        format="%d-%b-%Y",
        errors="coerce",
    )

    # --------------------------------------------------------
    # Normalize numeric fields
    # --------------------------------------------------------

    df["PAID UP VALUE"] = pd.to_numeric(
        df["PAID UP VALUE"],
        errors="coerce",
    )

    df["MARKET LOT"] = pd.to_numeric(
        df["MARKET LOT"],
        errors="coerce",
    )

    df["FACE VALUE"] = pd.to_numeric(
        df["FACE VALUE"],
        errors="coerce",
    )

    # --------------------------------------------------------
    # Rename to application-friendly names
    # --------------------------------------------------------

    df = df.rename(
        columns={
            "SYMBOL": "symbol",
            "NAME OF COMPANY": "company_name",
            "SERIES": "series",
            "DATE OF LISTING": "listing_date",
            "PAID UP VALUE": "paid_up_value",
            "MARKET LOT": "market_lot",
            "ISIN NUMBER": "isin",
            "FACE VALUE": "face_value",
        }
    )

    # --------------------------------------------------------
    # Add exchange
    # --------------------------------------------------------

    df["exchange"] = "NSE"

    # --------------------------------------------------------
    # Select final columns
    # --------------------------------------------------------

    df = df[
        [
            "symbol",
            "company_name",
            "series",
            "listing_date",
            "paid_up_value",
            "market_lot",
            "isin",
            "face_value",
            "exchange",
        ]
    ]

    # --------------------------------------------------------
    # Sort
    # --------------------------------------------------------

    df = df.sort_values(
        by=["symbol", "isin"],
        na_position="last",
    ).reset_index(drop=True)

    return df


# ============================================================
# Validate normalized data
# ============================================================

def validate_normalized_data(df: pd.DataFrame) -> None:
    """Validate normalized stock master."""

    if df.empty:
        raise ValueError("Normalized security master is empty.")

    # Symbol must exist.
    if df["symbol"].isna().any():
        raise ValueError("Found NULL symbols.")

    if (df["symbol"].str.len() == 0).any():
        raise ValueError("Found empty symbols.")

    # Duplicate symbols should not exist in the current NSE file.
    duplicate_symbols = df[
        df["symbol"].duplicated(keep=False)
    ]

    if not duplicate_symbols.empty:
        print("WARNING: Duplicate symbols found:")
        print(
            duplicate_symbols[
                ["symbol", "company_name", "isin"]
            ].to_string(index=False)
        )

    # ISIN duplicates are also worth reporting.
    duplicate_isins = df[
        df["isin"].notna()
        & df["isin"].duplicated(keep=False)
    ]

    if not duplicate_isins.empty:
        print("WARNING: Duplicate ISINs found:")
        print(
            duplicate_isins[
                ["symbol", "company_name", "isin"]
            ].to_string(index=False)
        )

    # Listing dates should normally be present.
    invalid_listing_dates = df["listing_date"].isna().sum()

    if invalid_listing_dates > 0:
        print(
            f"WARNING: {invalid_listing_dates} records "
            "have invalid/missing listing dates."
        )


# ============================================================
# Save Parquet
# ============================================================

def save_parquet(df: pd.DataFrame) -> None:
    """Save normalized security master as Parquet."""

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_parquet(
        OUTPUT_FILE,
        engine="pyarrow",
        index=False,
    )


# ============================================================
# Main pipeline
# ============================================================

def main() -> None:
    print("=" * 60)
    print("NSE Security Master Ingestion")
    print("=" * 60)

    print(f"Raw file: {RAW_FILE}")

    # 1. Load
    df = load_raw_security_master()

    print(f"Raw records: {len(df):,}")

    # 2. Validate source
    validate_source_schema(df)

    print("Source schema: OK")

    # 3. Normalize
    df = normalize_security_master(df)

    print(f"EQ records: {len(df):,}")

    # 4. Validate normalized data
    validate_normalized_data(df)

    print("Normalized data validation: OK")

    # 5. Save
    save_parquet(df)

    print(f"Parquet saved: {OUTPUT_FILE}")
    print(f"Final records: {len(df):,}")

    print("=" * 60)
    print("NSE Security Master ingestion completed successfully.")
    print("=" * 60)


if __name__ == "__main__":
    main()