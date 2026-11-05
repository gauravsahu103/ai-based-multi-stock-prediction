from pathlib import Path

import pandas as pd


# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SECURITY_MASTER_FILE = (
    PROJECT_ROOT
    / "data/processed/stock_master/"
    / "nse_equity_stock_master.parquet"
)

NIFTY100_FILE = (
    PROJECT_ROOT
    / "data/raw/nse/universe/"
    / "NIFTY100.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data/processed/universe"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "nifty100.parquet"
)


# ============================================================
# Load NSE Security Master
# ============================================================

def load_security_master() -> pd.DataFrame:
    """Load normalized NSE security master."""

    if not SECURITY_MASTER_FILE.exists():
        raise FileNotFoundError(
            f"Security master not found: {SECURITY_MASTER_FILE}"
        )

    df = pd.read_parquet(SECURITY_MASTER_FILE)

    return df


# ============================================================
# Load NIFTY 100
# ============================================================

def load_nifty100() -> pd.DataFrame:
    """Load official NSE NIFTY 100 constituent list."""

    if not NIFTY100_FILE.exists():
        raise FileNotFoundError(
            f"NIFTY 100 file not found: {NIFTY100_FILE}"
        )

    df = pd.read_csv(NIFTY100_FILE)

    # Normalize column names
    df.columns = (
        df.columns
        .str.strip()
        .str.upper()
    )

    return df


# ============================================================
# Validate NIFTY 100 source
# ============================================================

def validate_nifty100_source(df: pd.DataFrame) -> None:
    """Validate official NIFTY 100 source data."""

    expected_columns = {
        "COMPANY NAME",
        "INDUSTRY",
        "SYMBOL",
        "SERIES",
        "ISIN CODE",
    }

    actual_columns = set(df.columns)

    missing_columns = expected_columns - actual_columns

    if missing_columns:
        raise ValueError(
            f"Missing NIFTY 100 columns: "
            f"{sorted(missing_columns)}"
        )

    if len(df) != 100:
        raise ValueError(
            f"Expected exactly 100 NIFTY 100 records, "
            f"found {len(df)}."
        )

    # Normalize fields before validation
    df["SYMBOL"] = (
        df["SYMBOL"]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    df["ISIN CODE"] = (
        df["ISIN CODE"]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    duplicate_symbols = df[
        df["SYMBOL"].duplicated(keep=False)
    ]

    if not duplicate_symbols.empty:
        raise ValueError(
            "Duplicate symbols found in NIFTY 100:\n"
            + duplicate_symbols[
                ["SYMBOL", "COMPANY NAME"]
            ].to_string(index=False)
        )

    duplicate_isins = df[
        df["ISIN CODE"].duplicated(keep=False)
    ]

    if not duplicate_isins.empty:
        raise ValueError(
            "Duplicate ISINs found in NIFTY 100:\n"
            + duplicate_isins[
                ["SYMBOL", "ISIN CODE"]
            ].to_string(index=False)
        )

    invalid_series = df[
        df["SERIES"] != "EQ"
    ]

    if not invalid_series.empty:
        raise ValueError(
            "Non-EQ securities found in NIFTY 100:\n"
            + invalid_series[
                ["SYMBOL", "SERIES"]
            ].to_string(index=False)
        )


# ============================================================
# Build Universe
# ============================================================

def build_universe(
    security_master: pd.DataFrame,
    nifty100: pd.DataFrame,
) -> pd.DataFrame:
    """Build validated NIFTY 100 universe."""

    nifty100 = nifty100.copy()

    # Normalize source fields
    nifty100["SYMBOL"] = (
        nifty100["SYMBOL"]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    nifty100["ISIN CODE"] = (
        nifty100["ISIN CODE"]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    nifty100["COMPANY NAME"] = (
        nifty100["COMPANY NAME"]
        .astype("string")
        .str.strip()
    )

    nifty100["INDUSTRY"] = (
        nifty100["INDUSTRY"]
        .astype("string")
        .str.strip()
    )

    # Normalize security master fields
    security_master = security_master.copy()

    security_master["symbol"] = (
        security_master["symbol"]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    security_master["isin"] = (
        security_master["isin"]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    # --------------------------------------------------------
    # Join using ISIN
    #
    # ISIN is preferred over company name because it is a
    # security identifier and company names can change.
    # --------------------------------------------------------

    universe = nifty100.merge(
        security_master[
            [
                "symbol",
                "company_name",
                "isin",
                "listing_date",
                "exchange",
            ]
        ],
        left_on="ISIN CODE",
        right_on="isin",
        how="left",
        indicator=True,
    )

    # --------------------------------------------------------
    # Validate every NIFTY 100 security matched
    # --------------------------------------------------------

    unmatched = universe[
        universe["_merge"] != "both"
    ]

    if not unmatched.empty:
        raise ValueError(
            "NIFTY 100 securities not found "
            "in NSE security master:\n"
            + unmatched[
                [
                    "SYMBOL",
                    "COMPANY NAME",
                    "ISIN CODE",
                ]
            ].to_string(index=False)
        )

    # --------------------------------------------------------
    # Validate symbol consistency
    # --------------------------------------------------------

    symbol_mismatch = universe[
        universe["SYMBOL"] != universe["symbol"]
    ]

    if not symbol_mismatch.empty:
        print(
            "WARNING: Symbol mismatch between "
            "NIFTY 100 and security master:"
        )

        print(
            symbol_mismatch[
                [
                    "SYMBOL",
                    "symbol",
                    "ISIN CODE",
                ]
            ].to_string(index=False)
        )

    # --------------------------------------------------------
    # Build application-friendly schema
    # --------------------------------------------------------

    universe = universe[
        [
            "SYMBOL",
            "COMPANY NAME",
            "INDUSTRY",
            "SERIES",
            "ISIN CODE",
            "listing_date",
            "exchange",
        ]
    ].copy()

    universe = universe.rename(
        columns={
            "SYMBOL": "symbol",
            "COMPANY NAME": "company_name",
            "INDUSTRY": "industry",
            "SERIES": "series",
            "ISIN CODE": "isin",
        }
    )

    # --------------------------------------------------------
    # Add universe metadata
    # --------------------------------------------------------

    universe["universe"] = "NIFTY100"

    universe["universe_version"] = (
        pd.Timestamp.now(tz="Asia/Kolkata")
        .strftime("%Y-%m-%d")
    )

    # Stable ordering
    universe = universe.sort_values(
        "symbol"
    ).reset_index(drop=True)

    return universe


# ============================================================
# Validate final universe
# ============================================================

def validate_final_universe(
    universe: pd.DataFrame,
) -> None:
    """Validate final NIFTY 100 universe."""

    if len(universe) != 100:
        raise ValueError(
            f"Final universe must contain 100 stocks. "
            f"Found {len(universe)}."
        )

    if universe["symbol"].isna().any():
        raise ValueError(
            "Final universe contains NULL symbols."
        )

    if universe["isin"].isna().any():
        raise ValueError(
            "Final universe contains NULL ISINs."
        )

    if universe["symbol"].duplicated().any():
        raise ValueError(
            "Final universe contains duplicate symbols."
        )

    if universe["isin"].duplicated().any():
        raise ValueError(
            "Final universe contains duplicate ISINs."
        )

    print("Final universe validation: OK")


# ============================================================
# Save Universe
# ============================================================

def save_universe(
    universe: pd.DataFrame,
) -> None:
    """Save NIFTY 100 universe to Parquet."""

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    universe.to_parquet(
        OUTPUT_FILE,
        engine="pyarrow",
        index=False,
    )

    # Also create a dated snapshot.
    snapshot_date = (
        pd.Timestamp.now(tz="Asia/Kolkata")
        .strftime("%Y-%m-%d")
    )

    snapshot_file = (
        OUTPUT_DIR
        / f"nifty100_{snapshot_date}.parquet"
    )

    universe.to_parquet(
        snapshot_file,
        engine="pyarrow",
        index=False,
    )

    print(f"Universe saved: {OUTPUT_FILE}")
    print(f"Snapshot saved: {snapshot_file}")


# ============================================================
# Main
# ============================================================

def main() -> None:

    print("=" * 60)
    print("NIFTY 100 Universe Builder")
    print("=" * 60)

    # --------------------------------------------------------
    # Load Security Master
    # --------------------------------------------------------

    security_master = load_security_master()

    print(
        f"Security master records: "
        f"{len(security_master):,}"
    )

    # --------------------------------------------------------
    # Load NIFTY 100
    # --------------------------------------------------------

    nifty100 = load_nifty100()

    print(
        f"NIFTY 100 source records: "
        f"{len(nifty100):,}"
    )

    # --------------------------------------------------------
    # Validate source
    # --------------------------------------------------------

    validate_nifty100_source(nifty100)

    print("NIFTY 100 source validation: OK")

    # --------------------------------------------------------
    # Build universe
    # --------------------------------------------------------

    universe = build_universe(
        security_master,
        nifty100,
    )

    print(
        f"Universe records: "
        f"{len(universe):,}"
    )

    # --------------------------------------------------------
    # Validate final result
    # --------------------------------------------------------

    validate_final_universe(universe)

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_universe(universe)

    # --------------------------------------------------------
    # Display sample
    # --------------------------------------------------------

    print()
    print("Sample:")
    print(
        universe[
            [
                "symbol",
                "company_name",
                "industry",
                "isin",
            ]
        ]
        .head(10)
        .to_string(index=False)
    )

    print()
    print("=" * 60)
    print("NIFTY 100 Universe Build Completed Successfully")
    print("=" * 60)


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    main()