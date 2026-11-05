"""
Build the canonical Day-1 price foundation.

This module lives under src/preprocessing/day1/, so the project root is
three levels above this file.

Responsibilities
----------------
1. Read raw NSE historical CSV files, applying configured Day-1 eligibility exclusions.
2. Preserve historical alias rows instead of discarding them.
3. Resolve simple ticker changes to the requested/current symbol using
   NSE symbol-change history.
4. Apply supported backward price adjustments for split/bonus actions.
5. Preserve the raw source columns and record adjustment metadata.
6. Refuse to silently adjust unsupported corporate-action types.

Supported automatic price adjustments
-------------------------------------
- Face-value split / subdivision
- Bonus issue

Not automatically adjusted
--------------------------
- Dividend: retained as an economic price event for the price-return target
- Rights
- Merger
- Demerger
- Other complex actions

Those complex events are reported and require explicit treatment before the
canonical dataset is approved.

Price adjustment policy for split/bonus
---------------------------------------
For an event on EX-DATE D with adjustment factor f:

    adjusted_price(t) = raw_price(t) * cumulative_factor

for t < D.

For a split/bonus, historical share quantity fields are adjusted by 1/f,
while traded_value remains unchanged and deliverable percentage is unchanged.

This produces a split/bonus-consistent historical price series.

Inputs
------
data/raw/nse/prices/*_2021_2026.csv
data/raw/nse/reference/symbolchange.csv
data/raw/nse/reference/corporate_actions_2021_2026.csv

Output
------
data/processed/prices_canonical/*.parquet
data/processed/corporate_actions/corporate_action_reconciliation.parquet
data/evaluation/day1/price_canonical_reconciliation.txt
"""

from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd
import yaml

from src.utils.mvp_config import evaluation_dir, processed_dir

HORIZON = 1

# File:
#   Project/src/preprocessing/day1/build_canonical_price_dataset.py
#
# Therefore:
#   parents[0] -> day1
#   parents[1] -> preprocessing
#   parents[2] -> src
#   parents[3] -> Project
PROJECT_ROOT = Path(__file__).resolve().parents[3]

RAW_PRICE_DIR = PROJECT_ROOT / "data/raw/nse/prices"
REFERENCE_DIR = PROJECT_ROOT / "data/raw/nse/reference"

SYMBOL_CHANGE_FILE = REFERENCE_DIR / "symbolchange.csv"
CORPORATE_ACTION_FILE = REFERENCE_DIR / "corporate_actions_2021_2026.csv"
CONFIG_FILE = PROJECT_ROOT / "config/day1_exclusions.yaml"

# processed_dir is a function in src.utils.mvp_config.
# Use the configured project storage path rather than treating the function
# itself as a Path.
PROCESSED_DIR = processed_dir()

OUTPUT_DIR = PROCESSED_DIR / "prices_canonical"
CA_OUTPUT_DIR = PROCESSED_DIR / "corporate_actions"
EVALUATION_DIR = evaluation_dir(HORIZON)

PRICE_COLUMNS = [
    "prev_close",
    "open",
    "high",
    "low",
    "last",
    "close",
    "vwap",
]

QUANTITY_COLUMNS = [
    "volume",
    "deliverable_qty",
]

COMPLEX_ACTION_WORDS = (
    "rights",
    "demerger",
    "merger",
    "amalgamation",
)


def normalize_symbol(value: object) -> str:
    return str(value).strip().upper()


def load_day1_exclusions() -> tuple[set[str], str]:
    """
    Read Day-1 temporary stock exclusions from config/day1_exclusions.yaml.

    Expected configuration:

        day1:
          excluded_symbols:
            - GRASIM
            - HINDUNILVR
            - INDHOTEL
            - ITC
            - SIEMENS
          exclusion_note: >
            Temporarily excluded from the Day-1 MVP because of unresolved
            complex corporate actions.

    The exclusion is an MVP eligibility rule; it does not delete the
    security from the underlying universe definition.
    """
    if not CONFIG_FILE.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {CONFIG_FILE}"
        )

    with CONFIG_FILE.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    day1_config = config.get("day1", {}) or {}

    excluded_values = day1_config.get("excluded_symbols", []) or []

    if not isinstance(excluded_values, list):
        raise ValueError(
            "config/day1_exclusions.yaml: day1.excluded_symbols must be a list."
        )

    excluded_symbols = {
        normalize_symbol(symbol)
        for symbol in excluded_values
        if str(symbol).strip()
    }

    note = str(
        day1_config.get(
            "exclusion_note",
            "Temporary Day-1 eligibility exclusions.",
        )
    ).strip()

    return excluded_symbols, note


def load_symbol_changes() -> pd.DataFrame:
    if not SYMBOL_CHANGE_FILE.exists():
        raise FileNotFoundError(
            f"Symbol-change file not found: {SYMBOL_CHANGE_FILE}"
        )

    df = pd.read_csv(SYMBOL_CHANGE_FILE, header=None)

    # NSE archive is documented as:
    # Name, Old Symbol, New Symbol, Date of change.
    if df.shape[1] < 4:
        raise ValueError(
            f"Unexpected symbol-change file shape: {df.shape}"
        )

    df = df.iloc[:, :4].copy()
    df.columns = [
        "name",
        "old_symbol",
        "new_symbol",
        "change_date",
    ]

    for column in ["old_symbol", "new_symbol"]:
        df[column] = df[column].map(normalize_symbol)

    df["change_date"] = pd.to_datetime(
        df["change_date"],
        errors="coerce",
        dayfirst=True,
    )

    df = df.dropna(
        subset=["old_symbol", "new_symbol", "change_date"]
    ).reset_index(drop=True)

    return df


def resolve_to_requested(
    source_symbol: str,
    trade_date: pd.Timestamp,
    requested_symbol: str,
    changes: pd.DataFrame,
) -> bool:
    """
    Determine whether a historical source symbol/date belongs to the
    requested current-security series through simple ticker changes.

    This deliberately does not infer mergers/demergers from names or prices.
    """

    source_symbol = normalize_symbol(source_symbol)
    requested_symbol = normalize_symbol(requested_symbol)

    if source_symbol == requested_symbol:
        return True

    current = source_symbol
    visited: set[str] = set()

    for _ in range(20):
        if current in visited:
            return False
        visited.add(current)

        outgoing = (
            changes[
                (changes["old_symbol"] == current)
                & (changes["change_date"] > trade_date)
            ]
            .sort_values("change_date")
        )

        if outgoing.empty:
            return False

        next_symbol = normalize_symbol(
            outgoing.iloc[0]["new_symbol"]
        )

        if next_symbol == requested_symbol:
            return True

        current = next_symbol

    return False


def parse_split_bonus_factor(
    purpose: str,
) -> tuple[str | None, float | None]:
    """
    Return (action_type, factor) for supported mechanical actions.

    Split examples:
        Face Value Split ... From Rs 10 ... To Rs 2
        factor = 2 / 10

    Bonus examples:
        Bonus 1:1
        factor = 1 / (1 + 1)

        Bonus 4:1
        factor = 1 / (1 + 4)
    """

    text = str(purpose).strip().lower()

    # Face-value split / subdivision.
    if (
        "split" in text
        or "sub-division" in text
        or "subdivision" in text
    ):
        match = re.search(
            r"from\s*(?:rs\.?|₹)?\s*([0-9]+(?:\.[0-9]+)?)"
            r".*?to\s*(?:rs\.?|₹)?\s*([0-9]+(?:\.[0-9]+)?)",
            text,
        )
        if match:
            from_value = float(match.group(1))
            to_value = float(match.group(2))

            if from_value > 0 and to_value > 0:
                # Old 10 -> new 2 means 5 shares per old share.
                return "split", to_value / from_value

        raise ValueError(
            f"Supported split/subdivision could not be parsed: {purpose}"
        )

    # Bonus.
    if "bonus" in text:
        match = re.search(
            r"bonus\s*([0-9]+(?:\.[0-9]+)?)\s*:\s*"
            r"([0-9]+(?:\.[0-9]+)?)",
            text,
        )
        if match:
            bonus_new = float(match.group(1))
            old = float(match.group(2))

            if old > 0:
                return "bonus", old / (old + bonus_new)

        raise ValueError(f"Bonus ratio could not be parsed: {purpose}")

    return None, None


def load_corporate_actions() -> pd.DataFrame:
    if not CORPORATE_ACTION_FILE.exists():
        raise FileNotFoundError(
            f"Corporate-action file not found: {CORPORATE_ACTION_FILE}"
        )

    df = pd.read_csv(CORPORATE_ACTION_FILE)

    required = {"symbol", "purpose", "ex_date"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Corporate-action file missing columns: {sorted(missing)}"
        )

    df["symbol"] = df["symbol"].map(normalize_symbol)
    df["purpose"] = df["purpose"].astype(str).str.strip()
    df["ex_date"] = pd.to_datetime(
        df["ex_date"],
        errors="coerce",
        dayfirst=True,
    )

    df = df.dropna(
        subset=["symbol", "purpose", "ex_date"]
    ).reset_index(drop=True)

    return df


def load_raw_price(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    mapping = {
        "Symbol": "symbol",
        "Series": "series",
        "Date": "trade_date",
        "Prev Close": "prev_close",
        "Open Price": "open",
        "High Price": "high",
        "Low Price": "low",
        "Last Price": "last",
        "Close Price": "close",
        "Average Price": "vwap",
        "Total Traded Quantity": "volume",
        "Turnover ₹": "traded_value",
        "No. of Trades": "num_trades",
        "Deliverable Qty": "deliverable_qty",
        "% Dly Qt to Traded Qty": "deliverable_pct",
    }

    df.columns = [str(c).strip() for c in df.columns]
    df = df.rename(columns=mapping)

    required = {
        "symbol",
        "series",
        "trade_date",
        "prev_close",
        "open",
        "high",
        "low",
        "close",
        "vwap",
        "volume",
        "traded_value",
        "num_trades",
        "deliverable_qty",
        "deliverable_pct",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"{path}: missing columns: {sorted(missing)}"
        )

    df["symbol"] = df["symbol"].map(normalize_symbol)
    df["series"] = df["series"].astype(str).str.strip().str.upper()

    df["trade_date"] = pd.to_datetime(
        df["trade_date"],
        errors="coerce",
        dayfirst=True,
    )

    for column in PRICE_COLUMNS + QUANTITY_COLUMNS + [
        "traded_value",
        "num_trades",
        "deliverable_pct",
    ]:
        df[column] = (
            df[column]
            .astype(str)
            .str.replace(",", "", regex=False)
            .str.strip()
        )
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    # Corporate-action adjustments can produce fractional quantities.
    # Convert all adjustment targets to float before applying factors so
    # pandas does not attempt an unsafe int64 -> float64 assignment.
    for column in PRICE_COLUMNS + QUANTITY_COLUMNS:
        df[column] = df[column].astype("float64")

    df = df[df["series"] == "EQ"].copy()

    if df["trade_date"].isna().any():
        raise ValueError(f"{path}: invalid dates.")

    if df["trade_date"].duplicated().any():
        # Multiple historical aliases are allowed, but not duplicate
        # source-symbol/date records.
        if df.duplicated(["symbol", "trade_date"]).any():
            raise ValueError(
                f"{path}: duplicate source symbol/date rows."
            )

    return df.sort_values(
        ["trade_date", "symbol"]
    ).reset_index(drop=True)


def apply_split_bonus_adjustments(
    df: pd.DataFrame,
    actions: pd.DataFrame,
    requested_symbol: str,
    changes: pd.DataFrame,
) -> tuple[pd.DataFrame, list[dict]]:
    result = df.copy()

    # Split/bonus factors can be fractional (for example 1/2 or 1/5).
    # Ensure adjustment targets are float even when the source parquet/CSV
    # represented quantities as int64.
    for column in PRICE_COLUMNS + QUANTITY_COLUMNS:
        result[column] = result[column].astype("float64")

    reconciliation: list[dict] = []

    matched_actions = []

    for _, action in actions.iterrows():
        # Current symbol is the normal path. Alias rows are resolved through
        # the symbol-change table using the action date.
        action_symbol = normalize_symbol(action["symbol"])

        if action_symbol == requested_symbol:
            matched = True
        else:
            matched = resolve_to_requested(
                action_symbol,
                pd.Timestamp(action["ex_date"]),
                requested_symbol,
                changes,
            )

        if matched:
            matched_actions.append(action)

    matched_actions = sorted(
        matched_actions,
        key=lambda x: pd.Timestamp(x["ex_date"]),
    )

    for action in matched_actions:
        purpose = str(action["purpose"])
        ex_date = pd.Timestamp(action["ex_date"])

        try:
            action_type, factor = parse_split_bonus_factor(purpose)
        except ValueError:
            action_type, factor = None, None

        text = purpose.lower()

        if any(word in text for word in COMPLEX_ACTION_WORDS):
            reconciliation.append({
                "symbol": requested_symbol,
                "ex_date": ex_date,
                "purpose": purpose,
                "status": "REQUIRES_REVIEW",
                "action_type": "complex",
                "factor": np.nan,
            })
            continue

        if action_type in {"split", "bonus"} and factor is not None:
            mask = result["trade_date"] < ex_date

            # Backward-adjust prices/price-like fields.
            for column in PRICE_COLUMNS:
                result.loc[mask, column] *= factor

            # Split/bonus changes share count, so historical quantities
            # must be multiplied by the inverse factor.
            inverse = 1.0 / factor

            for column in QUANTITY_COLUMNS:
                result.loc[mask, column] *= inverse

            reconciliation.append({
                "symbol": requested_symbol,
                "ex_date": ex_date,
                "purpose": purpose,
                "status": "ADJUSTED",
                "action_type": action_type,
                "factor": factor,
                "rows_affected": int(mask.sum()),
            })
        else:
            reconciliation.append({
                "symbol": requested_symbol,
                "ex_date": ex_date,
                "purpose": purpose,
                "status": "ANNOTATED_ONLY",
                "action_type": (
                    "dividend"
                    if "dividend" in text
                    else "other"
                ),
                "factor": np.nan,
            })

    return result, reconciliation


def canonicalize_one(
    raw_file: Path,
    requested_symbol: str,
    changes: pd.DataFrame,
    actions: pd.DataFrame,
) -> tuple[pd.DataFrame, list[dict], list[dict]]:
    raw = load_raw_price(raw_file)

    # Keep only records that belong to this requested current security through
    # exact symbol identity or simple symbol-change continuity.
    membership = []

    for _, row in raw.iterrows():
        membership.append(
            resolve_to_requested(
                row["symbol"],
                pd.Timestamp(row["trade_date"]),
                requested_symbol,
                changes,
            )
        )

    series = raw.loc[membership].copy()

    if series.empty:
        raise ValueError(
            f"{requested_symbol}: no rows could be resolved "
            "to requested symbol."
        )

    series["source_symbol"] = series["symbol"]
    series["symbol"] = requested_symbol

    if series.duplicated(["symbol", "trade_date"]).any():
        duplicates = series[
            series.duplicated(
                ["symbol", "trade_date"],
                keep=False,
            )
        ][["source_symbol", "trade_date"]]

        raise ValueError(
            f"{requested_symbol}: duplicate canonical dates after "
            "symbol resolution:\n"
            f"{duplicates.head(20).to_string(index=False)}"
        )

    adjusted, reconciliation = apply_split_bonus_adjustments(
        series,
        actions,
        requested_symbol,
        changes,
    )

    adjusted = (
        adjusted
        .sort_values("trade_date")
        .reset_index(drop=True)
    )

    # Do not allow structural invalidity after adjustment.
    if (adjusted["close"] <= 0).any():
        raise ValueError(
            f"{requested_symbol}: non-positive adjusted close."
        )

    invalid_ohlc = adjusted[
        (adjusted["high"] < adjusted["low"])
        | (adjusted["high"] < adjusted["open"])
        | (adjusted["high"] < adjusted["close"])
        | (adjusted["low"] > adjusted["open"])
        | (adjusted["low"] > adjusted["close"])
    ]

    if not invalid_ohlc.empty:
        raise ValueError(
            f"{requested_symbol}: invalid OHLC after adjustment "
            f"({len(invalid_ohlc)} rows)."
        )

    complex_events = [
        item
        for item in reconciliation
        if item["status"] == "REQUIRES_REVIEW"
    ]

    return adjusted, reconciliation, complex_events


def main() -> None:
    print("=" * 80)
    print("CANONICAL PRICE DATASET BUILD")
    print("=" * 80)

    print(f"Project root      : {PROJECT_ROOT}")
    print(f"Raw price input   : {RAW_PRICE_DIR}")
    print(f"Reference input   : {REFERENCE_DIR}")
    print(f"Canonical output  : {OUTPUT_DIR}")
    print(f"CA reconciliation : {CA_OUTPUT_DIR}")
    print(f"Day-1 evaluation  : {EVALUATION_DIR}")
    print()

    if not RAW_PRICE_DIR.exists():
        raise FileNotFoundError(
            f"Raw price directory not found: {RAW_PRICE_DIR}"
        )

    if not REFERENCE_DIR.exists():
        raise FileNotFoundError(
            f"Reference directory not found: {REFERENCE_DIR}"
        )

    changes = load_symbol_changes()
    actions = load_corporate_actions()
    excluded_symbols, exclusion_note = load_day1_exclusions()

    raw_files = sorted(
        RAW_PRICE_DIR.glob("*_2021_2026.csv")
    )

    if not raw_files:
        raise FileNotFoundError(
            f"No raw NSE files found in {RAW_PRICE_DIR}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )
    CA_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )
    EVALUATION_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Remove stale canonical outputs for symbols that are now explicitly
    # excluded from the Day-1 pipeline. This prevents a previous successful
    # run from leaving excluded stocks available to downstream glob-based
    # readers.
    for excluded_symbol in excluded_symbols:
        stale_output = (
            OUTPUT_DIR / f"{excluded_symbol}.parquet"
        )
        if stale_output.exists():
            stale_output.unlink()
            print(
                f"[REMOVED] stale excluded output: "
                f"{stale_output.name}"
            )

    candidate_symbols = {
        raw_file.name
        .replace("_2021_2026.csv", "")
        .upper()
        for raw_file in raw_files
    }

    unknown_exclusions = excluded_symbols - candidate_symbols

    if unknown_exclusions:
        raise ValueError(
            "Configured Day-1 exclusions are not present in the raw "
            f"price universe: {sorted(unknown_exclusions)}"
        )

    selected_raw_files = []
    excluded_records = []

    for raw_file in raw_files:
        requested = (
            raw_file.name
            .replace("_2021_2026.csv", "")
            .upper()
        )

        if requested in excluded_symbols:
            excluded_records.append({
                "symbol": requested,
                "status": "EXCLUDED_DAY1",
                "reason": exclusion_note,
            })
            continue

        selected_raw_files.append(raw_file)

    all_reconciliation: list[dict] = []
    all_complex: list[dict] = []

    print(f"Raw files           : {len(raw_files)}")
    print(f"Configured exclusions: {len(excluded_symbols)}")
    print(f"Day-1 selected files : {len(selected_raw_files)}")

    if excluded_symbols:
        print(
            "Excluded symbols    : "
            + ", ".join(sorted(excluded_symbols))
        )

    for raw_file in selected_raw_files:
        requested = (
            raw_file.name
            .replace("_2021_2026.csv", "")
            .upper()
        )

        canonical, reconciliation, complex_events = canonicalize_one(
            raw_file,
            requested,
            changes,
            actions,
        )

        output = OUTPUT_DIR / f"{requested}.parquet"

        canonical.to_parquet(
            output,
            index=False,
        )

        all_reconciliation.extend(reconciliation)
        all_complex.extend(complex_events)

        print(
            f"[OK] {requested:<15} "
            f"rows={len(canonical):>7,} "
            f"source_symbols={canonical['source_symbol'].nunique():>2}"
        )

    reconciliation_records = (
        excluded_records + all_reconciliation
    )

    reconciliation_df = pd.DataFrame(
        reconciliation_records
    )

    reconciliation_file = (
        CA_OUTPUT_DIR
        / "corporate_action_reconciliation.parquet"
    )

    reconciliation_df.to_parquet(
        reconciliation_file,
        index=False,
    )

    report = (
        EVALUATION_DIR
        / "price_canonical_reconciliation.txt"
    )

    adjusted_count = 0

    if (
        not reconciliation_df.empty
        and "status" in reconciliation_df.columns
    ):
        adjusted_count = int(
            (
                reconciliation_df["status"]
                == "ADJUSTED"
            ).sum()
        )

    lines = [
        "=" * 80,
        "PRICE CANONICAL RECONCILIATION",
        "=" * 80,
        f"Raw price files (universe candidates): {len(raw_files)}",
        f"Day-1 selected price files: {len(selected_raw_files)}",
        f"Day-1 excluded symbols: {len(excluded_symbols)}",
        (
            "Corporate action records considered: "
            f"{len(all_reconciliation):,}"
        ),
        "",
        "DAY-1 EXCLUSION NOTE",
        "-" * 80,
        exclusion_note,
        "Excluded symbols: "
        + (
            ", ".join(sorted(excluded_symbols))
            if excluded_symbols
            else "None"
        ),
        (
            "Adjusted split/bonus events: "
            f"{adjusted_count}"
        ),
        (
            "Complex events requiring review: "
            f"{len(all_complex):,}"
        ),
        "",
        "COMPLEX EVENTS",
        "-" * 80,
    ]

    if all_complex:
        for event in all_complex[:200]:
            lines.append(
                f"{event['symbol']} | "
                f"{event['ex_date'].date()} | "
                f"{event['purpose']}"
            )

        if len(all_complex) > 200:
            lines.append(
                f"... {len(all_complex) - 200} more"
            )
    else:
        lines.append("None.")

    report.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    if all_complex:
        raise RuntimeError(
            "Canonical price build completed with complex corporate "
            "actions requiring review. Inspect "
            f"{reconciliation_file} and {report} before approving "
            "the dataset."
        )

    print()
    print(f"Reconciliation: {reconciliation_file}")
    print(f"Report        : {report}")
    print(f"Day-1 universe : {len(selected_raw_files)}")
    print(f"Excluded       : {len(excluded_symbols)}")
    print("STATUS        : PASSED")


if __name__ == "__main__":
    main()
