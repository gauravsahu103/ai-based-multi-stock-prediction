import os
import warnings
import numpy as np
import pandas as pd

from src.utils.mvp_config import evaluation_dir, modeling_dir

warnings.filterwarnings("ignore")


DATASET_PATH = modeling_dir() / "day5" / "nse100_technical_model_dataset.parquet"

OUTPUT_DIR = evaluation_dir()

KEY_COLUMNS = [
    "symbol",
    "trade_date",
]

TARGET_COLUMNS = [
    "target_return_5d",
    "target_direction_5d",
    "target_excess_return_5d",
]

FEATURE_COLUMNS = [
    "return_1d",
    "return_3d",
    "return_5d",
    "return_10d",
    "return_20d",
    "rsi_14",
    "roc_10",
    "roc_20",
    "volatility_5d",
    "volatility_20d",
    "atr_pct_14",
    "volume_change_1d",
    "volume_ratio_20d",
    "close_vs_sma20",
    "close_vs_sma50",
    "close_vs_ema20",
    "nifty50_return_1d",
    "nifty50_return_5d",
    "nifty50_return_20d",
    "stock_vs_nifty50_5d",
    "stock_vs_nifty50_20d",
    "rolling_beta_20d",
    "nifty50_volatility_20d",
    "relative_volatility_20d",
]


def section(title):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def print_table(df, max_rows=20):
    if df.empty:
        print("No records.")
    else:
        print(df.head(max_rows).to_string(index=False))


def check_basic_structure(df):
    section("1. BASIC DATASET STRUCTURE")

    print(f"Rows       : {len(df):,}")
    print(f"Columns    : {len(df.columns):,}")
    print(f"Stocks     : {df['symbol'].nunique():,}")
    print(
        f"Date range : "
        f"{df['trade_date'].min().date()} -> "
        f"{df['trade_date'].max().date()}"
    )

    missing_columns = [
        c for c in KEY_COLUMNS + FEATURE_COLUMNS + TARGET_COLUMNS
        if c not in df.columns
    ]

    if missing_columns:
        print("\nMISSING EXPECTED COLUMNS:")
        for column in missing_columns:
            print(f"  - {column}")
    else:
        print("\nRequired columns: PASS")


def check_duplicates(df):
    section("2. DUPLICATE CHECK")

    duplicate_mask = df.duplicated(
        subset=["symbol", "trade_date"],
        keep=False,
    )

    duplicate_rows = df[duplicate_mask].copy()

    print(f"Duplicate symbol/date rows : {len(duplicate_rows):,}")
    print(
        f"Duplicate symbol/date keys : "
        f"{duplicate_rows[['symbol', 'trade_date']].drop_duplicates().shape[0]:,}"
    )

    if not duplicate_rows.empty:
        print("\nSample duplicates:")
        print_table(
            duplicate_rows[
                ["symbol", "trade_date"]
            ].sort_values(["symbol", "trade_date"])
        )


def check_missingness(df):
    section("3. MISSING VALUE CHECK")

    columns = [
        c for c in FEATURE_COLUMNS + TARGET_COLUMNS
        if c in df.columns
    ]

    rows = []

    for column in columns:
        missing = int(df[column].isna().sum())
        missing_pct = missing / len(df) * 100

        rows.append(
            {
                "column": column,
                "missing_rows": missing,
                "missing_pct": missing_pct,
            }
        )

    result = (
        pd.DataFrame(rows)
        .sort_values("missing_pct", ascending=False)
        .reset_index(drop=True)
    )

    print_table(result, max_rows=len(result))

    result.to_csv(
        os.path.join(OUTPUT_DIR, "missingness_by_feature.csv"),
        index=False,
    )


def check_infinite_values(df):
    section("4. INFINITE VALUE CHECK")

    rows = []

    numeric_columns = [
        c for c in FEATURE_COLUMNS + TARGET_COLUMNS
        if c in df.columns
    ]

    for column in numeric_columns:
        values = pd.to_numeric(df[column], errors="coerce")

        inf_count = int(np.isinf(values).sum())

        rows.append(
            {
                "column": column,
                "infinite_rows": inf_count,
            }
        )

    result = pd.DataFrame(rows)

    print_table(result, max_rows=len(result))

    total_inf = int(result["infinite_rows"].sum())

    print()
    print(f"Total infinite values: {total_inf:,}")


def check_stock_coverage(df):
    section("5. STOCK COVERAGE")

    coverage = (
        df.groupby("symbol")
        .agg(
            rows=("trade_date", "size"),
            first_date=("trade_date", "min"),
            last_date=("trade_date", "max"),
            trading_days=("trade_date", "nunique"),
        )
        .reset_index()
        .sort_values("rows")
    )

    print("Stocks with lowest row counts:")
    print_table(coverage, 20)

    coverage.to_csv(
        os.path.join(OUTPUT_DIR, "stock_coverage.csv"),
        index=False,
    )

    print()
    print("Stock count by year:")

    yearly = (
        df.groupby(df["trade_date"].dt.year)["symbol"]
        .nunique()
        .reset_index(name="stocks")
        .rename(columns={"trade_date": "year"})
    )

    print(yearly.to_string(index=False))

    yearly.to_csv(
        os.path.join(OUTPUT_DIR, "stock_count_by_year.csv"),
        index=False,
    )


def check_daily_universe(df):
    section("6. DAILY CROSS-SECTIONAL UNIVERSE")

    daily = (
        df.groupby("trade_date")
        .agg(
            stocks=("symbol", "nunique"),
            rows=("symbol", "size"),
        )
        .reset_index()
    )

    print("Daily universe statistics:")
    print(daily["stocks"].describe().to_string())

    print()
    print("Days with fewer than 90 stocks:")

    low_days = daily[daily["stocks"] < 90]

    if low_days.empty:
        print("None")
    else:
        print(low_days.to_string(index=False))

    daily.to_csv(
        os.path.join(OUTPUT_DIR, "daily_universe.csv"),
        index=False,
    )


def target_statistics(df):
    section("7. TARGET STATISTICS")

    target_columns = [
        c for c in TARGET_COLUMNS
        if c in df.columns
    ]

    result_rows = []

    for column in target_columns:
        series = pd.to_numeric(df[column], errors="coerce").dropna()

        if series.empty:
            continue

        result_rows.append(
            {
                "target": column,
                "count": len(series),
                "mean": series.mean(),
                "std": series.std(),
                "min": series.min(),
                "p01": series.quantile(0.01),
                "p05": series.quantile(0.05),
                "median": series.median(),
                "p95": series.quantile(0.95),
                "p99": series.quantile(0.99),
                "max": series.max(),
            }
        )

    result = pd.DataFrame(result_rows)

    print(result.to_string(index=False))

    result.to_csv(
        os.path.join(OUTPUT_DIR, "target_statistics.csv"),
        index=False,
    )


def extreme_target_analysis(df):
    section("8. EXTREME TARGET ANALYSIS")

    target = "target_excess_return_5d"

    if target not in df.columns:
        print(f"{target} not found.")
        return

    columns = [
        "symbol",
        "trade_date",
        "target_return_5d",
        "target_excess_return_5d",
    ]

    columns = [c for c in columns if c in df.columns]

    series = df[target]

    thresholds = [
        ("<= -50%", series <= -0.50),
        ("<= -20%", series <= -0.20),
        (">= +20%", series >= 0.20),
        (">= +50%", series >= 0.50),
    ]

    for label, mask in thresholds:
        subset = df.loc[mask, columns].copy()

        print()
        print(f"{label}")
        print(f"Rows: {len(subset):,}")

        if not subset.empty:
            print_table(
                subset.sort_values(
                    target,
                    ascending=True,
                ),
                20,
            )

            safe_label = (
                label.replace("<=", "le")
                .replace(">=", "ge")
                .replace("%", "")
                .replace(" ", "_")
                .replace("+", "plus")
                .replace("-", "minus")
            )

            subset.to_csv(
                os.path.join(
                    OUTPUT_DIR,
                    f"extreme_target_{safe_label}.csv",
                ),
                index=False,
            )


def feature_statistics(df):
    section("9. FEATURE DISTRIBUTION CHECK")

    rows = []

    for column in FEATURE_COLUMNS:
        if column not in df.columns:
            continue

        series = pd.to_numeric(
            df[column],
            errors="coerce",
        ).replace([np.inf, -np.inf], np.nan).dropna()

        if series.empty:
            continue

        rows.append(
            {
                "feature": column,
                "count": len(series),
                "mean": series.mean(),
                "std": series.std(),
                "min": series.min(),
                "p01": series.quantile(0.01),
                "p05": series.quantile(0.05),
                "median": series.median(),
                "p95": series.quantile(0.95),
                "p99": series.quantile(0.99),
                "max": series.max(),
            }
        )

    result = pd.DataFrame(rows)

    print_table(result, len(result))

    result.to_csv(
        os.path.join(OUTPUT_DIR, "feature_statistics.csv"),
        index=False,
    )


def feature_extreme_counts(df):
    section("10. FEATURE EXTREME VALUE COUNTS")

    rows = []

    for column in FEATURE_COLUMNS:
        if column not in df.columns:
            continue

        series = pd.to_numeric(
            df[column],
            errors="coerce",
        )

        valid = series.dropna()

        if valid.empty:
            continue

        p01 = valid.quantile(0.01)
        p99 = valid.quantile(0.99)

        rows.append(
            {
                "feature": column,
                "below_p01": int((valid < p01).sum()),
                "above_p99": int((valid > p99).sum()),
                "below_p01_pct": float((valid < p01).mean() * 100),
                "above_p99_pct": float((valid > p99).mean() * 100),
            }
        )

    result = pd.DataFrame(rows)

    print_table(result, len(result))

    result.to_csv(
        os.path.join(OUTPUT_DIR, "feature_extreme_counts.csv"),
        index=False,
    )


def yearly_data_quality(df):
    section("11. DATA QUALITY BY YEAR")

    df = df.copy()
    df["year"] = df["trade_date"].dt.year

    result = (
        df.groupby("year")
        .agg(
            rows=("symbol", "size"),
            stocks=("symbol", "nunique"),
            trading_days=("trade_date", "nunique"),
        )
        .reset_index()
    )

    print(result.to_string(index=False))

    result.to_csv(
        os.path.join(OUTPUT_DIR, "data_quality_by_year.csv"),
        index=False,
    )


def cross_sectional_target_analysis(df):
    section("12. CROSS-SECTIONAL TARGET QUALITY")

    target = "target_excess_return_5d"

    if target not in df.columns:
        print(f"{target} not found.")
        return

    daily = (
        df.groupby("trade_date")[target]
        .agg(
            mean="mean",
            median="median",
            std="std",
            positive_rate=lambda x: (x > 0).mean(),
            stocks="count",
        )
        .reset_index()
    )

    print("Daily target statistics:")
    print(daily[["mean", "median", "std", "positive_rate", "stocks"]].describe())

    print()
    print("Days with fewer than 90 valid target observations:")

    low_days = daily[daily["stocks"] < 90]

    if low_days.empty:
        print("None")
    else:
        print(low_days.to_string(index=False))

    daily.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "daily_cross_sectional_target.csv",
        ),
        index=False,
    )


def check_target_feature_correlations(df):
    section("13. FEATURE / TARGET CORRELATION")

    target = "target_excess_return_5d"

    if target not in df.columns:
        print(f"{target} not found.")
        return

    columns = [
        c for c in FEATURE_COLUMNS
        if c in df.columns
    ]

    data = df[columns + [target]].copy()

    corr = (
        data.corr(numeric_only=True)[target]
        .drop(target)
        .sort_values()
    )

    result = corr.reset_index()
    result.columns = ["feature", "correlation"]

    print(result.to_string(index=False))

    result.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "feature_target_correlations.csv",
        ),
        index=False,
    )


def check_temporal_order(df):
    section("14. TEMPORAL ORDER CHECK")

    ordered = df.sort_values(
        ["symbol", "trade_date"]
    ).copy()

    duplicate_dates = (
        ordered.groupby("symbol")["trade_date"]
        .apply(lambda x: x.duplicated().sum())
        .sum()
    )

    backwards = 0

    for _, group in ordered.groupby("symbol"):
        dates = group["trade_date"]

        if not dates.is_monotonic_increasing:
            backwards += 1

    print(f"Duplicate dates across stocks : {duplicate_dates:,}")
    print(f"Stocks with backward dates     : {backwards:,}")

    if duplicate_dates == 0 and backwards == 0:
        print("\nTemporal ordering: PASS")
    else:
        print("\nTemporal ordering: REVIEW")


def check_feature_target_sanity(df):
    section("15. FEATURE / TARGET SANITY CHECK")

    target = "target_excess_return_5d"

    if target not in df.columns:
        return

    checks = []

    for feature in FEATURE_COLUMNS:
        if feature not in df.columns:
            continue

        valid = df[[feature, target]].dropna()

        if len(valid) < 2:
            continue

        correlation = valid[feature].corr(valid[target])

        checks.append(
            {
                "feature": feature,
                "correlation": correlation,
                "abs_correlation": abs(correlation),
            }
        )

    result = (
        pd.DataFrame(checks)
        .sort_values(
            "abs_correlation",
            ascending=False,
        )
    )

    print("Highest absolute feature/target correlations:")
    print_table(result, 15)

    result.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "feature_target_sanity.csv",
        ),
        index=False,
    )


def main():

    section("STOCK PREDICTION DATA QUALITY EVALUATION")

    print(f"Loading dataset:")
    print(DATASET_PATH)

    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}"
        )

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = pd.read_parquet(DATASET_PATH)

    df["trade_date"] = pd.to_datetime(
        df["trade_date"]
    )

    check_basic_structure(df)

    check_duplicates(df)

    check_missingness(df)

    check_infinite_values(df)

    check_stock_coverage(df)

    check_daily_universe(df)

    target_statistics(df)

    extreme_target_analysis(df)

    feature_statistics(df)

    feature_extreme_counts(df)

    yearly_data_quality(df)

    cross_sectional_target_analysis(df)

    check_target_feature_correlations(df)

    check_temporal_order(df)

    check_feature_target_sanity(df)

    section("DATA QUALITY EVALUATION COMPLETE")

    print("Reports saved under:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()