"""Build auditable evidence for the Day-1 stock-prediction report.

Run from the repository root in the project's activated virtual environment:
    python final_submission_evidence_runner.py

Scope:
- One-session prediction target only: target_return_1d / target_direction_1d.
- Uses the existing chronological train/validation/test Parquet partitions; it
  does not regenerate or rebalance the split.
- Freezes the existing E0 baseline as the final candidate BEFORE opening the
  test partition. E0 is retained because previous validation reviews did not
  establish a robust cross-metric winner; TUNE_012 improved regression errors
  but had weaker validation Rank IC and Top-5 outcomes.
- Fits no model and changes no source project data. It loads the stored E0
  models, which were fitted on training data only, and performs the final test
  scoring once. A guard prevents accidental repeat evaluation.
- Writes new audit evidence under data/evaluation/day1/final_submission_evidence.

Important: statistical outlier flags are not automatically treated as data
errors. Candidate extremes are exported for review; this script does not delete
observations or assert that a market event explains an error.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, average_precision_score, brier_score_loss,
    f1_score, log_loss, mean_absolute_error, mean_squared_error,
    precision_score, recall_score, r2_score, roc_auc_score,
)

from src.models.day1 import train_xgboost_baseline as baseline
from src.utils.mvp_config import modeling_dir, evaluation_dir

HORIZON = 1
TARGET_RETURN = "target_return_1d"
TARGET_DIRECTION = "target_direction_1d"
KEYS = ["symbol", "trade_date"]
MOMENTUM_FEATURE = "return_5d"  # lagged signal for a one-day target only
TOP_K = [5, 10, 20]
BOOTSTRAP_REPS = 5000
BLOCK_LENGTH = 5
RANDOM_SEED = 42017

ROOT = Path.cwd()
TRAIN_FILE = modeling_dir(HORIZON) / "train.parquet"
VALIDATION_FILE = modeling_dir(HORIZON) / "validation.parquet"
TEST_FILE = modeling_dir(HORIZON) / "test.parquet"
INVENTORY_FILE = evaluation_dir(HORIZON) / "feature_inventory.json"
REG_MODEL_FILE = baseline.REGRESSION_MODEL_FILE
CLS_MODEL_FILE = baseline.CLASSIFICATION_MODEL_FILE
VAL_PRED_FILE = baseline.VALIDATION_PREDICTIONS_FILE
OUT_DIR = evaluation_dir(HORIZON) / "final_submission_evidence"

# Final candidate chosen using validation evidence only, before the test file is
# opened. Do not alter this after inspecting test results.
FINAL_MODEL_CHOICE = {
    "model": "E0_BASELINE",
    "feature_set": "E0",
    "prediction_horizon_sessions": 1,
    "reason": (
        "E0 remains the frozen reference. E0-E6 and TUNE_012 did not establish "
        "a consistently superior ranking result on validation; TUNE_012's "
        "validation Rank IC and Top-5 return were below E0 despite improved "
        "regression errors. Final test is used once for confirmation, not selection."
    ),
    "test_opened_before_freeze": False,
}


def iso(value: Any) -> str | None:
    if pd.isna(value):
        return None
    return pd.Timestamp(value).isoformat()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_inventory() -> list[str]:
    payload = json.loads(INVENTORY_FILE.read_text(encoding="utf-8"))
    features = payload.get("feature_list")
    if payload.get("feature_set") != "E0" or not isinstance(features, list):
        raise ValueError(f"Unexpected feature inventory: {INVENTORY_FILE}")
    if len(features) != len(set(features)):
        raise ValueError("Duplicate feature names in feature_inventory.json")
    return list(features)


def load_frame(path: Path, label: str, features: list[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{label} partition missing: {path}")
    df = pd.read_parquet(path)
    if df.empty:
        raise ValueError(f"{label} partition is empty: {path}")
    required = set(KEYS + [TARGET_RETURN, TARGET_DIRECTION])
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{label} missing required columns: {sorted(missing)}")
    missing_features = [x for x in features if x not in df.columns]
    if missing_features:
        raise ValueError(f"{label} missing E0 features: {missing_features}")
    df["trade_date"] = pd.to_datetime(df["trade_date"], errors="raise").dt.normalize()
    duplicates = int(df.duplicated(KEYS).sum())
    if duplicates:
        raise ValueError(f"{label} has {duplicates:,} duplicate symbol/date rows")
    y = pd.to_numeric(df[TARGET_RETURN], errors="coerce")
    d = pd.to_numeric(df[TARGET_DIRECTION], errors="coerce")
    if y.isna().any() or d.isna().any():
        raise ValueError(f"{label} has missing target labels; inspect split before evaluation")
    if not np.isfinite(y.to_numpy(dtype=float)).all():
        raise ValueError(f"{label} has non-finite return target values")
    if not d.isin([0, 1]).all():
        raise ValueError(f"{label} direction target must be binary 0/1")
    X = df[features].to_numpy(dtype=float)
    if np.isinf(X).any():
        raise ValueError(f"{label} has infinite predictor values")
    return df


def partition_info(name: str, df: pd.DataFrame, path: Path) -> dict[str, Any]:
    dates = pd.to_datetime(df.trade_date).dt.normalize()
    return {
        "partition": name,
        "path": str(path),
        "sha256": sha256(path),
        "rows": int(len(df)),
        "unique_stocks": int(df.symbol.nunique()),
        "unique_dates": int(dates.nunique()),
        "min_date": iso(dates.min()),
        "max_date": iso(dates.max()),
        "duplicate_symbol_date_rows": int(df.duplicated(KEYS).sum()),
        "positive_target_prevalence": float((df[TARGET_DIRECTION].astype(int) == 1).mean()),
    }


def check_temporal_order(parts: list[tuple[str, pd.DataFrame]]) -> None:
    bounds = []
    for name, df in parts:
        dates = pd.to_datetime(df.trade_date).dt.normalize()
        bounds.append((name, dates.min(), dates.max()))
    for prior, current in zip(bounds, bounds[1:]):
        if prior[2] >= current[1]:
            raise ValueError(
                f"Partitions overlap or are not chronological: {prior[0]} {prior[2]} / "
                f"{current[0]} {current[1]}"
            )


def classify_family(name: str) -> str:
    n = name.lower()
    if n in {"prev_close", "last", "vwap"}:
        return "Source-reported price fields"
    if n in {"deliverable_qty", "deliverable_pct", "num_trades", "traded_value"}:
        return "Source-reported market activity / liquidity"
    if n in {"sma_20", "sma_50", "ema_20"}:
        return "Trend / technical indicators"
    if n.startswith("sector_") or "sector" in n:
        return "Sector-relative"
    if n.startswith("market_relative_return_"):
        return "Relative performance"
    if "rank" in n or "percentile" in n:
        return "Cross-sectional ranking"
    if "nifty" in n or n.startswith("market_"):
        return "Market context / regime"
    if "volume" in n or "dollar_volume" in n or "liquidity" in n:
        return "Volume / activity / liquidity"
    if any(k in n for k in ["volatility", "atr", "beta", "risk"]):
        return "Volatility / market sensitivity"
    if any(k in n for k in ["gap", "intraday", "range", "wick", "body", "position_in_range", "candle"]):
        return "Price action"
    if any(k in n for k in ["sma", "ema", "trend", "slope", "macd", "adx"]):
        return "Trend / technical indicators"
    if any(k in n for k in ["rsi", "roc", "stochastic"]):
        return "Technical momentum / oscillator"
    if n.startswith("stock_vs_") or "relative_strength" in n or "relative_volatility" in n:
        return "Relative performance"
    if n.startswith("return_") or "momentum" in n:
        return "Returns / momentum"
    if "corr" in n:
        return "Rolling association"
    return "Other engineered structured feature"


def formula_for(name: str) -> str:
    """Map E0 columns to formulas or explicitly documented source-field mappings."""
    n = name.lower()

    # These columns are source-reported NSE fields, not engineered formulas.
    # Their names and adjustment behavior are verified in
    # build_canonical_price_dataset.py (load_raw_price / PRICE_COLUMNS /
    # QUANTITY_COLUMNS / apply_split_bonus_adjustments).
    source_fields = {
        "prev_close": (
            "NSE source field 'Prev Close' mapped to prev_close; source-reported, "
            "not recalculated from the modeling frame. Canonical historical values "
            "are backward-adjusted for supported split/bonus events before ex-date."
        ),
        "last": (
            "NSE source field 'Last Price' mapped to last; source-reported, not "
            "recalculated. Canonical historical values are backward-adjusted for "
            "supported split/bonus events before ex-date."
        ),
        "vwap": (
            "NSE source field 'Average Price' mapped to vwap; source-reported and "
            "not recomputed in the feature generator. Canonical historical values "
            "are backward-adjusted for supported split/bonus events before ex-date."
        ),
        "traded_value": (
            "NSE source field 'Turnover ₹' mapped to traded_value; source-reported "
            "turnover value. The canonical builder leaves it unchanged during "
            "supported split/bonus adjustments."
        ),
        "num_trades": (
            "NSE source field 'No. of Trades' mapped to num_trades; source-reported "
            "daily trade count, with no engineered transformation in the feature generator."
        ),
        "deliverable_qty": (
            "NSE source field 'Deliverable Qty' mapped to deliverable_qty; source-reported "
            "quantity. Historical values before supported split/bonus ex-dates are "
            "adjusted by the inverse price factor (1/f)."
        ),
        "deliverable_pct": (
            "NSE source field '% Dly Qt to Traded Qty' mapped to deliverable_pct; "
            "source-reported percentage. The canonical builder leaves it unchanged "
            "during supported split/bonus adjustments."
        ),
    }
    if n in source_fields:
        return source_fields[n]

    if n == "sma_20":
        return "rolling_mean(close, window=20, min_periods=20); computed per stock after sorting by trade_date"
    if n == "sma_50":
        return "rolling_mean(close, window=50, min_periods=50); computed per stock after sorting by trade_date"
    if n == "ema_20":
        return "exponentially weighted mean of close, span=20, adjust=False, min_periods=20; computed per stock after sorting by trade_date"

    m = re.fullmatch(r"return_(\d+)d", n)
    if m:
        k = m.group(1)
        return f"close_t / close_(t-{k}) - 1; trailing {k}-session close return"
    m = re.fullmatch(r"roc_(\d+)", n)
    if m:
        k = m.group(1)
        return f"close_t / close_(t-{k}) - 1; rate of change over {k} sessions"
    m = re.fullmatch(r"volatility_(\d+)d", n)
    if m:
        k = m.group(1)
        return f"sample standard deviation of return_1d over trailing {k} sessions"
    m = re.fullmatch(r"atr_pct_(\d+)", n)
    if m:
        k = m.group(1)
        return f"rolling_mean(TrueRange, {k}) / close_t; TrueRange=max(high-low, abs(high-prev_close), abs(low-prev_close))"
    m = re.fullmatch(r"volume_ratio_(\d+)d", n)
    if m:
        k = m.group(1)
        return f"volume_t / rolling_mean(volume, {k})"
    m = re.fullmatch(r"nifty50_return_(\d+)d", n)
    if m:
        k = m.group(1)
        return f"nifty50_close_t / nifty50_close_(t-{k}) - 1"
    m = re.fullmatch(r"nifty50_volatility_(\d+)d", n)
    if m:
        k = m.group(1)
        return f"sample standard deviation of nifty50_return_1d over trailing {k} sessions"
    m = re.fullmatch(r"rolling_beta_(\d+)d", n)
    if m:
        k = m.group(1)
        return f"rolling_cov(return_1d_stock, nifty50_return_1d, {k}) / rolling_var(nifty50_return_1d, {k})"
    m = re.fullmatch(r"close_vs_(sma|ema)(\d+)", n)
    if m:
        avg, k = m.groups()
        return f"close_t / {avg.upper()}({k})_t - 1"
    m = re.fullmatch(r"(sma|ema)(\d+)_slope", n)
    if m:
        avg, k = m.groups()
        return f"{avg.upper()}({k})_t / {avg.upper()}({k})_(t-5) - 1"
    m = re.fullmatch(r"close_vs_sma(\d+)", n)
    if m:
        k = m.group(1)
        return f"close_t / rolling_mean(close, {k}) - 1"
    m = re.fullmatch(r"(sma|ema)(\d+)_vs_(sma|ema)(\d+)", n)
    if m:
        a, ka, b, kb = m.groups()
        return f"{a.upper()}({ka})_t / {b.upper()}({kb})_t - 1"
    m = re.fullmatch(r"stock_vs_nifty50_(\d+)d", n)
    if m:
        k = m.group(1)
        return f"return_{k}d - nifty50_return_{k}d"
    m = re.fullmatch(r"stock_vs_nifty50_(\d+)d", n)
    if m:
        return f"return_{m.group(1)}d - nifty50_return_{m.group(1)}d"
    if n == "volume_change_1d": return "volume_t / volume_(t-1) - 1"
    if n == "dollar_volume": return "close_t * volume_t"
    if n == "relative_volatility_20d": return "volatility_20d / nifty50_volatility_20d"
    if n in {"market_relative_return_1d", "market_relative_return_5d", "market_relative_return_20d"}:
        k = n.rsplit("_", 1)[1].removesuffix("d")
        return f"return_{k}d - nifty50_return_{k}d"
    if n == "stock_return_vol_adjusted": return "return_1d / volatility_20d; zero volatility is converted to missing"
    if n == "return_acceleration_1d_5d": return "return_1d - (return_5d / 5.0); approximate one-session return minus average daily rate implied by the trailing five-session return"
    if n == "trend_strength_score": return "((close_vs_sma20 + close_vs_sma50 + sma20_vs_sma50) / 3) * (0.5 + 0.5 * clip(adx_14, 0, 50) / 50)"
    if n == "momentum_consistency_score": return "mean(sign(x) for x in [return_1d, return_3d, return_5d, return_10d, return_20d, roc_10, roc_20])"
    if n == "market_alignment_score": return "((sign(stock_vs_nifty50_5d) + sign(stock_vs_nifty50_20d))/2) * ((encode(market_trend_regime) + encode(market_momentum_regime))/2); bullish/up=+1, bearish/down=-1, neutral=0"
    if n == "volatility_ratio_5_20": return "volatility_5d / volatility_20d"
    if n == "volatility_ratio_20_60": return "volatility_20d / volatility_60d"
    if n == "price_volume_corr_20d": return "rolling correlation(return_1d, volume_pct_change, 20)"
    if n == "rsi_7": return "RSI(close, 7); 100 - 100/(1 + average_gain/average_loss)"
    if n == "rsi_14": return "RSI(close, 14); 100 - 100/(1 + average_gain/average_loss)"
    if n == "rsi_21": return "RSI(close, 21); 100 - 100/(1 + average_gain/average_loss)"
    if n == "macd": return "EMA(close, 12) - EMA(close, 26)"
    if n == "macd_signal": return "EMA(macd, 9)"
    if n == "macd_histogram": return "macd - macd_signal"
    if n == "stochastic_k": return "100 * (close - rolling_low(low, 14)) / (rolling_high(high, 14) - rolling_low(low, 14))"
    if n == "stochastic_d": return "rolling_mean(stochastic_k, 3)"
    if n == "adx_14": return "14-session rolling mean of DX derived from plus_DI and minus_DI"
    if n == "gap_pct": return "(open_t - close_(t-1)) / close_(t-1)"
    if n == "intraday_return": return "(close_t - open_t) / open_t"
    if n == "high_low_range_pct": return "(high_t - low_t) / close_(t-1)"
    if n == "close_position_in_range": return "(close_t - low_t) / (high_t - low_t)"
    if n == "body_pct": return "abs(close_t - open_t) / open_t"
    if n == "upper_wick_pct": return "(high_t - max(open_t, close_t)) / open_t"
    if n == "lower_wick_pct": return "(min(open_t, close_t) - low_t) / open_t"
    if n.startswith("return_rank_"):
        window = n.removeprefix("return_rank_")
        return f"within-trade-date percentile rank of return_{window}"
    if n == "relative_strength_rank": return "within-date percentile rank of stock_vs_nifty50_20d"
    if n == "volume_rank": return "within-date percentile rank of volume_ratio_20d"
    if n == "market_trend_regime": return "nifty50_close / rolling_mean(nifty50_close, 50) - 1"
    if n == "market_volatility_regime": return "nifty50_volatility_20d / rolling_std(nifty50_return_1d, 60)"
    if n == "market_momentum_regime": return "nifty50_return_20d"
    if n == "sector_return_1d": return "same-date cross-sectional mean of member return_1d by mapped sector"
    if n == "sector_return_5d": return "same-date cross-sectional mean of member return_5d by mapped sector"
    if n == "sector_return_20d": return "same-date cross-sectional mean of member return_20d by mapped sector"
    if n == "sector_volatility_20d": return "same-date cross-sectional mean of member volatility_20d by mapped sector"
    if n == "stock_vs_sector_1d": return "return_1d - sector_return_1d"
    if n == "stock_vs_sector_5d": return "return_5d - sector_return_5d"
    if n == "stock_vs_sector_20d" or n == "sector_relative_strength": return "return_20d - sector_return_20d"
    if n.startswith("close_vs_"):
        return "close_t / corresponding rolling average ending at T - 1; verify exact window and average type in generator"
    if n.startswith("nifty50_") and "volatility" in n:
        return "rolling standard deviation of NIFTY 50 daily returns; see generator for exact window"
    if n.startswith("stock_vs_nifty50_"):
        return "stock trailing return over named window minus NIFTY 50 return over same window; verify exact window in generator"
    if "rank" in n or "percentile" in n:
        return "within-date cross-sectional percentile rank of the named underlying feature; verify exact grouping in generator"
    if n.startswith("sector_") or "sector" in n:
        return "same-date sector-level aggregate or stock-minus-sector comparison; exact measure is named in feature and generator"
    # Do not invent a formula for a column whose implementation was not mapped.
    return "FORMULA REVIEW REQUIRED: locate exact transformation in the versioned Day-1 feature generator before submission."


def inventory_table(features: list[str]) -> pd.DataFrame:
    df = pd.DataFrame({"feature_name": features})
    df["feature_family"] = df.feature_name.map(classify_family)
    df["operational_formula"] = df.feature_name.map(formula_for)
    df["timing_contract"] = ("Uses canonical stock/market information indexed to trade_date T; the supervised outcome is the exact close-to-close return from T to the next NSE trading session. Operational scoring must occur after the T-session data are published and before the T+1 decision point; an exact source-publication timestamp is not inferred by this inventory.")
    df["missingness_policy"] = "Rolling-window/invalid-denominator missingness retained; XGBoost native missing-value handling."
    df["source_of_truth"] = ("data/evaluation/day1/feature_inventory.json; base technical calculations in technical_features_xgboost_v2.py; expanded one-day features in day1_phase1_merged_validation_with_additional_features.py; source-field mappings and corporate-action adjustment rules in build_canonical_price_dataset.py")
    source_passthrough = df.feature_name.isin({
        "prev_close", "last", "vwap", "traded_value", "num_trades",
        "deliverable_qty", "deliverable_pct",
    })
    needs_review = df.operational_formula.str.startswith("FORMULA REVIEW REQUIRED")
    df["formula_status"] = np.select(
        [needs_review, source_passthrough],
        ["REVIEW", "mapped source field; no engineered formula"],
        default="mapped to documented feature-generation formula",
    )
    return df


def feature_eda(parts: dict[str, pd.DataFrame], features: list[str], out: Path) -> dict[str, Any]:
    stats_rows = []
    missing_rows = []
    outlier_rows = []
    target_stats = []

    # Training-derived Tukey bounds apply to all partitions to avoid defining
    # outliers based on validation/test outcomes.
    tr = parts["train"]
    tukey_bounds: dict[str, tuple[float, float]] = {}
    for feature in features:
        s = pd.to_numeric(tr[feature], errors="coerce")
        q1, q3 = s.quantile([0.25, 0.75])
        iqr = q3 - q1
        tukey_bounds[feature] = (q1 - 1.5 * iqr, q3 + 1.5 * iqr)

    for part_name, df in parts.items():
        for feature in features:
            s = pd.to_numeric(df[feature], errors="coerce")
            finite = s[np.isfinite(s)]
            stats_rows.append({
                "partition": part_name,
                "feature": feature,
                "rows": len(df),
                "non_missing_count": int(s.notna().sum()),
                "missing_count": int(s.isna().sum()),
                "missing_pct": float(s.isna().mean() * 100),
                "mean": finite.mean() if len(finite) else np.nan,
                "std": finite.std() if len(finite) else np.nan,
                "min": finite.min() if len(finite) else np.nan,
                "p01": finite.quantile(.01) if len(finite) else np.nan,
                "p05": finite.quantile(.05) if len(finite) else np.nan,
                "p25": finite.quantile(.25) if len(finite) else np.nan,
                "median": finite.median() if len(finite) else np.nan,
                "p75": finite.quantile(.75) if len(finite) else np.nan,
                "p95": finite.quantile(.95) if len(finite) else np.nan,
                "p99": finite.quantile(.99) if len(finite) else np.nan,
                "max": finite.max() if len(finite) else np.nan,
            })
            missing_rows.append({
                "partition": part_name,
                "feature": feature,
                "rows": len(df),
                "missing_count": int(s.isna().sum()),
                "missing_pct": float(s.isna().mean() * 100),
            })
            lo, hi = tukey_bounds[feature]
            flagged = ((s < lo) | (s > hi)) & s.notna()
            outlier_rows.append({
                "partition": part_name,
                "feature": feature,
                "training_q1": float(tr[feature].quantile(.25)) if tr[feature].notna().any() else None,
                "training_q3": float(tr[feature].quantile(.75)) if tr[feature].notna().any() else None,
                "training_iqr_lower_bound": float(lo) if pd.notna(lo) else None,
                "training_iqr_upper_bound": float(hi) if pd.notna(hi) else None,
                "observed_nonmissing": int(s.notna().sum()),
                "tukey_iqr_outlier_count": int(flagged.sum()),
                "tukey_iqr_outlier_pct_of_nonmissing": float(flagged.sum() / max(1, s.notna().sum()) * 100),
                "interpretation": "Statistical flag only; not proof of a bad record. Verify against price source/corporate actions before classifying as an error.",
            })
        target = pd.to_numeric(df[TARGET_RETURN], errors="coerce")
        target_stats.append({
            "partition": part_name,
            "rows": len(df),
            "target_return_mean": float(target.mean()),
            "target_return_std": float(target.std()),
            "target_return_min": float(target.min()),
            "target_return_p01": float(target.quantile(.01)),
            "target_return_p05": float(target.quantile(.05)),
            "target_return_median": float(target.median()),
            "target_return_p95": float(target.quantile(.95)),
            "target_return_p99": float(target.quantile(.99)),
            "target_return_max": float(target.max()),
            "target_positive_prevalence": float((df[TARGET_DIRECTION].astype(int) == 1).mean()),
            "return_gt_20pct_count": int((target > .20).sum()),
            "return_lt_minus20pct_count": int((target < -.20).sum()),
            "return_gt_50pct_count": int((target > .50).sum()),
            "return_lt_minus50pct_count": int((target < -.50).sum()),
            "unique_stocks": int(df.symbol.nunique()),
            "unique_dates": int(df.trade_date.nunique()),
        })

    pd.DataFrame(stats_rows).to_csv(out / "feature_descriptive_statistics_by_partition.csv", index=False)
    pd.DataFrame(missing_rows).to_csv(out / "feature_missingness_by_partition.csv", index=False)
    pd.DataFrame(outlier_rows).to_csv(out / "feature_outlier_counts_by_partition.csv", index=False)
    pd.DataFrame(target_stats).to_csv(out / "target_descriptive_statistics_by_partition.csv", index=False)

    # Export all next-session price-return extremes >= 20% absolute for raw
    # record/source/corporate-action review. They remain in the dataset.
    extreme_frames = []
    for part_name, df in parts.items():
        ret = pd.to_numeric(df[TARGET_RETURN], errors="coerce")
        mask = ret.abs() >= 0.20
        if mask.any():
            cols = [c for c in [
                "symbol", "trade_date", TARGET_RETURN, TARGET_DIRECTION,
                "open", "high", "low", "close", "prev_close", "volume",
                "return_1d", "return_5d", "gap_pct", "intraday_return",
                "volume_ratio_20d", "nifty50_return_1d", "stock_vs_nifty50_5d",
            ] if c in df.columns]
            tab = df.loc[mask, cols].copy()
            tab.insert(0, "partition", part_name)
            tab["extreme_abs_target_return_pct"] = ret.loc[mask].abs() * 100
            tab["review_status"] = "PENDING SOURCE PRICE / CORPORATE-ACTION CHECK; DO NOT REMOVE AUTOMATICALLY"
            extreme_frames.append(tab)
    extremes = pd.concat(extreme_frames, ignore_index=True) if extreme_frames else pd.DataFrame(
        columns=["partition", "symbol", "trade_date", TARGET_RETURN, "extreme_abs_target_return_pct", "review_status"]
    )
    extremes.to_csv(out / "target_return_extreme_candidates_for_source_review.csv", index=False)

    # Correlations are estimated using TRAIN only; no validation/test outcome is
    # involved in defining the correlation structure or selecting features.
    train_num = tr[features + [TARGET_RETURN]].apply(pd.to_numeric, errors="coerce")
    corr = train_num.corr(method="pearson")
    corr.to_csv(out / "train_feature_and_target_correlation_matrix.csv")
    target_corr = corr[TARGET_RETURN].drop(TARGET_RETURN).sort_values(
        key=lambda x: x.abs(), ascending=False
    )
    target_corr.rename("pearson_correlation_with_target_return_1d").to_csv(
        out / "train_feature_target_correlations.csv", header=True
    )

    # Figures for the report: target distribution by partition, selected
    # example features across the broad families, and top correlations.
    fig, ax = plt.subplots(figsize=(9, 5.4))
    for name, df in parts.items():
        vals = pd.to_numeric(df[TARGET_RETURN], errors="coerce")
        vals = vals[np.isfinite(vals)]
        ax.hist(vals.clip(-.20, .20), bins=100, histtype="step", density=True, label=name)
    ax.set_title("One-session target return distribution (values clipped for display only)")
    ax.set_xlabel("target_return_1d (decimal return)")
    ax.set_ylabel("Density")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "fig_target_return_distribution.png", dpi=180)
    plt.close(fig)

    preferred_examples = [
        "return_1d", "return_5d", "rsi_14", "volatility_20d",
        "volume_ratio_20d", "gap_pct", "close_vs_sma50",
        "nifty50_return_1d", "stock_vs_nifty50_5d", "rolling_beta_20d",
    ]
    to_plot = [f for f in preferred_examples if f in features][:10]
    if to_plot:
        ncols = 2
        nrows = int(np.ceil(len(to_plot) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3.1*nrows))
        axes = np.array(axes).reshape(-1)
        for ax, feature in zip(axes, to_plot):
            vals = pd.to_numeric(tr[feature], errors="coerce")
            vals = vals[np.isfinite(vals)]
            if len(vals):
                lo, hi = vals.quantile(.01), vals.quantile(.99)
                ax.hist(vals.clip(lo, hi), bins=50)
            ax.set_title(feature)
        for ax in axes[len(to_plot):]:
            ax.set_visible(False)
        fig.suptitle("Representative E0 feature distributions — training partition", y=1.005)
        fig.tight_layout()
        fig.savefig(out / "fig_representative_feature_distributions.png", dpi=180, bbox_inches="tight")
        plt.close(fig)

    strongest = target_corr.dropna().head(20)
    fig, ax = plt.subplots(figsize=(9, 7))
    strongest.sort_values().plot(kind="barh", ax=ax)
    ax.set_title("Twenty largest absolute training-set correlations with next-session return")
    ax.set_xlabel("Pearson correlation")
    fig.tight_layout()
    fig.savefig(out / "fig_train_feature_target_correlations.png", dpi=180)
    plt.close(fig)

    return {
        "feature_count": len(features),
        "formula_review_count": int(inventory_table(features).formula_status.eq("REVIEW").sum()),
        "top_target_correlations": target_corr.head(10).to_dict(),
        "feature_missingness_file": "feature_missingness_by_partition.csv",
        "feature_outliers_method": "Tukey 1.5*IQR bounds derived from train features, counted separately in each partition",
    }


def daily_metrics(frame: pd.DataFrame, pred_col: str, probability_col: str | None = None) -> pd.DataFrame:
    rows = []
    for date, g in frame.groupby("trade_date", sort=True):
        y = g[TARGET_RETURN].to_numpy(dtype=float)
        pred = g[pred_col].to_numpy(dtype=float)
        row: dict[str, Any] = {
            "trade_date": date,
            "daily_rmse_model": float(np.sqrt(np.mean((y - pred) ** 2))),
            "daily_rmse_zero": float(np.sqrt(np.mean(y ** 2))),
            "daily_mae_model": float(np.mean(np.abs(y - pred))),
            "daily_mae_zero": float(np.mean(np.abs(y))),
        }
        model_scores_vary = g[pred_col].nunique() > 1
        momentum_scores_vary = g["prediction_5d_momentum"].nunique() > 1
        targets_vary = g[TARGET_RETURN].nunique() > 1
        same_date_rankings_valid = model_scores_vary and momentum_scores_vary
        if same_date_rankings_valid and targets_vary:
            row["spearman_model"] = g[pred_col].corr(g[TARGET_RETURN], method="spearman")
            row["spearman_momentum"] = g["prediction_5d_momentum"].corr(g[TARGET_RETURN], method="spearman")
        if probability_col:
            ycls = g[TARGET_DIRECTION].to_numpy(dtype=int)
            prob = g[probability_col].to_numpy(dtype=float)
            pprev = g["prevalence_probability"].to_numpy(dtype=float)
            row["brier_model"] = float(np.mean((ycls - prob) ** 2))
            row["brier_prevalence"] = float(np.mean((ycls - pprev) ** 2))
            eps = 1e-15
            prob_clip = np.clip(prob, eps, 1 - eps)
            pprev_clip = np.clip(pprev, eps, 1 - eps)
            row["logloss_model"] = float(-np.mean(ycls*np.log(prob_clip)+(1-ycls)*np.log(1-prob_clip)))
            row["logloss_prevalence"] = float(-np.mean(ycls*np.log(pprev_clip)+(1-ycls)*np.log(1-pprev_clip)))
            if len(np.unique(ycls)) == 2:
                row["auc_model"] = float(roc_auc_score(ycls, prob))
                row["auc_chance"] = 0.5
        # Top-K comparison is recorded only when both signals can rank this
        # date; this ensures E0 and momentum use identical dates for each K.
        for k in TOP_K:
            if len(g) >= k and same_date_rankings_valid:
                gm = g.sort_values([pred_col, "symbol"], ascending=[False, True], kind="mergesort").head(k)
                gr = g.sort_values(["prediction_5d_momentum", "symbol"], ascending=[False, True], kind="mergesort").head(k)
                row[f"top{k}_return_model"] = float(gm[TARGET_RETURN].mean())
                row[f"top{k}_return_momentum"] = float(gr[TARGET_RETURN].mean())
                row[f"top{k}_hit_model"] = float((gm[TARGET_RETURN] > 0).mean())
                row[f"top{k}_hit_momentum"] = float((gr[TARGET_RETURN] > 0).mean())
        rows.append(row)
    return pd.DataFrame(rows).sort_values("trade_date").reset_index(drop=True)


def circular_block_ci(values: np.ndarray, seed: int = RANDOM_SEED) -> tuple[float, float, float, int]:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < BLOCK_LENGTH:
        return (float(np.mean(x)) if n else float("nan"), float("nan"), float("nan"), n)
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / BLOCK_LENGTH))
    offsets = np.arange(BLOCK_LENGTH)
    boot = np.empty(BOOTSTRAP_REPS, dtype=float)
    for i in range(BOOTSTRAP_REPS):
        starts = rng.integers(0, n, size=n_blocks)
        ids = ((starts[:, None] + offsets[None, :]) % n).ravel()[:n]
        boot[i] = np.mean(x[ids])
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return float(x.mean()), float(lo), float(hi), n


def ci_record(name: str, diff: pd.Series | np.ndarray, better: str) -> dict[str, Any]:
    mean, low, high, n = circular_block_ci(np.asarray(diff, dtype=float))
    return {
        "metric_comparison": name,
        "difference_convention": "model/reference difference (model minus reference)",
        "direction_of_benefit": better,
        "paired_dates": n,
        "mean_daily_difference": mean,
        "nominal_95pct_ci_lower": low,
        "nominal_95pct_ci_upper": high,
        "bootstrap_method": "paired circular moving-block bootstrap over trading dates",
        "block_length_trading_days": BLOCK_LENGTH,
        "bootstrap_iterations": BOOTSTRAP_REPS,
        "multiple_comparison_adjustment": "none; exploratory nominal interval",
    }


def evaluate_final_test(test: pd.DataFrame, features: list[str], train: pd.DataFrame, out: Path) -> dict[str, Any]:
    # Reuse the stored E0 models; do not retrain after opening the final test.
    config = baseline.load_config(baseline.CONFIG_FILE)
    regressor = baseline.build_regressor(config)
    classifier = baseline.build_classifier(config)
    if not REG_MODEL_FILE.exists() or not CLS_MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Stored E0 models not found: {REG_MODEL_FILE} / {CLS_MODEL_FILE}. "
            "Do not train/reselect after examining test results."
        )
    regressor.load_model(str(REG_MODEL_FILE))
    classifier.load_model(str(CLS_MODEL_FILE))

    X_test, y_reg, y_cls = baseline.build_xy(test, features)
    predicted_return = regressor.predict(X_test)
    predicted_probability = classifier.predict_proba(X_test)[:, 1]
    prevalence = float(train[TARGET_DIRECTION].astype(int).mean())

    predictions = test[KEYS + [TARGET_RETURN, TARGET_DIRECTION]].copy()
    predictions["predicted_return_1d"] = predicted_return
    predictions["predicted_probability_1d"] = predicted_probability
    predictions["prevalence_probability"] = prevalence
    predictions["prediction_5d_momentum"] = pd.to_numeric(
        test[MOMENTUM_FEATURE], errors="coerce"
    ) if MOMENTUM_FEATURE in test.columns else np.nan
    if predictions["prediction_5d_momentum"].notna().sum() == 0:
        raise ValueError(
            "return_5d is missing/unavailable in all test rows. Cannot compare "
            "the one-day model with the trailing-five-session momentum reference."
        )
    predictions["prediction_rank"] = predictions.groupby("trade_date")["predicted_return_1d"].rank(method="first", ascending=False).astype("Int64")
    predictions.to_parquet(out / "final_test_predictions.parquet", index=False)

    y = predictions[TARGET_RETURN].to_numpy(dtype=float)
    yp = predictions["predicted_return_1d"].to_numpy(dtype=float)
    zero = np.zeros_like(y)
    reg_metrics = {
        "model": "E0_BASELINE",
        "horizon_sessions": 1,
        "rows": len(predictions),
        "dates": int(predictions.trade_date.nunique()),
        "stocks": int(predictions.symbol.nunique()),
        "model_rmse": float(np.sqrt(mean_squared_error(y, yp))),
        "zero_return_rmse": float(np.sqrt(mean_squared_error(y, zero))),
        "model_mae": float(mean_absolute_error(y, yp)),
        "zero_return_mae": float(mean_absolute_error(y, zero)),
        "model_r2": float(r2_score(y, yp)),
        "model_pearson": float(np.corrcoef(y, yp)[0, 1]) if np.std(yp)>0 else None,
    }
    reg_metrics["interpretation"] = "Error comparisons are descriptive; lower is better. No profitability claim is implied."
    (out / "final_test_regression_metrics.json").write_text(json.dumps(reg_metrics, indent=2), encoding="utf-8")

    yc = predictions[TARGET_DIRECTION].astype(int).to_numpy()
    p = predictions["predicted_probability_1d"].to_numpy(dtype=float)
    pp = np.full_like(p, prevalence, dtype=float)
    yhat = (p >= 0.5).astype(int)
    cls_metrics = {
        "positive_class_prevalence_from_training": prevalence,
        "rows": len(yc),
        "dates": int(predictions.trade_date.nunique()),
        "model_threshold_fixed_at": 0.5,
        "model_accuracy": float(accuracy_score(yc, yhat)),
        "model_precision": float(precision_score(yc, yhat, zero_division=0)),
        "model_recall": float(recall_score(yc, yhat, zero_division=0)),
        "model_f1": float(f1_score(yc, yhat, zero_division=0)),
        "model_roc_auc": float(roc_auc_score(yc, p)) if len(np.unique(yc)) == 2 else None,
        "model_pr_auc_average_precision": float(average_precision_score(yc, p)) if len(np.unique(yc)) == 2 else None,
        "model_brier_score": float(brier_score_loss(yc, p)),
        "prevalence_baseline_brier_score": float(brier_score_loss(yc, pp)),
        "model_log_loss": float(log_loss(yc, p, labels=[0, 1])),
        "prevalence_baseline_log_loss": float(log_loss(yc, pp, labels=[0, 1])),
        "prevalence_baseline_probability_is_constant": True,
        "note": "Prevalence probability is estimated only from training labels; threshold fixed at 0.5 for model classifications. Lower Brier/log loss is better.",
    }
    (out / "final_test_classification_metrics.json").write_text(json.dumps(cls_metrics, indent=2), encoding="utf-8")

    # Daily regression/classification metrics use every valid test observation.
    # Ranking metrics below use only matched stock/date rows where the lagged
    # return_5d reference is available, applying the same eligible rows to both
    # ranking signals. This remains a one-session supervised target.
    daily = daily_metrics(predictions, "predicted_return_1d", "predicted_probability_1d")
    daily.to_csv(out / "final_test_daily_metrics.csv", index=False)
    eligible = predictions.dropna(subset=["prediction_5d_momentum"]).copy()
    ranking_daily = daily_metrics(eligible, "predicted_return_1d", "predicted_probability_1d")
    ranking_daily.to_csv(out / "final_test_ranking_daily_metrics_matched_dates.csv", index=False)

    ranking_summary: dict[str, Any] = {
        "horizon_sessions": 1,
        "model_rank_signal": "E0 predicted_return_1d",
        "reference_rank_signal": "lagged return_5d feature (trailing five-session return); outcome remains target_return_1d",
        "dates": int(ranking_daily.trade_date.nunique()),
        "eligible_dates_with_momentum_reference": int(eligible.trade_date.nunique()),
        "stocks": int(eligible.symbol.nunique()),
        "eligible_stock_date_rows_for_matched_ranking": int(len(eligible)),
        "stock_date_rows_missing_momentum_reference": int(predictions["prediction_5d_momentum"].isna().sum()),
    }
    for who, col in [("model", "spearman_model"), ("momentum_reference", "spearman_momentum")]:
        vals = ranking_daily[col].dropna() if col in ranking_daily else pd.Series(dtype=float)
        ranking_summary[f"{who}_daily_spearman_mean"] = float(vals.mean()) if len(vals) else None
        ranking_summary[f"{who}_daily_spearman_dates"] = int(len(vals))
    for k in TOP_K:
        for metric in [f"top{k}_return", f"top{k}_hit"]:
            for who in ["model", "momentum"]:
                col = f"{metric}_{who}"
                vals = ranking_daily[col].dropna() if col in ranking_daily else pd.Series(dtype=float)
                ranking_summary[f"{who}_{metric}"] = float(vals.mean()) if len(vals) else None
    (out / "final_test_ranking_metrics.json").write_text(json.dumps(ranking_summary, indent=2), encoding="utf-8")

    ci_rows = []
    # Regression: model versus zero-return predictor; negative differences favor model.
    ci_rows.append(ci_record("daily RMSE: E0 minus zero-return forecast", daily["daily_rmse_model"]-daily["daily_rmse_zero"], "negative favors E0"))
    ci_rows.append(ci_record("daily MAE: E0 minus zero-return forecast", daily["daily_mae_model"]-daily["daily_mae_zero"], "negative favors E0"))
    # Classification: model minus prevalence-only baseline; negative favours model.
    ci_rows.append(ci_record("daily Brier: E0 minus training-prevalence baseline", daily["brier_model"]-daily["brier_prevalence"], "negative favors E0"))
    ci_rows.append(ci_record("daily log loss: E0 minus training-prevalence baseline", daily["logloss_model"]-daily["logloss_prevalence"], "negative favors E0"))
    if "auc_model" in daily:
        ci_rows.append(ci_record("daily ROC-AUC: E0 minus chance reference 0.5", daily["auc_model"]-daily["auc_chance"], "positive favors E0"))
    # Ranking: E0 minus momentum; positive favours E0 for Rank IC and return/hit metrics.
    ci_rows.append(ci_record("daily Spearman: E0 minus trailing-5-session momentum", ranking_daily["spearman_model"]-ranking_daily["spearman_momentum"], "positive favors E0"))
    for k in TOP_K:
        ci_rows.append(ci_record(f"Top-{k} mean return: E0 minus momentum", ranking_daily[f"top{k}_return_model"]-ranking_daily[f"top{k}_return_momentum"], "positive favors E0"))
        ci_rows.append(ci_record(f"Top-{k} hit rate: E0 minus momentum", ranking_daily[f"top{k}_hit_model"]-ranking_daily[f"top{k}_hit_momentum"], "positive favors E0"))
    ci_df = pd.DataFrame(ci_rows)
    ci_df.to_csv(out / "final_test_paired_confidence_intervals.csv", index=False)

    return {"regression": reg_metrics, "classification": cls_metrics, "ranking": ranking_summary, "paired_ci_rows": len(ci_rows)}


def validation_outlier_diagnostics(val: pd.DataFrame, features: list[str], out: Path) -> dict[str, Any]:
    """Find poor validation Rank-IC days and high row/stock errors; do not remove them."""
    if not VAL_PRED_FILE.exists():
        return {"warning": f"Validation prediction file not found: {VAL_PRED_FILE}"}
    pred = pd.read_parquet(VAL_PRED_FILE)
    required = set(KEYS + [TARGET_RETURN, "predicted_return_1d"])
    if not required.issubset(pred.columns):
        return {"warning": f"Validation prediction file missing: {sorted(required-set(pred.columns))}"}
    pred["trade_date"] = pd.to_datetime(pred.trade_date).dt.normalize()
    pred = pred.copy()
    pred["actual_return"] = pd.to_numeric(pred[TARGET_RETURN], errors="coerce")
    pred["predicted_return"] = pd.to_numeric(pred["predicted_return_1d"], errors="coerce")
    pred["absolute_error"] = (pred.actual_return - pred.predicted_return).abs()
    # Add the actual validation-row context available in the modeling dataset.
    # This is a review table only; it does not change the model or filter rows.
    candidate_context = [
        "open", "high", "low", "close", "prev_close", "volume",
        "return_1d", "return_5d", "volatility_20d", "atr_pct_14",
        "volume_ratio_20d", "gap_pct", "intraday_return",
        "high_low_range_pct", "nifty50_return_1d", "stock_vs_nifty50_5d",
    ]
    use_context = [c for c in candidate_context if c in val.columns]
    if use_context:
        context = val[KEYS + use_context].drop_duplicates(KEYS)
        pred = pred.merge(context, on=KEYS, how="left", validate="one_to_one")
    daily = []
    for date, g in pred.groupby("trade_date", sort=True):
        corr = g.predicted_return.corr(g.actual_return, method="spearman") if g.predicted_return.nunique()>1 and g.actual_return.nunique()>1 else np.nan
        daily.append({"trade_date": date, "daily_spearman": corr, "stocks": g.symbol.nunique(), "daily_rmse": float(np.sqrt(np.mean((g.actual_return-g.predicted_return)**2)))})
    daily = pd.DataFrame(daily).sort_values("daily_spearman")
    daily.head(20).to_csv(out / "validation_weakest_rank_ic_dates.csv", index=False)
    pred.sort_values("absolute_error", ascending=False).head(100).to_csv(out / "validation_largest_stock_level_errors.csv", index=False)
    stock = pred.groupby("symbol").agg(
        observations=("absolute_error", "size"),
        stock_mae=("absolute_error", "mean"),
        stock_max_abs_error=("absolute_error", "max"),
        stock_rmse=("absolute_error", lambda x: float(np.sqrt(np.mean(np.square(x)))),),
    ).reset_index().sort_values("stock_mae", ascending=False)
    stock.to_csv(out / "validation_stock_error_summary.csv", index=False)

    # Make a human-review template from the worst Rank-IC date and largest errors.
    candidates = pred.merge(daily.head(1)[["trade_date", "daily_spearman"]], on="trade_date", how="inner")
    if not candidates.empty:
        candidates["review_question"] = "Check canonical OHLCV, adjustment/symbol history, target T+1 calculation, and contemporaneous corporate announcements; do not delete based on size alone."
        candidates["source_check_status"] = "PENDING MANUAL VERIFICATION"
        candidates.to_csv(out / "weakest_rank_ic_stock_level_event_review_template.csv", index=False)
    return {
        "validation_dates": int(pred.trade_date.nunique()),
        "weakest_rank_ic_date": iso(daily.iloc[0].trade_date) if len(daily) else None,
        "weakest_rank_ic": float(daily.iloc[0].daily_spearman) if len(daily) and pd.notna(daily.iloc[0].daily_spearman) else None,
        "largest_error_rows_exported": min(100, len(pred)),
        "outlier_rows_are_removed": False,
    }


def build_text_summary(out: Path, freeze: dict[str, Any], eda: dict[str, Any], final: dict[str, Any], validation_diag: dict[str, Any], formula_review: pd.DataFrame) -> None:
    lines = [
        "DAY-1 FINAL SUBMISSION EVIDENCE SUMMARY",
        "=======================================",
        "Prediction horizon: one NSE trading session ahead (T close to next NSE session close).",
        f"Final candidate frozen before test evaluation: {FINAL_MODEL_CHOICE['model']} / {FINAL_MODEL_CHOICE['feature_set']}",
        "Test data used for model selection: NO; test was evaluated only after the freeze manifest was written.",
        "",
        "FROZEN PARTITION MANIFEST",
        json.dumps(freeze, indent=2),
        "",
        "EDA SUMMARY",
        json.dumps(eda, indent=2),
        "",
        "FEATURE FORMULA CATALOG QA",
        f"Feature inventory count: {len(formula_review)}",
        f"Formula review rows remaining: {int(formula_review.formula_status.eq('REVIEW').sum())}",
        "If formula review rows remain, trace those exact names to the versioned generator before claiming a complete formula inventory.",
        "",
        "VALIDATION ANOMALY DIAGNOSTICS (NOT USED FOR MODEL SELECTION)",
        json.dumps(validation_diag, indent=2),
        "",
        "FINAL TEST METRICS (E0, evaluated once)",
        json.dumps(final, indent=2),
        "",
        "INTERPRETATION GUARDRAILS",
        "1. The trailing-five-session momentum feature is a reference ranking signal only; the supervised target remains one session ahead.",
        "2. Tukey flags are statistical outlier indicators, not proof of erroneous records; no rows were removed.",
        "3. Bootstrap intervals are nominal and unadjusted for the number of metrics examined.",
        "4. Top-K returns are gross close-to-close returns, not evidence of profitability after fees, slippage, turnover, liquidity, and portfolio constraints.",
        "5. The current constituent cohort is not a reconstructed historical NIFTY 100 constituent universe; conclusions are restricted to the frozen current-constituent cohort.",
        "6. Do not revise model/features after viewing final test metrics. If any data issue is found on test, disclose it and do not silently change the test set or model.",
    ]
    (out / "final_submission_evidence_summary.txt").write_text("\n".join(lines)+"\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    test_guard = OUT_DIR / "FINAL_TEST_EVALUATION_ALREADY_RECORDED.json"
    if test_guard.exists():
        raise SystemExit(
            "Guard stopped this run: a final test evaluation is already recorded at "
            f"{test_guard}. Do not inspect the test repeatedly. Review existing outputs or use a separate, justified audit process."
        )
    for path in [TRAIN_FILE, VALIDATION_FILE, TEST_FILE, INVENTORY_FILE, REG_MODEL_FILE, CLS_MODEL_FILE]:
        if not path.exists():
            raise FileNotFoundError(f"Required Day-1 artifact not found: {path}")

    features = load_inventory()
    # Read train + validation first. These are the only labelled partitions used
    # for the model-selection decision included in the freeze record.
    train = load_frame(TRAIN_FILE, "Train", features)
    validation = load_frame(VALIDATION_FILE, "Validation", features)
    check_temporal_order([("train", train), ("validation", validation)])

    inventory_df = inventory_table(features)
    inventory_csv = OUT_DIR / "feature_inventory_90_with_formulas.csv"
    inventory_df.to_csv(inventory_csv, index=False)
    unresolved = inventory_df.loc[inventory_df.formula_status.eq("REVIEW"), "feature_name"].tolist()
    if unresolved:
        raise SystemExit(
            "Stopped BEFORE opening the final test partition because some E0 feature formulas "
            "still need source mapping. The actual 90-feature inventory is in: "
            f"{inventory_csv}. Unresolved names: {unresolved}. Upload that CSV so the formulas "
            "can be mapped to the local versioned generator before the one-time final test."
        )

    freeze = {
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "prediction_horizon_sessions": 1,
        "split_policy": "Freeze and retain the already-trained model's existing chronological train/validation/test Parquet partitions; do not rebuild to the previously planned 65/15/20 split late in the study. Record actual date cutoffs and row/date counts. The test partition is not used to choose the model or features.",
        "model_choice": FINAL_MODEL_CHOICE,
        "partitions_before_test_open": [
            partition_info("train", train, TRAIN_FILE),
            partition_info("validation", validation, VALIDATION_FILE),
        ],
        "pending_test_partition_path": str(TEST_FILE),
        "feature_inventory_path": str(INVENTORY_FILE),
        "feature_inventory_sha256": sha256(INVENTORY_FILE),
        "E0_regression_model_path": str(REG_MODEL_FILE),
        "E0_classification_model_path": str(CLS_MODEL_FILE),
        "note": "E0 model/feature choice is frozen using pre-existing validation evidence before test file is opened. The existing split is retained rather than replaced with new boundaries late in the analysis.",
    }
    freeze_path = OUT_DIR / "final_model_and_split_freeze.json"
    freeze_path.write_text(json.dumps(freeze, indent=2), encoding="utf-8")
    # From this point onward, the model choice and split are considered frozen.

    # Create a durable in-progress guard BEFORE loading the test partition. If
    # this run fails after test access starts, a rerun must not inspect the
    # same holdout again as if it were untouched.
    test_guard.write_text(json.dumps({
        "evaluation_started_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_status": "IN_PROGRESS",
        "test_access_started": True,
        "test_file": str(TEST_FILE),
        "test_used_for_model_selection": False,
        "selected_model": FINAL_MODEL_CHOICE["model"],
        "instruction": "Do not rerun this script if status remains IN_PROGRESS. Inspect outputs and document the failure; the test partition has already been accessed after the freeze.",
    }, indent=2), encoding="utf-8")

    # Load test only after writing the freeze record and in-progress guard.
    test = load_frame(TEST_FILE, "Test", features)
    check_temporal_order([("train", train), ("validation", validation), ("test", test)])
    freeze["partitions_after_final_test_open"] = [
        partition_info("train", train, TRAIN_FILE),
        partition_info("validation", validation, VALIDATION_FILE),
        partition_info("test", test, TEST_FILE),
    ]
    freeze["final_test_opened_after_freeze"] = True
    freeze["final_test_sha256"] = sha256(TEST_FILE)
    freeze_path.write_text(json.dumps(freeze, indent=2), encoding="utf-8")

    parts = {"train": train, "validation": validation, "test": test}
    eda = feature_eda(parts, features, OUT_DIR)

    # Create validation-only anomaly diagnostics; never use test to select or drop records.
    validation_diag = validation_outlier_diagnostics(validation, features, OUT_DIR)

    # Final single holdout evaluation: saved E0 models only; no test refit.
    final = evaluate_final_test(test, features, train, OUT_DIR)
    guard = {
        "evaluation_recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_status": "COMPLETED",
        "test_access_started": True,
        "test_file_sha256": sha256(TEST_FILE),
        "test_used": True,
        "used_for_model_selection": False,
        "selected_model": FINAL_MODEL_CHOICE["model"],
        "evaluation_files": [
            "final_test_regression_metrics.json",
            "final_test_classification_metrics.json",
            "final_test_ranking_metrics.json",
            "final_test_paired_confidence_intervals.csv",
            "final_test_predictions.parquet",
        ],
    }
    test_guard.write_text(json.dumps(guard, indent=2), encoding="utf-8")
    build_text_summary(OUT_DIR, freeze, eda, final, validation_diag, inventory_df)

    zip_path = OUT_DIR / "final_submission_evidence_bundle.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for item in sorted(OUT_DIR.iterdir()):
            if item.is_file() and item != zip_path:
                z.write(item, arcname=item.name)
    print("\nSTATUS: EVIDENCE GENERATED")
    print("Output directory:", OUT_DIR.resolve())
    print("Bundle:", zip_path.resolve())
    print("Feature count:", len(features))
    print("Feature formulas needing manual source mapping:", eda["formula_review_count"])
    print("Train/validation/test rows:", len(train), len(validation), len(test))
    print("Train/validation/test date counts:", train.trade_date.nunique(), validation.trade_date.nunique(), test.trade_date.nunique())
    print("Final candidate:", FINAL_MODEL_CHOICE["model"])
    print("TEST USED FOR FINAL SCORING: YES (after freeze; do not use for later selection)")
    print("Review summary:", (OUT_DIR / "final_submission_evidence_summary.txt").resolve())


if __name__ == "__main__":
    main()
