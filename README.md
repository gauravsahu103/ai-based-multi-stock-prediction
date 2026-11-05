# AI-Based Multi-Stock Market Prediction & Ranking System

## Overview

This project builds a daily, leakage-aware stock prediction pipeline for NSE equities.

- Universe: NIFTY 100
- Market: NSE (cash equities)
- Main horizons: Day-1 and Day-5
- Main models: XGBoost (classification + regression), LSTM (excess-return experiment)

## Pipeline (Current)

1. Ingest raw NSE master, universe, index, and stock prices.
2. Normalize and validate processed datasets.
3. Build technical features.
4. Generate Day-1 and Day-5 targets.
5. Build day-specific modeling datasets.
6. Train Day-5 models.
7. Run diagnostics/ranking evaluation.

## Configuration

- Primary config file: `config/mvp.yaml`
- Central path helpers: `src/utils/mvp_config.py`

All core day1/day5 scripts now use the config-driven storage paths.

## Directory Structure

```text
Project/
├── config/
│   └── mvp.yaml
├── data/
│   ├── raw/                 # downloaded NSE source files
│   ├── processed/           # normalized datasets (prices/index/universe/modeling)
│   ├── features/            # engineered features (day-specific/versioned)
│   ├── targets/             # generated day1/day5 targets
│   ├── predictions/         # model predictions + ranking diagnostics
│   ├── backtests/
│   └── evaluation/          # data-quality and evaluation outputs
├── docs/
│   └── SOURCE_FILE_GUIDE.md
├── models/                  # serialized trained models
├── src/
│   ├── ingestion/           # raw data download + normalization
│   ├── features/
│   │   └── day1/            # day1 feature generators
│   ├── targets/             # day1/day5 target generation
│   ├── preprocessing/       # day1/day5 modeling dataset builders/inspectors/validators
│   ├── models/              # split logic + day5 trainers
│   ├── evaluation/          # day5 diagnostics/evaluations
│   └── utils/               # config/path helpers
├── tests/
└── README.md
```

## File-by-File Documentation

Complete module-level documentation (purpose, inputs/outputs, and logic) is in:

- `docs/SOURCE_FILE_GUIDE.md`

This includes every Python file under `src/`.

## Common Run Commands

From project root:

```bash
# Ingestion
python -m src.ingestion.download_nse_security_master
python -m src.ingestion.nifty100_download
python -m src.ingestion.nse_security_master
python -m src.ingestion.universe_builder
python -m src.ingestion.nse_index_history
python -m src.ingestion.nse_price_history
python -m src.ingestion.nse_price_parser

# Feature engineering (baseline + expanded v2)
python -m src.features.day1.technical_features_xgboost
python -m src.features.day1.technical_features_xgboost_v2

# Day-1 integrated dataset validation + final modeling split build
python -m src.preprocessing.day1.dataset_validation_and_target

# Targets
python -m src.targets.day1.return_targets
python -m src.targets.day5.return_targets

# Modeling datasets
python -m src.preprocessing.day1.build_model_dataset
python -m src.preprocessing.day1.inspect_model_dataset
python -m src.preprocessing.day5.build_model_dataset
python -m src.preprocessing.day5.inspect_model_dataset

# Day-5 model training
python -m src.models.day5.xgboost_trainer
python -m src.models.day5.xgboost_return_trainer
python -m src.models.day5.xgboost_excess_return_trainer
python -m src.models.day5.lstm_excess_return_trainer

# Day-5 evaluation
python -m src.evaluation.day5.baseline_diagnostics
python -m src.evaluation.day5.return_ranking_diagnostics
python -m src.evaluation.day5.excess_return_baseline_comparison
python -m src.evaluation.day5.excess_return_ranking_diagnostics
python -m src.evaluation.day5.data_quality_diagnostics
```

Notes:

- `technical_features_xgboost` writes baseline feature files to `data/features/technical`.
- `technical_features_xgboost_v2` writes expanded features to `data/features/technical_v2`.
- `dataset_validation_and_target` validates day1 features + targets and writes final day1 modeling datasets/splits.
- If you want preprocessing scripts to consume v2 features, point the `features` storage path in `config/mvp.yaml` to `data/features/technical_v2`.

## Day-1 Notes (Chronological Worklog)

### Stage Flow (as executed)

| Order | Stage | What it creates | Input | Output | Why | Command |
|---:|---|---|---|---|---|---|
| 1 | Target Generation | The answer the model should learn | Price data | `target_return_1d`, `target_direction_1d` | Defines what to predict | `python -m src.targets.day1.return_targets` |
| 2 | Feature Generation | Information available at T | Price/market/sector data | Technical + additional feature columns | Defines what the model can use | `python -m src.features.day1.technical_features_xgboost_v2` |
| 3 | Modeling Dataset Assembly | One clean ML-ready table | Features + targets | `nse100_technical_model_dataset.parquet` + train/validation/test | Combines everything for training | `python -m src.preprocessing.day1.dataset_validation_and_target` |

### Baseline Setup Notes

- Data prepared through the above 3 stages.
- Next baseline steps were executed/planned:
  - `python -m src.models.day1.baseline_train`
  - Setup `day1.yaml` config usage.
  - Created `train_xgboost_baseline` flow for baseline XGBoost.

### Baseline Diagnosis Commands

Command 1:

```bash
python - <<'PY'
import pandas as pd

p = "data/predictions/day1/validation_predictions.parquet"
df = pd.read_parquet(p)

print("\n=== TARGET RETURN ===")
print(df["target_return_1d"].describe())

print("\n=== TARGET DIRECTION ===")
print(df["target_direction_1d"].value_counts(normalize=True).sort_index())

print("\n=== PREDICTED RETURN ===")
print(df["predicted_return_1d"].describe())

print("\n=== PREDICTED PROBABILITY ===")
print(df["predicted_probability_1d"].describe())

print("\n=== PREDICTED DIRECTION ===")
print(df["predicted_direction_1d"].value_counts(normalize=True).sort_index())

print("\n=== DAILY STOCK COUNT ===")
print(df.groupby("trade_date")["symbol"].count().describe())
PY
```

Command 2:

```bash
python - <<'PY'
import pandas as pd

p = "data/evaluation/day1/feature_importance.parquet"
df = pd.read_parquet(p)

print("Columns:")
print(df.columns.tolist())

print("\n=== REGRESSION TOP 30 ===")
print(
    df[df["model"] == "regression"]
    .sort_values("importance_gain", ascending=False)
    .head(30)
    .to_string(index=False)
)

print("\n=== CLASSIFICATION TOP 30 ===")
print(
    df[df["model"] == "classification"]
    .sort_values("importance_gain", ascending=False)
    .head(30)
    .to_string(index=False)
)
PY
```

Command 3:

```bash
python - <<'PY'
import pandas as pd

p = "data/processed/modeling/day1/validation.parquet"
df = pd.read_parquet(p)

cols = [
    "symbol",
    "trade_date",
    "target_return_1d",
    "target_direction_1d",
]

print(
    df.loc[
        df["target_return_1d"].abs().nlargest(20).index,
        cols
    ].sort_values("target_return_1d")
    .to_string(index=False)
)
PY
```

### Key Finding

Corporate-action contamination exists in target/price data.

| Stock | Target date | 1-day target | Corporate action around T+1 |
|---|---|---:|---|
| BAJFINANCE | 2025-06-13 | **-89.95%** | 1:2 split + 4:1 bonus → combined adjustment factor 10 |
| ADANIPOWER | 2025-09-19 | **-76.00%** | 1:5 face-value split, ex-date 2025-09-22 |
| PIDILITIND | 2025-09-22 | **-50.98%** | 1:1 bonus, ex-date 2025-09-23 |
| HDFCBANK | 2025-08-25 | **-50.44%** | 1:1 bonus around 25–26 Aug 2025 |
| HDFCAMC | 2025-11-25 | **-49.80%** | 1:1 bonus, ex-date 2025-11-26 |

Also observed: since target is based on `close[T+1] / close[T] - 1` and technical features also use `close`, split/bonus discontinuity contaminates both target and features.

### Priorities

**P0 — Must fix**
1. Corporate-action consistency
2. NIFTY feature calculation on stock-specific rows
3. Daily universe coverage
4. Target calendar-gap validation (currently considered okay)
5. Feature/target extreme-value audit
6. E0 feature eligibility / unwanted absolute-price fields
7. Exact duplicate-feature detection

**P1 — Must validate before final baseline**
8. Sector mapping/coverage
9. Cross-sectional rank correctness
10. Missingness by feature + stock
11. Train/validation distribution drift
12. Prediction-time consistency
13. Stock/sector/regime diagnostics

**P2 — Research limitation to document**
14. Current-universe / survivorship effect
15. Raw-return target vs future excess-return target

### Consolidated Correction Plan (recorded)

- Run `download_nse_reference_data`.
- Filter complex events (needs additional effort).
- Run `build_canonical_price_dataset`.
- Build calendar.
- Build calendar-aware day1 target.
- Update `technical_features_xgboost_v2`.

Reference merge logic:

```text
stock prices ------------------┐
                               ├── merge by trade_date
nifty50_market_features -------┘
                               ↓
                    stock-vs-NIFTY features
```

### Special Session Repair (NIFTY 2024-03-02)

Issue observed: `nifty50_daily.parquet` was missing 2024-03-02 (special NSE trading session).

Repair command (version 1):

```bash
python - <<'PY'
from pathlib import Path
import pandas as pd

path = Path("data/processed/index/nifty50_daily.parquet")

df = pd.read_parquet(path)
df["trade_date"] = pd.to_datetime(df["trade_date"])

target_date = pd.Timestamp("2024-03-02")

if target_date in set(df["trade_date"]):
    print("2024-03-02 already exists. Nothing to change.")
else:
    values = {
        "trade_date": target_date,
        "close": 22378.40,
    }

    optional_values = {
        "open": 22406.95,
        "high": 22419.55,
        "low": 22367.05,
    }

    for column, value in optional_values.items():
        if column in df.columns:
            values[column] = value

    row = {
        column: values.get(column, pd.NA)
        for column in df.columns
    }

    df = pd.concat(
        [df, pd.DataFrame([row])],
        ignore_index=True,
    )

    df = (
        df.drop_duplicates("trade_date", keep="first")
        .sort_values("trade_date")
        .reset_index(drop=True)
    )

    df.to_parquet(path, index=False)

    print("Added 2024-03-02 to NIFTY daily data.")
    print(f"Rows: {len(df):,}")
    print(
        df[
            df["trade_date"].between(
                "2024-02-29",
                "2024-03-04",
            )
        ].to_string(index=False)
    )

print("STATUS : PASSED")
PY
```

Re-run sequence after repair attempt:

- `python -m src.features.day1.build_nifty_market_reference`
- `python -m src.preprocessing.day1.build_nse_trading_calendar`
- `python -m src.features.day1.technical_features_xgboost_v2`

### Diagnostics and Re-check Loop

Coverage diagnostics command:

```bash
python - <<'PY'
import pandas as pd

files = {
    "NIFTY DAILY": "data/processed/index/nifty50_daily.parquet",
    "MARKET REF": "data/processed/index/nifty50_market_features.parquet",
    "CALENDAR": "data/processed/calendar/nse_trading_calendar.parquet",
}

target = pd.Timestamp("2024-03-02")

for name, path in files.items():
    df = pd.read_parquet(path)
    df["trade_date"] = pd.to_datetime(df["trade_date"])

    print("=" * 80)
    print(name)
    print("Rows:", len(df))
    print("Has 2024-03-02:", target in set(df["trade_date"]))

    print(
        df[
            df["trade_date"].between(
                "2024-02-29",
                "2024-03-04"
            )
        ][["trade_date"] + (
            ["close"] if "close" in df.columns else []
        )].to_string(index=False)
    )
PY
```

Repair command (version 2):

```bash
python - <<'PY'
from pathlib import Path
import pandas as pd

path = Path("data/processed/index/nifty50_daily.parquet")

df = pd.read_parquet(path)
df["trade_date"] = pd.to_datetime(df["trade_date"], errors="raise")

target_date = pd.Timestamp("2024-03-02")
target_close = 22378.40

if target_date in set(df["trade_date"]):
    print("2024-03-02 already exists.")
else:
    row = {column: pd.NA for column in df.columns}

    row["trade_date"] = target_date
    row["close"] = target_close

    df = pd.concat(
        [df, pd.DataFrame([row])],
        ignore_index=True,
    )

    df = (
        df
        .drop_duplicates("trade_date", keep="first")
        .sort_values("trade_date")
        .reset_index(drop=True)
    )

    df.to_parquet(path, index=False)

    print("Added NIFTY 2024-03-02")
    print(f"Close: {target_close:.2f}")

print(f"Rows: {len(df):,}")

print(
    df[
        df["trade_date"].between(
            "2024-02-29",
            "2024-03-04",
        )
    ].to_string(index=False)
)

print("STATUS : PASSED")
PY
```

Then re-run:

- `python -m src.features.day1.build_nifty_market_reference`
- `python -m src.preprocessing.day1.build_nse_trading_calendar`
- `python -m src.targets.day1.calendar_aware_day1_target_generator`
- `python -m src.features.day1.technical_features_xgboost_v2`

Coverage check command:

```bash
python - <<'PY'
from pathlib import Path
import pandas as pd

feature_dir = Path("data/features/day1/technical_v2")
market_file = Path(
    "data/processed/index/nifty50_market_features.parquet"
)

market = pd.read_parquet(market_file)
market_dates = set(
    pd.to_datetime(market["trade_date"])
)

missing_rows = []

for path in sorted(feature_dir.glob("*_features.parquet")):
    df = pd.read_parquet(path)
    df["trade_date"] = pd.to_datetime(df["trade_date"])

    missing = df.loc[
        ~df["trade_date"].isin(market_dates),
        ["symbol", "trade_date"]
    ]

    if not missing.empty:
        missing_rows.append(missing)

if missing_rows:
    result = pd.concat(missing_rows, ignore_index=True)
else:
    result = pd.DataFrame(columns=["symbol", "trade_date"])

print("=" * 80)
print("NIFTY MARKET DATE COVERAGE")
print("=" * 80)
print(f"Missing market-date rows : {len(result):,}")

if not result.empty:
    print(result.to_string(index=False))

print()
print("STATUS :", "PASSED" if result.empty else "FAILED")
PY
```

### Full Rebuild Sequence (recorded)

- `python -m src.ingestion.download_nse_reference_data`
- `python -m src.preprocessing.build_canonical_price_dataset`
- `python -m src.features.day1.build_nifty_market_reference`
- `python -m src.preprocessing.day1.build_nse_trading_calendar`
- `python -m src.targets.day1.calendar_aware_day1_target_generator`
- `python -m src.features.day1.technical_features_xgboost_v2`

### Remaining Missing Market Feature Rows Analysis

```bash
python - <<'PY'
from pathlib import Path
import pandas as pd

feature_dir = Path("data/features/day1/technical_v2")

rows = []

for path in sorted(feature_dir.glob("*_features.parquet")):
    df = pd.read_parquet(path)

    market_cols = [
        "nifty50_return_1d",
        "nifty50_return_3d",
        "nifty50_return_5d",
        "nifty50_return_20d",
        "nifty50_volatility_5d",
        "nifty50_volatility_10d",
        "nifty50_volatility_20d",
    ]

    missing = df[market_cols].isna().any(axis=1)

    if missing.any():
        tmp = df.loc[missing, ["symbol", "trade_date"]].copy()
        rows.append(tmp)

out = pd.concat(rows, ignore_index=True)

print("\nMarket-missing rows:", len(out))
print("\nDates:")
print(out.groupby("trade_date").size().to_string())

print("\nSymbols:")
print(out.groupby("symbol").size().to_string())

print("\nSample:")
print(out.sort_values(["trade_date", "symbol"]).to_string(index=False))
PY
```

Then executed:

- `python -m src.preprocessing.day1.dataset_validation_and_target`

### Target Mismatch Validation Command

```bash
python - <<'PY'
from pathlib import Path
import pandas as pd
import numpy as np

TARGET_DIR = Path("data/targets/day1")
CALENDAR_FILE = Path("data/processed/calendar/nse_trading_calendar.parquet")

cal = pd.read_parquet(CALENDAR_FILE)
cal["trade_date"] = pd.to_datetime(cal["trade_date"], errors="coerce")

if "is_trading_day" in cal.columns:
    cal = cal[cal["is_trading_day"].astype(bool)]

dates = (
    cal["trade_date"]
    .dropna()
    .drop_duplicates()
    .sort_values()
    .reset_index(drop=True)
)

calendar_next = pd.Series(
    dates.values[1:],
    index=dates.values[:-1],
)

total_valid = 0
total_mismatch = 0
total_return_mismatch = 0
total_missing_ok = 0

for path in sorted(TARGET_DIR.glob("*.parquet")):
    df = pd.read_parquet(path)

    df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce")

    valid = df["target_return_1d"].notna()

    total_valid += int(valid.sum())

    expected_next = df.loc[valid, "trade_date"].map(calendar_next)

    actual_next = pd.to_datetime(
        df.loc[valid, "next_trade_date"],
        errors="coerce",
    )

    mismatch = (
        expected_next.reset_index(drop=True)
        != actual_next.reset_index(drop=True)
    )

    total_mismatch += int(mismatch.sum())

    expected_return = (
        pd.to_numeric(
            df.loc[valid, "future_close_1d"],
            errors="coerce",
        )
        /
        pd.to_numeric(
            df.loc[valid, "close"],
            errors="coerce",
        )
        - 1.0
    )

    actual_return = pd.to_numeric(
        df.loc[valid, "target_return_1d"],
        errors="coerce",
    )

    return_mismatch = ~np.isclose(
        actual_return.to_numpy(dtype=float),
        expected_return.to_numpy(dtype=float),
        rtol=1e-8,
        atol=1e-10,
    )

    total_return_mismatch += int(return_mismatch.sum())

    missing = df["target_return_1d"].isna()
    missing_ok = df.loc[missing, "future_close_1d"].isna()

    total_missing_ok += int(missing_ok.sum())

print("=" * 80)
print("CALENDAR-AWARE T+1 TARGET VALIDATION")
print("=" * 80)
print(f"Valid target rows              : {total_valid:,}")
print(f"Calendar successor mismatches  : {total_mismatch:,}")
print(f"Target return mismatches       : {total_return_mismatch:,}")
print(f"Missing-target rows valid      : {total_missing_ok:,}")

if total_mismatch == 0 and total_return_mismatch == 0:
    print("STATUS                         : PASSED")
else:
    print("STATUS                         : FAILED")
PY
```

Observed note: `99` valid missing row(s), that are correct as marked NA flagged .

### Baseline Training Executed

- `python -m src.models.day1.baseline_train`
- `python -m src.models.day1.train_xgboost_baseline`

### Baseline Result Summary (Step 2.2)

Result was weak.

```text
==========================================================================================
DAY-1 XGBOOST BASELINE
STEP 2.2 — CLEAN REPRODUCIBLE BASELINE TRAINING
==========================================================================================

DATA
------------------------------------------------------------------------------------------
Train rows      : 78,194
Validation rows : 17,283
E0 feature count: 90
Config          : config/day1.yaml

REGRESSION — VALIDATION
------------------------------------------------------------------------------------------
rmse                    : 0.03887825430955638
mae                     : 0.012693415597189879
r2                      : -0.04514842303027766
pearson_correlation     : 0.0167923331383685

CLASSIFICATION — VALIDATION
------------------------------------------------------------------------------------------
accuracy                : 0.5194121390962217
precision               : 0.5262255396409118
recall                  : 0.5910944935418083
f1                      : 0.556776947705443
roc_auc                 : 0.5171303471707596
pr_auc                  : 0.525847494726366
log_loss                : 0.7072045803070068
threshold               : 0.5

RANKING — VALIDATION
------------------------------------------------------------------------------------------
Daily Spearman mean     : 0.006212707696238234
Days evaluated          : 186
Top-K average returns   : {'5': 0.0014837136119793157, '10': 0.0011636921495492213, '20': 0.0021821631930360346}
Top-K hit rates         : {'5': 0.5494623655913979, '10': 0.5274193548387096, '20': 0.5123655913978494}

MISSING-VALUE POLICY
------------------------------------------------------------------------------------------
XGBoost native missing-value handling. No imputation or scaling is used.

BASELINE BOUNDARY
------------------------------------------------------------------------------------------
No hyperparameter tuning, calibration, test evaluation, or test-driven
feature decisions are performed in Step 2.2.
```

### Next Actions (Recorded)

- Executed feature experiment command:
    - `python -m src.models.day1.run_feature_ablation_experiments`
- Feature freeze decision: not finalized.
- Created tuning config/script plan:
    - `config/day1_tuning.yaml`
    - `tune_xgboost_hyperparameters.py`

Comparison objective (recorded): compare E0 and TUNE_012 on matching stocks/dates using regression errors, daily Spearman, Top-5/10/20 returns, and hit rates.

```bash
python - <<'PY'
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from src.models.day1 import train_xgboost_baseline as baseline

e0_path = Path(baseline.VALIDATION_PREDICTIONS_FILE)
tuned_path = Path(
    "data/evaluation/day1/hyperparameter_tuning/"
    "selected_candidate_validation_predictions.parquet"
)

target_col = baseline.TARGET_RETURN_COLUMN
keys = ["symbol", "trade_date"]
prediction_col = "predicted_return_1d"

for path in [e0_path, tuned_path]:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")

e0 = pd.read_parquet(e0_path)
tuned = pd.read_parquet(tuned_path)

required = set(keys + [target_col, prediction_col])

for name, frame in [("E0", e0), ("TUNE_012", tuned)]:
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(
            f"{name} predictions are missing columns: {sorted(missing)}"
        )

    frame["trade_date"] = pd.to_datetime(
        frame["trade_date"], errors="raise"
    ).dt.normalize()

    if frame.duplicated(keys).any():
        raise ValueError(
            f"{name} has duplicate symbol/date observations."
        )

print("\n1. INPUT FILES")
print("==============")
print("E0:", e0_path)
print("TUNE_012:", tuned_path)
print(f"E0 rows: {len(e0):,}")
print(f"TUNE_012 rows: {len(tuned):,}")

e0_keys = set(
    e0[keys].itertuples(index=False, name=None)
)
tuned_keys = set(
    tuned[keys].itertuples(index=False, name=None)
)

print("\n2. VALIDATION ALIGNMENT")
print("=======================")
print("E0 unique stock/date pairs:", len(e0_keys))
print("TUNE_012 unique stock/date pairs:", len(tuned_keys))
print("E0 pairs missing from TUNE_012:", len(e0_keys - tuned_keys))
print("TUNE_012 pairs missing from E0:", len(tuned_keys - e0_keys))

if e0_keys != tuned_keys:
    print("\nComparison stopped: prediction files do not contain")
    print("the same stock/date observations. Do not compare their")
    print("metrics until the difference has been investigated.")
    raise SystemExit(1)

e0 = e0[
    keys + [target_col, prediction_col]
].rename(columns={
    target_col: "actual_e0",
    prediction_col: "e0_prediction",
})

tuned = tuned[
    keys + [target_col, prediction_col]
].rename(columns={
    target_col: "actual_tuned",
    prediction_col: "tuned_prediction",
})

data = e0.merge(
    tuned,
    on=keys,
    how="inner",
    validate="one_to_one",
)

actual_e0 = pd.to_numeric(data["actual_e0"], errors="coerce")
actual_tuned = pd.to_numeric(data["actual_tuned"], errors="coerce")

if actual_e0.isna().any() or actual_tuned.isna().any():
    raise ValueError("Actual returns contain missing or invalid values.")

if not np.allclose(
    actual_e0.to_numpy(),
    actual_tuned.to_numpy(),
    rtol=0,
    atol=1e-12,
):
    raise ValueError(
        "Actual target returns differ between the two prediction files."
    )

data["actual_return"] = actual_e0
for col in ["e0_prediction", "tuned_prediction"]:
    data[col] = pd.to_numeric(data[col], errors="coerce")

data = data.replace([np.inf, -np.inf], np.nan).dropna(
    subset=["actual_return", "e0_prediction", "tuned_prediction"]
)

print("Matched rows:", f"{len(data):,}")
print("Matched trading dates:", data["trade_date"].nunique())
print("Stocks represented:", data["symbol"].nunique())

y = data["actual_return"].to_numpy()

def regression_metrics(pred):
    pred = data[pred].to_numpy()
    return {
        "RMSE": np.sqrt(mean_squared_error(y, pred)),
        "MAE": mean_absolute_error(y, pred),
        "R2": r2_score(y, pred),
    }

print("\n3. REGRESSION METRICS ON MATCHED ROWS")
print("=====================================")
reg = {
    "E0": regression_metrics("e0_prediction"),
    "TUNE_012": regression_metrics("tuned_prediction"),
}

for metric in ["RMSE", "MAE", "R2"]:
    print(
        f"{metric:8s} "
        f"E0={reg['E0'][metric]:.6f}  "
        f"TUNE_012={reg['TUNE_012'][metric]:.6f}"
    )

daily_rows = []

for date, group in data.groupby("trade_date", sort=True):
    actual = group["actual_return"]

    row = {"trade_date": date}

    predictions_vary = {
        "E0": group["e0_prediction"].nunique() > 1,
        "TUNE_012": group["tuned_prediction"].nunique() > 1,
    }

    # Spearman is compared only on dates eligible for both models.
    if actual.nunique() > 1 and all(predictions_vary.values()):
        row["spearman_e0"] = group["e0_prediction"].corr(
            actual, method="spearman"
        )
        row["spearman_tuned"] = group["tuned_prediction"].corr(
            actual, method="spearman"
        )

    # Top-K is compared on the same date whenever both models
    # produce varying predictions and at least K stocks are present.
    for k in [5, 10, 20]:
        if len(group) >= k and all(predictions_vary.values()):
            for label, col in [
                ("e0", "e0_prediction"),
                ("tuned", "tuned_prediction"),
            ]:
                ranked = group.sort_values(
                    [col, "symbol"],
                    ascending=[False, True],
                    kind="mergesort",
                ).head(k)

                row[f"{label}_top{k}_return"] = (
                    ranked["actual_return"].mean()
                )
                row[f"{label}_top{k}_hit_rate"] = (
                    ranked["actual_return"].gt(0).mean()
                )

    daily_rows.append(row)

daily = pd.DataFrame(daily_rows)

def compare_daily(metric, e0_col, tuned_col):
    paired = daily[[e0_col, tuned_col]].dropna()
    if paired.empty:
        print(f"{metric}: no dates eligible for both models")
        return

    e0_values = paired[e0_col]
    tuned_values = paired[tuned_col]
    differences = tuned_values - e0_values

    print(f"\n{metric}")
    print("  Paired dates:", len(paired))
    print(f"  E0 mean:      {e0_values.mean():.6f}")
    print(f"  TUNE_012 mean:{tuned_values.mean():.6f}")
    print(f"  Difference:   {differences.mean():+.6f}")
    print(
        "  TUNE_012 better on:",
        f"{int((differences > 0).sum())}/{len(paired)} dates",
    )

print("\n4. PAIRED DAILY RANKING COMPARISON")
print("==================================")

compare_daily(
    "Daily Spearman",
    "spearman_e0",
    "spearman_tuned",
)

for k in [5, 10, 20]:
    compare_daily(
        f"Top-{k} average realized return",
        f"e0_top{k}_return",
        f"tuned_top{k}_return",
    )
    compare_daily(
        f"Top-{k} hit rate",
        f"e0_top{k}_hit_rate",
        f"tuned_top{k}_hit_rate",
    )

print("\nNo project files were written or modified.")
print("Final test dataset was not loaded.")
PY
```

Next note: paired 95% confidence intervals for daily metric differences.

Then changed:

- `tune_xgboost_hyperparameters.py`
- `config/day1_tuning.yaml`

Then:

```bash
python - <<'PY'
from pathlib import Path
import json
import pandas as pd
import yaml

cfg = yaml.safe_load(
    Path("config/day1_tuning.yaml").read_text()
)
out = Path(cfg["outputs"]["directory"])

print("OUTPUT DIRECTORY:", out.resolve())

# 1. Verify candidate coverage and selection scores.
trials_path = out / cfg["outputs"]["trials_csv"]
trials = pd.read_csv(trials_path)

columns = [
    "candidate_id",
    "daily_spearman_selection_score",
    "selection_days_evaluated",
    "constant_prediction_dates",
    "daily_spearman_mean",
    "top_5_average_realized_return",
    "rmse",
    "mae",
]

columns = [c for c in columns if c in trials.columns]

print("\nALL CANDIDATES — UPDATED SCORING")
print(trials[columns].to_string(index=False))

if "selection_days_evaluated" in trials.columns:
    print("\nSELECTION-DAY COVERAGE")
    print(
        trials["selection_days_evaluated"]
        .value_counts(dropna=False)
        .sort_index()
        .to_string()
    )

# 2. Inspect the selected candidate.
for key in ["selected_params_json", "validation_metrics_json"]:
    path = out / cfg["outputs"][key]
    print(f"\n{key.upper()}")
    print(json.dumps(json.loads(path.read_text()), indent=2))

# 3. Display the updated report.
report_path = out / cfg["outputs"]["report_txt"]
print("\nUPDATED REPORT")
print(report_path.read_text())

print("\nTest dataset loaded: NO")
PY
```

Next review of existing feature experiments:

```bash
python - <<'PY'
from pathlib import Path
import pandas as pd

root = Path("data/evaluation/day1/validation_robustness_review")

for name in [
    "paired_comparisons.csv",
    "validation_robustness_report.txt",
]:
    path = root / name
    print(f"\n{'=' * 20} {name} {'=' * 20}")

    if not path.exists():
        print(f"File not found: {path}")
        continue

    if path.suffix == ".csv":
        df = pd.read_csv(path)
        print("Rows:", len(df))
        print("Columns:", list(df.columns))
        print(df.to_string(index=False))
    else:
        print(path.read_text())

print("\nRead-only inspection complete. Test data was not loaded.")
PY
```

