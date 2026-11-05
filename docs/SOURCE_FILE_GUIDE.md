# Source File Guide

This document explains **what each Python file in `src/` does**, including its main logic and role in the pipeline.

---

## 1) Package Structure

- `src/`
  - `ingestion/`: download + normalize raw NSE data
  - `features/`: technical feature engineering
  - `targets/`: day-specific target generation
  - `preprocessing/`: build/inspect modeling datasets
  - `models/`: split logic + day5 model training
  - `evaluation/`: day5 diagnostics and ranking evaluation
  - `utils/`: config/path utilities
  - `backtesting/`, `calibration/`, `ranking/`, `validation/`: reserved package areas

---

## 2) Common Utility and Base Packages

### `src/__init__.py`
- Purpose: marks `src` as a Python package root for module imports.

### `src/utils/__init__.py`
- Purpose: package marker for shared utilities.

### `src/utils/mvp_config.py`
- Purpose: centralized loading of `config/mvp.yaml` and path resolution.
- Core logic:
  - resolves project root
  - reads YAML once (cached)
  - exposes path helpers like `modeling_dir()`, `features_technical_dir(day)`, `targets_day_dir(day)`, `models_dir()`, `evaluation_dir()`, `predictions_dir()`.
- Why important: removes hardcoded paths and keeps day1/day5 scripts consistent.

### `src/backtesting/__init__.py`
- Purpose: package marker; backtesting modules can be added here.

### `src/calibration/__init__.py`
- Purpose: package marker; probability calibration modules can be added here.

### `src/ranking/__init__.py`
- Purpose: package marker; ranking-specific reusable logic can be added here.

### `src/validation/__init__.py`
- Purpose: package marker; data/model validation modules can be added here.

---

## 3) Ingestion Layer (`src/ingestion`)

### `src/ingestion/__init__.py`
- Purpose: package marker.

### `src/ingestion/download_nse_security_master.py`
- Purpose: downloads raw NSE equity master CSV (`EQUITY_L.csv`).
- Core logic:
  - calls NSE archive endpoint with browser-like headers
  - writes raw CSV to `data/raw/nse/EQUITY_L.csv`.

### `src/ingestion/nifty100_download.py`
- Purpose: downloads official NIFTY100 constituent CSV.
- Core logic:
  - fetches `ind_nifty100list.csv`
  - stores it under `data/raw/nse/universe/NIFTY100.csv`.

### `src/ingestion/nse_security_master.py`
- Purpose: normalizes raw NSE security master into cleaned parquet.
- Core logic:
  - validates expected columns
  - keeps only `EQ` series
  - standardizes text/date/numeric fields
  - renames columns to application-friendly names
  - validates duplicates/nulls
  - outputs `data/processed/stock_master/nse_equity_stock_master.parquet`.

### `src/ingestion/universe_builder.py`
- Purpose: builds validated NIFTY100 universe parquet.
- Core logic:
  - loads normalized stock master and official NIFTY100 list
  - validates schema/record count/duplicates
  - joins by ISIN (preferred over names)
  - verifies full matches and quality checks
  - writes processed universe dataset.

### `src/ingestion/nse_index_history.py`
- Purpose: downloads and parses historical NIFTY index data in chunked windows.
- Core logic:
  - creates NSE session/cookies
  - splits long date range into <=365-day chunks
  - retries failed requests with backoff
  - validates JSON response structure
  - normalizes index fields and dtypes
  - writes raw chunk JSON + processed index parquet.

### `src/ingestion/nse_price_history.py`
- Purpose: downloads historical per-symbol stock CSVs for universe symbols.
- Core logic:
  - loads 100-symbol universe
  - requests symbol CSV with retry/backoff
  - validates CSV content and required columns
  - handles NSE historical alias symbols by filtering to requested symbol
  - stores raw CSVs for later parsing.

### `src/ingestion/nse_price_parser.py`
- Purpose: parses raw NSE stock CSVs into normalized per-symbol parquet.
- Core logic:
  - validates expected NSE CSV schema
  - renames columns to canonical names
  - handles multi-symbol historical alias rows
  - parses numerics/dates, checks quality
  - writes one parquet per symbol in processed prices.

### `src/ingestion/test_nse_historical_api.py`
- Purpose: manual API smoke-test script for one symbol/date range.
- Core logic:
  - initializes NSE session
  - calls historical endpoint
  - prints status/content preview
  - saves raw response for inspection.

### `src/ingestion/test_nse_price_history.py`
- Purpose: alternate/manual test script for NSE price endpoint behavior.
- Core logic:
  - sends test request with known params
  - prints response metadata/body sample
  - saves response payload for debugging.

---

## 4) Feature Engineering (`src/features`)

### `src/features/__init__.py`
- Purpose: package marker.

### `src/features/day1/technical_features_xgboost.py`
- Purpose: baseline day1 technical feature generator for XGBoost pipelines.
- Core logic:
  - loads per-symbol processed prices and index data
  - computes returns, moving averages, EMA, RSI, ROC, volatility, ATR, volume features
  - computes benchmark-relative features (e.g., stock vs NIFTY, beta-style features)
  - performs data checks and exports per-symbol feature parquet files to `data/features/technical`.

### `src/features/day1/technical_features_xgboost_v2.py`
- Purpose: expanded day1 feature generator with baseline + extended feature families.
- Core logic:
  - keeps baseline 24 XGBoost features and adds additional price action/momentum/trend/volatility/volume/market regime features
  - computes date-wise cross-sectional rank features (e.g., return ranks, relative strength rank)
  - optionally joins sector mapping for sector-relative features when `data/reference/stock_sector_mapping.parquet` exists
  - writes one parquet per symbol to `data/features/technical_v2` with `feature_version = technical_v2`.

---

## 5) Targets (`src/targets`)

### `src/targets/__init__.py`
- Purpose: package marker.

### `src/targets/day1/__init__.py`
- Purpose: package marker for day1 targets.

### `src/targets/day1/return_targets.py`
- Purpose: creates day1 future return and direction targets.
- Core logic:
  - for each symbol: computes `future_close_1d`, `target_return_1d`, `target_direction_1d`
  - validates schema/duplicates/empties
  - keeps final no-future rows as NA targets
  - writes one parquet per symbol to day1 target folder.

### `src/targets/day1_return_targets.py`
- Purpose: compatibility wrapper entrypoint.
- Core logic: imports and runs `src.targets.day1.return_targets.main()`.

### `src/targets/day5/__init__.py`
- Purpose: package marker for day5 targets.

### `src/targets/day5/return_targets.py`
- Purpose: creates day5 return, direction, and excess-return targets.
- Core logic:
  - computes per-symbol `target_return_5d`, `target_direction_5d`
  - loads NIFTY 5-day return and merges by `trade_date`
  - computes `target_excess_return_5d = stock_return_5d - nifty50_return_5d`
  - keeps tail rows without forward data as NA targets
  - writes one parquet per symbol to day5 target folder.

### `src/targets/day5_return_targets.py`
- Purpose: compatibility wrapper entrypoint.
- Core logic: imports and runs `src.targets.day5.return_targets.main()`.

---

## 6) Preprocessing / Modeling Dataset Build (`src/preprocessing`)

### `src/preprocessing/__init__.py`
- Purpose: package marker.

### `src/preprocessing/day1/__init__.py`
- Purpose: package marker for day1 preprocessing.

### `src/preprocessing/day1/build_model_dataset.py`
- Purpose: builds combined day1 modeling dataset from features + day1 targets.
- Core logic:
  - loads each symbol feature file and matching day1 target file
  - validates required columns, symbol/date uniqueness, one-symbol-per-file
  - merges on `(symbol, trade_date)`
  - concatenates all symbols
  - validates numeric infinities
  - writes day1 modeling parquet.

### `src/preprocessing/day1/inspect_model_dataset.py`
- Purpose: audits day1 modeling dataset quality and feature candidate set.
- Core logic:
  - checks required columns/targets
  - reports dataset shape/date range/stocks
  - identifies non-feature columns vs candidate feature columns
  - runs leakage and missingness diagnostics.

### `src/preprocessing/day1/dataset_validation_and_target.py`
- Purpose: integrated day1 validation pipeline that verifies feature-target integrity and creates final modeling datasets.
- Core logic:
  - loads generated day1 features and day1 targets, then validates schema/symbol/date alignment
  - validates target semantics (`target_return_1d`, `target_direction_1d`) and direction consistency
  - checks duplicates, missing target rows, finite numeric values, leakage-risk structure, and feature distributions
  - builds final day1 modeling dataset and chronological train/validation/test splits
  - writes validation artifacts including a detailed phase report and distribution outputs.

### `src/preprocessing/day5/__init__.py`
- Purpose: package marker for day5 preprocessing.

### `src/preprocessing/day5/build_model_dataset.py`
- Purpose: builds combined day5 modeling dataset from features + day5 targets.
- Core logic:
  - similar to day1 build, with additional `target_excess_return_5d`
  - validates merge consistency and numeric quality
  - writes day5 modeling parquet.

### `src/preprocessing/day5/inspect_model_dataset.py`
- Purpose: audits day5 modeling dataset structure, target presence, leakage/missingness.
- Core logic:
  - checks required target columns (`return`, `direction`, `excess_return`)
  - prints all columns and candidate features
  - validates excluded/non-feature fields and NaN distribution.

---

## 7) Models (`src/models`)

### `src/models/__init__.py`
- Purpose: package marker.

### `src/models/time_split.py`
- Purpose: canonical leakage-aware time split utility.
- Core logic:
  - defines fixed boundaries for train/validation/test
  - applies 5-trading-day purge near boundaries to avoid forward-label overlap leakage
  - validates no date overlap and non-empty splits
  - returns `(train_df, validation_df, test_df)`.

### `src/models/day5/__init__.py`
- Purpose: package marker for day5 trainers.

### `src/models/day5/xgboost_trainer.py`
- Purpose: trains day5 direction classifier (`target_direction_5d`).
- Core logic:
  - loads day5 modeling dataset
  - applies `create_time_split`
  - prepares feature matrix/target
  - trains `XGBClassifier`
  - reports classification metrics (accuracy/ROC-AUC/PR-AUC/logloss/brier)
  - saves model + metadata.

### `src/models/day5/xgboost_return_trainer.py`
- Purpose: trains day5 absolute-return regressor (`target_return_5d`).
- Core logic:
  - split + feature prep
  - trains `XGBRegressor`
  - reports MAE/RMSE/R²/correlation
  - prints feature importances
  - saves model + metadata.

### `src/models/day5/xgboost_excess_return_trainer.py`
- Purpose: trains day5 excess-return regressor (`target_excess_return_5d`).
- Core logic:
  - mirrors return trainer structure with excess-return target
  - keeps feature set parity with absolute-return experiment
  - saves model + metadata for downstream ranking diagnostics.

### `src/models/day5/lstm_excess_return_trainer.py`
- Purpose: trains sequence-based LSTM model for day5 excess return.
- Core logic:
  - loads day5 modeling dataset
  - identifies columns/features and cleans usable data
  - scales using train-only fit
  - constructs rolling lookback sequences
  - trains PyTorch LSTM with early stopping
  - evaluates regression and ranking-related metrics
  - saves `.pt` model and test predictions parquet.

---

## 8) Evaluation (`src/evaluation`)

### `src/evaluation/__init__.py`
- Purpose: package marker.

### `src/evaluation/day5/__init__.py`
- Purpose: package marker for day5 diagnostics.

### `src/evaluation/day5/baseline_diagnostics.py`
- Purpose: diagnostics for trained day5 direction model on split datasets.
- Core logic:
  - loads model + dataset
  - computes probability metrics by split
  - analyzes yearly performance, probability buckets, cross-sectional ranking quality
  - prints feature importance.

### `src/evaluation/day5/data_quality_diagnostics.py`
- Purpose: deep quality audit for day5 modeling dataset.
- Core logic:
  - checks schema, duplicates, missingness, infinite values, stock/date coverage
  - summarizes target distributions and feature behavior
  - writes multiple CSV diagnostics into evaluation folder.

### `src/evaluation/day5/return_ranking_diagnostics.py`
- Purpose: evaluates cross-sectional ranking quality for day5 absolute-return model.
- Core logic:
  - predicts on test split
  - computes daily Spearman rank correlation
  - evaluates Top-5/10/20 actual returns and hit rates vs universe
  - writes ranking diagnostics parquet.

### `src/evaluation/day5/excess_return_baseline_comparison.py`
- Purpose: compares day5 excess-return XGBoost model vs non-ML baselines.
- Core logic:
  - evaluates model against zero, historical mean, and daily mean baselines
  - uses same test rows for fair comparison
  - computes regression metrics and persists comparison outputs.

### `src/evaluation/day5/excess_return_ranking_diagnostics.py`
- Purpose: cross-sectional ranking diagnostics for excess-return model.
- Core logic:
  - predicts excess returns on test period
  - computes daily rank correlation and Top-N portfolio statistics
  - reports spread/hit-rate behavior
  - writes diagnostics parquet.

---

## 9) Notes on Wrappers and Day Separation

- Day-specific logic is implemented under day folders (`day1`, `day5`) instead of merged scripts.
- Wrapper files (`day1_return_targets.py`, `day5_return_targets.py`) are retained for backward-compatible command usage.
- Core path resolution is centralized in `src/utils/mvp_config.py` and backed by `config/mvp.yaml`.

---

## 10) Quick Start Order (Recommended)

1. Run ingestion scripts to populate raw + processed market data.
2. Run technical feature generation.
3. Run day1/day5 target generators.
4. Build day1/day5 modeling datasets.
5. Train day5 models.
6. Run day5 diagnostics and ranking evaluations.
