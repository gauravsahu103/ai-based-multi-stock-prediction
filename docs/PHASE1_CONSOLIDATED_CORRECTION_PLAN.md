# Phase 1 — Consolidated Data Correction Plan

## Purpose

This is the correction plan after the read-only Phase 1 integrity audit and the first XGBoost diagnostic baseline.

The current XGBoost baseline remains diagnostic only. Do not tune the model until this correction pass is completed.

## Confirmed findings

### P0 — Mandatory corrections

1. Corporate-action / price-series discontinuities
   - Large one-day moves are present.
   - Examples include approximately -90%, -76%, and -50% moves.
   - These must be reconciled against corporate actions before technical features and T+1 targets are generated.

2. NIFTY-derived features are calculated on stock-specific rows
   - The current feature generator merges NIFTY close into each stock and then calculates NIFTY returns/rolling statistics.
   - The corrected pipeline must calculate the complete NIFTY market feature table once from the standalone NIFTY series, then merge it into stocks by date.

3. Stock identity / historical symbol aliases
   - The current NSE parser explicitly discards historical alias rows when a response contains more than one symbol.
   - This can create missing historical observations and can contribute to long T+1 calendar gaps.
   - The canonical price layer must preserve raw source symbols and resolve them to a stable security identity/current symbol using symbol-change history.

4. T+1 semantics
   - The current target generator uses the next available row for the stock.
   - The corrected Day-1 target must use the canonical exchange trading calendar.
   - If the stock has no observation on the next canonical trading date, the target should be missing for that row rather than silently jumping many calendar days forward.

### P1 — Mandatory validation/correction

5. E0 feature eligibility
   Exclude source-scale/absolute-price fields from the primary structured feature inventory, including:
   - prev_close
   - vwap
   - sma_20
   - sma_50
   - ema_20
   - deliverable_qty
   - traded_value
   - other absolute source-scale fields identified by the validator

6. Duplicate feature removal
   Detect exact duplicate feature columns before forming E0.
   Example:
   - market_relative_return_5d duplicates stock_vs_nifty50_5d
   - market_relative_return_20d duplicates stock_vs_nifty50_20d

7. Extreme feature validation
   Review all features with very large magnitude or invalid numeric values after the price-series correction.

8. Missingness
   Separate expected warm-up NaNs from unexpected data gaps.
   Report missingness by feature and by stock.

9. Universe coverage
   Explain every date with fewer than 100 available securities.
   Preserve the distinction between:
   - legitimate stock non-trading/suspension
   - missing source data
   - historical symbol issue
   - listing-history issue

10. Sector
   Sector features are optional. The current run has no sector column, so this is not a blocker for the technical branch.

11. Cross-sectional ranks
   Existing date-local rank calculation passed the current audit. Re-run after correcting the stock universe/identity layer.

12. Prediction-time contract
   EOD-T features such as close, intraday return, candle range, and close-position features are valid only because the MVP prediction cut-off is the end of trading session T. Freeze this contract in documentation/configuration.

13. Split drift
   Keep the generated train/validation/test distribution report. Use validation for model decisions; keep test untouched until the corrected baseline is frozen.

## Research limitation to document

The MVP uses the current NIFTY-100 universe snapshot for historical observations. It is not a historically reconstructed NIFTY-100 constituent history. This should remain explicitly documented as a research limitation rather than being silently changed during this correction pass.

## Corrected execution order

```text
Raw NSE price CSV
      |
      +--> Symbol-change / identity resolution
      |
      +--> Corporate-action reconciliation
      |
      v
Canonical price dataset
      |
      +--> Canonical NIFTY market reference table
      |
      v
Technical feature generation
      |
      v
Calendar-aware Day-1 target generation
      |
      v
Modeling dataset + validation
      |
      v
Phase 2.1 input validation
      |
      v
Phase 2.2 clean XGBoost baseline
```

## Exit criteria before rerunning XGBoost

- No unexplained split/bonus-style price discontinuities remain.
- NIFTY-derived features match an independently calculated NIFTY reference table.
- Historical symbol aliases are resolved rather than discarded.
- T+1 gaps are based on the canonical exchange calendar.
- No absolute/source-scale fields are silently admitted into E0.
- Exact duplicate features are removed or explicitly excluded.
- Feature missingness and extreme values are documented.
- Same-date universe coverage is understood.
- The regenerated Phase-1 report passes.
- Only then rerun the clean XGBoost baseline.
/Users/gauravsahu/Documents/ML_AI/StockPrediction/Project/
├── docs/
│   └── PHASE1_CONSOLIDATED_CORRECTION_PLAN.md
│
src/
├── ingestion/
│   └── download_nse_reference_data.py
│
├── preprocessing/
│   └── build_canonical_price_dataset.py
│
├── features/
│   └── build_nifty_market_reference.py
│
├── targets/
│   └── calendar_aware_day1_target_generator.py