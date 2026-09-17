# AI-Based Multi-Stock Market Prediction & Ranking System

## MVP

Market:
- NSE

Universe:
- NIFTY 500

Frequency:
- Daily

Primary prediction horizon:
- 5 trading days

Primary model:
- XGBoost

Primary objective:
- Predict 5-day stock return and probability of positive return.
- Rank stocks cross-sectionally.
- Evaluate predictions using leakage-controlled walk-forward backtesting.

## Architecture

Data Sources
→ Validation
→ Preprocessing
→ Feature Engineering
→ XGBoost
→ Probability Calibration
→ Ranking
→ Backtesting
→ Evaluation
→ Dashboard

## Storage

PostgreSQL:
- Stock metadata
- Trading calendar
- Corporate actions
- Model registry
- Pipeline runs
- Predictions
- Experiments
- Backtest summaries

Parquet:
- Historical prices
- Features
- Targets
- Large analytical datasets

## Current Status

Environment setup: Complete
PostgreSQL setup: Complete
Database schema: Complete
Project structure: Complete
Git setup: Complete

Next:
NSE historical market data ingestion.
