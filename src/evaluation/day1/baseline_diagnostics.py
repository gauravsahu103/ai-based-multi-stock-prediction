"""Phase 2.3 read-only Day-1 baseline diagnostics. Train + validation only.
Run from project root: python -m src.evaluation.day1.baseline_diagnostics
"""
from __future__ import annotations
import json
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from src.utils.mvp_config import evaluation_dir, modeling_dir, storage_path

H = 1
RET, DIR = f"target_return_{H}d", f"target_direction_{H}d"
TRAIN = modeling_dir(H) / "train.parquet"
VAL = modeling_dir(H) / "validation.parquet"
PRED = storage_path("predictions") / "day1" / "validation_predictions.parquet"
INV = evaluation_dir(H) / "feature_inventory.json"
IMP = evaluation_dir(H) / "feature_importance.parquet"
OUT = evaluation_dir(H) / "baseline_diagnostics"


def load_parquet(path, label):
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    df = pd.read_parquet(path)
    if df.empty:
        raise ValueError(f"{label} is empty: {path}")
    return df


def corr(a, b, method="pearson"):
    pair = pd.concat([a, b], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    if len(pair) < 2 or pair.iloc[:, 0].nunique() < 2 or pair.iloc[:, 1].nunique() < 2:
        return None
    v = pair.iloc[:, 0].corr(pair.iloc[:, 1], method=method)
    return float(v) if pd.notna(v) else None


def summary(s):
    s = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if s.empty:
        return {"count": 0}
    return {"count": int(s.count()), "mean": float(s.mean()), "std": float(s.std()),
            "min": float(s.min()), "p01": float(s.quantile(.01)),
            "p05": float(s.quantile(.05)), "median": float(s.median()),
            "p95": float(s.quantile(.95)), "p99": float(s.quantile(.99)), "max": float(s.max())}


def main():
    print("=" * 84, "\nPHASE 2.3 — BASELINE DIAGNOSTICS (TRAIN + VALIDATION ONLY)\n", "=" * 84, sep="")
    if not INV.exists():
        raise FileNotFoundError(f"Feature inventory not found: {INV}")
    inventory = json.loads(INV.read_text(encoding="utf-8"))
    features = inventory.get("feature_list")
    if inventory.get("feature_set") != "E0" or not isinstance(features, list) or not features:
        raise ValueError("Expected feature_set=E0 and non-empty feature_list.")

    train = load_parquet(TRAIN, "Train dataset")
    val = load_parquet(VAL, "Validation dataset")
    pred = load_parquet(PRED, "Validation predictions")
    train["trade_date"] = pd.to_datetime(train["trade_date"], errors="raise")
    val["trade_date"] = pd.to_datetime(val["trade_date"], errors="raise")
    pred["trade_date"] = pd.to_datetime(pred["trade_date"], errors="raise")

    for name, df in (("train", train), ("validation", val)):
        missing = sorted(set(features) - set(df.columns))
        if missing:
            raise ValueError(f"{name} missing E0 features: {missing}")
        needed = {"symbol", "trade_date", RET, DIR}
        if needed - set(df.columns):
            raise ValueError(f"{name} missing columns: {sorted(needed - set(df.columns))}")
        if df.duplicated(["symbol", "trade_date"]).any():
            raise ValueError(f"{name} has duplicate symbol/trade_date rows")
        if np.isinf(df[features].to_numpy(dtype=float)).any():
            raise ValueError(f"{name} E0 contains infinity")

    pred_required = {"symbol", "trade_date", RET, DIR, "predicted_return_1d", "predicted_probability_1d"}
    if pred_required - set(pred.columns):
        raise ValueError(f"Predictions missing columns: {sorted(pred_required - set(pred.columns))}")
    if pred.duplicated(["symbol", "trade_date"]).any():
        raise ValueError("Predictions have duplicate symbol/trade_date rows")
    key_check = val[["symbol", "trade_date"]].merge(
        pred[["symbol", "trade_date"]], on=["symbol", "trade_date"], how="outer", indicator=True)
    mismatch_count = int((key_check["_merge"] != "both").sum())
    if mismatch_count:
        raise ValueError(f"Validation/prediction keys differ; mismatch rows={mismatch_count}")

    OUT.mkdir(parents=True, exist_ok=True)
    pred[RET] = pd.to_numeric(pred[RET], errors="raise")
    pred[DIR] = pd.to_numeric(pred[DIR], errors="raise").astype(int)
    pred["predicted_return_1d"] = pd.to_numeric(pred["predicted_return_1d"], errors="raise")
    pred["predicted_probability_1d"] = pd.to_numeric(pred["predicted_probability_1d"], errors="raise")
    pred["regression_error"] = pred["predicted_return_1d"] - pred[RET]
    pred["absolute_error"] = pred["regression_error"].abs()
    pred["predicted_direction_at_0_5"] = (pred["predicted_probability_1d"] >= .5).astype(int)
    pred["direction_correct_at_0_5"] = pred["predicted_direction_at_0_5"] == pred[DIR]

    stock_rows = []
    for symbol, g in pred.groupby("symbol", sort=True):
        stock_rows.append({"symbol": symbol, "rows": len(g),
            "actual_return_mean": float(g[RET].mean()),
            "mae": float(g["absolute_error"].mean()),
            "rmse": float(np.sqrt(np.mean(g["regression_error"] ** 2))),
            "return_pearson": corr(g["predicted_return_1d"], g[RET]),
            "return_spearman": corr(g["predicted_return_1d"], g[RET], "spearman"),
            "direction_accuracy_at_0_5": float(g["direction_correct_at_0_5"].mean()),
            "actual_positive_rate": float(g[DIR].mean()),
            "mean_predicted_probability": float(g["predicted_probability_1d"].mean())})
    per_stock = pd.DataFrame(stock_rows).sort_values("mae", ascending=False)

    date_rows = []
    for date, g in pred.groupby("trade_date", sort=True):
        row = {"trade_date": date, "stock_count": len(g),
            "daily_spearman": corr(g["predicted_return_1d"], g[RET], "spearman"),
            "daily_pearson": corr(g["predicted_return_1d"], g[RET]),
            "actual_return_mean": float(g[RET].mean()),
            "predicted_return_mean": float(g["predicted_return_1d"].mean()),
            "predicted_return_std": float(g["predicted_return_1d"].std()),
            "predicted_return_spread": float(g["predicted_return_1d"].max() - g["predicted_return_1d"].min()),
            "probability_mean": float(g["predicted_probability_1d"].mean()),
            "probability_std": float(g["predicted_probability_1d"].std())}
        ranked = g.sort_values("predicted_return_1d", ascending=False)
        for k in (5, 10, 20):
            top = ranked.head(k) if len(ranked) >= k else None
            row[f"top_{k}_mean_return"] = float(top[RET].mean()) if top is not None else None
            row[f"top_{k}_hit_rate"] = float((top[RET] > 0).mean()) if top is not None else None
        date_rows.append(row)
    per_date = pd.DataFrame(date_rows)

    missing_rows = []
    for f in features:
        missing_rows.append({"feature": f, "train_missing_count": int(train[f].isna().sum()),
            "train_missing_pct": float(train[f].isna().mean()),
            "validation_missing_count": int(val[f].isna().sum()),
            "validation_missing_pct": float(val[f].isna().mean()),
            "train_unique_non_null": int(train[f].nunique(dropna=True)),
            "validation_unique_non_null": int(val[f].nunique(dropna=True))})
    missingness = pd.DataFrame(missing_rows).sort_values(
        ["validation_missing_pct", "train_missing_pct", "feature"], ascending=[False, False, True])

    distribution = {
        "actual_return": summary(pred[RET]),
        "predicted_return": summary(pred["predicted_return_1d"]),
        "predicted_probability": summary(pred["predicted_probability_1d"]),
        "regression_error": summary(pred["regression_error"]),
        "absolute_error": summary(pred["absolute_error"]),
        "prediction_actual_pearson": corr(pred["predicted_return_1d"], pred[RET]),
        "prediction_actual_spearman": corr(pred["predicted_return_1d"], pred[RET], "spearman"),
        "actual_positive_rate": float(pred[DIR].mean()),
        "predicted_up_rate_at_0_5": float(pred["predicted_direction_at_0_5"].mean()),
        "direction_accuracy_at_0_5": float(pred["direction_correct_at_0_5"].mean()),
        "validation_rows": len(val), "prediction_rows": len(pred), "key_mismatches": mismatch_count,
        "test_loaded": False}

    pred.to_parquet(OUT / "validation_row_diagnostics.parquet", index=False)
    per_stock.to_csv(OUT / "per_stock_diagnostics.csv", index=False)
    per_date.to_csv(OUT / "per_date_diagnostics.csv", index=False)
    missingness.to_csv(OUT / "feature_missingness.csv", index=False)
    (OUT / "prediction_error_distribution.json").write_text(json.dumps(distribution, indent=2, allow_nan=False), encoding="utf-8")
    if IMP.exists():
        importance = pd.read_parquet(IMP)
        importance.to_parquet(OUT / "feature_importance_copy.parquet", index=False)
        if {"model", "importance_gain"}.issubset(importance.columns):
            importance.sort_values("importance_gain", ascending=False).groupby("model", group_keys=False).head(20).to_csv(
                OUT / "top_20_feature_importance.csv", index=False)

    lines = ["PHASE 2.3 — DAY-1 BASELINE DIAGNOSTICS", "=" * 70,
        "Train + validation only. Test partition not loaded.", "No model fitting, tuning, threshold optimization, or feature selection.",
        f"Train rows: {len(train):,}", f"Validation rows: {len(val):,}",
        f"Prediction rows: {len(pred):,}", f"E0 features: {len(features)}",
        f"Pearson: {distribution['prediction_actual_pearson']}",
        f"Spearman: {distribution['prediction_actual_spearman']}",
        f"Direction accuracy at fixed 0.5: {distribution['direction_accuracy_at_0_5']:.6f}",
        "", "Highest-MAE stocks:"]
    for _, r in per_stock.head(10).iterrows():
        lines.append(f"{r['symbol']}: MAE={r['mae']:.6f}; Pearson={r['return_pearson']}; rows={r['rows']}")
    lines.extend(["", "Lowest daily Spearman dates:"])
    for _, r in per_date.dropna(subset=["daily_spearman"]).sort_values("daily_spearman").head(10).iterrows():
        lines.append(f"{pd.Timestamp(r['trade_date']).date()}: Spearman={r['daily_spearman']:.6f}; stocks={r['stock_count']}")
    lines.extend(["", "Highest validation feature missingness:"])
    for _, r in missingness.head(15).iterrows():
        lines.append(f"{r['feature']}: train={r['train_missing_pct']:.2%}; validation={r['validation_missing_pct']:.2%}")
    (OUT / "baseline_diagnostics_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    metadata = {"phase": "day1", "step": "2.3", "feature_set": "E0", "feature_count": len(features),
        "train_rows": len(train), "validation_rows": len(val), "test_loaded": False,
        "model_fitted": False, "threshold_tuned": False, "feature_selection_performed": False,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(), "output_directory": str(OUT)}
    (OUT / "diagnostics_run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Train rows                : {len(train):,}")
    print(f"Validation rows           : {len(val):,}")
    print(f"Validation prediction rows: {len(pred):,}")
    print(f"E0 features               : {len(features)}")
    print(f"Output directory          : {OUT}")
    print(f"Report                    : {OUT / 'baseline_diagnostics_report.txt'}")
    print("TEST DATA LOADED          : NO")
    print("MODEL FITTED              : NO")
    print("STATUS                    : PASSED")

if __name__ == "__main__":
    main()
