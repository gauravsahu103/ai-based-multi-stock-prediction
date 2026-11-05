"""Day-1 stock prediction — validation robustness review.

Run from project root:
    python -m src.evaluation.day1.validation_robustness_review

Reads only the existing validation partition and saved E0/E1-E6 validation
predictions. It never opens train or test partition files and does not fit models.
It compares ranking strategies using paired moving-block bootstrap intervals over
trading dates and adds zero-return regression and 5-day-momentum ranking references.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score


ROOT = Path.cwd()
VALIDATION_FILE = ROOT / "data/processed/modeling/day1/validation.parquet"
E0_PREDICTIONS_FILE = ROOT / "data/predictions/day1/validation_predictions.parquet"
EXPERIMENT_PREDICTIONS_ROOT = ROOT / "data/predictions/day1/feature_experiments"
OUTPUT_DIR = ROOT / "data/evaluation/day1/validation_robustness_review"

KEY_COLUMNS = ["symbol", "trade_date"]
TARGET_RETURN = "target_return_1d"
TARGET_DIRECTION = "target_direction_1d"
PREDICTED_RETURN = "predicted_return_1d"
PREDICTED_PROBABILITY = "predicted_probability_1d"
MOMENTUM_SIGNAL = "return_5d"
THRESHOLD = 0.5
TOP_K_VALUES = [5, 10, 20]
BOOTSTRAP_ITERATIONS = 5000
BOOTSTRAP_BLOCK_LENGTH = 5  # trading sessions; accounts partially for serial dependence
BOOTSTRAP_SEED = 20261010

EXPERIMENTS = ["E0", "E1", "E2", "E3", "E4", "E5", "E6"]
FEATURE_COUNTS = {"E0": 90, "E1": 24, "E2": 85, "E3": 70, "E4": 80, "E5": 55, "E6": 70}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_file(path: Path, description: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Required {description} not found: {path}")


def safe_corr(left: pd.Series, right: pd.Series, method: str = "pearson") -> float:
    pair = pd.concat([left, right], axis=1).dropna()
    if len(pair) < 2 or pair.iloc[:, 0].nunique() < 2 or pair.iloc[:, 1].nunique() < 2:
        return float("nan")
    value = pair.iloc[:, 0].corr(pair.iloc[:, 1], method=method)
    return float(value) if pd.notna(value) else float("nan")


def check_keys_and_targets(
    prediction: pd.DataFrame,
    validation_truth: pd.DataFrame,
    strategy: str,
) -> pd.DataFrame:
    required = set(KEY_COLUMNS + [TARGET_RETURN, TARGET_DIRECTION, PREDICTED_RETURN, PREDICTED_PROBABILITY])
    missing = sorted(required - set(prediction.columns))
    if missing:
        raise ValueError(f"{strategy} predictions missing required columns: {missing}")
    if prediction.duplicated(KEY_COLUMNS).any():
        raise ValueError(f"{strategy} predictions contain duplicate symbol/trade_date keys.")

    pred = prediction.copy()
    pred["trade_date"] = pd.to_datetime(pred["trade_date"], errors="raise")
    truth = validation_truth.copy()
    truth["trade_date"] = pd.to_datetime(truth["trade_date"], errors="raise")

    key_check = truth[KEY_COLUMNS].merge(
        pred[KEY_COLUMNS], on=KEY_COLUMNS, how="outer", indicator=True
    )
    if not key_check["_merge"].eq("both").all():
        counts = key_check["_merge"].value_counts().to_dict()
        raise ValueError(f"{strategy} prediction keys do not match validation: {counts}")

    pred[TARGET_RETURN] = pd.to_numeric(pred[TARGET_RETURN], errors="raise")
    pred[TARGET_DIRECTION] = pd.to_numeric(pred[TARGET_DIRECTION], errors="raise").astype(int)
    pred[PREDICTED_RETURN] = pd.to_numeric(pred[PREDICTED_RETURN], errors="raise")
    pred[PREDICTED_PROBABILITY] = pd.to_numeric(pred[PREDICTED_PROBABILITY], errors="raise")
    if pred[[TARGET_RETURN, TARGET_DIRECTION, PREDICTED_RETURN, PREDICTED_PROBABILITY]].isna().any().any():
        raise ValueError(f"{strategy} predictions contain missing target/prediction values.")
    if not np.isfinite(pred[[TARGET_RETURN, PREDICTED_RETURN, PREDICTED_PROBABILITY]].to_numpy(dtype=float)).all():
        raise ValueError(f"{strategy} predictions contain non-finite numeric values.")
    probabilities = pred[PREDICTED_PROBABILITY].to_numpy(dtype=float)
    if ((probabilities < 0.0) | (probabilities > 1.0)).any():
        raise ValueError(f"{strategy} probabilities fall outside [0, 1].")
    if not set(pred[TARGET_DIRECTION].unique()).issubset({0, 1}):
        raise ValueError(f"{strategy} target direction contains values outside 0/1.")

    truth_small = truth[KEY_COLUMNS + [TARGET_RETURN, TARGET_DIRECTION]]
    aligned = pred.merge(
        truth_small,
        on=KEY_COLUMNS,
        how="left",
        validate="one_to_one",
        suffixes=("", "_validation"),
    )
    ret_diff = (aligned[TARGET_RETURN] - aligned[f"{TARGET_RETURN}_validation"]).abs()
    if (ret_diff > 1e-12).any():
        raise ValueError(f"{strategy} return targets differ from the current validation dataset.")
    direction_match = aligned[TARGET_DIRECTION].eq(aligned[f"{TARGET_DIRECTION}_validation"])
    if not direction_match.all():
        raise ValueError(f"{strategy} direction targets differ from the current validation dataset.")
    aligned = aligned.drop(columns=[f"{TARGET_RETURN}_validation", f"{TARGET_DIRECTION}_validation"])
    return aligned.sort_values(KEY_COLUMNS, kind="mergesort").reset_index(drop=True)


def prediction_path(strategy: str) -> Path:
    if strategy == "E0":
        return E0_PREDICTIONS_FILE
    return EXPERIMENT_PREDICTIONS_ROOT / strategy / "validation_predictions.parquet"


def read_predictions(strategy: str, validation_truth: pd.DataFrame) -> pd.DataFrame:
    path = prediction_path(strategy)
    require_file(path, f"{strategy} validation predictions")
    raw = pd.read_parquet(path)
    return check_keys_and_targets(raw, validation_truth, strategy)


def pooled_regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    residual = predicted - actual
    total_sum_squares = float(np.square(actual - np.mean(actual)).sum())
    residual_sum_squares = float(np.square(residual).sum())
    r2 = float("nan") if total_sum_squares == 0 else 1.0 - residual_sum_squares / total_sum_squares
    return {
        "rmse": float(np.sqrt(np.mean(np.square(residual)))),
        "mae": float(np.mean(np.abs(residual))),
        "r2": float(r2),
        "pearson": safe_corr(pd.Series(actual), pd.Series(predicted), "pearson"),
    }


def pooled_classification_metrics(actual: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    predicted_class = (probability >= THRESHOLD).astype(int)
    result: dict[str, float] = {
        "accuracy": float(np.mean(predicted_class == actual)),
        "precision": float(np.sum((predicted_class == 1) & (actual == 1)) / max(1, np.sum(predicted_class == 1))),
        "recall": float(np.sum((predicted_class == 1) & (actual == 1)) / max(1, np.sum(actual == 1))),
    }
    precision = result["precision"]
    recall = result["recall"]
    result["f1"] = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    try:
        result["roc_auc"] = float(roc_auc_score(actual, probability))
    except ValueError:
        result["roc_auc"] = float("nan")
    try:
        result["pr_auc"] = float(average_precision_score(actual, probability))
    except ValueError:
        result["pr_auc"] = float("nan")
    try:
        result["log_loss"] = float(log_loss(actual, probability, labels=[0, 1]))
    except ValueError:
        result["log_loss"] = float("nan")
    return result


def date_metrics(
    frame: pd.DataFrame,
    strategy: str,
    ranking_signal_column: str | None,
    include_classification: bool,
    include_regression: bool,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for trade_date, original_group in frame.groupby("trade_date", sort=True):
        group = original_group.dropna(subset=[TARGET_RETURN]).copy()
        if group.empty:
            continue
        actual = group[TARGET_RETURN].astype(float).to_numpy()
        row: dict[str, Any] = {
            "strategy": strategy,
            "trade_date": pd.Timestamp(trade_date),
            "stock_count": int(len(group)),
        }
        if include_regression:
            pred = group[PREDICTED_RETURN].astype(float).to_numpy()
            row["daily_rmse"] = float(np.sqrt(np.mean(np.square(pred - actual))))
            row["daily_mae"] = float(np.mean(np.abs(pred - actual)))
            row["daily_pearson"] = safe_corr(group[TARGET_RETURN], group[PREDICTED_RETURN], "pearson")
        if include_classification:
            actual_direction = group[TARGET_DIRECTION].astype(int).to_numpy()
            probability = group[PREDICTED_PROBABILITY].astype(float).to_numpy()
            predicted_direction = (probability >= THRESHOLD).astype(int)
            row["daily_direction_accuracy"] = float(np.mean(predicted_direction == actual_direction))
        else:
            row["daily_direction_accuracy"] = float("nan")
        if ranking_signal_column:
            rank_frame = group.dropna(subset=[ranking_signal_column]).copy()
            row["daily_spearman"] = safe_corr(
                rank_frame[ranking_signal_column], rank_frame[TARGET_RETURN], "spearman"
            )
            ranked = rank_frame.sort_values(
                [ranking_signal_column, "symbol"], ascending=[False, True], kind="mergesort"
            )
            for k in TOP_K_VALUES:
                if len(ranked) >= k:
                    top = ranked.head(k)
                    row[f"top_{k}_average_realized_return"] = float(top[TARGET_RETURN].mean())
                    row[f"top_{k}_hit_rate"] = float((top[TARGET_RETURN] > 0).mean())
                else:
                    row[f"top_{k}_average_realized_return"] = float("nan")
                    row[f"top_{k}_hit_rate"] = float("nan")
        else:
            row["daily_spearman"] = float("nan")
            for k in TOP_K_VALUES:
                row[f"top_{k}_average_realized_return"] = float("nan")
                row[f"top_{k}_hit_rate"] = float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def strategy_summary(
    strategy: str,
    frame: pd.DataFrame,
    daily: pd.DataFrame,
    signal_column: str | None,
    model_strategy: bool,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "strategy": strategy,
        "strategy_type": "xgboost_model" if model_strategy else ("momentum_ranking_reference" if signal_column else "regression_reference"),
        "feature_count": FEATURE_COUNTS.get(strategy, 0),
        "rows": int(len(frame)),
        "dates": int(frame["trade_date"].nunique()),
    }
    if model_strategy:
        actual = frame[TARGET_RETURN].to_numpy(dtype=float)
        predicted = frame[PREDICTED_RETURN].to_numpy(dtype=float)
        row.update({f"regression_{key}": value for key, value in pooled_regression_metrics(actual, predicted).items()})
        y = frame[TARGET_DIRECTION].to_numpy(dtype=int)
        p = frame[PREDICTED_PROBABILITY].to_numpy(dtype=float)
        row.update({f"classification_{key}": value for key, value in pooled_classification_metrics(y, p).items()})
    elif strategy == "ZERO_RETURN_REGRESSION":
        actual = frame[TARGET_RETURN].to_numpy(dtype=float)
        row.update({f"regression_{key}": value for key, value in pooled_regression_metrics(actual, np.zeros_like(actual)).items()})
    row["daily_spearman_mean"] = float(daily["daily_spearman"].mean()) if daily["daily_spearman"].notna().any() else float("nan")
    row["daily_spearman_median"] = float(daily["daily_spearman"].median()) if daily["daily_spearman"].notna().any() else float("nan")
    row["ranking_days_evaluated"] = int(daily["daily_spearman"].notna().sum())
    for k in TOP_K_VALUES:
        returns_col = f"top_{k}_average_realized_return"
        hit_col = f"top_{k}_hit_rate"
        row[returns_col] = float(daily[returns_col].mean()) if daily[returns_col].notna().any() else float("nan")
        row[hit_col] = float(daily[hit_col].mean()) if daily[hit_col].notna().any() else float("nan")
    if model_strategy:
        for metric in ["daily_rmse", "daily_mae", "daily_direction_accuracy"]:
            row[f"{metric}_mean"] = float(daily[metric].mean()) if daily[metric].notna().any() else float("nan")
    return row


def moving_block_bootstrap_ci(
    paired_differences: np.ndarray,
    iterations: int = BOOTSTRAP_ITERATIONS,
    block_length: int = BOOTSTRAP_BLOCK_LENGTH,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float, float]:
    values = np.asarray(paired_differences, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan"), float("nan")
    point = float(values.mean())
    if values.size < 2:
        return point, float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    n = len(values)
    block = max(1, min(int(block_length), n))
    blocks_needed = int(np.ceil(n / block))
    offsets = np.arange(block)
    samples = np.empty(iterations, dtype=float)
    starts = np.arange(n)
    for i in range(iterations):
        block_starts = rng.choice(starts, size=blocks_needed, replace=True)
        indices = np.concatenate([(start + offsets) % n for start in block_starts])[:n]
        samples[i] = float(values[indices].mean())
    low, high = np.quantile(samples, [0.025, 0.975])
    return point, float(low), float(high)


def add_comparison(
    output_rows: list[dict[str, Any]],
    daily_scores: pd.DataFrame,
    candidate: str,
    reference: str,
    metric: str,
    higher_is_better: bool,
    comparison_seed: int,
) -> None:
    left = daily_scores.loc[daily_scores["strategy"].eq(candidate), ["trade_date", metric]].rename(columns={metric: "candidate_value"})
    right = daily_scores.loc[daily_scores["strategy"].eq(reference), ["trade_date", metric]].rename(columns={metric: "reference_value"})
    paired = left.merge(right, on="trade_date", how="inner", validate="one_to_one").dropna()
    diff = (paired["candidate_value"] - paired["reference_value"]).to_numpy(dtype=float)
    point, low, high = moving_block_bootstrap_ci(diff, seed=comparison_seed)
    direction = "positive_favors_candidate" if higher_is_better else "negative_favors_candidate"
    output_rows.append({
        "candidate": candidate,
        "reference": reference,
        "metric": metric,
        "direction_of_benefit": direction,
        "paired_dates": int(len(diff)),
        "mean_daily_difference_candidate_minus_reference": point,
        "nominal_95pct_ci_lower": low,
        "nominal_95pct_ci_upper": high,
        "bootstrap_method": "paired circular moving-block bootstrap over trading dates",
        "block_length_trading_days": int(min(BOOTSTRAP_BLOCK_LENGTH, len(diff))) if len(diff) else 0,
        "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
    })


def write_report(
    path: Path,
    summary: pd.DataFrame,
    comparisons: pd.DataFrame,
    daily: pd.DataFrame,
    input_row_count: int,
) -> None:
    lines = [
        "VALIDATION ROBUSTNESS REVIEW",
        "=" * 88,
        "",
        "DATA BOUNDARY",
        "Only the saved validation partition and validation prediction artifacts were read.",
        "No train or test partition was loaded. No models were fitted and no thresholds were tuned.",
        f"Validation prediction rows per model: {input_row_count:,}",
        f"Model strategies: {', '.join(EXPERIMENTS)}",
        "Classification threshold is fixed at 0.5.",
        f"Bootstrap: {BOOTSTRAP_ITERATIONS:,} paired circular moving-block resamples; block length {BOOTSTRAP_BLOCK_LENGTH} trading sessions.",
        "Confidence intervals are nominal 95% intervals and are not adjusted for multiple comparisons.",
        "",
        "STRATEGY SUMMARY",
        summary.to_string(index=False, float_format=lambda value: f"{value:.6f}"),
        "",
        "PAIRED COMPARISONS WITH INTERVALS",
    ]
    if comparisons.empty:
        lines.append("No paired comparisons could be computed.")
    else:
        for _, row in comparisons.iterrows():
            lines.append(
                f"{row['candidate']} vs {row['reference']} | {row['metric']} | "
                f"mean difference={row['mean_daily_difference_candidate_minus_reference']:.6f} | "
                f"nominal 95% CI=[{row['nominal_95pct_ci_lower']:.6f}, {row['nominal_95pct_ci_upper']:.6f}] | "
                f"paired dates={int(row['paired_dates'])} | {row['direction_of_benefit']}"
            )
    lines.extend(["", "E0 DATE-LEVEL EXTREMES", "Lowest E0 daily Spearman values:"])
    e0_daily = daily.loc[daily["strategy"].eq("E0")].dropna(subset=["daily_spearman"])
    if e0_daily.empty:
        lines.append("No valid E0 daily Spearman values were available.")
    else:
        for _, row in e0_daily.nsmallest(10, "daily_spearman").iterrows():
            lines.append(f"  {pd.Timestamp(row['trade_date']).date()}: Spearman={row['daily_spearman']:.6f}, stocks={int(row['stock_count'])}")
        lines.append("Highest E0 daily Spearman values:")
        for _, row in e0_daily.nlargest(10, "daily_spearman").iterrows():
            lines.append(f"  {pd.Timestamp(row['trade_date']).date()}: Spearman={row['daily_spearman']:.6f}, stocks={int(row['stock_count'])}")
    lines.extend([
        "",
        "INTERPRETATION GUARDRAILS",
        "1. A confidence interval spanning zero does not show a clear improvement over the reference in that metric.",
        "2. A confidence interval excluding zero is still exploratory because many metrics and candidates are compared.",
        "3. Date-block resampling preserves local sequences of nearby trading days better than independently resampling individual rows.",
        "4. Top-K values are gross next-session returns; they are not proof of tradable profitability.",
        "5. E0 remains the frozen reference during this review. Do not use these results to open the final test partition.",
        "6. The 5-day momentum strategy is a simple ranking reference using only the validation row's lagged return_5d feature.",
        "",
        "FILES",
        "  strategy_summary.csv",
        "  daily_strategy_metrics.csv",
        "  paired_comparisons.csv",
        "  e0_extreme_dates.csv",
        "  review_metadata.json",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    print("=" * 88)
    print("VALIDATION ROBUSTNESS REVIEW")
    print("Saved validation predictions only | No fitting | Test partition remains sealed")
    print("=" * 88)

    require_file(VALIDATION_FILE, "validation partition")
    require_file(E0_PREDICTIONS_FILE, "E0 validation predictions")

    # Read only the validation partition; never open train.parquet or test.parquet.
    validation_raw = pd.read_parquet(VALIDATION_FILE, columns=KEY_COLUMNS + [TARGET_RETURN, TARGET_DIRECTION, MOMENTUM_SIGNAL])
    if validation_raw.empty:
        raise ValueError("Validation partition is empty.")
    if validation_raw.duplicated(KEY_COLUMNS).any():
        raise ValueError("Validation partition has duplicate symbol/trade_date keys.")
    validation_raw["trade_date"] = pd.to_datetime(validation_raw["trade_date"], errors="raise")
    validation_raw[TARGET_RETURN] = pd.to_numeric(validation_raw[TARGET_RETURN], errors="raise")
    validation_raw[TARGET_DIRECTION] = pd.to_numeric(validation_raw[TARGET_DIRECTION], errors="raise").astype(int)
    validation_raw[MOMENTUM_SIGNAL] = pd.to_numeric(validation_raw[MOMENTUM_SIGNAL], errors="coerce")
    validation_truth = validation_raw[KEY_COLUMNS + [TARGET_RETURN, TARGET_DIRECTION]].copy()

    all_predictions: dict[str, pd.DataFrame] = {}
    daily_frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []
    file_hashes: dict[str, str] = {
        "validation_partition": sha256_file(VALIDATION_FILE),
        "E0_predictions": sha256_file(E0_PREDICTIONS_FILE),
    }

    for strategy in EXPERIMENTS:
        path = prediction_path(strategy)
        require_file(path, f"{strategy} predictions")
        if strategy != "E0":
            file_hashes[f"{strategy}_predictions"] = sha256_file(path)
        pred = read_predictions(strategy, validation_truth)
        all_predictions[strategy] = pred
        daily = date_metrics(pred, strategy, PREDICTED_RETURN, True, True)
        daily_frames.append(daily)
        summary_rows.append(strategy_summary(strategy, pred, daily, PREDICTED_RETURN, True))
        print(f"Loaded {strategy}: {len(pred):,} rows, {pred['trade_date'].nunique()} validation dates")

    # Zero-return regression benchmark (predict the next return is zero for all stocks).
    zero_pred = validation_truth.copy()
    zero_pred[PREDICTED_RETURN] = 0.0
    zero_daily = date_metrics(zero_pred, "ZERO_RETURN_REGRESSION", None, False, True)
    daily_frames.append(zero_daily)
    summary_rows.append(strategy_summary("ZERO_RETURN_REGRESSION", zero_pred, zero_daily, None, False))

    # Simple out-of-model ranking benchmark: rank stocks by trailing 5-session return.
    momentum = validation_raw.merge(
        validation_truth[KEY_COLUMNS + [TARGET_RETURN, TARGET_DIRECTION]],
        on=KEY_COLUMNS,
        how="left",
        validate="one_to_one",
        suffixes=("", "_truth"),
    )
    # Remove duplicate target columns if the original selected validation table already had them.
    if f"{TARGET_RETURN}_truth" in momentum.columns:
        momentum = momentum.drop(columns=[f"{TARGET_RETURN}_truth", f"{TARGET_DIRECTION}_truth"])
    momentum[PREDICTED_RETURN] = momentum[MOMENTUM_SIGNAL]
    momentum[PREDICTED_PROBABILITY] = np.nan
    momentum_daily = date_metrics(momentum, "MOMENTUM_5D_RANK", MOMENTUM_SIGNAL, False, False)
    daily_frames.append(momentum_daily)
    summary_rows.append(strategy_summary("MOMENTUM_5D_RANK", momentum, momentum_daily, MOMENTUM_SIGNAL, False))

    daily_scores = pd.concat(daily_frames, ignore_index=True, sort=False)
    summary = pd.DataFrame(summary_rows)

    # Paired date-level model comparisons against E0.
    comparison_rows: list[dict[str, Any]] = []
    model_metrics = [
        ("daily_rmse", False),
        ("daily_mae", False),
        ("daily_direction_accuracy", True),
        ("daily_spearman", True),
        ("top_5_average_realized_return", True),
        ("top_5_hit_rate", True),
        ("top_10_average_realized_return", True),
        ("top_10_hit_rate", True),
        ("top_20_average_realized_return", True),
        ("top_20_hit_rate", True),
    ]
    for idx, candidate in enumerate(["E1", "E2", "E3", "E4", "E5", "E6"], start=1):
        for metric, higher_is_better in model_metrics:
            add_comparison(comparison_rows, daily_scores, candidate, "E0", metric, higher_is_better, BOOTSTRAP_SEED + idx * 101 + len(comparison_rows))

    # Compare model ranking strategies with the trailing-5-day momentum reference.
    ranking_metrics = [
        ("daily_spearman", True),
        ("top_5_average_realized_return", True),
        ("top_5_hit_rate", True),
        ("top_10_average_realized_return", True),
        ("top_10_hit_rate", True),
        ("top_20_average_realized_return", True),
        ("top_20_hit_rate", True),
    ]
    for idx, candidate in enumerate(EXPERIMENTS, start=20):
        for metric, higher_is_better in ranking_metrics:
            add_comparison(comparison_rows, daily_scores, candidate, "MOMENTUM_5D_RANK", metric, higher_is_better, BOOTSTRAP_SEED + idx * 103 + len(comparison_rows))

    comparisons = pd.DataFrame(comparison_rows)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT_DIR / "strategy_summary.csv", index=False)
    daily_scores.to_csv(OUTPUT_DIR / "daily_strategy_metrics.csv", index=False, date_format="%Y-%m-%d")
    comparisons.to_csv(OUTPUT_DIR / "paired_comparisons.csv", index=False)
    e0_extremes = daily_scores.loc[daily_scores["strategy"].eq("E0")].dropna(subset=["daily_spearman"]).copy()
    if not e0_extremes.empty:
        e0_extremes["spearman_rank_low_to_high"] = e0_extremes["daily_spearman"].rank(method="first")
        e0_extremes.sort_values("daily_spearman").to_csv(OUTPUT_DIR / "e0_extreme_dates.csv", index=False, date_format="%Y-%m-%d")
    else:
        pd.DataFrame().to_csv(OUTPUT_DIR / "e0_extreme_dates.csv", index=False)

    metadata = {
        "review": "validation_robustness_review",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "validation_file": str(VALIDATION_FILE),
        "validation_rows": int(len(validation_raw)),
        "validation_dates": int(validation_raw["trade_date"].nunique()),
        "strategies": EXPERIMENTS + ["ZERO_RETURN_REGRESSION", "MOMENTUM_5D_RANK"],
        "test_partition_loaded": False,
        "train_partition_loaded": False,
        "models_fitted": False,
        "threshold_tuned": False,
        "threshold": THRESHOLD,
        "top_k": TOP_K_VALUES,
        "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
        "bootstrap_method": "paired circular moving-block bootstrap over trading dates",
        "bootstrap_block_length_trading_days": BOOTSTRAP_BLOCK_LENGTH,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "nominal_confidence_level": 0.95,
        "multiple_comparison_adjustment": False,
        "input_sha256": file_hashes,
        "output_directory": str(OUTPUT_DIR),
    }
    (OUTPUT_DIR / "review_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    write_report(OUTPUT_DIR / "validation_robustness_report.txt", summary, comparisons, daily_scores, len(validation_raw))

    print("\nSTRATEGY SUMMARY")
    relevant = [
        "strategy", "feature_count", "regression_rmse", "regression_mae", "regression_r2",
        "classification_accuracy", "classification_roc_auc", "daily_spearman_mean",
        "top_5_average_realized_return", "top_10_average_realized_return", "top_20_average_realized_return",
    ]
    print(summary[[column for column in relevant if column in summary.columns]].to_string(index=False))
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print(f"Report: {OUTPUT_DIR / 'validation_robustness_report.txt'}")
    print("TRAIN PARTITION LOADED: NO")
    print("TEST PARTITION LOADED : NO")
    print("MODELS FITTED         : NO")
    print("STATUS                : PASSED")


if __name__ == "__main__":
    main()
