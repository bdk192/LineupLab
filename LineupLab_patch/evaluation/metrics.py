"""Metrics used by the LineupLab evaluation/backtest layer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RegressionMetrics:
    """Core out-of-sample regression metrics."""

    n: int
    mae: float
    rmse: float
    bias: float
    pearson_r: float | None
    spearman_r: float | None


def _clean(actual: Iterable[float], predicted: Iterable[float]) -> tuple[np.ndarray, np.ndarray]:
    y = pd.to_numeric(pd.Series(actual), errors="coerce").to_numpy(dtype=float)
    p = pd.to_numeric(pd.Series(predicted), errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(y) & np.isfinite(p)
    return y[mask], p[mask]


def regression_metrics(actual: Iterable[float], predicted: Iterable[float]) -> RegressionMetrics:
    """Return metrics for a prediction series.

    Bias is defined as mean(predicted - actual), so positive values mean the
    model tends to over-project.
    """
    y, p = _clean(actual, predicted)
    if len(y) == 0:
        return RegressionMetrics(0, np.nan, np.nan, np.nan, None, None)

    errors = p - y
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors**2)))
    bias = float(np.mean(errors))

    pearson = None
    spearman = None
    if len(y) >= 2 and np.std(y) > 0 and np.std(p) > 0:
        pearson = float(np.corrcoef(y, p)[0, 1])
        spearman = float(pd.Series(y).corr(pd.Series(p), method="spearman"))

    return RegressionMetrics(len(y), mae, rmse, bias, pearson, spearman)


def compare_models(
    df: pd.DataFrame,
    actual_col: str,
    prediction_cols: list[str],
) -> pd.DataFrame:
    """Compare multiple prediction columns on the same observations."""
    rows = []
    for col in prediction_cols:
        if col not in df.columns:
            continue
        m = regression_metrics(df[actual_col], df[col])
        rows.append(
            {
                "model": col,
                "n": m.n,
                "mae": m.mae,
                "rmse": m.rmse,
                "bias": m.bias,
                "pearson_r": m.pearson_r,
                "spearman_r": m.spearman_r,
            }
        )
    return pd.DataFrame(rows)


def incremental_improvement(
    baseline_mae: float,
    candidate_mae: float,
) -> float:
    """Percent MAE improvement from baseline to candidate.

    Positive = candidate is better (lower MAE).
    """
    if not np.isfinite(baseline_mae) or baseline_mae == 0:
        return np.nan
    return float((baseline_mae - candidate_mae) / baseline_mae * 100.0)


def bucket_performance(
    df: pd.DataFrame,
    signal_col: str,
    actual_col: str,
    bins: int = 5,
) -> pd.DataFrame:
    """Describe actual outcomes across signal quantiles.

    Quantile buckets are useful for the first-pass question:
    "Do higher signal values correspond to different subsequent outcomes?"
    """
    work = df[[signal_col, actual_col]].copy()
    work[signal_col] = pd.to_numeric(work[signal_col], errors="coerce")
    work[actual_col] = pd.to_numeric(work[actual_col], errors="coerce")
    work = work.dropna()

    if work.empty:
        return pd.DataFrame()

    # qcut can fail when a signal has too many identical values. Fall back to
    # rank-based buckets in that case.
    try:
        work["bucket"] = pd.qcut(
            work[signal_col],
            q=bins,
            labels=[f"Q{i}" for i in range(1, bins + 1)],
            duplicates="drop",
        )
    except ValueError:
        ranks = work[signal_col].rank(method="first")
        work["bucket"] = pd.qcut(
            ranks,
            q=bins,
            labels=[f"Q{i}" for i in range(1, bins + 1)],
        )

    return (
        work.groupby("bucket", observed=False)
        .agg(
            observations=(actual_col, "size"),
            mean_signal=(signal_col, "mean"),
            mean_actual=(actual_col, "mean"),
            median_actual=(actual_col, "median"),
            std_actual=(actual_col, "std"),
        )
        .reset_index()
    )
