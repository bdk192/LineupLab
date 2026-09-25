"""Backtest simple LineupLab baselines and optional signal columns.

The backtest intentionally starts with no learned model.  It answers:
"How accurate are simple pre-week baselines, and do added columns improve
out-of-sample accuracy?"

For signal testing, provide a dataset containing columns such as
`lineuplab_composite`, `matchup_z`, etc.  A later stage can add a learned
regression model, but this module keeps the first evaluation transparent.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .metrics import bucket_performance, compare_models, incremental_improvement


TARGET = "actual_fantasy_points_ppr"


def evaluate(
    df: pd.DataFrame,
    prediction_cols: list[str],
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    metrics = compare_models(df, TARGET, prediction_cols)

    buckets: dict[str, pd.DataFrame] = {}
    for col in prediction_cols:
        if col in df.columns:
            buckets[col] = bucket_performance(df, col, TARGET)

    return metrics, buckets


def print_report(metrics: pd.DataFrame, buckets: dict[str, pd.DataFrame]) -> None:
    if metrics.empty:
        print("No valid predictions found.")
        return

    print("\n=== LineupLab Baseline Evaluation ===")
    print(metrics.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    baseline_row = metrics.iloc[0]
    baseline_mae = float(baseline_row["mae"])

    if len(metrics) > 1:
        print("\n=== MAE improvement vs first model ===")
        for _, row in metrics.iloc[1:].iterrows():
            improvement = incremental_improvement(baseline_mae, float(row["mae"]))
            print(f"{row['model']}: {improvement:.2f}%")

    for name, table in buckets.items():
        print(f"\n=== Outcome by {name} bucket ===")
        print(table.to_string(index=False, float_format=lambda x: f"{x:.3f}"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("evaluation/data/player_week_baseline.parquet"),
    )
    parser.add_argument(
        "--predictions",
        nargs="+",
        default=["last_4_games_ppr", "season_to_date_ppr"],
    )
    args = parser.parse_args()

    df = pd.read_parquet(args.input)

    # The first game(s) of a player's career/season have no valid historical
    # baseline.  They are excluded per model rather than filling with zero.
    usable = df.dropna(subset=[c for c in args.predictions if c in df.columns], how="all")

    metrics, buckets = evaluate(usable, args.predictions)
    print_report(metrics, buckets)


if __name__ == "__main__":
    main()
