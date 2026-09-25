"""Build the first LineupLab historical player-week evaluation dataset.

This is deliberately a baseline-first implementation.  It uses nflverse's
weekly player statistics and creates predictions for week N using only data
from weeks before N.

Current output is a clean foundation for later merging of LineupLab signals:
the signal engine can add columns keyed by (season, week, player_id) without
changing the outcome/evaluation logic.

Example:
    python -m evaluation.build_player_week_dataset --seasons 2022 2023 2024 2025

Output:
    evaluation/data/player_week_baseline.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_POSITIONS = ("QB", "RB", "WR", "TE")


def load_weekly_stats(seasons: list[int]) -> pd.DataFrame:
    import nflreadpy as nfl

    stats = nfl.load_player_stats(seasons, summary_level="week").to_pandas()

    # Keep offensive fantasy-relevant positions for the first evaluation
    # version.  Kickers/defenses should get their own outcome definitions
    # rather than being mixed into this baseline.
    if "position" in stats.columns:
        stats = stats[stats["position"].isin(DEFAULT_POSITIONS)].copy()

    return stats


def prepare_dataset(stats: pd.DataFrame) -> pd.DataFrame:
    required = {
        "player_id",
        "player_display_name",
        "position",
        "season",
        "week",
        "season_type",
        "team",
        "opponent_team",
        "fantasy_points_ppr",
    }
    missing = required - set(stats.columns)
    if missing:
        raise ValueError(f"nflverse player stats missing columns: {sorted(missing)}")

    # Regular season only for the first version.  Postseason is a different
    # population and should not silently be mixed into weekly regular-season
    # prediction tests.
    df = stats[stats["season_type"].eq("REG")].copy()

    # nflverse weekly player stats can include rows that did not actually
    # contribute to the relevant offensive stat line.  Keep all player-weeks
    # for now, but normalize numeric fields and explicitly mark zero-outcome
    # games rather than dropping them.
    numeric_cols = [
        "fantasy_points_ppr",
        "carries",
        "targets",
        "receptions",
        "rushing_yards",
        "receiving_yards",
        "rushing_tds",
        "receiving_tds",
        "attempts",
        "passing_yards",
        "passing_tds",
        "passing_interceptions",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    df["opportunities"] = (
        df.get("carries", 0).fillna(0)
        + df.get("targets", 0).fillna(0)
    )

    # Sort first, then create strictly pre-week baselines.
    from .baselines import add_fantasy_baselines

    df = add_fantasy_baselines(df, opportunity_col="opportunities", window=4)

    # Make the target explicit.  Keeping it separate from prediction columns
    # makes accidental leakage easier to spot in downstream code.
    df = df.rename(columns={"fantasy_points_ppr": "actual_fantasy_points_ppr"})

    # Stable, compact schema for downstream signal merges.
    keep = [
        "player_id",
        "player_display_name",
        "position",
        "season",
        "week",
        "team",
        "opponent_team",
        "actual_fantasy_points_ppr",
        "opportunities",
        "previous_game_ppr",
        "last_4_games_ppr",
        "season_to_date_ppr",
        "last_4_games_opportunity",
        "season_player_games_before",
    ]
    keep = [c for c in keep if c in df.columns]

    return df[keep].sort_values(["season", "week", "position", "player_display_name"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seasons", nargs="+", type=int, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("evaluation/data/player_week_baseline.parquet"),
    )
    args = parser.parse_args()

    stats = load_weekly_stats(args.seasons)
    dataset = prepare_dataset(stats)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(args.output, index=False)

    print(f"Wrote {len(dataset):,} player-week rows to {args.output}")
    print(f"Seasons: {sorted(dataset['season'].unique().tolist())}")
    print(f"Columns: {', '.join(dataset.columns)}")


if __name__ == "__main__":
    main()
