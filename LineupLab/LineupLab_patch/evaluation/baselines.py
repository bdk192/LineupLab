"""Point-in-time fantasy baselines for LineupLab.

All baselines are deliberately simple.  Their purpose is not to be a final
projection model; it is to establish a fair bar that LineupLab must beat.
"""

from __future__ import annotations

import pandas as pd


def add_fantasy_baselines(
    stats: pd.DataFrame,
    *,
    player_col: str = "player_id",
    season_col: str = "season",
    week_col: str = "week",
    fantasy_col: str = "fantasy_points_ppr",
    opportunity_col: str = "opportunities",
    window: int = 4,
) -> pd.DataFrame:
    """Add strictly pre-game rolling baselines.

    Each prediction for week N only uses games through week N-1.
    Rows are sorted by player/season/week before shifting/rolling.
    """
    required = {player_col, season_col, week_col, fantasy_col}
    missing = required - set(stats.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = stats.copy()
    df = df.sort_values([player_col, season_col, week_col]).reset_index(drop=True)

    group = df.groupby(player_col, group_keys=False)

    # Explicitly shift before rolling: current-week outcome can never leak into
    # the prediction for that same week.
    prior_fantasy = group[fantasy_col].shift(1)
    df["previous_game_ppr"] = prior_fantasy

    df[f"last_{window}_games_ppr"] = (
        prior_fantasy.groupby(df[player_col])
        .rolling(window=window, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )

    df["season_to_date_ppr"] = (
        prior_fantasy.groupby(df[player_col])
        .expanding(min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )

    if opportunity_col in df.columns:
        prior_opp = group[opportunity_col].shift(1)
        df[f"last_{window}_games_opportunity"] = (
            prior_opp.groupby(df[player_col])
            .rolling(window=window, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )

    # A player-season baseline is useful for evaluating whether contextual
    # signals add value beyond simply knowing who the player is that year.
    df["season_player_games_before"] = group.cumcount()

    return df


def build_opportunity_column(
    stats: pd.DataFrame,
    *,
    carries_col: str = "carries",
    targets_col: str = "targets",
) -> pd.DataFrame:
    """Create a simple RB/WR/TE opportunity measure: carries + targets."""
    df = stats.copy()
    carries = pd.to_numeric(df.get(carries_col, 0), errors="coerce").fillna(0)
    targets = pd.to_numeric(df.get(targets_col, 0), errors="coerce").fillna(0)
    df["opportunities"] = carries + targets
    return df
