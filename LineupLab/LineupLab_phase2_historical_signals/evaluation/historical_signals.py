"""
Historical, leakage-safe versions of LineupLab's current signals.

The key design choice is that every function receives a target season/week and
uses only information that would have been available BEFORE that target week.

This module deliberately does not modify app.py yet.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

GAME_SCRIPT_BUCKETS = [
    (-100, -9, "trailing_big"),
    (-8, -1, "trailing_close"),
    (0, 0, "tied"),
    (1, 8, "leading_close"),
    (9, 100, "leading_big"),
]

PRACTICE_SEVERITY = {
    "Full Participation in Practice": 0,
    "Limited Participation in Practice": 1,
    "Did Not Participate In Practice": 2,
}


def _week_index(season: pd.Series, week: pd.Series) -> pd.Series:
    # nfl seasons/weeks are not continuous integers across seasons, so use a
    # sortable tuple-like integer representation.
    return season.astype(int) * 100 + week.astype(int)


def filter_before_week(
    df: pd.DataFrame,
    target_season: int,
    target_week: int,
    season_col: str = "season",
    week_col: str = "week",
) -> pd.DataFrame:
    """Return only rows from games/weeks completed before the target week."""
    if df.empty:
        return df.copy()
    idx = _week_index(df[season_col], df[week_col])
    cutoff = target_season * 100 + target_week
    return df.loc[idx < cutoff].copy()


def zscore(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    std = s.std()
    if std == 0 or pd.isna(std):
        return pd.Series(0.0, index=series.index)
    return (s - s.mean()) / std


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    v = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    w = pd.to_numeric(weights, errors="coerce").fillna(0).to_numpy(dtype=float)
    mask = np.isfinite(v) & np.isfinite(w)
    if not mask.any() or w[mask].sum() <= 0:
        return float(np.nanmean(v)) if np.isfinite(v).any() else np.nan
    return float(np.average(v[mask], weights=w[mask]))


def add_recency_weight(
    df: pd.DataFrame,
    target_season: int,
    target_week: int,
    half_life_weeks: float = 6.0,
) -> pd.DataFrame:
    """Recency weights relative to the prediction date, never relative to future data."""
    out = df.copy()
    current = target_season * 100 + target_week
    idx = _week_index(out["season"], out["week"])
    weeks_ago = current - idx
    # The season/week encoding is only a convenient ordering; cross-season
    # gaps are slightly approximate. This is preferable to using future data.
    out["recency_weight"] = 0.5 ** (weeks_ago / half_life_weeks)
    return out


def bucket_score_diff(diff):
    for lo, hi, label in GAME_SCRIPT_BUCKETS:
        if lo <= diff <= hi:
            return label
    return "unknown"


def compute_historical_usage_signal(
    pbp: pd.DataFrame,
    target_season: int,
    target_week: int,
    min_opportunities: int = 20,
) -> pd.DataFrame:
    """Historical version of role volatility using only pre-target plays."""
    df = filter_before_week(pbp, target_season, target_week)
    df = df[df["play_type"].isin(["run", "pass"])].copy()
    df = df.dropna(subset=["score_differential"])
    df["game_script"] = df["score_differential"].apply(bucket_score_diff)

    rush = df[df["play_type"] == "run"].dropna(subset=["rusher_player_id"]).copy()
    rush_usage = (
        rush.groupby(["rusher_player_id", "rusher_player_name", "game_script"])
        .size().reset_index(name="opportunities")
        .rename(columns={"rusher_player_id": "player_id",
                         "rusher_player_name": "player_name"})
    )
    rush_usage["usage_type"] = "rush"

    targets = df[df["play_type"] == "pass"].dropna(subset=["receiver_player_id"]).copy()
    target_usage = (
        targets.groupby(["receiver_player_id", "receiver_player_name", "game_script"])
        .size().reset_index(name="opportunities")
        .rename(columns={"receiver_player_id": "player_id",
                         "receiver_player_name": "player_name"})
    )
    target_usage["usage_type"] = "target"

    combined = pd.concat([rush_usage, target_usage], ignore_index=True)
    if combined.empty:
        return pd.DataFrame()

    pivot = combined.pivot_table(
        index=["player_id", "player_name", "usage_type"],
        columns="game_script",
        values="opportunities",
        fill_value=0,
        aggfunc="sum",
    ).reset_index()

    script_cols = [label for _, _, label in GAME_SCRIPT_BUCKETS if label in pivot.columns]
    pivot["total_opportunities"] = pivot[script_cols].sum(axis=1)
    for c in script_cols:
        pivot[f"{c}_share"] = pivot[c] / pivot["total_opportunities"].replace(0, 1)

    share_cols = [f"{c}_share" for c in script_cols]
    pivot["role_volatility"] = pivot[share_cols].std(axis=1)
    return pivot[pivot["total_opportunities"] >= min_opportunities].copy()


def compute_historical_team_baselines(
    pbp: pd.DataFrame,
    target_season: int,
    target_week: int,
) -> pd.DataFrame:
    """Team offensive EPA baseline using only games before target week."""
    df = filter_before_week(pbp, target_season, target_week)
    off = df[df["posteam"].notna() & df["play_type"].isin(["run", "pass"])].copy()
    if off.empty:
        return pd.DataFrame(columns=["team", "season", "epa_per_play"])
    return (
        off.groupby(["season", "posteam"])
        .agg(epa_per_play=("epa", "mean"), plays=("epa", "size"))
        .reset_index()
        .rename(columns={"posteam": "team"})
    )


def compute_historical_short_rest_signal(
    schedule: pd.DataFrame,
    pbp: pd.DataFrame,
    target_season: int,
    target_week: int,
) -> pd.DataFrame:
    """Estimate a team's historical EPA change in short-rest games, pre-target only."""
    hist_sched = filter_before_week(schedule, target_season, target_week)
    if hist_sched.empty:
        return pd.DataFrame()

    rows = []
    for _, g in hist_sched.iterrows():
        for side, team_col, rest_col, opp_col in [
            ("home", "home_team", "home_rest", "away_team"),
            ("away", "away_team", "away_rest", "home_team"),
        ]:
            if pd.isna(g.get(rest_col)):
                continue
            bucket = (
                "short_rest" if g[rest_col] <= 4
                else "long_rest" if g[rest_col] >= 9
                else "normal_rest"
            )
            rows.append({
                "season": g["season"],
                "week": g["week"],
                "team": g[team_col],
                "opponent": g[opp_col],
                "rest_days": g[rest_col],
                "rest_bucket": bucket,
            })

    rest = pd.DataFrame(rows)
    perf = compute_historical_team_baselines(pbp, target_season, target_week)
    merged = rest.merge(perf, on=["season", "week", "team"], how="inner")
    if merged.empty:
        return pd.DataFrame()

    # Season-to-date baseline for each team, calculated from completed games.
    team_baseline = (
        merged.groupby(["season", "team"])["epa_per_play"]
        .mean().reset_index(name="season_baseline_epa")
    )
    merged = merged.merge(team_baseline, on=["season", "team"], how="left")
    merged["epa_delta_vs_baseline"] = (
        merged["epa_per_play"] - merged["season_baseline_epa"]
    )

    merged = add_recency_weight(merged, target_season, target_week)
    return (
        merged[merged["rest_bucket"] == "short_rest"]
        .groupby("team")
        .apply(lambda g: pd.Series({
            "short_rest_games": len(g),
            "short_rest_epa_delta": weighted_mean(
                g["epa_delta_vs_baseline"], g["recency_weight"]
            ),
        }))
        .reset_index()
    )


def compute_historical_pressure_signal(
    participation: pd.DataFrame,
    pbp: pd.DataFrame,
    target_season: int,
    target_week: int,
) -> pd.DataFrame:
    """Team pressure allowed before the target week."""
    part = filter_before_week(participation, target_season, target_week)
    plays = filter_before_week(pbp, target_season, target_week)

    game_col = next((c for c in ["game_id", "nflverse_game_id"] if c in part.columns), None)
    if not game_col or "play_id" not in part.columns or "was_pressure" not in part.columns:
        return pd.DataFrame()

    keep = [game_col, "play_id", "was_pressure"]
    if "time_to_throw" in part.columns:
        keep.append("time_to_throw")
    part = part[keep].rename(columns={game_col: "game_id"})

    if not {"game_id", "play_id", "posteam"}.issubset(plays.columns):
        return pd.DataFrame()

    merged = part.merge(
        plays[["game_id", "play_id", "posteam", "season", "week"]],
        on=["game_id", "play_id"], how="inner",
    )
    if merged.empty:
        return pd.DataFrame()

    merged = add_recency_weight(merged, target_season, target_week)
    rows = []
    for team, g in merged.groupby("posteam"):
        rows.append({
            "team": team,
            "plays": len(g),
            "pressure_rate": weighted_mean(g["was_pressure"], g["recency_weight"]),
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["pressure_rate_delta_vs_league"] = (
        out["pressure_rate"] - out["pressure_rate"].mean()
    )
    out["pressure_rate_z"] = -zscore(out["pressure_rate_delta_vs_league"])
    return out


def compute_historical_defense_signal(
    pbp: pd.DataFrame,
    target_season: int,
    target_week: int,
) -> pd.DataFrame:
    """Opponent defensive EPA allowed before target week."""
    df = filter_before_week(pbp, target_season, target_week)
    off = df[df["defteam"].notna() & df["play_type"].isin(["run", "pass"])].copy()
    if off.empty:
        return pd.DataFrame()

    off = add_recency_weight(off, target_season, target_week)
    out = (
        off.groupby("defteam")
        .apply(lambda g: pd.Series({
            "plays_faced": len(g),
            "def_epa_allowed": weighted_mean(g["epa"], g["recency_weight"]),
        }))
        .reset_index()
        .rename(columns={"defteam": "team"})
    )
    out["matchup_z"] = zscore(out["def_epa_allowed"])
    return out


def compute_target_opponents(
    schedule: pd.DataFrame,
    target_season: int,
    target_week: int,
) -> pd.DataFrame:
    """The opponent for each team in the target week, not a future-game lookup."""
    s = schedule[
        (schedule["season"] == target_season) &
        (schedule["week"] == target_week)
    ].copy()

    rows = []
    for _, g in s.iterrows():
        if pd.isna(g.get("home_team")) or pd.isna(g.get("away_team")):
            continue
        rows.append({
            "team": g["home_team"],
            "opponent": g["away_team"],
            "is_home": True,
            "rest_days": g.get("home_rest"),
        })
        rows.append({
            "team": g["away_team"],
            "opponent": g["home_team"],
            "is_home": False,
            "rest_days": g.get("away_rest"),
        })
    return pd.DataFrame(rows)


def compute_historical_signal_snapshot(
    pbp: pd.DataFrame,
    schedule: pd.DataFrame,
    participation: pd.DataFrame,
    target_season: int,
    target_week: int,
) -> pd.DataFrame:
    """
    Produce a single player/team-context snapshot for the target week.

    This is intentionally limited to signals that can be calculated from the
    currently available data without introducing additional data sources.
    """
    usage = compute_historical_usage_signal(pbp, target_season, target_week)
    if usage.empty:
        return pd.DataFrame()

    # A player may have separate rush/target rows. Keep them separate initially,
    # matching the current dashboard's structure.
    target = compute_target_opponents(schedule, target_season, target_week)
    pressure = compute_historical_pressure_signal(
        participation, pbp, target_season, target_week
    )
    defense = compute_historical_defense_signal(
        pbp, target_season, target_week
    )
    rest = compute_historical_short_rest_signal(
        schedule, pbp, target_season, target_week
    )

    out = usage.copy()

    # Derive the player's latest pre-target team from PBP, avoiding current-roster leakage.
    pre = filter_before_week(pbp, target_season, target_week)
    team_rows = []
    for player_col, name_col in [
        ("rusher_player_id", "rusher_player_name"),
        ("receiver_player_id", "receiver_player_name"),
    ]:
        if player_col not in pre.columns or "posteam" not in pre.columns:
            continue
        x = pre[[player_col, "posteam", "season", "week"]].dropna().rename(
            columns={player_col: "player_id", "posteam": "team"}
        )
        team_rows.append(x)
    if team_rows:
        teams = pd.concat(team_rows, ignore_index=True)
        teams["_idx"] = _week_index(teams["season"], teams["week"])
        teams = teams.sort_values("_idx").drop_duplicates("player_id", keep="last")
        out = out.merge(teams[["player_id", "team"]], on="player_id", how="left")

    out = out.merge(target[["team", "opponent", "is_home", "rest_days"]],
                    on="team", how="left")

    if not pressure.empty:
        out = out.merge(
            pressure[["team", "pressure_rate", "pressure_rate_delta_vs_league",
                      "pressure_rate_z"]],
            on="team", how="left"
        )

    if not rest.empty:
        out = out.merge(rest, on="team", how="left")

    if not defense.empty:
        out = out.merge(
            defense[["team", "def_epa_allowed", "matchup_z"]]
            .rename(columns={"team": "opponent"}),
            on="opponent", how="left"
        )

    # Current dashboard composite components, but now computed strictly before target week.
    components = []
    if "short_rest_epa_delta" in out.columns:
        out["short_rest_z"] = zscore(out["short_rest_epa_delta"])
        components.append("short_rest_z")
    if "pressure_rate_z" in out.columns:
        components.append("pressure_rate_z")
    if "matchup_z" in out.columns:
        components.append("matchup_z")

    # Role volatility is currently displayed as a signal but is not part of the
    # existing composite. Preserve that behavior for this evaluation snapshot.
    if components:
        out["historical_composite"] = out[components].fillna(0).mean(axis=1)
    else:
        out["historical_composite"] = np.nan

    out["target_season"] = target_season
    out["target_week"] = target_week
    return out
