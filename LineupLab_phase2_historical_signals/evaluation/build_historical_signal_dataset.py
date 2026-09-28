"""Build a leakage-safe historical signal dataset across many season/weeks.

This is the deliverable both README.md and README_PHASE2.md point to as the
"next step": loop historical_signals.compute_historical_signal_snapshot()
over every (season, week) already present in the Phase 1 baseline dataset,
then merge the result with that dataset on (player_id, season, week) so
signals and real outcomes live in one evaluation-ready table.

It also adds NAIVE (unweighted, unfitted) baseline+signal columns. This is
deliberate and should not be mistaken for a real model: Section 20 of the
project roadmap is explicit that weights should not be optimized before
first checking whether signals contain useful information at all. Adding a
raw z-scored signal directly onto a points-scale baseline assumes "one
standard deviation of signal = one fantasy point," which is almost
certainly the wrong MAGNITUDE -- but it's sufficient to check DIRECTION and
whether error moves at all, which is the actual Phase 3/4 question. A real
weighted model comes later, only if this cruder test says these signals
carry real information.

Usage:
    python -m evaluation.build_historical_signal_dataset \
        --baseline evaluation/data/player_week_baseline.parquet \
        --output evaluation/data/player_week_with_signals.parquet

Requires evaluation/data/player_week_baseline.parquet to already exist
(built by build_player_week_dataset.py).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .historical_signals import compute_historical_signal_snapshot


PBP_NEEDED = [
    "game_id", "play_id", "play_type", "score_differential",
    "season", "week", "rusher_player_id", "rusher_player_name",
    "receiver_player_id", "receiver_player_name",
    "passer_player_id", "passer_player_name",
    "epa", "posteam", "defteam",
]
PARTICIPATION_NEEDED = [
    "game_id", "nflverse_game_id", "play_id",
    "was_pressure", "time_to_throw", "defense_man_zone_type",
]


def load_season_data(season: int):
    """Load one season's pbp/schedule/participation once, reused across every
    week of that season -- the signal functions filter internally via
    filter_before_week(), so there's no need to reload per week."""
    import nflreadpy as nfl

    pbp = nfl.load_pbp([season])
    pbp = pbp.select([c for c in PBP_NEEDED if c in pbp.columns]).to_pandas()

    schedule = nfl.load_schedules([season]).to_pandas()

    try:
        part = nfl.load_participation([season])
        part = part.select(
            [c for c in PARTICIPATION_NEEDED if c in part.columns]
        ).to_pandas()
    except ValueError:
        # Participation (FTN charting) isn't published for every season --
        # notably not for an in-progress one. Signals depending on it
        # (pressure) simply come back empty for those weeks; downstream
        # code treats that as missing, not a crash.
        part = pd.DataFrame()

    return pbp, schedule, part


def collapse_to_one_row_per_player(snap: pd.DataFrame) -> pd.DataFrame:
    """A player can appear as multiple usage_type rows (rush/target/
    pass_attempt) in one snapshot -- e.g. a receiving back, or a scrambling
    QB. Merging 1:1 against the outcome dataset needs one row per player, so
    keep only the row with the most opportunities. Team-level signals
    (rest/pressure/matchup) are identical across a player's rows regardless
    -- this only affects which usage_type's role_volatility is retained."""
    if snap.empty:
        return snap
    snap = snap.sort_values("total_opportunities", ascending=False)
    return snap.drop_duplicates(subset=["player_id"], keep="first")


def build_all_snapshots(baseline: pd.DataFrame) -> pd.DataFrame:
    season_weeks = (
        baseline[["season", "week"]].drop_duplicates().sort_values(["season", "week"])
    )

    snapshots = []
    for season in sorted(season_weeks["season"].unique()):
        print(f"Loading season {season} data...")
        pbp, schedule, participation = load_season_data(int(season))
        weeks = sorted(season_weeks.loc[season_weeks["season"] == season, "week"].unique())

        for week in weeks:
            # Week 1 has no prior-week data to build signals from -- skip
            # rather than silently returning an empty/meaningless row.
            if week <= 1:
                continue
            snap = compute_historical_signal_snapshot(
                pbp, schedule, participation, int(season), int(week)
            )
            snap = collapse_to_one_row_per_player(snap)
            if not snap.empty:
                snapshots.append(snap)
            print(f"  season={season} week={week}: {len(snap)} player rows")

    if not snapshots:
        return pd.DataFrame()
    return pd.concat(snapshots, ignore_index=True)


def add_naive_combinations(merged: pd.DataFrame) -> pd.DataFrame:
    out = merged.copy()
    base = out["last_4_games_ppr"]
    if "season_to_date_ppr" in out.columns:
        base = base.fillna(out["season_to_date_ppr"])
    out["naive_base"] = base

    # Level-shifting signals: naive add onto a points-scale baseline.
    signal_cols = {
        "short_rest_epa_delta_z": "naive_base_plus_short_rest",
        "pressure_rate_z": "naive_base_plus_pressure",
        "matchup_z": "naive_base_plus_matchup",
        "historical_composite": "naive_base_plus_composite",
    }
    # historical_signals.py computes short_rest_z internally but doesn't
    # rename/export it under a fixed name on the merged snapshot -- guard
    # for both possible spellings rather than assuming one.
    if "short_rest_z" in out.columns and "short_rest_epa_delta_z" not in out.columns:
        out["short_rest_epa_delta_z"] = out["short_rest_z"]

    for src, dest in signal_cols.items():
        if src in out.columns:
            out[dest] = base + out[src].fillna(0)

    # role_volatility is a DISPERSION signal, not level-shifting -- it does
    # NOT belong added to a point prediction. Instead expose what's needed
    # to test it on its own terms: does higher volatility correspond to a
    # BIGGER baseline prediction error (regardless of direction)?
    if "actual_fantasy_points_ppr" in out.columns:
        out["baseline_abs_error"] = (out["actual_fantasy_points_ppr"] - base).abs()

    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline", type=Path,
        default=Path("evaluation/data/player_week_baseline.parquet"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("evaluation/data/player_week_with_signals.parquet"),
    )
    args = parser.parse_args()

    baseline = pd.read_parquet(args.baseline)
    print(f"Loaded {len(baseline):,} baseline player-weeks")

    print(
        "Building leakage-safe historical signal snapshots -- this loads "
        "full play-by-play per season, so it can take a while..."
    )
    signals = build_all_snapshots(baseline)
    print(f"\nBuilt {len(signals):,} total signal rows across all season/weeks")

    if signals.empty:
        print("No signal rows were produced -- nothing to merge.")
        return

    merged = baseline.merge(
        signals,
        left_on=["player_id", "season", "week"],
        right_on=["player_id", "target_season", "target_week"],
        how="left",
        suffixes=("", "_signal"),
    )

    merged = add_naive_combinations(merged)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(args.output, index=False)

    match_col = "historical_composite" if "historical_composite" in merged.columns else None
    matched_pct = merged[match_col].notna().mean() * 100 if match_col else float("nan")

    print(f"\nWrote {len(merged):,} rows to {args.output}")
    print(f"{matched_pct:.1f}% of player-weeks matched to a historical signal snapshot")
    print(
        "\nNext: run\n"
        "  python -m evaluation.backtest --input "
        f"{args.output} --predictions naive_base naive_base_plus_composite "
        "naive_base_plus_matchup naive_base_plus_pressure naive_base_plus_short_rest\n"
        "to compare baseline-alone vs baseline+signal MAE (Section 11)."
    )


if __name__ == "__main__":
    main()
