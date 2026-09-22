"""
Fantasy Signal Dashboard -- Cloud-deployable version
------------------------------------------------------
This combines what were previously four separate local scripts
(game_script_usage.py, travel_rest_effects.py, combine_signals.py,
dashboard.py) into ONE app, designed to run entirely on Streamlit
Community Cloud -- no local Python installs, no local storage needed.

Data pulling and heavy computation are wrapped in @st.cache_data so they
only run once (per day, per the ttl below) rather than on every click.

DEPLOYMENT (see chat for the full walkthrough):
  1. Push this file + requirements.txt to a GitHub repo
  2. Connect that repo at https://share.streamlit.io
  3. Streamlit Cloud installs requirements.txt and runs this file for you
"""

import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Fantasy Signal Dashboard", layout="wide")

# Kept modest on purpose -- more seasons = more data = slower/heavier on
# a free-tier cloud instance. Expand once you've confirmed this runs
# comfortably within Streamlit Cloud's resource limits.
SEASONS = [2023, 2024]

GAME_SCRIPT_BUCKETS = [
    (-100, -9, "trailing_big"),
    (-8, -1, "trailing_close"),
    (0, 0, "tied"),
    (1, 8, "leading_close"),
    (9, 100, "leading_big"),
]


def bucket_score_diff(diff):
    for lo, hi, label in GAME_SCRIPT_BUCKETS:
        if lo <= diff <= hi:
            return label
    return "unknown"


# ---------------------------------------------------------------------------
# DATA LOADING (cached -- runs once per day per unique input, not per click)
# ---------------------------------------------------------------------------
@st.cache_data(ttl=86400, show_spinner="Loading play-by-play data (first load can take a minute)...")
def load_pbp(seasons):
    import nflreadpy as nfl
    # Full play-by-play has ~372 columns; this app only uses about a dozen.
    # Selecting down to just what's needed BEFORE converting from Polars to
    # pandas cuts memory usage substantially -- likely the main fix for a
    # memory-related crash on Streamlit Cloud's free tier (1GB RAM).
    needed = [
        "game_id", "play_id", "play_type", "score_differential",
        "rusher_player_id", "rusher_player_name",
        "receiver_player_id", "receiver_player_name",
        "epa", "posteam", "defteam", "penalty",
    ]
    df = nfl.load_pbp(seasons)
    available = [c for c in needed if c in df.columns]
    return df.select(available).to_pandas()


@st.cache_data(ttl=86400, show_spinner="Loading schedule data...")
def load_schedule(seasons):
    import nflreadpy as nfl
    return nfl.load_schedules(seasons).to_pandas()


@st.cache_data(ttl=86400, show_spinner="Loading roster data...")
def load_rosters(seasons):
    import nflreadpy as nfl
    # load_rosters() gives one row per player per season, already reflecting
    # that player's most recent team for the season (per nflverse docs) --
    # this is what lets us map a player_id to a current team.
    return nfl.load_rosters(seasons).to_pandas()


@st.cache_data(ttl=86400, show_spinner="Loading participation data (pressure/scheme)...")
def load_participation(seasons):
    import nflreadpy as nfl
    # Sourced from FTN Data via nflverse; confirmed available for 2023-2024
    # as of this writing -- if SEASONS expands beyond that range, this may
    # come back empty/partial for years outside that coverage.
    # Also trimmed to needed columns for the same memory reason as load_pbp --
    # the full table includes heavy list-type columns (offense_players,
    # defense_players) this app never uses.
    needed = [
        "game_id", "nflverse_game_id", "play_id", "was_pressure",
        "time_to_throw", "defense_man_zone_type",
    ]
    df = nfl.load_participation(seasons)
    available = [c for c in needed if c in df.columns]
    return df.select(available).to_pandas()


@st.cache_data(ttl=86400, show_spinner="Loading contract data...")
def load_contracts():
    import nflreadpy as nfl
    # Sourced from OverTheCap.com -- no season argument; this returns the
    # full historical contract database, which we filter ourselves below.
    return nfl.load_contracts().to_pandas()


def build_player_team_map(rosters: pd.DataFrame) -> pd.DataFrame:
    """One row per player: their most recent known team across the loaded
    seasons. Handles mid-season trades by keeping the latest season's row."""
    latest = rosters.sort_values("season").drop_duplicates(subset="gsis_id", keep="last")
    cols = [c for c in ["gsis_id", "team", "full_name", "position"] if c in latest.columns]
    return latest[cols].rename(columns={"gsis_id": "player_id"})


# ---------------------------------------------------------------------------
# SIGNAL 1: game-script conditional usage
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Computing game-script usage signal...")
def compute_usage_by_game_script(pbp: pd.DataFrame) -> pd.DataFrame:
    df = pbp[pbp["play_type"].isin(["run", "pass"])].copy()
    df = df.dropna(subset=["score_differential"])
    df["game_script"] = df["score_differential"].apply(bucket_score_diff)

    rush = df[df["play_type"] == "run"].copy()
    rush_usage = (
        rush.groupby(["rusher_player_id", "rusher_player_name", "game_script"])
        .size()
        .reset_index(name="carries")
        .rename(columns={"rusher_player_id": "player_id", "rusher_player_name": "player_name"})
    )
    rush_usage["usage_type"] = "rush"

    pas = df[df["play_type"] == "pass"].dropna(subset=["receiver_player_id"]).copy()
    tgt_usage = (
        pas.groupby(["receiver_player_id", "receiver_player_name", "game_script"])
        .size()
        .reset_index(name="carries")
        .rename(columns={"receiver_player_id": "player_id", "receiver_player_name": "player_name"})
    )
    tgt_usage["usage_type"] = "target"

    combined = pd.concat([rush_usage, tgt_usage], ignore_index=True)
    combined = combined.rename(columns={"carries": "opportunities"})

    pivot = combined.pivot_table(
        index=["player_id", "player_name", "usage_type"],
        columns="game_script",
        values="opportunities",
        fill_value=0,
        aggfunc="sum",
    ).reset_index()

    script_cols = [c for _, _, c in GAME_SCRIPT_BUCKETS if c in pivot.columns]
    pivot["total_opportunities"] = pivot[script_cols].sum(axis=1)
    for c in script_cols:
        pivot[f"{c}_share"] = pivot[c] / pivot["total_opportunities"].replace(0, 1)

    share_cols = [f"{c}_share" for c in script_cols]
    pivot["role_volatility"] = pivot[share_cols].std(axis=1)
    pivot = pivot[pivot["total_opportunities"] >= 20]
    return pivot


# ---------------------------------------------------------------------------
# SIGNAL 2: travel/rest effects
# ---------------------------------------------------------------------------
def compute_rest_situations(schedule: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, g in schedule.iterrows():
        for side, team_col, rest_col, opp_col in [
            ("home", "home_team", "home_rest", "away_team"),
            ("away", "away_team", "away_rest", "home_team"),
        ]:
            rest_days = g.get(rest_col)
            if pd.isna(rest_days):
                continue
            bucket = "short_rest" if rest_days <= 4 else "long_rest" if rest_days >= 9 else "normal_rest"
            rows.append({
                "season": g["season"], "week": g["week"], "team": g[team_col],
                "opponent": g[opp_col], "is_home": side == "home",
                "rest_days": rest_days, "rest_bucket": bucket,
            })
    return pd.DataFrame(rows)


def compute_offensive_output_by_game(pbp: pd.DataFrame) -> pd.DataFrame:
    off = pbp[pbp["posteam"].notna() & pbp["play_type"].isin(["run", "pass"])].copy()
    return (
        off.groupby(["season", "week", "posteam"])
        .agg(epa_per_play=("epa", "mean"), plays=("epa", "size"))
        .reset_index()
        .rename(columns={"posteam": "team"})
    )


@st.cache_data(show_spinner="Computing travel/rest signal...")
def compute_schedule_effects(schedule: pd.DataFrame, pbp: pd.DataFrame) -> pd.DataFrame:
    rest_df = compute_rest_situations(schedule)
    perf_df = compute_offensive_output_by_game(pbp)
    merged = rest_df.merge(perf_df, on=["season", "week", "team"], how="inner")

    baseline = (
        merged.groupby(["season", "team"])["epa_per_play"].mean()
        .reset_index().rename(columns={"epa_per_play": "season_baseline_epa"})
    )
    merged = merged.merge(baseline, on=["season", "team"])
    merged["epa_delta_vs_baseline"] = merged["epa_per_play"] - merged["season_baseline_epa"]

    return (
        merged.groupby(["team", "rest_bucket"])
        .agg(games=("epa_delta_vs_baseline", "size"), avg_epa_delta=("epa_delta_vs_baseline", "mean"))
        .reset_index()
    )


# ---------------------------------------------------------------------------
# SIGNAL 3: referee/pace tendencies
# ---------------------------------------------------------------------------
def compute_game_level_stats(pbp: pd.DataFrame) -> pd.DataFrame:
    """Plays and penalties per game -- the raw material for pace/strictness."""
    agg_kwargs = {"plays": ("epa", "size")}
    if "penalty" in pbp.columns:
        agg_kwargs["penalties"] = ("penalty", "sum")
    return pbp.groupby("game_id").agg(**agg_kwargs).reset_index()


@st.cache_data(show_spinner="Computing referee/pace signal...")
def compute_referee_signal(schedule: pd.DataFrame, pbp: pd.DataFrame):
    """
    Uses the 'referee' column already present in load_schedules() output,
    rather than a separate officials table -- this avoids a game_id join
    across two different sources, since schedule and pbp share the same
    game_id lineage. Still written defensively in case the column name
    or join doesn't hold up as expected.
    """
    warnings = []
    cols = schedule.columns.tolist()

    ref_col = next((c for c in ["referee", "official_referee"] if c in cols), None)
    game_col = next((c for c in ["game_id", "nflverse_game_id"] if c in cols), None)

    if not (ref_col and game_col):
        warnings.append(
            f"[referee_signal] Couldn't find a referee/game_id column in schedule "
            f"data. Actual columns found: {cols}. Skipping this signal."
        )
        return pd.DataFrame(), warnings

    referees = schedule[[game_col, ref_col]].dropna(subset=[ref_col]).rename(
        columns={game_col: "game_id", ref_col: "referee_name"}
    )

    game_stats = compute_game_level_stats(pbp)
    merged = referees.merge(game_stats, on="game_id", how="inner")

    if merged.empty:
        warnings.append(
            f"[referee_signal] Schedule and play-by-play still didn't join on "
            f"game_id. Sample schedule game_id: {referees['game_id'].iloc[0] if len(referees) else 'n/a'} "
            f"vs sample pbp game_id: {pbp['game_id'].iloc[0] if 'game_id' in pbp.columns and len(pbp) else 'n/a'}."
        )
        return pd.DataFrame(), warnings

    league_avg_plays = merged["plays"].mean()
    summary = merged.groupby("referee_name").agg(
        games=("game_id", "nunique"),
        avg_plays=("plays", "mean"),
    )
    summary["plays_delta_vs_league"] = summary["avg_plays"] - league_avg_plays

    if "penalties" in merged.columns:
        league_avg_penalties = merged["penalties"].mean()
        pen_summary = merged.groupby("referee_name")["penalties"].mean()
        summary["avg_penalties"] = pen_summary
        summary["penalties_delta_vs_league"] = pen_summary - league_avg_penalties
    else:
        warnings.append(
            "[referee_signal] No 'penalty' column found in play-by-play data -- "
            "penalty-rate tendencies weren't computed, only pace (plays/game)."
        )

    summary = summary.reset_index()
    summary = summary[summary["games"] >= 5]  # drop refs with too few games to trust
    warnings.extend(validate_signal("plays_delta_vs_league", summary["plays_delta_vs_league"], (-10, 10)))

    return summary.sort_values("plays_delta_vs_league", ascending=False), warnings


# ---------------------------------------------------------------------------
# SIGNAL 4a: O-line / pressure allowed (team offensive line quality proxy)
# ---------------------------------------------------------------------------
def _find_participation_join_cols(participation: pd.DataFrame):
    cols = participation.columns.tolist()
    game_col = next((c for c in ["game_id", "nflverse_game_id"] if c in cols), None)
    play_col = "play_id" if "play_id" in cols else None
    return game_col, play_col, cols


@st.cache_data(show_spinner="Computing O-line/pressure signal...")
def compute_pressure_signal(participation: pd.DataFrame, pbp: pd.DataFrame):
    warnings = []
    game_col, play_col, cols = _find_participation_join_cols(participation)
    pressure_col = "was_pressure" if "was_pressure" in cols else None

    if not (game_col and play_col and pressure_col):
        warnings.append(
            f"[pressure_signal] Missing expected join/pressure columns in "
            f"participation data. Actual columns found: {cols}. Skipping this signal."
        )
        return pd.DataFrame(), warnings

    keep = [game_col, play_col, pressure_col]
    if "time_to_throw" in cols:
        keep.append("time_to_throw")
    part = participation[keep].rename(columns={game_col: "game_id", play_col: "play_id"})

    if "game_id" not in pbp.columns or "play_id" not in pbp.columns:
        warnings.append("[pressure_signal] pbp is missing game_id/play_id -- can't join.")
        return pd.DataFrame(), warnings

    pbp_small = pbp[["game_id", "play_id", "posteam"]].dropna(subset=["posteam"])
    merged = part.merge(pbp_small, on=["game_id", "play_id"], how="inner")

    if merged.empty:
        warnings.append(
            "[pressure_signal] participation and pbp didn't join on game_id/play_id -- "
            "sample participation game_id: "
            f"{part['game_id'].iloc[0] if len(part) else 'n/a'} vs sample pbp game_id: "
            f"{pbp['game_id'].iloc[0] if len(pbp) else 'n/a'}."
        )
        return pd.DataFrame(), warnings

    agg_kwargs = {"plays": (pressure_col, "size"), "pressure_rate": (pressure_col, "mean")}
    summary = merged.groupby("posteam").agg(**agg_kwargs).reset_index().rename(columns={"posteam": "team"})

    if "time_to_throw" in merged.columns:
        ttt = merged.groupby("posteam")["time_to_throw"].mean().reset_index()
        ttt.columns = ["team", "avg_time_to_throw"]
        summary = summary.merge(ttt, on="team", how="left")

    league_avg = summary["pressure_rate"].mean()
    # Negative delta = allows pressure LESS than league average = better O-line.
    summary["pressure_rate_delta_vs_league"] = summary["pressure_rate"] - league_avg
    warnings.extend(validate_signal("pressure_rate", summary["pressure_rate"], (0.0, 1.0)))

    return summary.sort_values("pressure_rate_delta_vs_league"), warnings


# ---------------------------------------------------------------------------
# SIGNAL 4b: defensive scheme tendency (man vs. zone rate)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Computing defensive scheme-matchup signal...")
def compute_scheme_signal(participation: pd.DataFrame, pbp: pd.DataFrame):
    warnings = []
    game_col, play_col, cols = _find_participation_join_cols(participation)
    scheme_col = "defense_man_zone_type" if "defense_man_zone_type" in cols else None

    if not (game_col and play_col and scheme_col):
        warnings.append(
            f"[scheme_signal] Missing expected join/scheme columns in participation "
            f"data. Actual columns found: {cols}. Skipping this signal."
        )
        return pd.DataFrame(), warnings

    part = participation[[game_col, play_col, scheme_col]].rename(
        columns={game_col: "game_id", play_col: "play_id"}
    )

    if "game_id" not in pbp.columns or "play_id" not in pbp.columns:
        warnings.append("[scheme_signal] pbp is missing game_id/play_id -- can't join.")
        return pd.DataFrame(), warnings

    pbp_small = pbp[["game_id", "play_id", "defteam"]].dropna(subset=["defteam"])
    merged = part.merge(pbp_small, on=["game_id", "play_id"], how="inner").dropna(subset=[scheme_col])

    if merged.empty:
        warnings.append(
            "[scheme_signal] participation and pbp didn't join on game_id/play_id, "
            "or no non-null scheme values were present after joining."
        )
        return pd.DataFrame(), warnings

    counts = merged.groupby(["defteam", scheme_col]).size().reset_index(name="plays")
    pivot = counts.pivot_table(index="defteam", columns=scheme_col, values="plays", fill_value=0)
    pivot["total_plays"] = pivot.sum(axis=1)
    for c in [c for c in pivot.columns if c != "total_plays"]:
        pivot[f"{c}_rate"] = pivot[c] / pivot["total_plays"]
    pivot = pivot.reset_index().rename(columns={"defteam": "team"})

    return pivot, warnings


# ---------------------------------------------------------------------------
# SIGNAL 5: contract-year status (context signal -- NOT added to composite,
# since the "contract year effect" itself is genuinely disputed in sports
# analytics research, not a settled positive/negative direction to assume)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Computing contract-year signal...")
def compute_contract_year_signal(contracts: pd.DataFrame, reference_season: int):
    warnings = []
    cols = contracts.columns.tolist()
    required = ["gsis_id", "year_signed", "years"]
    missing = [c for c in required if c not in cols]

    if missing:
        warnings.append(
            f"[contract_signal] Missing expected columns {missing}. "
            f"Actual columns found: {cols}. Skipping this signal."
        )
        return pd.DataFrame(), warnings

    df = contracts.dropna(subset=["gsis_id", "year_signed", "years"]).copy()
    df["contract_end_year"] = df["year_signed"] + df["years"] - 1
    df["is_contract_year"] = df["contract_end_year"] == reference_season

    # A player can have multiple historical contracts on file -- keep only
    # the one that was current as of the reference season.
    df = df[df["year_signed"] <= reference_season].sort_values("year_signed")
    current = df.groupby("gsis_id", as_index=False).tail(1)

    keep = ["gsis_id", "is_contract_year", "contract_end_year"]
    if "apy_cap_pct" in cols:
        keep.append("apy_cap_pct")
    result = current[keep].rename(columns={"gsis_id": "player_id"})

    warnings.extend(
        validate_signal(
            "contract_end_year", result["contract_end_year"],
            (reference_season - 15, reference_season + 10),
        )
    )
    return result, warnings


# ---------------------------------------------------------------------------
# VALIDATION + COMPOSITE (same logic/spirit as combine_signals.py)
# ---------------------------------------------------------------------------
def zscore(series: pd.Series) -> pd.Series:
    mean, std = series.mean(), series.std()
    if std == 0 or np.isnan(std):
        return pd.Series(0, index=series.index)
    return (series - mean) / std


def validate_signal(name, series, expected_range):
    warnings = []
    missing_pct = series.isna().mean() * 100
    if missing_pct > 30:
        warnings.append(f"[{name}] {missing_pct:.0f}% missing values.")
    valid = series.dropna()
    if len(valid) == 0:
        warnings.append(f"[{name}] no valid values at all -- excluded from composite.")
        return warnings
    lo, hi = expected_range
    out_pct = ((valid < lo) | (valid > hi)).mean() * 100
    if out_pct > 10:
        warnings.append(f"[{name}] {out_pct:.0f}% of values outside expected range {expected_range}.")
    if valid.std() == 0:
        warnings.append(f"[{name}] zero variance -- likely a bug.")
    return warnings


def build_dashboard_data(usage_df: pd.DataFrame, schedule_df: pd.DataFrame, player_team_map: pd.DataFrame, pressure_df: pd.DataFrame):
    warnings = []
    warnings.extend(validate_signal("role_volatility", usage_df["role_volatility"], (0.0, 0.5)))

    short_rest = schedule_df[schedule_df["rest_bucket"] == "short_rest"][["team", "avg_epa_delta"]]
    short_rest = short_rest.rename(columns={"avg_epa_delta": "short_rest_epa_delta"})
    warnings.extend(validate_signal("short_rest_epa_delta", short_rest["short_rest_epa_delta"], (-0.3, 0.3)))

    usage_df = usage_df.copy()
    usage_df["role_volatility_z"] = zscore(usage_df["role_volatility"])
    short_rest["short_rest_epa_delta_z"] = zscore(short_rest["short_rest_epa_delta"])

    # Join each player to their current team, then bring the team-level
    # short-rest signal onto the player's own row -- this is the merge
    # that was previously just a placeholder.
    usage_df = usage_df.merge(player_team_map, on="player_id", how="left")
    missing_team_pct = usage_df["team"].isna().mean() * 100
    if missing_team_pct > 15:
        warnings.append(
            f"[player-team join] {missing_team_pct:.0f}% of players couldn't be "
            f"matched to a team -- likely retired/inactive players still present "
            f"in play-by-play data, or an ID mismatch worth double-checking."
        )

    usage_df = usage_df.merge(
        short_rest[["team", "short_rest_epa_delta", "short_rest_epa_delta_z"]],
        on="team",
        how="left",
    )

    composite_components = ["short_rest_epa_delta_z"]

    if not pressure_df.empty and "pressure_rate_delta_vs_league" in pressure_df.columns:
        pressure_df = pressure_df.copy()
        # Lower pressure-allowed is better, so flip the sign before z-scoring
        # such that higher composite = better O-line, consistent with the
        # other components (higher z = better).
        pressure_df["pressure_rate_z"] = -zscore(pressure_df["pressure_rate_delta_vs_league"])
        usage_df = usage_df.merge(
            pressure_df[["team", "pressure_rate_delta_vs_league", "pressure_rate_z"]],
            on="team",
            how="left",
        )
        composite_components.append("pressure_rate_z")
    else:
        warnings.append(
            "[pressure_signal] Not available this run -- composite score built "
            "without it. See the referee/pressure tabs for the specific reason."
        )

    # Use the roster's real full name for display/search when available --
    # the pbp-derived player_name column is abbreviated (e.g. "T.Kelce"),
    # which breaks searches for a full first name.
    if "full_name" in usage_df.columns:
        usage_df["display_name"] = usage_df["full_name"].fillna(usage_df["player_name"])
    else:
        usage_df["display_name"] = usage_df["player_name"]

    # Composite = average of whichever z-scored components are available
    # this run. Averaging (not summing) keeps the scale roughly comparable
    # even when a signal is temporarily missing, rather than silently
    # shrinking everyone's score toward zero.
    usage_df["composite_score"] = usage_df[composite_components].fillna(0).mean(axis=1)

    return usage_df, short_rest, warnings


# ---------------------------------------------------------------------------
# APP
# ---------------------------------------------------------------------------
st.title("Fantasy Signal Dashboard")
st.caption("Every raw signal is shown alongside any composite score — nothing is hidden inside a black-box number.")

pbp = load_pbp(SEASONS)
schedule = load_schedule(SEASONS)
rosters = load_rosters(SEASONS)
participation = load_participation(SEASONS)
contracts = load_contracts()
usage_df = compute_usage_by_game_script(pbp)
schedule_df = compute_schedule_effects(schedule, pbp)
player_team_map = build_player_team_map(rosters)
pressure_df, pressure_warnings = compute_pressure_signal(participation, pbp)
scheme_df, scheme_warnings = compute_scheme_signal(participation, pbp)
contract_df, contract_warnings = compute_contract_year_signal(contracts, max(SEASONS))
usage_df, short_rest_df, warnings = build_dashboard_data(usage_df, schedule_df, player_team_map, pressure_df)

# Contract-year status merges directly by player_id (a true player-level
# signal, unlike the team-level ones) -- shown as context, not part of the
# composite, per the caveat above about disputed research on the effect.
if not contract_df.empty:
    usage_df = usage_df.merge(contract_df, on="player_id", how="left")

referee_df, referee_warnings = compute_referee_signal(schedule, pbp)
warnings = warnings + referee_warnings + pressure_warnings + scheme_warnings + contract_warnings

with st.expander(f"Validation report ({len(warnings)} issue(s))", expanded=len(warnings) > 0):
    if warnings:
        for w in warnings:
            st.warning(w)
    else:
        st.success("All signals passed validation checks cleanly.")

# --- Search any player: one row, every signal, in one place ---
st.subheader("Search any player")
player_search = st.text_input(
    "Type a player's name", key="global_player_search",
    help="Searches across every player who has at least one signal computed.",
)
if player_search:
    match = usage_df[
        usage_df["display_name"].str.contains(player_search, case=False, na=False)
        | usage_df["player_name"].str.contains(player_search, case=False, na=False)
    ]
    if match.empty:
        st.info("No player matched that name in the current signal data.")
    else:
        show_cols = [
            c for c in [
                "display_name", "team", "usage_type", "total_opportunities",
                "role_volatility", "short_rest_epa_delta", "pressure_rate_delta_vs_league",
                "is_contract_year", "composite_score",
            ] if c in match.columns
        ]
        st.dataframe(match[show_cols], use_container_width=True)

st.divider()

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Player usage signal", "Team rest/travel signal", "Referee/pace signal",
    "O-line/pressure signal", "Defensive scheme signal", "Contract-year signal",
])

with tab1:
    st.subheader("Game-script conditional usage")
    search = st.text_input("Search player name", key="player_search")
    df_show = usage_df
    if search:
        df_show = df_show[
            df_show["display_name"].str.contains(search, case=False, na=False)
            | df_show["player_name"].str.contains(search, case=False, na=False)
        ]
    min_opp = st.slider("Minimum total opportunities", 0, int(usage_df["total_opportunities"].max()), 20)
    df_show = df_show[df_show["total_opportunities"] >= min_opp]
    sort_col = st.selectbox("Sort by", ["role_volatility", "total_opportunities"], key="usage_sort")
    df_show = df_show.sort_values(sort_col, ascending=False)
    st.dataframe(df_show, use_container_width=True, height=500)

with tab2:
    st.subheader("Short-rest performance delta by team")
    st.dataframe(
        short_rest_df.sort_values("short_rest_epa_delta", ascending=False),
        use_container_width=True,
    )
    st.bar_chart(short_rest_df.set_index("team")["short_rest_epa_delta"])

with tab3:
    st.subheader("Referee tendencies: pace and penalty rate")
    st.caption(
        "Historical only, by design -- referee assignments for upcoming games "
        "usually aren't public until shortly before kickoff, so this isn't fed "
        "into the composite score yet. Useful for understanding past game "
        "context (e.g. why a game ran unusually hot or cold for volume)."
    )
    if referee_df.empty:
        st.info("No referee data available this run -- see the validation report above for why.")
    else:
        st.dataframe(referee_df, use_container_width=True)
        st.bar_chart(referee_df.set_index("referee_name")["plays_delta_vs_league"])

with tab4:
    st.subheader("O-line / pressure allowed by team")
    st.caption(
        "This IS fed into the composite score above (as pressure_rate_z) -- "
        "a team whose O-line allows less pressure than league average boosts "
        "its skill-position players' composite scores."
    )
    if pressure_df.empty:
        st.info("No pressure data available this run -- see the validation report above for why.")
    else:
        st.dataframe(pressure_df, use_container_width=True)
        st.bar_chart(pressure_df.set_index("team")["pressure_rate_delta_vs_league"])

with tab5:
    st.subheader("Defensive scheme tendency: man vs. zone rate")
    st.caption(
        "Team-level only for now, not yet fed into the composite. The real "
        "value of this signal comes from matching a specific defense's "
        "man/zone rate against a specific receiver's route profile -- that "
        "player-level matchup layer is a natural next addition, not yet built."
    )
    if scheme_df.empty:
        st.info("No scheme data available this run -- see the validation report above for why.")
    else:
        st.dataframe(scheme_df, use_container_width=True)

with tab6:
    st.subheader("Contract-year status")
    st.caption(
        "Context only, NOT in the composite score -- whether a 'contract year' "
        "genuinely boosts performance is disputed in sports analytics research, "
        "so this isn't fed in as an assumed positive or negative. Use your own "
        "judgment on how much weight to give it."
    )
    if "is_contract_year" in usage_df.columns:
        contract_view_cols = [
            c for c in ["display_name", "team", "is_contract_year", "contract_end_year", "apy_cap_pct"]
            if c in usage_df.columns
        ]
        st.dataframe(
            usage_df[contract_view_cols].dropna(subset=["is_contract_year"]),
            use_container_width=True,
        )
    else:
        st.info("No contract data available this run -- see the validation report above for why.")

st.caption(
    "Player and team signals are now joined via each player's current roster "
    "team, feeding the composite score above. Tabs below still show each "
    "signal in its original, unmerged form for detailed inspection."
)
