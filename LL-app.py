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
    import nfl_data_py as nfl
    return nfl.import_pbp_data(seasons, downcast=True)


@st.cache_data(ttl=86400, show_spinner="Loading schedule data...")
def load_schedule(seasons):
    import nfl_data_py as nfl
    return nfl.import_schedules(seasons)


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


def build_dashboard_data(usage_df: pd.DataFrame, schedule_df: pd.DataFrame):
    warnings = []
    warnings.extend(validate_signal("role_volatility", usage_df["role_volatility"], (0.0, 0.5)))

    short_rest = schedule_df[schedule_df["rest_bucket"] == "short_rest"][["team", "avg_epa_delta"]]
    short_rest = short_rest.rename(columns={"avg_epa_delta": "short_rest_epa_delta"})
    warnings.extend(validate_signal("short_rest_epa_delta", short_rest["short_rest_epa_delta"], (-0.3, 0.3)))

    usage_df = usage_df.copy()
    usage_df["role_volatility_z"] = zscore(usage_df["role_volatility"])
    short_rest["short_rest_epa_delta_z"] = zscore(short_rest["short_rest_epa_delta"])

    # Composite currently uses only short_rest_epa_delta (higher = better);
    # role_volatility is a confidence signal, not part of the score, per
    # the earlier design decision.
    usage_df["composite_score"] = np.nan  # placeholder until player->team join exists
    return usage_df, short_rest, warnings


# ---------------------------------------------------------------------------
# APP
# ---------------------------------------------------------------------------
st.title("Fantasy Signal Dashboard")
st.caption("Every raw signal is shown alongside any composite score — nothing is hidden inside a black-box number.")

pbp = load_pbp(SEASONS)
schedule = load_schedule(SEASONS)
usage_df = compute_usage_by_game_script(pbp)
schedule_df = compute_schedule_effects(schedule, pbp)
usage_df, short_rest_df, warnings = build_dashboard_data(usage_df, schedule_df)

with st.expander(f"Validation report ({len(warnings)} issue(s))", expanded=len(warnings) > 0):
    if warnings:
        for w in warnings:
            st.warning(w)
    else:
        st.success("All signals passed validation checks cleanly.")

tab1, tab2 = st.tabs(["Player usage signal", "Team rest/travel signal"])

with tab1:
    st.subheader("Game-script conditional usage")
    search = st.text_input("Search player name", key="player_search")
    df_show = usage_df
    if search:
        df_show = df_show[df_show["player_name"].str.contains(search, case=False, na=False)]
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

st.caption(
    "Note: player-level and team-level signals aren't merged into one row yet "
    "(needs a player-to-team mapping) -- shown as separate tabs for now."
)
