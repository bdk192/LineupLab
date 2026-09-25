"""
Streamlit test harness for the leakage-safe historical LineupLab signals.

Run this as the Streamlit entrypoint on Streamlit Community Cloud if local
developer tools are unavailable:

    streamlit run evaluation_app.py

Or temporarily set Streamlit Cloud's Main file path to evaluation_app.py.

This app is intentionally small: choose one season/week, build the signals as
they would have existed BEFORE that week, and inspect the result.
"""

import streamlit as st
import pandas as pd

st.set_page_config(page_title="LineupLab Historical Signal Test", layout="wide")
st.title("LineupLab — Historical Signal Test")
st.caption(
    "Leakage-safe test harness: every signal is calculated using information "
    "available before the selected target week."
)

season = st.number_input("Target season", min_value=2022, max_value=2026, value=2025, step=1)
week = st.number_input("Target week", min_value=1, max_value=22, value=10, step=1)

@st.cache_data(ttl=86400, show_spinner="Loading NFL data...")
def load_data(season):
    import nflreadpy as nfl

    pbp_needed = [
        "game_id", "play_id", "play_type", "score_differential",
        "season", "week", "rusher_player_id", "rusher_player_name",
        "receiver_player_id", "receiver_player_name", "epa", "posteam", "defteam",
    ]
    pbp = nfl.load_pbp([season])
    pbp = pbp.select([c for c in pbp_needed if c in pbp.columns]).to_pandas()

    schedule = nfl.load_schedules([season]).to_pandas()

    try:
        part = nfl.load_participation([season])
        part = part.select([
            c for c in [
                "game_id", "nflverse_game_id", "play_id",
                "was_pressure", "time_to_throw", "defense_man_zone_type"
            ] if c in part.columns
        ]).to_pandas()
    except ValueError:
        part = pd.DataFrame()

    return pbp, schedule, part

if st.button("Build historical snapshot", type="primary"):
    from evaluation.historical_signals import compute_historical_signal_snapshot

    pbp, schedule, participation = load_data(int(season))

    with st.spinner("Calculating pre-week signals..."):
        result = compute_historical_signal_snapshot(
            pbp,
            schedule,
            participation,
            int(season),
            int(week),
        )

    if result.empty:
        st.error("No signal rows were produced. Check the selected season/week and data availability.")
    else:
        st.success(
            f"Built {len(result):,} signal rows for Week {int(week)}. "
            "These rows use only data from earlier weeks."
        )

        cols = [
            c for c in [
                "player_id", "player_name", "team", "opponent", "usage_type",
                "total_opportunities", "role_volatility",
                "short_rest_epa_delta", "pressure_rate",
                "pressure_rate_delta_vs_league",
                "def_epa_allowed", "historical_composite",
            ] if c in result.columns
        ]
        st.dataframe(
            result[cols].sort_values(
                "historical_composite", ascending=False, na_position="last"
            ),
            use_container_width=True,
            height=600,
        )

        st.download_button(
            "Download snapshot CSV",
            result.to_csv(index=False),
            file_name=f"lineuplab_signals_{int(season)}_week_{int(week)}.csv",
            mime="text/csv",
        )

        st.info(
            "Next step: merge this snapshot with the actual Week "
            f"{int(week)} player outcomes and compare predictions against "
            "simple baselines."
        )
