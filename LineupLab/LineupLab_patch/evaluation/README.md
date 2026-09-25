# LineupLab Evaluation Engine — Phase 1

This folder starts the historical evaluation layer without changing the
existing Streamlit dashboard.

## Why this exists

The core LineupLab research question is:

> When LineupLab says something is different about a player this week, does
> that information actually improve our prediction of what happens next?

The first step is establishing a fair, point-in-time baseline.

## Current phase

The code here:

1. Loads weekly offensive player stats from `nflreadpy`.
2. Keeps regular-season QB/RB/WR/TE player-weeks.
3. Creates simple pre-week baselines:
   - previous game PPR
   - last 4 games PPR average
   - season-to-date PPR average
   - last 4 games opportunity average
4. Stores the actual next-game PPR outcome separately.
5. Evaluates MAE, RMSE, bias, Pearson correlation, and Spearman correlation.
6. Supports signal columns later without changing the outcome logic.

`nflreadpy.load_player_stats(..., summary_level="week")` provides weekly
player-level statistics, including fantasy points and opportunity variables.

## Run

Install the existing project requirements, then:

```bash
python -m evaluation.build_player_week_dataset --seasons 2022 2023 2024 2025
python -m evaluation.backtest
```

The output dataset is written to:

```text
evaluation/data/player_week_baseline.parquet
```

## Important methodological rule

For week N, every baseline is calculated using only games through week N-1.

Do not use current-week or future-week data to construct a prediction.

## What this does NOT do yet

It does not yet recreate every current LineupLab signal historically.

That is intentional.

The next phase should refactor the existing signal functions so they can accept
a `reference_season` and `reference_week`, then generate historical signal
values keyed by:

```text
season
week
player_id
```

Those values can then be merged into this player-week dataset and tested for
incremental predictive value.

## Planned next step

Compare:

```text
Baseline
Baseline + one LineupLab signal
Baseline + all current LineupLab signals
```

using walk-forward/out-of-sample evaluation before changing signal weights.
