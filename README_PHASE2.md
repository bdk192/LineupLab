# Phase 2 — Leakage-Safe Historical Signals

This adds a historical signal layer without changing the existing `app.py`.

## What it does

For a target season/week, it calculates the current LineupLab-style signals using only data from weeks **before** the target week:

- Game-script role volatility
- Historical short-rest effect
- Team pressure allowed
- Opponent defensive EPA allowed
- Target-week opponent
- A preliminary historical composite using the same general directionality as the current composite

The key difference from the live app is the cutoff. The live app currently derives recency from the latest available data, while this module derives it relative to the requested target week.

## Streamlit testing

You do NOT need local Python/developer tools to test this.

On Streamlit Community Cloud, temporarily set the app's main file to:

`evaluation_app.py`

The app will load one season, let you choose a target week, calculate the historical snapshot, and provide a CSV download.

Alternatively, if your current Streamlit deployment lets you add the files to the repo, you can use the same deployment workflow you already use.

## Important limitations

This is intentionally the first historical layer, not the finished evaluation engine.

1. It does not yet reproduce every signal in `app.py`.
2. The injury/practice signal is not included yet because the first priority is establishing the historical cutoff architecture cleanly.
3. Scheme/referee/contract-year are not yet included in the historical composite.
4. The current role-volatility definition is preserved rather than redesigned.
5. The historical composite is a research snapshot, not a claim that the current equal-weight composite is optimal.

## Next step

Merge these historical signal snapshots with the actual player-week outcome dataset from Phase 1.

Then run:

Baseline
vs.
Baseline + each signal
vs.
Baseline + historical composite

That will tell us whether LineupLab's existing information actually improves prediction.
