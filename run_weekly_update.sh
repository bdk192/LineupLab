#!/usr/bin/env bash
# run_weekly_update.sh
#
# Runs ON the VPS itself (unlike run_historical_build.sh), on a normal
# schedule via cron. Processes ONE week at a time, so it's light enough to
# fit comfortably in the permanent 2GB size -- no resizing required.
#
# SETUP (one-time, on the VPS):
#   mkdir -p ~/lineuplab && cd ~/lineuplab
#   # copy the evaluation/, evaluation_app.py, Dockerfile,
#   # requirements-evaluation.txt, and this script here
#   docker build -t lineuplab-eval .
#   crontab -e
#   # add a line like:
#   # 0 10 * * 3 /home/YOUR_USER/lineuplab/run_weekly_update.sh >> /home/YOUR_USER/lineuplab/weekly.log 2>&1
#   # (10am every Wednesday, once injury reports for the week start rolling in)

set -euo pipefail
cd "$(dirname "$0")"

TARGET_SEASON="${1:-$(date +%Y)}"
TARGET_WEEK="${2:?Usage: $0 [season] <week>}"

echo "$(date): Running weekly update for season=$TARGET_SEASON week=$TARGET_WEEK"

docker run --rm \
  --memory="1g" \
  -v "$(pwd)/evaluation/data:/app/evaluation/data" \
  lineuplab-eval \
  python -c "
from evaluation.build_historical_signal_dataset import load_season_data
from evaluation.historical_signals import compute_historical_signal_snapshot
import pandas as pd
from pathlib import Path

season, week = $TARGET_SEASON, $TARGET_WEEK
pbp, schedule, participation = load_season_data(season)
snap = compute_historical_signal_snapshot(pbp, schedule, participation, season, week)

out_path = Path('evaluation/data/weekly_snapshots.parquet')
if out_path.exists():
    existing = pd.read_parquet(out_path)
    existing = existing[
        ~((existing['target_season'] == season) & (existing['target_week'] == week))
    ]
    snap = pd.concat([existing, snap], ignore_index=True)

snap.to_parquet(out_path, index=False)
print(f'Appended {len(snap)} total rows (this week + prior history) to {out_path}')
"

echo "$(date): Done."
