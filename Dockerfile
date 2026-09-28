# LineupLab evaluation environment -- isolated from other services on the VPS.
# Build once; used by both the heavy (resize-up) historical job and the
# lightweight recurring weekly job, just with different commands/memory caps.

FROM python:3.12-slim

WORKDIR /app

# Only what the evaluation package actually needs -- keep this lean since
# it's running on a resource-constrained box.
COPY requirements-evaluation.txt .
RUN pip install --no-cache-dir -r requirements-evaluation.txt

COPY evaluation/ ./evaluation/
COPY evaluation_app.py .

# No CMD here on purpose -- the heavy job and the weekly job run different
# module entrypoints (see run_historical_build.sh and run_weekly_update.sh).
# Data/output lives under /app/evaluation/data, which should be bind-mounted
# from the host so results persist across container runs.
