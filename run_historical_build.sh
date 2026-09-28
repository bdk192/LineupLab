#!/usr/bin/env bash
# run_historical_build.sh
#
# Runs on your LOCAL machine (Mac), NOT on the VPS -- resizing a droplet
# requires powering it off, so the script doing the resizing can't be
# running on the box being resized.
#
# What this does, in order:
#   1. Shuts the droplet down (required before a RAM/CPU resize)
#   2. Resizes it UP temporarily (e.g. 2GB -> 4GB)
#   3. Powers it back on
#   4. SSHes in and runs the heavy historical-build job inside a
#      memory-capped Docker container
#   5. Resizes it back DOWN to your normal permanent size
#   6. Powers it back on
#
# PREREQUISITES (one-time setup on your Mac):
#   brew install doctl
#   doctl auth init                     # paste a DigitalOcean API token
#   Make sure you can already `ssh <user>@<droplet-ip>` without issues.
#
# USAGE:
#   ./run_historical_build.sh <droplet-id> <droplet-ip> <ssh-user>
#
# Find your droplet ID with:
#   doctl compute droplet list

set -euo pipefail

DROPLET_ID="${1:?Usage: $0 <droplet-id> <droplet-ip> <ssh-user>}"
DROPLET_IP="${2:?Usage: $0 <droplet-id> <droplet-ip> <ssh-user>}"
SSH_USER="${3:?Usage: $0 <droplet-id> <droplet-ip> <ssh-user>}"

NORMAL_SIZE="s-1vcpu-2gb"   # your permanent baseline size
BIG_SIZE="s-2vcpu-4gb"      # temporary size for the heavy job
DOCKER_MEM_LIMIT="3g"       # leave some headroom under the 4GB ceiling

echo "== Step 1/6: Shutting down droplet (required for resize) =="
doctl compute droplet-action shutdown "$DROPLET_ID" --wait

echo "== Step 2/6: Resizing UP to $BIG_SIZE =="
doctl compute droplet-action resize "$DROPLET_ID" --size "$BIG_SIZE" --resize-disk=false --wait

echo "== Step 3/6: Powering back on =="
doctl compute droplet-action power-on "$DROPLET_ID" --wait
echo "Waiting 20s for SSH to become available..."
sleep 20

echo "== Step 4/6: Running the heavy historical build over SSH =="
ssh "${SSH_USER}@${DROPLET_IP}" bash -s <<EOF
set -e
cd ~/lineuplab
docker build -t lineuplab-eval .
docker run --rm \
  --memory="${DOCKER_MEM_LIMIT}" \
  -v \$(pwd)/evaluation/data:/app/evaluation/data \
  lineuplab-eval \
  python -m evaluation.build_historical_signal_dataset
EOF

echo "== Step 5/6: Resizing back DOWN to $NORMAL_SIZE =="
doctl compute droplet-action shutdown "$DROPLET_ID" --wait
doctl compute droplet-action resize "$DROPLET_ID" --size "$NORMAL_SIZE" --resize-disk=false --wait

echo "== Step 6/6: Powering back on =="
doctl compute droplet-action power-on "$DROPLET_ID" --wait

echo "Done. Results are on the VPS at ~/lineuplab/evaluation/data/"
echo "You were only charged the price difference for the time spent at $BIG_SIZE."
