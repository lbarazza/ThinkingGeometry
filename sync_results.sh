#!/usr/bin/env bash
# sync_results.sh — pull results/ from a remote GPU to this machine periodically.
#
# Run this on your LOCAL Mac, not on the GPU instance.
#
# Usage:
#   ./sync_results.sh <remote> [local_dir] [interval_secs]
#
#   <remote>        SSH path to the project root on the GPU, e.g.
#                   user@203.0.113.5:/home/user/ThinkingGeometry
#   [local_dir]     Local directory to sync into (default: ./remote_results)
#   [interval_secs] Seconds between syncs (default: 60)
#
# Keep your Mac awake during a long sync session:
#   caffeinate -s ./sync_results.sh user@host:/path/to/project
#
# Press Ctrl-C to stop.

set -euo pipefail

REMOTE="${1:?Usage: $0 <user@host:/path/to/project> [local_dir] [interval_secs]}"
LOCAL_DIR="${2:-./remote_results}"
INTERVAL="${3:-60}"

mkdir -p "$LOCAL_DIR"

echo "Syncing  $REMOTE/results/  ->  $LOCAL_DIR/"
echo "Interval: ${INTERVAL}s   (Ctrl-C to stop)"
echo

while true; do
    echo "[$(date '+%H:%M:%S')] syncing..."
    rsync -avz --progress "$REMOTE/results/" "$LOCAL_DIR/"
    echo "[$(date '+%H:%M:%S')] done. Next sync in ${INTERVAL}s."
    sleep "$INTERVAL"
done
