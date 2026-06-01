#!/usr/bin/env bash
# sync_results.sh — pull results/ from a remote GPU to this machine periodically.
#
# Run this on your LOCAL Mac, not on the GPU instance.
#
# Usage:
#   ./sync_results.sh <remote> [local_dir] [interval_secs] [ssh_opts]
#
#   <remote>        SSH path to the project root on the GPU, e.g.
#                   user@203.0.113.5:/home/user/ThinkingGeometry
#   [local_dir]     Local directory to sync into (default: ./remote_results)
#   [interval_secs] Seconds between syncs (default: 60)
#   [ssh_opts]      Extra SSH options, e.g. "-p 17037 -i ~/.ssh/id_ed25519_runpod"
#
# Examples:
#   # Standard (default port, default key):
#   caffeinate -s ./sync_results.sh user@host:/path/to/project
#
#   # RunPod with custom port and key:
#   caffeinate -s ./sync_results.sh "root@64.247.201.49:/workspace/ThinkingGeometry" ./remote_results 60 "-p 17037 -i ~/.ssh/id_ed25519_runpod"
#
# Keep your Mac awake during a long sync session by prefixing with caffeinate -s
#
# Press Ctrl-C to stop.

set -euo pipefail

REMOTE="${1:?Usage: $0 <user@host:/path/to/project> [local_dir] [interval_secs] [ssh_opts]}"
LOCAL_DIR="${2:-./remote_results}"
INTERVAL="${3:-60}"
SSH_OPTS="${4:-}"

mkdir -p "$LOCAL_DIR"

echo "Syncing  $REMOTE/results/  ->  $LOCAL_DIR/"
echo "Interval: ${INTERVAL}s   (Ctrl-C to stop)"
echo

while true; do
    echo "[$(date '+%H:%M:%S')] syncing..."
    if [ -n "$SSH_OPTS" ]; then
        rsync -avz --progress -e "ssh $SSH_OPTS" "$REMOTE/results/" "$LOCAL_DIR/"
    else
        rsync -avz --progress "$REMOTE/results/" "$LOCAL_DIR/"
    fi
    echo "[$(date '+%H:%M:%S')] done. Next sync in ${INTERVAL}s."
    sleep "$INTERVAL"
done
