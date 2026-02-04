#!/bin/bash

source ./.sync_config

EXCLUDES=(
  ".git"
  ".sync_config"
  ".sync_config.sample"
  "__pycache__"
  "*.pyc"
  "node_modules"
  "venv"
  ".DS_Store"
  "*.log"
  ".env"
  "checkpoints/"
)

EXCLUDE_FLAGS=""
for pattern in "${EXCLUDES[@]}"; do
  EXCLUDE_FLAGS="$EXCLUDE_FLAGS --exclude=$pattern"
done

echo "Syncing to ${REMOTE_HOST}..."

rsync -az --progress ./ $EXCLUDE_FLAGS "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}/"

rsync -az --progress  "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}/checkpoints/" ./checkpoints/

echo "✓ Sync complete"
