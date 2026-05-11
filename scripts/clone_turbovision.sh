#!/usr/bin/env bash
# Clone the official Score TurboVision repo as a read-only reference.
# Not vendored into this repo because:
#   1. it's external code under Score Technologies' license,
#   2. it's large, and
#   3. our eval/verify_against_official.py imports from it directly.
#
# Usage:  bash scripts/clone_turbovision.sh
set -euo pipefail

REPO_DIR="turbovision"
REPO_URL="https://github.com/score-technologies/turbovision.git"

if [[ -d "$REPO_DIR/.git" ]]; then
  echo "[clone_turbovision] $REPO_DIR already cloned; pulling latest."
  git -C "$REPO_DIR" pull --ff-only
else
  echo "[clone_turbovision] cloning $REPO_URL into $REPO_DIR ..."
  git clone "$REPO_URL" "$REPO_DIR"
fi

echo "[clone_turbovision] done. last commit:"
git -C "$REPO_DIR" log -1 --format='%h %ad %s' --date=short
