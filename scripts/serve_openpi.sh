#!/usr/bin/env bash
# Start an openpi policy server (websocket + msgpack) for one checkpoint.
# Usage: scripts/serve_openpi.sh [config_name] [port] [checkpoint]
#   config_name  openpi training config, default pi05_droid
#   port         websocket port, default 8000 (health check: http://localhost:<port>/healthz)
#   checkpoint   checkpoint dir/URL, default gs://openpi-assets/checkpoints/<config_name>
set -euo pipefail

CONFIG="${1:-pi05_droid}"
PORT="${2:-8000}"
CKPT="${3:-gs://openpi-assets/checkpoints/$CONFIG}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OPENPI="$ROOT/third_party/openpi"

# Checkpoint cache: OPENPI_DATA_HOME from the environment or from <repo>/.env (git-ignored, machine-specific),
# else openpi's default ~/.cache/openpi.
if [ -z "${OPENPI_DATA_HOME:-}" ] && [ -f "$ROOT/.env" ]; then
    OPENPI_DATA_HOME="$(sed -n 's/^OPENPI_DATA_HOME=//p' "$ROOT/.env" | tail -1)"
fi
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-$HOME/.cache/openpi}"
# Let JAX grow GPU memory instead of grabbing 75% up front (Isaac shares this GPU).
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export GIT_LFS_SKIP_SMUDGE=1

# Keep system ROS 2 (Python 3.10) out of openpi's Python 3.11 environment.
unset PYTHONPATH AMENT_PREFIX_PATH CMAKE_PREFIX_PATH COLCON_PREFIX_PATH ROS_PACKAGE_PATH

cd "$OPENPI"
exec uv run --no-sync scripts/serve_policy.py --port "$PORT" \
    policy:checkpoint \
    --policy.config="$CONFIG" \
    --policy.dir="$CKPT"
