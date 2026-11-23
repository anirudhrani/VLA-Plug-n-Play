#!/usr/bin/env bash
# Start the vla_bridge ROS 2 node (system ROS 2 Humble, Python 3.10, own venv with openpi_client).
# Usage: scripts/run_bridge.sh [--model pi05_droid] [--host localhost] [--port 8000] [ros args...]
# ROS_DOMAIN_ID / RMW_IMPLEMENTATION are inherited if set (the VLA Sim panel passes Isaac's), else 0 / FastDDS.
set -eo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WS="$ROOT/ros2_ws"

unset PYTHONPATH AMENT_PREFIX_PATH CMAKE_PREFIX_PATH COLCON_PREFIX_PATH LD_LIBRARY_PATH
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}"
# shellcheck disable=SC1091
source /opt/ros/humble/setup.bash
export PYTHONPATH="$WS/src/vla_bridge:$PYTHONPATH"
export PYTHONUNBUFFERED=1

exec "$WS/.venv/bin/python" -m vla_bridge.vla_node "$@"
