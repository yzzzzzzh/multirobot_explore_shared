#!/bin/bash
set -e

if [ -f "/opt/ros/humble/setup.bash" ]; then
  source /opt/ros/humble/setup.bash
fi

if [ -f "${COLCON_WS:-/fishbot_ws}/install/setup.bash" ]; then
  source "${COLCON_WS:-/fishbot_ws}/install/setup.bash"
fi

exec "$@"
