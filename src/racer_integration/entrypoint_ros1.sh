#!/usr/bin/env bash
set -e

source /opt/ros/noetic/setup.bash
if [[ -f /racer_ws/devel/setup.bash ]]; then
  source /racer_ws/devel/setup.bash
fi

exec "$@"
