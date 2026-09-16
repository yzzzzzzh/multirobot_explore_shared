#!/bin/bash
set -e
source /opt/ros/humble/setup.bash
[ -f /fishbot_ws/install/setup.bash ] && source /fishbot_ws/install/setup.bash
[ -f /legged_ws/install/setup.bash ] && source /legged_ws/install/setup.bash
export LD_LIBRARY_PATH=/opt/libtorch/lib:${LD_LIBRARY_PATH}
exec "$@"
