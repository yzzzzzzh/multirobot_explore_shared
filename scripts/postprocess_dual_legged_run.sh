#!/usr/bin/env bash
# Collect a finished two-robot exploration run into an artifact directory and
# compute the planar coverage / SLAM metrics (v124 evaluation pipeline).
# Usage: scripts/postprocess_dual_legged_run.sh <container_run_dir> <artifact_dir> [layout_json]
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="${1:?container run dir, e.g. /tmp/go2dual}"
RUN="${2:?artifact dir}"
LAYOUT="${3:-src/gazebo_sim/worlds/teaching_building_atrium.layout.json}"
cd "$REPO"
mkdir -p "$RUN/ros1_logs" "$RUN/ros2_logs"
docker cp "swarm_lio2_ros2:$SRC/." "$RUN/" 2>/dev/null
timeout 60 docker exec racer_ros1 bash -lc 'source /racer_ws/devel/setup.bash >/dev/null 2>&1; python3 /racer_integration/ros1_final_snapshot.py --bots 1,2 --state-timeout 10 --service-timeout 15' 2>/dev/null | grep RACER_FINAL_SNAPSHOT > "$RUN/final_snapshot.txt" || echo "snapshot failed"
docker exec racer_ros1 tar -C /root/.ros/log/latest -cf - . 2>/dev/null | tar -xf - -C "$RUN/ros1_logs" 2>/dev/null
docker exec swarm_lio2_ros2 tar -C /root/.ros/log -cf - . 2>/dev/null | tar -xf - -C "$RUN/ros2_logs" 2>/dev/null
docker logs racer_controller > "$RUN/ros2_logs/racer_controller.log" 2>&1
docker logs racer_ros1 > "$RUN/ros1_logs/racer_ros1_container.log" 2>&1
docker logs swarm_lio2_ros2 > "$RUN/ros2_logs/swarm_lio2_container.log" 2>&1
docker logs fishbot_gazebo > "$RUN/ros2_logs/gazebo_container.log" 2>&1
if [ -f "$RUN/run.npz" ]; then
  python3 src/racer_integration/analyze_planar_exploration.py --run "$RUN/run.npz" --layout "$LAYOUT" --output "$RUN/planar_metrics" 2>&1 | tail -1
  python3 src/racer_integration/analyze_exploration_run.py "$RUN/run.summary.json" "$RUN/run.npz" "$RUN/ros1_logs/rosout.log" --output "$RUN/run.analysis.json" 2>&1 | tail -1
else
  echo "run.npz missing in $RUN (recorder still running?)"
fi
echo "postprocess-done: $RUN"
