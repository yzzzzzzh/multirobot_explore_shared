#!/usr/bin/env bash
set -euo pipefail

# Workspace: /racer_ws inside the simulation container, ~/racer_ws when RACER is
# built natively on a robot's onboard computer (deploy/go2_real).  Override with
# RACER_WS.  `set +u` around the sourcing: ROS 1's setup scripts read variables
# that may be unset (e.g. ROS_MASTER_URI), which trips `set -u`.
RACER_WS="${RACER_WS:-}"
if [[ -z "${RACER_WS}" ]]; then
  if [[ -f /racer_ws/devel/setup.bash ]]; then
    RACER_WS=/racer_ws
  else
    RACER_WS="${HOME}/racer_ws"
  fi
fi
set +u
source /opt/ros/noetic/setup.bash
source "${RACER_WS}/devel/setup.bash"
set -u

RACER_BOT_IDS="${RACER_BOTS:-1}"
RACER_UAV_COUNT="$(awk -F, '{print NF}' <<<"${RACER_BOT_IDS}")"
RACER_TRAJECTORY_SERVER="${RACER_ENABLE_TRAJECTORY_SERVER:-false}"
RACER_TRIGGER="${RACER_AUTO_TRIGGER:-false}"
RACER_WARMUP="${RACER_MAP_WARMUP_SECONDS:-10}"
RACER_PLANNER_VEL="${RACER_PLANNER_MAX_VEL:-1.5}"
RACER_PLANNER_ACC="${RACER_PLANNER_MAX_ACC:-1.0}"
RACER_MAP_SIZE_X_VALUE="${RACER_MAP_SIZE_X:-52.0}"
RACER_MAP_SIZE_Y_VALUE="${RACER_MAP_SIZE_Y:-52.0}"
RACER_MAP_SIZE_Z_VALUE="${RACER_MAP_SIZE_Z:-52.0}"
RACER_MAP_RESOLUTION_VALUE="${RACER_MAP_RESOLUTION:-0.25}"
RACER_GROUND_HEIGHT_VALUE="${RACER_GROUND_HEIGHT:--1.0}"
RACER_BOX_MIN_X_VALUE="${RACER_BOX_MIN_X:--25.0}"
RACER_BOX_MIN_Y_VALUE="${RACER_BOX_MIN_Y:--25.0}"
RACER_BOX_MIN_Z_VALUE="${RACER_BOX_MIN_Z:-0.0}"
RACER_BOX_MAX_X_VALUE="${RACER_BOX_MAX_X:-25.0}"
RACER_BOX_MAX_Y_VALUE="${RACER_BOX_MAX_Y:-25.0}"
RACER_BOX_MAX_Z_VALUE="${RACER_BOX_MAX_Z:-50.0}"
RACER_OBSTACLES_INFLATION_VALUE="${RACER_OBSTACLES_INFLATION:-0.76}"
RACER_MIN_TARGET_EXECUTION_TIME_VALUE="${RACER_MIN_TARGET_EXECUTION_TIME:--1.0}"
RACER_EUCLIDEAN_INFLATION_VALUE="${RACER_EUCLIDEAN_INFLATION:-false}"
RACER_USE_MEASURED_REPLAN_START_VALUE="${RACER_USE_MEASURED_REPLAN_START:-false}"
RACER_MAX_RAY_LENGTH_VALUE="${RACER_MAX_RAY_LENGTH:-10.0}"
RACER_VIRTUAL_CEIL_HEIGHT_VALUE="${RACER_VIRTUAL_CEIL_HEIGHT:-50.0}"
RACER_VISUALIZATION_HEIGHT_VALUE="${RACER_VISUALIZATION_HEIGHT:-50.0}"
RACER_PARTITIONING_GRID_SIZE_VALUE="${RACER_PARTITIONING_GRID_SIZE:-20.0}"
RACER_PARTITIONING_GRID_SIZE_Z_VALUE="${RACER_PARTITIONING_GRID_SIZE_Z:-20.0}"
RACER_PARTITIONING_MIN_UNKNOWN_RATIO_VALUE="${RACER_PARTITIONING_MIN_UNKNOWN_RATIO:-0.40}"
RACER_PARTITIONING_REQUIRE_FRONTIER_VALUE="${RACER_PARTITIONING_REQUIRE_FRONTIER:-false}"
RACER_PARTITIONING_EXCLUDE_VISITED_NO_FRONTIER_VALUE="${RACER_PARTITIONING_EXCLUDE_VISITED_NO_FRONTIER:-false}"
RACER_SENSOR_MODEL_VALUE="${RACER_SENSOR_MODEL:-lidar_3d}"
RACER_LIDAR_MIN_ELEVATION_VALUE="${RACER_LIDAR_MIN_ELEVATION:--0.7}"
RACER_LIDAR_MAX_ELEVATION_VALUE="${RACER_LIDAR_MAX_ELEVATION:-0.9}"
RACER_MIN_TRANSLATION_PROGRESS_VALUE="${RACER_MIN_TRANSLATION_PROGRESS:-0.5}"
RACER_FRONTIER_MIN_CANDIDATE_CLEARANCE_VALUE="${RACER_FRONTIER_MIN_CANDIDATE_CLEARANCE:-1.0}"
RACER_FRONTIER_CLUSTER_MIN_VALUE="${RACER_FRONTIER_CLUSTER_MIN:-16}"
RACER_FRONTIER_MIN_VISIB_NUM_VALUE="${RACER_FRONTIER_MIN_VISIB_NUM:-5}"
RACER_FRONTIER_WALL_GAP_MAX_CELLS_VALUE="${RACER_FRONTIER_WALL_GAP_MAX_CELLS:-0}"
RACER_FRONTIER_WALL_GAP_MIN_SUPPORT_CELLS_VALUE="${RACER_FRONTIER_WALL_GAP_MIN_SUPPORT_CELLS:-1}"
RACER_KINO_MAX_SEARCH_TIME_VALUE="${RACER_KINO_MAX_SEARCH_TIME:-1.0}"

roscore &
ROSCORE_PID=$!
GATEWAY_PID=
TRIGGER_PID=

cleanup() {
  if [[ -n "${GATEWAY_PID}" ]]; then
    kill "${GATEWAY_PID}" 2>/dev/null || true
  fi
  if [[ -n "${TRIGGER_PID}" ]]; then
    kill "${TRIGGER_PID}" 2>/dev/null || true
  fi
  kill "${ROSCORE_PID}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

for _ in $(seq 1 30); do
  if rostopic list >/dev/null 2>&1; then
    break
  fi
  sleep 0.2
done

# RACER's trajectory server and replanning timers must advance with Gazebo,
# not wall time.  /clock is forwarded by the narrow ROS2<->ROS1 gateway.
rosparam set /use_sim_time "${RACER_USE_SIM_TIME:-true}"   # false on the real robot

# RACER_GATEWAY_BIND: 0.0.0.0 when other robots' gateways connect over the LAN.
# (Keep this comment OUT of the continuation lines: a trailing "\   # ..."
# ends the command early, runs the gateway in the foreground and roslaunch is
# never reached -- the exploration nodes silently never start.)
/usr/bin/python3 /racer_integration/ros1_racer_gateway.py \
  --bots "${RACER_BOT_IDS}" \
  --bind "${RACER_GATEWAY_BIND:-127.0.0.1}" \
  --port 47100 &
GATEWAY_PID=$!

if [[ "${RACER_TRIGGER}" == "true" ]]; then
  (
    # Six-UAV trajectory matching advances on simulation time and can take
    # several wall-clock minutes at a reduced Gazebo real-time factor.
    # Do not let a short wall timeout silently kill the one-shot trigger.
    set +e
    IFS=',' read -ra BOT_IDS <<<"${RACER_BOT_IDS}"
    ALL_ODOMETRY_READY=true
    for bot_id in "${BOT_IDS[@]}"; do
      if ! timeout 1800 rostopic echo -n 1 \
        "/racer/bot${bot_id}/odom_world" >/dev/null; then
        echo "[trigger] timed out waiting for bot${bot_id} common-frame odometry"
        ALL_ODOMETRY_READY=false
        break
      fi
    done
    if [[ "${ALL_ODOMETRY_READY}" != "true" ]]; then
      exit 1
    fi
    # Common-frame odometry can become available before every exploration FSM
    # finishes allocating its map and subscribes to the global trigger.  Wait
    # for all requested subscribers so a fast 1/2-UAV startup cannot miss the
    # one-shot trigger.
    for _ in $(seq 1 180); do
      TRIGGER_INFO="$(rostopic info /move_base_simple/goal 2>/dev/null || true)"
      ALL_TRIGGER_SUBSCRIBERS=true
      for bot_id in "${BOT_IDS[@]}"; do
        if ! grep -q "/exploration_node_${bot_id}" <<<"${TRIGGER_INFO}"; then
          ALL_TRIGGER_SUBSCRIBERS=false
          break
        fi
      done
      if [[ "${ALL_TRIGGER_SUBSCRIBERS}" == "true" ]]; then
        break
      fi
      sleep 0.5
    done
    echo "[trigger] all common odometry and exploration subscribers ready; warming map for ${RACER_WARMUP}s"
    sleep "${RACER_WARMUP}"
    # Upstream RACER expects pairwise ACVRP exchanges to seed every route
    # while all FSMs are still in WAIT_TRIGGER. Triggering after a fixed short
    # delay lets the first pair monopolize the global HGrid and leaves later
    # UAVs idle. Require a fresh non-empty official allocation for every UAV.
    if ! /usr/bin/python3 /racer_integration/ros1_wait_for_seed_allocation.py \
      --bots "${RACER_BOT_IDS}" --timeout "${RACER_SEED_TIMEOUT:-90}" --stable-seconds 2; then
      # The runtime idle-rebalance seeds an empty vehicle within a few seconds
      # of the trigger, so a missing pre-trigger seed must not block the run.
      echo "[trigger] WARNING: no pre-trigger seed allocation; triggering anyway (runtime rebalance will seed idle UAVs)"
    fi
    # A duplicate is harmless once an FSM has left WAIT_TRIGGER, and protects
    # against a subscriber reconnect exactly as the first message is sent.
    for _ in 1 2; do
      rostopic pub -1 /move_base_simple/goal geometry_msgs/PoseStamped \
        "{header: {frame_id: world}, pose: {orientation: {w: 1.0}}}"
      sleep 1
    done
    echo "[trigger] exploration trigger published"
  ) &
  TRIGGER_PID=$!
fi

exec roslaunch exploration_manager swarm_lio_lidar.launch \
  uav_count:="${RACER_UAV_COUNT}" \
  enable_trajectory_server:="${RACER_TRAJECTORY_SERVER}" \
  enable_planning_visualization:="${RACER_ENABLE_PLANNING_VISUALIZATION:-true}" \
  show_all_map:="${RACER_SHOW_ALL_MAP:-true}" \
  map_size_x:="${RACER_MAP_SIZE_X_VALUE}" \
  map_size_y:="${RACER_MAP_SIZE_Y_VALUE}" \
  map_size_z:="${RACER_MAP_SIZE_Z_VALUE}" \
  map_resolution:="${RACER_MAP_RESOLUTION_VALUE}" \
  ground_height:="${RACER_GROUND_HEIGHT_VALUE}" \
  box_min_x:="${RACER_BOX_MIN_X_VALUE}" \
  box_min_y:="${RACER_BOX_MIN_Y_VALUE}" \
  box_min_z:="${RACER_BOX_MIN_Z_VALUE}" \
  box_max_x:="${RACER_BOX_MAX_X_VALUE}" \
  box_max_y:="${RACER_BOX_MAX_Y_VALUE}" \
  box_max_z:="${RACER_BOX_MAX_Z_VALUE}" \
  obstacles_inflation:="${RACER_OBSTACLES_INFLATION_VALUE}" \
  min_target_execution_time:="${RACER_MIN_TARGET_EXECUTION_TIME_VALUE}" \
  min_target_execution_distance:="${RACER_MIN_TARGET_EXECUTION_DISTANCE:--1.0}" \
  w_dir:="${RACER_W_DIR:-1.5}" \
  empty_grid_center_first:="${RACER_EMPTY_GRID_CENTER_FIRST:-auto}" \
  frontier_exit_fixes:="${RACER_FRONTIER_EXIT_FIXES:-false}" \
  euclidean_inflation:="${RACER_EUCLIDEAN_INFLATION_VALUE}" \
  use_measured_replan_start:="${RACER_USE_MEASURED_REPLAN_START_VALUE}" \
  max_ray_length:="${RACER_MAX_RAY_LENGTH_VALUE}" \
  virtual_ceil_height:="${RACER_VIRTUAL_CEIL_HEIGHT_VALUE}" \
  visualization_truncate_height:="${RACER_VISUALIZATION_HEIGHT_VALUE}" \
  frontier_min_candidate_clearance:="${RACER_FRONTIER_MIN_CANDIDATE_CLEARANCE_VALUE}" \
  frontier_cluster_min:="${RACER_FRONTIER_CLUSTER_MIN_VALUE}" \
  frontier_min_visib_num:="${RACER_FRONTIER_MIN_VISIB_NUM_VALUE}" \
  frontier_wall_gap_max_cells:="${RACER_FRONTIER_WALL_GAP_MAX_CELLS_VALUE}" \
  frontier_wall_gap_min_support_cells:="${RACER_FRONTIER_WALL_GAP_MIN_SUPPORT_CELLS_VALUE}" \
  partitioning_grid_size:="${RACER_PARTITIONING_GRID_SIZE_VALUE}" \
  partitioning_grid_size_z:="${RACER_PARTITIONING_GRID_SIZE_Z_VALUE}" \
  partitioning_min_unknown_ratio:="${RACER_PARTITIONING_MIN_UNKNOWN_RATIO_VALUE}" \
  partitioning_exact_drone_grid_dist:="${RACER_EXACT_DRONE_GRID_DIST:-15.0}" \
  partitioning_consistency_sign:="${RACER_CONSISTENCY_SIGN:-1.0}" \
  partitioning_first_grid_bonus:="${RACER_FIRST_GRID_BONUS:-0.0}" \
  replan_time:="${RACER_REPLAN_TIME:-0.2}" \
  partitioning_require_frontier_for_relevance:="${RACER_PARTITIONING_REQUIRE_FRONTIER_VALUE}" \
  partitioning_exclude_visited_without_frontier:="${RACER_PARTITIONING_EXCLUDE_VISITED_NO_FRONTIER_VALUE}" \
  sensor_model:="${RACER_SENSOR_MODEL_VALUE}" \
  lidar_min_elevation:="${RACER_LIDAR_MIN_ELEVATION_VALUE}" \
  lidar_max_elevation:="${RACER_LIDAR_MAX_ELEVATION_VALUE}" \
  min_translation_progress:="${RACER_MIN_TRANSLATION_PROGRESS_VALUE}" \
  kino_max_search_time:="${RACER_KINO_MAX_SEARCH_TIME_VALUE}" \
  max_vel:="${RACER_PLANNER_VEL}" \
  max_acc:="${RACER_PLANNER_ACC}"
