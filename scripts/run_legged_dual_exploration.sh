#!/usr/bin/env bash
# Run the validated dual-Go2 RACER + Swarm-LIO2 v128 configuration.
# Usage: scripts/run_legged_dual_exploration.sh [duration_sim_s] [run_name]
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

if [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

DURATION="${1:-700}"
RUN_NAME="${2:-run_$(date +%Y%m%d_%H%M%S)}"
COMPOSE=(docker compose -f compose.run1-v128.yml)
CONTAINER_RUN_DIR="/tmp/${RUN_NAME}"
OUTPUT_DIR="$REPO/runs/$RUN_NAME"
containers_started=false

cleanup_on_exit() {
  status=$?
  trap - EXIT INT TERM
  if [ "$containers_started" = true ]; then
    "${COMPOSE[@]}" stop --timeout 30 \
      racer_controller racer_ros1 swarm_lio2 gazebo >/dev/null 2>&1 || true
  fi
  exit "$status"
}
trap cleanup_on_exit EXIT INT TERM

if ! [[ "$DURATION" =~ ^[0-9]+([.][0-9]+)?$ ]] || [ "$DURATION" = "0" ]; then
  echo "duration_sim_s must be a positive number" >&2
  exit 2
fi
if ! [[ "$RUN_NAME" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "run_name may contain only letters, digits, dot, underscore, and dash" >&2
  exit 2
fi
if [ -e "$OUTPUT_DIR/run.npz" ] || [ -e "$OUTPUT_DIR/run.summary.json" ]; then
  echo "refusing to overwrite completed run: $OUTPUT_DIR" >&2
  exit 2
fi

bash scripts/preflight.sh
mkdir -p "$OUTPUT_DIR"

# Fixed container names are part of the recording/post-processing interface.
# Remove only stopped containers with those exact names. Never interrupt a
# running experiment implicitly.
for name in fishbot_gazebo swarm_lio2_ros2 racer_ros1 racer_controller; do
  if docker container inspect "$name" >/dev/null 2>&1; then
    state="$(docker inspect -f '{{.State.Running}}' "$name")"
    if [ "$state" = "true" ]; then
      echo "container $name is already running; stop it explicitly first" >&2
      exit 3
    fi
    docker container rm "$name" >/dev/null
  fi
done

swarm_image="${SWARM_LIO_IMAGE:-swarm-lio2-ros2:run1-v128-validated}"
docker run --rm --ipc=host --entrypoint bash "$swarm_image" -lc \
  "find /dev/shm -maxdepth 1 -type f \( -name 'fastrtps_*' -o -name 'sem.fastrtps_*' \) -delete"

"${COMPOSE[@]}" up -d --force-recreate gazebo swarm_lio2 racer_ros1 racer_controller
containers_started=true

echo "waiting for Swarm-LIO2 map initialization"
deadline=$((SECONDS + 600))
while true; do
  ikd_count="$(docker logs --since 30s swarm_lio2_ros2 2>&1 | grep -c 'ikd-tree size' || true)"
  if [ "$ikd_count" -gt 0 ]; then
    break
  fi
  if (( SECONDS >= deadline )); then
    echo "timed out waiting for Swarm-LIO2 map initialization" >&2
    exit 4
  fi
  sleep 5
done

for bot in 1 2; do
  ready=false
  for _ in $(seq 1 40); do
    if docker exec swarm_lio2_ros2 bash -lc \
      "source /opt/ros/humble/setup.bash; source /opt/swarm_lio_ws/install/setup.bash; timeout 15 ros2 topic echo --once /bot${bot}/lidar_slam/odom >/dev/null 2>&1"; then
      ready=true
      echo "bot${bot} LIO odometry ready"
      break
    fi
    sleep 3
  done
  if [ "$ready" != true ]; then
    echo "timed out waiting for bot${bot} LIO odometry" >&2
    exit 5
  fi
done

sleep 10
echo "releasing both Go2 gaits"
for bot in 1 2; do
  released=false
  for _ in $(seq 1 6); do
    if docker exec fishbot_gazebo bash -lc \
      "source /opt/ros/humble/setup.bash; timeout 20 ros2 param set /bot${bot}/twist_to_control_input auto_trot true" \
      | grep -qi 'successful'; then
      released=true
      break
    fi
    sleep 3
  done
  if [ "$released" != true ]; then
    echo "failed to release bot${bot} gait" >&2
    exit 6
  fi
done

docker exec swarm_lio2_ros2 mkdir -p "$CONTAINER_RUN_DIR"
docker exec -d swarm_lio2_ros2 bash -lc \
  "source /opt/ros/humble/setup.bash; source /opt/swarm_lio_ws/install/setup.bash; exec /usr/bin/python3 /racer_integration/ros2_exploration_recorder.py --bots 1,2 --duration '$DURATION' --duration-basis sim --platform quadruped --wait-for-tracking --startup-timeout 1800 --progress-interval 30 --divergence-error 5.0 --cloud-bounds -50 -25 0 50 25 1 --output-prefix '$CONTAINER_RUN_DIR/run' --ros-args -p use_sim_time:=true > '$CONTAINER_RUN_DIR/recorder.log' 2>&1"

echo "recorder started: duration=${DURATION} simulation seconds"
echo "host output: $OUTPUT_DIR"

bash scripts/monitor_dual_run.sh "$CONTAINER_RUN_DIR" "$OUTPUT_DIR" &
monitor_pid=$!

set +e
bash scripts/finish_dual_run.sh "$CONTAINER_RUN_DIR" "$OUTPUT_DIR" "go2_dual_atrium_v128_${DURATION}s" "$DURATION"
finish_status=$?
wait "$monitor_pid"
monitor_status=$?
set -e

"${COMPOSE[@]}" stop --timeout 30 racer_controller racer_ros1 swarm_lio2 gazebo >/dev/null 2>&1 || true

if [ "$finish_status" -ne 0 ] || [ "$monitor_status" -ne 0 ]; then
  echo "run or post-processing failed; inspect $OUTPUT_DIR" >&2
  exit 7
fi

echo "completed: $OUTPUT_DIR"
