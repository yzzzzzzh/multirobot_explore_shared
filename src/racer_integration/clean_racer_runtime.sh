#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd "${script_dir}/../.." && pwd)"

cd "${repo_dir}"
docker compose \
  -f docker-compose.yml \
  -f docker-compose.racer.yml \
  -f docker-compose.racer-live.yml \
  down --remove-orphans

# Fast-DDS shared-memory files are safe to purge only after every producer and
# consumer has stopped.  Refuse the cleanup if a robotics process survived the
# compose shutdown or belongs to another active stack.
active_processes="$(
  pgrep -af \
    'ign gazebo|gz sim|parameter_bridge|swarm_lio|ros2_racer|racer_frame_adapter|robot_state_publisher|exploration_node|traj_server' \
    || true
)"
if [[ -n "${active_processes}" ]]; then
  echo "Refusing Fast-DDS cleanup because ROS/Gazebo processes are still active:" >&2
  echo "${active_processes}" >&2
  exit 1
fi

docker run --rm --ipc=host fishbot_base:latest bash -lc \
  "find /dev/shm -maxdepth 1 -type f \\( -name 'fastrtps_*' -o -name 'sem.fastrtps_*' \\) -delete"

remaining="$(
  docker run --rm --ipc=host fishbot_base:latest bash -lc \
    "find /dev/shm -maxdepth 1 -type f \\( -name 'fastrtps_*' -o -name 'sem.fastrtps_*' \\) | wc -l"
)"
echo "RACER runtime stopped; remaining Fast-DDS IPC files: ${remaining}"
