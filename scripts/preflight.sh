#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

if [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

for command_name in docker python3 ffmpeg ffprobe sha256sum; do
  command -v "$command_name" >/dev/null || {
    echo "missing required command: $command_name" >&2
    exit 10
  }
done

docker info >/dev/null
docker compose version >/dev/null
nvidia-smi >/dev/null

if ! docker info --format '{{json .Runtimes}}' | grep -q 'nvidia'; then
  echo "Docker NVIDIA runtime is not configured" >&2
  exit 11
fi

for image in \
  "${LEGGED_IMAGE:-fishbot_multirobot_sim-legged:run1-v128-historical}" \
  "${RACER_IMAGE:-fishbot_multirobot_sim-racer_ros1:v128-dual-first-grid-bonus}" \
  "${SWARM_LIO_IMAGE:-swarm-lio2-ros2:run1-v128-validated}"; do
  docker image inspect "$image" >/dev/null || {
    echo "required image is unavailable: $image" >&2
    echo "pull it from GHCR or restore the validated local image" >&2
    exit 12
  }
done

for file in \
  src/gazebo_sim/worlds/teaching_building_atrium_legged.world \
  src/gazebo_sim/worlds/teaching_building_atrium.layout.json \
  src/legged/qrc/go2_description/config/himloco/himloco.pt \
  src/racer_adapter/config/racer_adapter_quadruped_go2_dual.yaml \
  src/racer_integration/ros2_exploration_recorder.py; do
  [ -f "$file" ] || {
    echo "required file is missing: $file" >&2
    exit 13
  }
done

docker compose -f compose.run1-v128.yml config --quiet

available_kb="$(df -Pk "$REPO" | awk 'NR==2 {print $4}')"
if [ "$available_kb" -lt 1048576 ]; then
  echo "less than 1 GiB is available on the repository filesystem" >&2
  exit 14
fi

echo "preflight passed"
