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

python3 -c 'import matplotlib, numpy' >/dev/null || {
  echo "Python packages numpy and matplotlib are required" >&2
  exit 10
}

docker info >/dev/null
docker compose version >/dev/null
nvidia-smi >/dev/null

if ! docker info --format '{{json .Runtimes}}' | grep -q 'nvidia'; then
  echo "Docker NVIDIA runtime is not configured" >&2
  exit 11
fi

for image in \
  "${LEGGED_IMAGE:-multirobot_explore_shared-legged:local}" \
  "${RACER_IMAGE:-multirobot_explore_shared-racer-ros1:local}" \
  "${SWARM_LIO_IMAGE:-multirobot_explore_shared-swarm-lio2:local}"; do
  docker image inspect "$image" >/dev/null || {
    echo "required image is unavailable: $image" >&2
    echo "run ./build_docker_environment.sh first" >&2
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

docker compose -f docker-compose.yml config --quiet

available_kb="$(df -Pk "$REPO" | awk 'NR==2 {print $4}')"
if [ "$available_kb" -lt 1048576 ]; then
  echo "less than 1 GiB is available on the repository filesystem" >&2
  exit 14
fi

echo "preflight passed"
