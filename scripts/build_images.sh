#!/usr/bin/env bash
# Build the complete runtime environment from the source in this repository.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

# Accept the same explicit proxy variables as fishbot_multirobot_sim_shared,
# while preserving standard Docker proxy variables when they are already set.
HTTP_PROXY="${FISHBOT_HTTP_PROXY:-${HTTP_PROXY:-}}"
HTTPS_PROXY="${FISHBOT_HTTPS_PROXY:-${HTTPS_PROXY:-}}"
NO_PROXY="${FISHBOT_NO_PROXY:-${NO_PROXY:-localhost,127.0.0.1,::1}}"
export HTTP_PROXY HTTPS_PROXY NO_PROXY

proxy_args=()
for name in HTTP_PROXY HTTPS_PROXY NO_PROXY http_proxy https_proxy no_proxy; do
  if [ -n "${!name:-}" ]; then
    proxy_args+=(--build-arg "$name=${!name}")
  fi
done

docker build "${proxy_args[@]}" --network=host \
  -t multirobot_explore_shared-base:local \
  -f docker/Dockerfile .

docker build "${proxy_args[@]}" --network=host \
  --build-arg BASE_IMAGE=multirobot_explore_shared-base:local \
  -t multirobot_explore_shared-gazebo:local \
  -f docker/Dockerfile.gazebo .

docker build "${proxy_args[@]}" --network=host \
  --build-arg BASE_IMAGE=multirobot_explore_shared-gazebo:local \
  -t multirobot_explore_shared-legged:local \
  -f docker/Dockerfile.legged .

docker build "${proxy_args[@]}" --network=host \
  -t multirobot_explore_shared-racer-ros1:local \
  -f src/racer_integration/Dockerfile.ros1 .

docker build "${proxy_args[@]}" --network=host \
  -t multirobot_explore_shared-swarm-lio2:local \
  -f src/Swarm-LIO2-ROS2-Docker/docker/Dockerfile.ros2 \
  src/Swarm-LIO2-ROS2-Docker

echo "source-build images completed"
echo "Run: bash scripts/preflight.sh"
