#!/usr/bin/env bash
# Source-build alternatives. For result reproduction, prefer the immutable
# validated images documented in config/provenance.json.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

proxy_args=()
for name in HTTP_PROXY HTTPS_PROXY NO_PROXY http_proxy https_proxy no_proxy; do
  if [ -n "${!name:-}" ]; then
    proxy_args+=(--build-arg "$name=${!name}")
  fi
done

docker build "${proxy_args[@]}" --network=host \
  -t fishbot_base:run1-v128-source \
  -f docker/Dockerfile .

docker build "${proxy_args[@]}" --network=host \
  --build-arg BASE_IMAGE=fishbot_base:run1-v128-source \
  -t fishbot_multirobot_sim-gazebo:run1-v128-source \
  -f docker/Dockerfile.gazebo .

docker build "${proxy_args[@]}" --network=host \
  --build-arg BASE_IMAGE=fishbot_multirobot_sim-gazebo:run1-v128-source \
  -t fishbot_multirobot_sim-legged:run1-v128-source \
  -f docker/Dockerfile.legged .

docker build "${proxy_args[@]}" --network=host \
  -t fishbot_multirobot_sim-racer_ros1:run1-v128-source \
  -f src/racer_integration/Dockerfile.ros1 .

docker build "${proxy_args[@]}" --network=host \
  -t swarm-lio2-ros2:run1-v128-source \
  -f src/Swarm-LIO2-ROS2-Docker/docker/Dockerfile.ros2 \
  src/Swarm-LIO2-ROS2-Docker

echo "source-build images completed"
echo "These images are rebuilds, not byte-identical substitutes for the validated image IDs."
