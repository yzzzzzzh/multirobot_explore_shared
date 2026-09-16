#!/usr/bin/env bash
# Tag the three validated local images before pushing to GHCR.
# Usage: scripts/tag_images_for_ghcr.sh [github_owner]
set -euo pipefail

OWNER="${1:-yzzzzzzh}"
OWNER="${OWNER,,}"

docker image inspect fishbot_multirobot_sim-legged:run1-v128-historical >/dev/null
docker image inspect fishbot_multirobot_sim-racer_ros1:v128-dual-first-grid-bonus >/dev/null
docker image inspect swarm-lio2-ros2:run1-v128-validated >/dev/null

docker tag fishbot_multirobot_sim-legged:run1-v128-historical \
  "ghcr.io/$OWNER/fishbot-dual-go2-legged:run1-v128"
docker tag fishbot_multirobot_sim-racer_ros1:v128-dual-first-grid-bonus \
  "ghcr.io/$OWNER/fishbot-dual-go2-racer:run1-v128"
docker tag swarm-lio2-ros2:run1-v128-validated \
  "ghcr.io/$OWNER/fishbot-dual-go2-swarm-lio2:run1-v128"

echo "tagged images under ghcr.io/$OWNER"
echo "Run scripts/push_images_to_ghcr.sh only after docker login ghcr.io succeeds."
