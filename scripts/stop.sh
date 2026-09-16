#!/usr/bin/env bash
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
docker compose -f docker-compose.yml stop --timeout 30 \
  racer_controller racer_ros1 swarm_lio2 gazebo
