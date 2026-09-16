#!/usr/bin/env bash
# Build all runtime images from this clone and validate the host configuration.
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$repo_dir"

command -v docker >/dev/null || {
  echo "Docker is required" >&2
  exit 10
}
docker info >/dev/null
docker compose version >/dev/null

docker_root="$(docker info --format '{{.DockerRootDir}}')"
available_kb="$(df -Pk "$docker_root" | awk 'NR==2 {print $4}')"
if [ "$available_kb" -lt 26214400 ]; then
  echo "warning: less than 25 GiB is free on the Docker storage filesystem" >&2
fi

bash scripts/build_images.sh
bash scripts/preflight.sh

echo "Docker environment build and preflight completed"
