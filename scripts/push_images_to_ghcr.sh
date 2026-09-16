#!/usr/bin/env bash
# Requires: docker login ghcr.io -u <owner>
# Usage: scripts/push_images_to_ghcr.sh [github_owner]
set -euo pipefail

OWNER="${1:-yzzzzzzh}"
OWNER="${OWNER,,}"

images=(
  "ghcr.io/$OWNER/fishbot-dual-go2-legged:run1-v128"
  "ghcr.io/$OWNER/fishbot-dual-go2-racer:run1-v128"
  "ghcr.io/$OWNER/fishbot-dual-go2-swarm-lio2:run1-v128"
)

for image in "${images[@]}"; do
  docker image inspect "$image" >/dev/null
  docker push "$image"
done

echo "Push complete. Record the registry digests printed above and replace"
echo "the tag references in .env.example with ghcr.io/...@sha256:... values."
