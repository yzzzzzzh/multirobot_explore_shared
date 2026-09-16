#!/usr/bin/env bash
# Build from source, run the dual-Go2 experiment, and generate result videos.
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$repo_dir"

duration_sim_s="${1:-200}"
run_name="${2:-one_click_$(date +%Y%m%d_%H%M%S)}"

bash build_docker_environment.sh
exec bash scripts/run_legged_dual_exploration.sh "$duration_sim_s" "$run_name"
