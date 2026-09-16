#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE="${1:?usage: prepare_release_assets.sh PATH_TO_run1_700s_v128 [OUTPUT_DIR]}"
OUTPUT="${2:-$REPO/release_assets/run1_700s_v128}"

if [ ! -d "$SOURCE" ]; then
  echo "reference result directory not found: $SOURCE" >&2
  exit 2
fi

videos=(
  go2_dual_atrium_700s_run1_debug.mp4
  go2_dual_atrium_700s_run1_hgrid_debug.mp4
  go2_dual_atrium_700s_run1_slam_view_edited.mp4
)

result_files=(
  run.npz
  run.summary.json
  run.analysis.json
  planar_metrics
  planar_metrics.coverage.npz
  go2_dual_atrium_700s_run1_debug.png
  go2_dual_atrium_700s_run1_hgrid_debug.png
  go2_dual_atrium_700s_run1_hgrid_debug.json
  go2_dual_atrium_700s_run1_slam_view_edited.png
  comparison_table.md
  final_snapshot.txt
)

for file in "${videos[@]}" "${result_files[@]}"; do
  if [ ! -f "$SOURCE/$file" ]; then
    echo "required release file missing: $SOURCE/$file" >&2
    exit 3
  fi
done

mkdir -p "$OUTPUT"
for file in "${videos[@]}"; do
  install -m 0644 "$SOURCE/$file" "$OUTPUT/$file"
done

tar -C "$SOURCE" -czf "$OUTPUT/run1_700s_v128_results.tar.gz" \
  "${result_files[@]}"

log_inputs=(
  ros1_logs
  ros2_logs
  recorder.log
  monitor.log
  finish.log
  render_planar.log
  render_hgrid.log
  render_slam_view_edited.log
)
for item in "${log_inputs[@]}"; do
  if [ ! -e "$SOURCE/$item" ]; then
    echo "required log input missing: $SOURCE/$item" >&2
    exit 4
  fi
done

tar -C "$SOURCE" -czf "$OUTPUT/run1_700s_v128_logs.tar.gz" \
  "${log_inputs[@]}"

(
  cd "$OUTPUT"
  sha256sum \
    "${videos[@]}" \
    run1_700s_v128_results.tar.gz \
    run1_700s_v128_logs.tar.gz \
    > SHA256SUMS
  sha256sum -c SHA256SUMS
)

echo "release assets prepared: $OUTPUT"
