#!/usr/bin/env bash
# Wait for recording, collect the result, analyze it, and render both videos.
# Usage: scripts/finish_dual_run.sh <container_run_dir> <artifact_dir> <video_stem> [requested_duration]
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
SRC="${1:?container run directory is required}"
RUN="${2:?artifact directory is required}"
STEM="${3:?video stem is required}"
DURATION="${4:-700}"
LAYOUT="src/gazebo_sim/worlds/teaching_building_atrium.layout.json"

until docker exec swarm_lio2_ros2 sh -c "grep -q 'RACER_RECORD_RESULT=' '$SRC/recorder.log' 2>/dev/null"; do
  sleep 10
done
sleep 5

bash scripts/postprocess_dual_legged_run.sh "$SRC" "$RUN" "$LAYOUT"

python3 src/racer_integration/render_planar_quadruped_video.py \
  --run "$RUN/run.npz" \
  --summary "$RUN/run.summary.json" \
  --layout "$LAYOUT" \
  --coverage "$RUN/planar_metrics.coverage.npz" \
  --output "$RUN/${STEM}_debug.mp4" \
  --thumbnail "$RUN/${STEM}_debug.png" \
  --speedup 10 \
  --title "Two Go2 + Swarm-LIO2 + RACER v128, atrium, ${DURATION} sim s" \
  > "$RUN/render_planar.log" 2>&1

python3 src/racer_integration/render_quadruped_hgrid_diagnostic_video.py \
  --run "$RUN/run.npz" \
  --layout "$LAYOUT" \
  --rosout "$RUN/ros1_logs/rosout.log" \
  --output "$RUN/${STEM}_hgrid_debug.mp4" \
  --thumbnail "$RUN/${STEM}_hgrid_debug.png" \
  --metadata "$RUN/${STEM}_hgrid_debug.json" \
  --speedup 10 \
  --max-align-median 1.0 \
  --max-align-p90 2.0 \
  > "$RUN/render_hgrid.log" 2>&1

python3 - "$RUN" <<'PY'
import json
import pathlib
import sys

import numpy as np

run = pathlib.Path(sys.argv[1])
with np.load(run / "run.npz", allow_pickle=False) as data:
    if not data.files:
        raise SystemExit("run.npz has no arrays")
for name in ("run.summary.json", "run.analysis.json", "planar_metrics"):
    with (run / name).open("r", encoding="utf-8") as stream:
        json.load(stream)
PY

ffmpeg -v error -i "$RUN/${STEM}_debug.mp4" -f null -
ffmpeg -v error -i "$RUN/${STEM}_hgrid_debug.mp4" -f null -

(
  cd "$RUN"
  sha256sum \
    run.npz run.summary.json run.analysis.json planar_metrics \
    planar_metrics.coverage.npz "${STEM}_debug.mp4" "${STEM}_hgrid_debug.mp4" \
    > SHA256SUMS
)

echo "post-processing completed: $RUN"
