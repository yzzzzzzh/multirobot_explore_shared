#!/usr/bin/env bash
# Per-minute progress monitor for a two-robot run: recorder progress line
# (from the recorder log inside swarm_lio2_ros2) + RACER snapshot.  Appends one
# JSON line per minute to <out_dir>/monitor.log; exits when the recorder ends.
# Usage: scripts/monitor_dual_run.sh <container_run_dir> <out_dir>
SRC="${1:?container run dir}"; OUT_DIR="${2:?out dir}"; mkdir -p "$OUT_DIR"
OUT="$OUT_DIR/monitor.log"
while true; do
  ts="$(date +%H:%M:%S)"
  rec="$(docker exec swarm_lio2_ros2 cat "$SRC/recorder.log" 2>/dev/null)"
  prog="$(grep 'RACER_MONITOR_PROGRESS=' <<<"$rec" | tail -1 | sed 's/^RACER_MONITOR_PROGRESS=//')"
  snap="$(timeout 40 docker exec racer_ros1 bash -lc 'source /racer_ws/devel/setup.bash >/dev/null 2>&1; python3 /racer_integration/ros1_final_snapshot.py --bots 1,2 --state-timeout 5 --service-timeout 8 2>/dev/null' | sed -n 's/^RACER_FINAL_SNAPSHOT=//p')"
  ctrl="$(docker logs --since 70s racer_controller 2>&1 | grep -c 'stuck_recovery\|separation\|brake' )"
  python3 - "$ts" "$prog" "$snap" "$ctrl" >> "$OUT" <<'PY'
import json, sys
ts, prog, snap, ctrl = sys.argv[1:5]
row = {"wall": ts}
try:
    p = json.loads(prog) if prog else {}
    row.update({"sim_s": round(p.get("sim_elapsed_s", -1), 1),
        "pos_err_m": {k: round(v, 3) for k, v in p.get("position_error_m", {}).items()},
        "slam_valid": p.get("slam_valid"), "invalid": p.get("slam_invalid_samples"),
        "dist_m": {k: round(v, 1) for k, v in p.get("distance_m", {}).items()},
        "min_dist_m": round(p.get("min_inter_uav_3d_m", -1), 2), "rtf": round(p.get("rtf", -1), 3),
        "tracking": p.get("tracking"), "warnings": p.get("warning_count")})
except Exception as e:
    row["prog_err"] = str(e)[:80]
try:
    s = json.loads(snap) if snap else {}
    cov = s.get("coverage", {})
    row["racer_obs"] = {k: round(v.get("stats", {}).get("observed_fraction", -1), 4) for k, v in cov.items()}
    row["tasks"] = s.get("ownership", {}).get("per_bot_task_count")
    row["racer_t"] = round(s.get("sim_time_s", -1), 1)
    row["pos"] = {k: [round(x, 1) for x in v.get("position_m", [])[:2]] for k, v in s.get("states", {}).items()}
except Exception as e:
    row["snap_err"] = str(e)[:80]
row["ctrl_events_1min"] = int(ctrl or 0)
print(json.dumps(row, ensure_ascii=False))
PY
  if grep -q "RACER_RECORD_RESULT=" <<<"$rec" 2>/dev/null; then
    echo "{\"wall\": \"$(date +%H:%M:%S)\", \"event\": \"recorder_finished\"}" >> "$OUT"; break
  fi
  sleep 60
done
