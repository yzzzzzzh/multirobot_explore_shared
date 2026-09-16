# Repository contents

## Active execution path

- `compose.run1-v128.yml`: the only supported runtime Compose file.
- `scripts/run_legged_dual_exploration.sh`: startup and experiment orchestration.
- `scripts/preflight.sh`: host, image, model and configuration checks.
- `scripts/monitor_dual_run.sh`: periodic progress snapshots.
- `scripts/postprocess_dual_legged_run.sh`: collection and metric calculation.
- `scripts/finish_dual_run.sh`: rendering and output integrity checks.
- `scripts/prepare_release_assets.sh`: packages the archived 700-s videos,
  result data and logs for a GitHub Release without adding them to Git history.
- `src/RACER`: modified v128 working tree extracted from the RACER image.
- `src/Swarm-LIO2-ROS2-Docker`: ROS 2 LIO source used by the validated rerun.
- `src/racer_adapter`: ROS 2 planar adapter.
- `src/racer_integration`: ROS1/ROS2 gateway, controller, recorder and analysis.
- `src/legged`: Go2 model, controllers, HIMLoco policy and local adaptations.
- `src/gazebo_sim`: only the package metadata, launch code and atrium world/layout
  needed by this experiment.

## Historical configuration evidence

The files under `config/docker-compose*.yml` are the five original layered
Compose files from which the successful run was launched. They are retained for
auditability but are not the public entrypoint. Their relevant values were
collapsed into `compose.run1-v128.yml`.

## Included result evidence

`results/reference/run1_700s_v128` contains small text/JSON metrics and hashes.
The complete `run.npz`, videos and logs are not Git objects; they belong in the
GitHub Release named `run1-700s-v128`.

## Deliberately excluded

- unrelated single-Go2, wheeled, UAV and navigation experiment outputs;
- ROS build/install/log directories and Python caches;
- Docker image tar archives;
- the original parent repository Git history;
- credentials, personal environment files and machine-specific absolute paths.

Some upstream source packages remain vendored because the historical RACER
image was built as one catkin workspace. Their presence is a build dependency
boundary, not evidence that every upstream demonstration is active at runtime.
