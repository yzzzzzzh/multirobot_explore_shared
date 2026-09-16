# Dual Go2 RACER + Swarm-LIO2

## Runtime architecture

```text
Gazebo Fortress + 2 x Go2 + HIMLoco
        │ LiDAR / IMU / ground truth used only by evaluation
        ▼
Swarm-LIO2 ROS 2 ── common-frame estimation ── racer_adapter
        │ planarized cloud, odometry and state over local TCP gateway
        ▼
RACER ROS 1 ── HGrid / ACVRP / frontier / trajectory planning
        │ trajectory commands returned through the gateway
        ▼
ROS 2 safety/controller ── ground_omni cmd_vel ── two Go2 robots
```

Swarm-LIO2 continues to estimate full SE(3) poses. The adapter presents a
planar slice to RACER because this experiment is ground exploration.

## Build and reproducibility boundary

This repository builds its complete Docker environment locally and does not
require any prebuilt private image. The historical runtime identities remain
recorded in `config/provenance.json` for audit purposes.

The source-built environment is not claimed to be byte-identical to the
historical images because upstream apt repositories can change, and the
original Swarm-LIO2 image digest was not preserved. See
`DOCKER_ENVIRONMENT.md` for the complete build graph, dependencies, runtime
topology and tested host.

## Requirements

- Linux amd64; tested on Ubuntu with an NVIDIA RTX 3060 12 GB
- NVIDIA driver and `nvidia-smi`
- Docker Engine and Docker Compose v2
- NVIDIA Container Toolkit with the Docker `nvidia` runtime
- Python 3, NumPy, Matplotlib and FFmpeg on the host for analysis/rendering
- internet access for the first Docker build
- at least 25 GB free in Docker storage

The four runtime containers use fixed host networking and fixed names. Do not
run another copy of this stack at the same time.

## Clone and build

```bash
git clone git@github.com:yzzzzzzh/multirobot_explore_shared.git
cd multirobot_explore_shared
./build_docker_environment.sh
```

The command builds all five local images in dependency order and runs the
preflight checks. It does not download project-specific images from a registry.
Optional proxy usage and all image details are documented in
`DOCKER_ENVIRONMENT.md`.

## Run

First perform a shorter 200-simulation-second acceptance run:

```bash
./run_one_click.sh 200 smoke_200s
```

Then run the complete experiment:

```bash
./run_one_click.sh 700 run_700s
```

`run_one_click.sh` performs the source build, starts the experiment, records the
requested simulation duration, computes the metrics and renders both videos.
Docker reuses its build cache on later runs. If the images are already built,
`bash scripts/run_legged_dual_exploration.sh ...` skips the build step.

The script performs the configured startup sequence:

1. start both Go2 robots in fixed stand;
2. wait for both Swarm-LIO2 odometry topics;
3. release both HIMLoco gaits and verify the parameter calls;
4. execute the root-only common-frame excitation;
5. wait until both robots are tracking;
6. record for the requested amount of simulation time;
7. collect logs, calculate metrics and render two videos.

Results appear under `runs/<run_name>/`. Simulation time is the acceptance
duration; wall time depends on real-time factor, and the video duration is
shorter because rendering uses a speed-up factor.

To stop a running stack explicitly:

```bash
bash scripts/stop.sh
```

## Expected outputs

Each successful run includes at least:

```text
run.npz
run.summary.json
run.analysis.json
planar_metrics
planar_metrics.coverage.npz
*_debug.mp4
*_hgrid_debug.mp4
SHA256SUMS
ros1_logs/
ros2_logs/
```

The finish script loads the NPZ/JSON files and fully decodes both videos before
declaring success.

## Local repository verification

```bash
bash scripts/verify_repository.sh
```

This checks shell syntax, JSON, Compose resolution, host-specific paths, large
files, nested build trees, critical hashes, reusable integration unit tests and
the focused release regression tests. The original monolithic
`test_racer_no_progress_guard.py` is retained as historical source but excluded
from release verification because it also asserts removed wheeled/UAV files and
controller behavior added after the archived snapshot.

## Proxy build

The build accepts the same explicit proxy variables as the single-robot shared
repository:

```bash
FISHBOT_HTTP_PROXY=http://proxy-host:7890 \
FISHBOT_HTTPS_PROXY=http://proxy-host:7890 \
./build_docker_environment.sh
```

Every Docker build uses host networking, so a proxy listening on
`127.0.0.1` is reachable during the build.

Prepare the archived 700-s result for the GitHub Release without adding large
artifacts to Git history:

```bash
bash scripts/prepare_release_assets.sh /path/to/run1_700s_v128
```

The generated files appear under `release_assets/`, which is ignored by Git.
Upload them to a release tagged `run1-700s`; keep the repository and packages
private unless the redistribution review in `LICENSE_SCOPE.md` has been
completed.

## Reproducibility criteria

A run is considered operationally valid only when:

- both LIO streams become ready;
- the common-frame graph reaches tracking for bots 1 and 2;
- raw LiDAR includes retro-reflector returns during bootstrap;
- `slam_validity.final_valid` is true for both robots;
- no emergency stop or wall/inter-robot collision is reported;
- both videos decode successfully.

Trajectory and coverage are not expected to be numerically identical across
all machines because simulation scheduling and planning are not fully
deterministic. Compare them against the archived metrics rather than requiring
byte-identical NPZ output.

## Licensing

This aggregate does not currently assert a single blanket open-source license.
Read `LICENSE_SCOPE.md` and `THIRD_PARTY_NOTICES.md` before sharing or changing
repository/package visibility. In particular, confirm redistribution rights
for the RACER snapshot and pretrained policy before making the project public.
