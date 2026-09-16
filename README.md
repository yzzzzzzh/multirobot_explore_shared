# Dual Go2 RACER + Swarm-LIO2 v128

This repository is a focused, private reproduction package for the result
formerly stored as:

```text
artifacts/legged_go2_dual/atrium/run1_700s_v128
```

It contains only the source, Docker definitions, runtime configuration,
recording, analysis and rendering code needed for the two-robot teaching
building experiment. It intentionally excludes unrelated FishBot experiments.

## Reference result

The archived run requested 700 simulation seconds and recorded 699.8975
simulation seconds. The planar evaluator reported:

| Metric | Reference value |
|---|---:|
| Navigable area | 4406.5 m² |
| Final observed area | 4406.5 m² |
| Final planar coverage | 100% |
| Time to 80% | 365.4975 s |
| Time to 90% | 400.1975 s |
| Time to 95% | 415.4975 s |
| Total planar distance | 893.7524 m |
| Inter-robot collision samples | 0 |
| Emergency stop | false |

Machine-readable reference metrics are under
`results/reference/run1_700s_v128`. The full NPZ, videos and logs are intended
to be distributed as a GitHub Release rather than normal Git objects.

## Runtime architecture

```text
Gazebo Fortress + 2 x Go2 + HIMLoco
        │ LiDAR / IMU / ground truth used only by evaluation
        ▼
Swarm-LIO2 ROS 2 ── common-frame estimation ── racer_adapter
        │ planarized cloud, odometry and state over local TCP gateway
        ▼
RACER ROS 1 v128 ── HGrid / ACVRP / frontier / trajectory planning
        │ trajectory commands returned through the gateway
        ▼
ROS 2 safety/controller ── ground_omni cmd_vel ── two Go2 robots
```

Swarm-LIO2 continues to estimate full SE(3) poses. The adapter presents a
planar slice to RACER because this experiment is ground exploration.

## Important version boundary

The validated execution path uses three pinned container images described in
`config/provenance.json`:

- historical Go2/Gazebo image with working retro-reflector LiDAR returns;
- RACER ROS 1 v128 image with per-vehicle first-grid hysteresis;
- a Swarm-LIO2 ROS 2 runtime validated in the 2026-09-16 rerun.

Do not replace these with `latest`. A later locally rebuilt Go2 image produced
zero-intensity reflector returns and could not initialize the two-robot common
frame. The repository includes source-build Dockerfiles for inspection and
future development, but an image rebuilt later is not claimed to be
bit-identical to the validated image.

The original 2026-09-05 result did not preserve an independently identifiable
Swarm-LIO2 image digest. This limitation is recorded explicitly rather than
claiming full bit-for-bit reconstruction.

## Requirements

- Linux amd64; tested on Ubuntu with an NVIDIA RTX 3060 12 GB
- NVIDIA driver and `nvidia-smi`
- Docker Engine and Docker Compose v2
- NVIDIA Container Toolkit with the Docker `nvidia` runtime
- Python 3, NumPy, Matplotlib and FFmpeg on the host for analysis/rendering
- access to this repository and its private GHCR packages
- at least 30 GB free for container images and build/runtime data is recommended

The four runtime containers use fixed host networking and fixed names. Do not
run another copy of this stack at the same time.

## Clone and retrieve images

```bash
git clone git@github.com:yzzzzzzh/fishbot_multirobot_sim_dual_v128.git
cd fishbot_multirobot_sim_dual_v128
```

For a private package, authenticate to GHCR with a token having
`read:packages`:

```bash
printf '%s' "$CR_PAT" | docker login ghcr.io -u yzzzzzzh --password-stdin
```

After the images are published, copy the example environment and use the
immutable GHCR references recorded there:

```bash
cp .env.example .env
docker compose -f compose.run1-v128.yml pull
bash scripts/preflight.sh
```

Never commit `.env`, tokens or Docker credentials.

## Run

First perform a shorter 200-simulation-second acceptance run:

```bash
bash scripts/run_legged_dual_exploration.sh 200 smoke_200s
```

Then run the complete experiment:

```bash
bash scripts/run_legged_dual_exploration.sh 700 run_700s
```

The script performs the validated startup sequence:

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
the focused v128 release regression tests. The original monolithic
`test_racer_no_progress_guard.py` is retained as historical source but excluded
from release verification because it also asserts removed wheeled/UAV files and
post-v128 controller behavior.

## Source-build images

The exact result should use the published validated images. For development,
the included Dockerfiles can be built with:

```bash
bash scripts/build_images.sh
```

Proxy variables are optional and inherited when present:

```bash
HTTP_PROXY=http://proxy-host:7890 \
HTTPS_PROXY=http://proxy-host:7890 \
NO_PROXY=localhost,127.0.0.1 \
bash scripts/build_images.sh
```

Do not use `127.0.0.1` for a host proxy unless the build uses host networking
or the proxy is genuinely reachable from the builder.

## Publishing the validated images

The repository owner performs this once:

```bash
bash scripts/tag_images_for_ghcr.sh yzzzzzzh
printf '%s' "$CR_PAT" | docker login ghcr.io -u yzzzzzzh --password-stdin
bash scripts/push_images_to_ghcr.sh yzzzzzzh
```

After pushing, replace tag-only entries in `.env.example` with immutable
`ghcr.io/...@sha256:...` references and link each package to this repository so
private collaborators inherit access.

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
