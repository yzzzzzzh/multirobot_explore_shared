# RACER + Swarm-LIO2 LiDAR-only Gazebo integration

Last validated: 2026-07-24.

## Result

The closed loop is operational in Gazebo:

```text
Gazebo 3D LiDAR + IMU
       |
       v
Swarm-LIO2 (ROS 2 Humble, one LIO per UAV)
       |
       +-- local odometry and current registered sparse LiDAR scan
       +-- peer trajectory matching and full SE(3) common-frame transforms
       |
       v
racer_adapter (ROS 2, transform/validate/latch)
       |
       v
standard-message TCP bridge on localhost:47100
       |
       v
official RACER (ROS 1 Noetic, mapping/planning/trajectory server)
       |
       v
ROS 2 velocity controller -> Gazebo quadrotors
```

- Simulator: Gazebo (`ros_gz`), six `quad_mid360` UAVs.
- SLAM: Swarm-LIO2, ROS 2 Humble.
- Exploration: official RACER checkout at commit
  `049c332e3634ef72d8beb155b4c13dc91ca52916`, ROS 1 Noetic.
- Production feedback: Swarm-LIO2 odometry only. Gazebo truth is subscribed
  only by the optional evaluator.
- Mapping input: current registered 3D LiDAR scans only.
- Cross-world transform: full SE(3). The prior `se2_legacy` implementation is
  retained as an explicit fallback.

## Sensor boundary

The Gazebo UAV contains a `gpu_lidar` and an IMU; it contains no camera or
depth camera.

| Item | Setting |
|---|---:|
| LiDAR update rate | 10 Hz |
| horizontal samples/FOV | 180 / 360 deg |
| vertical samples/FOV | 96 / -40 to +52 deg |
| range | 0.12 to 6.0 m |
| range noise | Gaussian, 0.01 m standard deviation |
| IMU update rate | 100 Hz |
| LiDAR origin relative to IMU | `[0, 0, +0.095]` m |

RACER's depth-image input is deliberately configured as
`/racer/disabled_depth`. Runtime inspection showed no publisher on that topic.
`/sdf_map/depth_cloud` is a RACER map-visualization output, not a camera input.

For UAV `N`, the data boundary is:

- Gazebo to LIO: `/botN/lidar_points/points`, `/botN/imu`.
- LIO to adapter: `/botN/cloud_registered_sparse`,
  `/botN/lidar_slam/odom`, and the common-frame TF.
- Adapter to bridge/RACER: `/racer/botN/cloud_world`,
  `/racer/botN/lidar_pose`, `/racer/botN/odom_world`.
- RACER to controller: `/racer/botN/position_setpoint` and feed-forward
  velocity.
- Controller to Gazebo: `/botN/cmd_vel` and `/botN/enable`.

The cloud and the LiDAR ray origin carry the same timestamp and are transformed
by the same latched rigid transform. RACER raycasts the cloud from that LiDAR
origin to update occupancy; it does not receive an accumulated perfect map.

For multi-UAV launch semantics, RACER's official examples reserve the final
`exploration/drone_num` entry for the ground/map relay (two UAVs therefore use
`drone_num=3`). The integration now preserves that convention. RACER's
partitioning counts were also rescaled for the 0.25 m grid: volumetric
unknown/free thresholds use `(0.10/0.25)^3`, while the frontier-surface
threshold uses `(0.10/0.25)^2`.

## Coordinate frames and initialization

Swarm-LIO2 estimates a transform from every `botN/world` into the root
`map_origin` frame. The adapter:

1. waits for every requested transform;
2. checks it over a 3 s window (`<=0.05 m`, `<=0.5 deg` drift);
3. composes the configured root-to-RACER map offset;
4. latches all transforms before RACER starts accumulating occupancy;
5. applies the full rotation and x/y/z translation to point clouds, poses,
   orientations, and velocity vectors.

The system does not use a global positioning topic to initialize this frame.
Before the common frame exists, the controller uses each UAV's local LIO
odometry to execute a closed, non-planar excitation trajectory. Non-root UAVs
are excited sequentially, farthest first, to make trajectory matching
observable without Gazebo truth. After the transforms latch, the controller
switches to RACER commands.

The RACER integration enables
`gravity_constrained_extrinsic_rotation=true`: trajectory matching estimates
yaw, while the two gravity-aligned LIO frames robustify roll/pitch; translation
x/y/z is then fitted again. This remains full SE(3). As direct runtime evidence,
the final bot6-to-root transform included
`RPY=[-0.059, 0.029, -1.401] deg` and
`t=[0.978, 5.719, -0.060] m`; z and roll/pitch were not cleared.

## Changes to Swarm-LIO2

- Added `trajectory_alignment.hpp` with full 3D Kabsch alignment.
- Retained old yaw+x+y behavior as `se2_legacy`.
- Kept full `Pose3` transforms through graph propagation and TF publication.
- Added a gravity-constrained full-SO(3) option for weak vertical excitation.
- Added `enable_mutual_observation_update`. The RACER overlay sets it false:
  peer detection, tracking, trajectory matching, and transform-graph
  propagation still operate, while noisy reflective detections cannot update
  the ego ESIKF. Its default remains true outside this integration.
- Corrected simulated LiDAR-to-IMU translation to `+0.095 m`.
- Disabled unbounded PCD accumulation and the unused accumulated local-map
  publisher on this launch path.

## ROS 1/ROS 2 boundary

RACER's custom messages remain inside its ROS 1 container. A narrow
length-framed TCP protocol on `127.0.0.1:47100` carries only standard semantic
data:

- ROS 2 to ROS 1: `PointCloud2`, `PoseStamped`, and `Odometry`.
- ROS 1 to ROS 2: position setpoints and feed-forward twist.

The wire protocol rejects malformed headers, oversized payloads, unknown
message types, and invalid point-field layouts. This avoids a full custom
message port while keeping the RACER source largely intact.

## How to run

Run commands from the repository root. The tested world and irregular start
poses are supplied by `docker-compose.racer.yml`.

Single UAV:

```bash
RACER_UAV_COUNT=1 \
RACER_BOTS=1 \
RACER_ADAPTER_CONFIG=racer_adapter_single.yaml \
SWARM_FORCE_UAV_NUM=1 \
RACER_MAP_WARMUP_SECONDS=10 \
docker compose -f docker-compose.yml -f docker-compose.racer.yml \
  -f docker-compose.racer-live.yml up -d --build --force-recreate \
  gazebo swarm_lio2 racer_ros1 racer_controller
```

Two UAVs:

```bash
RACER_UAV_COUNT=2 \
RACER_BOTS=1,2 \
RACER_ADAPTER_CONFIG=racer_adapter_two.yaml \
SWARM_FORCE_UAV_NUM=2 \
RACER_MAP_WARMUP_SECONDS=15 \
docker compose -f docker-compose.yml -f docker-compose.racer.yml \
  -f docker-compose.racer-live.yml up -d --build --force-recreate \
  gazebo swarm_lio2 racer_ros1 racer_controller
```

Six UAVs:

```bash
RACER_UAV_COUNT=6 \
RACER_BOTS=1,2,3,4,5,6 \
RACER_ADAPTER_CONFIG=racer_adapter.yaml \
SWARM_FORCE_UAV_NUM=6 \
RACER_MAP_WARMUP_SECONDS=15 \
docker compose -f docker-compose.yml -f docker-compose.racer.yml \
  -f docker-compose.racer-live.yml up -d --build --force-recreate \
  gazebo swarm_lio2 racer_ros1 racer_controller
```

The optional ATE evaluator consumes Gazebo truth for measurement only:

```bash
docker exec swarm_lio2_ros2 bash -lc \
  'source /opt/ros/humble/setup.bash &&
   source /opt/swarm_lio_ws/install/setup.bash &&
   python3 /racer_integration/ros2_racer_evaluator.py \
     --bots 1,2,3,4,5,6 --duration 60 --alignment first_pose \
     --ros-args -p use_sim_time:=true'
```

Use `--alignment root_trajectory` for conventional post-run full-SE(3)
trajectory alignment.

## Validation

### Automated tests

- Python protocol, controller, and evaluation math: 11 tests passed.
- Adapter point-cloud/pose/full-SE(3) transform utilities: 5 tests passed.
- Full-SE(3) synthetic alignment: passed.
- Legacy SE(2) regression: passed.
- Gravity-constrained full-SO(3) alignment: passed.
- Full `swarm_lio` and `racer_adapter` Release build: passed.
- Official RACER ROS 1 build and launch smoke test: passed.

### Closed-loop Gazebo tests

The final comparison uses fresh 60 s wall-clock recordings after every UAV
entered RACER command tracking. `online ATE` applies one full-SE(3) alignment
latched at the first pose; it does not use future samples. `root-fit ATE`
applies one offline full-SE(3) Kabsch fit from the root trajectory to every
member, preserving the common frame rather than fitting each UAV separately.

| Scenario | sim time | GT distance | online ATE, pooled | root-fit ATE, pooled |
|---|---:|---:|---:|---:|
| 1 UAV | 59.69 s | 13.35 m | 0.436 m | 0.267 m |
| 2 UAV | 58.54 s | 56.16 m | 1.029 m | 1.207 m |
| 6 UAV | 19.32 s | 63.00 m | 1.025 m | 1.135 m |

Per-UAV online ATE RMSE:

| Scenario | bot1 | bot2 | bot3 | bot4 | bot5 | bot6 |
|---|---:|---:|---:|---:|---:|---:|
| 1 UAV | 0.436 | - | - | - | - | - |
| 2 UAV | 0.973 | 1.081 | - | - | - | - |
| 6 UAV | 0.187 | 0.867 | 2.019 | 0.583 | 0.845 | 0.623 |

The exploration proxy counts 1 m static-world surface cells observed by the
aligned accumulated registered LiDAR scans. It is a reproducible progress
proxy, not RACER's internal frontier-completion criterion.

| Scenario | 0.25 m map voxels | voxels/wall s | surface coverage | completed in 60 s |
|---|---:|---:|---:|---:|
| 1 UAV | 6,361 | 106.0 | 0.616% | no |
| 2 UAV | 11,785 | 196.3 | 0.997% | no |
| 6 UAV | 18,641 | 310.6 | 1.368% | no |

The six-UAV run was CPU-bound: 60.01 s wall time advanced Gazebo by 19.32 s
(real-time factor 0.322). The single- and two-UAV factors were 0.994 and 0.975.
All containers remained running. No inter-UAV envelope overlap occurred. The
single- and two-UAV recordings had no static-envelope overlap; the six-UAV
recording found conservative wall-envelope overlaps for bot2 and bot4. No
Gazebo contact sensor was installed, so those are collision-risk detections,
not confirmed physics contact events.

The rendered H.264 videos and machine-readable recordings are under
`artifacts/racer_videos/`.

## Known limitations

- bot3 was the weakest common-frame link in the final six-UAV recording
  (2.019 m online ATE); multi-hop transform quality remains the main accuracy
  risk.
- bot4 travelled only 1.23 m in the final six-UAV 60 s window despite entering
  command tracking. It also had a brief conservative wall-envelope overlap,
  so the run is not a collision-free six-UAV acceptance result.
- Common-frame transforms are intentionally latched before mapping. Online
  frame corrections would otherwise warp an already-built RACER occupancy
  grid; supporting them requires map deformation or controlled map reset.
- RACER occasionally logs `No path` in dense/partially observed regions and
  `Total time too long!!!` when a planning cycle exceeds its time budget.
  These remain planner/performance limitations rather than SLAM data-path
  failures.
- Six-UAV operation is CPU-bound on the current host: the registered cloud
  throughput can fall to roughly 3--4 Hz even though Gazebo generates scans at
  10 Hz.
- The current integration passes the current registered sparse scan to RACER.
  It does not fuse the Swarm-LIO2 accumulated map, and it intentionally does
  not use a camera.

## Baseline OOM finding

The old six-UAV configuration ran for about 15 hours before bot2, bot5, and
bot6 were killed with exit code `-9`; kernel logs identified the OOM killer.
At inspection, Swarm-LIO2 used about 15.3 GiB and `map_fusion` about 7.3 GiB.
The main cause was `pcd_save_en=true` with `interval=-1`, which retained every
registered scan in every LIO process, plus a second accumulated map in
`map_fusion`.

The RACER launch path now forces `pcd_save_en=false` and
`local_map_pub_en=false`, and RACER receives only the bounded current sparse
scan.
