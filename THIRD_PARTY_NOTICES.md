# Third-party notices and redistribution boundary

This repository combines modified upstream projects and locally authored
integration code. It does not claim that every file is covered by one common
license. Keep the repository and its container packages private until the
redistribution terms of every component, especially the RACER tree and the
pretrained policy files, have been reviewed by the repository owner.

## RACER

- Upstream: <https://github.com/Robotics-STAR-Lab/RACER>
- Base commit recorded in `config/provenance.json`.
- The vendored snapshot contains local changes for planar ground exploration,
  multi-robot allocation, recovery behavior and per-vehicle first-grid
  hysteresis.
- The inspected upstream snapshot did not contain a top-level license file.
  Source availability alone must not be represented as permission to
  redistribute under a particular open-source license.
- Cite the RACER paper identified in `src/RACER/README.md` when appropriate.

## Swarm-LIO2 ROS 2 port

- Port upstream: <https://github.com/V-Roman-V/Swarm-LIO2-ROS2-Docker>
- Original project: <https://github.com/hku-mars/Swarm-LIO2>
- The port repository includes GPL-2.0 text at
  `src/Swarm-LIO2-ROS2-Docker/LICENSE`.
- The `swarm_lio` package includes its own BSD-style notice at
  `src/Swarm-LIO2-ROS2-Docker/src/swarm_lio/LICENSE`.
- Preserve both notices when distributing source or binaries.

## quadruped_ros2_control and Go2 assets

- Upstream: <https://github.com/legubiao/quadruped_ros2_control>
- Selected packages, meshes, controller code and policy files are vendored
  under `src/legged/qrc` and include local Gazebo Fortress and namespace fixes.
- `gz_quadruped_hardware` carries Apache-2.0 text in its package directory.
- Other vendored packages and pretrained policies must be reviewed separately;
  the presence of one package license does not automatically license the whole
  vendored tree.

## HIMLoco policy

- Project: <https://github.com/OpenRobotLab/HIMLoco>
- The file used by this experiment is
  `src/legged/qrc/go2_description/config/himloco/himloco.pt`.
- SHA-256 is recorded in `config/provenance.json`.
- Confirm checkpoint redistribution permission before making the repository or
  its container image public.

## Other dependencies

The container builds also install ROS 1 Noetic, ROS 2 Humble, Gazebo Fortress,
LibTorch, GTSAM 4.2, NLopt 2.7.1 and LKH 3.0.6. Their licenses remain with the
respective upstream projects and distributions.
