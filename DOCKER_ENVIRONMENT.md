# Docker environment

The repository builds every runtime image locally. No prebuilt private image,
container registry login, or sibling source checkout is required.

## Tested host

- Linux amd64
- Docker Engine 28.1.1
- Docker Compose v2.35.1
- NVIDIA driver 570.153.02
- NVIDIA Container Toolkit with the Docker `nvidia` runtime
- NVIDIA GeForce RTX 3060 with 12 GiB VRAM

Equivalent newer Linux, Docker and NVIDIA versions may work, but have not been
validated by this repository.

## Build graph

| Local image | Dockerfile | Base image | Purpose |
|---|---|---|---|
| `multirobot_explore_shared-base:local` | `docker/Dockerfile` | `osrf/ros:humble-desktop` | ROS 2 build base |
| `multirobot_explore_shared-gazebo:local` | `docker/Dockerfile.gazebo` | local base image | Gazebo Fortress and teaching-building world |
| `multirobot_explore_shared-legged:local` | `docker/Dockerfile.legged` | local Gazebo image | two Go2 models, ros2_control and HIMLoco |
| `multirobot_explore_shared-racer-ros1:local` | `src/racer_integration/Dockerfile.ros1` | `osrf/ros:noetic-desktop-full` | RACER planner and ROS 1 gateway |
| `multirobot_explore_shared-swarm-lio2:local` | `src/Swarm-LIO2-ROS2-Docker/docker/Dockerfile.ros2` | `ros:humble-ros-base` | Swarm-LIO2, adapter, controller and recorder |

The three external base images are pinned in their Dockerfiles to these
registry digests:

```text
osrf/ros:humble-desktop@sha256:fb07245b32187d74350be25323d8ad2f8ca5c25c325759911a1eff2267a49c1e
osrf/ros:noetic-desktop-full@sha256:7dbfb9576d8e6d226c31e06129a82aaab8702695f38eca2116918cb9b9308797
ros:humble-ros-base@sha256:1813d3c85d7f96ff7d3012d865204583255740182db5d0065f8f8cd029a83138
```

The important explicitly selected dependencies are GTSAM 4.2, LibTorch 2.5.0
CPU with the C++11 ABI, NLopt 2.7.1 and LKH 3.0.6. ROS and Ubuntu packages are
installed from their upstream apt repositories during the build. The HIMLoco
policy is included in the repository and checked by
`config/critical-files.sha256`.

## Runtime topology

`docker-compose.yml` starts four services with host networking and fixed
container names:

| Service | Container | Local image |
|---|---|---|
| `gazebo` | `fishbot_gazebo` | legged image |
| `swarm_lio2` | `swarm_lio2_ros2` | Swarm-LIO2 image |
| `racer_ros1` | `racer_ros1` | RACER image |
| `racer_controller` | `racer_controller` | Swarm-LIO2 image |

All runtime paths are repository-relative bind mounts. The stack uses host IPC,
host networking, the NVIDIA runtime, and ROS domain ID 0. Do not run a second
copy simultaneously because container names and network ports are fixed.

## One-command build

```bash
./build_docker_environment.sh
```

This builds the five images in dependency order and then runs the host, image,
model and Compose preflight checks. The first build needs internet access and
approximately 25 GiB of free Docker storage.

For a proxy reachable on the Docker host:

```bash
FISHBOT_HTTP_PROXY=http://127.0.0.1:7890 \
FISHBOT_HTTPS_PROXY=http://127.0.0.1:7890 \
./build_docker_environment.sh
```

Every build uses host networking, so a proxy listening on host loopback is
reachable from BuildKit. Proxy values are passed as build arguments and are not
written into the repository.

## Reproducibility boundary

This is a complete source-build environment. It is not claimed to be
byte-identical to the historical local images: upstream apt repositories can
change, and the original Swarm-LIO2 image digest was not preserved.
`config/provenance.json` records the historical image IDs and source origins
for audit purposes. The Dockerfiles, Compose resolution and build graph are
checked automatically; a newly built environment still requires runtime
acceptance with both LIO streams, common-frame tracking, valid SLAM,
collision-free motion and decodable result videos.
