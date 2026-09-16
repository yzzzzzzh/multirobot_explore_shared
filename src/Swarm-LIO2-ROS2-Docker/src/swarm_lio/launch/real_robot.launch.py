"""Swarm-LIO2 on a real robot (Unitree Go2 + L1 / any PointCloud2 lidar).

Differences from single_drone_sim.launch.py:
  * the parameter file is an argument (config_file), so tuning is
    "edit yaml -> restart this node" with nothing else touched;
  * lidar / imu topics are arguments (default: bot<N>/lidar_points/points and
    bot<N>/imu, the names the adapter/controller/gateway already expect);
  * use_sim_time defaults to false; no ground-truth subscriptions;
  * any yaml key can be overridden on the command line through
    `param_overrides:="mapping/filter_size_surf=0.3,preprocess/blind=0.5"`.

Example:
  ros2 launch swarm_lio real_robot.launch.py drone_id:=1 \
      config_file:=/deploy/config/swarm_lio_go2_l1.yaml uav_num:=1
"""
import os

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def flatten_dict(d, parent_key="", sep="/"):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def parse_override(text):
    text = text.strip()
    if text.lower() in ("true", "false"):
        return text.lower() == "true"
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text


def launch_setup(context, *args, **kwargs):
    cfg = lambda name: LaunchConfiguration(name).perform(context)  # noqa: E731
    drone_id = int(cfg("drone_id"))
    config_file = cfg("config_file")
    if not config_file:
        config_file = os.path.join(get_package_share_directory("swarm_lio"), "config", "mid360.yaml")
    with open(config_file, "r") as f:
        params = flatten_dict(yaml.safe_load(f).get("ros__parameters", {}))
    params["common/drone_id"] = drone_id
    params["common/lid_topic"] = cfg("lid_topic") or f"bot{drone_id}/lidar_points/points"
    params["common/imu_topic"] = cfg("imu_topic") or f"bot{drone_id}/imu"
    params["use_sim_time"] = cfg("use_sim_time").lower() in ("true", "1", "yes")
    params["evaluation/ground_truth_logging_en"] = False
    uav_num = int(cfg("uav_num"))
    if uav_num > 0:
        params["multiuav/actual_uav_num"] = uav_num
    params["multiuav/enable_mutual_observation_update"] = (
        cfg("enable_mutual_observation_update").lower() in ("true", "1", "yes"))
    params["pcd_save/pcd_save_en"] = cfg("pcd_save_en").lower() in ("true", "1", "yes")
    for item in [x for x in cfg("param_overrides").split(",") if "=" in x]:
        key, value = item.split("=", 1)
        params[key.strip()] = parse_override(value)
    print(f"[swarm_lio real] drone {drone_id} config={config_file} "
          f"lidar={params['common/lid_topic']} imu={params['common/imu_topic']}")
    nodes = [
        Node(
            package="swarm_lio",
            executable="swarm_lio",
            name=f"laserMapping_bot{drone_id}",
            output="screen",
            parameters=[params],
            emulate_tty=True,
        ),
        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            name=f"bot{drone_id}_aft_mapped_to_base_link_static_tf",
            arguments=["--x", "0", "--y", "0", "--z", "0", "--roll", "0", "--pitch", "0",
                       "--yaw", "0", "--frame-id", f"bot{drone_id}/aft_mapped",
                       "--child-frame-id", f"bot{drone_id}/base_link"],
            output="log",
        ),
    ]
    return nodes


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("drone_id", default_value="1"),
        DeclareLaunchArgument("config_file", default_value="",
                              description="swarm_lio parameter yaml (ros__parameters tree)"),
        DeclareLaunchArgument("lid_topic", default_value=""),
        DeclareLaunchArgument("imu_topic", default_value=""),
        DeclareLaunchArgument("use_sim_time", default_value="false"),
        DeclareLaunchArgument("uav_num", default_value="1", description="swarm size; 0 keeps yaml"),
        DeclareLaunchArgument("enable_mutual_observation_update", default_value="false"),
        DeclareLaunchArgument("pcd_save_en", default_value="false"),
        DeclareLaunchArgument("param_overrides", default_value="",
                              description='comma list "key=value" applied last, e.g. mapping/filter_size_surf=0.3'),
        OpaqueFunction(function=launch_setup),
    ])
