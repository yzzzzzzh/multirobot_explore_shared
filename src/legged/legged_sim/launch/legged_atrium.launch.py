"""Two Unitree Go2 (quadruped_ros2_control, unitree_guide trot) in the teaching-building
world under Gazebo Fortress, with the same ROS topics the RACER stack expects:
/botN/cmd_vel (Twist in), /botN/imu, /botN/lidar_points/points, /world/default/pose/info."""
import json
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction, TimerAction
from launch_ros.actions import Node


def render_controller_yaml(template_path: str, ns: str) -> str:
    text = open(template_path).read().replace("__NS__", ns)
    text = text.replace("__RL_MODEL__", os.environ.get("LEGGED_RL_MODEL", "legged_gym"))
    out = f"/tmp/go2_controllers_{ns}.yaml"
    with open(out, "w") as f:
        f.write(text)
    return out


def launch_setup(context, *args, **kwargs):
    pkg = get_package_share_directory("legged_sim")
    sim_pkg = get_package_share_directory("gazebo_sim")
    world = context.launch_configurations["world"]
    headless = context.launch_configurations["headless"] == "true"
    count = int(context.launch_configurations["count"])
    spawn_delay = float(context.launch_configurations["spawn_delay"])
    controller_delay = float(context.launch_configurations["controller_delay"])
    world_path = world if os.path.isabs(world) else os.path.join(sim_pkg, "worlds", world)
    poses = json.loads(os.environ.get("SPAWN_POSES", "[[-4.0, 0.0, 0.45, 0.0], [4.0, 0.0, 0.45, 0.0]]"))

    # The robots are written into the world file instead of being spawned at
    # runtime: in Gazebo Fortress the force-torque sensors (foot contact for
    # the gait controller) are only created for models present at world load.
    robot_descriptions = {}
    model_sdfs = []
    for i in range(count):
        ns = f"bot{i + 1}"
        x, y, z, yaw = (list(poses[i]) + [0.0])[:4]
        yaml_path = render_controller_yaml(os.path.join(pkg, "config", "go2_controllers.yaml.in"), ns)
        xacro_path = os.path.join(pkg, "xacro", "go2_mid360.xacro")
        urdf = os.popen(f"xacro {xacro_path} robot_namespace:={ns} controllers_yaml:={yaml_path}").read()
        robot_descriptions[ns] = urdf
        urdf_path = f"/tmp/{ns}.urdf"
        with open(urdf_path, "w") as f:
            f.write(urdf)
        sdf = os.popen(f"ign sdf -p {urdf_path}").read()
        m0 = sdf.index("<model "); m1 = sdf.rindex("</model>") + len("</model>")
        model = sdf[m0:m1]
        model = model.replace("<model name='go2'>", f"<model name='{ns}'><pose>{x} {y} {z} 0 0 {yaw}</pose>", 1)
        model_sdfs.append(model)
    world_text = open(world_path).read()
    end = world_text.rindex("</world>")
    world_text = world_text[:end] + "\n".join(model_sdfs) + "\n" + world_text[end:]
    world_path = "/tmp/legged_world.sdf"
    with open(world_path, "w") as f:
        f.write(world_text)

    gz_args = ["ign", "gazebo", "-r", "-v", "3"]
    if headless:
        gz_args += ["-s", "--headless-rendering"]
    gz_args.append(world_path)
    plugin_dirs = ["/legged_ws/install/lib", "/legged_ws/install/gz_quadruped_hardware/lib"]
    plugin_path = ":".join(plugin_dirs + [os.environ.get("IGN_GAZEBO_SYSTEM_PLUGIN_PATH", "")])
    actions = [ExecuteProcess(cmd=gz_args, output="screen",
                              additional_env={"IGN_GAZEBO_SYSTEM_PLUGIN_PATH": plugin_path,
                                              "GZ_SIM_SYSTEM_PLUGIN_PATH": plugin_path})]

    bridge_args = [
        "/clock@rosgraph_msgs/msg/Clock[ignition.msgs.Clock",
        "/world/default/pose/info@tf2_msgs/msg/TFMessage[ignition.msgs.Pose_V",
    ]
    for i in range(count):
        ns = f"bot{i + 1}"
        bridge_args += [
            f"/{ns}/imu@sensor_msgs/msg/Imu[ignition.msgs.IMU",
            f"/{ns}/lidar_points/points@sensor_msgs/msg/PointCloud2[ignition.msgs.PointCloudPacked",
        ]
    actions.append(Node(package="ros_gz_bridge", executable="parameter_bridge", name="legged_bridge",
                        arguments=bridge_args, parameters=[{"use_sim_time": True}], output="screen"))

    spawn_actions = []
    for i in range(count):
        ns = f"bot{i + 1}"
        yaml_path = f"/tmp/go2_controllers_{ns}.yaml"
        spawn_actions.append(Node(package="robot_state_publisher", executable="robot_state_publisher",
                                  namespace=ns, output="screen",
                                  parameters=[{"use_sim_time": True, "robot_description": robot_descriptions[ns],
                                               "frame_prefix": f"{ns}/"}]))
        cm = f"/{ns}/controller_manager"
        locomotion = os.environ.get("LEGGED_CONTROLLER", context.launch_configurations["locomotion"])
        loco_controller = "rl_quadruped_controller" if locomotion == "rl" else "unitree_guide_controller"
        spawners = [
            Node(package="controller_manager", executable="spawner", output="screen",
                 arguments=["joint_state_broadcaster", "--controller-manager", cm, "-p", yaml_path]),
            Node(package="controller_manager", executable="spawner", output="screen",
                 arguments=["imu_sensor_broadcaster", "--controller-manager", cm, "-p", yaml_path]),
            Node(package="controller_manager", executable="spawner", output="screen",
                 arguments=[loco_controller, "--controller-manager", cm, "-p", yaml_path]),
        ]
        # Controller managers of all robots live in the Gazebo process; loading the same
        # controller plugin concurrently races inside pluginlib, so stagger per robot.
        spawn_actions.append(TimerAction(period=controller_delay + 4.0 * i, actions=spawners))
        spawn_actions.append(Node(package="legged_sim", executable="twist_to_control_input.py",
                                  namespace=ns, name="twist_to_control_input", output="screen",
                                  parameters=[{"use_sim_time": True,
                                               "vx_limit": float(context.launch_configurations["vx_limit"]),
                                               "vy_limit": float(context.launch_configurations["vy_limit"]),
                                               "wz_limit": float(context.launch_configurations["wz_limit"]),
                                               "auto_trot": os.environ.get("LEGGED_AUTO_TROT", context.launch_configurations["auto_trot"]) == "true",
                                               "walk_state": "rl" if locomotion == "rl" else "trotting",
                                               "raw_velocity_commands": locomotion == "rl",
                                               "min_step_speed": float(os.environ.get(
                                                   "LEGGED_MIN_STEP_SPEED",
                                                   "0.28" if locomotion == "rl" else "0.0")),
                                               "park_when_idle": os.environ.get(
                                                   "LEGGED_PARK_WHEN_IDLE",
                                                   "true" if locomotion == "rl" else "false") == "true",
                                               "idle_park_seconds": float(os.environ.get(
                                                   "LEGGED_IDLE_PARK_SECONDS", "3.0")),
                                               "max_linear_accel": float(os.environ.get(
                                                   "LEGGED_MAX_LINEAR_ACCEL",
                                                   "0.35" if locomotion == "rl" else "0.0")),
                                               "max_angular_accel": float(os.environ.get(
                                                   "LEGGED_MAX_ANGULAR_ACCEL",
                                                   "1.0" if locomotion == "rl" else "0.0"))}]))
    actions.append(TimerAction(period=spawn_delay, actions=spawn_actions))
    return actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("world", default_value="teaching_building_atrium.world"),
        DeclareLaunchArgument("headless", default_value="true"),
        DeclareLaunchArgument("count", default_value="2"),
        DeclareLaunchArgument("spawn_delay", default_value="6.0"),
        DeclareLaunchArgument("controller_delay", default_value="6.0"),
        DeclareLaunchArgument("vx_limit", default_value="0.4"),
        DeclareLaunchArgument("vy_limit", default_value="0.3"),
        DeclareLaunchArgument("wz_limit", default_value="0.5"),
        DeclareLaunchArgument("auto_trot", default_value="true"),
        DeclareLaunchArgument("locomotion", default_value="guide"),
        OpaqueFunction(function=launch_setup),
    ])
