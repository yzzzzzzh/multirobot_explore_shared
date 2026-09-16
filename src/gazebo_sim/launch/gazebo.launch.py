import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, SetEnvironmentVariable
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    package_name = 'gazebo_sim'
    pkg_share = FindPackageShare(package=package_name).find(package_name)
    default_world = os.environ.get('GAZEBO_WORLD', 'fishbot.world')

    resource_paths = [
        pkg_share,
        os.path.join(pkg_share, 'worlds'),
        os.path.join(pkg_share, 'robots'),
    ]
    existing_resource = os.environ.get('IGN_GAZEBO_RESOURCE_PATH')
    if existing_resource:
        resource_paths.append(existing_resource)
    resource_value = ':'.join(dict.fromkeys(resource_paths))

    world_arg = DeclareLaunchArgument('world', default_value=default_world)
    headless_arg = DeclareLaunchArgument('headless', default_value='false')
    world_path = PathJoinSubstitution([pkg_share, 'worlds', LaunchConfiguration('world')])

    start_gazebo_gui_cmd = ExecuteProcess(
        cmd=['ign', 'gazebo', '-r', '-v', '4', '--render-engine', 'ogre2', world_path],
        output='screen',
        condition=UnlessCondition(LaunchConfiguration('headless')),
    )
    start_gazebo_headless_cmd = ExecuteProcess(
        cmd=[
            'ign', 'gazebo', '-r', '-s', '--headless-rendering',
            '-v', '4', '--render-engine', 'ogre2', world_path,
        ],
        output='screen',
        condition=IfCondition(LaunchConfiguration('headless')),
    )

    # Bridge simulation clock from the world topic into ROS /clock
    clock_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        # Gazebo publishes the clock on /world/<name>/clock; remap it to /clock for ROS
        arguments=['/world/default/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock'],
        remappings=[('/world/default/clock', '/clock')],
        output='screen',
    )

    launch_description = LaunchDescription()
    launch_description.add_action(world_arg)
    launch_description.add_action(headless_arg)
    launch_description.add_action(SetEnvironmentVariable('IGN_GAZEBO_RESOURCE_PATH', resource_value))
    launch_description.add_action(SetEnvironmentVariable('GZ_SIM_RESOURCE_PATH', resource_value))
    launch_description.add_action(start_gazebo_gui_cmd)
    launch_description.add_action(start_gazebo_headless_cmd)
    launch_description.add_action(clock_bridge)
    return launch_description
