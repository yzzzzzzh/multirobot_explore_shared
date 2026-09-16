import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description() -> LaunchDescription:
    """Launch Gazebo and delegate robot spawning to spawn_robots.launch.py."""
    pkg_share = get_package_share_directory('gazebo_sim')

    default_world = os.environ.get('GAZEBO_WORLD', 'fishbot.world')

    declare_args = [
        DeclareLaunchArgument('world', default_value=default_world),
        DeclareLaunchArgument('headless', default_value='false'),
        DeclareLaunchArgument('world_name', default_value='default'),
        DeclareLaunchArgument('robot', default_value='fishbot_v2_3d'),
        DeclareLaunchArgument('count', default_value='3'),
        DeclareLaunchArgument('name_prefix', default_value='bot'),
        DeclareLaunchArgument('start_index', default_value='1'),
        DeclareLaunchArgument('x', default_value='0.0'),
        DeclareLaunchArgument('y', default_value='0.0'),
        # Ground robots spawn at z=0; flying robots need a non-zero spawn height.
        DeclareLaunchArgument('z', default_value='0.0'),
        DeclareLaunchArgument('pattern', default_value='matrix'),
        DeclareLaunchArgument('spacing', default_value='1'),
    ]

    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_share, 'launch', 'gazebo.launch.py')
        ),
        launch_arguments={
            'world': LaunchConfiguration('world'),
            'headless': LaunchConfiguration('headless'),
        }.items(),
    )

    spawn_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_share, 'launch', 'spawn_robots.launch.py')
        ),
        launch_arguments={
            'world_name': LaunchConfiguration('world_name'),
            'robot': LaunchConfiguration('robot'),
            'count': LaunchConfiguration('count'),
            'name_prefix': LaunchConfiguration('name_prefix'),
            'start_index': LaunchConfiguration('start_index'),
            'x': LaunchConfiguration('x'),
            'y': LaunchConfiguration('y'),
            'z': LaunchConfiguration('z'),
            'pattern': LaunchConfiguration('pattern'),
            'spacing': LaunchConfiguration('spacing'),
        }.items(),
    )

    ld = LaunchDescription()
    for action in declare_args:
        ld.add_action(action)
    ld.add_action(gazebo_launch)
    ld.add_action(spawn_launch)
    return ld
