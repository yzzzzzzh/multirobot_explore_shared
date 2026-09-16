from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    default_config = str(
        Path(get_package_share_directory("racer_adapter"))
        / "config"
        / "racer_adapter.yaml"
    )
    config = LaunchConfiguration("config")
    return LaunchDescription(
        [
            DeclareLaunchArgument("config", default_value=default_config),
            Node(
                package="racer_adapter",
                executable="racer_frame_adapter",
                name="racer_frame_adapter",
                output="screen",
                parameters=[config],
            ),
        ]
    )
