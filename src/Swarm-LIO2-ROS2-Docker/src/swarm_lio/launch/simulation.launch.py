from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os
import tempfile


def make_rviz_config(template_path: str, prefix: str, bot_id: int) -> str:
    with open(template_path, 'r') as f:
        text = f.read()

    text = text.replace('BOT_PREFIX', prefix)

    # Put a predictable name into /tmp so the RViz title is readable
    cfg_dir = tempfile.gettempdir()
    cfg_path = os.path.join(cfg_dir, f'Bot{bot_id}.rviz')

    with open(cfg_path, 'w') as f:
        f.write(text)

    return cfg_path


def generate_launch_description():
    declare_bot_list = DeclareLaunchArgument(
        "bot_list",
        default_value="1,2,3,4",
        description="Comma-separated list of bot IDs to simulate",
    )

    declare_rviz_list = DeclareLaunchArgument(
        "rviz_list",
        default_value="",
        description="Comma-separated list of bot IDs to visualize in RViz (defaults to first bot only)",
    )
    declare_use_sim_time = DeclareLaunchArgument(
        "use_sim_time",
        default_value="true",
        description="Use simulation time from /clock"
    )

    pkg_share = get_package_share_directory('swarm_lio')
    launch_dir = os.path.join(pkg_share, 'launch')
    rviz_cfg_dir = os.path.join(pkg_share, 'rviz_cfg')
    rviz_template = os.path.join(rviz_cfg_dir, 'ros2_generic.rviz')

    def launch_everything(context, *args, **kwargs):
        actions = []
        bot_ids = [int(x) for x in context.launch_configurations["bot_list"].split(',')]
        use_sim_time_str = context.launch_configurations.get('use_sim_time', 'true')
        use_sim_time_bool = str(use_sim_time_str).lower() in ('true', '1', 'yes')

        # --- Launch sim for each bot ---
        for bot_id in bot_ids:
            output_mode = 'screen' if bot_id == bot_ids[0] else 'log'
            actions.append(
                IncludeLaunchDescription(
                    PythonLaunchDescriptionSource(
                        os.path.join(launch_dir, 'single_drone_sim.launch.py')
                    ),
                    launch_arguments={
                        'drone_id': str(bot_id),
                        'output_mode': output_mode,
                        'use_sim_time': use_sim_time_str,
                        'ground_truth_logging_en':
                            context.launch_configurations.get(
                                'ground_truth_logging_en', 'false'),
                        # swarm size follows the actual bot count (yaml is stale);
                        # override to 1 to run PURE per-drone LIO with teammate
                        # detection off (independent-exploration scenarios where
                        # the mutual-observation code hits degenerate crashes).
                        'uav_num': (context.launch_configurations.get('force_uav_num', '0')
                                    if int(context.launch_configurations.get('force_uav_num', '0')) > 0
                                    else str(len(bot_ids))),
                        # robot-size-dependent cluster gate; 0 = keep yaml value
                        'cluster_size_thresh': context.launch_configurations.get(
                            'cluster_size_thresh', '0'),
                        'cross_world_transform_mode': context.launch_configurations.get(
                            'cross_world_transform_mode', 'se3'),
                        'enable_mutual_observation_update': context.launch_configurations.get(
                            'enable_mutual_observation_update', 'true'),
                        'gravity_constrained_extrinsic_rotation': context.launch_configurations.get(
                            'gravity_constrained_extrinsic_rotation', 'false'),
                        'max_abs_match_yaw_deg': context.launch_configurations.get(
                            'max_abs_match_yaw_deg', '5.0'),
                        'traj_matching_start_thresh':
                            context.launch_configurations.get(
                                'traj_matching_start_thresh', '0.0'),
                        'traj_matching_time_tolerance':
                            context.launch_configurations.get(
                                'traj_matching_time_tolerance', '0.0'),
                        'ave_match_error_thresh':
                            context.launch_configurations.get(
                                'ave_match_error_thresh', '0.0'),
                        'temp_tracker_lost_timeout':
                            context.launch_configurations.get(
                                'temp_tracker_lost_timeout', '0.0'),
                        'valid_temp_cluster_dist_thresh':
                            context.launch_configurations.get(
                                'valid_temp_cluster_dist_thresh', '0.0'),
                        'use_raw_temp_measurement_for_traj_matching':
                            context.launch_configurations.get(
                                'use_raw_temp_measurement_for_traj_matching',
                                'false'),
                        'trajectory_samples_require_measurement':
                            context.launch_configurations.get(
                                'trajectory_samples_require_measurement',
                                'false'),
                        'stop_tracking_after_bootstrap':
                            context.launch_configurations.get(
                                'stop_tracking_after_bootstrap',
                                'true'),
                        # Long exploration runs must not retain every scan in RAM.
                        # The RACER mapper consumes the current registered scan,
                        # so neither Swarm-LIO's PCD accumulator nor its local-map
                        # publisher is needed on this path.
                        'pcd_save_en': context.launch_configurations.get(
                            'pcd_save_en', 'false'),
                        'local_map_pub_en': context.launch_configurations.get(
                            'local_map_pub_en', 'false'),
                        'planar_odom_prior_en':
                            context.launch_configurations.get(
                                'planar_odom_prior_en', 'false'),
                        'planar_odom_position_gain':
                            context.launch_configurations.get(
                                'planar_odom_position_gain', '0.50'),
                        'planar_odom_velocity_gain':
                            context.launch_configurations.get(
                                'planar_odom_velocity_gain', '0.0'),
                        'planar_odom_yaw_gain':
                            context.launch_configurations.get(
                                'planar_odom_yaw_gain', '0.0'),
                        'point_filter_num': context.launch_configurations.get(
                            'point_filter_num', '0'),
                        'filter_size_surf': context.launch_configurations.get(
                            'filter_size_surf', '0.0'),
                        'filter_size_map': context.launch_configurations.get(
                            'filter_size_map', '0.0'),
                    }.items()
                )
            )
            actions.append(
                Node(
                    package='tf2_ros',
                    executable='static_transform_publisher',
                    name=f'bot{bot_id}_aft_mapped_to_base_link_static_tf',
                    arguments=[ # Identity transform
                        '--x', '0', '--y', '0', '--z', '0',
                        '--roll', '0', '--pitch', '0', '--yaw', '0',
                        '--frame-id', f'bot{bot_id}/aft_mapped',
                        '--child-frame-id', f'bot{bot_id}/base_link',
                    ],
                    parameters=[{'use_sim_time': use_sim_time_bool}],
                    output='log',
                )
            )


        # --- Decide which bots get RViz ---
        rviz_list_str = context.launch_configurations.get('rviz_list', '').strip()
        if rviz_list_str.lower() in ('none', 'off', 'false'):
            rviz_ids = []
        elif rviz_list_str:
            rviz_ids = [int(x) for x in rviz_list_str.split(',') if x]
        else:
            rviz_ids = [bot_ids[0]] if bot_ids else []

        # --- Publish extrinsics between robots ---
        if bot_ids:
            actions.append(
                Node(
                    package='swarm_lio',
                    executable='robot_extrinsic_publisher',
                    name='robot_extrinsic_publisher',
                    parameters=[{
                        'root_id': int(bot_ids[0]),
                        'robot_ids': bot_ids,
                        'robot_prefix': 'bot',
                        'robot_frame_suffix': 'world',
                        'use_sim_time': use_sim_time_bool,
                    }],
                    output='log',
                )
            )

        # --- Launch one RViz per requested bot ---
        for bot_id in rviz_ids:
            bot_prefix = f'/bot{bot_id}'
            rviz_config = make_rviz_config(rviz_template, bot_prefix, bot_id)

            actions.append(
                Node(
                    package='rviz2',
                    executable='rviz2',
                    name=f'rviz_{bot_id}',
                    arguments=[
                        '-d', rviz_config,
                        '--ros-args',
                        '--log-level', 'error',          # or 'fatal' if you want only crashes
                        '--disable-stdout-logs',         # no logs to terminal
                        # optionally also:
                        # '--disable-rosout-logs',
                        # '--disable-external-lib-logs',
                    ],
                    output='log',
                    emulate_tty=False,
                )

            )

        return actions

    declare_cluster_size = DeclareLaunchArgument(
        "cluster_size_thresh",
        default_value="0",
        description="valid_cluster_size_thresh override for big robots; 0 = keep yaml"
    )
    declare_force_uav_num = DeclareLaunchArgument(
        "force_uav_num",
        default_value="0",
        description="override actual_uav_num; 1 = pure per-drone LIO (teammate off)"
    )
    declare_cross_world_transform_mode = DeclareLaunchArgument(
        "cross_world_transform_mode",
        default_value="se3",
        description="cross-world trajectory alignment: se3 or se2_legacy"
    )
    declare_enable_mutual_observation_update = DeclareLaunchArgument(
        "enable_mutual_observation_update",
        default_value="true",
        description="feed active/passive teammate observations into each local ESIKF"
    )
    declare_ground_truth_logging_en = DeclareLaunchArgument(
        "ground_truth_logging_en",
        default_value="false",
        description="subscribe to simulator truth for estimator-side logging only"
    )
    declare_gravity_constrained_extrinsic_rotation = DeclareLaunchArgument(
        "gravity_constrained_extrinsic_rotation",
        default_value="false",
        description="robustify SE3 roll/pitch using each LIO gravity frame"
    )
    declare_max_abs_match_yaw_deg = DeclareLaunchArgument(
        "max_abs_match_yaw_deg",
        default_value="5.0",
        description="initial-heading plausibility gate for trajectory matching"
    )
    declare_traj_matching_time_tolerance = DeclareLaunchArgument(
        "traj_matching_time_tolerance",
        default_value="0.0",
        description="nearest trajectory timestamp association gate; 0 = keep yaml"
    )
    declare_use_raw_temp_measurement = DeclareLaunchArgument(
        "use_raw_temp_measurement_for_traj_matching",
        default_value="false",
        description="use current lidar cluster positions for trajectory matching"
    )
    declare_require_measured_trajectory_samples = DeclareLaunchArgument(
        "trajectory_samples_require_measurement",
        default_value="false",
        description="exclude prediction-only temporary-tracker samples"
    )
    declare_pcd_save_en = DeclareLaunchArgument(
        "pcd_save_en",
        default_value="false",
        description="accumulate scans for PCD saving (unsafe for long runs when interval=-1)"
    )
    declare_local_map_pub_en = DeclareLaunchArgument(
        "local_map_pub_en",
        default_value="false",
        description="publish Swarm-LIO's accumulated local map; RACER uses current scans instead"
    )
    declare_planar_odom_prior_en = DeclareLaunchArgument(
        "planar_odom_prior_en",
        default_value="false",
        description="fuse a bounded wheel/leg-odometry prior for planar robots"
    )
    declare_planar_odom_position_gain = DeclareLaunchArgument(
        "planar_odom_position_gain",
        default_value="0.50",
        description="complementary planar position gain"
    )
    declare_planar_odom_velocity_gain = DeclareLaunchArgument(
        "planar_odom_velocity_gain",
        default_value="0.0",
        description="complementary planar velocity gain"
    )
    declare_planar_odom_yaw_gain = DeclareLaunchArgument(
        "planar_odom_yaw_gain",
        default_value="0.0",
        description="complementary planar yaw gain"
    )
    declare_point_filter_num = DeclareLaunchArgument(
        "point_filter_num",
        default_value="0",
        description="LiDAR input decimation override; 0 = keep yaml"
    )
    declare_filter_size_surf = DeclareLaunchArgument(
        "filter_size_surf",
        default_value="0.0",
        description="scan voxel size override in metres; 0 = keep yaml"
    )
    declare_filter_size_map = DeclareLaunchArgument(
        "filter_size_map",
        default_value="0.0",
        description="map voxel size override in metres; 0 = keep yaml"
    )

    return LaunchDescription([
        declare_bot_list,
        declare_rviz_list,
        declare_use_sim_time,
        declare_cluster_size,
        declare_force_uav_num,
        declare_cross_world_transform_mode,
        declare_enable_mutual_observation_update,
        declare_ground_truth_logging_en,
        declare_gravity_constrained_extrinsic_rotation,
        declare_max_abs_match_yaw_deg,
        declare_traj_matching_time_tolerance,
        declare_use_raw_temp_measurement,
        declare_require_measured_trajectory_samples,
        declare_pcd_save_en,
        declare_local_map_pub_en,
        declare_planar_odom_prior_en,
        declare_planar_odom_position_gain,
        declare_planar_odom_velocity_gain,
        declare_planar_odom_yaw_gain,
        declare_point_filter_num,
        declare_filter_size_surf,
        declare_filter_size_map,
        OpaqueFunction(function=launch_everything),
    ])
