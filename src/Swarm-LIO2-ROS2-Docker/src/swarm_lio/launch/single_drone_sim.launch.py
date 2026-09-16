from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os
import yaml


def flatten_dict(d, parent_key='', sep='/'):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def launch_setup(context, *args, **kwargs):
    # Resolve launch arguments in the current context
    drone_id = LaunchConfiguration('drone_id').perform(context)
    output_mode = LaunchConfiguration('output_mode').perform(context)
    use_sim_time_raw = LaunchConfiguration('use_sim_time').perform(context)
    use_sim_time = str(use_sim_time_raw).lower() in ('true', '1', 'yes', 'on')

    pkg_share = get_package_share_directory('swarm_lio')

    # Load YAML and flatten dict
    params_file = os.path.join(pkg_share, 'config', 'simulation.yaml')
    with open(params_file, 'r') as f:
        yaml_data = yaml.safe_load(f)
    params = flatten_dict(yaml_data.get('ros__parameters', {}))

    # override drone specific params using drone_id
    params['common/drone_id'] = int(drone_id)
    params['common/lid_topic'] = f'bot{drone_id}/lidar_points/points'
    params['common/imu_topic'] = f'bot{drone_id}/imu'
    params['sub_gt_pose_topic'] = f"/bot{drone_id}/gt/odom"
    params['evaluation/ground_truth_logging_en'] = (
        LaunchConfiguration('ground_truth_logging_en').perform(context).lower()
        in ('true', '1', 'yes', 'on')
    )
    params['mapping/planar_odom_topic'] = f"/bot{drone_id}/wheel_odom"
    params['use_sim_time'] = use_sim_time
    params['multiuav/cross_world_transform_mode'] = (
        LaunchConfiguration('cross_world_transform_mode').perform(context)
    )
    params['multiuav/enable_mutual_observation_update'] = (
        LaunchConfiguration('enable_mutual_observation_update').perform(context).lower()
        in ('true', '1', 'yes', 'on')
    )
    params['multiuav/gravity_constrained_extrinsic_rotation'] = (
        LaunchConfiguration('gravity_constrained_extrinsic_rotation').perform(context).lower()
        in ('true', '1', 'yes', 'on')
    )
    params['multiuav/max_abs_match_yaw_deg'] = float(
        LaunchConfiguration('max_abs_match_yaw_deg').perform(context)
    )
    trajectory_excitation = float(
        LaunchConfiguration('traj_matching_start_thresh').perform(context)
    )
    if trajectory_excitation > 0.0:
        params['multiuav/traj_matching_start_thresh'] = trajectory_excitation
    trajectory_time_tolerance = float(
        LaunchConfiguration('traj_matching_time_tolerance').perform(context)
    )
    if trajectory_time_tolerance > 0.0:
        params['multiuav/traj_matching_time_tolerance'] = (
            trajectory_time_tolerance
        )
    average_match_error = float(
        LaunchConfiguration('ave_match_error_thresh').perform(context)
    )
    if average_match_error > 0.0:
        params['multiuav/ave_match_error_thresh'] = average_match_error
    temp_lost_timeout = float(
        LaunchConfiguration('temp_tracker_lost_timeout').perform(context)
    )
    if temp_lost_timeout > 0.0:
        params['multiuav/temp_tracker_lost_timeout'] = temp_lost_timeout
    temp_cluster_gate = float(
        LaunchConfiguration('valid_temp_cluster_dist_thresh').perform(context)
    )
    if temp_cluster_gate > 0.0:
        params['multiuav/valid_temp_cluster_dist_thresh'] = temp_cluster_gate
    params['multiuav/use_raw_temp_measurement_for_traj_matching'] = (
        LaunchConfiguration(
            'use_raw_temp_measurement_for_traj_matching'
        ).perform(context).lower()
        in ('true', '1', 'yes', 'on')
    )
    params['multiuav/trajectory_samples_require_measurement'] = (
        LaunchConfiguration(
            'trajectory_samples_require_measurement'
        ).perform(context).lower()
        in ('true', '1', 'yes', 'on')
    )
    params['multiuav/stop_tracking_after_bootstrap'] = (
        LaunchConfiguration(
            'stop_tracking_after_bootstrap'
        ).perform(context).lower()
        in ('true', '1', 'yes', 'on')
    )
    params['pcd_save/pcd_save_en'] = (
        LaunchConfiguration('pcd_save_en').perform(context).lower()
        in ('true', '1', 'yes', 'on')
    )
    params['publish/local_map_pub_en'] = (
        LaunchConfiguration('local_map_pub_en').perform(context).lower()
        in ('true', '1', 'yes', 'on')
    )
    params['mapping/planar_odom_prior_en'] = (
        LaunchConfiguration('planar_odom_prior_en').perform(context).lower()
        in ('true', '1', 'yes', 'on')
    )
    params['mapping/planar_odom_position_gain'] = float(
        LaunchConfiguration('planar_odom_position_gain').perform(context)
    )
    params['mapping/planar_odom_velocity_gain'] = float(
        LaunchConfiguration('planar_odom_velocity_gain').perform(context)
    )
    params['mapping/planar_odom_yaw_gain'] = float(
        LaunchConfiguration('planar_odom_yaw_gain').perform(context)
    )
    point_filter_num = int(
        LaunchConfiguration('point_filter_num').perform(context)
    )
    if point_filter_num > 0:
        params['mapping/point_filter_num'] = point_filter_num
    filter_size_surf = float(
        LaunchConfiguration('filter_size_surf').perform(context)
    )
    if filter_size_surf > 0.0:
        params['mapping/filter_size_surf'] = filter_size_surf
    filter_size_map = float(
        LaunchConfiguration('filter_size_map').perform(context)
    )
    if filter_size_map > 0.0:
        params['mapping/filter_size_map'] = filter_size_map

    # swarm size must match the actual bot count: "found all teammates" fires at
    # actual_uav_num-1 trackers, so a stale yaml value silently caps discovery.
    uav_num = int(LaunchConfiguration('uav_num').perform(context))
    if uav_num > 0:
        params['multiuav/actual_uav_num'] = uav_num

    # robot-size-dependent gate: predict-region clusters larger than this are
    # DROPPED (MultiUAV.cpp max_dist check). The yaml 0.15 fits a fishbot; a
    # 0.64 m quad exceeds it whenever seen well, starving its tracker.
    cluster_size = float(LaunchConfiguration('cluster_size_thresh').perform(context))
    if cluster_size > 0:
        params['multiuav/valid_cluster_size_thresh'] = cluster_size

    print(
        f"Launching drone {drone_id} with LIDAR topic {params['common/lid_topic']} "
        f"and IMU topic {params['common/imu_topic']}"
    )

    node = Node(
        package='swarm_lio',
        executable='swarm_lio',
        name=f'laserMapping_bot{drone_id}',
        output=output_mode,
        parameters=[params],
        emulate_tty=True,
    )

    return [node]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('drone_id'),
        DeclareLaunchArgument('output_mode', default_value='screen'),
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        DeclareLaunchArgument(
            'ground_truth_logging_en',
            default_value='false',
            description='subscribe to simulator truth for estimator-side logging only',
        ),
        DeclareLaunchArgument('uav_num', default_value='0',
                              description='swarm size; 0 = keep yaml value'),
        DeclareLaunchArgument('cluster_size_thresh', default_value='0',
                              description='valid_cluster_size_thresh override; 0 = keep yaml'),
        DeclareLaunchArgument(
            'cross_world_transform_mode',
            default_value='se3',
            description='cross-world trajectory alignment: se3 or se2_legacy',
        ),
        DeclareLaunchArgument(
            'enable_mutual_observation_update',
            default_value='true',
            description='feed active/passive teammate observations into the local ESIKF',
        ),
        DeclareLaunchArgument(
            'gravity_constrained_extrinsic_rotation',
            default_value='false',
            description='use both LIO gravity frames to robustify SE3 roll/pitch',
        ),
        DeclareLaunchArgument(
            'max_abs_match_yaw_deg',
            default_value='5.0',
            description='initial-heading plausibility gate for trajectory matching',
        ),
        DeclareLaunchArgument(
            'traj_matching_start_thresh',
            default_value='0.0',
            description='trajectory excitation singular-value threshold; 0 keeps yaml',
        ),
        DeclareLaunchArgument(
            'traj_matching_time_tolerance',
            default_value='0.0',
            description='nearest trajectory timestamp association gate in seconds; 0 keeps yaml',
        ),
        DeclareLaunchArgument(
            'ave_match_error_thresh',
            default_value='0.0',
            description='maximum trajectory alignment mean residual; 0 keeps yaml',
        ),
        DeclareLaunchArgument(
            'temp_tracker_lost_timeout',
            default_value='0.0',
            description='temporary high-intensity tracker loss timeout; 0 keeps yaml',
        ),
        DeclareLaunchArgument(
            'valid_temp_cluster_dist_thresh',
            default_value='0.0',
            description='temporary tracker association distance; 0 keeps yaml',
        ),
        DeclareLaunchArgument(
            'use_raw_temp_measurement_for_traj_matching',
            default_value='false',
            description='align time-synchronised raw cluster positions instead of lagged tracker states',
        ),
        DeclareLaunchArgument(
            'trajectory_samples_require_measurement',
            default_value='false',
            description='exclude prediction-only tracker samples from trajectory alignment',
        ),
        DeclareLaunchArgument(
            'stop_tracking_after_bootstrap',
            default_value='true',
            description='stop teammate tracking once the RACER common frame is initialized',
        ),
        DeclareLaunchArgument(
            'pcd_save_en',
            default_value='false',
            description='accumulate registered scans for PCD output',
        ),
        DeclareLaunchArgument(
            'local_map_pub_en',
            default_value='false',
            description='publish the accumulated Swarm-LIO local map',
        ),
        DeclareLaunchArgument(
            'planar_odom_prior_en',
            default_value='false',
            description='fuse a bounded planar wheel/leg-odometry prior',
        ),
        DeclareLaunchArgument(
            'planar_odom_position_gain',
            default_value='0.50',
            description='complementary position gain for the planar prior',
        ),
        DeclareLaunchArgument(
            'planar_odom_velocity_gain',
            default_value='0.0',
            description='complementary velocity gain for the planar prior',
        ),
        DeclareLaunchArgument(
            'planar_odom_yaw_gain',
            default_value='0.0',
            description='complementary yaw gain for the planar prior',
        ),
        DeclareLaunchArgument(
            'point_filter_num',
            default_value='0',
            description='LiDAR input decimation override; 0 keeps yaml',
        ),
        DeclareLaunchArgument(
            'filter_size_surf',
            default_value='0.0',
            description='scan voxel size override in metres; 0 keeps yaml',
        ),
        DeclareLaunchArgument(
            'filter_size_map',
            default_value='0.0',
            description='map voxel size override in metres; 0 keeps yaml',
        ),
        OpaqueFunction(function=launch_setup),
    ])
