import json
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]


class RacerNoProgressGuardTest(unittest.TestCase):
    def test_quadruped_kinodynamic_astar_has_one_second_budget(self):
        compose = (ROOT / "docker-compose.quadruped.yml").read_text()
        self.assertIn('RACER_KINO_MAX_SEARCH_TIME: "1.0"', compose)

        launch = (
            ROOT
            / "src/RACER/swarm_exploration/exploration_manager/launch/"
            "swarm_lio_lidar.launch"
        ).read_text()
        self.assertIn(
            '<arg name="kino_max_search_time" default="1.000"/>', launch
        )
        self.assertEqual(
            launch.count(
                '<arg name="kino_max_search_time" '
                'value="$(arg kino_max_search_time)"/>'
            ),
            6,
        )

        runner = (ROOT / "src/racer_integration/run_ros1_stack.sh").read_text()
        self.assertIn(
            'RACER_KINO_MAX_SEARCH_TIME_VALUE="'
            '${RACER_KINO_MAX_SEARCH_TIME:-1.0}"',
            runner,
        )
        self.assertIn(
            'kino_max_search_time:="${RACER_KINO_MAX_SEARCH_TIME_VALUE}"',
            runner,
        )

        source = (
            ROOT
            / "src/RACER/swarm_exploration/path_searching/src/"
            "kinodynamic_astar.cpp"
        ).read_text()
        self.assertIn(
            'nh.param("search/max_search_time", max_search_time_, 1.0)',
            source,
        )

    def test_lidar_launch_enables_translation_progress_guard(self):
        launch = (
            ROOT
            / "src/RACER/swarm_exploration/exploration_manager/launch/swarm_lio_lidar.launch"
        ).read_text()
        self.assertIn('<arg name="min_translation_progress" default="0.5"/>', launch)
        self.assertEqual(
            launch.count(
                '<arg name="min_translation_progress" '
                'value="$(arg min_translation_progress)"/>'
            ),
            6,
        )

    def test_manager_rejects_viewpoint_and_path_degeneracy(self):
        source = (
            ROOT
            / "src/RACER/swarm_exploration/exploration_manager/src/"
            "fast_exploration_manager.cpp"
        ).read_text()
        self.assertIn("no_progress_viewpoint=1", source)
        self.assertIn("no_progress_path=1", source)
        self.assertGreaterEqual(
            source.count("last_plan_requires_grid_release_ = true"), 3
        )
        self.assertIn(
            "last_plan_requires_grid_release_ = start_planner_clear",
            source,
        )

    def test_grid_quarantine_blacklist_is_removed(self):
        source = (
            ROOT
            / "src/RACER/swarm_exploration/exploration_manager/src/"
            "fast_exploration_fsm.cpp"
        ).read_text()
        header = (
            ROOT
            / "src/RACER/swarm_exploration/exploration_manager/include/"
            "exploration_manager/fast_exploration_fsm.h"
        ).read_text()
        manager = (
            ROOT
            / "src/RACER/swarm_exploration/exploration_manager/src/"
            "fast_exploration_manager.cpp"
        ).read_text()
        planner = (
            ROOT
            / "src/RACER/swarm_exploration/exploration_manager/launch/"
            "single_drone_planner.xml"
        ).read_text()
        # No per-robot HGrid blacklist: no quarantine routine, no failure
        # counters, no TTL parameters, and the lease path no longer writes
        # failed_grid_until_.
        self.assertNotIn("quarantineOwnGrid", source)
        self.assertNotIn("quarantineOwnGrid", header)
        self.assertNotIn("failed_grid_counts_", source)
        self.assertNotIn("failed_grid_counts_", header)
        self.assertNotIn("fsm/failed_grid_ttl", planner)
        self.assertNotIn("fsm/release_grid_after_failures", planner)
        self.assertNotIn("state.failed_grid_until_[grid_id] = dormant_until", manager)
        # A blocked execution still drops only the stale target.
        blocked = source.split("executionBlockedRelease", 1)[0]
        self.assertIn("fd_->target_initialized_ = false", blocked)
        self.assertIn("fd_->active_target_grid_ = -1", blocked)

    def test_hgrid_cost_uses_exact_astar_within_fifteen_metres(self):
        hgrid = (
            ROOT
            / "src/RACER/swarm_exploration/active_perception/src/hgrid.cpp"
        ).read_text()
        planner = (
            ROOT
            / "src/RACER/swarm_exploration/exploration_manager/launch/"
            "single_drone_planner.xml"
        ).read_text()
        self.assertNotIn("dist1 < 5.0", hgrid)
        self.assertIn("dist1 < exact_drone_grid_dist_", hgrid)
        self.assertIn(
            'nh.param("partitioning/exact_drone_grid_dist", exact_drone_grid_dist_, 15.0)',
            hgrid,
        )
        self.assertIn(
            "path_finder_->max_search_time_ = hgrid_astar_max_search_time_", hgrid
        )
        self.assertIn(
            '<param name="partitioning/exact_drone_grid_dist" value="$(arg partitioning_exact_drone_grid_dist)"', planner
        )
        self.assertIn(
            '<param name="partitioning/hgrid_astar_max_search_time" value="0.05"',
            planner,
        )

    def test_fsm_can_start_and_resume_from_hgrid_without_frontier(self):
        source = (
            ROOT
            / "src/RACER/swarm_exploration/exploration_manager/src/"
            "fast_exploration_fsm.cpp"
        ).read_text()
        self.assertGreaterEqual(
            source.count("frontier_count != 0 || !"), 2
        )
        self.assertIn("trigger_no_work", source)
        self.assertIn("active_hgrids=%zu", source)

    def test_lidar_map_marks_the_sensor_origin_free(self):
        source = (
            ROOT
            / "src/RACER/swarm_exploration/plan_env/src/sdf_map.cpp"
        ).read_text()
        point_cloud_update = source.split(
            "void SDFMap::inputPointCloud", 1
        )[1].split(
            "void SDFMap::clearAndInflateLocalMap", 1
        )[0]
        self.assertIn("posToIndex(camera_pos, sensor_idx)", point_cloud_update)
        self.assertIn(
            "setCacheOccupancy(toAddress(sensor_idx), 0)",
            point_cloud_update,
        )

    def test_astar_can_only_escape_the_unknown_start_voxel(self):
        source = (
            ROOT
            / "src/RACER/swarm_exploration/path_searching/src/astar2.cpp"
        ).read_text()
        self.assertGreaterEqual(
            source.count("start_occupancy == SDFMap::UNKNOWN"), 2
        )
        self.assertIn(
            "ckpt_occupancy == SDFMap::UNKNOWN &&\n"
            "                    !start_voxel_checkpoint",
            source,
        )
        # General unknown neighbours remain forbidden; the exception is only
        # for checkpoints that still lie inside the measured start voxel.
        self.assertIn(
            "(!optimistic && nbr_occupancy == SDFMap::UNKNOWN)",
            source,
        )

    def test_quadruped_uses_original_lidar_teammate_observation_initialization(self):
        compose = (ROOT / "docker-compose.quadruped.yml").read_text()
        self.assertIn(
            'SWARM_ENABLE_MUTUAL_OBSERVATION_UPDATE: "true"', compose
        )
        self.assertIn("enable_mutual_observation_update:=true", compose)
        config = (
            ROOT
            / "src/Swarm-LIO2-ROS2-Docker/src/swarm_lio/config/"
            "simulation.yaml"
        ).read_text()
        self.assertIn("mo_gate_residual_norm: 0.50", config)
        self.assertIn("mo_gate_residual_z: 0.25", config)
        self.assertIn("mo_gate_rollpitch_deg: 5.0", config)
        source = (
            ROOT
            / "src/Swarm-LIO2-ROS2-Docker/src/swarm_lio/src/"
            "laserMapping.cpp"
        ).read_text()
        self.assertIn("if (enable_mutual_observation_update) {", source)
        self.assertNotIn("racer_common_frame_ready", source)

    def test_swarm_lio_estimator_does_not_subscribe_to_gazebo_truth(self):
        compose = (ROOT / "docker-compose.quadruped.yml").read_text()
        self.assertIn("ground_truth_logging_en:=false", compose)
        source = (
            ROOT
            / "src/Swarm-LIO2-ROS2-Docker/src/swarm_lio/src/"
            "laserMapping.cpp"
        ).read_text()
        self.assertIn(
            '"evaluation/ground_truth_logging_en", false', source
        )
        self.assertIn(
            "if (ground_truth_logging_en) {", source
        )
        self.assertIn(
            "Estimator-side ground-truth subscription disabled", source
        )

    def test_atrium_lidar_features_stay_above_planar_robot_envelope(self):
        layout_path = (
            ROOT
            / "src/gazebo_sim/worlds/teaching_building_atrium.layout.json"
        )
        layout = json.loads(layout_path.read_text())
        features = [
            box for box in layout["boxes"]
            if box["name"].startswith("atrium_lidar_feature_")
        ]
        self.assertEqual(len(features), 40)
        for feature in features:
            bottom = feature["center"][2] - 0.5 * feature["size"][2]
            self.assertGreaterEqual(bottom, 1.05)

    def test_quadruped_controller_xy_bounds_match_racer_map(self):
        compose = (ROOT / "docker-compose.quadruped.yml").read_text()
        self.assertIn('RACER_BOX_MIN_X: "-50.0"', compose)
        self.assertIn('RACER_BOX_MIN_Y: "-25.0"', compose)
        self.assertIn('RACER_BOX_MAX_X: "50.0"', compose)
        self.assertIn('RACER_BOX_MAX_Y: "25.0"', compose)
        self.assertIn("bounds_lower:='[-50.0,-25.0,0.0]'", compose)
        self.assertIn("bounds_upper:='[50.0,25.0,3.0]'", compose)

    def test_planar_odom_prior_is_ground_only_and_bounded(self):
        compose = (ROOT / "docker-compose.quadruped.yml").read_text()
        # The optional guard remains bounded and ground-only.  The quadruped
        # profile enables its short-window body-speed fault detector so a
        # corridor-direction LIO freeze cannot silently corrupt RACER; UAV
        # profiles and the Swarm-LIO2 default remain pure LiDAR-inertial.
        self.assertIn("planar_odom_prior_en:=true", compose)
        source = (
            ROOT
            / "src/Swarm-LIO2-ROS2-Docker/src/swarm_lio/src/"
            "laserMapping.cpp"
        ).read_text()
        self.assertIn("bool planar_odom_prior_en{false};", source)
        self.assertIn("planar_odom_max_position_correction", source)
        self.assertIn("PLANAR_ODOM_PRIOR_TRIGGER", source)
        self.assertIn("planar_odom_yaw_consistency_gate", source)
        self.assertIn(
            "!gravity_align_finished || !racer_bootstrap_complete.load()",
            source,
        )
        self.assertIn("apply_planar_odom_prior(lidar_end_time)", source)
        config = (
            ROOT
            / "src/Swarm-LIO2-ROS2-Docker/src/swarm_lio/config/"
            "simulation.yaml"
        ).read_text()
        self.assertIn("planar_odom_prior_en: false", config)

    def test_quadruped_contact_proxy_has_anisotropic_wheel_friction(self):
        model = (
            ROOT
            / "src/gazebo_sim/robots/quadruped_mid360/urdf/"
            "quadruped_mid360.xacro"
        ).read_text()
        self.assertIn('<fdir1 value="1 0 0"/>', model)
        self.assertIn('<mu1 value="2.0"/>', model)
        self.assertIn('<mu2 value="0.20"/>', model)

    def test_ground_stagnation_release_is_removed(self):
        source = (
            ROOT
            / "src/RACER/swarm_exploration/exploration_manager/src/"
            "fast_exploration_fsm.cpp"
        ).read_text()
        planner = (
            ROOT
            / "src/RACER/swarm_exploration/exploration_manager/launch/"
            "single_drone_planner.xml"
        ).read_text()
        self.assertNotIn("stagnant_viewpoint", source)
        self.assertNotIn("stagnant_hgrid", source)
        self.assertNotIn("target_stagnation", source)
        self.assertNotIn("fsm/target_stagnation_timeout", planner)
        self.assertNotIn("fsm/grid_stagnation_timeout", planner)

    def test_target_hold_allows_timeout_or_target_reach(self):
        source = (
            ROOT
            / "src/RACER/swarm_exploration/exploration_manager/src/"
            "fast_exploration_fsm.cpp"
        ).read_text()
        target_hold = source.split(
            "bool FastExplorationFSM::targetHoldComplete", 1
        )[1].split(
            "void FastExplorationFSM::executionBlockedCallback", 1
        )[0]
        self.assertIn(
            "return elapsed >= fp_->min_target_execution_time_ || reached;",
            target_hold,
        )
        self.assertNotIn("progress >=", target_hold)

    def test_empty_frontier_fallback_uses_execution_progress_floor(self):
        source = (
            ROOT
            / "src/RACER/swarm_exploration/exploration_manager/src/"
            "fast_exploration_manager.cpp"
        ).read_text()
        empty_frontier = source.split(
            "} else if (frontier_ids.size() == 0)", 1
        )[1].split(
            "} else if (frontier_ids.size() == 1)", 1
        )[0]
        self.assertIn("ep_->min_translation_progress_", empty_frontier)
        self.assertIn("grid_center", empty_frontier)

    def test_quadruped_uses_mid360_density_overrides(self):
        compose = (ROOT / "docker-compose.quadruped.yml").read_text()
        self.assertIn("point_filter_num:=2", compose)
        self.assertIn("filter_size_surf:=0.2", compose)
        self.assertIn("filter_size_map:=0.3", compose)
        # The compact 1.2 x 0.45 m planar initialization ellipse reaches a
        # second covariance singular value around 1.0--1.4.  The residual and
        # yaw gates remain responsible for rejecting bad rigid alignments.
        self.assertIn("traj_matching_start_thresh:=1.0", compose)
        self.assertIn("traj_matching_time_tolerance:=0.06", compose)
        self.assertIn("temp_tracker_lost_timeout:=8.0", compose)

    def test_quadruped_planner_and_raw_lidar_clearances_are_compatible(self):
        compose = (ROOT / "docker-compose.quadruped.yml").read_text()
        self.assertIn('RACER_OBSTACLES_INFLATION: "${RACER_OBSTACLES_INFLATION:-0.60}"', compose)
        self.assertIn('RACER_EUCLIDEAN_INFLATION: "true"', compose)
        self.assertIn("obstacle_hard_clearance:=${RACER_OBSTACLE_HARD_CLEARANCE:-0.65}", compose)
        self.assertIn("obstacle_braking_margin:=${RACER_OBSTACLE_BRAKING_MARGIN:-0.25}", compose)
        self.assertIn("obstacle_repulsion_activation:=1.05", compose)
        self.assertIn("lidar_point_stride:=1", compose)
        self.assertIn("lidar_self_filter_radius:=${RACER_LIDAR_SELF_FILTER_RADIUS:-0.20}", compose)
        self.assertIn("ground_turn_in_place_angle_deg:=35.0", compose)
        self.assertIn("ground_obstacle_z_min:=${RACER_GROUND_OBSTACLE_Z_MIN:--0.10}", compose)

    def test_quadruped_uses_planar_frontier_thresholds(self):
        compose = (ROOT / "docker-compose.quadruped.yml").read_text()
        self.assertIn(
            'RACER_FRONTIER_MIN_CANDIDATE_CLEARANCE: "0.21"', compose
        )
        self.assertIn('RACER_FRONTIER_CLUSTER_MIN: "6"', compose)
        self.assertIn('RACER_FRONTIER_MIN_VISIB_NUM: "5"', compose)

        launch = (
            ROOT
            / "src/RACER/swarm_exploration/exploration_manager/launch/"
            "swarm_lio_lidar.launch"
        ).read_text()
        self.assertEqual(
            launch.count(
                '<arg name="frontier_min_candidate_clearance" '
                'value="$(arg frontier_min_candidate_clearance)"/>'
            ),
            6,
        )

        runner = (ROOT / "src/racer_integration/run_ros1_stack.sh").read_text()
        self.assertIn(
            'frontier_min_candidate_clearance:="'
            '${RACER_FRONTIER_MIN_CANDIDATE_CLEARANCE_VALUE}"',
            runner,
        )
        self.assertIn(
            'frontier_cluster_min:="${RACER_FRONTIER_CLUSTER_MIN_VALUE}"',
            runner,
        )
        self.assertIn(
            'frontier_min_visib_num:="${RACER_FRONTIER_MIN_VISIB_NUM_VALUE}"',
            runner,
        )

    def test_quadruped_closes_supported_wall_pinhole_before_frontier_search(self):
        compose = (ROOT / "docker-compose.quadruped.yml").read_text()
        self.assertIn('RACER_FRONTIER_WALL_GAP_MAX_CELLS: "2"', compose)
        self.assertIn(
            'RACER_FRONTIER_WALL_GAP_MIN_SUPPORT_CELLS: "1"', compose
        )

        runner = (ROOT / "src/racer_integration/run_ros1_stack.sh").read_text()
        self.assertIn(
            'frontier_wall_gap_max_cells:="'
            '${RACER_FRONTIER_WALL_GAP_MAX_CELLS_VALUE}"',
            runner,
        )
        self.assertIn(
            'frontier_wall_gap_min_support_cells:="'
            '${RACER_FRONTIER_WALL_GAP_MIN_SUPPORT_CELLS_VALUE}"',
            runner,
        )

        swarm_launch = (
            ROOT
            / "src/RACER/swarm_exploration/exploration_manager/launch/"
            "swarm_lio_lidar.launch"
        ).read_text()
        self.assertEqual(
            swarm_launch.count(
                '<arg name="frontier_wall_gap_max_cells" '
                'value="$(arg frontier_wall_gap_max_cells)"/>'
            ),
            6,
        )

        finder = (
            ROOT
            / "src/RACER/swarm_exploration/active_perception/src/"
            "frontier_finder.cpp"
        ).read_text()
        self.assertIn("updateWallGapMask();", finder)
        # Plain (2r+1)x(2r+1) dilation of occupied voxels, no gap bridging or
        # L-corner closing.
        self.assertIn("for (int dx = -r; dx <= r; ++dx)", finder)
        self.assertIn("for (int dy = -r; dy <= r; ++dy)", finder)
        self.assertIn("wall_mask_dilation", finder)
        self.assertNotIn("supported(start, -dir)", finder)
        self.assertNotIn("close_corner", finder)
        self.assertIn("isWallGapClosed(nbr)", finder)
        self.assertIn(
            "state == SDFMap::OCCUPIED || isWallGapClosed(idx)", finder
        )
        # The configured threshold rejects one/two isolated frontier voxels:
        # a cluster must contain strictly more than cluster_min cells.
        self.assertIn("expanded.size() > cluster_min_", finder)

        manager = (
            ROOT
            / "src/RACER/swarm_exploration/exploration_manager/src/"
            "fast_exploration_manager.cpp"
        ).read_text()
        self.assertIn("view_gain=%d frontier_cells=%d", manager)

    def test_planar_odom_fault_guard_aborts_worsening_recovery(self):
        mapping = (
            ROOT
            / "src/Swarm-LIO2-ROS2-Docker/src/swarm_lio/src/laserMapping.cpp"
        ).read_text()
        self.assertIn("planar_odom_worsening_count >= 5", mapping)
        self.assertIn("residual_norm >= 1.0", mapping)
        self.assertIn("PLANAR_ODOM_PRIOR_ABORT_WORSENING", mapping)

    def test_quadruped_keeps_unvisited_unknown_hgrids_and_retires_visited_empty_ones(self):
        compose = (ROOT / "docker-compose.quadruped.yml").read_text()
        self.assertIn(
            'RACER_PARTITIONING_MIN_UNKNOWN_RATIO: "${RACER_PARTITIONING_MIN_UNKNOWN_RATIO:-0.05}"', compose
        )
        self.assertIn(
            'RACER_PARTITIONING_REQUIRE_FRONTIER: "false"', compose
        )
        self.assertIn(
            'RACER_PARTITIONING_EXCLUDE_VISITED_NO_FRONTIER: "true"',
            compose,
        )

        source = (
            ROOT
            / "src/RACER/swarm_exploration/active_perception/src/uniform_grid.cpp"
        ).read_text()
        self.assertIn(
            'nh.param("partitioning/require_frontier_for_relevance"', source
        )
        self.assertIn(
            'nh.param("partitioning/exclude_visited_without_frontier"', source
        )
        self.assertIn(
            "return enough_unknown && (!grid.ever_visited_ || has_valid_frontier);",
            source,
        )
        self.assertIn(
            "return enough_unknown || has_valid_frontier;", source
        )

        header = (
            ROOT
            / "src/RACER/swarm_exploration/active_perception/include/"
            "active_perception/uniform_grid.h"
        ).read_text()
        self.assertIn("bool ever_visited_;", header)

        fsm = (
            ROOT
            / "src/RACER/swarm_exploration/exploration_manager/src/"
            "fast_exploration_fsm.cpp"
        ).read_text()
        self.assertIn(
            "expl_manager_->recordTrajectoryPosition(fd_->odom_pos_);", fsm
        )

        launch = (
            ROOT
            / "src/RACER/swarm_exploration/exploration_manager/launch/"
            "swarm_lio_lidar.launch"
        ).read_text()
        self.assertEqual(
            launch.count(
                '<arg name="partitioning_require_frontier_for_relevance" '
                'value="$(arg partitioning_require_frontier_for_relevance)"/>'
            ),
            6,
        )
        self.assertEqual(
            launch.count(
                '<arg name="partitioning_exclude_visited_without_frontier" '
                'value="$(arg partitioning_exclude_visited_without_frontier)"/>'
            ),
            6,
        )

        runner = (ROOT / "src/racer_integration/run_ros1_stack.sh").read_text()
        self.assertIn(
            'partitioning_min_unknown_ratio:="'
            '${RACER_PARTITIONING_MIN_UNKNOWN_RATIO_VALUE}"',
            runner,
        )
        self.assertIn(
            'partitioning_require_frontier_for_relevance:="'
            '${RACER_PARTITIONING_REQUIRE_FRONTIER_VALUE}"',
            runner,
        )
        self.assertIn(
            'partitioning_exclude_visited_without_frontier:="'
            '${RACER_PARTITIONING_EXCLUDE_VISITED_NO_FRONTIER_VALUE}"',
            runner,
        )

    def test_pair_opt_uses_official_id_handoff_and_optional_viewpoint_hints(self):
        message = (
            ROOT
            / "src/RACER/swarm_exploration/exploration_manager/msg/PairOpt.msg"
        ).read_text()
        for field in (
            "int32[] evidence_grid_ids",
            "geometry_msgs/Point[] evidence_viewpoints",
            "float64[] evidence_yaws",
            "int32[] evidence_visible_cells",
            "float64[] evidence_lease_until",
            "uint64[] evidence_versions",
        ):
            self.assertIn(field, message)

        fsm = (
            ROOT
            / "src/RACER/swarm_exploration/exploration_manager/src/"
            "fast_exploration_fsm.cpp"
        ).read_text()
        self.assertIn("action=id_only_transfer", fsm)
        self.assertIn("if (evidence == proposed_remote_evidence.end()) continue", fsm)
        self.assertIn("remote_task_received", fsm)
        self.assertIn("invalid_remote_evidence", fsm)
        self.assertNotIn("remote_evidence_complete", fsm)

        launch = (
            ROOT
            / "src/RACER/swarm_exploration/exploration_manager/launch/"
            "single_drone_planner.xml"
        ).read_text()
        self.assertIn(
            'name="exploration/remote_task_lease_enabled"\n'
            '           value="false"', launch
        )

    def test_idle_vehicle_rebuilds_unreserved_peer_frontier_locally(self):
        message = (
            ROOT
            / "src/RACER/swarm_exploration/exploration_manager/msg/"
            "FrontierShare.msg"
        ).read_text()
        for field in (
            "uint64 claimed_signature",
            "uint64[] signatures",
            "uint8[] reserved",
            "geometry_msgs/Point[] suggested_viewpoints",
            "int32[] cell_offsets",
            "geometry_msgs/Point[] cells",
        ):
            self.assertIn(field, message)

        fsm = (
            ROOT
            / "src/RACER/swarm_exploration/exploration_manager/src/"
            "fast_exploration_fsm.cpp"
        ).read_text()
        self.assertIn("flushPendingChunk", fsm)
        self.assertIn("frontierShareTimerCallback", fsm)
        self.assertIn("frontierShareMsgCallback", fsm)
        self.assertIn("candidate.reserved_ = msg->reserved[index] != 0", fsm)
        self.assertIn("peer_frontier_claim_yield", fsm)

        finder = (
            ROOT
            / "src/RACER/swarm_exploration/active_perception/src/"
            "frontier_finder.cpp"
        ).read_text()
        self.assertIn("getPeerFrontierViewpoints", finder)
        self.assertIn("knownfree(index) && isNeighborUnknown(index)", finder)
        self.assertIn("sampleViewpoints(frontier)", finder)
        self.assertIn("getInflateOccupancy(suggested_pos) != 1", finder)

        manager = (
            ROOT
            / "src/RACER/swarm_exploration/exploration_manager/src/"
            "fast_exploration_manager.cpp"
        ).read_text()
        self.assertIn("selectPeerFrontierRescue", manager)
        self.assertIn("candidate.reserved_", manager)
        self.assertIn("claimed.count(candidate.signature_)", manager)
        self.assertIn("Astar::REACH_END", manager)
        self.assertIn("receivedChunkCount", manager)

        multi_map = (
            ROOT
            / "src/RACER/swarm_exploration/plan_env/src/"
            "multi_map_manager.cpp"
        ).read_text()
        self.assertIn("void MultiMapManager::flushPendingChunk()", multi_map)
        self.assertIn("getOccOfChunk", multi_map)
        self.assertIn("voxel_occ_", multi_map)

        launch = (
            ROOT
            / "src/RACER/swarm_exploration/exploration_manager/launch/"
            "single_drone_planner.xml"
        ).read_text()
        self.assertIn("/swarm_expl/frontier_share_send", launch)
        self.assertIn("/swarm_expl/frontier_share_recv", launch)
        self.assertIn("exploration/peer_frontier_rescue_enabled", launch)

    def test_planar_platform_can_select_euclidean_inflation(self):
        source = (
            ROOT
            / "src/RACER/swarm_exploration/plan_env/src/sdf_map.cpp"
        ).read_text()
        self.assertIn(
            'nh.param("sdf_map/euclidean_inflation", '
            "mp_->euclidean_inflation_, false)",
            source,
        )
        self.assertIn("if (mp_->euclidean_inflation_)", source)
        launch = (
            ROOT
            / "src/RACER/swarm_exploration/exploration_manager/launch/"
            "swarm_lio_lidar.launch"
        ).read_text()
        self.assertEqual(
            launch.count(
                '<arg name="euclidean_inflation" '
                'value="$(arg euclidean_inflation)"/>'
            ),
            6,
        )

    def test_ground_replans_start_from_current_odometry(self):
        fsm = (
            ROOT
            / "src/RACER/swarm_exploration/exploration_manager/src/"
            "fast_exploration_fsm.cpp"
        ).read_text()
        self.assertIn(
            'nh.param(\n'
            '      "fsm/use_measured_replan_start", '
            "fp_->use_measured_replan_start_, false)",
            fsm,
        )
        self.assertIn("fp_->use_measured_replan_start_", fsm)
        self.assertIn("RACER_METRIC measured_replan_start", fsm)
        launch = (
            ROOT
            / "src/RACER/swarm_exploration/exploration_manager/launch/"
            "swarm_lio_lidar.launch"
        ).read_text()
        self.assertEqual(
            launch.count(
                '<arg name="use_measured_replan_start" '
                'value="$(arg use_measured_replan_start)"/>'
            ),
            6,
        )
        compose = (ROOT / "docker-compose.quadruped.yml").read_text()
        self.assertIn(
            'RACER_USE_MEASURED_REPLAN_START: "true"', compose
        )

    def test_emergency_replan_uses_measured_state_and_preserves_tasks(self):
        fsm = (
            ROOT
            / "src/RACER/swarm_exploration/exploration_manager/src/"
            "fast_exploration_fsm.cpp"
        ).read_text()
        manager = (
            ROOT
            / "src/RACER/swarm_exploration/exploration_manager/src/"
            "fast_exploration_manager.cpp"
        ).read_text()
        self.assertIn(
            "fd_->static_state_ || fd_->emergency_replan_", fsm
        )
        self.assertIn(
            "ed_->last_plan_requires_grid_release_ = start_planner_clear",
            manager,
        )
        self.assertIn(
            "all_astar_starts_planner_clear", manager
        )
        self.assertIn(
            "lastSearchStartedPlannerClear()", manager
        )
        controller = (
            ROOT / "src/racer_integration/ros2_racer_controller.py"
        ).read_text()
        self.assertIn(
            "float(self.obstacle_hard_clearance)\n"
            "            if self.ground_unicycle",
            controller,
        )
        compose = (ROOT / "docker-compose.quadruped.yml").read_text()
        self.assertIn('RACER_MIN_TRANSLATION_PROGRESS: "0.18"', compose)

    def test_quadruped_has_one_compact_retroreflector(self):
        model = (
            ROOT
            / "src/gazebo_sim/robots/quadruped_mid360/urdf/"
            "quadruped_mid360.xacro"
        ).read_text()
        self.assertEqual(model.count("<visual><laser_retro>"), 1)
        self.assertIn('reference="retro_beacon_link"', model)
        self.assertIn('<origin xyz="0 0 0.32"', model)

    def test_swarm_teammate_state_subscription_drops_stale_backlog(self):
        source = (
            ROOT
            / "src/Swarm-LIO2-ROS2-Docker/src/swarm_lio/src/MultiUAV.cpp"
        ).read_text()
        self.assertIn("rclcpp::QoS(rclcpp::KeepLast(1)).best_effort()", source)
        self.assertIn("QuadState_peer_subscribers.push_back", source)
        self.assertIn('"/quadstate_to_teammate"', source)
        self.assertIn("quadstate_publish_topic, teammate_state_qos", source)

    def test_viewpoint_unknown_gain_reaches_hgrid_route_cost(self):
        finder_header = (
            ROOT
            / "src/RACER/swarm_exploration/active_perception/include/"
            "active_perception/frontier_finder.h"
        ).read_text()
        finder = (
            ROOT
            / "src/RACER/swarm_exploration/active_perception/src/"
            "frontier_finder.cpp"
        ).read_text()
        grid = (
            ROOT
            / "src/RACER/swarm_exploration/active_perception/src/"
            "uniform_grid.cpp"
        ).read_text()
        hgrid = (
            ROOT
            / "src/RACER/swarm_exploration/active_perception/src/hgrid.cpp"
        ).read_text()
        manager = (
            ROOT
            / "src/RACER/swarm_exploration/exploration_manager/src/"
            "fast_exploration_manager.cpp"
        ).read_text()
        planner = (
            ROOT
            / "src/RACER/swarm_exploration/exploration_manager/launch/"
            "single_drone_planner.xml"
        ).read_text()

        # A valid view must still see the known-free frontier, then the same
        # physical rays continue past it and collect unique UNKNOWN voxels
        # until a known occupied surface terminates the ray.
        self.assertIn("int unknown_gain_{0};", finder_header)
        evaluate = finder.split("int FrontierFinder::evaluateViewpoint", 1)[1]
        self.assertIn("ray_start = cell +", evaluate)
        self.assertIn("state == SDFMap::OCCUPIED", evaluate)
        self.assertIn("state == SDFMap::UNKNOWN", evaluate)
        self.assertIn("unique_unknown.insert(toadr(idx))", evaluate)
        # Gain is a ranking score only; no hard minimum-gain gate drops a
        # geometrically valid viewpoint.
        self.assertNotIn(">= min_unknown_gain_", finder)
        self.assertNotIn("min_unknown_gain_", finder)
        self.assertIn("v1.unknown_gain_ > v2.unknown_gain_", finder)

        # Overlapping view rays/frontiers are de-duplicated before becoming a
        # single HGrid expected-information gain.
        self.assertIn("vector<std::unordered_set<int>> grid_unknown_voxels", grid)
        self.assertIn("grid.viewpoint_unknown_gain_ = unique.size()", grid)
        self.assertIn("hgrid_->inputFrontiers(ed_->averages_", manager)
        self.assertIn("ed_->viewpoint_unknown_voxels_", manager)

        # A fixed additive reward is constant over a tour that visits all
        # tasks. The implementation therefore scales the incoming travel edge
        # and leaves remote, unvisited HGrids without local evidence neutral.
        gain_cost = hgrid.split("double HGrid::applyUnknownGainCost", 1)[1]
        self.assertIn("grid.contained_frontier_ids_.empty()", gain_cost)
        self.assertIn("!grid.ever_visited_", gain_cost)
        self.assertIn("return travel_cost *", gain_cost)
        self.assertIn("unknown_gain_cost_weight_ * (1.0 - normalized_gain)", gain_cost)
        self.assertNotIn("travel_cost + unknown_gain", gain_cost)

        self.assertNotIn('name="frontier/min_unknown_gain"', planner)
        self.assertIn('name="frontier/unknown_gain_max_dist"', planner)
        self.assertIn(
            '<param name="partitioning/min_viewpoint_unknown_gain" value="0"', planner
        )
        self.assertIn('name="partitioning/unknown_gain_cost_weight"', planner)
        self.assertIn("hgrid_gain=%d view_gain=%d", manager)

    def test_frontier_overlap_test_tolerates_half_a_voxel(self):
        finder = (
            ROOT
            / "src/RACER/swarm_exploration/active_perception/src/"
            "frontier_finder.cpp"
        ).read_text()
        overlap = finder.split("bool FrontierFinder::haveOverlap(", 1)[1].split("}", 1)[0]
        # The planar map's frontier boxes (cell centres, z = 0.625) and the
        # updated box (sensor/cloud height, z = 0.64..0.645) never overlapped
        # under the upstream 1e-3 tolerance, so stale frontiers were never
        # removed and isFrontierCovered() never fired.
        self.assertIn("const double slack = 0.5 * resolution_ + 1e-3;", overlap)
        self.assertIn("if (bmin[i] > bmax[i] + slack) return false;", overlap)
        self.assertNotIn("bmax[i] + 1e-3", overlap)


if __name__ == "__main__":
    unittest.main()
