#ifndef _EXPL_DATA_H_
#define _EXPL_DATA_H_

#include <Eigen/Eigen>
#include <vector>
#include <cstdint>
#include <unordered_map>
#include <bspline/Bspline.h>

using Eigen::Vector3d;
using std::vector;

namespace fast_planner {
struct FSMData {
  // FSM data
  bool trigger_, have_odom_, static_state_;
  vector<string> state_str_;

  Eigen::Vector3d odom_pos_, odom_vel_;  // odometry state
  Eigen::Quaterniond odom_orient_;
  double odom_yaw_;

  Eigen::Vector3d start_pt_, start_vel_, start_acc_, start_yaw_;  // start state
  vector<Eigen::Vector3d> start_poss;
  bspline::Bspline newest_traj_;

  // Swarm collision avoidance
  bool avoid_collision_, go_back_, emergency_replan_;
  ros::Time fsm_init_time_;
  ros::Time last_check_frontier_time_;
  ros::Time last_plan_attempt_time_;
  ros::Time trigger_ready_time_;
  ros::Time pair_opt_ready_time_;
  bool pair_opt_phase_started_;
  int consecutive_plan_failures_;
  bool replan_same_target_;
  bool target_initialized_;
  bool execution_blocked_;
  bool execution_blocked_handled_;
  int target_block_replans_;
  ros::Time target_started_time_;
  ros::Time target_arrived_time_;
  ros::Time target_near_time_;
  ros::Time progress_anchor_time_;
  Eigen::Vector3d progress_anchor_pos_;
  double target_best_dist_;
  ros::Time target_best_time_;
  ros::Time active_grid_started_time_;
  ros::Time execution_blocked_since_;
  Eigen::Vector3d target_started_pos_;
  Eigen::Vector3d active_target_pos_;
  int active_target_grid_;

  Eigen::Vector3d start_pos_;
};

struct FSMParam {
  double replan_thresh1_;
  double replan_thresh2_;
  double replan_thresh3_;
  double replan_time_;  // second
  double plan_retry_interval_;
  double idle_check_interval_;
  double frontier_update_interval_;
  double trigger_stagger_;
  double pair_opt_timer_stagger_;
  int reachability_filter_max_tasks_;
  double pair_opt_min_keep_fraction_;
  double arrived_no_gain_timeout_;
  double viewpoint_block_ttl_;
  double viewpoint_stagnation_timeout_;
  double viewpoint_stagnation_radius_;
  int plan_fail_block_after_;
  double no_progress_block_timeout_;
  double no_progress_block_distance_;
  // Block the target when the straight-line distance to it has not improved
  // by approach_progress_distance_ for approach_stall_timeout_ seconds while
  // still farther than viewpoint_stagnation_radius_ (moving but never
  // getting closer: door limit cycles, unreachable fallback viewpoints).
  double approach_stall_timeout_;
  // Release (mark visited) a frontier-less grid after this many consecutive
  // planning failures on its fallback target (0 disables).
  int fallback_grid_release_after_;
  double approach_progress_distance_;
  double pair_opt_reshuffle_min_gain_;
  int recovery_batch_size_;
  double min_target_execution_time_;
  double min_target_execution_distance_;
  double execution_blocked_replan_delay_;
  bool use_measured_replan_start_;
  bool enable_planning_visualization_;

  // Swarm
  double attempt_interval_;   // Min interval of opt attempt
  double pair_opt_interval_;  // Min interval of successful pair opt
  double state_freshness_;    // Maximum accepted age of a teammate state
  int repeat_send_num_;
  bool enable_idle_rebalance_;
  double idle_rebalance_min_makespan_gain_;
  double idle_rebalance_max_sum_ratio_;
};

// Evidence attached to an HGrid transferred from another vehicle.  Absence
// of a matching local frontier is UNKNOWN while this lease is active, not a
// proof that the globally discovered task is invalid.
struct RemoteTaskEvidence {
  Eigen::Vector3d viewpoint_;
  double yaw_;
  double lease_until_;
  uint64_t version_;
  int source_drone_id_;
  int expected_visible_cells_;
  uint64_t last_validation_frontier_seq_;
  int validation_misses_;
  int restore_count_;
};

// A frontier observed by another vehicle.  The receiver never executes the
// suggested viewpoint blindly: it rebuilds candidate viewpoints from cells_
// against its own map after RACER's free/occupied ChunkData has been merged.
struct PeerFrontierCandidate {
  int source_drone_id_;
  uint64_t signature_;
  double stamp_;
  bool reserved_;
  vector<Eigen::Vector3d> cells_;
  Eigen::Vector3d suggested_viewpoint_;
  double suggested_yaw_;
  int suggested_visible_cells_;
};

struct PeerFrontierShareState {
  double stamp_;
  uint64_t frontier_epoch_;
  uint64_t claimed_signature_;
  double claim_until_;
  vector<PeerFrontierCandidate> candidates_;
};

struct DroneState {
  Eigen::Vector3d pos_;
  Eigen::Vector3d vel_;
  double yaw_;
  double stamp_;                // Stamp of pos,vel,yaw
  double recent_attempt_time_;  // Stamp of latest opt attempt with any drone

  vector<int> grid_ids_;         // Id of grid tour
  // Route-level version guards pair transactions. Per-grid versions make
  // duplicate ownership converge deterministically after asynchronous state
  // delivery.
  uint64_t assignment_epoch_;
  std::unordered_map<int, uint64_t> grid_epochs_;
  // A failed task remains globally active, but this vehicle is temporarily
  // ineligible for it. Expiry is ROS simulation time in seconds.
  std::unordered_map<int, double> failed_grid_until_;
  // Only authoritative for this process's own route. Peer DroneState packets
  // intentionally remain lightweight; new evidence is transferred in the
  // transactional PairOpt request.
  std::unordered_map<int, RemoteTaskEvidence> remote_tasks_;
  double recent_interact_time_;  // Stamp of latest opt with this drone
};

struct ExplorationData {
  vector<vector<Vector3d>> frontiers_;
  vector<vector<Vector3d>> dead_frontiers_;
  vector<pair<Vector3d, Vector3d>> frontier_boxes_;
  vector<Vector3d> points_;
  vector<Vector3d> averages_;
  vector<Vector3d> views_;
  vector<double> yaws_;
  // Aligned with points_/averages_: unique unknown voxels predicted visible
  // from the selected viewpoint, and their counts.
  vector<vector<int>> viewpoint_unknown_voxels_;
  vector<int> viewpoint_unknown_gains_;
  vector<Vector3d> frontier_tour_;
  vector<vector<Vector3d>> other_tours_;

  vector<int> refined_ids_;
  vector<vector<Vector3d>> n_points_;
  vector<Vector3d> unrefined_points_;
  vector<Vector3d> refined_points_;
  vector<Vector3d> refined_views_;  // points + dir(yaw)
  vector<Vector3d> refined_views1_, refined_views2_;
  vector<Vector3d> refined_tour_;

  Vector3d next_goal_;
  vector<Vector3d> path_next_goal_, kino_path_;
  Vector3d next_pos_;
  double next_yaw_;

  // viewpoint planning
  // vector<Vector4d> views_;
  vector<Vector3d> views_vis1_, views_vis2_;
  vector<Vector3d> centers_, scales_;

  // Swarm, other drones' state
  vector<DroneState> swarm_state_;
  vector<double> pair_opt_stamps_, pair_opt_res_stamps_;
  vector<int> ego_ids_, other_ids_;
  double pair_opt_stamp_;
  uint64_t pair_expected_ego_epoch_;
  uint64_t pair_expected_other_epoch_;
  uint64_t pair_assignment_epoch_;
  bool reallocated_, wait_response_;

  // Coverage planning
  vector<Vector3d> grid_tour_, grid_tour2_;
  // int prev_first_id_;
  vector<int> last_grid_ids_;

  int plan_num_;
  // Diagnostics/recovery state for the latest local trajectory attempt.
  bool last_plan_failed_astar_;
  bool last_plan_requires_grid_release_;
  // Grid id whose frontier-less fallback produced the last target (-1 if the
  // target was a real frontier viewpoint).
  int last_fallback_grid_;
  bool escape_inflation_;
  uint64_t frontier_update_seq_;

  // Peer-frontier rescue is activated only while this vehicle has no locally
  // coverable frontier.  Claims are short lived and are refreshed in the
  // periodic FrontierShare packet.
  std::unordered_map<int, PeerFrontierShareState> peer_frontier_shares_;
  std::unordered_map<uint64_t, double> peer_frontier_failed_until_;
  bool peer_rescue_valid_;
  PeerFrontierCandidate peer_rescue_candidate_;
  Eigen::Vector3d peer_rescue_viewpoint_;
  double peer_rescue_yaw_;
  int peer_rescue_visible_cells_;
  double peer_rescue_claim_until_;
};

struct ExplorationParam {
  // params
  bool refine_local_;
  int refined_num_;
  double refined_radius_;
  int top_view_num_;
  double max_decay_;
  string tsp_dir_;   // resource dir of tsp solver
  string mtsp_dir_;  // resource dir of tsp solver
  double relax_time_;
  int init_plan_num_;
  double optimistic_fallback_range_;
  double astar_timeout_retry_resolution_;
  double min_translation_progress_;
  bool empty_grid_center_first_;
  double empty_grid_center_min_dist_;
  // Release a frontier-less grid instead of chasing a fallback frontier
  // farther than this from the grid centre (<=0 disables).
  double empty_grid_max_frontier_dist_;
  int fast_grid_tour_threshold_;
  int fast_frontier_tour_threshold_;
  int fast_pair_allocation_threshold_;
  bool exact_pair_objective_;
  bool remote_task_lease_enabled_;
  double remote_task_min_lease_;
  double remote_task_max_lease_;
  double remote_task_lease_factor_;
  double remote_task_assumed_speed_;
  double remote_task_arrival_radius_;
  int remote_task_validation_updates_;
  double remote_task_dormant_ttl_;
  bool peer_frontier_rescue_enabled_;
  double peer_frontier_share_interval_;
  double peer_frontier_max_age_;
  double peer_frontier_claim_ttl_;
  double peer_frontier_failure_ttl_;
  double peer_frontier_reserved_radius_;
  int peer_frontier_share_limit_;
  int peer_frontier_cell_limit_;
  int peer_frontier_search_limit_;
  int peer_frontier_max_views_;

  // Swarm
  int drone_num_;
  int drone_id_;
};

}  // namespace fast_planner

#endif
