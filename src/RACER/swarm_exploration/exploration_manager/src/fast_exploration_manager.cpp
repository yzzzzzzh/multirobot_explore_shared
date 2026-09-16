// #include <fstream>
#include <algorithm>
#include <cmath>
#include <exploration_manager/fast_exploration_manager.h>
#include <thread>
#include <iostream>
#include <fstream>
#include <limits>
#include <unordered_set>
#include <active_perception/graph_node.h>
#include <active_perception/graph_search.h>
#include <active_perception/perception_utils.h>
#include <active_perception/frontier_finder.h>
// #include <active_perception/uniform_grid.h>
#include <active_perception/hgrid.h>
#include <plan_env/raycast.h>
#include <plan_env/sdf_map.h>
#include <plan_env/multi_map_manager.h>
#include <plan_env/edt_environment.h>
#include <plan_manage/planner_manager.h>
// #include <lkh_tsp_solver/lkh_interface.h>
// #include <lkh_mtsp_solver/lkh3_interface.h>
#include <lkh_tsp_solver/SolveTSP.h>
#include <lkh_mtsp_solver/SolveMTSP.h>

#include <exploration_manager/expl_data.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <visualization_msgs/Marker.h>

using namespace Eigen;

namespace fast_planner {

namespace {
// Large global tours are replanned frequently and only their first edge is
// executed.  A deterministic nearest-neighbour open tour avoids LKH process
// and file-I/O overhead while leaving small problems on the original solver.
vector<int> greedyOpenTour(
    const Eigen::MatrixXd& mat, const int start_node, const vector<int>& candidates) {
  vector<int> remaining = candidates;
  vector<int> route;
  route.reserve(remaining.size());
  int current = start_node;
  while (!remaining.empty()) {
    auto best = remaining.begin();
    double best_cost = mat(current, *best);
    for (auto it = remaining.begin() + 1; it != remaining.end(); ++it) {
      const double cost = mat(current, *it);
      if (cost < best_cost) {
        best = it;
        best_cost = cost;
      }
    }
    current = *best;
    route.push_back(current);
    remaining.erase(best);
  }
  return route;
}
}  // namespace

// SECTION interfaces for setup and query

FastExplorationManager::FastExplorationManager() {
}

FastExplorationManager::~FastExplorationManager() {
  ViewNode::astar_.reset();
  ViewNode::caster_.reset();
  ViewNode::map_.reset();
}

void FastExplorationManager::initialize(ros::NodeHandle& nh) {
  planner_manager_.reset(new FastPlannerManager);
  planner_manager_->initPlanModules(nh);

  edt_environment_ = planner_manager_->edt_environment_;
  sdf_map_ = edt_environment_->sdf_map_;
  frontier_finder_.reset(new FrontierFinder(edt_environment_, nh));
  // uniform_grid_.reset(new UniformGrid(edt_environment_, nh));
  hgrid_.reset(new HGrid(edt_environment_, nh));
  // view_finder_.reset(new ViewFinder(edt_environment_, nh));

  ed_.reset(new ExplorationData);
  ep_.reset(new ExplorationParam);

  nh.param("exploration/refine_local", ep_->refine_local_, true);
  nh.param("exploration/refined_num", ep_->refined_num_, -1);
  nh.param("exploration/refined_radius", ep_->refined_radius_, -1.0);
  nh.param("exploration/top_view_num", ep_->top_view_num_, -1);
  nh.param("exploration/max_decay", ep_->max_decay_, -1.0);
  nh.param("exploration/tsp_dir", ep_->tsp_dir_, string("null"));
  nh.param("exploration/mtsp_dir", ep_->mtsp_dir_, string("null"));
  nh.param("exploration/relax_time", ep_->relax_time_, 1.0);
  nh.param("exploration/drone_num", ep_->drone_num_, 1);
  nh.param("exploration/drone_id", ep_->drone_id_, 1);
  nh.param("exploration/init_plan_num", ep_->init_plan_num_, 2);
  nh.param("exploration/optimistic_fallback_range",
      ep_->optimistic_fallback_range_, 10.0);
  nh.param("exploration/astar_timeout_retry_resolution",
      ep_->astar_timeout_retry_resolution_, 0.0);
  // Upstream camera exploration may deliberately keep position fixed while
  // changing yaw. A 360-degree LiDAR gains no new coverage from that motion,
  // so the LiDAR launch enables a positive translation threshold while the
  // upstream-compatible default remains disabled.
  nh.param("exploration/empty_grid_center_first", ep_->empty_grid_center_first_, false);
  nh.param("exploration/empty_grid_center_min_dist", ep_->empty_grid_center_min_dist_, 1.5);
  nh.param("exploration/empty_grid_max_frontier_dist", ep_->empty_grid_max_frontier_dist_, -1.0);
  nh.param("exploration/min_translation_progress",
      ep_->min_translation_progress_, 0.0);
  // Zero disables the local greedy approximations and selects RACER's
  // upstream LKH ATSP / ACVRP task-allocation path.
  nh.param("exploration/fast_grid_tour_threshold", ep_->fast_grid_tour_threshold_, 0);
  nh.param("exploration/fast_frontier_tour_threshold", ep_->fast_frontier_tour_threshold_, 0);
  nh.param("exploration/fast_pair_allocation_threshold", ep_->fast_pair_allocation_threshold_, 0);
  nh.param("exploration/exact_pair_objective", ep_->exact_pair_objective_, true);
  nh.param("exploration/remote_task_lease_enabled",
      ep_->remote_task_lease_enabled_, false);
  nh.param("exploration/remote_task_min_lease",
      ep_->remote_task_min_lease_, 15.0);
  nh.param("exploration/remote_task_max_lease",
      ep_->remote_task_max_lease_, 180.0);
  nh.param("exploration/remote_task_lease_factor",
      ep_->remote_task_lease_factor_, 1.5);
  nh.param("exploration/remote_task_assumed_speed",
      ep_->remote_task_assumed_speed_, 0.8);
  nh.param("exploration/remote_task_arrival_radius",
      ep_->remote_task_arrival_radius_, 0.75);
  nh.param("exploration/remote_task_validation_updates",
      ep_->remote_task_validation_updates_, 3);
  nh.param("exploration/remote_task_dormant_ttl",
      ep_->remote_task_dormant_ttl_, 120.0);
  nh.param("exploration/peer_frontier_rescue_enabled",
      ep_->peer_frontier_rescue_enabled_, false);
  nh.param("exploration/peer_frontier_share_interval",
      ep_->peer_frontier_share_interval_, 0.5);
  nh.param("exploration/peer_frontier_max_age",
      ep_->peer_frontier_max_age_, 5.0);
  nh.param("exploration/peer_frontier_claim_ttl",
      ep_->peer_frontier_claim_ttl_, 15.0);
  nh.param("exploration/peer_frontier_failure_ttl",
      ep_->peer_frontier_failure_ttl_, 30.0);
  nh.param("exploration/peer_frontier_reserved_radius",
      ep_->peer_frontier_reserved_radius_, 0.75);
  nh.param("exploration/peer_frontier_share_limit",
      ep_->peer_frontier_share_limit_, 16);
  nh.param("exploration/peer_frontier_cell_limit",
      ep_->peer_frontier_cell_limit_, 96);
  nh.param("exploration/peer_frontier_search_limit",
      ep_->peer_frontier_search_limit_, 8);
  nh.param("exploration/peer_frontier_max_views",
      ep_->peer_frontier_max_views_, 4);

  ed_->swarm_state_.resize(ep_->drone_num_);
  ed_->pair_opt_stamps_.resize(ep_->drone_num_);
  ed_->pair_opt_res_stamps_.resize(ep_->drone_num_);
  for (int i = 0; i < ep_->drone_num_; ++i) {
    ed_->swarm_state_[i].stamp_ = 0.0;
    ed_->swarm_state_[i].assignment_epoch_ = 0;
    ed_->swarm_state_[i].grid_epochs_.clear();
    ed_->swarm_state_[i].failed_grid_until_.clear();
    ed_->swarm_state_[i].remote_tasks_.clear();
    ed_->pair_opt_stamps_[i] = 0.0;
    ed_->pair_opt_res_stamps_[i] = 0.0;
  }
  planner_manager_->swarm_traj_data_.init(ep_->drone_id_, ep_->drone_num_);

  nh.param("exploration/vm", ViewNode::vm_, -1.0);
  nh.param("exploration/am", ViewNode::am_, -1.0);
  nh.param("exploration/yd", ViewNode::yd_, -1.0);
  nh.param("exploration/ydd", ViewNode::ydd_, -1.0);
  nh.param("exploration/w_dir", ViewNode::w_dir_, -1.0);

  ViewNode::astar_.reset(new Astar);
  ViewNode::astar_->init(nh, edt_environment_);
  ViewNode::map_ = sdf_map_;

  double resolution_ = sdf_map_->getResolution();
  Eigen::Vector3d origin, size;
  sdf_map_->getRegion(origin, size);
  ViewNode::caster_.reset(new RayCaster);
  ViewNode::caster_->setParams(resolution_, origin);

  planner_manager_->path_finder_->lambda_heu_ = 1.0;
  // Keep the wall-time budget loaded by Astar::init() from
  // astar/max_search_time.  Upstream overwrote it with 1 s here, which makes
  // a blocked local viewpoint stall one planner callback for hundreds of
  // milliseconds and defeats the launch-time real-time bound.

  tsp_client_ =
      nh.serviceClient<lkh_mtsp_solver::SolveMTSP>("/solve_tsp_" + to_string(ep_->drone_id_), true);
  acvrp_client_ = nh.serviceClient<lkh_mtsp_solver::SolveMTSP>(
      "/solve_acvrp_" + to_string(ep_->drone_id_), true);

  // Swarm
  for (auto& state : ed_->swarm_state_) {
    state.stamp_ = 0.0;
    state.recent_interact_time_ = 0.0;
    state.recent_attempt_time_ = 0.0;
  }
  ed_->last_grid_ids_ = {};
  ed_->reallocated_ = true;
  ed_->pair_opt_stamp_ = 0.0;
  ed_->pair_expected_ego_epoch_ = 0;
  ed_->pair_expected_other_epoch_ = 0;
  ed_->pair_assignment_epoch_ = 0;
  ed_->wait_response_ = false;
  ed_->plan_num_ = 0;
  ed_->last_plan_failed_astar_ = false;
  ed_->last_plan_requires_grid_release_ = false;
  ed_->last_fallback_grid_ = -1;
  ed_->escape_inflation_ = false;
  ed_->frontier_update_seq_ = 0;
  ed_->peer_rescue_valid_ = false;
  ed_->peer_rescue_visible_cells_ = 0;
  ed_->peer_rescue_claim_until_ = 0.0;

  ROS_WARN(
      "RACER_CONFIG_REMOTE drone=%d enabled=%d lease=[%.1f,%.1f] "
      "factor=%.2f speed=%.2f arrival=%.2f validation_updates=%d "
      "dormant_ttl=%.1f",
      ep_->drone_id_, static_cast<int>(ep_->remote_task_lease_enabled_),
      ep_->remote_task_min_lease_, ep_->remote_task_max_lease_,
      ep_->remote_task_lease_factor_, ep_->remote_task_assumed_speed_,
      ep_->remote_task_arrival_radius_, ep_->remote_task_validation_updates_,
      ep_->remote_task_dormant_ttl_);

  // Analysis
  // ofstream fout;
  // fout.open("/home/boboyu/Desktop/RAL_Time/frontier.txt");
  // fout.close();
}

int FastExplorationManager::planExploreMotion(
    const Vector3d& pos, const Vector3d& vel, const Vector3d& acc, const Vector3d& yaw) {
  ros::Time t1 = ros::Time::now();
  auto t2 = t1;
  const ros::WallTime wall_plan_start = ros::WallTime::now();
  ed_->last_plan_failed_astar_ = false;
  ed_->last_plan_requires_grid_release_ = false;
  ed_->last_fallback_grid_ = -1;

  std::cout << "start pos: " << pos.transpose() << ", vel: " << vel.transpose()
            << ", acc: " << acc.transpose() << std::endl;

  // Do global and local tour planning and retrieve the next viewpoint

  ed_->frontier_tour_.clear();
  Vector3d next_pos;
  double next_yaw;

  // A robot with no locally coverable frontier may temporarily assist a peer.
  // The target was regenerated and A*-checked against this robot's merged map
  // in selectPeerFrontierRescue(); planTrajToView performs the final check.
  if (ed_->frontiers_.empty() && ed_->peer_rescue_valid_) {
    next_pos = ed_->peer_rescue_viewpoint_;
    next_yaw = ed_->peer_rescue_yaw_;
    ed_->next_pos_ = next_pos;
    ed_->next_yaw_ = next_yaw;
    const uint64_t signature = ed_->peer_rescue_candidate_.signature_;
    const int source = ed_->peer_rescue_candidate_.source_drone_id_;
    ROS_WARN(
        "RACER_METRIC peer_frontier_plan drone=%d source=%d signature=%llu "
        "target=[%.3f,%.3f,%.3f] visible=%d",
        ep_->drone_id_, source,
        static_cast<unsigned long long>(signature), next_pos[0], next_pos[1],
        next_pos[2], ed_->peer_rescue_visible_cells_);
    const int result =
        planTrajToView(pos, vel, acc, yaw, next_pos, next_yaw);
    if (result != SUCCEED) rejectPeerFrontierRescue("trajectory");
    return result;
  }

  // Find the tour passing through viewpoints
  // Optimal tour is returned as indices of frontier
  vector<int> grid_ids, frontier_ids;
  // findGlobalTour(pos, vel, yaw, indices);
  findGridAndFrontierPath(pos, vel, yaw, grid_ids, frontier_ids);
  const double route_wall_ms =
      (ros::WallTime::now() - wall_plan_start).toSec() * 1000.0;

  if (grid_ids.empty()) {
    ROS_WARN(
        "RACER_METRIC full_plan drone=%d result=%d grids=0 frontiers=0 "
        "route_ms=%.3f viewpoint_ms=0.000 trajectory_ms=0.000 total_ms=%.3f",
        ep_->drone_id_, NO_GRID, route_wall_ms, route_wall_ms);
    return NO_GRID;

    // No grid is assigned to this drone, but keep moving is necessary
    // Move to the closest targets
    ROS_WARN("Empty grid");

    double min_cost = 100000;
    int min_cost_id = -1;
    vector<Vector3d> tmp_path;
    for (int i = 0; i < ed_->averages_.size(); ++i) {
      auto tmp_cost =
          ViewNode::computeCost(pos, ed_->points_[i], yaw[0], ed_->yaws_[i], vel, yaw[1], tmp_path);
      if (tmp_cost < min_cost) {
        min_cost = tmp_cost;
        min_cost_id = i;
      }
    }
    next_pos = ed_->points_[min_cost_id];
    next_yaw = ed_->yaws_[min_cost_id];

  } else if (frontier_ids.size() == 0) {
    // // The assigned grid contains no frontier, find the one closest to the grid
    // ROS_WARN("No frontier in grid");

    Eigen::Vector3d grid_center = ed_->grid_tour_[1];
    ed_->last_fallback_grid_ = grid_ids.front();
    bool remote_task_viewpoint = false;
    const auto& own_state = ed_->swarm_state_[ep_->drone_id_ - 1];
    const auto remote = own_state.remote_tasks_.find(grid_ids.front());
    if (ep_->remote_task_lease_enabled_ &&
        remote != own_state.remote_tasks_.end() &&
        remote->second.lease_until_ > ros::Time::now().toSec()) {
      next_pos = remote->second.viewpoint_;
      next_yaw = remote->second.yaw_;
      remote_task_viewpoint = true;
      ROS_WARN(
          "RACER_METRIC remote_task_viewpoint drone=%d grid=%d source=%d "
          "distance_m=%.3f expected_visible=%d lease_left_s=%.3f",
          ep_->drone_id_, grid_ids.front(),
          remote->second.source_drone_id_, (next_pos - pos).norm(),
          remote->second.expected_visible_cells_,
          remote->second.lease_until_ - ros::Time::now().toSec());
    }

    // Centre-first: an assigned HGrid without any frontier is still mostly
    // unknown; RACER's intent is to advance into it.  Drive to its unknown
    // centroid when that point is free of (inflated) obstacles, not blocked
    // and reasonably far, instead of chasing the globally nearest frontier
    // (which in the planar runs was regularly 10-40 m away and caused
    // block/re-select churn).
    bool center_first = false;
    if (!remote_task_viewpoint && ep_->empty_grid_center_first_) {
      const double center_dist = (grid_center - pos).norm();
      if (center_dist >= ep_->empty_grid_center_min_dist_ &&
          edt_environment_->sdf_map_->isInBox(grid_center) &&
          edt_environment_->sdf_map_->getOccupancy(grid_center) != SDFMap::OCCUPIED &&
          edt_environment_->sdf_map_->getInflateOccupancy(grid_center) != 1 &&
          !frontier_finder_->isViewpointBlocked(grid_center)) {
        next_pos = grid_center;
        const Eigen::Vector3d grid_dir = grid_center - pos;
        next_yaw = grid_dir.head<2>().norm() > 1e-3 ? atan2(grid_dir[1], grid_dir[0]) : yaw[0];
        center_first = true;
        ROS_WARN(
            "RACER_METRIC empty_grid_center_first drone=%d grid=%d distance_m=%.3f",
            ep_->drone_id_, grid_ids.front(), center_dist);
      }
    }
    if (!remote_task_viewpoint && !center_first) {
      double min_cost = 100000;
      int min_cost_id = -1;
      for (int i = 0; i < ed_->points_.size(); ++i) {
        // Upstream rule: the frontier closest to the grid by straight-line
        // distance.  The A*-based variant returned a constant for every
        // frontier the 20 ms search could not reach, so all far frontiers
        // tied and the first one in the list (often 40 m away) was chosen.
        if (frontier_finder_->isViewpointBlocked(ed_->points_[i])) continue;
        double cost = (grid_center - ed_->averages_[i]).norm();
        if (cost < min_cost) {
          min_cost = cost;
          min_cost_id = i;
        }
      }
      if (min_cost_id >= 0 && ep_->empty_grid_max_frontier_dist_ > 0.0 &&
          min_cost > ep_->empty_grid_max_frontier_dist_) {
        // The nearest remaining frontier is nowhere near this frontier-less
        // grid: chasing it (v111: 20-31 m away, blocked by the approach
        // watchdog, then the next far one) is pure churn.  Release the grid
        // for this robot; it is re-evaluated when a frontier appears in it.
        ed_->last_plan_requires_grid_release_ = true;
        hgrid_->markGridVisited(grid_ids.front());
        const double total_wall_ms =
            (ros::WallTime::now() - wall_plan_start).toSec() * 1000.0;
        ROS_ERROR(
            "RACER_RECOVERY drone=%d empty_grid_far_frontier_release=1 grid=%d "
            "nearest_frontier_m=%.3f limit_m=%.3f",
            ep_->drone_id_, grid_ids.front(), min_cost, ep_->empty_grid_max_frontier_dist_);
        ROS_WARN(
            "RACER_METRIC full_plan drone=%d result=%d grids=%zu frontiers=0 "
            "route_ms=%.3f viewpoint_ms=%.3f trajectory_ms=0.000 total_ms=%.3f",
            ep_->drone_id_, FAIL, grid_ids.size(), route_wall_ms,
            total_wall_ms - route_wall_ms, total_wall_ms);
        return FAIL;
      }
      if (min_cost_id >= 0) {
        next_pos = ed_->points_[min_cost_id];
        next_yaw = ed_->yaws_[min_cost_id];
      } else if (frontier_finder_->isViewpointBlocked(grid_center)) {
        // The cell centre itself has been blocked by the watchdogs (repeated
        // planning failures): release the cell for this robot instead of
        // selecting the same unreachable point again.
        ed_->last_plan_requires_grid_release_ = true;
        hgrid_->markGridVisited(grid_ids.front());
        ROS_ERROR(
            "RACER_RECOVERY drone=%d fallback_grid_released=1 grid=%d reason=center_blocked",
            ep_->drone_id_, grid_ids.front());
        return FAIL;
      } else {
        next_pos = grid_center;
        const Eigen::Vector3d grid_dir = grid_center - pos;
        next_yaw = grid_dir.head<2>().norm() > 1e-3
                       ? atan2(grid_dir[1], grid_dir[0])
                       : yaw[0];
      }
    }

    // The upstream fallback selects the global frontier closest to the current
    // HGrid.  With independently updated LiDAR maps that viewpoint can already
    // equal the current pose while the assigned HGrid is still unexplored.
    // Publishing that zero-length path as success resets the FSM failure
    // counter forever.  Preserve the upstream selection normally, but in this
    // degenerate case advance toward the assigned 3-D HGrid center instead.
    // If the center itself is also current, explicitly release this stale grid.
    // Use the same translation floor as the final trajectory guard.  Keeping
    // this at 0.15 m while the ground controller required 0.25 m created a
    // narrow failure band: a 0.156 m global-frontier fallback was accepted
    // here, rejected below, and every HGrid in the route was quarantined in
    // succession.  A near fallback now advances to the assigned grid centre
    // (or releases only a genuinely current/empty grid) in one planning pass.
    const double min_grid_progress =
        std::max(ep_->min_translation_progress_,
            std::max(0.15, 0.5 * edt_environment_->sdf_map_->getResolution()));
    if ((next_pos - pos).norm() < min_grid_progress) {
      if (remote_task_viewpoint) {
        // Wait in IDLE for a genuinely new frontier update. The versioned
        // lease validator will either convert this task to locally ACTIVE or
        // retire it after the configured number of no-frontier updates.
        ROS_WARN(
            "RACER_METRIC remote_task_arrived_wait_validation drone=%d "
            "grid=%d distance_m=%.3f frontier_seq=%llu",
            ep_->drone_id_, grid_ids.front(), (next_pos - pos).norm(),
            static_cast<unsigned long long>(ed_->frontier_update_seq_));
        return NO_GRID;
      }
      const Eigen::Vector3d grid_dir = grid_center - pos;
      if (grid_dir.norm() >= min_grid_progress) {
        next_pos = grid_center;
        next_yaw =
            grid_dir.head<2>().norm() > 1e-3 ? atan2(grid_dir[1], grid_dir[0]) : yaw[0];
        ROS_WARN(
            "RACER_RECOVERY drone=%d empty_frontier_grid_center_fallback=1 "
            "progress_m=%.3f",
            ep_->drone_id_, grid_dir.norm());
      } else {
        ed_->last_plan_requires_grid_release_ = true;
        hgrid_->markGridVisited(grid_ids.front());
        const double total_wall_ms =
            (ros::WallTime::now() - wall_plan_start).toSec() * 1000.0;
        ROS_ERROR(
            "RACER_RECOVERY drone=%d no_progress_empty_frontier=1 "
            "grid=%d target_distance_m=%.3f center_distance_m=%.3f",
            ep_->drone_id_, grid_ids.front(), (next_pos - pos).norm(),
            grid_dir.norm());
        ROS_WARN(
            "RACER_METRIC full_plan drone=%d result=%d grids=%zu frontiers=0 "
            "route_ms=%.3f viewpoint_ms=%.3f trajectory_ms=0.000 total_ms=%.3f",
            ep_->drone_id_, FAIL, grid_ids.size(), route_wall_ms,
            total_wall_ms - route_wall_ms, total_wall_ms);
        return FAIL;
      }
    }

    // // Simply go to the center of the unknown grid
    // next_pos = grid_center;
    // Eigen::Vector3d dir = grid_center - pos;
    // next_yaw = atan2(dir[1], dir[0]);

  } else if (frontier_ids.size() == 1) {
    // ROS_WARN("Single frontier");
    if (ep_->refine_local_) {
      // Single frontier, find the min cost viewpoint for it
      ed_->refined_ids_ = { frontier_ids[0] };
      ed_->unrefined_points_ = { ed_->points_[frontier_ids[0]] };
      ed_->n_points_.clear();
      vector<vector<double>> n_yaws;
      frontier_finder_->getViewpointsInfo(
          pos, { frontier_ids[0] }, ep_->top_view_num_, ep_->max_decay_, ed_->n_points_, n_yaws);

      if (grid_ids.size() <= 1) {
        // Only one grid is assigned
        double min_cost = 100000;
        int min_cost_id = -1;
        vector<Vector3d> tmp_path;
        for (int i = 0; i < ed_->n_points_[0].size(); ++i) {
          auto tmp_cost = ViewNode::computeCost(
              pos, ed_->n_points_[0][i], yaw[0], n_yaws[0][i], vel, yaw[1], tmp_path);
          if (tmp_cost < min_cost) {
            min_cost = tmp_cost;
            min_cost_id = i;
          }
        }
        next_pos = ed_->n_points_[0][min_cost_id];
        next_yaw = n_yaws[0][min_cost_id];
      } else {
        // More than one grid, the next grid is considered for path planning
        // vector<Eigen::Vector3d> grid_pos = { ed_->grid_tour_[2] };
        // Eigen::Vector3d dir = ed_->grid_tour_[2] - ed_->grid_tour_[1];
        // vector<double> grid_yaw = { atan2(dir[1], dir[0]) };

        Eigen::Vector3d grid_pos;
        double grid_yaw;
        if (hgrid_->getNextGrid(grid_ids, grid_pos, grid_yaw)) {
          ed_->n_points_.push_back({ grid_pos });
          n_yaws.push_back({ grid_yaw });
        }

        ed_->refined_points_.clear();
        ed_->refined_views_.clear();
        vector<double> refined_yaws;
        refineLocalTour(pos, vel, yaw, ed_->n_points_, n_yaws, ed_->refined_points_, refined_yaws);
        next_pos = ed_->refined_points_[0];
        next_yaw = refined_yaws[0];
      }
      ed_->refined_points_ = { next_pos };
      ed_->refined_views_ = { next_pos + 2.0 * Vector3d(cos(next_yaw), sin(next_yaw), 0) };
    }
  } else {
    // ROS_WARN("Multiple frontier");
    // More than two frontiers are assigned
    // Do refinement for the next few viewpoints in the global tour
    t1 = ros::Time::now();

    ed_->refined_ids_.clear();
    ed_->unrefined_points_.clear();
    int knum = min(int(frontier_ids.size()), ep_->refined_num_);
    for (int i = 0; i < knum; ++i) {
      auto tmp = ed_->points_[frontier_ids[i]];
      ed_->unrefined_points_.push_back(tmp);
      ed_->refined_ids_.push_back(frontier_ids[i]);
      if ((tmp - pos).norm() > ep_->refined_radius_ && ed_->refined_ids_.size() >= 2) break;
    }

    // Get top N viewpoints for the next K frontiers
    ed_->n_points_.clear();
    vector<vector<double>> n_yaws;
    frontier_finder_->getViewpointsInfo(
        pos, ed_->refined_ids_, ep_->top_view_num_, ep_->max_decay_, ed_->n_points_, n_yaws);

    ed_->refined_points_.clear();
    ed_->refined_views_.clear();
    vector<double> refined_yaws;
    refineLocalTour(pos, vel, yaw, ed_->n_points_, n_yaws, ed_->refined_points_, refined_yaws);
    next_pos = ed_->refined_points_[0];
    next_yaw = refined_yaws[0];

    // Get marker for view visualization
    for (int i = 0; i < ed_->refined_points_.size(); ++i) {
      Vector3d view =
          ed_->refined_points_[i] + 2.0 * Vector3d(cos(refined_yaws[i]), sin(refined_yaws[i]), 0);
      ed_->refined_views_.push_back(view);
    }
    ed_->refined_views1_.clear();
    ed_->refined_views2_.clear();
    for (int i = 0; i < ed_->refined_points_.size(); ++i) {
      vector<Vector3d> v1, v2;
      frontier_finder_->percep_utils_->setPose(ed_->refined_points_[i], refined_yaws[i]);
      frontier_finder_->percep_utils_->getFOV(v1, v2);
      ed_->refined_views1_.insert(ed_->refined_views1_.end(), v1.begin(), v1.end());
      ed_->refined_views2_.insert(ed_->refined_views2_.end(), v2.begin(), v2.end());
    }
    double local_time = (ros::Time::now() - t1).toSec();
    ROS_INFO("Local refine time: %lf", local_time);
  }

  const double viewpoint_wall_ms =
      (ros::WallTime::now() - wall_plan_start).toSec() * 1000.0 - route_wall_ms;

  // RACER's camera model can use a nearby viewpoint for a pure-yaw scan.
  // With the 360-degree LiDAR model that is a no-progress command: A* returns
  // REACH_END for start==goal and the FSM otherwise resets its failure count
  // forever. Reject it before trajectory generation so the normal bounded
  // HGrid-release path can move this UAV on to useful work.
  const double target_translation = (next_pos - pos).norm();
  if (ep_->min_translation_progress_ > 0.0 &&
      target_translation < ep_->min_translation_progress_) {
    ed_->last_plan_requires_grid_release_ = true;
    const double total_wall_ms =
        (ros::WallTime::now() - wall_plan_start).toSec() * 1000.0;
    ROS_ERROR(
        "RACER_RECOVERY drone=%d no_progress_viewpoint=1 grid=%d "
        "frontiers=%zu target_distance_m=%.3f threshold_m=%.3f",
        ep_->drone_id_, grid_ids.front(), frontier_ids.size(),
        target_translation, ep_->min_translation_progress_);
    ROS_WARN(
        "RACER_METRIC full_plan drone=%d result=%d grids=%zu frontiers=%zu "
        "route_ms=%.3f viewpoint_ms=%.3f trajectory_ms=0.000 total_ms=%.3f",
        ep_->drone_id_, FAIL, grid_ids.size(), frontier_ids.size(),
        route_wall_ms, viewpoint_wall_ms, total_wall_ms);
    return FAIL;
  }

  const int active_grid_id = grid_ids.empty() ? -1 : grid_ids.front();
  const int active_frontier_id =
      frontier_ids.empty() ? -1 : frontier_ids.front();
  Eigen::Vector3d active_frontier_average =
      Eigen::Vector3d::Constant(std::numeric_limits<double>::quiet_NaN());
  if (active_frontier_id >= 0 &&
      active_frontier_id < static_cast<int>(ed_->averages_.size())) {
    active_frontier_average = ed_->averages_[active_frontier_id];
  }
  const int active_hgrid_unknown =
      active_grid_id >= 0 ? hgrid_->getUnknownCellsNum(active_grid_id) : 0;
  const int active_hgrid_gain =
      active_grid_id >= 0 ? hgrid_->getViewpointUnknownGain(active_grid_id) : 0;
  const int active_view_gain =
      active_frontier_id >= 0 &&
              active_frontier_id <
                  static_cast<int>(ed_->viewpoint_unknown_gains_.size())
          ? ed_->viewpoint_unknown_gains_[active_frontier_id]
          : 0;
  const int active_frontier_cells =
      active_frontier_id >= 0 &&
              active_frontier_id < static_cast<int>(ed_->frontiers_.size())
          ? static_cast<int>(ed_->frontiers_[active_frontier_id].size())
          : 0;
  ROS_WARN(
      "RACER_METRIC next_view drone=%d grid=%d frontier=%d "
      "start=[%.3f,%.3f,%.3f] target=[%.3f,%.3f,%.3f] "
      "frontier_average=[%.3f,%.3f,%.3f] distance_m=%.3f "
      "hgrid_unknown=%d hgrid_gain=%d view_gain=%d frontier_cells=%d",
      ep_->drone_id_, active_grid_id, active_frontier_id,
      pos[0], pos[1], pos[2], next_pos[0], next_pos[1], next_pos[2],
      active_frontier_average[0], active_frontier_average[1],
      active_frontier_average[2], target_translation, active_hgrid_unknown,
      active_hgrid_gain, active_view_gain, active_frontier_cells);
  std::cout << "Next view: " << next_pos.transpose() << ", " << next_yaw << std::endl;
  ed_->next_pos_ = next_pos;
  ed_->next_yaw_ = next_yaw;
  if (!frontier_ids.empty()) frontier_finder_->setNextFrontier(frontier_ids.front());

  const ros::WallTime wall_traj_start = ros::WallTime::now();
  if (planTrajToView(pos, vel, acc, yaw, next_pos, next_yaw) == FAIL) {
    const double trajectory_wall_ms =
        (ros::WallTime::now() - wall_traj_start).toSec() * 1000.0;
    ROS_WARN(
        "RACER_METRIC full_plan drone=%d result=%d grids=%zu frontiers=%zu "
        "route_ms=%.3f viewpoint_ms=%.3f trajectory_ms=%.3f total_ms=%.3f",
        ep_->drone_id_, FAIL, grid_ids.size(), frontier_ids.size(), route_wall_ms,
        viewpoint_wall_ms, trajectory_wall_ms,
        (ros::WallTime::now() - wall_plan_start).toSec() * 1000.0);
    return FAIL;
  }
  const double trajectory_wall_ms =
      (ros::WallTime::now() - wall_traj_start).toSec() * 1000.0;

  double total = (ros::Time::now() - t2).toSec();
  ROS_INFO("Total time: %lf", total);
  ROS_ERROR_COND(total > 0.1, "Total time too long!!!");
  ROS_WARN(
      "RACER_METRIC full_plan drone=%d result=%d grids=%zu frontiers=%zu "
      "route_ms=%.3f viewpoint_ms=%.3f trajectory_ms=%.3f total_ms=%.3f",
      ep_->drone_id_, SUCCEED, grid_ids.size(), frontier_ids.size(), route_wall_ms,
      viewpoint_wall_ms, trajectory_wall_ms,
      (ros::WallTime::now() - wall_plan_start).toSec() * 1000.0);

  return SUCCEED;
}

int FastExplorationManager::planTrajToView(const Vector3d& pos, const Vector3d& vel,
    const Vector3d& acc, const Vector3d& yaw, const Vector3d& next_pos, const double& next_yaw) {

  // Plan trajectory (position and yaw) to the next viewpoint
  auto t1 = ros::Time::now();
  const ros::WallTime wall_total_start = ros::WallTime::now();
  ed_->last_plan_failed_astar_ = false;
  ed_->last_plan_requires_grid_release_ = false;
  ed_->last_fallback_grid_ = -1;
  ed_->escape_inflation_ =
      edt_environment_->sdf_map_->getInflateOccupancy(pos) == 1 &&
      edt_environment_->sdf_map_->getOccupancy(pos) == SDFMap::FREE;

  // A frontier viewpoint can become occupied between selection and execution
  // after a delayed LiDAR map chunk arrives. Revalidate it immediately and
  // project only to a nearby known-free, non-inflated voxel. This preserves
  // RACER's selected task while avoiding three identical failed replans.
  Vector3d safe_next_pos = next_pos;
  const int requested_goal_occ =
      edt_environment_->sdf_map_->getOccupancy(next_pos);
  const bool requested_goal_inflated =
      edt_environment_->sdf_map_->getInflateOccupancy(next_pos) == 1;
  if (requested_goal_occ == SDFMap::OCCUPIED ||
      requested_goal_inflated) {
    Eigen::Vector3i center_idx;
    edt_environment_->sdf_map_->posToIndex(next_pos, center_idx);
    const double resolution = edt_environment_->sdf_map_->getResolution();
    constexpr double max_safe_goal_shift = 3.0;
    const int max_steps = std::max(
        1, static_cast<int>(
               std::ceil(max_safe_goal_shift / resolution)));
    double best_score = std::numeric_limits<double>::infinity();
    bool found_safe_goal = false;
    for (int radius = 1; radius <= max_steps; ++radius) {
      for (int dx = -radius; dx <= radius; ++dx) {
        for (int dy = -radius; dy <= radius; ++dy) {
          for (int dz = -radius; dz <= radius; ++dz) {
            if (std::max({std::abs(dx), std::abs(dy), std::abs(dz)}) != radius)
              continue;
            const Eigen::Vector3i idx =
                center_idx + Eigen::Vector3i(dx, dy, dz);
            if (!edt_environment_->sdf_map_->isInBox(idx) ||
                edt_environment_->sdf_map_->getOccupancy(idx) != SDFMap::FREE ||
                edt_environment_->sdf_map_->getInflateOccupancy(idx) == 1)
              continue;
            Vector3d candidate;
            edt_environment_->sdf_map_->indexToPos(idx, candidate);
            const double score =
                (candidate - next_pos).squaredNorm() +
                2e-2 * (candidate - pos).squaredNorm();
            if (score < best_score) {
              best_score = score;
              safe_next_pos = candidate;
              found_safe_goal = true;
            }
          }
        }
      }
      if (found_safe_goal) break;
    }
    if (found_safe_goal) {
      ROS_WARN(
          "RACER_RECOVERY drone=%d adjusted_unsafe_goal=1 shift_m=%.3f "
          "requested_occ=%d requested_inflated=%d",
          ep_->drone_id_, (safe_next_pos - next_pos).norm(),
          requested_goal_occ, static_cast<int>(requested_goal_inflated));
    } else {
      ROS_ERROR(
          "RACER_RECOVERY drone=%d unsafe_goal_no_projection=1 "
          "radius_m=%.1f requested_occ=%d requested_inflated=%d",
          ep_->drone_id_, max_safe_goal_shift, requested_goal_occ,
          static_cast<int>(requested_goal_inflated));
    }
  }

  // Compute time lower bound of yaw and use in trajectory generation
  double diff0 = next_yaw - yaw[0];
  double diff1 = fabs(diff0);
  double time_lb = min(diff1, 2 * M_PI - diff1) / ViewNode::yd_;

  // Generate trajectory of x,y,z
  bool goal_unknown =
      (edt_environment_->sdf_map_->getOccupancy(safe_next_pos) == SDFMap::UNKNOWN);
  // bool start_unknown = (edt_environment_->sdf_map_->getOccupancy(pos) == SDFMap::UNKNOWN);
  double goal_distance = (safe_next_pos - pos).norm();
  // An HGrid center/CP is intentionally allowed to lie just beyond the known
  // free-space boundary. If that unknown goal is still inside LiDAR range,
  // searching conservatively can never reach it and only burns the A* budget.
  bool optimistic =
      ed_->plan_num_ < ep_->init_plan_num_ ||
      (goal_unknown && goal_distance <= ep_->optimistic_fallback_range_);
  const ros::WallTime wall_astar_start = ros::WallTime::now();
  const double normal_astar_resolution =
      planner_manager_->path_finder_->getResolution();
  const double coarse_astar_resolution =
      std::max(normal_astar_resolution, ep_->astar_timeout_retry_resolution_);
  // A map callback may change the current voxel between consecutive searches.
  // Release a destination only when every actual A* attempt started in the
  // planner-clear state captured inside Astar::search().
  bool all_astar_starts_planner_clear = true;
  auto bounded_astar_search = [&](bool search_optimistic) {
    planner_manager_->path_finder_->reset();
    planner_manager_->path_finder_->setResolution(normal_astar_resolution);
    int status = planner_manager_->path_finder_->search(
        pos, safe_next_pos, search_optimistic);
    all_astar_starts_planner_clear &=
        planner_manager_->path_finder_->lastSearchStartedPlannerClear();
    if (status != Astar::TIMEOUT ||
        coarse_astar_resolution <= normal_astar_resolution + 1e-6)
      return status;

    // A bounded fine-grid timeout is not evidence that a free-space component
    // is disconnected. Retry once at the configured LiDAR coarse resolution.
    // Every coarse edge is still checked at 0.1 m, and normal resolution is
    // restored immediately. A non-positive launch value disables the retry.
    planner_manager_->path_finder_->reset();
    planner_manager_->path_finder_->setResolution(coarse_astar_resolution);
    status = planner_manager_->path_finder_->search(
        pos, safe_next_pos, search_optimistic);
    all_astar_starts_planner_clear &=
        planner_manager_->path_finder_->lastSearchStartedPlannerClear();
    planner_manager_->path_finder_->setResolution(normal_astar_resolution);
    ROS_WARN(
        "RACER_METRIC astar_timeout_retry drone=%d optimistic=%d "
        "coarse_resolution=%.3f result=%d",
        ep_->drone_id_, static_cast<int>(search_optimistic),
        coarse_astar_resolution, status);
    return status;
  };
  int astar_status = bounded_astar_search(optimistic);
  vector<Vector3d> safe_prefix_path;
  // RACER's viewpoint/HGrid ordering intentionally uses optimistic path
  // estimates, while trajectory generation is conservative after the
  // initialization plans.  In a large 3-D LiDAR map this can select a
  // known-free viewpoint in a different currently-known component.  Dropping
  // that task makes an unseen HGrid circulate forever; executing the complete
  // optimistic path would instead fly through unobserved space.
  //
  // On a conservative failure, search the same collision-aware optimistic
  // route and retain only its continuous known-free prefix.  The vehicle
  // advances to the sensing boundary, extends the map with real LiDAR, and
  // replans there.  Successful conservative paths and the official
  // LKH/ACVRP task ordering are unchanged.
  if (astar_status != Astar::REACH_END && !optimistic) {
    const int optimistic_status = bounded_astar_search(true);
    if (optimistic_status == Astar::REACH_END) {
      const vector<Vector3d> optimistic_path =
          planner_manager_->path_finder_->getPath();
      safe_prefix_path = { optimistic_path.front() };
      Vector3d last_safe = optimistic_path.front();
      bool crossed_unobserved = false;
      const double map_resolution =
          edt_environment_->sdf_map_->getResolution();
      const double sample_step = std::min(0.1, 0.5 * map_resolution);

      for (size_t i = 1; i < optimistic_path.size() && !crossed_unobserved; ++i) {
        const Vector3d delta = optimistic_path[i] - optimistic_path[i - 1];
        const double segment_length = delta.norm();
        if (segment_length < 1e-6) continue;
        const Vector3d direction = delta / segment_length;
        for (double distance = std::min(sample_step, segment_length);
             distance <= segment_length + 1e-6; distance += sample_step) {
          const Vector3d sample =
              optimistic_path[i - 1] +
              direction * std::min(distance, segment_length);
          const bool sample_safe =
              edt_environment_->sdf_map_->getOccupancy(sample) == SDFMap::FREE &&
              edt_environment_->sdf_map_->getInflateOccupancy(sample) == 0;
          if (!sample_safe) {
            crossed_unobserved = true;
            break;
          }
          last_safe = sample;
        }
        if (!crossed_unobserved) {
          last_safe = optimistic_path[i];
          safe_prefix_path.push_back(last_safe);
        }
      }

      const double safe_progress = (last_safe - pos).norm();
      const double min_progress = std::max(0.15, 0.5 * map_resolution);
      if (crossed_unobserved && safe_progress >= min_progress) {
        if ((safe_prefix_path.back() - last_safe).norm() > 1e-3)
          safe_prefix_path.push_back(last_safe);
        safe_next_pos = last_safe;
        goal_distance = safe_progress;
        goal_unknown = false;
        optimistic = true;
        astar_status = Astar::REACH_END;
        ROS_WARN(
            "RACER_RECOVERY drone=%d advanced_known_boundary=1 "
            "progress_m=%.3f requested_distance_m=%.3f",
            ep_->drone_id_, safe_progress, (next_pos - pos).norm());
      } else if (!crossed_unobserved) {
        // The optimistic search found a path that is entirely known-free.
        // The conservative attempt was therefore budget-limited rather than
        // topologically blocked; the recovered path is equally safe.
        safe_prefix_path = optimistic_path;
        astar_status = Astar::REACH_END;
        ROS_WARN(
            "RACER_RECOVERY drone=%d recovered_known_path_after_budget=1 "
            "distance_m=%.3f",
            ep_->drone_id_, goal_distance);
      }
    }
  }
  if (astar_status != Astar::REACH_END) {
    const double astar_wall_ms =
        (ros::WallTime::now() - wall_astar_start).toSec() * 1000.0;
    const bool start_planner_clear = all_astar_starts_planner_clear;
    ed_->last_plan_failed_astar_ = true;
    // An invalid start belongs to the current execution/recovery state, not
    // to the destination HGrid.  Releasing a different first task after
    // every retry emptied the whole route while the vehicle remained at the
    // same wall.  Only quarantine the destination when A* started from
    // planner-clear free space.
    ed_->last_plan_requires_grid_release_ = start_planner_clear;
    ROS_ERROR(
        "No path to next viewpoint; start_occ=%d start_inflated=%d "
        "goal_occ=%d goal_inflated=%d distance=%.3f release_grid=%d",
        planner_manager_->path_finder_->lastStartOccupancy(),
        static_cast<int>(planner_manager_->path_finder_->lastStartInflated()),
        edt_environment_->sdf_map_->getOccupancy(safe_next_pos),
        edt_environment_->sdf_map_->getInflateOccupancy(safe_next_pos),
        goal_distance, static_cast<int>(start_planner_clear));
    ROS_WARN(
        "RACER_METRIC local_traj drone=%d result=%d optimistic=%d "
        "astar_ms=%.3f shorten_ms=0.000 position_ms=0.000 yaw_ms=0.000 "
        "path_m=0.000 total_ms=%.3f",
        ep_->drone_id_, FAIL, static_cast<int>(optimistic), astar_wall_ms,
        (ros::WallTime::now() - wall_total_start).toSec() * 1000.0);
    return FAIL;
  }
  const double astar_wall_ms =
      (ros::WallTime::now() - wall_astar_start).toSec() * 1000.0;
  ed_->path_next_goal_ =
      safe_prefix_path.empty() ? planner_manager_->path_finder_->getPath()
                               : safe_prefix_path;
  const ros::WallTime wall_shorten_start = ros::WallTime::now();
  shortenPath(ed_->path_next_goal_);
  const double shorten_wall_ms =
      (ros::WallTime::now() - wall_shorten_start).toSec() * 1000.0;
  ed_->kino_path_.clear();

  const double radius_far = 7.0;
  const double radius_close = 1.5;
  const double len = Astar::pathLength(ed_->path_next_goal_);
  // Goal projection and path shortening can still collapse a non-degenerate
  // request to one point. Keep this second guard adjacent to the trajectory
  // publisher so no future target-selection branch can reintroduce a
  // successful hover loop.
  if (ep_->min_translation_progress_ > 0.0 &&
      len < ep_->min_translation_progress_) {
    ed_->last_plan_requires_grid_release_ = true;
    ROS_ERROR(
        "RACER_RECOVERY drone=%d no_progress_path=1 path_m=%.3f "
        "threshold_m=%.3f requested_distance_m=%.3f",
        ep_->drone_id_, len, ep_->min_translation_progress_,
        (next_pos - pos).norm());
    ROS_WARN(
        "RACER_METRIC local_traj drone=%d result=%d optimistic=%d "
        "astar_ms=%.3f shorten_ms=%.3f position_ms=0.000 yaw_ms=0.000 "
        "path_m=%.3f total_ms=%.3f",
        ep_->drone_id_, FAIL, static_cast<int>(optimistic), astar_wall_ms,
        shorten_wall_ms, len,
        (ros::WallTime::now() - wall_total_start).toSec() * 1000.0);
    return FAIL;
  }
  const ros::WallTime wall_position_start = ros::WallTime::now();
  if (len < radius_close || optimistic) {
    // Next viewpoint is very close, no need to search kinodynamic path, just use waypoints-based
    // optimization
    planner_manager_->planExploreTraj(ed_->path_next_goal_, vel, acc, time_lb);
    ed_->next_goal_ = safe_next_pos;
    // std::cout << "Close goal." << std::endl;
    if (ed_->plan_num_ < ep_->init_plan_num_) {
      ed_->plan_num_++;
      ROS_WARN("init plan.");
    }
  } else if (len > radius_far) {
    // Next viewpoint is far away, select intermediate goal on geometric path (this also deal with
    // dead end)
    std::cout << "Far goal." << std::endl;
    double len2 = 0.0;
    vector<Eigen::Vector3d> truncated_path = { ed_->path_next_goal_.front() };
    for (int i = 1; i < ed_->path_next_goal_.size() && len2 < radius_far; ++i) {
      auto cur_pt = ed_->path_next_goal_[i];
      len2 += (cur_pt - truncated_path.back()).norm();
      truncated_path.push_back(cur_pt);
    }
    ed_->next_goal_ = truncated_path.back();
    planner_manager_->planExploreTraj(truncated_path, vel, acc, time_lb);
  } else {
    // Search kino path to exactly next viewpoint and optimize
    std::cout << "Mid goal" << std::endl;
    ed_->next_goal_ = safe_next_pos;

    if (!planner_manager_->kinodynamicReplan(
            pos, vel, acc, ed_->next_goal_, Vector3d(0, 0, 0), time_lb)) {
      // The geometric A* path is already collision checked.  A failed
      // kinodynamic seed must not strand the UAV at a reachable viewpoint;
      // use RACER's waypoint optimizer as the bounded fallback.
      ROS_WARN("Kinodynamic search failed; using geometric-path trajectory fallback");
      planner_manager_->planExploreTraj(ed_->path_next_goal_, vel, acc, time_lb);
    } else {
      ed_->kino_path_ = planner_manager_->kino_path_finder_->getKinoTraj(0.02);
    }
  }
  const double position_wall_ms =
      (ros::WallTime::now() - wall_position_start).toSec() * 1000.0;

  if (planner_manager_->local_data_.position_traj_.getTimeSum() < time_lb - 0.5)
    ROS_ERROR("Lower bound not satified!");

  double traj_plan_time = (ros::Time::now() - t1).toSec();

  t1 = ros::Time::now();
  const ros::WallTime wall_yaw_start = ros::WallTime::now();
  planner_manager_->planYawExplore(yaw, next_yaw, true, ep_->relax_time_);
  const double yaw_wall_ms =
      (ros::WallTime::now() - wall_yaw_start).toSec() * 1000.0;
  double yaw_time = (ros::Time::now() - t1).toSec();
  ROS_INFO("Traj: %lf, yaw: %lf", traj_plan_time, yaw_time);
  ROS_WARN(
      "RACER_METRIC local_traj drone=%d result=%d optimistic=%d "
      "astar_ms=%.3f shorten_ms=%.3f position_ms=%.3f yaw_ms=%.3f "
      "path_m=%.3f total_ms=%.3f",
      ep_->drone_id_, SUCCEED, static_cast<int>(optimistic), astar_wall_ms,
      shorten_wall_ms, position_wall_ms, yaw_wall_ms, len,
      (ros::WallTime::now() - wall_total_start).toSec() * 1000.0);

  return SUCCEED;
}

// 更新前沿边界
int FastExplorationManager::updateFrontierStruct(const Eigen::Vector3d& pos) {

  ++ed_->frontier_update_seq_;
  auto t1 = ros::Time::now();
  auto t2 = t1;
  const ros::WallTime wall_total_start = ros::WallTime::now();
  ros::WallTime wall_stage_start = wall_total_start;
  ed_->views_.clear();

  // Search frontiers and group them into clusters 这些得是空闲且紧邻位置的体素，并且会聚成边界簇
  frontier_finder_->searchFrontiers();
  const double search_wall_ms =
      (ros::WallTime::now() - wall_stage_start).toSec() * 1000.0;

  double frontier_time = (ros::Time::now() - t1).toSec();
  t1 = ros::Time::now();
  wall_stage_start = ros::WallTime::now();

  // Find viewpoints (x,y,z,yaw) for all clusters; find the informative ones
  frontier_finder_->computeFrontiersToVisit();

  // Retrieve the updated info
  frontier_finder_->getFrontiers(ed_->frontiers_);
  frontier_finder_->getDormantFrontiers(ed_->dead_frontiers_);
  frontier_finder_->getFrontierBoxes(ed_->frontier_boxes_);

  frontier_finder_->getTopViewpointsInfo(pos, ed_->points_, ed_->yaws_,
      ed_->averages_, &ed_->viewpoint_unknown_voxels_,
      &ed_->viewpoint_unknown_gains_);
  for (int i = 0; i < ed_->points_.size(); ++i)
    ed_->views_.push_back(
        ed_->points_[i] + 2.0 * Vector3d(cos(ed_->yaws_[i]), sin(ed_->yaws_[i]), 0));

  if (ed_->frontiers_.empty()) {
    const double view_wall_ms =
        (ros::WallTime::now() - wall_stage_start).toSec() * 1000.0;
    ROS_WARN(
        "RACER_METRIC frontier drone=%d count=0 search_ms=%.3f "
        "view_ms=%.3f matrix_ms=0.000 total_ms=%.3f",
        ep_->drone_id_, search_wall_ms, view_wall_ms,
        (ros::WallTime::now() - wall_total_start).toSec() * 1000.0);
    if (selectPeerFrontierRescue(pos)) {
      ROS_WARN(
          "RACER_METRIC frontier drone=%d count=0 peer_rescue=1 "
          "source=%d signature=%llu",
          ep_->drone_id_, ed_->peer_rescue_candidate_.source_drone_id_,
          static_cast<unsigned long long>(
              ed_->peer_rescue_candidate_.signature_));
      // A synthetic positive work count keeps the FSM out of IDLE.  No remote
      // cluster is inserted into the local frontier database.
      return 1;
    }
    ed_->peer_rescue_valid_ = false;
    ROS_WARN("No coverable local or peer frontier.");
    return 0;
  }

  // Local information always takes precedence over assistance work.
  if (ed_->peer_rescue_valid_) {
    ROS_WARN(
        "RACER_METRIC peer_frontier_released drone=%d signature=%llu "
        "reason=local_frontier_available",
        ep_->drone_id_, static_cast<unsigned long long>(
            ed_->peer_rescue_candidate_.signature_));
    ed_->peer_rescue_valid_ = false;
    ed_->peer_rescue_claim_until_ = 0.0;
  }

  if (!ed_->viewpoint_unknown_gains_.empty()) {
    const auto gain_range = std::minmax_element(
        ed_->viewpoint_unknown_gains_.begin(),
        ed_->viewpoint_unknown_gains_.end());
    long long gain_sum = 0;
    for (const int gain : ed_->viewpoint_unknown_gains_)
      gain_sum += std::max(0, gain);
    const double gain_mean =
        static_cast<double>(gain_sum) /
        static_cast<double>(ed_->viewpoint_unknown_gains_.size());
    const double planar_cell_area =
        std::pow(sdf_map_->getResolution(), 2);
    ROS_WARN(
        "RACER_METRIC viewpoint_gain drone=%d views=%zu "
        "min_cells=%d mean_cells=%.3f max_cells=%d "
        "mean_planar_area_m2=%.3f max_planar_area_m2=%.3f",
        ep_->drone_id_, ed_->viewpoint_unknown_gains_.size(),
        *gain_range.first, gain_mean, *gain_range.second,
        gain_mean * planar_cell_area,
        static_cast<double>(*gain_range.second) * planar_cell_area);
  }

  const double view_wall_ms =
      (ros::WallTime::now() - wall_stage_start).toSec() * 1000.0;
  double view_time = (ros::Time::now() - t1).toSec();

  t1 = ros::Time::now();
  wall_stage_start = ros::WallTime::now();
  frontier_finder_->updateFrontierCostMatrix();
  const double matrix_wall_ms =
      (ros::WallTime::now() - wall_stage_start).toSec() * 1000.0;

  double mat_time = (ros::Time::now() - t1).toSec();
  double total_time = frontier_time + view_time + mat_time;
  ROS_INFO("Drone %d: frontier t: %lf, viewpoint t: %lf, mat: %lf", ep_->drone_id_, frontier_time,
      view_time, mat_time);

  ROS_INFO("Total t: %lf", (ros::Time::now() - t2).toSec());
  ROS_WARN(
      "RACER_METRIC frontier drone=%d count=%zu search_ms=%.3f "
      "view_ms=%.3f matrix_ms=%.3f total_ms=%.3f",
      ep_->drone_id_, ed_->frontiers_.size(), search_wall_ms, view_wall_ms,
      matrix_wall_ms, (ros::WallTime::now() - wall_total_start).toSec() * 1000.0);
  return ed_->frontiers_.size();
}

void FastExplorationManager::recordTrajectoryPosition(
    const Eigen::Vector3d& pos) {
  hgrid_->recordTrajectoryPosition(pos);
}

uint64_t FastExplorationManager::frontierSignature(
    const Eigen::Vector3d& average) const {
  // Quantization makes independently clustered copies of the same physical
  // boundary converge to one claim key without depending on local IDs.
  constexpr double kResolution = 0.5;
  uint64_t hash = 1469598103934665603ULL;
  for (int axis = 0; axis < 3; ++axis) {
    const int64_t quantized =
        static_cast<int64_t>(std::llround(average[axis] / kResolution));
    const uint64_t word = static_cast<uint64_t>(quantized);
    for (int byte = 0; byte < 8; ++byte) {
      hash ^= (word >> (8 * byte)) & 0xffULL;
      hash *= 1099511628211ULL;
    }
  }
  return hash == 0 ? 1 : hash;
}

bool FastExplorationManager::selectPeerFrontierRescue(
    const Eigen::Vector3d& position) {
  if (!ep_->peer_frontier_rescue_enabled_) return false;

  const double now = ros::Time::now().toSec();
  for (auto it = ed_->peer_frontier_failed_until_.begin();
       it != ed_->peer_frontier_failed_until_.end();) {
    if (it->second <= now)
      it = ed_->peer_frontier_failed_until_.erase(it);
    else
      ++it;
  }

  std::unordered_set<uint64_t> claimed;
  vector<const PeerFrontierCandidate*> candidates;
  for (const auto& share_item : ed_->peer_frontier_shares_) {
    const auto& share = share_item.second;
    if (now - share.stamp_ > ep_->peer_frontier_max_age_) continue;
    if (share.claimed_signature_ != 0 && share.claim_until_ > now)
      claimed.insert(share.claimed_signature_);
    for (const auto& candidate : share.candidates_) {
      if (candidate.reserved_ ||
          ed_->peer_frontier_failed_until_.count(candidate.signature_) > 0)
        continue;
      candidates.push_back(&candidate);
    }
  }

  sort(candidates.begin(), candidates.end(),
      [&position](const PeerFrontierCandidate* first,
          const PeerFrontierCandidate* second) {
        return (first->suggested_viewpoint_ - position).squaredNorm() <
               (second->suggested_viewpoint_ - position).squaredNorm();
      });

  const int search_count = std::min(
      ep_->peer_frontier_search_limit_, static_cast<int>(candidates.size()));
  const PeerFrontierCandidate* best_candidate = nullptr;
  Viewpoint best_view;
  double best_score = std::numeric_limits<double>::infinity();
  double best_path_length = std::numeric_limits<double>::infinity();

  for (int index = 0; index < search_count; ++index) {
    const auto& candidate = *candidates[index];
    if (claimed.count(candidate.signature_) > 0) continue;

    vector<Viewpoint> local_views;
    if (!frontier_finder_->getPeerFrontierViewpoints(candidate.cells_,
            candidate.suggested_viewpoint_, candidate.suggested_yaw_, position,
            ep_->peer_frontier_max_views_, local_views)) {
      continue;
    }

    for (const auto& view : local_views) {
      const double progress = (view.pos_ - position).norm();
      if (progress < std::max(0.25, ep_->min_translation_progress_)) continue;
      hgrid_->path_finder_->reset();
      if (hgrid_->path_finder_->search(position, view.pos_, false) !=
          Astar::REACH_END)
        continue;
      const auto path = hgrid_->path_finder_->getPath();
      const double path_length = Astar::pathLength(path);
      if (!std::isfinite(path_length)) continue;
      const double score = path_length /
                           (1.0 + 0.02 * std::max(0, view.visib_num_) +
                               0.04 * std::max(0, view.unknown_gain_));
      if (score < best_score) {
        best_score = score;
        best_path_length = path_length;
        best_candidate = &candidate;
        best_view = view;
      }
    }
  }

  if (best_candidate == nullptr) return false;
  ed_->peer_rescue_candidate_ = *best_candidate;
  ed_->peer_rescue_viewpoint_ = best_view.pos_;
  ed_->peer_rescue_yaw_ = best_view.yaw_;
  ed_->peer_rescue_visible_cells_ = best_view.visib_num_;
  ed_->peer_rescue_claim_until_ = now + ep_->peer_frontier_claim_ttl_;
  ed_->peer_rescue_valid_ = true;
  const int chunks = sdf_map_->mm_->receivedChunkCount(
      best_candidate->source_drone_id_);
  ROS_WARN(
      "RACER_METRIC peer_frontier_selected drone=%d source=%d signature=%llu "
      "path_m=%.3f visible=%d unknown_gain=%d source_chunks=%d",
      ep_->drone_id_, best_candidate->source_drone_id_,
      static_cast<unsigned long long>(best_candidate->signature_),
      best_path_length, best_view.visib_num_, best_view.unknown_gain_, chunks);
  return true;
}

void FastExplorationManager::rejectPeerFrontierRescue(const char* reason) {
  if (!ed_->peer_rescue_valid_) return;
  const uint64_t signature = ed_->peer_rescue_candidate_.signature_;
  ed_->peer_frontier_failed_until_[signature] =
      ros::Time::now().toSec() + ep_->peer_frontier_failure_ttl_;
  ROS_WARN(
      "RACER_METRIC peer_frontier_rejected drone=%d source=%d "
      "signature=%llu ttl_s=%.1f reason=%s",
      ep_->drone_id_, ed_->peer_rescue_candidate_.source_drone_id_,
      static_cast<unsigned long long>(signature),
      ep_->peer_frontier_failure_ttl_, reason);
  ed_->peer_rescue_valid_ = false;
  ed_->peer_rescue_claim_until_ = 0.0;
}

int FastExplorationManager::checkGridReachability(
    const Eigen::Vector3d& pos, int grid_id, double& path_cost, int max_views) {
  path_cost = std::numeric_limits<double>::infinity();
  vector<int> frontier_ids;
  hgrid_->getFrontiersInGrid({ grid_id }, frontier_ids);
  if (frontier_ids.empty()) return GRID_REACHABILITY_UNKNOWN;

  int checked = 0;
  bool found_candidate = false;
  for (const int frontier_id : frontier_ids) {
    if (checked >= std::max(1, max_views)) break;
    vector<vector<Vector3d>> candidates;
    vector<vector<double>> candidate_yaws;
    frontier_finder_->getViewpointsInfo(
        pos, { frontier_id }, std::max(1, max_views - checked), 0.7,
        candidates, candidate_yaws);
    if (candidates.empty()) continue;
    for (const auto& goal : candidates.front()) {
      if (checked++ >= std::max(1, max_views)) break;
      found_candidate = true;
      hgrid_->path_finder_->reset();
      if (hgrid_->path_finder_->search(pos, goal, false) != Astar::REACH_END)
        continue;
      const auto path = hgrid_->path_finder_->getPath();
      path_cost = std::min(path_cost, Astar::pathLength(path));
    }
  }
  if (std::isfinite(path_cost)) return GRID_REACHABLE;
  return found_candidate ? GRID_UNREACHABLE : GRID_REACHABILITY_UNKNOWN;
}

bool FastExplorationManager::getGridTaskViewpoint(
    const Eigen::Vector3d& start, int grid_id, Eigen::Vector3d& viewpoint,
    double& yaw, int& visible_cells, double& path_length, int max_views) {
  path_length = std::numeric_limits<double>::infinity();
  visible_cells = 0;
  vector<int> frontier_ids;
  hgrid_->getFrontiersInGrid({ grid_id }, frontier_ids);
  if (frontier_ids.empty()) return false;

  int checked = 0;
  // The sender's local occupancy map is not a global routing oracle.  In a
  // distributed run the sender may own a valid frontier/viewpoint while the
  // receiver's current position (or an ACVRP-predicted previous task) lies in
  // space that the sender has not mapped.  Preserve the nearest source-
  // validated viewpoint as evidence even when the sender cannot construct the
  // whole route to it.  The receiver will run A* in its own map and the remote
  // task lease/failed-task TTL will retire an actually unreachable target.
  bool found_geometric_evidence = false;
  double geometric_length = std::numeric_limits<double>::infinity();
  Vector3d geometric_viewpoint = Vector3d::Zero();
  double geometric_yaw = 0.0;
  int geometric_visible_cells = 0;
  auto consider = [&](const Eigen::Vector3d& goal, double goal_yaw,
                      int cells) {
    if (checked++ >= std::max(1, max_views)) return;
    const double direct_length = (start - goal).norm();
    if (std::isfinite(direct_length) && direct_length < geometric_length) {
      found_geometric_evidence = true;
      geometric_length = direct_length;
      geometric_viewpoint = goal;
      geometric_yaw = goal_yaw;
      geometric_visible_cells = cells;
    }
    hgrid_->path_finder_->reset();
    if (hgrid_->path_finder_->search(start, goal, false) !=
        Astar::REACH_END)
      return;
    const auto path = hgrid_->path_finder_->getPath();
    const double length = Astar::pathLength(path);
    if (!std::isfinite(length) || length >= path_length) return;
    path_length = length;
    viewpoint = goal;
    yaw = goal_yaw;
    visible_cells = cells;
  };

  for (const int frontier_id : frontier_ids) {
    if (checked >= std::max(1, max_views)) break;
    const int cells =
        frontier_id >= 0 &&
                frontier_id < static_cast<int>(ed_->frontiers_.size())
            ? static_cast<int>(ed_->frontiers_[frontier_id].size())
            : 0;
    if (frontier_id >= 0 &&
        frontier_id < static_cast<int>(ed_->points_.size()) &&
        frontier_id < static_cast<int>(ed_->yaws_.size())) {
      consider(ed_->points_[frontier_id], ed_->yaws_[frontier_id], cells);
    }
    if (checked >= std::max(1, max_views)) break;

    vector<vector<Vector3d>> candidates;
    vector<vector<double>> candidate_yaws;
    frontier_finder_->getViewpointsInfo(start, { frontier_id },
        std::max(1, max_views - checked), 0.7, candidates, candidate_yaws);
    if (candidates.empty() || candidate_yaws.empty()) continue;
    const int count = std::min(candidates.front().size(),
        candidate_yaws.front().size());
    for (int index = 0;
         index < count && checked < std::max(1, max_views); ++index) {
      consider(candidates.front()[index], candidate_yaws.front()[index],
          cells);
    }
  }
  if (std::isfinite(path_length)) return true;
  if (!found_geometric_evidence) return false;

  viewpoint = geometric_viewpoint;
  yaw = geometric_yaw;
  visible_cells = geometric_visible_cells;
  path_length = geometric_length;
  ROS_WARN(
      "RACER_METRIC remote_viewpoint_geometric_fallback drone=%d grid=%d "
      "direct_m=%.3f visible=%d checked=%d",
      ep_->drone_id_, grid_id, path_length, visible_cells, checked);
  return true;
}

void FastExplorationManager::reconcileRemoteTaskLeases(
    const Eigen::Vector3d& position, const vector<int>& previous_ids,
    vector<int>& current_ids) {
  if (!ep_->remote_task_lease_enabled_) return;

  auto& state = ed_->swarm_state_[ep_->drone_id_ - 1];
  const double now = ros::Time::now().toSec();
  const std::unordered_set<int> previous(
      previous_ids.begin(), previous_ids.end());
  std::unordered_set<int> current(current_ids.begin(), current_ids.end());

  for (auto it = state.remote_tasks_.begin();
       it != state.remote_tasks_.end();) {
    const int grid_id = it->first;
    auto& evidence = it->second;
    if (previous.count(grid_id) == 0) {
      it = state.remote_tasks_.erase(it);
      continue;
    }

    // A divided coarse parent is replaced by newly discovered fine HGrids;
    // retaining the inactive parent would create a stale task ID.
    if (!hgrid_->isGridActive(grid_id)) {
      ROS_WARN(
          "RACER_METRIC remote_task_retired drone=%d grid=%d "
          "reason=partitioned source=%d",
          ep_->drone_id_, grid_id, evidence.source_drone_id_);
      it = state.remote_tasks_.erase(it);
      continue;
    }

    if (hgrid_->isGridLocallyRelevant(grid_id) &&
        hgrid_->gridHasValidFrontier(grid_id)) {
      ROS_WARN(
          "RACER_METRIC remote_task_local_confirmed drone=%d grid=%d "
          "source=%d frontier_seq=%llu",
          ep_->drone_id_, grid_id, evidence.source_drone_id_,
          static_cast<unsigned long long>(ed_->frontier_update_seq_));
      it = state.remote_tasks_.erase(it);
      continue;
    }

    const double distance = (position - evidence.viewpoint_).norm();
    const bool arrived = distance <= ep_->remote_task_arrival_radius_;
    if (arrived &&
        evidence.last_validation_frontier_seq_ !=
            ed_->frontier_update_seq_) {
      evidence.last_validation_frontier_seq_ = ed_->frontier_update_seq_;
      ++evidence.validation_misses_;
      ROS_WARN(
          "RACER_METRIC remote_task_validation drone=%d grid=%d "
          "miss=%d/%d distance_m=%.3f frontier_seq=%llu",
          ep_->drone_id_, grid_id, evidence.validation_misses_,
          ep_->remote_task_validation_updates_, distance,
          static_cast<unsigned long long>(ed_->frontier_update_seq_));
    }

    const bool lease_expired = now >= evidence.lease_until_;
    const bool validation_failed =
        arrived && evidence.validation_misses_ >=
                       std::max(1, ep_->remote_task_validation_updates_);
    if (lease_expired || validation_failed) {
      // Grid blacklisting removed: retire the remote task from this route
      // without marking the HGrid ineligible for future allocation.
      current.erase(grid_id);
      current_ids.erase(
          std::remove(current_ids.begin(), current_ids.end(), grid_id),
          current_ids.end());
      ROS_WARN(
          "RACER_METRIC remote_task_retired drone=%d grid=%d "
          "reason=%s source=%d distance_m=%.3f misses=%d "
          "dormant_ttl_s=%.1f",
          ep_->drone_id_, grid_id,
          validation_failed ? "arrival_no_frontier" : "lease_expired",
          evidence.source_drone_id_, distance, evidence.validation_misses_,
          ep_->remote_task_dormant_ttl_);
      it = state.remote_tasks_.erase(it);
      continue;
    }

    if (current.insert(grid_id).second) current_ids.push_back(grid_id);
    ++evidence.restore_count_;
    if (evidence.restore_count_ == 1 || evidence.restore_count_ % 20 == 0) {
      ROS_WARN(
          "RACER_METRIC remote_task_restored drone=%d grid=%d "
          "source=%d distance_m=%.3f lease_left_s=%.3f restores=%d",
          ep_->drone_id_, grid_id, evidence.source_drone_id_, distance,
          evidence.lease_until_ - now, evidence.restore_count_);
    }
    ++it;
  }

  // HGrid updates may erase and then append tasks. Preserve the route order
  // that ACVRP assigned, including a restored remote first task, and append
  // only genuinely rediscovered tasks afterwards.
  std::unordered_set<int> remaining(current_ids.begin(), current_ids.end());
  vector<int> ordered;
  ordered.reserve(current_ids.size());
  for (const int grid_id : previous_ids) {
    if (remaining.erase(grid_id) > 0) ordered.push_back(grid_id);
  }
  for (const int grid_id : current_ids) {
    if (remaining.erase(grid_id) > 0) ordered.push_back(grid_id);
  }
  current_ids.swap(ordered);
}

void FastExplorationManager::syncOwnGridEpochs(const vector<int>& previous_ids) {
  auto& state = ed_->swarm_state_[ep_->drone_id_ - 1];
  const auto previous_epochs = state.grid_epochs_;
  vector<int> previous = previous_ids;
  vector<int> current = state.grid_ids_;
  sort(previous.begin(), previous.end());
  previous.erase(unique(previous.begin(), previous.end()), previous.end());
  sort(current.begin(), current.end());
  current.erase(unique(current.begin(), current.end()), current.end());

  const bool route_changed = previous != current;
  if (route_changed) {
    const uint64_t clock_epoch =
        (static_cast<uint64_t>(ros::Time::now().toNSec()) << 4) |
        static_cast<uint64_t>(ep_->drone_id_ & 0x0f);
    state.assignment_epoch_ =
        std::max(state.assignment_epoch_ + 1, clock_epoch);
    ROS_WARN(
        "RACER_METRIC ownership_local_update drone=%d epoch=%llu "
        "tasks=%zu->%zu",
        ep_->drone_id_,
        static_cast<unsigned long long>(state.assignment_epoch_),
        previous.size(), current.size());
  }

  std::unordered_set<int> current_set(current.begin(), current.end());
  std::unordered_map<int, uint64_t> removed_coarse_epochs;
  for (const int previous_id : previous_ids) {
    if (current_set.count(previous_id) > 0) continue;
    // Only a removed coarse parent transfers ownership to its children.
    // A fine HGrid that merely becomes relevant again is a new unallocated
    // discovery and must not inherit an unrelated sibling's recent epoch.
    if (hgrid_->getCoarseGridId(previous_id) != previous_id) continue;
    const auto epoch = previous_epochs.find(previous_id);
    if (epoch != previous_epochs.end())
      removed_coarse_epochs[previous_id] = epoch->second;
  }

  std::unordered_map<int, uint64_t> epochs;
  for (const int grid_id : state.grid_ids_) {
    const auto existing = previous_epochs.find(grid_id);
    if (existing != previous_epochs.end()) {
      epochs[grid_id] = existing->second;
      continue;
    }
    const int coarse_id = hgrid_->getCoarseGridId(grid_id);
    const auto parent = removed_coarse_epochs.find(coarse_id);
    if (parent != removed_coarse_epochs.end()) {
      epochs[grid_id] = parent->second;
    } else {
      // A deterministic low epoch makes simultaneous rediscovery converge by
      // the existing owner-ID tie break. Pairwise ACVRP assignments always
      // carry a newer epoch and therefore cannot be stolen by rediscovery.
      epochs[grid_id] = 1;
    }
  }
  state.grid_epochs_.swap(epochs);
}

void FastExplorationManager::findGridAndFrontierPath(const Vector3d& cur_pos,
    const Vector3d& cur_vel, const Vector3d& cur_yaw, vector<int>& grid_ids,
    vector<int>& frontier_ids) {
  auto t1 = ros::Time::now();
  const ros::WallTime wall_total_start = ros::WallTime::now();

  // Select nearby drones according to their states' stamp
  vector<Eigen::Vector3d> positions = { cur_pos };
  // vector<Eigen::Vector3d> velocities = { Eigen::Vector3d(0, 0, 0) };
  vector<Eigen::Vector3d> velocities = { cur_vel };
  vector<double> yaws = { cur_yaw[0] };

  // Partitioning-based tour planning
  vector<int> ego_ids;
  vector<vector<int>> other_ids;
  if (!findGlobalTourOfGrid(positions, velocities, ego_ids, other_ids)) {
    grid_ids = {};
    ROS_WARN(
        "RACER_METRIC route drone=%d grids=0 frontiers=0 grid_ms=%.3f "
        "frontier_ms=0.000 total_ms=%.3f",
        ep_->drone_id_, (ros::WallTime::now() - wall_total_start).toSec() * 1000.0,
        (ros::WallTime::now() - wall_total_start).toSec() * 1000.0);
    return;
  }
  grid_ids = ego_ids;

  const double grid_wall_ms =
      (ros::WallTime::now() - wall_total_start).toSec() * 1000.0;
  double grid_time = (ros::Time::now() - t1).toSec();

  // Frontier-based single drone tour planning
  // Restrict frontier within the first visited grid
  t1 = ros::Time::now();

  vector<int> ftr_ids;
  // uniform_grid_->getFrontiersInGrid(ego_ids[0], ftr_ids);
  hgrid_->getFrontiersInGrid(ego_ids, ftr_ids);
  ROS_INFO("Find frontier tour, %d involved------------", ftr_ids.size());

  if (ftr_ids.empty()) {
    frontier_ids = {};
    ROS_WARN(
        "RACER_METRIC route drone=%d grids=%zu frontiers=0 grid_ms=%.3f "
        "frontier_ms=%.3f total_ms=%.3f",
        ep_->drone_id_, grid_ids.size(), grid_wall_ms,
        (ros::WallTime::now() - wall_total_start).toSec() * 1000.0 - grid_wall_ms,
        (ros::WallTime::now() - wall_total_start).toSec() * 1000.0);
    return;
  }

  // Consider next grid in frontier tour planning
  Eigen::Vector3d grid_pos;
  double grid_yaw;
  vector<Eigen::Vector3d> grid_pos_vec;
  if (hgrid_->getNextGrid(ego_ids, grid_pos, grid_yaw)) {
    grid_pos_vec = { grid_pos };
  }

  findTourOfFrontier(cur_pos, cur_vel, cur_yaw, ftr_ids, grid_pos_vec, frontier_ids);
  const double total_wall_ms =
      (ros::WallTime::now() - wall_total_start).toSec() * 1000.0;
  double ftr_time = (ros::Time::now() - t1).toSec();
  ROS_INFO("Grid tour t: %lf, frontier tour t: %lf.", grid_time, ftr_time);
  ROS_WARN(
      "RACER_METRIC route drone=%d grids=%zu frontiers=%zu grid_ms=%.3f "
      "frontier_ms=%.3f total_ms=%.3f",
      ep_->drone_id_, grid_ids.size(), frontier_ids.size(), grid_wall_ms,
      total_wall_ms - grid_wall_ms, total_wall_ms);
}

void FastExplorationManager::shortenPath(vector<Vector3d>& path) {
  if (path.empty()) {
    ROS_ERROR("Empty path to shorten");
    return;
  }
  // Shorten the tour, only critical intermediate points are reserved.
  const double dist_thresh = 3.0;
  vector<Vector3d> short_tour = { path.front() };
  for (int i = 1; i < path.size() - 1; ++i) {
    if ((path[i] - short_tour.back()).norm() > dist_thresh)
      short_tour.push_back(path[i]);
    else {
      // Add waypoints to shorten path only to avoid collision
      ViewNode::caster_->input(short_tour.back(), path[i + 1]);
      Eigen::Vector3i idx;
      while (ViewNode::caster_->nextId(idx) && ros::ok()) {
        if (edt_environment_->sdf_map_->getInflateOccupancy(idx) == 1 ||
            edt_environment_->sdf_map_->getOccupancy(idx) == SDFMap::UNKNOWN) {
          short_tour.push_back(path[i]);
          break;
        }
      }
    }
  }
  if ((path.back() - short_tour.back()).norm() > 1e-3) short_tour.push_back(path.back());

  // Ensure at least three points in the path
  if (short_tour.size() == 2)
    short_tour.insert(short_tour.begin() + 1, 0.5 * (short_tour[0] + short_tour[1]));
  path = short_tour;
}

void FastExplorationManager::findGlobalTour(const Vector3d& cur_pos, const Vector3d& cur_vel,
    const Vector3d cur_yaw, vector<int>& indices) {
  auto t1 = ros::Time::now();

  // Get cost matrix for current state and clusters
  Eigen::MatrixXd cost_mat;
  frontier_finder_->getFullCostMatrix(cur_pos, cur_vel, cur_yaw, cost_mat);
  const int dimension = cost_mat.rows();
  std::cout << "mat:   " << cost_mat.rows() << std::endl;

  double mat_time = (ros::Time::now() - t1).toSec();
  t1 = ros::Time::now();

  // Initialize TSP par file
  ofstream par_file(ep_->tsp_dir_ + "/drone_" + to_string(ep_->drone_id_) + ".par");
  par_file << "PROBLEM_FILE = " << ep_->tsp_dir_ + "/drone_" + to_string(ep_->drone_id_) + ".tsp\n";
  par_file << "GAIN23 = NO\n";
  par_file << "OUTPUT_TOUR_FILE ="
           << ep_->tsp_dir_ + "/drone_" + to_string(ep_->drone_id_) + ".tou"
                                                                      "r\n";
  par_file << "RUNS = 1\n";
  par_file.close();

  // Write params and cost matrix to problem file
  ofstream prob_file(ep_->tsp_dir_ + "/drone_" + to_string(ep_->drone_id_) + ".tsp");
  // Problem specification part, follow the format of TSPLIB
  string prob_spec;
  prob_spec = "NAME : single\nTYPE : ATSP\nDIMENSION : " + to_string(dimension) +
              "\nEDGE_WEIGHT_TYPE : "
              "EXPLICIT\nEDGE_WEIGHT_FORMAT : FULL_MATRIX\nEDGE_WEIGHT_SECTION\n";
  prob_file << prob_spec;
  // prob_file << "TYPE : TSP\n";
  // prob_file << "EDGE_WEIGHT_FORMAT : LOWER_ROW\n";
  // Problem data part
  const int scale = 100;
  for (int i = 0; i < dimension; ++i) {
    for (int j = 0; j < dimension; ++j) {
      int int_cost = cost_mat(i, j) * scale;
      prob_file << int_cost << " ";
    }
    prob_file << "\n";
  }
  prob_file << "EOF";
  prob_file.close();

  // solveTSPLKH((ep_->tsp_dir_ + "/drone_" + to_string(ep_->drone_id_) + ".par").c_str());
  lkh_tsp_solver::SolveTSP srv;
  if (!tsp_client_.call(srv)) {
    ROS_ERROR("Fail to solve TSP.");
    return;
  }

  // Read optimal tour from the tour section of result file
  ifstream res_file(ep_->tsp_dir_ + "/drone_" + to_string(ep_->drone_id_) + ".tour");
  string res;
  while (getline(res_file, res)) {
    // Go to tour section
    if (res.compare("TOUR_SECTION") == 0) break;
  }

  // Read path for ATSP formulation
  while (getline(res_file, res)) {
    // Read indices of frontiers in optimal tour
    int id = stoi(res);
    if (id == 1)  // Ignore the current state
      continue;
    if (id == -1) break;
    indices.push_back(id - 2);  // Idx of solver-2 == Idx of frontier
  }

  res_file.close();

  std::cout << "Tour " << ep_->drone_id_ << ": ";
  for (auto id : indices) std::cout << id << ", ";
  std::cout << "" << std::endl;

  // Get the path of optimal tour from path matrix
  frontier_finder_->getPathForTour(cur_pos, indices, ed_->frontier_tour_);

  double tsp_time = (ros::Time::now() - t1).toSec();
  ROS_INFO("Cost mat: %lf, TSP: %lf", mat_time, tsp_time);

  // if (tsp_time > 0.1) ROS_BREAK();
}

void FastExplorationManager::refineLocalTour(const Vector3d& cur_pos, const Vector3d& cur_vel,
    const Vector3d& cur_yaw, const vector<vector<Vector3d>>& n_points,
    const vector<vector<double>>& n_yaws, vector<Vector3d>& refined_pts,
    vector<double>& refined_yaws) {
  double create_time, search_time, parse_time;
  auto t1 = ros::Time::now();

  // Create graph for viewpoints selection
  GraphSearch<ViewNode> g_search;
  vector<ViewNode::Ptr> last_group, cur_group;

  // Add the current state
  ViewNode::Ptr first(new ViewNode(cur_pos, cur_yaw[0]));
  first->vel_ = cur_vel;
  g_search.addNode(first);
  last_group.push_back(first);
  ViewNode::Ptr final_node;

  // Add viewpoints
  std::cout << "Local refine graph size: 1, ";
  for (int i = 0; i < n_points.size(); ++i) {
    // Create nodes for viewpoints of one frontier
    for (int j = 0; j < n_points[i].size(); ++j) {
      ViewNode::Ptr node(new ViewNode(n_points[i][j], n_yaws[i][j]));
      g_search.addNode(node);
      // Connect a node to nodes in last group
      for (auto nd : last_group) g_search.addEdge(nd->id_, node->id_);
      cur_group.push_back(node);

      // Only keep the first viewpoint of the last local frontier
      if (i == n_points.size() - 1) {
        final_node = node;
        break;
      }
    }
    // Store nodes for this group for connecting edges
    std::cout << cur_group.size() << ", ";
    last_group = cur_group;
    cur_group.clear();
  }
  std::cout << "" << std::endl;
  create_time = (ros::Time::now() - t1).toSec();
  t1 = ros::Time::now();

  // Search optimal sequence
  vector<ViewNode::Ptr> path;
  g_search.DijkstraSearch(first->id_, final_node->id_, path);

  search_time = (ros::Time::now() - t1).toSec();
  t1 = ros::Time::now();

  // Return searched sequence
  for (int i = 1; i < path.size(); ++i) {
    refined_pts.push_back(path[i]->pos_);
    refined_yaws.push_back(path[i]->yaw_);
  }

  // Extract optimal local tour (for visualization)
  ed_->refined_tour_.clear();
  ed_->refined_tour_.push_back(cur_pos);
  ViewNode::astar_->lambda_heu_ = 1.0;
  ViewNode::astar_->setResolution(0.2);
  for (auto pt : refined_pts) {
    vector<Vector3d> path;
    if (ViewNode::searchPath(ed_->refined_tour_.back(), pt, path))
      ed_->refined_tour_.insert(ed_->refined_tour_.end(), path.begin(), path.end());
    else
      ed_->refined_tour_.push_back(pt);
  }
  ViewNode::astar_->lambda_heu_ = 10000;

  parse_time = (ros::Time::now() - t1).toSec();
  // ROS_WARN("create: %lf, search: %lf, parse: %lf", create_time, search_time, parse_time);
}

void FastExplorationManager::allocateGrids(const vector<Eigen::Vector3d>& positions,
    const vector<Eigen::Vector3d>& velocities, const vector<vector<int>>& first_ids,
    const vector<vector<int>>& second_ids, const vector<int>& grid_ids, vector<int>& ego_ids,
    vector<int>& other_ids) {
  // ROS_INFO("Allocate grid.");

  auto t1 = ros::Time::now();
  auto t2 = t1;

  if (grid_ids.size() == 1) {  // Only one grid, no need to run ACVRP
    auto pt = hgrid_->getCenter(grid_ids.front());
    // double d1 = (positions[0] - pt).norm();
    // double d2 = (positions[1] - pt).norm();
    vector<Eigen::Vector3d> path;
    double d1 = ViewNode::computeCost(positions[0], pt, 0, 0, Eigen::Vector3d(0, 0, 0), 0, path);
    double d2 = ViewNode::computeCost(positions[1], pt, 0, 0, Eigen::Vector3d(0, 0, 0), 0, path);
    if (d1 < d2) {
      ego_ids = grid_ids;
      other_ids = {};
    } else {
      ego_ids = {};
      other_ids = grid_ids;
    }
    return;
  }

  Eigen::MatrixXd mat;
  // uniform_grid_->getCostMatrix(positions, velocities, prev_first_ids, grid_ids, mat);
  hgrid_->getCostMatrix(positions, velocities, first_ids, second_ids, grid_ids, mat);

  if (ep_->fast_pair_allocation_threshold_ > 0 &&
      static_cast<int>(grid_ids.size()) > ep_->fast_pair_allocation_threshold_) {
    // Preserve each vehicle's in-progress first task when it is still active,
    // then grow the two routes by minimum projected makespan.  This keeps
    // execution continuity and reduces a large ACVRP call to O(N^2).
    ego_ids.clear();
    other_ids.clear();
    vector<vector<int>*> routes = { &ego_ids, &other_ids };
    vector<bool> assigned(grid_ids.size(), false);
    vector<int> current_node = { 1, 2 };
    vector<double> route_cost = { 0.0, 0.0 };

    for (int drone = 0; drone < 2; ++drone) {
      if (drone >= first_ids.size() || first_ids[drone].empty()) continue;
      for (int i = 0; i < grid_ids.size(); ++i) {
        if (assigned[i] || grid_ids[i] != first_ids[drone].front()) continue;
        routes[drone]->push_back(grid_ids[i]);
        current_node[drone] = 3 + i;
        route_cost[drone] = mat(1 + drone, current_node[drone]);
        assigned[i] = true;
        break;
      }
    }

    for (int remaining = grid_ids.size() - ego_ids.size() - other_ids.size(); remaining > 0;
         --remaining) {
      int best_drone = -1, best_grid = -1;
      double best_score = std::numeric_limits<double>::infinity();
      for (int i = 0; i < grid_ids.size(); ++i) {
        if (assigned[i]) continue;
        const int task_node = 3 + i;
        for (int drone = 0; drone < 2; ++drone) {
          const double projected = route_cost[drone] + mat(current_node[drone], task_node);
          const double other = route_cost[1 - drone];
          const double score = std::max(projected, other) + 0.02 * (projected + other);
          if (score < best_score) {
            best_score = score;
            best_drone = drone;
            best_grid = i;
          }
        }
      }
      if (best_grid < 0) break;
      const int task_node = 3 + best_grid;
      route_cost[best_drone] += mat(current_node[best_drone], task_node);
      current_node[best_drone] = task_node;
      routes[best_drone]->push_back(grid_ids[best_grid]);
      assigned[best_grid] = true;
    }
    ROS_INFO("Fast pair allocation: %zu/%zu HGrid tasks", ego_ids.size(), other_ids.size());
    return;
  }

  // int unknown = hgrid_->getTotalUnknwon();
  int unknown;

  double mat_time = (ros::Time::now() - t1).toSec();

  // Find optimal path through AmTSP
  t1 = ros::Time::now();
  const int dimension = mat.rows();
  const int drone_num = positions.size();

  vector<int> demands;
  int total_demand = 0;
  int largest_demand = 0;
  for (int i = 0; i < grid_ids.size(); ++i) {
    int unum = hgrid_->getUnknownCellsNum(grid_ids[i]);
    // LKH's ACVRP transformation can omit or duplicate zero-demand
    // customers. A nearly observed HGrid is still a real frontier task, so
    // retain the upstream unknown-volume weight while assigning every task a
    // strictly positive integer demand.
    const int demand = max(1, static_cast<int>(ceil(unum * 0.1)));
    demands.push_back(demand);
    total_demand += demand;
    largest_demand = max(largest_demand, demand);
    // std::cout << "Grid " << i << ": " << unum << std::endl;
  }
  // Preserve RACER's 75%-of-total per-vehicle capacity, rounded up and made
  // feasible for the largest individual HGrid.
  const int capacity =
      max(largest_demand, static_cast<int>(ceil(total_demand * 0.75)));

  // int prob_type;
  // if (grid_ids.size() >= 3)
  //   prob_type = 2;  // Use ACVRP
  // else
  //   prob_type = 1;  // Use AmTSP

  const int prob_type = 2;

  // Create problem file--------------------------
  ofstream file(ep_->mtsp_dir_ + "/amtsp3_" + to_string(ep_->drone_id_) + ".atsp");
  file << "NAME : pairopt\n";

  if (prob_type == 1)
    file << "TYPE : ATSP\n";
  else if (prob_type == 2)
    file << "TYPE : ACVRP\n";

  file << "DIMENSION : " + to_string(dimension) + "\n";
  file << "EDGE_WEIGHT_TYPE : EXPLICIT\n";
  file << "EDGE_WEIGHT_FORMAT : FULL_MATRIX\n";

  if (prob_type == 2) {
    file << "CAPACITY : " + to_string(capacity) + "\n";   // ACVRP
    file << "VEHICLES : " + to_string(drone_num) + "\n";  // ACVRP
  }

  // Cost matrix
  file << "EDGE_WEIGHT_SECTION\n";
  for (int i = 0; i < dimension; ++i) {
    for (int j = 0; j < dimension; ++j) {
      int int_cost = 100 * mat(i, j);
      file << int_cost << " ";
    }
    file << "\n";
  }

  if (prob_type == 2) {  // Demand section, ACVRP only
    file << "DEMAND_SECTION\n";
    file << "1 0\n";
    for (int i = 0; i < drone_num; ++i) {
      file << to_string(i + 2) + " 0\n";
    }
    for (int i = 0; i < grid_ids.size(); ++i) {
      file << to_string(i + 2 + drone_num) + " " + to_string(demands[i]) + "\n";
    }
    file << "DEPOT_SECTION\n";
    file << "1\n";
    file << "EOF";
  }

  file.close();

  // Create par file------------------------------------------
  int min_size = int(grid_ids.size()) / 2;
  int max_size = ceil(int(grid_ids.size()) / 2.0);
  file.open(ep_->mtsp_dir_ + "/amtsp3_" + to_string(ep_->drone_id_) + ".par");
  file << "SPECIAL\n";
  file << "PROBLEM_FILE = " + ep_->mtsp_dir_ + "/amtsp3_" + to_string(ep_->drone_id_) + ".atsp\n";
  if (prob_type == 1) {
    file << "SALESMEN = " << to_string(drone_num) << "\n";
    file << "MTSP_OBJECTIVE = MINSUM\n";
    // file << "MTSP_OBJECTIVE = MINMAX\n";
    file << "MTSP_MIN_SIZE = " << to_string(min_size) << "\n";
    file << "MTSP_MAX_SIZE = " << to_string(max_size) << "\n";
    file << "TRACE_LEVEL = 0\n";
  } else if (prob_type == 2) {
    file << "TRACE_LEVEL = 1\n";  // ACVRP
    file << "SEED = 0\n";         // ACVRP
  }
  file << "RUNS = 1\n";
  file << "TOUR_FILE = " + ep_->mtsp_dir_ + "/amtsp3_" + to_string(ep_->drone_id_) + ".tour\n";

  file.close();

  auto par_dir = ep_->mtsp_dir_ + "/amtsp3_" + to_string(ep_->drone_id_) + ".atsp";
  t1 = ros::Time::now();

  lkh_mtsp_solver::SolveMTSP srv;
  srv.request.prob = 3;
  // if (!tsp_client_.call(srv)) {
  if (!acvrp_client_.call(srv)) {
    ROS_ERROR("Fail to solve ACVRP.");
    return;
  }
  // system("/home/boboyu/software/LKH-3.0.6/LKH
  // /home/boboyu/workspaces/hkust_swarm_ws/src/swarm_exploration/utils/lkh_mtsp_solver/resource/amtsp3_1.par");

  double mtsp_time = (ros::Time::now() - t1).toSec();
  std::cout << "Allocation time: " << mtsp_time << std::endl;

  // Read results
  t1 = ros::Time::now();

  ifstream fin(ep_->mtsp_dir_ + "/amtsp3_" + to_string(ep_->drone_id_) + ".tour");
  string res;
  vector<int> ids;
  while (getline(fin, res)) {
    if (res.compare("TOUR_SECTION") == 0) break;
  }
  while (getline(fin, res)) {
    int id = stoi(res);
    ids.push_back(id - 1);
    if (id == -1) break;
  }
  fin.close();

  // Parse the m-tour of grid
  vector<vector<int>> tours;
  vector<int> tour;
  for (auto id : ids) {
    if (id > 0 && id <= drone_num) {
      tour.clear();
      tour.push_back(id);
    } else if (id >= dimension || id <= 0) {
      tours.push_back(tour);
    } else {
      tour.push_back(id);
    }
  }
  // // Print tour ids
  // for (auto tr : tours) {
  //   std::cout << "tour: ";
  //   for (auto id : tr) std::cout << id << ", ";
  //   std::cout << "" << std::endl;
  // }

  for (int i = 1; i < tours.size(); ++i) {
    if (tours[i][0] == 1) {
      ego_ids.insert(ego_ids.end(), tours[i].begin() + 1, tours[i].end());
    } else {
      other_ids.insert(other_ids.end(), tours[i].begin() + 1, tours[i].end());
    }
  }
  for (auto& id : ego_ids) {
    id = grid_ids[id - 1 - drone_num];
  }
  for (auto& id : other_ids) {
    id = grid_ids[id - 1 - drone_num];
  }
  // // Remove repeated grid
  // unordered_map<int, int> ego_map, other_map;
  // for (auto id : ego_ids) ego_map[id] = 1;
  // for (auto id : other_ids) other_map[id] = 1;

  // ego_ids.clear();
  // other_ids.clear();
  // for (auto p : ego_map) ego_ids.push_back(p.first);
  // for (auto p : other_map) other_ids.push_back(p.first);

  // sort(ego_ids.begin(), ego_ids.end());
  // sort(other_ids.begin(), other_ids.end());
}

double FastExplorationManager::computeGridPathCost(int drone_index, const Eigen::Vector3d& pos,
    const vector<int>& grid_ids, const vector<int>& first, const vector<vector<int>>& firsts,
    const vector<vector<int>>& seconds, bool allow_exact_search) {
  if (grid_ids.empty()) return 0.0;

  double cost = 0.0;
  vector<Eigen::Vector3d> path;
  cost += hgrid_->getCostDroneToGrid(pos, grid_ids[0], first, drone_index);
  for (int i = 0; i < grid_ids.size() - 1; ++i) {
    cost += hgrid_->getCostGridToGrid(
        grid_ids[i], grid_ids[i + 1], firsts, seconds, firsts.size(), allow_exact_search);
  }
  return cost;
}

bool FastExplorationManager::findGlobalTourOfGrid(const vector<Eigen::Vector3d>& positions,
    const vector<Eigen::Vector3d>& velocities, vector<int>& indices, vector<vector<int>>& others,
    bool init) {

  ROS_INFO("Find grid tour---------------");

  auto t1 = ros::Time::now();
  const ros::WallTime wall_total_start = ros::WallTime::now();

  auto& grid_ids = ed_->swarm_state_[ep_->drone_id_ - 1].grid_ids_;
  const vector<int> ownership_before_update = grid_ids;

  // hgrid_->updateBaseCoor();  // Use the latest basecoor transform of swarm

  vector<int> first_ids, second_ids;
  hgrid_->inputFrontiers(ed_->averages_, ed_->viewpoint_unknown_voxels_);

  hgrid_->updateGridData(
      ep_->drone_id_, grid_ids, ed_->reallocated_, ed_->last_grid_ids_, first_ids, second_ids);
  reconcileRemoteTaskLeases(positions[0], ownership_before_update, grid_ids);
  syncOwnGridEpochs(ownership_before_update);

  if (grid_ids.empty()) {
    ROS_WARN(
        "RACER_METRIC grid_tour drone=%d result=0 tasks=0 dimension=0 "
        "prepare_ms=%.3f solver_ms=0.000 parse_ms=0.000 total_ms=%.3f",
        ep_->drone_id_, (ros::WallTime::now() - wall_total_start).toSec() * 1000.0,
        (ros::WallTime::now() - wall_total_start).toSec() * 1000.0);
    ROS_WARN("Empty dominance.");
    ed_->grid_tour_.clear();
    return false;
  }

  ROS_INFO("Allocated HGrid tasks: %zu", grid_ids.size());

  Eigen::MatrixXd mat;
  // uniform_grid_->getCostMatrix(positions, velocities, first_ids, grid_ids, mat);
  if (!init)
    hgrid_->getCostMatrix(positions, velocities, { first_ids }, { second_ids }, grid_ids, mat);
  else
    hgrid_->getCostMatrix(positions, velocities, { {} }, { {} }, grid_ids, mat);

  const double prepare_wall_ms =
      (ros::WallTime::now() - wall_total_start).toSec() * 1000.0;
  double mat_time = (ros::Time::now() - t1).toSec();

  // Find optimal path through ATSP
  t1 = ros::Time::now();
  const int dimension = mat.rows();
  const int drone_num = 1;

  if (ep_->fast_grid_tour_threshold_ > 0 &&
      static_cast<int>(grid_ids.size()) > ep_->fast_grid_tour_threshold_) {
    const ros::WallTime wall_solver_start = ros::WallTime::now();
    vector<int> candidates;
    candidates.reserve(grid_ids.size());
    for (int node = 1 + drone_num; node < dimension; ++node) candidates.push_back(node);
    const auto route = greedyOpenTour(mat, 1, candidates);
    for (const int node : route) indices.push_back(grid_ids[node - 1 - drone_num]);
    others.clear();
    grid_ids = indices;
    hgrid_->getGridTour(grid_ids, positions[0], ed_->grid_tour_, ed_->grid_tour2_);
    ed_->last_grid_ids_ = grid_ids;
    ed_->reallocated_ = false;
    const double solver_wall_ms =
        (ros::WallTime::now() - wall_solver_start).toSec() * 1000.0;
    ROS_WARN(
        "RACER_METRIC grid_tour drone=%d result=1 tasks=%zu dimension=%d "
        "prepare_ms=%.3f solver_ms=%.3f parse_ms=0.000 total_ms=%.3f mode=greedy",
        ep_->drone_id_, indices.size(), dimension, prepare_wall_ms, solver_wall_ms,
        (ros::WallTime::now() - wall_total_start).toSec() * 1000.0);
    ROS_INFO("Fast HGrid tour: %zu tasks", indices.size());
    return true;
  }

  const ros::WallTime wall_solver_start = ros::WallTime::now();
  // Create problem file
  ofstream file(ep_->mtsp_dir_ + "/amtsp2_" + to_string(ep_->drone_id_) + ".atsp");
  file << "NAME : amtsp\n";
  file << "TYPE : ATSP\n";
  file << "DIMENSION : " + to_string(dimension) + "\n";
  file << "EDGE_WEIGHT_TYPE : EXPLICIT\n";
  file << "EDGE_WEIGHT_FORMAT : FULL_MATRIX\n";
  file << "EDGE_WEIGHT_SECTION\n";
  for (int i = 0; i < dimension; ++i) {
    for (int j = 0; j < dimension; ++j) {
      int int_cost = 100 * mat(i, j);
      file << int_cost << " ";
    }
    file << "\n";
  }
  file.close();

  // Create par file
  file.open(ep_->mtsp_dir_ + "/amtsp2_" + to_string(ep_->drone_id_) + ".par");
  file << "SPECIAL\n";
  file << "PROBLEM_FILE = " + ep_->mtsp_dir_ + "/amtsp2_" + to_string(ep_->drone_id_) + ".atsp\n";
  file << "SALESMEN = " << to_string(drone_num) << "\n";
  file << "MTSP_OBJECTIVE = MINSUM\n";
  // file << "MTSP_MIN_SIZE = " << to_string(min(int(ed_->frontiers_.size()) / drone_num, 4)) <<
  // "\n"; file << "MTSP_MAX_SIZE = "
  //      << to_string(max(1, int(ed_->frontiers_.size()) / max(1, drone_num - 1))) << "\n";
  file << "RUNS = 1\n";
  file << "TRACE_LEVEL = 0\n";
  file << "TOUR_FILE = " + ep_->mtsp_dir_ + "/amtsp2_" + to_string(ep_->drone_id_) + ".tour\n";
  file.close();

  auto par_dir = ep_->mtsp_dir_ + "/amtsp2_" + to_string(ep_->drone_id_) + ".atsp";
  t1 = ros::Time::now();

  lkh_mtsp_solver::SolveMTSP srv;
  srv.request.prob = 2;
  if (!tsp_client_.call(srv)) {
    const double solver_wall_ms =
        (ros::WallTime::now() - wall_solver_start).toSec() * 1000.0;
    ROS_WARN(
        "RACER_METRIC grid_tour drone=%d result=0 tasks=%zu dimension=%d "
        "prepare_ms=%.3f solver_ms=%.3f parse_ms=0.000 total_ms=%.3f mode=lkh",
        ep_->drone_id_, grid_ids.size(), dimension, prepare_wall_ms, solver_wall_ms,
        (ros::WallTime::now() - wall_total_start).toSec() * 1000.0);
    ROS_ERROR("Fail to solve ATSP.");
    return false;
  }

  const double solver_wall_ms =
      (ros::WallTime::now() - wall_solver_start).toSec() * 1000.0;
  double mtsp_time = (ros::Time::now() - t1).toSec();
  // std::cout << "AmTSP time: " << mtsp_time << std::endl;

  // Read results
  t1 = ros::Time::now();
  const ros::WallTime wall_parse_start = ros::WallTime::now();

  ifstream fin(ep_->mtsp_dir_ + "/amtsp2_" + to_string(ep_->drone_id_) + ".tour");
  string res;
  vector<int> ids;
  while (getline(fin, res)) {
    if (res.compare("TOUR_SECTION") == 0) break;
  }
  while (getline(fin, res)) {
    int id = stoi(res);
    ids.push_back(id - 1);
    if (id == -1) break;
  }
  fin.close();

  // Parse the m-tour of grid
  vector<vector<int>> tours;
  vector<int> tour;
  for (auto id : ids) {
    if (id > 0 && id <= drone_num) {
      tour.clear();
      tour.push_back(id);
    } else if (id >= dimension || id <= 0) {
      tours.push_back(tour);
    } else {
      tour.push_back(id);
    }
  }

  // for (auto tr : tours) {
  //   std::cout << "tour: ";
  //   for (auto id : tr) std::cout << id << ", ";
  //   std::cout << "" << std::endl;
  // }
  others.resize(drone_num - 1);
  for (int i = 1; i < tours.size(); ++i) {
    if (tours[i][0] == 1) {
      indices.insert(indices.end(), tours[i].begin() + 1, tours[i].end());
    } else {
      others[tours[i][0] - 2].insert(
          others[tours[i][0] - 2].end(), tours[i].begin(), tours[i].end());
    }
  }
  for (auto& id : indices) {
    id -= 1 + drone_num;
  }
  for (auto& other : others) {
    for (auto& id : other) id -= 1 + drone_num;
  }
  for (auto& id : indices) id = grid_ids[id];
  ROS_INFO("Solved HGrid tour with %zu tasks", indices.size());

  // uniform_grid_->getGridTour(indices, ed_->grid_tour_);
  grid_ids = indices;
  hgrid_->getGridTour(grid_ids, positions[0], ed_->grid_tour_, ed_->grid_tour2_);

  ed_->last_grid_ids_ = grid_ids;
  ed_->reallocated_ = false;

  // hgrid_->checkFirstGrid(grid_ids.front());

  const double parse_wall_ms =
      (ros::WallTime::now() - wall_parse_start).toSec() * 1000.0;
  ROS_WARN(
      "RACER_METRIC grid_tour drone=%d result=1 tasks=%zu dimension=%d "
      "prepare_ms=%.3f solver_ms=%.3f parse_ms=%.3f total_ms=%.3f mode=lkh",
      ep_->drone_id_, indices.size(), dimension, prepare_wall_ms, solver_wall_ms,
      parse_wall_ms, (ros::WallTime::now() - wall_total_start).toSec() * 1000.0);

  return true;
}

void FastExplorationManager::findTourOfFrontier(const Vector3d& cur_pos, const Vector3d& cur_vel,
    const Vector3d& cur_yaw, const vector<int>& ftr_ids, const vector<Eigen::Vector3d>& grid_pos,
    vector<int>& indices) {

  auto t1 = ros::Time::now();
  const ros::WallTime wall_total_start = ros::WallTime::now();

  vector<Eigen::Vector3d> positions = { cur_pos };
  vector<Eigen::Vector3d> velocities = { cur_vel };
  vector<double> yaws = { cur_yaw[0] };

  // frontier_finder_->getSwarmCostMatrix(positions, velocities, yaws, mat);
  Eigen::MatrixXd mat;
  frontier_finder_->getSwarmCostMatrix(positions, velocities, yaws, ftr_ids, grid_pos, mat);
  const int dimension = mat.rows();
  // std::cout << "dim of frontier TSP mat: " << dimension << std::endl;

  const double matrix_wall_ms =
      (ros::WallTime::now() - wall_total_start).toSec() * 1000.0;
  double mat_time = (ros::Time::now() - t1).toSec();
  // ROS_INFO("mat time: %lf", mat_time);

  // Find optimal allocation through AmTSP
  t1 = ros::Time::now();

  if (ep_->fast_frontier_tour_threshold_ > 0 &&
      static_cast<int>(ftr_ids.size()) > ep_->fast_frontier_tour_threshold_) {
    const ros::WallTime wall_solver_start = ros::WallTime::now();
    const int drone_num = 1;
    vector<int> candidates;
    candidates.reserve(ftr_ids.size());
    for (int node = 1 + drone_num; node < 1 + drone_num + ftr_ids.size(); ++node)
      candidates.push_back(node);
    const auto route = greedyOpenTour(mat, 1, candidates);
    for (const int node : route) indices.push_back(ftr_ids[node - 1 - drone_num]);
    frontier_finder_->getPathForTour(cur_pos, indices, ed_->frontier_tour_);
    if (!grid_pos.empty()) ed_->frontier_tour_.push_back(grid_pos[0]);
    ROS_WARN(
        "RACER_METRIC frontier_tour drone=%d result=1 tasks=%zu dimension=%d "
        "matrix_ms=%.3f solver_ms=%.3f parse_ms=0.000 total_ms=%.3f mode=greedy",
        ep_->drone_id_, indices.size(), dimension, matrix_wall_ms,
        (ros::WallTime::now() - wall_solver_start).toSec() * 1000.0,
        (ros::WallTime::now() - wall_total_start).toSec() * 1000.0);
    ROS_INFO("Fast frontier tour: %zu tasks", indices.size());
    return;
  }

  const ros::WallTime wall_solver_start = ros::WallTime::now();
  // Create problem file
  ofstream file(ep_->mtsp_dir_ + "/amtsp_" + to_string(ep_->drone_id_) + ".atsp");
  file << "NAME : amtsp\n";
  file << "TYPE : ATSP\n";
  file << "DIMENSION : " + to_string(dimension) + "\n";
  file << "EDGE_WEIGHT_TYPE : EXPLICIT\n";
  file << "EDGE_WEIGHT_FORMAT : FULL_MATRIX\n";
  file << "EDGE_WEIGHT_SECTION\n";
  for (int i = 0; i < dimension; ++i) {
    for (int j = 0; j < dimension; ++j) {
      int int_cost = 100 * mat(i, j);
      file << int_cost << " ";
    }
    file << "\n";
  }
  file.close();

  // Create par file
  const int drone_num = 1;

  file.open(ep_->mtsp_dir_ + "/amtsp_" + to_string(ep_->drone_id_) + ".par");
  file << "SPECIAL\n";
  file << "PROBLEM_FILE = " + ep_->mtsp_dir_ + "/amtsp_" + to_string(ep_->drone_id_) + ".atsp\n";
  file << "SALESMEN = " << to_string(drone_num) << "\n";
  file << "MTSP_OBJECTIVE = MINSUM\n";
  file << "MTSP_MIN_SIZE = " << to_string(min(int(ed_->frontiers_.size()) / drone_num, 4)) << "\n";
  file << "MTSP_MAX_SIZE = "
       << to_string(max(1, int(ed_->frontiers_.size()) / max(1, drone_num - 1))) << "\n";
  file << "RUNS = 1\n";
  file << "TRACE_LEVEL = 0\n";
  file << "TOUR_FILE = " + ep_->mtsp_dir_ + "/amtsp_" + to_string(ep_->drone_id_) + ".tour\n";
  file.close();

  auto par_dir = ep_->mtsp_dir_ + "/amtsp_" + to_string(ep_->drone_id_) + ".atsp";
  t1 = ros::Time::now();

  lkh_mtsp_solver::SolveMTSP srv;
  srv.request.prob = 1;
  if (!tsp_client_.call(srv)) {
    ROS_WARN(
        "RACER_METRIC frontier_tour drone=%d result=0 tasks=%zu dimension=%d "
        "matrix_ms=%.3f solver_ms=%.3f parse_ms=0.000 total_ms=%.3f mode=lkh",
        ep_->drone_id_, ftr_ids.size(), dimension, matrix_wall_ms,
        (ros::WallTime::now() - wall_solver_start).toSec() * 1000.0,
        (ros::WallTime::now() - wall_total_start).toSec() * 1000.0);
    ROS_ERROR("Fail to solve ATSP.");
    return;
  }

  const double solver_wall_ms =
      (ros::WallTime::now() - wall_solver_start).toSec() * 1000.0;
  double mtsp_time = (ros::Time::now() - t1).toSec();
  // ROS_INFO("AmTSP time: %lf", mtsp_time);

  // Read results
  t1 = ros::Time::now();
  const ros::WallTime wall_parse_start = ros::WallTime::now();

  ifstream fin(ep_->mtsp_dir_ + "/amtsp_" + to_string(ep_->drone_id_) + ".tour");
  string res;
  vector<int> ids;
  while (getline(fin, res)) {
    if (res.compare("TOUR_SECTION") == 0) break;
  }
  while (getline(fin, res)) {
    int id = stoi(res);
    ids.push_back(id - 1);
    if (id == -1) break;
  }
  fin.close();

  // Parse the m-tour
  vector<vector<int>> tours;
  vector<int> tour;
  for (auto id : ids) {
    if (id > 0 && id <= drone_num) {
      tour.clear();
      tour.push_back(id);
    } else if (id >= dimension || id <= 0) {
      tours.push_back(tour);
    } else {
      tour.push_back(id);
    }
  }

  vector<vector<int>> others(drone_num - 1);
  for (int i = 1; i < tours.size(); ++i) {
    if (tours[i][0] == 1) {
      indices.insert(indices.end(), tours[i].begin() + 1, tours[i].end());
    }
    // else {
    //   others[tours[i][0] - 2].insert(
    //       others[tours[i][0] - 2].end(), tours[i].begin() + 1, tours[i].end());
    // }
  }
  for (auto& id : indices) {
    id -= 1 + drone_num;
  }
  // for (auto& other : others) {
  //   for (auto& id : other)
  //     id -= 1 + drone_num;
  // }

  if (ed_->grid_tour_.size() > 2) {  // Remove id for next grid, since it is considered in the TSP
    indices.pop_back();
  }
  // Subset of frontier inside first grid
  for (int i = 0; i < indices.size(); ++i) {
    indices[i] = ftr_ids[indices[i]];
  }

  // Get the path of optimal tour from path matrix
  frontier_finder_->getPathForTour(cur_pos, indices, ed_->frontier_tour_);
  if (!grid_pos.empty()) {
    ed_->frontier_tour_.push_back(grid_pos[0]);
  }

  // ed_->other_tours_.clear();
  // for (int i = 1; i < positions.size(); ++i) {
  //   ed_->other_tours_.push_back({});
  //   frontier_finder_->getPathForTour(positions[i], others[i - 1], ed_->other_tours_[i - 1]);
  // }

  double parse_time = (ros::Time::now() - t1).toSec();
  const double parse_wall_ms =
      (ros::WallTime::now() - wall_parse_start).toSec() * 1000.0;
  ROS_WARN(
      "RACER_METRIC frontier_tour drone=%d result=1 tasks=%zu dimension=%d "
      "matrix_ms=%.3f solver_ms=%.3f parse_ms=%.3f total_ms=%.3f mode=lkh",
      ep_->drone_id_, indices.size(), dimension, matrix_wall_ms, solver_wall_ms,
      parse_wall_ms, (ros::WallTime::now() - wall_total_start).toSec() * 1000.0);
  // ROS_INFO("Cost mat: %lf, TSP: %lf, parse: %f, %d frontiers assigned.", mat_time, mtsp_time,
  //     parse_time, indices.size());
}

}  // namespace fast_planner
