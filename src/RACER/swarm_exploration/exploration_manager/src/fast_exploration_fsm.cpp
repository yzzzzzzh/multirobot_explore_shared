
#include <plan_manage/planner_manager.h>
#include <exploration_manager/fast_exploration_manager.h>
#include <traj_utils/planning_visualization.h>

#include <exploration_manager/fast_exploration_fsm.h>
#include <exploration_manager/expl_data.h>
#include <exploration_manager/HGrid.h>
#include <exploration_manager/GridTour.h>

#include <plan_env/edt_environment.h>
#include <plan_env/sdf_map.h>
#include <plan_env/multi_map_manager.h>
#include <active_perception/perception_utils.h>
#include <active_perception/hgrid.h>
// #include <active_perception/uniform_grid.h>
// #include <lkh_tsp_solver/lkh_interface.h>
// #include <lkh_mtsp_solver/lkh3_interface.h>

// 状态机，决定该干什么，包括七个状态：INIT、WAIT_TRIGGER、PLAN_TRAJ、PUB_TRAJ、EXEC_TRAJ、FINISH、IDLE
// WAIT_TRIGGER 收到 /move_base_simple/goal 后进 PLAN_TRAJ；
// PLAN_TRAJ 调 callExplorationPlanner() 规划，成功进 PUB_TRAJ 发布 B 样条，
// 再进 EXEC_TRAJ；EXEC_TRAJ 里检查到达、目标保持时间、阻塞回报、边界变化，决定回 PLAN_TRAJ 重规划还是 FINISH。
// 探索开始后，状态机只在 PLAN_TRAJ、PUB_TRAJ、EXEC_TRAJ、IDLE 四个状态之间循环

#include <cmath>
#include <fstream>
#include <limits>
#include <unordered_map>
#include <unordered_set>

using Eigen::Vector4d;

namespace fast_planner {
void FastExplorationFSM::init(ros::NodeHandle& nh) {
  fp_.reset(new FSMParam);
  fd_.reset(new FSMData);

  /*  Fsm param  */
  nh.param("fsm/thresh_replan1", fp_->replan_thresh1_, -1.0);
  nh.param("fsm/thresh_replan2", fp_->replan_thresh2_, -1.0);
  nh.param("fsm/thresh_replan3", fp_->replan_thresh3_, -1.0);
  nh.param("fsm/replan_time", fp_->replan_time_, -1.0);
  nh.param("fsm/plan_retry_interval", fp_->plan_retry_interval_, 0.2);
  nh.param("fsm/idle_check_interval", fp_->idle_check_interval_, 2.0);
  nh.param("fsm/frontier_update_interval", fp_->frontier_update_interval_, 0.5);
  nh.param("fsm/trigger_stagger", fp_->trigger_stagger_, 0.0);
  nh.param("fsm/pair_opt_timer_stagger", fp_->pair_opt_timer_stagger_, 0.0);
  nh.param("fsm/reachability_filter_max_tasks", fp_->reachability_filter_max_tasks_, 8);
  // Allocation hysteresis: a candidate that keeps less than this fraction of
  // each vehicle's current route is a reshuffle; accept it only when the pair
  // sum improves by at least pair_opt_reshuffle_min_gain (relative).  When
  // rejected but unallocated tasks exist, publish the greedy baseline
  // insertion instead so those tasks still get an owner without churn.
  nh.param("fsm/pair_opt_min_keep_fraction", fp_->pair_opt_min_keep_fraction_, 0.0);
  // Viewpoint-level exit: after reaching the target and waiting this long
  // without the target frontier changing, or after a second blocked
  // execution, the viewpoint itself is suppressed for viewpoint_block_ttl.
  nh.param("fsm/arrived_no_gain_timeout", fp_->arrived_no_gain_timeout_, 0.0);
  nh.param("fsm/viewpoint_block_ttl", fp_->viewpoint_block_ttl_, 60.0);
  // Wider variant of the arrival rule: the robot hovers within
  // viewpoint_stagnation_radius of an unchanged target (e.g. the controller
  // throttles it near a wall so it never "arrives") for this long.
  nh.param("fsm/viewpoint_stagnation_timeout", fp_->viewpoint_stagnation_timeout_, 0.0);
  nh.param("fsm/viewpoint_stagnation_radius", fp_->viewpoint_stagnation_radius_, 2.0);
  nh.param("fsm/plan_fail_block_after", fp_->plan_fail_block_after_, 0);
  nh.param("fsm/fallback_grid_release_after", fp_->fallback_grid_release_after_, 0);
  // Generic safety net: while a target is active, moving less than
  // no_progress_block_distance within no_progress_block_timeout suppresses
  // that viewpoint (covers wall-crawling and controller throttling cases).
  nh.param("fsm/no_progress_block_timeout", fp_->no_progress_block_timeout_, 0.0);
  nh.param("fsm/no_progress_block_distance", fp_->no_progress_block_distance_, 1.0);
  nh.param("fsm/approach_stall_timeout", fp_->approach_stall_timeout_, 0.0);
  nh.param("fsm/approach_progress_distance", fp_->approach_progress_distance_, 1.5);
  nh.param("fsm/pair_opt_reshuffle_min_gain", fp_->pair_opt_reshuffle_min_gain_, 0.0);
  nh.param("fsm/recovery_batch_size", fp_->recovery_batch_size_, 6);
  nh.param("fsm/min_target_execution_time", fp_->min_target_execution_time_, 1.0);
  nh.param(
      "fsm/min_target_execution_distance", fp_->min_target_execution_distance_, 0.75);
  nh.param(
      "fsm/execution_blocked_replan_delay", fp_->execution_blocked_replan_delay_, 0.5);
  nh.param(
      "fsm/use_measured_replan_start", fp_->use_measured_replan_start_, false);
  nh.param("fsm/enable_planning_visualization", fp_->enable_planning_visualization_, true);
  nh.param("fsm/attempt_interval", fp_->attempt_interval_, 0.2);
  nh.param("fsm/pair_opt_interval", fp_->pair_opt_interval_, 1.0);
  nh.param("fsm/state_freshness", fp_->state_freshness_, 0.2);
  nh.param("fsm/repeat_send_num", fp_->repeat_send_num_, 10);
  nh.param("fsm/enable_idle_rebalance", fp_->enable_idle_rebalance_, false);
  nh.param("fsm/idle_rebalance_min_makespan_gain",
      fp_->idle_rebalance_min_makespan_gain_, 0.05);
  nh.param("fsm/idle_rebalance_max_sum_ratio",
      fp_->idle_rebalance_max_sum_ratio_, 1.75);

  /* Initialize main modules */
  expl_manager_.reset(new FastExplorationManager);
  expl_manager_->initialize(nh);
  visualization_.reset(new PlanningVisualization(nh));

  planner_manager_ = expl_manager_->planner_manager_;
  state_ = EXPL_STATE::INIT;
  fd_->have_odom_ = false;
  fd_->state_str_ = { "INIT", "WAIT_TRIGGER", "PLAN_TRAJ", "PUB_TRAJ", "EXEC_TRAJ", "FINISH", "IDL"
                                                                                              "E" };
  fd_->static_state_ = true;
  fd_->trigger_ = false;
  fd_->avoid_collision_ = false;
  fd_->go_back_ = false;
  fd_->emergency_replan_ = false;
  fd_->last_plan_attempt_time_ = ros::Time(0);
  fd_->pair_opt_phase_started_ = false;
  fd_->consecutive_plan_failures_ = 0;
  fd_->replan_same_target_ = false;
  fd_->target_initialized_ = false;
  fd_->target_arrived_time_ = ros::Time(0);
  fd_->target_near_time_ = ros::Time(0);
  fd_->progress_anchor_time_ = ros::Time(0);
  fd_->progress_anchor_pos_ = Eigen::Vector3d::Zero();
  fd_->target_best_dist_ = -1.0;
  fd_->target_best_time_ = ros::Time(0);
  fd_->active_target_grid_ = -1;
  fd_->execution_blocked_ = false;
  fd_->execution_blocked_handled_ = false;
  fd_->target_block_replans_ = 0;
  fd_->execution_blocked_since_ = ros::Time(0);

  /* Ros sub, pub and timer */
  exec_timer_ = nh.createTimer(ros::Duration(0.01), &FastExplorationFSM::FSMCallback, this);
  safety_timer_ = nh.createTimer(ros::Duration(0.05), &FastExplorationFSM::safetyCallback, this);
  frontier_timer_ = nh.createTimer(
      ros::Duration(fp_->frontier_update_interval_), &FastExplorationFSM::frontierCallback, this);

  trigger_sub_ =
      nh.subscribe("/move_base_simple/goal", 1, &FastExplorationFSM::triggerCallback, this);
  odom_sub_ = nh.subscribe("/odom_world", 1, &FastExplorationFSM::odometryCallback, this);
  execution_blocked_sub_ = nh.subscribe(
      "/planning/execution_blocked", 10,
      &FastExplorationFSM::executionBlockedCallback, this);

  replan_pub_ = nh.advertise<std_msgs::Empty>("/planning/replan", 10);
  new_pub_ = nh.advertise<std_msgs::Empty>("/planning/new", 10);
  bspline_pub_ = nh.advertise<bspline::Bspline>("/planning/bspline", 10);

  // Swarm, timer, pub and sub
  drone_state_timer_ =
      nh.createTimer(ros::Duration(0.04), &FastExplorationFSM::droneStateTimerCallback, this);
  drone_state_pub_ =
      nh.advertise<exploration_manager::DroneState>("/swarm_expl/drone_state_send", 10);
  drone_state_sub_ = nh.subscribe(
      "/swarm_expl/drone_state_recv", 10, &FastExplorationFSM::droneStateMsgCallback, this);

  opt_timer_ = nh.createTimer(ros::Duration(0.05), &FastExplorationFSM::optTimerCallback, this);
  opt_pub_ = nh.advertise<exploration_manager::PairOpt>("/swarm_expl/pair_opt_send", 10);
  opt_sub_ = nh.subscribe("/swarm_expl/pair_opt_recv", 100, &FastExplorationFSM::optMsgCallback,
      this, ros::TransportHints().tcpNoDelay());

  opt_res_pub_ =
      nh.advertise<exploration_manager::PairOptResponse>("/swarm_expl/pair_opt_res_send", 10);
  opt_res_sub_ = nh.subscribe("/swarm_expl/pair_opt_res_recv", 100,
      &FastExplorationFSM::optResMsgCallback, this, ros::TransportHints().tcpNoDelay());

  frontier_share_timer_ = nh.createTimer(
      ros::Duration(std::max(0.2, expl_manager_->ep_->peer_frontier_share_interval_)),
      &FastExplorationFSM::frontierShareTimerCallback, this);
  frontier_share_pub_ = nh.advertise<exploration_manager::FrontierShare>(
      "/swarm_expl/frontier_share_send", 10);
  frontier_share_sub_ = nh.subscribe("/swarm_expl/frontier_share_recv", 20,
      &FastExplorationFSM::frontierShareMsgCallback, this,
      ros::TransportHints().tcpNoDelay());

  swarm_traj_pub_ = nh.advertise<bspline::Bspline>("/planning/swarm_traj_send", 100);
  swarm_traj_sub_ =
      nh.subscribe("/planning/swarm_traj_recv", 100, &FastExplorationFSM::swarmTrajCallback, this);
  swarm_traj_timer_ =
      nh.createTimer(ros::Duration(0.1), &FastExplorationFSM::swarmTrajTimerCallback, this);

  hgrid_pub_ = nh.advertise<exploration_manager::HGrid>("/swarm_expl/hgrid_send", 10);
  grid_tour_pub_ = nh.advertise<exploration_manager::GridTour>("/swarm_expl/grid_tour_send", 10);

  ROS_WARN(
      "RACER_CONFIG drone=%d replan_time=%.3f thresholds=[%.3f,%.3f,%.3f] "
      "retry=%.3f frontier_update=%.3f pair_opt=%.3f pair_stagger=%.3f "
      "state_freshness=%.3f idle_rebalance=%d "
      "idle_gain=%.3f idle_sum_ratio=%.3f "
      "reachability_cap=%d recovery_batch=%d "
      "target_hold=[%.2fs,%.2fm] grid_blacklist=disabled "
      "blocked_replan=%.2fs "
      "measured_replan_start=%d planning_visualization=%d",
      getId(), fp_->replan_time_, fp_->replan_thresh1_, fp_->replan_thresh2_,
      fp_->replan_thresh3_, fp_->plan_retry_interval_, fp_->frontier_update_interval_,
      fp_->pair_opt_interval_, fp_->pair_opt_timer_stagger_,
      fp_->state_freshness_, static_cast<int>(fp_->enable_idle_rebalance_),
      fp_->idle_rebalance_min_makespan_gain_,
      fp_->idle_rebalance_max_sum_ratio_,
      fp_->reachability_filter_max_tasks_,
      fp_->recovery_batch_size_,
      fp_->min_target_execution_time_, fp_->min_target_execution_distance_,
      fp_->execution_blocked_replan_delay_,
      static_cast<int>(fp_->use_measured_replan_start_),
      static_cast<int>(fp_->enable_planning_visualization_));
}

int FastExplorationFSM::getId() {
  return expl_manager_->ep_->drone_id_;
}

uint64_t FastExplorationFSM::nextAssignmentEpoch(
    const DroneState& first, const DroneState& second) const {
  const uint64_t clock_epoch =
      (static_cast<uint64_t>(ros::Time::now().toNSec()) << 4) |
      static_cast<uint64_t>(expl_manager_->ep_->drone_id_ & 0x0f);
  return std::max(
      std::max(first.assignment_epoch_, second.assignment_epoch_) + 1,
      clock_epoch);
}

void FastExplorationFSM::setRouteOwnership(
    DroneState& state, const vector<int>& ids, uint64_t epoch) {
  state.grid_ids_ = ids;
  state.assignment_epoch_ = epoch;
  state.grid_epochs_.clear();
  for (const int grid_id : ids) state.grid_epochs_[grid_id] = epoch;
  const std::unordered_set<int> retained(ids.begin(), ids.end());
  for (auto it = state.remote_tasks_.begin();
       it != state.remote_tasks_.end();) {
    if (retained.count(it->first) == 0)
      it = state.remote_tasks_.erase(it);
    else
      ++it;
  }
}

void FastExplorationFSM::purgeExpiredFailures(DroneState& state, double now) {
  for (auto it = state.failed_grid_until_.begin();
       it != state.failed_grid_until_.end();) {
    if (it->second <= now)
      it = state.failed_grid_until_.erase(it);
    else
      ++it;
  }
}

bool FastExplorationFSM::isGridBlacklisted(
    const DroneState& state, int grid_id, double now) const {
  const auto it = state.failed_grid_until_.find(grid_id);
  return it != state.failed_grid_until_.end() && it->second > now;
}

void FastExplorationFSM::reconcileDuplicateOwnership() {
  struct Winner {
    uint64_t epoch;
    int owner;
  };
  auto& states = expl_manager_->ed_->swarm_state_;
  const double now = ros::Time::now().toSec();
  std::unordered_map<int, Winner> winners;
  for (int owner = 0; owner < static_cast<int>(states.size()); ++owner) {
    purgeExpiredFailures(states[owner], now);
    for (const int grid_id : states[owner].grid_ids_) {
      const auto epoch_it = states[owner].grid_epochs_.find(grid_id);
      const uint64_t epoch =
          epoch_it == states[owner].grid_epochs_.end()
              ? states[owner].assignment_epoch_
              : epoch_it->second;
      const auto winner = winners.find(grid_id);
      if (winner == winners.end() || epoch > winner->second.epoch ||
          (epoch == winner->second.epoch && owner < winner->second.owner)) {
        winners[grid_id] = { epoch, owner };
      }
    }
  }

  // A DroneState received from a peer is only a cache of that peer's
  // authoritative route. Mutating all cached routes here made an old packet
  // repeatedly "remove" tasks, after which findUnallocated() treated the
  // transient cache state as real unassigned work. Each vehicle now corrects
  // only its own authoritative route; the losing peer performs the symmetric
  // correction when it receives the same winning state.
  auto& own = states[getId() - 1];
  vector<int> unique_route;
  unique_route.reserve(own.grid_ids_.size());
  std::unordered_set<int> seen;
  int removed = 0;
  for (const int grid_id : own.grid_ids_) {
    const auto winner = winners.find(grid_id);
    const bool keep =
        seen.insert(grid_id).second && winner != winners.end() &&
        winner->second.owner == getId() - 1;
    if (keep) {
      unique_route.push_back(grid_id);
      continue;
    }
    own.grid_epochs_.erase(grid_id);
    ++removed;
    ROS_WARN(
        "RACER_METRIC ownership_duplicate_resolved observer=%d "
        "grid=%d loser=%d winner=%d epoch=%llu",
        getId(), grid_id, getId(),
        winner == winners.end() ? -1 : winner->second.owner + 1,
        static_cast<unsigned long long>(
            winner == winners.end() ? 0 : winner->second.epoch));
  }
  if (removed > 0) {
    own.grid_ids_.swap(unique_route);
    const std::unordered_set<int> retained(
        own.grid_ids_.begin(), own.grid_ids_.end());
    for (auto it = own.remote_tasks_.begin();
         it != own.remote_tasks_.end();) {
      if (retained.count(it->first) == 0)
        it = own.remote_tasks_.erase(it);
      else
        ++it;
    }
    own.assignment_epoch_ = nextAssignmentEpoch(own, own);
    expl_manager_->ed_->last_grid_ids_.clear();
    expl_manager_->ed_->reallocated_ = true;
    ROS_WARN(
        "RACER_METRIC ownership_reconcile observer=%d removed=%d own_changed=1",
        getId(), removed);
  }
}

bool FastExplorationFSM::targetHoldComplete(double now) const {
  if (!fd_->target_initialized_) return true;
  const double elapsed = now - fd_->target_started_time_.toSec();
  const bool reached =
      (fd_->odom_pos_ - fd_->active_target_pos_).norm() <=
      std::max(0.35, 0.5 * fp_->min_target_execution_distance_);
  // Do not require a minimum travelled distance before accepting a target
  // switch. A blocked or short trajectory may legitimately make less progress
  // than that floor; keeping the old conjunction could then suppress every
  // frontier-covered replan. One second of execution is sufficient evidence
  // that the command was attempted, while reaching the target permits an
  // immediate switch.
  return elapsed >= fp_->min_target_execution_time_ || reached;
}

void FastExplorationFSM::executionBlockedCallback(
    const std_msgs::BoolConstPtr& msg) {
  if (msg->data) {
    // The controller republishes a latched blocked state every 0.5 s.  Treat
    // it as one obstacle episode until either the controller reports clear or
    // a replacement trajectory is published.  Otherwise one physical
    // blockage drains a different HGrid on every repeated True message.
    if (fd_->execution_blocked_handled_) return;
    if (!fd_->execution_blocked_) {
      fd_->execution_blocked_since_ = ros::Time::now();
    }
    fd_->execution_blocked_ = true;
  } else {
    fd_->execution_blocked_ = false;
    fd_->execution_blocked_handled_ = false;
    fd_->execution_blocked_since_ = ros::Time(0);
  }
}

void FastExplorationFSM::FSMCallback(const ros::TimerEvent& e) {
  ROS_INFO_STREAM_THROTTLE(
      1.0, "[FSM]: Drone " << getId() << " state: " << fd_->state_str_[int(state_)]);

  switch (state_) {
    case INIT: {
      // Wait for odometry ready
      if (!fd_->have_odom_) {
        ROS_WARN_THROTTLE(1.0, "no odom");
        return;
      }
      if ((ros::Time::now() - fd_->fsm_init_time_).toSec() < 2.0) {
        ROS_WARN_THROTTLE(1.0, "wait for init");
        return;
      }
      // Go to wait trigger when odom is ok
      transitState(WAIT_TRIGGER, "FSM");
      break;
    }

    // 一旦进入了WAIT_TRIGGER状态，就不会再回来了。
    case WAIT_TRIGGER: {
      // fd_->trigger_：有没有收到过开始探索的指令。一次触发之后，后面再也不会回去
      // fd_->trigger_ready_time_ = 现在 + trigger_stagger × (机器人编号 − 1)最早允许离开 WAIT_TRIGGER 的时刻，目标是多机错峰
      if (fd_->trigger_ && ros::Time::now() >= fd_->trigger_ready_time_) {
        const int frontier_count =
            expl_manager_->updateFrontierStruct(fd_->odom_pos_);
        const auto& own_grids =
            expl_manager_->ed_->swarm_state_[getId() - 1].grid_ids_;
        // A 360-degree planar LiDAR can temporarily have no coverable local
        // frontier while the HGrid layer already owns valid unknown-space
        // tasks.  PLAN_TRAJ has an explicit empty-frontier path that advances
        // toward the HGrid center/known boundary.  Skipping it here marked a
        // largely unknown building as FINISH before the first command.
        if (frontier_count != 0 || !own_grids.empty())
          transitState(PLAN_TRAJ, "FSM-trigger-stagger");
        else {
          ROS_WARN(
              "RACER_METRIC trigger_no_work drone=%d frontiers=0 grids=0",
              getId());
          transitState(FINISH, "FSM-trigger-stagger");
        }
      } else {
        ROS_WARN_THROTTLE(1.0, "wait for trigger.");
      }
      break;
    }

    case FINISH: {
      ROS_INFO_THROTTLE(1.0, "finish exploration.");
      break;
    }

    case IDLE: {
      double check_interval = (ros::Time::now() - fd_->last_check_frontier_time_).toSec();
      if (check_interval > fp_->idle_check_interval_) {
        fd_->last_check_frontier_time_ = ros::Time::now();
        const int frontier_count =
            expl_manager_->updateFrontierStruct(fd_->odom_pos_);
        const auto& refreshed_grids =
            expl_manager_->ed_->swarm_state_[getId() - 1].grid_ids_;
        if (frontier_count != 0 || !refreshed_grids.empty()) {
          ROS_WARN(
              "Restart exploration: frontiers=%d active_hgrids=%zu",
              frontier_count, refreshed_grids.size());
          transitState(PLAN_TRAJ, "FSM");
        }
      }
      break;
    }

    case PLAN_TRAJ: {
      const ros::Time now = ros::Time::now();
      if (!fd_->last_plan_attempt_time_.isZero() &&
          (now - fd_->last_plan_attempt_time_).toSec() <
              fp_->plan_retry_interval_) {
        break;
      }
      const bool had_safe_active_trajectory =
          !fd_->static_state_ && !fd_->emergency_replan_;
      // A collision/emergency callback truncates the old trajectory before
      // entering PLAN_TRAJ.  Its future B-spline state is therefore neither
      // executable nor necessarily collision-free.  Replanning from that
      // stale point made the same unsafe start fail against every HGrid and
      // drained the whole distributed allocation.  Emergency plans must
      // start from the latest measured state.
      if (fd_->static_state_ || fd_->emergency_replan_ ||
          fp_->use_measured_replan_start_) {
        // Plan from static state (hover)
        fd_->start_pt_ = fd_->odom_pos_;
        fd_->start_vel_ = fd_->odom_vel_;
        fd_->start_acc_.setZero();
        fd_->start_yaw_ << fd_->odom_yaw_, 0, 0;
        if (fp_->use_measured_replan_start_ && !fd_->static_state_ &&
            !fd_->emergency_replan_) {
          ROS_WARN_THROTTLE(
              1.0,
              "RACER_METRIC measured_replan_start drone=%d speed_mps=%.3f",
              getId(), fd_->odom_vel_.norm());
        }
      } else {
        // Replan from non-static state, starting from 'replan_time' seconds later
        LocalTrajData* info = &planner_manager_->local_data_;
        double t_r = (ros::Time::now() - info->start_time_).toSec() + fp_->replan_time_;
        fd_->start_pt_ = info->position_traj_.evaluateDeBoorT(t_r);
        fd_->start_vel_ = info->velocity_traj_.evaluateDeBoorT(t_r);
        fd_->start_acc_ = info->acceleration_traj_.evaluateDeBoorT(t_r);
        fd_->start_yaw_(0) = info->yaw_traj_.evaluateDeBoorT(t_r)[0];
        fd_->start_yaw_(1) = info->yawdot_traj_.evaluateDeBoorT(t_r)[0];
        fd_->start_yaw_(2) = info->yawdotdot_traj_.evaluateDeBoorT(t_r)[0];
      }
      fd_->last_plan_attempt_time_ = now;
      const ros::WallTime plan_wall_start = ros::WallTime::now();
      int res = callExplorationPlanner();
      const double plan_wall_ms =
          (ros::WallTime::now() - plan_wall_start).toSec() * 1000.0;
      ROS_WARN(
          "RACER_METRIC plan drone=%d result=%d wall_ms=%.3f "
          "consecutive_failures=%d",
          getId(), res, plan_wall_ms, fd_->consecutive_plan_failures_);
      if (res == SUCCEED) {
        // Only truncate the old trajectory after a complete replacement has
        // been generated.  A normal failed replan therefore cannot turn a
        // moving vehicle into a permanent hover.
        replan_pub_.publish(std_msgs::Empty());
        fd_->consecutive_plan_failures_ = 0;
        fd_->emergency_replan_ = false;
        transitState(PUB_TRAJ, "FSM");
      } else if (res == FAIL) {  // Keep trying to replan
        ++fd_->consecutive_plan_failures_;
        ROS_WARN("Plan fail (%d consecutive)", fd_->consecutive_plan_failures_);
        if (fp_->fallback_grid_release_after_ > 0 &&
            expl_manager_->ed_->last_fallback_grid_ >= 0 &&
            fd_->consecutive_plan_failures_ >= fp_->fallback_grid_release_after_) {
          const int released = expl_manager_->ed_->last_fallback_grid_;
          expl_manager_->hgrid_->markGridVisited(released);
          ROS_ERROR(
              "RACER_RECOVERY drone=%d fallback_grid_released=1 grid=%d reason=plan_fail "
              "failures=%d",
              getId(), released, fd_->consecutive_plan_failures_);
          expl_manager_->ed_->last_fallback_grid_ = -1;
          fd_->consecutive_plan_failures_ = 0;
        }
        // No grid quarantine/blacklist: a failed plan keeps the task in the
        // route and the next cycle re-solves the tour with fresh costs.
        // Viewpoint-level exit: after repeated failures towards the same goal
        // suppress that goal so the next selection picks another viewpoint.
        if (fp_->plan_fail_block_after_ > 0 &&
            fd_->consecutive_plan_failures_ >= fp_->plan_fail_block_after_) {
          const Eigen::Vector3d goal = expl_manager_->ed_->next_pos_;
          expl_manager_->frontier_finder_->blockViewpoint(
              goal, ros::Time::now().toSec() + fp_->viewpoint_block_ttl_);
          ROS_WARN(
              "RACER_RECOVERY drone=%d viewpoint_blocked=plan_fail "
              "target=[%.2f,%.2f] failures=%d",
              getId(), goal[0], goal[1], fd_->consecutive_plan_failures_);
          fd_->consecutive_plan_failures_ = 0;
          fd_->replan_same_target_ = false;
          fd_->target_initialized_ = false;
          fd_->active_target_grid_ = -1;
        }
        if (had_safe_active_trajectory) {
          // The trajectory server still owns the unmodified safe trajectory.
          transitState(EXEC_TRAJ, "FSM-plan-fallback");
        } else {
          fd_->static_state_ = true;
        }
      } else if (res == NO_GRID) {
        fd_->static_state_ = true;
        fd_->emergency_replan_ = false;
        fd_->consecutive_plan_failures_ = 0;
        replan_pub_.publish(std_msgs::Empty());
        fd_->last_check_frontier_time_ = ros::Time::now();
        ROS_WARN("No grid");
        transitState(IDLE, "FSM");
        visualize(1);
        // clearVisMarker();
      }
      break;
    }

    case PUB_TRAJ: {
      double dt = (ros::Time::now() - fd_->newest_traj_.start_time).toSec();
      if (dt > 0) {
        bspline_pub_.publish(fd_->newest_traj_);
        fd_->static_state_ = false;

        // fd_->newest_traj_.drone_id = planner_manager_->swarm_traj_data_.drone_id_;
        fd_->newest_traj_.drone_id = expl_manager_->ep_->drone_id_;
        swarm_traj_pub_.publish(fd_->newest_traj_);

        const Eigen::Vector3d target = expl_manager_->ed_->next_pos_;
        const auto& own_grids =
            expl_manager_->ed_->swarm_state_[getId() - 1].grid_ids_;
        const int committed_grid =
            own_grids.empty() ? -1 : own_grids.front();
        if (committed_grid != fd_->active_target_grid_) {
          fd_->active_target_grid_ = committed_grid;
          fd_->active_grid_started_time_ = ros::Time::now();
          ROS_WARN(
              "RACER_METRIC grid_commit drone=%d grid=%d",
              getId(), committed_grid);
        }
        if (!fd_->target_initialized_ ||
            (target - fd_->active_target_pos_).norm() > 0.5) {
          fd_->target_initialized_ = true;
          fd_->target_block_replans_ = 0;
          fd_->target_started_time_ = ros::Time::now();
          fd_->target_started_pos_ = fd_->odom_pos_;
          fd_->active_target_pos_ = target;
          fd_->target_best_dist_ = -1.0;
          fd_->target_best_time_ = ros::Time(0);
          ROS_WARN(
              "RACER_METRIC target_commit drone=%d x=%.3f y=%.3f z=%.3f",
              getId(), target[0], target[1], target[2]);
        }

        // A newly published trajectory starts a new execution attempt.  If
        // the raw-LiDAR/separation barrier remains active, its next latched
        // True report is handled as a fresh episode after the dwell time.
        fd_->execution_blocked_ = false;
        fd_->execution_blocked_handled_ = false;
        fd_->execution_blocked_since_ = ros::Time(0);

        thread vis_thread(&FastExplorationFSM::visualize, this, 2);
        vis_thread.detach();
        transitState(EXEC_TRAJ, "FSM");
      }
      break;
    }

    case EXEC_TRAJ: {
      auto tn = ros::Time::now();
      // Check whether replan is needed
      LocalTrajData* info = &planner_manager_->local_data_;
      double t_cur = (tn - info->start_time_).toSec();

      if (!fd_->go_back_) {
        if (fd_->execution_blocked_ &&
            (tn - fd_->execution_blocked_since_).toSec() >=
                fp_->execution_blocked_replan_delay_) {
          fd_->execution_blocked_handled_ = true;
          fd_->execution_blocked_ = false;
          fd_->execution_blocked_since_ = ros::Time(0);

          // First ask the obstacle-aware local planner for another route to
          // the same useful viewpoint.  A transient inter-UAV encounter or a
          // newly fused obstacle should not immediately blacklist the task.
          // Only a second blocked execution of the same target releases it.
          if (fd_->target_block_replans_ < 1 &&
              fd_->target_initialized_) {
            ++fd_->target_block_replans_;
            fd_->avoid_collision_ = true;
            fd_->emergency_replan_ = true;
            fd_->replan_same_target_ = false;
            fd_->static_state_ = true;
            replan_pub_.publish(std_msgs::Empty());
            ROS_WARN(
                "RACER_RECOVERY drone=%d execution_blocked_local_retry=1 "
                "target_retry=%d",
                getId(), fd_->target_block_replans_);
            transitState(PLAN_TRAJ, "executionBlockedLocalRetry");
          } else {
            // Blacklisting removed: release only the stale target, keep the
            // HGrid ownership and let the next full route selection decide.
            // Suppress the unreachable viewpoint itself for a while.
            if (fd_->target_initialized_)
              expl_manager_->frontier_finder_->blockViewpoint(
                  fd_->active_target_pos_,
                  tn.toSec() + fp_->viewpoint_block_ttl_);
            fd_->target_block_replans_ = 0;
            fd_->target_initialized_ = false;
            fd_->active_target_grid_ = -1;
            fd_->replan_same_target_ = false;
            fd_->static_state_ = true;
            ROS_ERROR(
                "RACER_RECOVERY drone=%d execution_blocked_replan=1",
                getId());
            transitState(PLAN_TRAJ, "executionBlockedRelease");
          }
          break;
        }

        bool need_replan = false;
        const bool hold_complete =
            targetHoldComplete(tn.toSec());
        const bool frontier_covered =
            t_cur > fp_->replan_thresh2_ &&
            expl_manager_->frontier_finder_->isFrontierCovered();
        // Viewpoint-level exit: the target was reached but the frontier it
        // serves has not changed for arrived_no_gain_timeout seconds.
        if (fd_->target_initialized_ && fp_->arrived_no_gain_timeout_ > 0.0) {
          const double target_distance =
              (fd_->odom_pos_ - fd_->active_target_pos_).norm();
          if (target_distance <= fp_->min_target_execution_distance_) {
            if (fd_->target_arrived_time_.toSec() <= 0.0)
              fd_->target_arrived_time_ = tn;
          } else {
            fd_->target_arrived_time_ = ros::Time(0);
          }
          if (target_distance <= fp_->viewpoint_stagnation_radius_) {
            if (fd_->target_near_time_.toSec() <= 0.0) fd_->target_near_time_ = tn;
          } else {
            fd_->target_near_time_ = ros::Time(0);
          }
          // Displacement watchdog for the active target.
          if (fd_->progress_anchor_time_.toSec() <= 0.0 ||
              (fd_->odom_pos_ - fd_->progress_anchor_pos_).norm() >
                  fp_->no_progress_block_distance_) {
            fd_->progress_anchor_time_ = tn;
            fd_->progress_anchor_pos_ = fd_->odom_pos_;
          }
          const bool no_progress =
              fp_->no_progress_block_timeout_ > 0.0 &&
              (tn - fd_->progress_anchor_time_).toSec() >= fp_->no_progress_block_timeout_;
          // Approach watchdog: best distance to the target must keep improving.
          if (fd_->target_best_dist_ < 0.0 ||
              target_distance < fd_->target_best_dist_ - fp_->approach_progress_distance_) {
            fd_->target_best_dist_ = target_distance;
            fd_->target_best_time_ = tn;
          }
          const bool no_approach =
              fp_->approach_stall_timeout_ > 0.0 &&
              fd_->target_best_time_.toSec() > 0.0 &&
              target_distance > fp_->viewpoint_stagnation_radius_ &&
              (tn - fd_->target_best_time_).toSec() >= fp_->approach_stall_timeout_;
          const bool arrived_no_gain =
              fd_->target_arrived_time_.toSec() > 0.0 &&
              (tn - fd_->target_arrived_time_).toSec() >= fp_->arrived_no_gain_timeout_;
          const bool near_stagnant =
              fp_->viewpoint_stagnation_timeout_ > 0.0 &&
              fd_->target_near_time_.toSec() > 0.0 &&
              (tn - fd_->target_near_time_).toSec() >= fp_->viewpoint_stagnation_timeout_;
          if (!frontier_covered && (arrived_no_gain || near_stagnant || no_progress || no_approach)) {
            expl_manager_->frontier_finder_->blockViewpoint(
                fd_->active_target_pos_,
                tn.toSec() + fp_->viewpoint_block_ttl_);
            ROS_WARN(
                "RACER_RECOVERY drone=%d viewpoint_blocked=%s "
                "target=[%.2f,%.2f] distance_m=%.2f",
                getId(), arrived_no_gain ? "arrived_no_gain"
                                         : (near_stagnant ? "near_stagnant"
                                                          : (no_progress ? "no_progress" : "no_approach")),
                fd_->active_target_pos_[0], fd_->active_target_pos_[1],
                target_distance);
            fd_->target_arrived_time_ = ros::Time(0);
            fd_->target_near_time_ = ros::Time(0);
            fd_->progress_anchor_time_ = ros::Time(0);
            fd_->target_best_dist_ = -1.0;
            fd_->target_best_time_ = ros::Time(0);
            fd_->replan_same_target_ = false;
            fd_->target_initialized_ = false;
            fd_->active_target_grid_ = -1;
            fd_->static_state_ = true;
            transitState(PLAN_TRAJ, "arrivedNoGain");
            break;
          }
        }
        if (frontier_covered && hold_complete) {
          ROS_WARN("Replan: cluster covered=====================================");
          fd_->replan_same_target_ = false;
          need_replan = true;
        } else if (frontier_covered) {
          ROS_WARN_THROTTLE(
              0.5,
              "RACER_METRIC target_switch_suppressed drone=%d "
              "reason=frontier_covered elapsed_s=%.3f progress_m=%.3f",
              getId(), tn.toSec() - fd_->target_started_time_.toSec(),
              (fd_->odom_pos_ - fd_->target_started_pos_).norm());
        }
        if (!need_replan && info->duration_ - t_cur < fp_->replan_thresh1_) {
          // Replan if traj is almost fully executed
          ROS_WARN("Replan: traj fully executed=================================");
          need_replan = true;
          fd_->replan_same_target_ = !hold_complete;
        } else if (!need_replan && t_cur > fp_->replan_thresh3_) {
          // Replan after some time
          ROS_WARN("Replan: periodic call=======================================");
          need_replan = true;
          fd_->replan_same_target_ = !hold_complete;
        }

        if (need_replan) {
          if (fd_->replan_same_target_) {
            ROS_WARN(
                "RACER_METRIC target_hold_replan drone=%d "
                "elapsed_s=%.3f progress_m=%.3f",
                getId(), tn.toSec() - fd_->target_started_time_.toSec(),
                (fd_->odom_pos_ - fd_->target_started_pos_).norm());
            transitState(PLAN_TRAJ, "FSM-target-hold");
          } else {
            const int frontier_count =
                expl_manager_->updateFrontierStruct(fd_->odom_pos_);
            const auto& own_grids =
                expl_manager_->ed_->swarm_state_[getId() - 1].grid_ids_;
            if (frontier_count != 0 || !own_grids.empty()) {
              // A REMOTE_PENDING HGrid remains executable even before its
              // source frontier appears in this vehicle's local map.
              thread vis_thread(&FastExplorationFSM::visualize, this, 1);
              vis_thread.detach();
              transitState(PLAN_TRAJ, "FSM");
            } else {
              fd_->last_check_frontier_time_ = ros::Time::now();
              transitState(IDLE, "FSM");
              ROS_WARN("Idle since no frontier or owned HGrid is detected");
              fd_->static_state_ = true;
              replan_pub_.publish(std_msgs::Empty());
              // clearVisMarker();
              visualize(1);
            }
          }
        }
      } else {
        // Check if reach goal
        auto pos = info->position_traj_.evaluateDeBoorT(t_cur);
        if ((pos - expl_manager_->ed_->next_pos_).norm() < 1.0) {
          replan_pub_.publish(std_msgs::Empty());
          clearVisMarker();
          transitState(FINISH, "FSM");
          return;
        }
        if (t_cur > fp_->replan_thresh3_ || info->duration_ - t_cur < fp_->replan_thresh1_) {
          // Replan for going back
          transitState(PLAN_TRAJ, "FSM");
          thread vis_thread(&FastExplorationFSM::visualize, this, 1);
          vis_thread.detach();
        }
      }

      break;
    }
  }
}

int FastExplorationFSM::callExplorationPlanner() {
  ros::Time time_r = ros::Time::now() + ros::Duration(fp_->replan_time_);

  int res;
  if (fd_->avoid_collision_ || fd_->go_back_ ||
      fd_->replan_same_target_) {  // Only replan trajectory
    res = expl_manager_->planTrajToView(fd_->start_pt_, fd_->start_vel_, fd_->start_acc_,
        fd_->start_yaw_, expl_manager_->ed_->next_pos_, expl_manager_->ed_->next_yaw_);
    fd_->avoid_collision_ = false;
    fd_->replan_same_target_ = false;
  } else {  // Do full planning normally
    res = expl_manager_->planExploreMotion(
        fd_->start_pt_, fd_->start_vel_, fd_->start_acc_, fd_->start_yaw_);
  }

  if (res == SUCCEED) {
    auto info = &planner_manager_->local_data_;
    info->start_time_ = (ros::Time::now() - time_r).toSec() > 0 ? ros::Time::now() : time_r;

    bspline::Bspline bspline;
    bspline.order = planner_manager_->pp_.bspline_degree_;
    bspline.start_time = info->start_time_;
    bspline.traj_id = info->traj_id_;
    Eigen::MatrixXd pos_pts = info->position_traj_.getControlPoint();
    for (int i = 0; i < pos_pts.rows(); ++i) {
      geometry_msgs::Point pt;
      pt.x = pos_pts(i, 0);
      pt.y = pos_pts(i, 1);
      pt.z = pos_pts(i, 2);
      bspline.pos_pts.push_back(pt);
    }
    Eigen::VectorXd knots = info->position_traj_.getKnot();
    for (int i = 0; i < knots.rows(); ++i) {
      bspline.knots.push_back(knots(i));
    }
    Eigen::MatrixXd yaw_pts = info->yaw_traj_.getControlPoint();
    for (int i = 0; i < yaw_pts.rows(); ++i) {
      double yaw = yaw_pts(i, 0);
      bspline.yaw_pts.push_back(yaw);
    }
    bspline.yaw_dt = info->yaw_traj_.getKnotSpan();
    fd_->newest_traj_ = bspline;
  }
  return res;
}

void FastExplorationFSM::visualize(int content) {
  if (!fp_->enable_planning_visualization_) return;

  // content 1: frontier; 2 paths & trajs
  auto info = &planner_manager_->local_data_;
  auto plan_data = &planner_manager_->plan_data_;
  auto ed_ptr = expl_manager_->ed_;

  auto getColorVal = [&](const int& id, const int& num, const int& drone_id) {
    double a = (drone_id - 1) / double(num + 1);
    double b = 1 / double(num + 1);
    return a + b * double(id) / ed_ptr->frontiers_.size();
  };

  if (content == 1) {
    // Draw frontier
    static int last_ftr_num = 0;
    static int last_dftr_num = 0;
    for (int i = 0; i < ed_ptr->frontiers_.size(); ++i) {
      visualization_->drawCubes(ed_ptr->frontiers_[i], 0.1,
          visualization_->getColor(double(i) / ed_ptr->frontiers_.size(), 0.4), "frontier", i, 4);

      // getColorVal(i, expl_manager_->ep_->drone_num_, expl_manager_->ep_->drone_id_)
      // double(i) / ed_ptr->frontiers_.size()

      // visualization_->drawBox(ed_ptr->frontier_boxes_[i].first,
      // ed_ptr->frontier_boxes_[i].second,
      //     Vector4d(0.5, 0, 1, 0.3), "frontier_boxes", i, 4);
    }
    for (int i = ed_ptr->frontiers_.size(); i < last_ftr_num; ++i) {
      visualization_->drawCubes({}, 0.1, Vector4d(0, 0, 0, 1), "frontier", i, 4);
      // visualization_->drawBox(Vector3d(0, 0, 0), Vector3d(0, 0, 0), Vector4d(1, 0, 0, 0.3),
      // "frontier_boxes", i, 4);
    }
    last_ftr_num = ed_ptr->frontiers_.size();

    // for (int i = 0; i < ed_ptr->dead_frontiers_.size(); ++i)
    //   visualization_->drawCubes(
    //       ed_ptr->dead_frontiers_[i], 0.1, Vector4d(0, 0, 0, 0.5), "dead_frontier", i, 4);
    // for (int i = ed_ptr->dead_frontiers_.size(); i < last_dftr_num; ++i)
    //   visualization_->drawCubes({}, 0.1, Vector4d(0, 0, 0, 0.5), "dead_frontier", i, 4);
    // last_dftr_num = ed_ptr->dead_frontiers_.size();

    // // Draw updated box
    // Vector3d bmin, bmax;
    // planner_manager_->edt_environment_->sdf_map_->getUpdatedBox(bmin, bmax, false);
    // visualization_->drawBox(
    //     (bmin + bmax) / 2.0, bmax - bmin, Vector4d(0, 1, 0, 0.3), "updated_box", 0, 4);

    // vector<Eigen::Vector3d> bmins, bmaxs;
    // planner_manager_->edt_environment_->sdf_map_->mm_->getChunkBoxes(bmins, bmaxs, false);
    // for (int i = 0; i < bmins.size(); ++i) {
    //   visualization_->drawBox((bmins[i] + bmaxs[i]) / 2.0, bmaxs[i] - bmins[i],
    //       Vector4d(0, 1, 1, 0.3), "updated_box", i + 1, 4);
    // }

  } else if (content == 2) {

    // Hierarchical grid and global tour --------------------------------
    // vector<Eigen::Vector3d> pts1, pts2;
    // expl_manager_->uniform_grid_->getPath(pts1, pts2);
    // visualization_->drawLines(pts1, pts2, 0.05, Eigen::Vector4d(1, 0.3, 0, 1), "partition", 0,
    // 6);

    if (expl_manager_->ep_->drone_id_ == 1) {
      vector<Eigen::Vector3d> pts1, pts2;
      expl_manager_->hgrid_->getGridMarker(pts1, pts2);
      visualization_->drawLines(pts1, pts2, 0.05, Eigen::Vector4d(1, 0, 1, 0.5), "partition", 1, 6);

      vector<Eigen::Vector3d> pts;
      vector<string> texts;
      expl_manager_->hgrid_->getGridMarker2(pts, texts);
      static int last_text_num = 0;
      for (int i = 0; i < pts.size(); ++i) {
        visualization_->drawText(pts[i], texts[i], 1, Eigen::Vector4d(0, 0, 0, 1), "text", i, 6);
      }
      for (int i = pts.size(); i < last_text_num; ++i) {
        visualization_->drawText(
            Eigen::Vector3d(0, 0, 0), string(""), 1, Eigen::Vector4d(0, 0, 0, 1), "text", i, 6);
      }
      last_text_num = pts.size();

      // // Pub hgrid to ground node
      // exploration_manager::HGrid hgrid;
      // hgrid.stamp = ros::Time::now().toSec();
      // for (int i = 0; i < pts1.size(); ++i) {
      //   geometry_msgs::Point pt1, pt2;
      //   pt1.x = pts1[i][0];
      //   pt1.y = pts1[i][1];
      //   pt1.z = pts1[i][2];
      //   hgrid.points1.push_back(pt1);
      //   pt2.x = pts2[i][0];
      //   pt2.y = pts2[i][1];
      //   pt2.z = pts2[i][2];
      //   hgrid.points2.push_back(pt2);
      // }
      // hgrid_pub_.publish(hgrid);
    }

    auto grid_tour = expl_manager_->ed_->grid_tour_;
    // auto grid_tour = expl_manager_->ed_->grid_tour2_;
    // for (auto& pt : grid_tour) pt = pt + trans;

    visualization_->drawLines(grid_tour, 0.05,
        PlanningVisualization::getColor(
            (expl_manager_->ep_->drone_id_ - 1) / double(expl_manager_->ep_->drone_num_)),
        "grid_tour", 0, 6);

    // Publish grid tour to ground node
    exploration_manager::GridTour tour;
    for (int i = 0; i < grid_tour.size(); ++i) {
      geometry_msgs::Point point;
      point.x = grid_tour[i][0];
      point.y = grid_tour[i][1];
      point.z = grid_tour[i][2];
      tour.points.push_back(point);
    }
    tour.drone_id = expl_manager_->ep_->drone_id_;
    tour.stamp = ros::Time::now().toSec();
    grid_tour_pub_.publish(tour);

    // visualization_->drawSpheres(
    //     expl_manager_->ed_->grid_tour_, 0.3, Eigen::Vector4d(0, 1, 0, 1), "grid_tour", 1, 6);
    // visualization_->drawLines(
    //     expl_manager_->ed_->grid_tour2_, 0.05, Eigen::Vector4d(0, 1, 0, 0.5), "grid_tour", 2, 6);

    // Top viewpoints and frontier tour-------------------------------------

    // visualization_->drawSpheres(ed_ptr->points_, 0.2, Vector4d(0, 0.5, 0, 1), "points", 0, 6);
    // visualization_->drawLines(
    //     ed_ptr->points_, ed_ptr->views_, 0.05, Vector4d(0, 1, 0.5, 1), "view", 0, 6);
    // visualization_->drawLines(
    //     ed_ptr->points_, ed_ptr->averages_, 0.03, Vector4d(1, 0, 0, 1), "point-average", 0, 6);

    // auto frontier = ed_ptr->frontier_tour_;
    // for (auto& pt : frontier) pt = pt + trans;
    // visualization_->drawLines(frontier, 0.07,
    //     PlanningVisualization::getColor(
    //         (expl_manager_->ep_->drone_id_ - 1) / double(expl_manager_->ep_->drone_num_), 0.6),
    //     "frontier_tour", 0, 6);

    // for (int i = 0; i < ed_ptr->other_tours_.size(); ++i) {
    //   visualization_->drawLines(
    //       ed_ptr->other_tours_[i], 0.07, Eigen::Vector4d(0, 0, 1, 1), "other_tours", i, 6);
    // }

    // Locally refined viewpoints and refined tour-------------------------------

    // visualization_->drawSpheres(
    //     ed_ptr->refined_points_, 0.2, Vector4d(0, 0, 1, 1), "refined_pts", 0, 6);
    // visualization_->drawLines(
    //     ed_ptr->refined_points_, ed_ptr->refined_views_, 0.05, Vector4d(0.5, 0, 1, 1),
    //     "refined_view", 0, 6);
    // visualization_->drawLines(
    //     ed_ptr->refined_tour_, 0.07,
    //     PlanningVisualization::getColor(
    //         (expl_manager_->ep_->drone_id_ - 1) / double(expl_manager_->ep_->drone_num_), 0.6),
    //     "refined_tour", 0, 6);

    // visualization_->drawLines(ed_ptr->refined_views1_, ed_ptr->refined_views2_, 0.04, Vector4d(0,
    // 0, 0, 1),
    //                           "refined_view", 0, 6);
    // visualization_->drawLines(ed_ptr->refined_points_, ed_ptr->unrefined_points_, 0.05,
    // Vector4d(1, 1, 0, 1),
    //                           "refine_pair", 0, 6);
    // for (int i = 0; i < ed_ptr->n_points_.size(); ++i)
    //   visualization_->drawSpheres(ed_ptr->n_points_[i], 0.1,
    //                               visualization_->getColor(double(ed_ptr->refined_ids_[i]) /
    //                               ed_ptr->frontiers_.size()),
    //                               "n_points", i, 6);
    // for (int i = ed_ptr->n_points_.size(); i < 15; ++i)
    //   visualization_->drawSpheres({}, 0.1, Vector4d(0, 0, 0, 1), "n_points", i, 6);

    // Trajectory-------------------------------------------

    // visualization_->drawSpheres(
    //     { ed_ptr->next_goal_ /* + trans */ }, 0.3, Vector4d(0, 0, 1, 1), "next_goal", 0, 6);

    // vector<Eigen::Vector3d> next_yaw_vis;
    // next_yaw_vis.push_back(ed_ptr->next_goal_ /* + trans */);
    // next_yaw_vis.push_back(
    //     ed_ptr->next_goal_ /* + trans */ +
    //     2.0 * Eigen::Vector3d(cos(ed_ptr->next_yaw_), sin(ed_ptr->next_yaw_), 0));
    // visualization_->drawLines(next_yaw_vis, 0.1, Eigen::Vector4d(0, 0, 1, 1), "next_goal", 1, 6);
    // visualization_->drawSpheres(
    //     { ed_ptr->next_pos_ /* + trans */ }, 0.3, Vector4d(0, 1, 0, 1), "next_pos", 0, 6);

    // Eigen::MatrixXd ctrl_pt = info->position_traj_.getControlPoint();
    // for (int i = 0; i < ctrl_pt.rows(); ++i) {
    //   for (int j = 0; j < 3; ++j) ctrl_pt(i, j) = ctrl_pt(i, j) + trans[j];
    // }
    // NonUniformBspline position_traj(ctrl_pt, 3, info->position_traj_.getKnotSpan());

    visualization_->drawBspline(info->position_traj_, 0.1,
        PlanningVisualization::getColor(
            (expl_manager_->ep_->drone_id_ - 1) / double(expl_manager_->ep_->drone_num_)),
        false, 0.15, Vector4d(1, 1, 0, 1));

    // visualization_->drawLines(
    //     expl_manager_->ed_->path_next_goal_, 0.1, Eigen::Vector4d(0, 1, 0, 1), "astar", 0, 6);
    // visualization_->drawSpheres(
    //     expl_manager_->ed_->kino_path_, 0.1, Eigen::Vector4d(0, 0, 1, 1), "kino", 0, 6);
    // visualization_->drawSpheres(plan_data->kino_path_, 0.1, Vector4d(1, 0, 1, 1), "kino_path", 0,
    // 0); visualization_->drawLines(ed_ptr->path_next_goal_, 0.05, Vector4d(0, 1, 1, 1),
    // "next_goal", 1, 6);

    // // Draw trajs of other drones
    // vector<NonUniformBspline> trajs;
    // planner_manager_->swarm_traj_data_.getValidTrajs(trajs);
    // for (int k = 0; k < trajs.size(); ++k) {
    //   visualization_->drawBspline(trajs[k], 0.1, Eigen::Vector4d(1, 1, 0, 1), false, 0.15,
    //       Eigen::Vector4d(0, 0, 1, 1), k + 1);
    // }
  }
}

void FastExplorationFSM::clearVisMarker() {
  if (!fp_->enable_planning_visualization_) return;

  for (int i = 0; i < 10; ++i) {
    visualization_->drawCubes({}, 0.1, Vector4d(0, 0, 0, 1), "frontier", i, 4);
    // visualization_->drawCubes({}, 0.1, Vector4d(0, 0, 0, 1), "dead_frontier", i, 4);
    // visualization_->drawBox(Vector3d(0, 0, 0), Vector3d(0, 0, 0), Vector4d(1, 0, 0, 0.3),
    // "frontier_boxes", i, 4);
  }
  // visualization_->drawSpheres({}, 0.2, Vector4d(0, 0.5, 0, 1), "points", 0, 6);
  visualization_->drawLines({}, 0.07, Vector4d(0, 0.5, 0, 1), "frontier_tour", 0, 6);
  visualization_->drawLines({}, 0.07, Vector4d(0, 0.5, 0, 1), "grid_tour", 0, 6);
  // visualization_->drawSpheres({}, 0.2, Vector4d(0, 0, 1, 1), "refined_pts", 0, 6);
  // visualization_->drawLines({}, {}, 0.05, Vector4d(0.5, 0, 1, 1), "refined_view", 0, 6);
  // visualization_->drawLines({}, 0.07, Vector4d(0, 0, 1, 1), "refined_tour", 0, 6);
  visualization_->drawSpheres({}, 0.1, Vector4d(0, 0, 1, 1), "B-Spline", 0, 0);

  // visualization_->drawLines({}, {}, 0.03, Vector4d(1, 0, 0, 1), "current_pose", 0, 6);
}

void FastExplorationFSM::frontierCallback(const ros::TimerEvent& e) {
  if (state_ == WAIT_TRIGGER) {
    auto ft = expl_manager_->frontier_finder_;
    auto ed = expl_manager_->ed_;

    auto getColorVal = [&](const int& id, const int& num, const int& drone_id) {
      double a = (drone_id - 1) / double(num + 1);
      double b = 1 / double(num + 1);
      return a + b * double(id) / ed->frontiers_.size();
    };

    // ft->searchFrontiers();
    // ft->computeFrontiersToVisit();
    // ft->updateFrontierCostMatrix();

    // ft->getFrontiers(ed->frontiers_);
    // ft->getFrontierBoxes(ed->frontier_boxes_);

    expl_manager_->updateFrontierStruct(fd_->odom_pos_);

    cout << "odom: " << fd_->odom_pos_.transpose() << endl;
    vector<int> tmp_id1;
    vector<vector<int>> tmp_id2;
    bool status = expl_manager_->findGlobalTourOfGrid(
        { fd_->odom_pos_ }, { fd_->odom_vel_ }, tmp_id1, tmp_id2, true);

    if (fp_->enable_planning_visualization_) {
      // Draw frontier and bounding box
      for (int i = 0; i < ed->frontiers_.size(); ++i) {
        visualization_->drawCubes(ed->frontiers_[i], 0.1,
            visualization_->getColor(double(i) / ed->frontiers_.size(), 0.4), "frontier", i, 4);
      }
      for (int i = ed->frontiers_.size(); i < 50; ++i) {
        visualization_->drawCubes({}, 0.1, Vector4d(0, 0, 0, 1), "frontier", i, 4);
      }
      if (status)
        visualize(2);
      else
        visualization_->drawLines({}, 0.07, Vector4d(0, 0.5, 0, 1), "grid_tour", 0, 6);
    }

    // Draw grid tour
  }
}

void FastExplorationFSM::triggerCallback(const geometry_msgs::PoseStampedConstPtr& msg) {

  // // Debug traj planner
  // Eigen::Vector3d pos;
  // pos << msg->pose.position.x, msg->pose.position.y, 1;
  // expl_manager_->ed_->next_pos_ = pos;

  // Eigen::Vector3d dir = pos - fd_->odom_pos_;
  // expl_manager_->ed_->next_yaw_ = atan2(dir[1], dir[0]);
  // fd_->go_back_ = true;
  // transitState(PLAN_TRAJ, "triggerCallback");
  // return;

  if (state_ != WAIT_TRIGGER || fd_->trigger_) return;
  fd_->trigger_ = true;
  cout << "Triggered!" << endl;
  fd_->start_pos_ = fd_->odom_pos_;
  fd_->trigger_ready_time_ =
      ros::Time::now() + ros::Duration(fp_->trigger_stagger_ * (getId() - 1));
  ROS_WARN_STREAM("Start expl pos: " << fd_->start_pos_.transpose());
  ROS_WARN(
      "RACER_METRIC trigger drone=%d stagger_s=%.3f", getId(),
      fp_->trigger_stagger_ * (getId() - 1));
}

void FastExplorationFSM::safetyCallback(const ros::TimerEvent& e) {
  if (state_ == EXPL_STATE::EXEC_TRAJ) {
    // Check safety and trigger replan if necessary
    double dist;
    bool safe = planner_manager_->checkTrajCollision(
        dist, expl_manager_->ed_->escape_inflation_);
    if (!safe) {
      ROS_WARN("Replan: collision detected==================================");
      fd_->avoid_collision_ = true;
      fd_->emergency_replan_ = true;
      fd_->static_state_ = true;
      // This is the one non-transactional case: brake the already unsafe
      // trajectory before attempting a replacement.
      replan_pub_.publish(std_msgs::Empty());
      transitState(PLAN_TRAJ, "safetyCallback");
    }
  }
}

void FastExplorationFSM::odometryCallback(const nav_msgs::OdometryConstPtr& msg) {
  fd_->odom_pos_(0) = msg->pose.pose.position.x;
  fd_->odom_pos_(1) = msg->pose.pose.position.y;
  fd_->odom_pos_(2) = msg->pose.pose.position.z;

  // Keep a per-robot HGrid visit history at odometry rate. This avoids
  // labelling an HGrid "unvisited" merely because a long trajectory crossed
  // it between two relatively sparse global replans.
  expl_manager_->recordTrajectoryPosition(fd_->odom_pos_);

  fd_->odom_vel_(0) = msg->twist.twist.linear.x;
  fd_->odom_vel_(1) = msg->twist.twist.linear.y;
  fd_->odom_vel_(2) = msg->twist.twist.linear.z;

  fd_->odom_orient_.w() = msg->pose.pose.orientation.w;
  fd_->odom_orient_.x() = msg->pose.pose.orientation.x;
  fd_->odom_orient_.y() = msg->pose.pose.orientation.y;
  fd_->odom_orient_.z() = msg->pose.pose.orientation.z;

  Eigen::Vector3d rot_x = fd_->odom_orient_.toRotationMatrix().block<3, 1>(0, 0);
  fd_->odom_yaw_ = atan2(rot_x(1), rot_x(0));

  if (!fd_->have_odom_) {
    fd_->have_odom_ = true;
    fd_->fsm_init_time_ = ros::Time::now();
  }
}

void FastExplorationFSM::transitState(EXPL_STATE new_state, string pos_call) {
  int pre_s = int(state_);
  state_ = new_state;
  ROS_INFO_STREAM("[" + pos_call + "]: Drone "
                  << getId()
                  << " from " + fd_->state_str_[pre_s] + " to " + fd_->state_str_[int(new_state)]);
}

void FastExplorationFSM::frontierShareTimerCallback(const ros::TimerEvent& e) {
  if (!expl_manager_->ep_->peer_frontier_rescue_enabled_ || !fd_->have_odom_)
    return;

  // Frontier cells without the matching free/occupied evidence are not useful
  // to another planner. Seal RACER's residual map chunk before advertising the
  // frontier snapshot; normal chunk-stamp exchange delivers it asynchronously.
  if (expl_manager_->sdf_map_->mm_)
    expl_manager_->sdf_map_->mm_->flushPendingChunk();

  exploration_manager::FrontierShare msg;
  msg.from_drone_id = getId();
  msg.stamp = ros::Time::now().toSec();
  msg.frontier_epoch = expl_manager_->ed_->frontier_update_seq_;

  auto& data = *expl_manager_->ed_;
  if (data.peer_rescue_valid_) {
    data.peer_rescue_claim_until_ =
        msg.stamp + expl_manager_->ep_->peer_frontier_claim_ttl_;
    msg.claimed_signature = data.peer_rescue_candidate_.signature_;
    msg.claim_until = data.peer_rescue_claim_until_;
  } else {
    msg.claimed_signature = 0;
    msg.claim_until = 0.0;
  }

  const int aligned_count = std::min(
      std::min(static_cast<int>(data.frontiers_.size()),
          static_cast<int>(data.points_.size())),
      std::min(static_cast<int>(data.averages_.size()),
          static_cast<int>(data.yaws_.size())));
  const int cell_limit =
      std::max(1, expl_manager_->ep_->peer_frontier_cell_limit_);
  std::unordered_set<int> reserved_ids(
      data.refined_ids_.begin(), data.refined_ids_.end());
  auto is_reserved = [&](const int index) {
    if (reserved_ids.count(index) > 0) return true;
    return fd_->target_initialized_ &&
           (data.points_[index] - fd_->active_target_pos_).norm() <=
               expl_manager_->ep_->peer_frontier_reserved_radius_;
  };
  vector<int> publish_indices(aligned_count);
  for (int index = 0; index < aligned_count; ++index)
    publish_indices[index] = index;
  // Put unreserved work first so a bounded packet cannot contain only this
  // vehicle's current local tour while omitting useful assistance targets.
  stable_sort(publish_indices.begin(), publish_indices.end(),
      [&](const int first, const int second) {
        return static_cast<int>(is_reserved(first)) <
               static_cast<int>(is_reserved(second));
      });
  const int candidate_count = std::min(
      std::max(0, expl_manager_->ep_->peer_frontier_share_limit_),
      aligned_count);

  msg.cell_offsets.push_back(0);
  for (int slot = 0; slot < candidate_count; ++slot) {
    const int index = publish_indices[slot];
    msg.signatures.push_back(
        expl_manager_->frontierSignature(data.averages_[index]));
    const bool reserved = is_reserved(index);
    msg.reserved.push_back(reserved ? 1 : 0);

    geometry_msgs::Point suggested;
    suggested.x = data.points_[index][0];
    suggested.y = data.points_[index][1];
    suggested.z = data.points_[index][2];
    msg.suggested_viewpoints.push_back(suggested);
    msg.suggested_yaws.push_back(data.yaws_[index]);
    msg.suggested_visible_cells.push_back(data.frontiers_[index].size());

    const auto& cells = data.frontiers_[index];
    const int stride = std::max(
        1, static_cast<int>(std::ceil(
               static_cast<double>(cells.size()) / cell_limit)));
    int inserted = 0;
    for (int cell_index = 0;
         cell_index < static_cast<int>(cells.size()) && inserted < cell_limit;
         cell_index += stride, ++inserted) {
      const auto& cell = cells[cell_index];
      geometry_msgs::Point point;
      point.x = cell[0];
      point.y = cell[1];
      point.z = cell[2];
      msg.cells.push_back(point);
    }
    msg.cell_offsets.push_back(msg.cells.size());
  }

  frontier_share_pub_.publish(msg);
  ROS_INFO_THROTTLE(
      5.0,
      "RACER_METRIC frontier_share drone=%d candidates=%zu cells=%zu "
      "claim=%llu epoch=%llu",
      getId(), msg.signatures.size(), msg.cells.size(),
      static_cast<unsigned long long>(msg.claimed_signature),
      static_cast<unsigned long long>(msg.frontier_epoch));
}

void FastExplorationFSM::frontierShareMsgCallback(
    const exploration_manager::FrontierShareConstPtr& msg) {
  if (!expl_manager_->ep_->peer_frontier_rescue_enabled_ ||
      msg->from_drone_id == getId())
    return;
  if (msg->from_drone_id <= 0 ||
      msg->from_drone_id > expl_manager_->ep_->drone_num_)
    return;

  auto& shares = expl_manager_->ed_->peer_frontier_shares_;
  const auto previous = shares.find(msg->from_drone_id);
  if (previous != shares.end() && previous->second.stamp_ + 1e-4 >= msg->stamp)
    return;

  const size_t count = msg->signatures.size();
  bool malformed = msg->reserved.size() != count ||
                   msg->suggested_viewpoints.size() != count ||
                   msg->suggested_yaws.size() != count ||
                   msg->suggested_visible_cells.size() != count ||
                   msg->cell_offsets.size() != count + 1;
  if (!malformed) {
    malformed = msg->cell_offsets.front() != 0 ||
                msg->cell_offsets.back() !=
                    static_cast<int>(msg->cells.size());
    for (size_t index = 0; !malformed && index + 1 < msg->cell_offsets.size(); ++index) {
      malformed = msg->cell_offsets[index] < 0 ||
                  msg->cell_offsets[index] > msg->cell_offsets[index + 1] ||
                  msg->cell_offsets[index + 1] >
                      static_cast<int>(msg->cells.size());
    }
  }
  if (malformed) {
    ROS_WARN(
        "Reject malformed FrontierShare from drone %d: candidates=%zu "
        "offsets=%zu cells=%zu",
        msg->from_drone_id, count, msg->cell_offsets.size(),
        msg->cells.size());
    return;
  }

  PeerFrontierShareState share;
  share.stamp_ = msg->stamp;
  share.frontier_epoch_ = msg->frontier_epoch;
  share.claimed_signature_ = msg->claimed_signature;
  share.claim_until_ = msg->claim_until;
  for (size_t index = 0; index < count; ++index) {
    const auto& suggested = msg->suggested_viewpoints[index];
    if (!std::isfinite(suggested.x) || !std::isfinite(suggested.y) ||
        !std::isfinite(suggested.z) ||
        !std::isfinite(msg->suggested_yaws[index]))
      continue;
    PeerFrontierCandidate candidate;
    candidate.source_drone_id_ = msg->from_drone_id;
    candidate.signature_ = msg->signatures[index];
    candidate.stamp_ = msg->stamp;
    candidate.reserved_ = msg->reserved[index] != 0;
    candidate.suggested_viewpoint_ = Eigen::Vector3d(
        suggested.x, suggested.y, suggested.z);
    candidate.suggested_yaw_ = msg->suggested_yaws[index];
    candidate.suggested_visible_cells_ =
        msg->suggested_visible_cells[index];
    for (int cell_index = msg->cell_offsets[index];
         cell_index < msg->cell_offsets[index + 1]; ++cell_index) {
      const auto& point = msg->cells[cell_index];
      if (std::isfinite(point.x) && std::isfinite(point.y) &&
          std::isfinite(point.z))
        candidate.cells_.emplace_back(point.x, point.y, point.z);
    }
    if (!candidate.cells_.empty())
      share.candidates_.push_back(std::move(candidate));
  }
  shares[msg->from_drone_id] = std::move(share);

  const double now = ros::Time::now().toSec();
  auto& data = *expl_manager_->ed_;
  if (data.peer_rescue_valid_ && msg->claimed_signature != 0 &&
      msg->claim_until > now &&
      msg->claimed_signature == data.peer_rescue_candidate_.signature_ &&
      msg->from_drone_id < getId()) {
    const uint64_t signature = data.peer_rescue_candidate_.signature_;
    expl_manager_->rejectPeerFrontierRescue("lower_id_claim");
    if (state_ == EXEC_TRAJ || state_ == PUB_TRAJ || state_ == PLAN_TRAJ) {
      replan_pub_.publish(std_msgs::Empty());
      fd_->static_state_ = true;
      fd_->last_check_frontier_time_ = ros::Time(0);
      transitState(IDLE, "peerFrontierClaimYield");
    }
    ROS_WARN(
        "RACER_METRIC peer_frontier_claim_yield drone=%d winner=%d "
        "signature=%llu",
        getId(), msg->from_drone_id,
        static_cast<unsigned long long>(signature));
    return;
  }

  // Wake a completed/idle vehicle immediately; updateFrontierStruct performs
  // the expensive local reconstruction and A* outside this message callback.
  if (!shares[msg->from_drone_id].candidates_.empty()) {
    fd_->last_check_frontier_time_ = ros::Time(0);
    if (state_ == FINISH && fd_->trigger_)
      transitState(IDLE, "peerFrontierAvailable");
  }
}

void FastExplorationFSM::droneStateTimerCallback(const ros::TimerEvent& e) {
  // Broadcast own state periodically
  reconcileDuplicateOwnership();
  exploration_manager::DroneState msg;
  msg.drone_id = getId();

  auto& state = expl_manager_->ed_->swarm_state_[msg.drone_id - 1];

  if (fd_->static_state_) {
    state.pos_ = fd_->odom_pos_;
    state.vel_ = fd_->odom_vel_;
    state.yaw_ = fd_->odom_yaw_;
  } else {
    LocalTrajData* info = &planner_manager_->local_data_;
    double t_r = (ros::Time::now() - info->start_time_).toSec();
    state.pos_ = info->position_traj_.evaluateDeBoorT(t_r);
    state.vel_ = info->velocity_traj_.evaluateDeBoorT(t_r);
    state.yaw_ = info->yaw_traj_.evaluateDeBoorT(t_r)[0];
  }
  state.stamp_ = ros::Time::now().toSec();
  msg.pos = { float(state.pos_[0]), float(state.pos_[1]), float(state.pos_[2]) };
  msg.vel = { float(state.vel_[0]), float(state.vel_[1]), float(state.vel_[2]) };
  msg.yaw = state.yaw_;
  for (auto id : state.grid_ids_) {
    msg.grid_ids.push_back(id);
    const auto epoch = state.grid_epochs_.find(id);
    msg.grid_epochs.push_back(
        epoch == state.grid_epochs_.end() ? state.assignment_epoch_
                                          : epoch->second);
  }
  msg.assignment_epoch = state.assignment_epoch_;
  purgeExpiredFailures(state, state.stamp_);
  vector<pair<int, double>> failures(
      state.failed_grid_until_.begin(), state.failed_grid_until_.end());
  sort(failures.begin(), failures.end());
  for (const auto& failure : failures) {
    msg.failed_grid_ids.push_back(failure.first);
    msg.failed_grid_until.push_back(failure.second);
  }
  msg.recent_attempt_time = state.recent_attempt_time_;
  msg.stamp = state.stamp_;

  drone_state_pub_.publish(msg);
}

void FastExplorationFSM::droneStateMsgCallback(const exploration_manager::DroneStateConstPtr& msg) {
  // Update other drones' states
  if (msg->drone_id == getId()) return;

  // Simulate swarm communication loss
  Eigen::Vector3d msg_pos(msg->pos[0], msg->pos[1], msg->pos[2]);
  // if ((msg_pos - fd_->odom_pos_).norm() > 6.0) return;

  auto& drone_state = expl_manager_->ed_->swarm_state_[msg->drone_id - 1];
  if (drone_state.stamp_ + 1e-4 >= msg->stamp) return;  // Avoid unordered msg

  drone_state.pos_ = Eigen::Vector3d(msg->pos[0], msg->pos[1], msg->pos[2]);
  drone_state.vel_ = Eigen::Vector3d(msg->vel[0], msg->vel[1], msg->vel[2]);
  drone_state.yaw_ = msg->yaw;
  drone_state.grid_ids_.clear();
  drone_state.grid_epochs_.clear();
  for (int i = 0; i < static_cast<int>(msg->grid_ids.size()); ++i) {
    const int id = msg->grid_ids[i];
    drone_state.grid_ids_.push_back(id);
    drone_state.grid_epochs_[id] =
        i < static_cast<int>(msg->grid_epochs.size())
            ? msg->grid_epochs[i]
            : msg->assignment_epoch;
  }
  drone_state.assignment_epoch_ = msg->assignment_epoch;
  drone_state.failed_grid_until_.clear();
  const int failed_count = std::min(
      msg->failed_grid_ids.size(), msg->failed_grid_until.size());
  for (int i = 0; i < failed_count; ++i) {
    if (msg->failed_grid_until[i] > ros::Time::now().toSec())
      drone_state.failed_grid_until_[msg->failed_grid_ids[i]] =
          msg->failed_grid_until[i];
  }
  drone_state.stamp_ = msg->stamp;
  drone_state.recent_attempt_time_ = msg->recent_attempt_time;
  reconcileDuplicateOwnership();

  // std::cout << "Drone " << getId() << " get drone " << int(msg->drone_id) << "'s state" <<
  // std::endl; std::cout << drone_state.pos_.transpose() << std::endl;
}

void FastExplorationFSM::optTimerCallback(const ros::TimerEvent& e) {
  if (state_ == INIT) return;

  // Select nearby drone not interacting with recently
  reconcileDuplicateOwnership();
  auto& states = expl_manager_->ed_->swarm_state_;
  auto& state1 = states[getId() - 1];
  // bool urgent = (state1.grid_ids_.size() <= 1 /* && !state1.grid_ids_.empty() */);
  bool urgent = state1.grid_ids_.empty();
  auto tn = ros::Time::now().toSec();

  // The upstream protocol assumes asynchronously running vehicles. Gazebo
  // launches all six FSM timers on the same callback phase, which can make
  // several senders lock the same receiver at exactly the same stamp forever.
  // Introduce only deterministic sender slots; the RACER ACVRP objective,
  // messages, acceptance rule and assignments remain unchanged.
  if (!fd_->pair_opt_phase_started_) {
    fd_->pair_opt_phase_started_ = true;
    fd_->pair_opt_ready_time_ =
        ros::Time::now() +
        ros::Duration(fp_->pair_opt_timer_stagger_ * (getId() - 1));
    ROS_WARN(
        "RACER_METRIC pair_phase drone=%d delay_s=%.3f", getId(),
        fp_->pair_opt_timer_stagger_ * (getId() - 1));
    return;
  }
  if (ros::Time::now() < fd_->pair_opt_ready_time_) return;

  // A one-time ready delay does not de-synchronize periodic ROS timers: after
  // the delay they still fire on the common 50 ms phase. Give each sender one
  // non-overlapping slot per attempt cycle. The cycle includes a guard longer
  // than attempt_interval_, so a receiver used in the previous cycle is
  // eligible again before the next low-ID sender starts. This prevents two
  // empty UAVs from repeatedly sending valid ACVRP results to the same busy
  // receiver, where all but one would be rejected by the official handshake.
  if (fp_->pair_opt_timer_stagger_ > 0.0 &&
      fp_->attempt_interval_ > 0.0) {
    const double slot_spacing = fp_->pair_opt_timer_stagger_;
    const double slot_width = 0.5 * slot_spacing;
    const double cycle =
        fp_->attempt_interval_ +
        slot_spacing * static_cast<double>(expl_manager_->ep_->drone_num_);
    const double slot_begin = slot_spacing * static_cast<double>(getId() - 1);
    double phase = std::fmod(tn, cycle);
    if (phase < 0.0) phase += cycle;
    if (phase < slot_begin || phase >= slot_begin + slot_width) return;
  }

  // Avoid frequent attempt
  if (tn - state1.recent_attempt_time_ < fp_->attempt_interval_) return;

  int select_id = -1;
  double max_interval = -1.0;
  int best_work_priority = -1;
  int skip_stale = 0, skip_attempt = 0, skip_interact = 0, skip_empty = 0;
  for (int i = 0; i < states.size(); ++i) {
    if (i + 1 <= getId()) continue;
    // Check if have communication recently
    // or the drone just experience another opt
    // or the drone is interacted with recently /* !urgent &&  */
    // or the candidate drone dominates enough grids
    if (tn - states[i].stamp_ > fp_->state_freshness_) { ++skip_stale; continue; }
    if (tn - states[i].recent_attempt_time_ < fp_->attempt_interval_) { ++skip_attempt; continue; }
    if (tn - states[i].recent_interact_time_ < fp_->pair_opt_interval_) { ++skip_interact; continue; }
    if (states[i].grid_ids_.size() + state1.grid_ids_.size() == 0) { ++skip_empty; continue; }

    double interval = tn - states[i].recent_interact_time_;
    // Upstream breaks equal interaction-age ties by ascending ID. With six
    // nodes launched together, those ties are exact and a high-ID empty UAV
    // can starve forever. Only before the global trigger, prefer an empty
    // receiver when the sender owns enough customers for ACVRP to split.
    const bool seed_empty_receiver =
        state_ == WAIT_TRIGGER && states[i].grid_ids_.empty() &&
        state1.grid_ids_.size() >= 2;
    const bool runtime_idle_pair =
        state_ != WAIT_TRIGGER && fp_->enable_idle_rebalance_ &&
        (state1.grid_ids_.empty() != states[i].grid_ids_.empty()) &&
        state1.grid_ids_.size() + states[i].grid_ids_.size() >= 2;
    const int work_priority =
        (seed_empty_receiver || runtime_idle_pair) ? 1 : 0;
    if (work_priority < best_work_priority) continue;
    if (work_priority == best_work_priority && interval <= max_interval) continue;
    select_id = i + 1;
    max_interval = interval;
    best_work_priority = work_priority;
  }
  if (select_id == -1) {
    ROS_WARN_THROTTLE(5.0,
        "RACER_METRIC pair_select_skip drone=%d stale=%d attempt=%d "
        "interact=%d empty=%d own_tasks=%zu state=%d",
        getId(), skip_stale, skip_attempt, skip_interact, skip_empty,
        state1.grid_ids_.size(), static_cast<int>(state_));
    return;
  }

  if (state_ == WAIT_TRIGGER && best_work_priority > 0) {
    ROS_WARN(
        "RACER_METRIC seed_idle_priority drone=%d partner=%d ego_tasks=%zu",
        getId(), select_id, state1.grid_ids_.size());
  } else if (state_ != WAIT_TRIGGER && best_work_priority > 0) {
    ROS_WARN(
        "RACER_METRIC runtime_idle_priority drone=%d partner=%d ego_tasks=%zu "
        "other_tasks=%zu",
        getId(), select_id, state1.grid_ids_.size(),
        states[select_id - 1].grid_ids_.size());
  }

  std::cout << "\nSelect: " << select_id << std::endl;
  ROS_WARN("Pair opt %d & %d", getId(), select_id);

  // Do pairwise optimization with selected drone, allocate the union of their domiance grids
  unordered_map<int, char> opt_ids_map;
  auto& state2 = states[select_id - 1];
  for (auto id : state1.grid_ids_) opt_ids_map[id] = 1;
  for (auto id : state2.grid_ids_) opt_ids_map[id] = 1;
  vector<int> opt_ids;
  for (auto pair : opt_ids_map) opt_ids.push_back(pair.first);
  // unordered_map iteration order is not stable.  A deterministic ACVRP node
  // order makes repeated pair optimizations comparable and reproducible.
  sort(opt_ids.begin(), opt_ids.end());

  std::cout << "Pair Opt id: ";
  for (auto id : opt_ids) std::cout << id << ", ";
  std::cout << "" << std::endl;

  // Find missed grids to reallocated them
  vector<int> actives, missed;
  expl_manager_->hgrid_->getActiveGrids(actives);
  findUnallocated(actives, missed);
  // Recover the nearest unallocated work first. Rediscovery and hierarchical
  // subdivision can expose dozens of HGrids in one map update; putting every
  // one into a single ACVRP creates an avoidable O(N^2) A* burst.
  sort(missed.begin(), missed.end(), [&](const int lhs, const int rhs) {
    const Eigen::Vector3d lhs_center = expl_manager_->hgrid_->getCenter(lhs);
    const Eigen::Vector3d rhs_center = expl_manager_->hgrid_->getCenter(rhs);
    const double lhs_distance =
        std::min((state1.pos_ - lhs_center).norm(),
            (state2.pos_ - lhs_center).norm());
    const double rhs_distance =
        std::min((state1.pos_ - rhs_center).norm(),
            (state2.pos_ - rhs_center).norm());
    if (std::abs(lhs_distance - rhs_distance) > 1e-6)
      return lhs_distance < rhs_distance;
    return lhs < rhs;
  });
  std::unordered_set<int> force_ego;
  std::unordered_set<int> force_other;
  vector<int> eligible_missed;
  int reachability_checked = 0;
  int reachability_deferred = 0;
  int batch_deferred = 0;
  const double allocation_now = ros::Time::now().toSec();
  for (const int grid_id : missed) {
    if (fp_->recovery_batch_size_ > 0 &&
        static_cast<int>(eligible_missed.size()) >=
            fp_->recovery_batch_size_) {
      ++batch_deferred;
      continue;
    }
    const bool ego_blacklisted =
        isGridBlacklisted(state1, grid_id, allocation_now);
    const bool other_blacklisted =
        isGridBlacklisted(state2, grid_id, allocation_now);
    int ego_reachability = GRID_REACHABILITY_UNKNOWN;
    int other_reachability = GRID_REACHABILITY_UNKNOWN;
    double ego_path_cost = std::numeric_limits<double>::infinity();
    double other_path_cost = std::numeric_limits<double>::infinity();

    if (state_ != WAIT_TRIGGER &&
        reachability_checked < fp_->reachability_filter_max_tasks_) {
      ++reachability_checked;
      if (!ego_blacklisted)
        ego_reachability = expl_manager_->checkGridReachability(
            state1.pos_, grid_id, ego_path_cost, 2);
      if (!other_blacklisted)
        other_reachability = expl_manager_->checkGridReachability(
            state2.pos_, grid_id, other_path_cost, 2);
    }

    const bool ego_ineligible =
        ego_blacklisted || ego_reachability == GRID_UNREACHABLE;
    const bool other_ineligible =
        other_blacklisted || other_reachability == GRID_UNREACHABLE;
    if (ego_ineligible && other_ineligible) {
      ++reachability_deferred;
      ROS_WARN(
          "RACER_METRIC allocation_task_deferred drone=%d partner=%d "
          "grid=%d ego_blacklist=%d other_blacklist=%d "
          "ego_reach=%d other_reach=%d",
          getId(), select_id, grid_id, static_cast<int>(ego_blacklisted),
          static_cast<int>(other_blacklisted), ego_reachability,
          other_reachability);
      continue;
    }
    if (ego_ineligible) force_other.insert(grid_id);
    if (other_ineligible) force_ego.insert(grid_id);
    eligible_missed.push_back(grid_id);
  }
  missed.swap(eligible_missed);
  std::cout << "Missed: ";
  for (auto id : missed) std::cout << id << ", ";
  std::cout << "" << std::endl;
  opt_ids.insert(opt_ids.end(), missed.begin(), missed.end());
  sort(opt_ids.begin(), opt_ids.end());
  opt_ids.erase(unique(opt_ids.begin(), opt_ids.end()), opt_ids.end());
  const bool has_missed_tasks = !missed.empty();
  if (reachability_checked > 0 || reachability_deferred > 0 ||
      batch_deferred > 0) {
    ROS_WARN(
        "RACER_METRIC reachability_filter drone=%d partner=%d "
        "checked=%d eligible=%zu deferred=%d batch_deferred=%d "
        "force_ego=%zu force_other=%zu",
        getId(), select_id, reachability_checked, missed.size(),
        reachability_deferred, batch_deferred, force_ego.size(),
        force_other.size());
  }

  // A one-customer ACVRP cannot give a task to both vehicles. During the
  // pre-trigger seed phase, sending that no-progress result to an empty
  // receiver merely locks it for attempt_interval and can repeatedly beat a
  // later sender that has enough work to split. Skip the transaction, not the
  // official allocator, and leave the empty receiver immediately available.
  if (state_ == WAIT_TRIGGER && state2.grid_ids_.empty() &&
      opt_ids.size() < 2) {
    state1.recent_attempt_time_ = tn;
    ROS_WARN(
        "RACER_METRIC pair_alloc_skipped drone=%d partner=%d tasks=%zu "
        "reason=seed_cannot_populate_empty",
        getId(), select_id, opt_ids.size());
    return;
  }

  // Do partition of the grid
  vector<Eigen::Vector3d> positions = { state1.pos_, state2.pos_ };
  vector<Eigen::Vector3d> velocities = { Eigen::Vector3d(0, 0, 0), Eigen::Vector3d(0, 0, 0) };
  constexpr double kMaxLocalCoordinate = 1.0e4;
  for (int i = 0; i < positions.size(); ++i) {
    if (!positions[i].allFinite() ||
        positions[i].cwiseAbs().maxCoeff() > kMaxLocalCoordinate) {
      ROS_ERROR(
          "Reject pair optimization: drone %d has invalid local position "
          "(%.3f, %.3f, %.3f)",
          i + 1, positions[i][0], positions[i][1], positions[i][2]);
      return;
    }
  }
  vector<int> first_ids1, second_ids1, first_ids2, second_ids2;
  if (state_ != WAIT_TRIGGER) {
    expl_manager_->hgrid_->getConsistentGrid(
        state1.grid_ids_, state1.grid_ids_, first_ids1, second_ids1);
    expl_manager_->hgrid_->getConsistentGrid(
        state2.grid_ids_, state2.grid_ids_, first_ids2, second_ids2);
  }

  auto t1 = ros::WallTime::now();

  // Per-vehicle first-grid hysteresis (first_grid_bonus): the exact grid each
  // vehicle is currently heading for keeps its discount inside the pair-wise
  // ACVRP too, so a near-tie cannot hand it to the partner every 5 s.
  if (state_ != WAIT_TRIGGER)
    expl_manager_->hgrid_->setPrevFirstGrids(
        { state1.grid_ids_.empty() ? -1 : state1.grid_ids_[0],
          state2.grid_ids_.empty() ? -1 : state2.grid_ids_[0] });
  else
    expl_manager_->hgrid_->setPrevFirstGrids({ -1, -1 });

  vector<int> ego_ids, other_ids;
  expl_manager_->allocateGrids(positions, velocities, { first_ids1, first_ids2 },
      { second_ids1, second_ids2 }, opt_ids, ego_ids, other_ids);

  double alloc_time = (ros::WallTime::now() - t1).toSec();

  auto move_task = [](int grid_id, vector<int>& source, vector<int>& destination) {
    const auto source_it = std::find(source.begin(), source.end(), grid_id);
    if (source_it == source.end()) return;
    source.erase(source_it);
    if (std::find(destination.begin(), destination.end(), grid_id) ==
        destination.end())
      destination.push_back(grid_id);
  };
  for (const int grid_id : force_ego)
    move_task(grid_id, other_ids, ego_ids);
  for (const int grid_id : force_other)
    move_task(grid_id, ego_ids, other_ids);

  // The blacklist is a per-vehicle constraint. ACVRP has no native
  // vehicle-customer exclusion, so enforce it transactionally on the
  // candidate before evaluating or publishing the result.
  vector<int> ego_copy = ego_ids;
  for (const int grid_id : ego_copy) {
    if (isGridBlacklisted(state1, grid_id, allocation_now) &&
        !isGridBlacklisted(state2, grid_id, allocation_now))
      move_task(grid_id, ego_ids, other_ids);
  }
  vector<int> other_copy = other_ids;
  for (const int grid_id : other_copy) {
    if (isGridBlacklisted(state2, grid_id, allocation_now) &&
        !isGridBlacklisted(state1, grid_id, allocation_now))
      move_task(grid_id, other_ids, ego_ids);
  }

  auto promote_reachable_first = [&](vector<int>& route,
                                     const Eigen::Vector3d& position,
                                     int drone_id) {
    if (route.empty()) return;
    const int limit = std::min(3, static_cast<int>(route.size()));
    int first_unknown = -1;
    for (int index = 0; index < limit; ++index) {
      double path_cost = std::numeric_limits<double>::infinity();
      const int reachability = expl_manager_->checkGridReachability(
          position, route[index], path_cost, 2);
      if (reachability == GRID_REACHABLE) {
        if (index > 0) {
          const int grid_id = route[index];
          route.erase(route.begin() + index);
          route.insert(route.begin(), grid_id);
          ROS_WARN(
              "RACER_METRIC reachable_task_promoted observer=%d "
              "vehicle=%d grid=%d from_index=%d path_m=%.3f",
              getId(), drone_id, grid_id, index, path_cost);
        }
        return;
      }
      if (reachability == GRID_REACHABILITY_UNKNOWN && first_unknown < 0)
        first_unknown = index;
    }
    if (first_unknown > 0) {
      const int grid_id = route[first_unknown];
      route.erase(route.begin() + first_unknown);
      route.insert(route.begin(), grid_id);
    }
  };
  if (state_ != WAIT_TRIGGER) {
    promote_reachable_first(ego_ids, state1.pos_, getId());
    promote_reachable_first(other_ids, state2.pos_, select_id);
  }

  // Upstream RACER transfers HGrid IDs, not mandatory viewpoints.  Keep the
  // older local viewpoint hint as an optional optimization only: failure to
  // produce it must not alter the ACVRP allocation.
  std::unordered_map<int, RemoteTaskEvidence> proposed_remote_evidence;
  if (expl_manager_->ep_->remote_task_lease_enabled_) {
    const std::unordered_set<int> receiver_before(
        state2.grid_ids_.begin(), state2.grid_ids_.end());
    Eigen::Vector3d route_cursor = state2.pos_;
    double cumulative_path_m = 0.0;
    const vector<int> receiver_candidate = other_ids;
    for (const int grid_id : receiver_candidate) {
      if (receiver_before.count(grid_id) > 0) {
        const Eigen::Vector3d center = expl_manager_->hgrid_->getCenter(grid_id);
        cumulative_path_m += (center - route_cursor).norm();
        route_cursor = center;
        continue;
      }

      Eigen::Vector3d viewpoint;
      double viewpoint_yaw = 0.0;
      double path_length = std::numeric_limits<double>::infinity();
      int visible_cells = 0;
      if (!expl_manager_->getGridTaskViewpoint(route_cursor, grid_id,
              viewpoint, viewpoint_yaw, visible_cells, path_length, 8)) {
        ROS_WARN(
            "RACER_METRIC remote_evidence_unavailable sender=%d "
            "receiver=%d grid=%d action=id_only_transfer",
            getId(), select_id, grid_id);
        continue;
      }

      cumulative_path_m += std::max(0.0, path_length);
      route_cursor = viewpoint;
      const double speed =
          std::max(0.1, expl_manager_->ep_->remote_task_assumed_speed_);
      const double lease_duration = std::min(
          expl_manager_->ep_->remote_task_max_lease_,
          std::max(expl_manager_->ep_->remote_task_min_lease_,
              5.0 + expl_manager_->ep_->remote_task_lease_factor_ *
                        cumulative_path_m / speed));
      RemoteTaskEvidence evidence;
      evidence.viewpoint_ = viewpoint;
      evidence.yaw_ = viewpoint_yaw;
      evidence.lease_until_ = allocation_now + lease_duration;
      evidence.version_ = 0;
      evidence.source_drone_id_ = getId();
      evidence.expected_visible_cells_ = visible_cells;
      evidence.last_validation_frontier_seq_ = 0;
      evidence.validation_misses_ = 0;
      evidence.restore_count_ = 0;
      proposed_remote_evidence[grid_id] = evidence;
    }
  }

  // LKH can terminate without a valid tour (for example after an internal
  // assertion) while the ROS service call itself still returns.  Never treat
  // an empty or partial candidate as a zero-cost improvement and erase the
  // current ownership.  The candidate must cover every requested grid exactly
  // once before any objective comparison is meaningful.
  vector<int> candidate_ids = ego_ids;
  candidate_ids.insert(candidate_ids.end(), other_ids.begin(), other_ids.end());
  sort(candidate_ids.begin(), candidate_ids.end());
  candidate_ids.erase(unique(candidate_ids.begin(), candidate_ids.end()), candidate_ids.end());
  if (candidate_ids != opt_ids ||
      ego_ids.size() + other_ids.size() != candidate_ids.size()) {
    ROS_ERROR(
        "Reject invalid reallocation: candidate covers %zu/%zu unique grids "
        "(raw assignments: %zu)",
        candidate_ids.size(), opt_ids.size(), ego_ids.size() + other_ids.size());
    // An invalid LKH result is still an expensive optimization attempt.
    // Pace the retry instead of invoking the same ACVRP at 20 Hz.
    state1.recent_attempt_time_ = tn;
    return;
  }

  std::cout << "Ego1  : ";
  for (auto id : state1.grid_ids_) std::cout << id << ", ";
  std::cout << "\nOther1: ";
  for (auto id : state2.grid_ids_) std::cout << id << ", ";
  std::cout << "\nEgo2  : ";
  for (auto id : ego_ids) std::cout << id << ", ";
  std::cout << "\nOther2: ";
  for (auto id : other_ids) std::cout << id << ", ";
  std::cout << "" << std::endl;

  // Check results. When unallocated tasks are recovered, compare the ACVRP
  // candidate with a complete feasible baseline that greedily inserts the
  // same tasks into the two old routes. Comparing against the incomplete old
  // routes made every recovery look more expensive and previously required
  // unconditional acceptance, including pathological reallocations.
  vector<int> baseline_ids1 = state1.grid_ids_;
  vector<int> baseline_ids2 = state2.grid_ids_;
  auto insertion_delta = [&](const vector<int>& route, int insert_index,
                             const Eigen::Vector3d& start,
                             int grid_id) {
    const Eigen::Vector3d point = expl_manager_->hgrid_->getCenter(grid_id);
    const Eigen::Vector3d previous =
        insert_index == 0
            ? start
            : expl_manager_->hgrid_->getCenter(route[insert_index - 1]);
    double delta = (previous - point).norm();
    if (insert_index < static_cast<int>(route.size())) {
      const Eigen::Vector3d next =
          expl_manager_->hgrid_->getCenter(route[insert_index]);
      delta += (point - next).norm() - (previous - next).norm();
    }
    return delta;
  };
  for (const int grid_id : missed) {
    int best_vehicle = -1;
    int best_index = -1;
    double best_delta = std::numeric_limits<double>::infinity();
    for (int vehicle = 0; vehicle < 2; ++vehicle) {
      if (vehicle == 0 && force_other.count(grid_id) > 0) continue;
      if (vehicle == 1 && force_ego.count(grid_id) > 0) continue;
      auto& route = vehicle == 0 ? baseline_ids1 : baseline_ids2;
      const auto& start = vehicle == 0 ? state1.pos_ : state2.pos_;
      for (int index = 0; index <= static_cast<int>(route.size()); ++index) {
        const double delta =
            insertion_delta(route, index, start, grid_id);
        if (delta < best_delta) {
          best_delta = delta;
          best_vehicle = vehicle;
          best_index = index;
        }
      }
    }
    if (best_vehicle == 0)
      baseline_ids1.insert(baseline_ids1.begin() + best_index, grid_id);
    else if (best_vehicle == 1)
      baseline_ids2.insert(baseline_ids2.begin() + best_index, grid_id);
  }

  const bool exact_pair_objective =
      expl_manager_->ep_->exact_pair_objective_;
  double prev_app1 = expl_manager_->computeGridPathCost(0, state1.pos_, baseline_ids1, first_ids1,
      { first_ids1, first_ids2 }, { second_ids1, second_ids2 }, exact_pair_objective);
  double prev_app2 = expl_manager_->computeGridPathCost(1, state2.pos_, baseline_ids2, first_ids2,
      { first_ids1, first_ids2 }, { second_ids1, second_ids2 }, exact_pair_objective);
  std::cout << "prev cost: " << prev_app1 << ", " << prev_app2 << ", " << prev_app1 + prev_app2
            << std::endl;
  double cur_app1 = expl_manager_->computeGridPathCost(0, state1.pos_, ego_ids, first_ids1,
      { first_ids1, first_ids2 }, { second_ids1, second_ids2 }, exact_pair_objective);
  double cur_app2 = expl_manager_->computeGridPathCost(1, state2.pos_, other_ids, first_ids2,
      { first_ids1, first_ids2 }, { second_ids1, second_ids2 }, exact_pair_objective);
  std::cout << "cur cost : " << cur_app1 << ", " << cur_app2 << ", " << cur_app1 + cur_app2
            << std::endl;
  const double prev_sum = prev_app1 + prev_app2;
  const double cur_sum = cur_app1 + cur_app2;
  const double prev_makespan = max(prev_app1, prev_app2);
  const double cur_makespan = max(cur_app1, cur_app2);
  const bool official_sum_improvement = cur_sum <= prev_sum + 0.1;
  // Upstream RACER deliberately accepts seed allocations while every FSM is
  // still in WAIT_TRIGGER: at that point opt_ids also contains globally active
  // grids that are not in either route yet, so comparing the new objective
  // against only the two old routes is not meaningful.  Retain that official
  // initialization behavior; apply the transactional objective guard only
  // after exploration has started.
  const bool seed_allocation = state_ == WAIT_TRIGGER;
  const bool recovery_missed_allocation =
      !seed_allocation && has_missed_tasks && official_sum_improvement;
  const bool had_one_idle_vehicle =
      state1.grid_ids_.empty() != state2.grid_ids_.empty();
  const bool candidate_keeps_both_working =
      !ego_ids.empty() && !other_ids.empty();
  // Keep the official LKH ACVRP candidate and sum-cost rule. The only
  // work-conserving extension is for a genuinely idle pair: accept a bounded
  // sum-cost increase when it materially reduces the pair makespan.
  const bool idle_makespan_improvement =
      fp_->enable_idle_rebalance_ && !seed_allocation &&
      had_one_idle_vehicle && candidate_keeps_both_working &&
      cur_makespan <=
          prev_makespan *
              (1.0 - fp_->idle_rebalance_min_makespan_gain_) &&
      cur_sum <=
          max(prev_sum + 0.1,
              prev_sum * fp_->idle_rebalance_max_sum_ratio_);
  const bool accept_reallocation =
      seed_allocation || official_sum_improvement ||
      idle_makespan_improvement;
  const char* acceptance_reason =
      seed_allocation
          ? "seed"
          : (recovery_missed_allocation
                    ? "recovery_missed"
                    : (official_sum_improvement
                              ? "official_sum"
                              : (idle_makespan_improvement
                                        ? "idle_makespan"
                                        : "rejected")));
  // ---- churn hysteresis (see parameter comment in the constructor) ----
  bool accept_reallocation_final = accept_reallocation;
  const char* acceptance_reason_final = acceptance_reason;
  // Never block a transfer to a vehicle that currently owns nothing: that is
  // the seed split or an idle-rebalance, not churn.
  const bool partner_or_self_idle =
      state1.grid_ids_.empty() || state2.grid_ids_.empty();
  if (!seed_allocation && accept_reallocation && !partner_or_self_idle &&
      fp_->pair_opt_min_keep_fraction_ > 0.0) {
    auto keep_fraction = [](const vector<int>& before, const vector<int>& after) {
      if (before.empty()) return 1.0;
      int kept = 0;
      for (const int id : before)
        if (std::find(after.begin(), after.end(), id) != after.end()) ++kept;
      return double(kept) / double(before.size());
    };
    const double keep1 = keep_fraction(state1.grid_ids_, ego_ids);
    const double keep2 = keep_fraction(state2.grid_ids_, other_ids);
    const bool large_reshuffle =
        std::min(keep1, keep2) < fp_->pair_opt_min_keep_fraction_;
    const bool strong_gain =
        cur_sum <= prev_sum * (1.0 - fp_->pair_opt_reshuffle_min_gain_);
    if (large_reshuffle && !strong_gain) {
      if (has_missed_tasks) {
        ego_ids = baseline_ids1;
        other_ids = baseline_ids2;
        acceptance_reason_final = "baseline_recovery";
      } else {
        accept_reallocation_final = false;
        acceptance_reason_final = "reshuffle_rejected";
      }
      ROS_WARN(
          "RACER_METRIC pair_hysteresis drone=%d partner=%d keep1=%.2f keep2=%.2f "
          "prev_sum=%.3f cur_sum=%.3f missed=%d outcome=%s",
          getId(), select_id, keep1, keep2, prev_sum, cur_sum,
          static_cast<int>(has_missed_tasks), acceptance_reason_final);
    }
  }
  const bool path1_consistent =
      state1.grid_ids_.empty() || ego_ids.empty() ||
      expl_manager_->hgrid_->isConsistent(state1.grid_ids_[0], ego_ids[0]);
  const bool path2_consistent =
      state2.grid_ids_.empty() || other_ids.empty() ||
      expl_manager_->hgrid_->isConsistent(state2.grid_ids_[0], other_ids[0]);

  // RACER's published pairwise sum objective remains the normal acceptance
  // rule; the bounded idle exception above prevents unfinished work from
  // remaining concentrated on one vehicle.
  ROS_WARN(
      "Pair allocation objective: sum %.3f -> %.3f, makespan %.3f -> %.3f",
      prev_sum, cur_sum, prev_makespan, cur_makespan);
  if (!path1_consistent || !path2_consistent) {
    // Upstream RACER reports this condition but still applies the ACVRP
    // result. The current safe B-spline remains active until replanning
    // commits a replacement, so rejecting here only starves an idle UAV.
    ROS_WARN(
        "Pair allocation changes an in-progress first HGrid "
        "(path1_consistent=%d path2_consistent=%d)",
        path1_consistent, path2_consistent);
  }
  if (!accept_reallocation_final) {
    ROS_WARN(
        "RACER_METRIC pair_alloc drone=%d partner=%d tasks=%zu "
        "ego_tasks=%zu other_tasks=%zu seed=%d "
        "wall_ms=%.3f prev_sum=%.3f cur_sum=%.3f "
        "prev_makespan=%.3f cur_makespan=%.3f accepted=0 reason=%s",
        getId(), select_id, opt_ids.size(), ego_ids.size(), other_ids.size(),
        seed_allocation, alloc_time * 1000.0,
        prev_sum, cur_sum, prev_makespan, cur_makespan,
        acceptance_reason_final);
    ROS_WARN("Reject reallocation: official pair-route sum cost would increase");
    // A rejected candidate is still an optimization attempt.  Recording its
    // timestamp prevents the 20 Hz timer from immediately solving the same
    // ACVRP again and consuming a CPU core.
    state1.recent_attempt_time_ = tn;
    // Rejected attempts do not update this field upstream, so deterministic
    // timer phases can select the same receiver forever and starve later
    // peers. Rotate the local peer age without changing any allocation.
    state2.recent_interact_time_ = tn;
    return;
  }

  // Update ego and other dominace grids
  auto last_ids2 = state2.grid_ids_;

  // Send the result to selected drone and wait for confirmation
  exploration_manager::PairOpt opt;
  opt.from_drone_id = getId();
  opt.to_drone_id = select_id;
  // opt.msg_type = 1;
  opt.stamp = tn;
  opt.expected_ego_epoch = state1.assignment_epoch_;
  opt.expected_other_epoch = state2.assignment_epoch_;
  opt.assignment_epoch = nextAssignmentEpoch(state1, state2);
  for (auto id : ego_ids) opt.ego_ids.push_back(id);
  for (auto id : other_ids) opt.other_ids.push_back(id);
  for (const int grid_id : other_ids) {
    const auto evidence = proposed_remote_evidence.find(grid_id);
    if (evidence == proposed_remote_evidence.end()) continue;
    geometry_msgs::Point point;
    point.x = evidence->second.viewpoint_[0];
    point.y = evidence->second.viewpoint_[1];
    point.z = evidence->second.viewpoint_[2];
    opt.evidence_grid_ids.push_back(grid_id);
    opt.evidence_viewpoints.push_back(point);
    opt.evidence_yaws.push_back(evidence->second.yaw_);
    opt.evidence_visible_cells.push_back(
        evidence->second.expected_visible_cells_);
    opt.evidence_lease_until.push_back(evidence->second.lease_until_);
    opt.evidence_versions.push_back(opt.assignment_epoch);
  }
  if (!opt.evidence_grid_ids.empty()) {
    ROS_WARN(
        "RACER_METRIC remote_evidence_sent sender=%d receiver=%d count=%zu "
        "assignment_epoch=%llu",
        getId(), select_id, opt.evidence_grid_ids.size(),
        static_cast<unsigned long long>(opt.assignment_epoch));
  }

  for (int i = 0; i < fp_->repeat_send_num_; ++i) opt_pub_.publish(opt);

  ROS_WARN("Drone %d send opt request to %d, pair opt t: %lf, allocate t: %lf", getId(), select_id,
      ros::Time::now().toSec() - tn, alloc_time);
  ROS_WARN(
      "RACER_METRIC pair_alloc drone=%d partner=%d tasks=%zu "
      "ego_tasks=%zu other_tasks=%zu seed=%d "
      "wall_ms=%.3f prev_sum=%.3f cur_sum=%.3f "
      "prev_makespan=%.3f cur_makespan=%.3f accepted=1 reason=%s",
      getId(), select_id, opt_ids.size(), ego_ids.size(), other_ids.size(),
      seed_allocation, alloc_time * 1000.0,
      prev_sum, cur_sum, prev_makespan, cur_makespan,
      acceptance_reason_final);

  // Reserve the result and wait...
  auto ed = expl_manager_->ed_;
  ed->ego_ids_ = ego_ids;
  ed->other_ids_ = other_ids;
  ed->pair_opt_stamp_ = opt.stamp;
  ed->pair_expected_ego_epoch_ = opt.expected_ego_epoch;
  ed->pair_expected_other_epoch_ = opt.expected_other_epoch;
  ed->pair_assignment_epoch_ = opt.assignment_epoch;
  ed->wait_response_ = true;
  state1.recent_attempt_time_ = tn;
}

void FastExplorationFSM::findUnallocated(const vector<int>& actives, vector<int>& missed) {
  // Create map of all active
  unordered_map<int, char> active_map;
  for (auto ativ : actives) {
    active_map[ativ] = 1;
  }

  // Remove allocated ones
  for (auto state : expl_manager_->ed_->swarm_state_) {
    for (auto id : state.grid_ids_) {
      if (active_map.find(id) != active_map.end()) {
        active_map.erase(id);
      } else {
        // ROS_ERROR("Inactive grid %d is allocated.", id);
      }
    }
  }

  missed.clear();
  for (auto p : active_map) {
    missed.push_back(p.first);
  }
}

void FastExplorationFSM::optMsgCallback(const exploration_manager::PairOptConstPtr& msg) {
  if (msg->from_drone_id == getId() || msg->to_drone_id != getId()) return;

  // Check stamp to avoid unordered/repeated msg
  if (msg->stamp <= expl_manager_->ed_->pair_opt_stamps_[msg->from_drone_id - 1] + 1e-4) return;
  expl_manager_->ed_->pair_opt_stamps_[msg->from_drone_id - 1] = msg->stamp;

  auto& state1 = expl_manager_->ed_->swarm_state_[msg->from_drone_id - 1];
  auto& state2 = expl_manager_->ed_->swarm_state_[getId() - 1];

  // auto tn = ros::Time::now().toSec();
  exploration_manager::PairOptResponse response;
  response.from_drone_id = msg->to_drone_id;
  response.to_drone_id = msg->from_drone_id;
  response.stamp = msg->stamp;  // reply with the same stamp for verificaiton
  response.expected_ego_epoch = msg->expected_ego_epoch;
  response.expected_other_epoch = msg->expected_other_epoch;
  response.assignment_epoch = msg->assignment_epoch;

  const bool stale_ownership =
      state1.assignment_epoch_ != msg->expected_ego_epoch ||
      state2.assignment_epoch_ != msg->expected_other_epoch ||
      msg->assignment_epoch <=
          std::max(msg->expected_ego_epoch, msg->expected_other_epoch);
  vector<int> candidate_ids(msg->ego_ids.begin(), msg->ego_ids.end());
  candidate_ids.insert(
      candidate_ids.end(), msg->other_ids.begin(), msg->other_ids.end());
  const size_t raw_candidate_size = candidate_ids.size();
  sort(candidate_ids.begin(), candidate_ids.end());
  candidate_ids.erase(unique(candidate_ids.begin(), candidate_ids.end()),
      candidate_ids.end());
  const bool duplicate_candidate =
      candidate_ids.size() != raw_candidate_size;
  const double now = ros::Time::now().toSec();
  const size_t evidence_count = msg->evidence_grid_ids.size();
  bool invalid_remote_evidence =
      msg->evidence_viewpoints.size() != evidence_count ||
      msg->evidence_yaws.size() != evidence_count ||
      msg->evidence_visible_cells.size() != evidence_count ||
      msg->evidence_lease_until.size() != evidence_count ||
      msg->evidence_versions.size() != evidence_count;
  std::unordered_map<int, size_t> evidence_indices;
  const std::unordered_set<int> proposed_other(
      msg->other_ids.begin(), msg->other_ids.end());
  if (!invalid_remote_evidence) {
    for (size_t index = 0; index < evidence_count; ++index) {
      const int grid_id = msg->evidence_grid_ids[index];
      const auto& point = msg->evidence_viewpoints[index];
      if (!evidence_indices.emplace(grid_id, index).second ||
          proposed_other.count(grid_id) == 0 ||
          !std::isfinite(point.x) || !std::isfinite(point.y) ||
          !std::isfinite(point.z) ||
          !std::isfinite(msg->evidence_yaws[index]) ||
          msg->evidence_lease_until[index] <= now ||
          msg->evidence_versions[index] != msg->assignment_epoch) {
        invalid_remote_evidence = true;
        break;
      }
    }
  }
  bool violates_blacklist = false;
  for (const int grid_id : msg->ego_ids)
    violates_blacklist =
        violates_blacklist || isGridBlacklisted(state1, grid_id, now);
  for (const int grid_id : msg->other_ids)
    violates_blacklist =
        violates_blacklist || isGridBlacklisted(state2, grid_id, now);

  if (stale_ownership || duplicate_candidate || violates_blacklist ||
      invalid_remote_evidence) {
    ROS_WARN(
        "RACER_METRIC pair_transaction_rejected receiver=%d sender=%d "
        "stale=%d duplicate=%d blacklist=%d evidence=%d "
        "expected=[%llu,%llu] "
        "actual=[%llu,%llu] proposed=%llu",
        getId(), msg->from_drone_id, static_cast<int>(stale_ownership),
        static_cast<int>(duplicate_candidate),
        static_cast<int>(violates_blacklist),
        static_cast<int>(invalid_remote_evidence),
        static_cast<unsigned long long>(msg->expected_ego_epoch),
        static_cast<unsigned long long>(msg->expected_other_epoch),
        static_cast<unsigned long long>(state1.assignment_epoch_),
        static_cast<unsigned long long>(state2.assignment_epoch_),
        static_cast<unsigned long long>(msg->assignment_epoch));
    response.status = 2;
  } else if (msg->stamp - state2.recent_attempt_time_ < fp_->attempt_interval_) {
    // Just made another pair opt attempt, should reject this attempt to avoid frequent changes
    ROS_WARN("Reject frequent attempt");
    response.status = 2;
  } else {
    // No opt attempt recently, and the grid info between drones are consistent, the pair opt
    // request can be accepted
    response.status = 1;

    // Update from the opt result
    setRouteOwnership(
        state1, vector<int>(msg->ego_ids.begin(), msg->ego_ids.end()),
        msg->assignment_epoch);
    setRouteOwnership(
        state2, vector<int>(msg->other_ids.begin(), msg->other_ids.end()),
        msg->assignment_epoch);

    for (const auto& item : evidence_indices) {
      const int grid_id = item.first;
      const size_t index = item.second;
      RemoteTaskEvidence evidence;
      evidence.viewpoint_ = Eigen::Vector3d(
          msg->evidence_viewpoints[index].x,
          msg->evidence_viewpoints[index].y,
          msg->evidence_viewpoints[index].z);
      evidence.yaw_ = msg->evidence_yaws[index];
      evidence.lease_until_ = msg->evidence_lease_until[index];
      evidence.version_ = msg->evidence_versions[index];
      evidence.source_drone_id_ = msg->from_drone_id;
      evidence.expected_visible_cells_ =
          msg->evidence_visible_cells[index];
      evidence.last_validation_frontier_seq_ =
          expl_manager_->ed_->frontier_update_seq_;
      evidence.validation_misses_ = 0;
      evidence.restore_count_ = 0;
      state2.remote_tasks_[grid_id] = evidence;
      ROS_WARN(
          "RACER_METRIC remote_task_received drone=%d grid=%d source=%d "
          "viewpoint=[%.3f,%.3f,%.3f] expected_visible=%d "
          "lease_s=%.3f version=%llu",
          getId(), grid_id, msg->from_drone_id, evidence.viewpoint_[0],
          evidence.viewpoint_[1], evidence.viewpoint_[2],
          evidence.expected_visible_cells_, evidence.lease_until_ - now,
          static_cast<unsigned long long>(evidence.version_));
    }

    state1.recent_interact_time_ = msg->stamp;
    state2.recent_attempt_time_ = ros::Time::now().toSec();
    expl_manager_->ed_->reallocated_ = true;

    if (state_ == IDLE && !state2.grid_ids_.empty()) {
      transitState(PLAN_TRAJ, "optMsgCallback");
      ROS_WARN("Restart after opt!");
    }

    // if (!check_consistency(tmp1, tmp2)) {
    //   response.status = 2;
    //   ROS_WARN("Inconsistent grid info, reject pair opt");
    // } else {
    // }
  }
  for (int i = 0; i < fp_->repeat_send_num_; ++i) opt_res_pub_.publish(response);
}

void FastExplorationFSM::optResMsgCallback(
    const exploration_manager::PairOptResponseConstPtr& msg) {
  if (msg->from_drone_id == getId() || msg->to_drone_id != getId()) return;

  // Check stamp to avoid unordered/repeated msg
  if (msg->stamp <= expl_manager_->ed_->pair_opt_res_stamps_[msg->from_drone_id - 1] + 1e-4) return;
  expl_manager_->ed_->pair_opt_res_stamps_[msg->from_drone_id - 1] = msg->stamp;

  auto ed = expl_manager_->ed_;
  // Verify the consistency of pair opt via time stamp
  if (!ed->wait_response_ || fabs(ed->pair_opt_stamp_ - msg->stamp) > 1e-5) return;

  ed->wait_response_ = false;
  ROS_WARN("get response %d", int(msg->status));

  if (msg->status != 1) return;  // Receive 1 for valid opt

  auto& state1 = ed->swarm_state_[getId() - 1];
  auto& state2 = ed->swarm_state_[msg->from_drone_id - 1];
  const bool transaction_matches =
      msg->expected_ego_epoch == ed->pair_expected_ego_epoch_ &&
      msg->expected_other_epoch == ed->pair_expected_other_epoch_ &&
      msg->assignment_epoch == ed->pair_assignment_epoch_ &&
      state1.assignment_epoch_ == ed->pair_expected_ego_epoch_ &&
      (state2.assignment_epoch_ == ed->pair_expected_other_epoch_ ||
          state2.assignment_epoch_ == ed->pair_assignment_epoch_);
  if (!transaction_matches) {
    ROS_WARN(
        "RACER_METRIC pair_response_stale drone=%d partner=%d "
        "expected=[%llu,%llu,%llu] actual=[%llu,%llu,%llu]",
        getId(), msg->from_drone_id,
        static_cast<unsigned long long>(ed->pair_expected_ego_epoch_),
        static_cast<unsigned long long>(ed->pair_expected_other_epoch_),
        static_cast<unsigned long long>(ed->pair_assignment_epoch_),
        static_cast<unsigned long long>(state1.assignment_epoch_),
        static_cast<unsigned long long>(state2.assignment_epoch_),
        static_cast<unsigned long long>(msg->assignment_epoch));
    return;
  }
  setRouteOwnership(state1, ed->ego_ids_, ed->pair_assignment_epoch_);
  setRouteOwnership(state2, ed->other_ids_, ed->pair_assignment_epoch_);
  state2.recent_interact_time_ = ros::Time::now().toSec();
  ed->reallocated_ = true;
  reconcileDuplicateOwnership();

  if (state_ == IDLE && !state1.grid_ids_.empty()) {
    transitState(PLAN_TRAJ, "optResMsgCallback");
    ROS_WARN("Restart after opt!");
  }
}

void FastExplorationFSM::swarmTrajCallback(const bspline::BsplineConstPtr& msg) {
  // Get newest trajs from other drones, for inter-drone collision avoidance
  auto& sdat = planner_manager_->swarm_traj_data_;

  // Ignore self trajectory
  if (msg->drone_id == sdat.drone_id_) return;

  // Ignore outdated trajectory
  if (sdat.receive_flags_[msg->drone_id - 1] == true &&
      msg->start_time.toSec() <= sdat.swarm_trajs_[msg->drone_id - 1].start_time_ + 1e-3)
    return;

  // Convert the msg to B-spline
  Eigen::MatrixXd pos_pts(msg->pos_pts.size(), 3);
  Eigen::VectorXd knots(msg->knots.size());
  for (int i = 0; i < msg->knots.size(); ++i) knots(i) = msg->knots[i];

  for (int i = 0; i < msg->pos_pts.size(); ++i) {
    pos_pts(i, 0) = msg->pos_pts[i].x;
    pos_pts(i, 1) = msg->pos_pts[i].y;
    pos_pts(i, 2) = msg->pos_pts[i].z;
  }

  // // Transform of drone's basecoor, optional step (skip if use swarm_pilot)
  // Eigen::Vector4d tf;
  // planner_manager_->edt_environment_->sdf_map_->getBaseCoor(msg->drone_id, tf);
  // double yaw = tf[3];
  // Eigen::Matrix3d rot;
  // rot << cos(yaw), -sin(yaw), 0, sin(yaw), cos(yaw), 0, 0, 0, 1;
  // Eigen::Vector3d trans = tf.head<3>();
  // for (int i = 0; i < pos_pts.rows(); ++i) {
  //   Eigen::Vector3d tmp = pos_pts.row(i);
  //   tmp = rot * tmp + trans;
  //   pos_pts.row(i) = tmp;
  // }

  sdat.swarm_trajs_[msg->drone_id - 1].setUniformBspline(pos_pts, msg->order, 0.1);
  sdat.swarm_trajs_[msg->drone_id - 1].setKnot(knots);
  sdat.swarm_trajs_[msg->drone_id - 1].start_time_ = msg->start_time.toSec();
  sdat.receive_flags_[msg->drone_id - 1] = true;

  if (state_ == EXEC_TRAJ) {
    // Check collision with received trajectory
    if (!planner_manager_->checkSwarmCollision(msg->drone_id)) {
      ROS_ERROR("Drone %d collide with drone %d.", sdat.drone_id_, msg->drone_id);
      fd_->avoid_collision_ = true;
      fd_->emergency_replan_ = true;
      replan_pub_.publish(std_msgs::Empty());
      transitState(PLAN_TRAJ, "swarmTrajCallback");
    }
  }
}

void FastExplorationFSM::swarmTrajTimerCallback(const ros::TimerEvent& e) {
  // Broadcast newest traj of this drone to others
  if (state_ == EXEC_TRAJ) {
    swarm_traj_pub_.publish(fd_->newest_traj_);

  } else if (state_ == WAIT_TRIGGER) {
    // Publish a virtual traj at current pose, to avoid collision
    bspline::Bspline bspline;
    bspline.order = planner_manager_->pp_.bspline_degree_;
    bspline.start_time = ros::Time::now();
    bspline.traj_id = planner_manager_->local_data_.traj_id_;

    Eigen::MatrixXd pos_pts(4, 3);
    for (int i = 0; i < 4; ++i) pos_pts.row(i) = fd_->odom_pos_.transpose();

    for (int i = 0; i < pos_pts.rows(); ++i) {
      geometry_msgs::Point pt;
      pt.x = pos_pts(i, 0);
      pt.y = pos_pts(i, 1);
      pt.z = pos_pts(i, 2);
      bspline.pos_pts.push_back(pt);
    }

    NonUniformBspline tmp(pos_pts, planner_manager_->pp_.bspline_degree_, 1.0);
    Eigen::VectorXd knots = tmp.getKnot();
    for (int i = 0; i < knots.rows(); ++i) {
      bspline.knots.push_back(knots(i));
    }
    bspline.drone_id = expl_manager_->ep_->drone_id_;
    swarm_traj_pub_.publish(bspline);
  }
}

}  // namespace fast_planner
