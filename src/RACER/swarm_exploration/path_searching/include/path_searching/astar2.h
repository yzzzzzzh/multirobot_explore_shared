#ifndef _ASTAR2_H
#define _ASTAR2_H

#include <Eigen/Eigen>
#include <iostream>
#include <map>
#include <ros/console.h>
#include <ros/ros.h>
#include <string>
#include <unordered_map>
#include "plan_env/edt_environment.h"
#include <boost/functional/hash.hpp>
#include <queue>
#include <path_searching/matrix_hash.h>
namespace fast_planner {
class Node {
public:
  Eigen::Vector3i index;
  Eigen::Vector3d position;
  double g_score, f_score;
  // True only during the goal-independent nearest-exit phase for a search
  // that started in an inflated voxel or in the vehicle's current raw
  // occupied voxel.  The latter can occur after delayed multi-map fusion;
  // only the start voxel itself may be exited, never re-entered.
  bool escaping_inflation;
  Node* parent;

  /* -------------------- */
  Node() {
    parent = NULL;
    escaping_inflation = false;
  }
  ~Node(){};
};
typedef Node* NodePtr;

class NodeComparator0 {
public:
  bool operator()(NodePtr node1, NodePtr node2) {
    return node1->f_score > node2->f_score;
  }
};

class Astar {
public:
  Astar();
  ~Astar();
  enum { REACH_END = 1, NO_PATH = 2, TIMEOUT = 3 };

  void init(ros::NodeHandle& nh, const EDTEnvironment::Ptr& env);
  void reset();
  int search(
      const Eigen::Vector3d& start_pt, const Eigen::Vector3d& end_pt, bool optimistic = true);
  void setResolution(const double& res);
  double getResolution() const {
    return resolution_;
  }
  static double pathLength(const vector<Eigen::Vector3d>& path);

  std::vector<Eigen::Vector3d> getPath();
  std::vector<Eigen::Vector3d> getVisited();
  double getEarlyTerminateCost();
  bool lastSearchStartedPlannerClear() const {
    return last_start_planner_clear_;
  }
  int lastStartOccupancy() const {
    return last_start_occupancy_;
  }
  bool lastStartInflated() const {
    return last_start_inflated_;
  }

  double lambda_heu_;
  double max_search_time_;

private:
  void backtrack(const NodePtr& end_node, const Eigen::Vector3d& end);
  void posToIndex(const Eigen::Vector3d& pt, Eigen::Vector3i& idx);
  double getDiagHeu(const Eigen::Vector3d& x1, const Eigen::Vector3d& x2);
  double getManhHeu(const Eigen::Vector3d& x1, const Eigen::Vector3d& x2);
  double getEuclHeu(const Eigen::Vector3d& x1, const Eigen::Vector3d& x2);

  // main data structure
  vector<NodePtr> path_node_pool_;
  int use_node_num_, iter_num_;
  std::priority_queue<NodePtr, std::vector<NodePtr>, NodeComparator0> open_set_;
  std::unordered_map<Eigen::Vector3i, NodePtr, matrix_hash<Eigen::Vector3i>> open_set_map_;
  std::unordered_map<Eigen::Vector3i, int, matrix_hash<Eigen::Vector3i>> close_set_map_;
  std::vector<Eigen::Vector3d> path_nodes_;
  double early_terminate_cost_;
  // Snapshot the exact map state used by search().  The occupancy map is
  // updated asynchronously, so recomputing this state after search returns
  // can turn an unsafe-start failure into a false destination failure.
  bool last_start_planner_clear_{false};
  int last_start_occupancy_{-1};
  bool last_start_inflated_{false};

  EDTEnvironment::Ptr edt_env_;

  // parameter
  double margin_;
  int allocate_num_;
  double tie_breaker_;
  double resolution_, inv_resolution_;
  Eigen::Vector3d map_size_3d_, origin_;
};

}  // namespace fast_planner

#endif
