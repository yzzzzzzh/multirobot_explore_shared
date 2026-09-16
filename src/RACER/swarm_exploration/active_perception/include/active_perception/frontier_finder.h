#ifndef _FRONTIER_FINDER_H_
#define _FRONTIER_FINDER_H_

#include <ros/ros.h>
#include <Eigen/Eigen>
#include <memory>
#include <vector>
#include <list>
#include <utility>

using Eigen::Vector3d;
using std::list;
using std::pair;
using std::shared_ptr;
using std::unique_ptr;
using std::vector;

class RayCaster;

namespace fast_planner {
class EDTEnvironment;
class PerceptionUtils;

// Viewpoint to cover a frontier cluster
struct Viewpoint {
  // Position and heading
  Vector3d pos_;
  double yaw_;
  // Number of visible known-free frontier cells.  This remains a hard
  // geometric validity check; it is not an information-gain estimate.
  int visib_num_{0};
  // Unique currently-unknown map voxels reached by rays that continue from
  // this viewpoint through visible frontier cells.  The addresses make gain
  // aggregation across frontiers exact (no double counting) for one map
  // update.
  int unknown_gain_{0};
  vector<int> unknown_voxel_addrs_;
};

// A frontier cluster, the viewpoints to cover it
struct Frontier {
  // Complete voxels belonging to the cluster
  vector<Vector3d> cells_;
  // down-sampled voxels filtered by voxel grid filter
  vector<Vector3d> filtered_cells_;
  // Average position of all voxels
  Vector3d average_;
  // Idx of cluster
  int id_;
  // Viewpoints that can cover the cluster
  vector<Viewpoint> viewpoints_;
  // Bounding box of cluster, center & 1/2 side length
  Vector3d box_min_, box_max_;
  // Path and cost from this cluster to other clusters
  list<vector<Vector3d>> paths_;
  list<double> costs_;
};

class FrontierFinder {
public:
  FrontierFinder(const shared_ptr<EDTEnvironment>& edt, ros::NodeHandle& nh);
  ~FrontierFinder();

  void searchFrontiers();
  void computeFrontiersToVisit();

  void getFrontiers(vector<vector<Vector3d>>& clusters);
  void getDormantFrontiers(vector<vector<Vector3d>>& clusters);
  void getFrontierBoxes(vector<pair<Vector3d, Vector3d>>& boxes);
  // Get viewpoint with highest coverage for each frontier
  void getTopViewpointsInfo(const Vector3d& cur_pos, vector<Vector3d>& points, vector<double>& yaws,
      vector<Vector3d>& averages, vector<vector<int>>* unknown_voxel_addrs = nullptr,
      vector<int>* unknown_gains = nullptr);
  // Get several viewpoints for a subset of frontiers
  void getViewpointsInfo(const Vector3d& cur_pos, const vector<int>& ids, const int& view_num,
      const double& max_decay, vector<vector<Vector3d>>& points, vector<vector<double>>& yaws);
  void updateFrontierCostMatrix();
  void getFullCostMatrix(const Vector3d& cur_pos, const Vector3d& cur_vel, const Vector3d cur_yaw,
      Eigen::MatrixXd& mat);
  void getSwarmCostMatrix(const vector<Vector3d>& positions, const vector<Vector3d>& velocities,
      const vector<double> yaws, Eigen::MatrixXd& mat);
  void getSwarmCostMatrix(const vector<Vector3d>& positions, const vector<Vector3d>& velocities,
      const vector<double>& yaws, const vector<int>& ftr_ids,
      const vector<Eigen::Vector3d>& grid_pos, Eigen::MatrixXd& mat);

  void getPathForTour(const Vector3d& pos, const vector<int>& frontier_ids, vector<Vector3d>& path);

  void setNextFrontier(const int& id);
  bool isFrontierCovered();
  // Viewpoint-level suppression: a viewpoint that was reached without any
  // map change, or that the controller could not reach, is skipped for a
  // while.  Frontiers and HGrid ownership are untouched.
  void blockViewpoint(const Eigen::Vector3d& pos, double until);
  bool isViewpointBlocked(const Eigen::Vector3d& pos) const;
  void wrapYaw(double& yaw);
  int computeGainOfView(const Eigen::Vector3d& pos, const double& yaw);
  int deleteFrontiers(const vector<uint16_t>& ids);
  int addFrontiers(const vector<pair<Eigen::Vector3d, double>>& views);

  // Reconstruct locally valid viewpoints for a frontier cluster supplied by
  // a teammate.  All cells, viewpoints and ray casts are rechecked against
  // this vehicle's merged free/occupied map; stale remote frontiers therefore
  // produce no candidate.
  bool getPeerFrontierViewpoints(const vector<Eigen::Vector3d>& remote_cells,
      const Eigen::Vector3d& suggested_pos, double suggested_yaw,
      const Eigen::Vector3d& cur_pos, int max_views,
      vector<Viewpoint>& viewpoints);

  shared_ptr<PerceptionUtils> percep_utils_;

private:
  void splitLargeFrontiers(list<Frontier>& frontiers);
  bool splitHorizontally(const Frontier& frontier, list<Frontier>& splits);
  void mergeFrontiers(Frontier& ftr1, const Frontier& ftr2);
  bool isFrontierChanged(const Frontier& ft);
  bool haveOverlap(
      const Vector3d& min1, const Vector3d& max1, const Vector3d& min2, const Vector3d& max2);
  bool haveAnyOverlap(const Vector3d& min1, const Vector3d& max1, const vector<Vector3d>& mins,
      const vector<Vector3d>& maxs);
  void computeFrontierInfo(Frontier& frontier);
  void downsample(const vector<Vector3d>& cluster_in, vector<Vector3d>& cluster_out);
  void sampleViewpoints(Frontier& frontier);

  int countVisibleCells(const Vector3d& pos, const double& yaw, const vector<Vector3d>& cluster);
  int evaluateViewpoint(const Vector3d& pos, const double& yaw,
      const vector<Vector3d>& cluster, vector<int>* unknown_voxel_addrs);
  void updateWallGapMask();
  bool isWallGapClosed(const Eigen::Vector3i& idx) const;
  bool isNearObstacle(const Eigen::Vector3d& pos);
  bool isNearUnknown(const Vector3d& pos);
  vector<Eigen::Vector3i> sixNeighbors(const Eigen::Vector3i& voxel);
  vector<Eigen::Vector3i> tenNeighbors(const Eigen::Vector3i& voxel);
  vector<Eigen::Vector3i> allNeighbors(const Eigen::Vector3i& voxel);
  bool isNeighborUnknown(const Eigen::Vector3i& voxel);
  void expandFrontier(const Eigen::Vector3i& first /* , const int& depth, const int& parent_id */);

  // Wrapper of sdf map
  int toadr(const Eigen::Vector3i& idx);
  bool knownfree(const Eigen::Vector3i& idx);
  bool inmap(const Eigen::Vector3i& idx);

  // Deprecated
  Eigen::Vector3i searchClearVoxel(const Eigen::Vector3i& pt);
  bool isInBoxes(const vector<pair<Vector3d, Vector3d>>& boxes, const Eigen::Vector3i& idx);
  bool canBeMerged(const Frontier& ftr1, const Frontier& ftr2);
  void findViewpoints(const Vector3d& sample, const Vector3d& ftr_avg, vector<Viewpoint>& vps);

  // Data
  vector<char> frontier_flag_;
  // Planner-only occupied cells inferred by a conservative 2-D wall closing.
  // The raw probabilistic SLAM map is intentionally left unchanged.
  vector<char> wall_gap_mask_;
  vector<pair<Eigen::Vector3d, double>> blocked_viewpoints_;
  list<Frontier> frontiers_, dormant_frontiers_, tmp_frontiers_;
  vector<int> removed_ids_;
  list<Frontier>::iterator first_new_ftr_;
  Frontier next_frontier_;
  bool next_frontier_valid_{false};

  // Params
  int cluster_min_;
  double cluster_size_xy_, cluster_size_z_;
  double candidate_rmax_, candidate_rmin_, candidate_dphi_, min_candidate_dist_,
      min_candidate_clearance_, min_candidate_obstacle_clearance_, viewpoint_block_radius_;
  int down_sample_;
  double min_view_finish_fraction_, resolution_;
  double unknown_gain_max_dist_;
  int min_visib_num_, unknown_gain_max_rays_, candidate_rnum_,
      max_exact_cost_matrix_size_;
  int wall_gap_max_cells_, wall_gap_min_support_cells_;

  // Utils
  shared_ptr<EDTEnvironment> edt_env_;
  unique_ptr<RayCaster> raycaster_;
};

}  // namespace fast_planner
#endif
