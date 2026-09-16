#ifndef _UNIFORM_GRID_H_
#define _UNIFORM_GRID_H_

#include <ros/ros.h>
#include <Eigen/Eigen>
#include <memory>
#include <vector>
#include <list>
#include <unordered_map>
#include <utility>

using Eigen::Vector3d;
using std::list;
using std::pair;
using std::shared_ptr;
using std::unique_ptr;
using std::unordered_map;
using std::vector;

class RayCaster;

namespace fast_planner {

class EDTEnvironment;
class Astar;
class HGrid;

// struct GridInfo {};

class GridInfo {
public:
  GridInfo() {
  }
  ~GridInfo() {
  }

  int unknown_num_;
  // Union of unknown voxels predicted visible from the currently selected
  // viewpoints of all frontier clusters assigned to this grid.
  int viewpoint_unknown_gain_;
  int frontier_num_;
  Eigen::Vector3d center_;
  unordered_map<int, int> frontier_cell_nums_;
  unordered_map<int, int> contained_frontier_ids_;
  bool is_updated_;
  bool need_divide_, active_;

  bool is_prev_relevant_;
  bool is_cur_relevant_;
  // True after this robot's own odometry trajectory has entered the HGrid.
  // This is deliberately local state: a peer may still be able to approach
  // the same geometric cell from a different connected free-space component.
  bool ever_visited_;

  // Eight box vertices (bottom face first) and their axis-aligned bounding box,
  // in the current drone's frame.
  Eigen::Vector3d vmin_, vmax_;
  vector<Eigen::Vector3d> vertices_;

  // Four separating directions in the xy plane, associated with the bottom
  // face vertices.
  vector<Eigen::Vector3d> normals_;
};

class UniformGrid {

public:
  UniformGrid(const shared_ptr<EDTEnvironment>& edt, ros::NodeHandle& nh, const int& level);
  ~UniformGrid();

  void initGridData();
  void updateBaseCoor();
  void updateGridData(const int& drone_id, vector<int>& grid_ids, vector<int>& parti_ids,
      vector<int>& parti_ids_all);
  void activateGrids(const vector<int>& ids);
  void recordTrajectoryPosition(const Eigen::Vector3d& pos);

  void inputFrontiers(const vector<Eigen::Vector3d>& avgs,
      const vector<vector<int>>& viewpoint_unknown_voxels);
  void getCostMatrix(const vector<Eigen::Vector3d>& positions,
      const vector<Eigen::Vector3d>& velocities, const vector<int>& prev_first_grid,
      const vector<int>& grid_ids, Eigen::MatrixXd& mat);
  void getGridTour(const vector<int>& ids, vector<Eigen::Vector3d>& tour);
  void getFrontiersInGrid(const int& grid_id, vector<int>& ftr_ids);
  void getGridMarker(vector<Eigen::Vector3d>& pts1, vector<Eigen::Vector3d>& pts2);

private:
  void updateGridInfo(const Eigen::Vector3i& id);
  void updateObservableUnknownMask();

  int toAddress(const Eigen::Vector3i& id);
  void adrToIndex(const int& adr, Eigen::Vector3i& idx);
  void posToIndex(const Eigen::Vector3d& pos, Eigen::Vector3i& id);
  void indexToPos(const Eigen::Vector3i& id, const double& inc, Eigen::Vector3d& pos);
  bool insideGrid(const Eigen::Vector3i& id);
  bool hasEnoughUnknown(const GridInfo& grid);
  bool isRelevant(const GridInfo& grid);

  shared_ptr<EDTEnvironment> edt_;
  unique_ptr<Astar> path_finder_;
  vector<GridInfo> grid_data_;

  vector<int> relevant_id_;
  unordered_map<int, int> relevant_map_;
  bool initialized_;
  vector<int> extra_ids_;

  Eigen::Vector3d resolution_;
  Eigen::Vector3d min_, max_;
  Eigen::Vector3i grid_num_;
  int level_;
  bool use_3d_;

  int min_unknown_, min_frontier_, min_free_, min_viewpoint_unknown_gain_;
  double split_known_ratio_, min_unknown_ratio_;
  bool require_frontier_for_relevance_;
  bool exclude_visited_without_frontier_;
  // Only count unknown voxels that are connected to free space through
  // unknown voxels (i.e. reachable by some future frontier).  Enclosed
  // unknown (wall interiors, outside the building) is treated as known.
  bool observable_unknown_only_;
  // A grid whose box lies within visit_margin_ of the robot counts as visited
  // (a frontier-less cell the robot is standing next to has nothing left for
  // this robot even if it never stepped inside).
  double visit_margin_;
  // Under exclude_visited_without_frontier: an unvisited cell with no frontier
  // stays a task only if its unknown ratio is at least this (<=0 disables).
  // Cells whose remaining unknown is just wall interiors / slivers behind
  // furniture (10-15%) can never be finished and would otherwise be handed
  // back and forth between the robots forever.
  double frontierless_min_unknown_ratio_;
  std::vector<uint8_t> observable_unknown_;
  int observable_unknown_count_, enclosed_unknown_count_;
  double consistent_cost_, inside_ratio_;
  double w_unknown_;

  // Swarm tf
  Eigen::Matrix3d rot_sw_;
  Eigen::Vector3d trans_sw_;
  bool use_swarm_tf_;

  friend HGrid;
};

}  // namespace fast_planner
#endif
