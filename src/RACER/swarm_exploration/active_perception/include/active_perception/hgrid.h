#ifndef _HGRID_H_
#define _HGRID_H_

#include <ros/ros.h>
#include <Eigen/Eigen>
#include <cstdint>
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
class GridInfo;
class UniformGrid;

// struct GridInfo {};

// Hierarchical grid, contains two levels currently
class HGrid {

public:
  HGrid(const shared_ptr<EDTEnvironment>& edt, ros::NodeHandle& nh);
  ~HGrid();

  void updateGridData(const int& drone_id, vector<int>& grid_ids, bool reallocated,
      const vector<int>& last_grid_ids, vector<int>& first_ids, vector<int>& second_ids);

  bool updateBaseCoor();
  void recordTrajectoryPosition(const Eigen::Vector3d& pos);
  // Mark a grid (coarse or fine id) as visited by this robot so that, under
  // exclude_visited_without_frontier, it drops out of the task pool until a
  // frontier appears inside it.
  void markGridVisited(const int& id);
  void inputFrontiers(const vector<Eigen::Vector3d>& avgs,
      const vector<vector<int>>& viewpoint_unknown_voxels);
  void getCostMatrix(const vector<Eigen::Vector3d>& positions,
      const vector<Eigen::Vector3d>& velocities, const vector<vector<int>>& first_ids,
      const vector<vector<int>>& second_ids, const vector<int>& grid_ids, Eigen::MatrixXd& mat);
  void getGridTour(const vector<int>& ids, const Eigen::Vector3d& pos,
      vector<Eigen::Vector3d>& tour, vector<Eigen::Vector3d>& tour2);
  void getFrontiersInGrid(const vector<int>& grid_ids, vector<int>& ftr_ids);
  bool getNextGrid(const vector<int>& grid_ids, Eigen::Vector3d& grid_pos, double& grid_yaw);
  void getConsistentGrid(const vector<int>& last_ids, const vector<int>& cur_ids,
      vector<int>& first_ids, vector<int>& second_ids);

  void getGridMarker(vector<Eigen::Vector3d>& pts1, vector<Eigen::Vector3d>& pts2);
  void getGridMarker2(vector<Eigen::Vector3d>& pts, vector<std::string>& texts);
  void checkFirstGrid(const int& id);
  int getUnknownCellsNum(const int& grid_id);
  int getViewpointUnknownGain(const int& grid_id);
  Eigen::Vector3d getCenter(const int& grid_id);
  int getCoarseGridId(const int& grid_id);
  bool isGridActive(const int& grid_id);
  bool isGridLocallyRelevant(const int& grid_id);
  bool gridHasValidFrontier(const int& grid_id);
  void getActiveGrids(vector<int>& grid_ids);
  bool isConsistent(const int& id1, const int& id2);
  double getCostDroneToGrid(const Eigen::Vector3d& pos, const int& grid_id,
      const vector<int>& first, int drone_index = 0);
  // Exact grid id (fine or coarse) that headed each vehicle's previous tour,
  // indexed like the positions passed to getCostMatrix.  -1 = none.  Set by
  // updateGridData for the own-tour ATSP (one vehicle) and by the pair
  // allocation FSM for the two-vehicle ACVRP.
  void setPrevFirstGrids(const vector<int>& ids) { prev_first_grid_ids_ = ids; }
  double getCostGridToGrid(const int& id1, const int& id2, const vector<vector<int>>& firsts,
      const vector<vector<int>>& seconds, const int& drone_num, bool allow_exact_search = true);
  unique_ptr<Astar> path_finder_;

private:
  void coarseToFineId(const int& coarse, vector<int>& fines);
  void fineToCoarseId(const int& fine, int& coarse);
  GridInfo& getGrid(const int& id);
  double applyUnknownGainCost(double travel_cost, const int& target_grid_id) const;

  bool isClose(const int& id1, const int& id2);
  bool inSameLevel1(const int& id1, const int& id2);

  unique_ptr<UniformGrid> grid1_;  // Coarse level
  unique_ptr<UniformGrid> grid2_;  // Fine level

  shared_ptr<EDTEnvironment> edt_;
  double consistent_cost_;
  double consistent_cost2_;
  double unknown_gain_cost_weight_;
  int unknown_gain_saturation_;
  int max_exact_pair_grids_;
  double exact_drone_grid_dist_;
  double hgrid_astar_max_search_time_;
  // Cost charged for an edge whose bounded A* timed out (see getCostMatrix).
  double hgrid_astar_timeout_cost_;
  // +1: upstream RACER behaviour (consistent_cost is ADDED to the previous
  // first grid / same-coarse-cell edges, i.e. it penalises consistency).
  // -1: the term acts as the hysteresis bonus its name and comments describe.
  double consistency_sign_;
  // Hysteresis on the exact grid (fine or coarse id) that headed each
  // vehicle's previous tour: its drone->grid cost is reduced by
  // first_grid_bonus_ so a near-tie cannot flip the committed target every
  // replan.  Applied per vehicle in both the own-tour ATSP and the pair-wise
  // ACVRP allocation (see setPrevFirstGrids).
  double first_grid_bonus_;
  vector<int> prev_first_grid_ids_;
  int astar_timeout_drone_grid_;
  int astar_timeout_grid_grid_;
  bool compute_exact_tour_visualization_;
  // Valid for the current cost-matrix build and the immediately following
  // objective evaluation in the same single-threaded ROS callback.
  unordered_map<uint64_t, double> exact_pair_cost_cache_;

  // Swarm tf
  Eigen::Matrix3d rot_sw_;
  Eigen::Vector3d trans_sw_;
  bool use_swarm_tf_;
  double w_first_;
};

}  // namespace fast_planner
#endif
