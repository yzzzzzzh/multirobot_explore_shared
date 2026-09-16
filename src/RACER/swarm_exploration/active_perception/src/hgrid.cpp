#include <active_perception/uniform_grid.h>
#include <active_perception/hgrid.h>
#include <active_perception/graph_node.h>
#include <path_searching/astar2.h>
#include <plan_env/sdf_map.h>
#include <plan_env/edt_environment.h>
#include <algorithm>
#include <cstdint>
#include <unordered_set>

// 会创建两个 UniformGrid 对象，对应的就是uniform_grid.cpp，分别对应粗网格和细网格，也就是不同的level
// 对应第 51-52 行

namespace fast_planner {

namespace {
uint64_t gridPairKey(int first, int second) {
  const uint32_t low = static_cast<uint32_t>(std::min(first, second));
  const uint32_t high = static_cast<uint32_t>(std::max(first, second));
  return (static_cast<uint64_t>(low) << 32) | high;
}
}  // namespace

HGrid::HGrid(const shared_ptr<EDTEnvironment>& edt, ros::NodeHandle& nh) {

  this->edt_ = edt;
  nh.param("partitioning/consistent_cost", consistent_cost_, 3.5);
  nh.param("partitioning/consistent_cost2", consistent_cost2_, 3.5);
  // Sign of the consistency terms; see hgrid.h.  Default keeps upstream
  // behaviour so existing runs stay comparable.
  nh.param("partitioning/consistency_sign", consistency_sign_, 1.0);
  nh.param("partitioning/first_grid_bonus", first_grid_bonus_, 0.0);
  prev_first_grid_ids_.clear();
  nh.param("partitioning/unknown_gain_cost_weight",
      unknown_gain_cost_weight_, 0.0);
  nh.param("partitioning/unknown_gain_saturation",
      unknown_gain_saturation_, 80);
  nh.param("partitioning/use_swarm_tf", use_swarm_tf_, false);
  nh.param("partitioning/w_first", w_first_, 1.0);
  nh.param("partitioning/max_exact_pair_grids", max_exact_pair_grids_, 120);
  // Drone-to-grid costs are exact (obstacle-aware A*) within this distance;
  // beyond it RACER falls back to a penalized straight-line estimate.
  nh.param("partitioning/exact_drone_grid_dist", exact_drone_grid_dist_, 15.0);
  // Dedicated computation budget for the HGrid/CVRP geometric A*.  The
  // trajectory-level A* keeps its own astar/max_search_time.
  nh.param("partitioning/hgrid_astar_max_search_time",
      hgrid_astar_max_search_time_, 0.2);
  // A timed-out HGrid A* means the target is not straightforwardly reachable
  // (typically the cell lies behind a wall and the true route is a long
  // detour).  The former straight-line + consistent_cost2 fallback made such
  // cells look cheaper than directly reachable ones, so the ATSP tour kept
  // committing to them and then abandoning them mid-detour.  Charge a fixed
  // large cost instead so the tour prefers cells with a verified path.
  nh.param("partitioning/hgrid_astar_timeout_cost",
      hgrid_astar_timeout_cost_, 100.0);
  astar_timeout_drone_grid_ = 0;
  astar_timeout_grid_grid_ = 0;
  nh.param(
      "partitioning/compute_exact_tour_visualization", compute_exact_tour_visualization_, false);

  path_finder_.reset(new Astar);
  path_finder_->init(nh, edt);
  if (hgrid_astar_max_search_time_ > 0.0)
    path_finder_->max_search_time_ = hgrid_astar_max_search_time_;

  grid1_.reset(new UniformGrid(edt, nh, 1));
  grid2_.reset(new UniformGrid(edt, nh, 2));

  // Swarm tf
  grid1_->use_swarm_tf_ = grid2_->use_swarm_tf_ = use_swarm_tf_;
  double yaw = 0.0;
  rot_sw_ << cos(yaw), -sin(yaw), 0, sin(yaw), cos(yaw), 0, 0, 0, 1;
  trans_sw_ << 0.0, 0.0, 0;
  grid1_->rot_sw_ = grid2_->rot_sw_ = rot_sw_;
  grid1_->trans_sw_ = grid2_->trans_sw_ = trans_sw_;

  // Wait for swarm basecoor transform and initialize grid
  // while (!updateBaseCoor()) {
  //   ROS_WARN("Wait for basecoor.");
  //   ros::Duration(0.5).sleep();
  //   ros::spinOnce();
  // }
  grid1_->initGridData();
  grid2_->initGridData();
  updateBaseCoor();
}

HGrid::~HGrid() {
}

bool HGrid::updateBaseCoor() {

  // Eigen::Vector4d tf;
  // if (!edt_->sdf_map_->getBaseCoor(1, tf)) return false;
  // double yaw = tf[3];
  // rot_sw_ << cos(yaw), -sin(yaw), 0, sin(yaw), cos(yaw), 0, 0, 0, 1;
  // trans_sw_ = tf.head<3>();

  rot_sw_ = Eigen::Matrix3d::Identity();
  trans_sw_ = Eigen::Vector3d::Zero();

  grid1_->rot_sw_ = grid2_->rot_sw_ = rot_sw_;
  grid1_->trans_sw_ = grid2_->trans_sw_ = trans_sw_;
  grid1_->updateBaseCoor();
  grid2_->updateBaseCoor();

  return true;
}

void HGrid::recordTrajectoryPosition(const Eigen::Vector3d& pos) {
  // Record both levels even when the fine child is not active yet. If its
  // coarse parent is divided later, the child still remembers that the local
  // robot's trajectory has already entered it.
  grid1_->recordTrajectoryPosition(pos);
  grid2_->recordTrajectoryPosition(pos);
}

void HGrid::inputFrontiers(const vector<Eigen::Vector3d>& avgs,
    const vector<vector<int>>& viewpoint_unknown_voxels) {
  // Input frontier to both levels
  grid1_->inputFrontiers(avgs, viewpoint_unknown_voxels);
  grid2_->inputFrontiers(avgs, viewpoint_unknown_voxels);
}

void HGrid::updateGridData(const int& drone_id, vector<int>& grid_ids, bool reallocated,
    const vector<int>& last_grid_ids, vector<int>& first_ids, vector<int>& second_ids) {

  // Convert grid_ids to the ids of bi-level uniform grid
  vector<int> grid_ids1, grid_ids2;
  const int grid_num1 = grid1_->grid_data_.size();
  for (auto id : grid_ids) {
    if (id < grid_num1)
      grid_ids1.push_back(id);
    else
      grid_ids2.push_back(id - grid_num1);  // Id of level 2 grid
  }

  // std::cout << "Input ids: ";
  // for (auto id : grid_ids)
  //   std::cout << id << ", ";
  // std::cout << "" << std::endl;

  // std::cout << "level 1 ids: ";
  // for (auto id : grid_ids1)
  //   std::cout << id << ", ";
  // std::cout << "" << std::endl;

  // std::cout << "level 2 ids: ";
  // for (auto id : grid_ids2)
  //   std::cout << id << ", ";
  // std::cout << "" << std::endl;

  // Update at level 1
  vector<int> tmp_ids1 = grid_ids1;
  vector<int> parti_ids1, parti_ids1_all;
  grid1_->updateGridData(drone_id, grid_ids1, parti_ids1, parti_ids1_all);

  // std::cout << "updated level 1 ids: ";
  // for (auto id : grid_ids1)
  //   std::cout << id << ", ";
  // std::cout << "" << std::endl;

  // std::cout << "divided level 1 ids: ";
  // for (auto id : parti_ids1)
  //   std::cout << id << ", ";
  // std::cout << "" << std::endl;

  // Merge the newly partitioned and original grid ids
  vector<int> fine_ids;
  for (auto id : parti_ids1) {
    vector<int> tmp_ids;
    coarseToFineId(id, tmp_ids);
    fine_ids.insert(fine_ids.end(), tmp_ids.begin(), tmp_ids.end());
  }
  grid_ids2.insert(grid_ids2.end(), fine_ids.begin(), fine_ids.end());

  // Activate newly divided grids
  vector<int> fine_ids_all;
  for (auto id : parti_ids1_all) {
    vector<int> tmp_ids;
    coarseToFineId(id, tmp_ids);
    fine_ids_all.insert(fine_ids_all.end(), tmp_ids.begin(), tmp_ids.end());
  }
  grid2_->activateGrids(fine_ids_all);

  // if (reallocated) grid2_->activateGrids(grid_ids2);

  // std::cout << "merged level 2 ids: ";
  // for (auto id : grid_ids2)
  //   std::cout << id << ", ";
  // std::cout << "" << std::endl;

  vector<int> parti_ids2, parti_ids2_all;  // Should be empty, no partition at level 2
  grid2_->updateGridData(drone_id, grid_ids2, parti_ids2, parti_ids2_all);

  // std::cout << "updated level 2 ids: ";
  // for (auto id : grid_ids2)
  //   std::cout << id << ", ";
  // std::cout << "" << std::endl;

  grid_ids = grid_ids1;
  for (auto& id : grid_ids2) {
    grid_ids.push_back(id + grid_num1);
  }

  // Maintain consistency of next visited grid
  // if (reallocated) return;
  getConsistentGrid(last_grid_ids, grid_ids, first_ids, second_ids);
  // The own-tour cost matrix that follows has exactly one vehicle.
  prev_first_grid_ids_ = { last_grid_ids.empty() ? -1 : last_grid_ids[0] };
}

void HGrid::getConsistentGrid(const vector<int>& last_ids, const vector<int>& cur_ids,
    vector<int>& first_ids, vector<int>& second_ids) {

  if (last_ids.empty()) return;
  // std::cout << "last id: ";
  // for (auto id : last_ids)
  //   std::cout << id << ", ";
  // std::cout << "" << std::endl;

  // Find the first two level 1 grids in last sequence
  const int grid_num1 = grid1_->grid_data_.size();
  int grid_id1 = last_ids[0];
  if (grid_id1 >= grid_num1) {
    int tmp = grid_id1 - grid_num1;
    fineToCoarseId(tmp, grid_id1);
  }

  // std::cout << "level 1 grid 1: " << grid_id1 << std::endl;

  int grid_id2 = -1;
  for (int i = 1; i < last_ids.size(); ++i) {
    if (last_ids[i] < grid_num1) {
      grid_id2 = last_ids[i];
      break;
    } else {
      int fine = last_ids[i] - grid_num1;
      int coarse;
      fineToCoarseId(fine, coarse);
      if (coarse != grid_id1) {
        grid_id2 = coarse;
        break;
      }
    }
  }
  // std::cout << "level 1 grid 2: " << grid_id2 << std::endl;

  first_ids.clear();
  // In the current sequence, try to find the first level 1 grid...
  for (auto id : cur_ids) {
    if (id == grid_id1) {
      first_ids = { id };
      // std::cout << "1" << std::endl;
    }
  }

  if (first_ids.empty()) {
    // or its sub-grids
    for (auto id : cur_ids) {
      if (id < grid_num1) continue;
      int coarse;
      fineToCoarseId(id - grid_num1, coarse);
      if (coarse == grid_id1) {
        first_ids.push_back(id);
      }
    }
  }

  vector<int>* ids_ptr;
  if (!first_ids.empty()) {
    // Already find the first, should find the second
    ids_ptr = &second_ids;
  } else {
    // No first yet, continue to find the first
    ids_ptr = &first_ids;
  }

  // Can not find first grid/sub-grid, try to find the second one
  if (grid_id2 == -1) return;

  for (auto id : cur_ids) {
    if (id == grid_id2) {
      *ids_ptr = { id };
      // std::cout << "3" << std::endl;
      return;
    }
  }

  for (auto id : cur_ids) {
    if (id < grid_num1) continue;
    int coarse;
    fineToCoarseId(id - grid_num1, coarse);
    if (coarse == grid_id2) {
      // std::cout << "4" << std::endl;
      ids_ptr->push_back(id);
    }
  }
  return;
}

void HGrid::coarseToFineId(const int& coarse, vector<int>& fines) {
  fines.clear();
  Eigen::Vector3i cidx;  // coarse idx
  grid1_->adrToIndex(coarse, cidx);

  const int z_children = grid1_->use_3d_ ? 2 : 1;
  for (int dx = 0; dx < 2; ++dx) {
    for (int dy = 0; dy < 2; ++dy) {
      for (int dz = 0; dz < z_children; ++dz) {
        Eigen::Vector3i idx(
            cidx[0] * 2 + dx, cidx[1] * 2 + dy,
            grid1_->use_3d_ ? cidx[2] * 2 + dz : cidx[2]);
        fines.push_back(grid2_->toAddress(idx));
      }
    }
  }
}

void HGrid::fineToCoarseId(const int& fine, int& coarse) {
  Eigen::Vector3i fidx;
  grid2_->adrToIndex(fine, fidx);

  Eigen::Vector3i cidx;
  cidx[0] = fidx[0] / 2;
  cidx[1] = fidx[1] / 2;
  cidx[2] = grid1_->use_3d_ ? fidx[2] / 2 : fidx[2];

  coarse = grid1_->toAddress(cidx);
}

void HGrid::getCostMatrix(const vector<Eigen::Vector3d>& positions,
    const vector<Eigen::Vector3d>& velocities, const vector<vector<int>>& first_ids,
    const vector<vector<int>>& second_ids, const vector<int>& grid_ids, Eigen::MatrixXd& mat) {
  // first_ids and second_ids are drone_num x 1-4 vectors

  // Fill the cost matrix
  const int drone_num = positions.size();
  const int grid_num = grid_ids.size();
  const int dimen = 1 + drone_num + grid_num;
  mat = Eigen::MatrixXd::Zero(dimen, dimen);
  astar_timeout_drone_grid_ = 0;
  astar_timeout_grid_grid_ = 0;

  // std::cout << "First id: ";
  // for (auto ids : first_ids)
  //   for (auto id : ids)
  //     std::cout << id << ", ";
  // std::cout << "" << std::endl;

  // std::cout << "Second id: ";
  // for (auto ids : second_ids)
  //   for (auto id : ids)
  //     std::cout << id << ", ";
  // std::cout << "" << std::endl;

  // Virtual depot to drones
  for (int i = 0; i < drone_num; ++i) {
    mat(0, 1 + i) = -1000;
    mat(1 + i, 0) = 1000;
  }
  // Virtual depot to grid
  for (int i = 0; i < grid_num; ++i) {
    mat(0, 1 + drone_num + i) = 1000;
    mat(1 + drone_num + i, 0) = 0;
  }
  // Costs between drones
  for (int i = 0; i < drone_num; ++i) {
    for (int j = 0; j < drone_num; ++j) {
      mat(1 + i, 1 + j) = 10000;
    }
  }

  // Costs from drones to grid
  for (int i = 0; i < drone_num; ++i) {
    for (int j = 0; j < grid_num; ++j) {
      double cost = getCostDroneToGrid(positions[i], grid_ids[j], first_ids[i], i);
      mat(1 + i, 1 + drone_num + j) = cost;
      mat(1 + drone_num + j, 1 + i) = 0;
    }
  }
  // Costs between grids. Preserve RACER's obstacle-aware A* costs for all
  // neighboring cells on ordinary-sized problems. For a large rediscovery
  // burst, retain exact costs on each cell's two nearest local graph edges
  // instead of launching A* for every O(N^2) neighboring pair.
  exact_pair_cost_cache_.clear();
  const bool allow_all_exact_pair_search =
      max_exact_pair_grids_ <= 0 || grid_num <= max_exact_pair_grids_;
  std::unordered_set<uint64_t> bounded_exact_pairs;
  if (!allow_all_exact_pair_search) {
    constexpr int kExactNeighborsPerGrid = 2;
    for (int i = 0; i < grid_num; ++i) {
      vector<pair<double, int>> neighbors;
      for (int j = 0; j < grid_num; ++j) {
        if (i == j || !isClose(grid_ids[i], grid_ids[j])) continue;
        const double distance =
            (getGrid(grid_ids[i]).center_ - getGrid(grid_ids[j]).center_)
                .squaredNorm();
        neighbors.emplace_back(distance, j);
      }
      sort(neighbors.begin(), neighbors.end());
      const int exact_count =
          std::min(kExactNeighborsPerGrid,
              static_cast<int>(neighbors.size()));
      for (int rank = 0; rank < exact_count; ++rank)
        bounded_exact_pairs.insert(
            gridPairKey(grid_ids[i], grid_ids[neighbors[rank].second]));
    }
    ROS_WARN(
        "HGrid allocation has %d tasks; exact A* is bounded to %zu nearest "
        "local edges above threshold %d. Other global edges retain RACER's "
        "penalized geometric cost; CP and trajectory planning remain "
        "obstacle-aware.",
        grid_num, bounded_exact_pairs.size(), max_exact_pair_grids_);
  }
  for (int i = 0; i < grid_num; ++i) {
    for (int j = i + 1; j < grid_num; ++j) {
      const bool allow_exact_pair_search =
          allow_all_exact_pair_search ||
          bounded_exact_pairs.count(
              gridPairKey(grid_ids[i], grid_ids[j])) > 0;
      const double forward_cost = getCostGridToGrid(
          grid_ids[i], grid_ids[j], first_ids, second_ids, drone_num,
          allow_exact_pair_search);
      const double reverse_cost = getCostGridToGrid(
          grid_ids[j], grid_ids[i], first_ids, second_ids, drone_num,
          allow_exact_pair_search);
      mat(1 + drone_num + i, 1 + drone_num + j) = forward_cost;
      mat(1 + drone_num + j, 1 + drone_num + i) = reverse_cost;
    }
  }

  // Diag
  for (int i = 0; i < dimen; ++i) {
    mat(i, i) = 1000;
  }
  if (astar_timeout_drone_grid_ > 0 || astar_timeout_grid_grid_ > 0) {
    ROS_WARN(
        "RACER_METRIC hgrid_astar_timeout tasks=%d drone_grid=%d "
        "grid_grid=%d budget_s=%.3f cost=%.1f",
        grid_num, astar_timeout_drone_grid_, astar_timeout_grid_grid_,
        hgrid_astar_max_search_time_, hgrid_astar_timeout_cost_);
  }
}

double HGrid::getCostDroneToGrid(const Eigen::Vector3d& pos, const int& grid_id,
    const vector<int>& first, int drone_index) {
  auto& grid = getGrid(grid_id);
  double dist1, cost;
  dist1 = (pos - grid.center_).norm();
  if (dist1 < exact_drone_grid_dist_) {
    path_finder_->reset();
    const int status = path_finder_->search(pos, grid.center_);
    if (status == Astar::REACH_END) {
      auto path = path_finder_->getPath();
      cost = path_finder_->pathLength(path);
    } else if (status == Astar::TIMEOUT) {
      cost = hgrid_astar_timeout_cost_;
      ++astar_timeout_drone_grid_;
    } else {
      cost = dist1 + consistent_cost2_;
    }
  } else {
    cost = 1.5 * dist1 + consistent_cost2_;
  }
  // Consistency cost with previous first grid
  if (!first.empty()) {
    for (auto first_id : first) {
      if (grid_id == first_id) {
        cost += consistency_sign_ * consistent_cost_;
        break;
      }
    }
  }
  // if (drone_num > 1) cost *= w_first_;
  if (first_grid_bonus_ > 0.0 && drone_index >= 0 &&
      drone_index < static_cast<int>(prev_first_grid_ids_.size()) &&
      grid_id == prev_first_grid_ids_[drone_index])
    cost -= first_grid_bonus_;
  return applyUnknownGainCost(cost, grid_id);
}

double HGrid::getCostGridToGrid(const int& id1, const int& id2, const vector<vector<int>>& firsts,
    const vector<vector<int>>& seconds, const int& drone_num, bool allow_exact_search) {
  auto& grid1 = getGrid(id1);
  auto& grid2 = getGrid(id2);
  double dist_cost = 0.0;

  if (allow_exact_search && isClose(id1, id2)) {
    const uint64_t cache_key = gridPairKey(id1, id2);
    const auto cached = exact_pair_cost_cache_.find(cache_key);
    if (cached != exact_pair_cost_cache_.end()) {
      dist_cost = cached->second;
    } else {
      // Neighbor grid, search path to compute exact cost.
      path_finder_->reset();
      const int status = path_finder_->search(grid1.center_, grid2.center_);
      if (status == Astar::REACH_END) {
        auto path = path_finder_->getPath();
        dist_cost = path_finder_->pathLength(path);
      } else if (status == Astar::TIMEOUT) {
        dist_cost = hgrid_astar_timeout_cost_;
        ++astar_timeout_grid_grid_;
      } else {
        dist_cost =
            (grid1.center_ - grid2.center_).norm() + consistent_cost2_;
      }
      exact_pair_cost_cache_[cache_key] = dist_cost;
    }

    // Make level 2 grids in the same level 1 grid adhere together
    if (drone_num <= 1 && inSameLevel1(id1, id2)) {
      dist_cost += consistency_sign_ * consistent_cost_;
    }

    if (!firsts.empty()) {
      // Consistency between firsts and second
      bool is_first, is_second;
      for (int k = 0; k < drone_num; ++k) {
        is_first = false;
        is_second = false;
        for (auto first_id : firsts[k]) {
          if (id1 == first_id) {
            is_first = true;
            break;
          }
        }
        for (auto second_id : seconds[k]) {
          if (id2 == second_id) {
            is_second = true;
            break;
          }
        }
        if (is_first && is_second) break;
      }
      if (is_first && is_second) {
        dist_cost += consistency_sign_ * consistent_cost_;
      }
    }
  } else {
    // Not nearby grids, approximate cost by straight line dist
    // double dist_cost = 1.5 * (grid1.center_ - grid2.center_).norm() - consistent_cost_;
    dist_cost = 1.5 * (grid1.center_ - grid2.center_).norm() + consistent_cost2_;
    // double dist_cost = (grid1.center_ - grid2.center_).norm();
  }
  return applyUnknownGainCost(dist_cost, id2);
}

double HGrid::applyUnknownGainCost(
    double travel_cost, const int& target_grid_id) const {
  if (unknown_gain_cost_weight_ <= 0.0 || unknown_gain_saturation_ <= 0)
    return travel_cost;
  const int coarse_count = grid1_->grid_data_.size();
  const GridInfo& grid = target_grid_id < coarse_count
                             ? grid1_->grid_data_[target_grid_id]
                             : grid2_->grid_data_[target_grid_id - coarse_count];
  // An additive target reward/penalty would sum to the same constant in a
  // route that visits every HGrid and therefore would not change either the
  // ATSP ordering or the ACVRP allocation.  Weight the incoming travel edge
  // instead, so a high-gain target can justify a longer incoming leg.
  //
  // An unvisited high-unknown HGrid may intentionally have no local frontier
  // yet: RACER uses that task to seed motion into remote unknown space.  Keep
  // such a task neutral instead of treating missing evidence as zero gain.
  if (grid.contained_frontier_ids_.empty() && !grid.ever_visited_)
    return travel_cost;
  const double normalized_gain = std::min(
      1.0, std::max(0.0, static_cast<double>(grid.viewpoint_unknown_gain_) /
                             static_cast<double>(unknown_gain_saturation_)));
  return travel_cost *
         (1.0 + unknown_gain_cost_weight_ * (1.0 - normalized_gain));
}

int HGrid::getUnknownCellsNum(const int& grid_id) {
  // Get unknown cell number of a grid
  return getGrid(grid_id).unknown_num_;
}

int HGrid::getViewpointUnknownGain(const int& grid_id) {
  return getGrid(grid_id).viewpoint_unknown_gain_;
}

Eigen::Vector3d HGrid::getCenter(const int& id) {
  return getGrid(id).center_;
}

void HGrid::markGridVisited(const int& id) {
  const int grid_num1 = grid1_->grid_data_.size();
  if (id < 0) return;
  if (id < grid_num1) {
    grid1_->grid_data_[id].ever_visited_ = true;
    return;
  }
  const int fine = id - grid_num1;
  if (fine >= 0 && fine < static_cast<int>(grid2_->grid_data_.size()))
    grid2_->grid_data_[fine].ever_visited_ = true;
}

int HGrid::getCoarseGridId(const int& id) {
  const int grid_num1 = grid1_->grid_data_.size();
  if (id < grid_num1) return id;
  int coarse = -1;
  fineToCoarseId(id - grid_num1, coarse);
  return coarse;
}

bool HGrid::isGridActive(const int& id) {
  const int grid_num1 = grid1_->grid_data_.size();
  if (id < 0) return false;
  if (id < grid_num1) return grid1_->grid_data_[id].active_;
  const int fine = id - grid_num1;
  return fine >= 0 && fine < static_cast<int>(grid2_->grid_data_.size()) &&
         grid2_->grid_data_[fine].active_;
}

bool HGrid::isGridLocallyRelevant(const int& id) {
  const int grid_num1 = grid1_->grid_data_.size();
  if (id < 0) return false;
  if (id < grid_num1) return grid1_->grid_data_[id].is_cur_relevant_;
  const int fine = id - grid_num1;
  return fine >= 0 && fine < static_cast<int>(grid2_->grid_data_.size()) &&
         grid2_->grid_data_[fine].is_cur_relevant_;
}

bool HGrid::gridHasValidFrontier(const int& id) {
  const int grid_num1 = grid1_->grid_data_.size();
  if (id < 0) return false;
  if (id < grid_num1)
    return !grid1_->grid_data_[id].contained_frontier_ids_.empty();
  const int fine = id - grid_num1;
  return fine >= 0 && fine < static_cast<int>(grid2_->grid_data_.size()) &&
         !grid2_->grid_data_[fine].contained_frontier_ids_.empty();
}

GridInfo& HGrid::getGrid(const int& id) {
  int grid_num1 = grid1_->grid_data_.size();
  if (id < grid_num1)
    return grid1_->grid_data_[id];
  else
    return grid2_->grid_data_[id - grid_num1];
}

void HGrid::getActiveGrids(vector<int>& grid_ids) {
  grid_ids.clear();
  const int grid_num1 = grid1_->grid_data_.size();
  for (int i = 0; i < grid_num1; ++i) {
    if (grid1_->grid_data_[i].active_ && grid1_->grid_data_[i].is_cur_relevant_) {
      grid_ids.push_back(i);
    }
  }
  for (int i = 0; i < grid2_->grid_data_.size(); ++i) {
    if (grid2_->grid_data_[i].active_ && grid2_->grid_data_[i].is_cur_relevant_) {
      grid_ids.push_back(i + grid_num1);
    }
  }
}

bool HGrid::getNextGrid(const vector<int>& grid_ids, Eigen::Vector3d& grid_pos, double& grid_yaw) {
  // Current level 1 grid id
  const int grid_num1 = grid1_->grid_data_.size();
  int grid_id1;
  if (grid_ids[0] < grid_num1)
    grid_id1 = grid_ids[0];
  else {
    int fine = grid_ids[0] - grid_num1;  // level 2 id
    fineToCoarseId(fine, grid_id1);
  }

  // std::cout << "current level 1 id: " << grid_id1 << std::endl;

  // Find the next different level 1 grid id
  int grid_id2 = -1;
  for (int i = 1; i < grid_ids.size(); ++i) {
    if (grid_ids[i] < grid_num1) {
      grid_id2 = grid_ids[i];
      break;
    } else {
      int fine = grid_ids[i] - grid_num1;
      int coarse;
      fineToCoarseId(fine, coarse);
      if (coarse != grid_id1) {
        grid_id2 = grid_ids[i];
        break;
      }
    }
  }

  // std::cout << "next level 1 id: " << grid_id2 << std::endl;

  if (grid_id2 == -1) return false;

  auto& grid1 = getGrid(grid_id1);
  auto& grid2 = getGrid(grid_id2);
  grid_pos = grid2.center_;
  Eigen::Vector3d dir = grid2.center_ - grid1.center_;
  grid_yaw = atan2(dir[1], dir[0]);

  // std::cout << "grid pos: " << grid_pos.transpose() << std::endl;

  return true;
}

bool HGrid::isClose(const int& id1, const int& id2) {
  // Convert to coarse level ids
  const int grid_num1 = grid1_->grid_data_.size();

  int tmp_id1 = id1;
  if (tmp_id1 >= grid_num1) {
    int fine = tmp_id1 - grid_num1;
    fineToCoarseId(fine, tmp_id1);
  }
  int tmp_id2 = id2;
  if (tmp_id2 >= grid_num1) {
    int fine = tmp_id2 - grid_num1;
    fineToCoarseId(fine, tmp_id2);
  }
  Eigen::Vector3i idx1, idx2;
  grid1_->adrToIndex(tmp_id1, idx1);
  grid1_->adrToIndex(tmp_id2, idx2);

  for (int i = 0; i < 3; ++i) {
    if (abs(idx1[i] - idx2[i]) > 1) return false;
  }
  return true;

  // int diff = abs(tmp_id1 - tmp_id2);
  // if (diff == 1 || diff == grid1_->grid_num_[1]) return true;
  // return false;
}

bool HGrid::inSameLevel1(const int& id1, const int& id2) {
  // Check whether two level 2 grids are contained in the same level 1 grid
  const int grid_num1 = grid1_->grid_data_.size();
  if (id1 < grid_num1 || id2 < grid_num1) return false;

  int tmp1 = id1 - grid_num1;
  int tmp2 = id2 - grid_num1;
  int coarse1, coarse2;
  fineToCoarseId(tmp1, coarse1);
  fineToCoarseId(tmp2, coarse2);
  if (coarse1 == coarse2) return true;
  return false;
}

bool HGrid::isConsistent(const int& id1, const int& id2) {
  const int grid_num1 = grid1_->grid_data_.size();

  int tmp1 = id1;
  if (tmp1 >= grid_num1) {
    tmp1 -= grid_num1;
    int coarse;
    fineToCoarseId(tmp1, coarse);
    tmp1 = coarse;
  }
  int tmp2 = id2;
  if (tmp2 >= grid_num1) {
    tmp2 -= grid_num1;
    int coarse;
    fineToCoarseId(tmp2, coarse);
    tmp2 = coarse;
  }

  if (tmp1 == tmp2) return true;
  return false;
}

void HGrid::getGridTour(const vector<int>& ids, const Eigen::Vector3d& pos,
    vector<Eigen::Vector3d>& tour, vector<Eigen::Vector3d>& tour2) {
  const int grid_num1 = grid1_->grid_data_.size();

  // Get the centers of the visited grids
  vector<Eigen::Vector3d> centers = { pos };
  for (auto id : ids) {
    if (id < grid_num1) {
      centers.push_back(grid1_->grid_data_[id].center_);
    } else {
      int tmp = id - grid_num1;
      centers.push_back(grid2_->grid_data_[tmp].center_);
    }
  }
  tour = centers;

  if (!compute_exact_tour_visualization_) {
    // tour2 is visualization-only. Obstacle-aware A* for every global edge
    // dominated replanning time in large 3-D maps without affecting control.
    tour2 = centers;
    return;
  }

  // Find the exact path visiting the grids
  tour2 = { pos };
  for (int i = 0; i < centers.size() - 1; ++i) {
    path_finder_->reset();
    if (path_finder_->search(centers[i], centers[i + 1]) == Astar::REACH_END) {
      auto path = path_finder_->getPath();
      tour2.insert(tour2.end(), path.begin() + 1, path.end());
    } else {
      tour2.push_back(centers[i + 1]);
    }
  }
}

void HGrid::getFrontiersInGrid(const vector<int>& grid_ids, vector<int>& ftr_ids) {
  ftr_ids.clear();
  int tmp = grid_ids.front();
  int grid_num1 = grid1_->grid_data_.size();

  if (tmp < grid_num1) {
    auto& grid = grid1_->grid_data_[tmp];
    for (auto pair : grid.contained_frontier_ids_) ftr_ids.push_back(pair.first);
  } else {
    // Find all frontier in the same level 1 grid
    tmp -= grid_num1;
    int coarse;
    fineToCoarseId(tmp, coarse);
    vector<int> fines;
    coarseToFineId(coarse, fines);

    vector<int> allocated_fines;  // level 2 grid allocated to current drone
    for (auto fine : fines) {
      for (auto id : grid_ids) {
        if (fine + grid_num1 == id) {
          allocated_fines.push_back(fine);
          break;
        }
      }
    }

    for (auto fine : allocated_fines) {
      auto& grid = grid2_->grid_data_[fine];
      for (auto pair : grid.contained_frontier_ids_) ftr_ids.push_back(pair.first);
    }
  }
}

void HGrid::checkFirstGrid(const int& id) {
  auto& grid = getGrid(id);

  std::cout << "grid id: " << id << std::endl;
  std::cout << "unknown num: " << grid.unknown_num_ << std::endl;
  std::cout << "center: " << grid.center_.transpose() << std::endl;
  std::cout << "relevant: " << grid.is_cur_relevant_ << ", " << grid.is_prev_relevant_ << std::endl;
}

void HGrid::getGridMarker(vector<Eigen::Vector3d>& pts1, vector<Eigen::Vector3d>& pts2) {
  pts1.clear();
  pts2.clear();

  auto add_box_edges = [&pts1, &pts2](const GridInfo& grid) {
    static const int edges[12][2] = {
      { 0, 1 }, { 1, 2 }, { 2, 3 }, { 3, 0 },
      { 4, 5 }, { 5, 6 }, { 6, 7 }, { 7, 4 },
      { 0, 4 }, { 1, 5 }, { 2, 6 }, { 3, 7 }
    };
    for (const auto& edge : edges) {
      pts1.push_back(grid.vertices_[edge[0]]);
      pts2.push_back(grid.vertices_[edge[1]]);
    }
  };

  for (auto& grid : grid1_->grid_data_) {
    if (!grid.active_) continue;
    add_box_edges(grid);
  }
  for (auto& grid : grid2_->grid_data_) {
    if (!grid.active_) continue;
    add_box_edges(grid);
  }
}

void HGrid::getGridMarker2(vector<Eigen::Vector3d>& pts, vector<string>& texts) {
  pts.clear();
  texts.clear();

  for (int i = 0; i < grid1_->grid_data_.size(); ++i) {
    auto& grid = grid1_->grid_data_[i];
    if (!grid.active_ || !grid.is_cur_relevant_) continue;
    pts.push_back(grid.center_);
    texts.push_back(to_string(i));
  }
  const int grid_num1 = grid1_->grid_data_.size();
  for (int i = 0; i < grid2_->grid_data_.size(); ++i) {
    auto& grid = grid2_->grid_data_[i];
    if (!grid.active_ || !grid.is_cur_relevant_) continue;
    pts.push_back(grid.center_);
    texts.push_back(to_string(i + grid_num1));
  }
}

}  // namespace fast_planner
