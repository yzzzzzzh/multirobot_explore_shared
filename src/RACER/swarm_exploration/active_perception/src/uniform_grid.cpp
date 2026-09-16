#include <algorithm>
#include <active_perception/uniform_grid.h>
#include <active_perception/graph_node.h>
#include <path_searching/astar2.h>
#include <plan_env/sdf_map.h>
#include <plan_env/edt_environment.h>
#include <plan_env/multi_map_manager.h>
#include <unordered_set>

// RACER 层次网格 HGrid 的单层区域管理器。它不直接规划轨迹或控制机器人，而是回答：
// 地图中哪些区域仍值得探索、哪些粗网格需要细分，以及当前机器人持有的区域任务应该保留还是删除。

namespace fast_planner {

UniformGrid::UniformGrid(
    const shared_ptr<EDTEnvironment>& edt, ros::NodeHandle& nh, const int& level) {

  this->edt_ = edt;

  // Read min, max, resolution here
  nh.param("sdf_map/box_min_x", min_[0], 0.0);
  nh.param("sdf_map/box_min_y", min_[1], 0.0);
  nh.param("sdf_map/box_min_z", min_[2], 0.0);
  nh.param("sdf_map/box_max_x", max_[0], 0.0);
  nh.param("sdf_map/box_max_y", max_[1], 0.0);
  nh.param("sdf_map/box_max_z", max_[2], 0.0);
  nh.param("partitioning/min_unknown", min_unknown_, 10000);
  nh.param("partitioning/min_frontier", min_frontier_, 100);
  nh.param("partitioning/min_free", min_free_, 3000);
  nh.param("partitioning/split_known_ratio", split_known_ratio_, -1.0);
  nh.param("partitioning/min_unknown_ratio", min_unknown_ratio_, -1.0);
  nh.param("partitioning/require_frontier_for_relevance",
      require_frontier_for_relevance_, false);
  nh.param("partitioning/exclude_visited_without_frontier",
      exclude_visited_without_frontier_, false);
  nh.param("partitioning/min_viewpoint_unknown_gain",
      min_viewpoint_unknown_gain_, 0);
  nh.param("partitioning/observable_unknown_only", observable_unknown_only_, false);
  nh.param("partitioning/visit_margin", visit_margin_, 0.0);
  nh.param("partitioning/frontierless_min_unknown_ratio", frontierless_min_unknown_ratio_, -1.0);
  observable_unknown_count_ = 0;
  enclosed_unknown_count_ = 0;
  nh.param("partitioning/consistent_cost", consistent_cost_, 3.5);
  nh.param("partitioning/w_unknown", w_unknown_, 3.5);

  double grid_size, grid_size_z;
  nh.param("partitioning/grid_size", grid_size, 5.0);
  nh.param("partitioning/grid_size_z", grid_size_z, -1.0);

  auto size = max_ - min_;
  use_3d_ = grid_size_z > 0.0;

  // 这个Hgrid，每一层都会创建出一套网格单元，在hgrid.cpp和FastExplorationManager里面是分别保存的
  for (int i = 0; i < 3; ++i) {
    // A non-positive grid_size_z retains RACER's original full-height column
    // partition. A positive value enables volumetric HGrid partitioning.
    const double requested_size = (i == 2 && use_3d_) ? grid_size_z : grid_size;
    if (i == 2 && !use_3d_) {
      resolution_[i] = size[i];
      continue;
    }
    int num = ceil(size[i] / requested_size);
    resolution_[i] = size[i] / double(num);
    for (int j = 1; j < level; ++j) resolution_[i] *= 0.5;
  }
  initialized_ = false;
  level_ = level;

  // path_finder_.reset(new Astar);
  // path_finder_->init(nh, edt);
}

UniformGrid::~UniformGrid() {
}

// 根据当前层的网格尺寸，提前创建这一层的所有 HGrid 单元，并为每个单元设置初始状态。
void UniformGrid::initGridData() {
  Eigen::Vector3d size = max_ - min_;
  for (int i = 0; i < 3; ++i) grid_num_(i) = ceil(size(i) / resolution_[i]);
  grid_data_.resize(grid_num_[0] * grid_num_[1] * grid_num_[2]);

  std::cout << "data size: " << grid_data_.size() << std::endl;
  std::cout << "grid num: " << grid_num_.transpose() << std::endl;
  std::cout << "resolution: " << resolution_.transpose() << std::endl;
  std::cout << "partitioning mode: " << (use_3d_ ? "3D volumetric" : "legacy XY columns")
            << std::endl;

  // Init each grid info
  for (int x = 0; x < grid_num_[0]; ++x) {
    for (int y = 0; y < grid_num_[1]; ++y) {
      for (int z = 0; z < grid_num_[2]; ++z) {
        Eigen::Vector3i id(x, y, z);
        auto& grid = grid_data_[toAddress(id)];

        Eigen::Vector3d pos;
        indexToPos(id, 0.5, pos);
        if (use_swarm_tf_) {
          pos = rot_sw_ * pos + trans_sw_;
        }

        grid.center_ = pos;

        // 假设这个Hgrid里面的所有体素都是未知的
        // 就是Hgrid的体积除以单个体素的体积
        grid.unknown_num_ = resolution_[0] * resolution_[1] * resolution_[2] /
                            pow(edt_->sdf_map_->getResolution(), 3);
        grid.viewpoint_unknown_gain_ = 0;

        // 上一轮更新时，这个hgrid是否值得探索
        grid.is_prev_relevant_ = true;
        // 当前更新后，这个网格是否仍值得探索
        grid.is_cur_relevant_ = true;
        // 本机器人是否曾进入该网格
        grid.ever_visited_ = false;
        // 这个粗网格是否需要拆成细网格
        grid.need_divide_ = false;
        // 如果是最粗的那一层，那么就是active的，也就是true；反之就是false
        if (level_ == 1)
          grid.active_ = true;
        else
          grid.active_ = false;
      }
    }
  }
}

// 计算每个 HGrid 单元在世界坐标系中的几何范围
void UniformGrid::updateBaseCoor() {
  for (int i = 0; i < grid_data_.size(); ++i) {
    auto& grid = grid_data_[i];
    // if (!grid.active_) continue;

    Eigen::Vector3i id;
    adrToIndex(i, id);

    // Compute all eight vertices and the enclosing box in the current drone's frame.
    Eigen::Vector3d left_bottom, right_top;
    indexToPos(id, 0.0, left_bottom);
    indexToPos(id, 1.0, right_top);
    vector<Eigen::Vector3d> vertices = {
      { left_bottom[0], left_bottom[1], left_bottom[2] },
      { right_top[0], left_bottom[1], left_bottom[2] },
      { right_top[0], right_top[1], left_bottom[2] },
      { left_bottom[0], right_top[1], left_bottom[2] },
      { left_bottom[0], left_bottom[1], right_top[2] },
      { right_top[0], left_bottom[1], right_top[2] },
      { right_top[0], right_top[1], right_top[2] },
      { left_bottom[0], right_top[1], right_top[2] }
    };
    if (use_swarm_tf_) {
      for (auto& vert : vertices) vert = rot_sw_ * vert + trans_sw_;
    }

    Eigen::Vector3d vmin, vmax;
    vmin = vmax = vertices[0];
    for (int j = 1; j < vertices.size(); ++j) {
      for (int k = 0; k < 3; ++k) {
        vmin[k] = min(vmin[k], vertices[j][k]);
        vmax[k] = max(vmax[k], vertices[j][k]);
      }
    }
    grid.vertices_ = vertices;
    grid.vmin_ = vmin;
    grid.vmax_ = vmax;

    // Compute normals of four separating lines
    grid.normals_.clear();
    for (int j = 0; j < 4; ++j) {
      Eigen::Vector3d dir = (vertices[(j + 1) % 4] - vertices[j]).normalized();
      grid.normals_.push_back(dir);
    }
    // std::cout << "Vertices of grid " << toAddress(id) << std::endl;
    // for (auto v : grid.vertices_)
    //   std::cout << v.transpose() << "; ";
    // std::cout << "\nNormals: " << std::endl;
    // for (auto n : grid.normals_)
    //   std::cout << n.transpose() << "; ";
    // std::cout << "\nbox: " << grid.vmin_.transpose() << ", " << grid.vmax_.transpose()
    //           << std::endl;
  }
}

// 最主要的更新函数，包含参数为：1）drone_id：当前机器人编号
// 2）grid_ids：当前机器人持有的任务列表
// 3）parti_ids：当前机器人拥有的任务中，需要从粗网格拆成细网格的那些任务
// 4）parti_ids_all：所有机器人拥有的任务中，需要从粗网格拆成细网格的那些任务
void UniformGrid::updateGridData(const int& drone_id, vector<int>& grid_ids, vector<int>& parti_ids,
    vector<int>& parti_ids_all) {

  // parti_ids are ids of grids that are assigned to THIS drone and should be divided
  // parti_ids_all are ids of ALL grids that should be divided
  
  // 把所有网格标记为本轮未更新，并清空当前机器人的细分输出
  for (auto& grid : grid_data_) {
    grid.is_updated_ = false;
  }
  parti_ids.clear();

  // 粗网格：reset=false；细网格: reset=true
  bool reset = (level_ == 2);
  Vector3d update_min, update_max;

  // 本机器人传感器产生的地图更新
  edt_->sdf_map_->getUpdatedBox(update_min, update_max, reset);

  vector<Eigen::Vector3d> update_mins, update_maxs;
  edt_->sdf_map_->mm_->getChunkBoxes(update_mins, update_maxs, reset);

  // Rediscovered grid
  vector<int> rediscovered_ids;

  if (observable_unknown_only_) updateObservableUnknownMask();

  // Half-voxel slack: boxes mix voxel centres and raw sensor positions
  // (see FrontierFinder::haveOverlap for the planar-map failure case).
  const double overlap_slack = 0.5 * edt_->sdf_map_->getResolution() + 1e-3;
  auto have_overlap = [overlap_slack](
      const Vector3d& min1, const Vector3d& max1, const Vector3d& min2, const Vector3d& max2) {
    const double slack = overlap_slack;
    for (int m = 0; m < 3; ++m) {
      double bmin = max(min1[m], min2[m]);
      double bmax = min(max1[m], max2[m]);
      if (bmin > bmax + slack) return false;
    }
    return true;
  };

  // For each grid, check overlap with updated box and update it if necessary
  for (int i = 0; i < grid_data_.size(); ++i) {
    auto& grid = grid_data_[i];
    if (!grid.active_) continue;

    // Check overlap with updated boxes
    bool overlap = false;
    for (int j = 0; j < update_mins.size(); ++j) {
      if (have_overlap(grid.vmin_, grid.vmax_, update_mins[j], update_maxs[j])) {
        overlap = true;
        break;
      }
    }
    bool overlap_with_fov = have_overlap(grid.vmin_, grid.vmax_, update_min, update_max);
    if (!overlap && !overlap_with_fov) continue;

    // Update the grid
    Eigen::Vector3i idx;
    adrToIndex(i, idx);
    updateGridInfo(idx);

    if (grid.need_divide_) {
      parti_ids_all.push_back(i);
      grid.active_ = false;
    }

    // Rediscovered relevant grid
    if (!overlap_with_fov) continue;
    if (!grid.is_prev_relevant_ && grid.is_cur_relevant_ && level_ > 1) {
      rediscovered_ids.push_back(i);
      ROS_WARN("Grid %d is rediscovered", i);
    }
  }

  // Patch code. To avoid incomplete update of level 2 grids...
  for (auto id : extra_ids_) {
    if (grid_data_[id].is_updated_) continue;

    Eigen::Vector3i idx;
    adrToIndex(id, idx);
    updateGridInfo(idx);
  }
  extra_ids_.clear();

  // Update the list of relevant grid
  relevant_id_.clear();
  relevant_map_.clear();
  int active_count = 0;
  int active_relevant_count = 0;
  int visited_count = 0;
  int excluded_visited_no_frontier_count = 0;
  int retained_unvisited_no_frontier_count = 0;
  for (int i = 0; i < grid_data_.size(); ++i) {
    auto& grid = grid_data_[i];
    const bool relevant = isRelevant(grid);
    if (relevant) {
      relevant_id_.push_back(i);
      relevant_map_[i] = 1;
    }
    if (!grid.active_) continue;
    ++active_count;
    if (relevant) ++active_relevant_count;
    if (grid.ever_visited_) ++visited_count;
    if (exclude_visited_without_frontier_ && hasEnoughUnknown(grid) &&
        grid.contained_frontier_ids_.empty()) {
      if (grid.ever_visited_)
        ++excluded_visited_no_frontier_count;
      else if (relevant)
        ++retained_unvisited_no_frontier_count;
    }
  }
  if (exclude_visited_without_frontier_) {
    ROS_WARN(
        "RACER_METRIC hgrid_relevance drone=%d level=%d active=%d "
        "relevant=%d visited=%d excluded_visited_no_frontier=%d "
        "retained_unvisited_no_frontier=%d",
        drone_id, level_, active_count, active_relevant_count, visited_count,
        excluded_visited_no_frontier_count,
        retained_unvisited_no_frontier_count);
  }

  // Update the dominance grid of ego drone
  if (!initialized_) {
    if (drone_id == 1 && level_ == 1) grid_ids = relevant_id_;
    // else
    //   grid_ids = {};
    ROS_WARN("Init grid allocation.");
    initialized_ = true;
  } else {
    for (auto it = grid_ids.begin(); it != grid_ids.end();) {
      if (relevant_map_.find(*it) == relevant_map_.end()) {
        // Remove irrelevant ones
        // std::cout << "Remove irrelevant: " << *it << std::endl;
        it = grid_ids.erase(it);
      } else if (grid_data_[*it].need_divide_) {
        // Partition coarse grid
        // std::cout << "Remove divided: " << *it << std::endl;
        parti_ids.push_back(*it);
        it = grid_ids.erase(it);
        // grid_data_[*it].active_ = false;
      } else {
        ++it;
      }
    }
    // Add rediscovered ones
    grid_ids.insert(grid_ids.end(), rediscovered_ids.begin(), rediscovered_ids.end());

    // sort(grid_ids.begin(), grid_ids.end());
  }
}

// 区分“可能继续探索的未知区域”和“被障碍完全封闭的未知区域”
// 找到与 FREE 体素相邻的未知体素，作为起点。
// 从这些起点扩展到相互连通的未知体素。
// 被访问到的标记为 observable_unknown_。
// 没有连到自由空间的未知体素计为 enclosed_unknown_。
void UniformGrid::updateObservableUnknownMask() {
  // Flood-fill the unknown voxels that touch free space (4-neighbourhood in
  // the plane, plus the vertical neighbours for 3D maps).  Anything unknown
  // that is not reached is enclosed by occupied voxels (wall interiors,
  // sealed pillars, the outside of the building) and can never become a
  // frontier, so it must not keep an HGrid cell "relevant" forever.
  auto& map = edt_->sdf_map_;
  const int n = map->getVoxelNum();
  observable_unknown_.assign(n, 0);
  Eigen::Vector3d bmin, bmax;
  map->getBox(bmin, bmax);
  Eigen::Vector3i imin, imax;
  map->posToIndex(bmin, imin);
  map->posToIndex(bmax, imax);
  const int dirs[6][3] = { { 1, 0, 0 }, { -1, 0, 0 }, { 0, 1, 0 }, { 0, -1, 0 }, { 0, 0, 1 },
    { 0, 0, -1 } };
  const int ndirs = use_3d_ ? 6 : 4;
  std::vector<Eigen::Vector3i> stack;
  stack.reserve(4096);
  int unknown_total = 0;
  for (int x = imin[0]; x <= imax[0]; ++x)
    for (int y = imin[1]; y <= imax[1]; ++y)
      for (int z = imin[2]; z <= imax[2]; ++z) {
        const Eigen::Vector3i idx(x, y, z);
        if (!map->isInMap(idx) || map->getOccupancy(idx) != SDFMap::UNKNOWN) continue;
        ++unknown_total;
        bool seed = false;
        for (int d = 0; d < ndirs && !seed; ++d) {
          const Eigen::Vector3i nb(x + dirs[d][0], y + dirs[d][1], z + dirs[d][2]);
          if (map->isInMap(nb) && map->getOccupancy(nb) == SDFMap::FREE) seed = true;
        }
        if (!seed) continue;
        const int adr = map->toAddress(idx);
        if (adr < 0 || adr >= n || observable_unknown_[adr]) continue;
        observable_unknown_[adr] = 1;
        stack.push_back(idx);
      }
  int observable = 0;
  auto inside_box = [&](const Eigen::Vector3i& v) {
    return v[0] >= imin[0] && v[0] <= imax[0] && v[1] >= imin[1] && v[1] <= imax[1] &&
           v[2] >= imin[2] && v[2] <= imax[2];
  };
  while (!stack.empty()) {
    const Eigen::Vector3i cur = stack.back();
    stack.pop_back();
    ++observable;
    for (int d = 0; d < ndirs; ++d) {
      const Eigen::Vector3i nb(cur[0] + dirs[d][0], cur[1] + dirs[d][1], cur[2] + dirs[d][2]);
      // Stay inside the exploration box so the observable/enclosed counts
      // refer to the same voxel set as the seed scan above.
      if (!inside_box(nb) || !map->isInMap(nb) || map->getOccupancy(nb) != SDFMap::UNKNOWN)
        continue;
      const int adr = map->toAddress(nb);
      if (adr < 0 || adr >= n || observable_unknown_[adr]) continue;
      observable_unknown_[adr] = 1;
      stack.push_back(nb);
    }
  }
  observable_unknown_count_ = observable;
  enclosed_unknown_count_ = unknown_total - observable;
  ROS_INFO_THROTTLE(10.0, "RACER_METRIC observable_unknown level=%d observable=%d enclosed=%d",
      level_, observable_unknown_count_, enclosed_unknown_count_);
}

// 重新统计某一个网格内部的地图状态。
void UniformGrid::updateGridInfo(const Eigen::Vector3i& id) {
  int adr = toAddress(id);
  auto& grid = grid_data_[adr];
  if (grid.is_updated_) {  // Ensure only one update to avoid repeated computation
    return;
  }
  grid.is_updated_ = true;

  grid.is_prev_relevant_ = grid.is_cur_relevant_;

  // Check if a voxel is inside the rotated box
  auto inside_box = [](const Eigen::Vector3d& vox, const GridInfo& grid) {
    // Check four separating planes(lines)
    for (int m = 0; m < 4; ++m) {
      if ((vox - grid.vertices_[m]).dot(grid.normals_[m]) <= 0.0) return false;
    }
    if (vox[2] < grid.vmin_[2] - 1e-6 || vox[2] > grid.vmax_[2] + 1e-6) return false;
    return true;
  };

  // Count known
  const double res = edt_->sdf_map_->getResolution();
  grid.center_.setZero();
  grid.unknown_num_ = 0;
  int free = 0;
  int known = 0;
  int voxel_count = 0;
  for (double x = grid.vmin_[0]; x <= grid.vmax_[0]; x += res) {
    for (double y = grid.vmin_[1]; y <= grid.vmax_[1]; y += res) {
      for (double z = grid.vmin_[2]; z <= grid.vmax_[2]; z += res) {

        Eigen::Vector3d pos(x, y, z);
        if (!inside_box(pos, grid)) continue;

        ++voxel_count;
        int state = edt_->sdf_map_->getOccupancy(pos);
        if (state == SDFMap::FREE) {
          free += 1;
          known += 1;
        } else if (state == SDFMap::OCCUPIED) {
          known += 1;
        } else if (state == SDFMap::UNKNOWN) {
          if (observable_unknown_only_ && !observable_unknown_.empty()) {
            Eigen::Vector3i vid;
            edt_->sdf_map_->posToIndex(pos, vid);
            const int vadr = edt_->sdf_map_->toAddress(vid);
            if (vadr >= 0 && vadr < static_cast<int>(observable_unknown_.size()) &&
                !observable_unknown_[vadr]) {
              // Enclosed unknown can never be observed: count it as known so
              // that it neither keeps the grid relevant nor biases its centre.
              known += 1;
              continue;
            }
          }
          grid.center_ = (grid.center_ * grid.unknown_num_ + pos) / (grid.unknown_num_ + 1);
          grid.unknown_num_ += 1;
        }
      }
    }
  }

  grid.is_cur_relevant_ = isRelevant(grid);

  // cout << "level: " << level_ << ", grid id: " << id.transpose() << ", adr: " << adr
  //      << ", unknown: " << grid.unknown_num_ << ", center: " << grid.center_.transpose()
  //      << ", rele: " << grid.is_cur_relevant_ << endl;

  const bool enough_known =
      split_known_ratio_ > 0.0
          ? voxel_count > 0 && double(known) / double(voxel_count) >= split_known_ratio_
          : free > min_free_;
  if (level_ == 1 && grid.active_ && enough_known) {
    grid.need_divide_ = true;
  }
}

int UniformGrid::toAddress(const Eigen::Vector3i& id) {
  return id[0] * grid_num_(1) * grid_num_(2) + id[1] * grid_num_(2) + id[2];
}

void UniformGrid::adrToIndex(const int& adr, Eigen::Vector3i& idx) {
  // id[0] * grid_num_(1) * grid_num_(2) + id[1] * grid_num_(2) + id[2];
  int tmp_adr = adr;
  const int a = grid_num_(1) * grid_num_(2);
  const int b = grid_num_(2);

  idx[0] = tmp_adr / a;
  tmp_adr = tmp_adr % a;
  idx[1] = tmp_adr / b;
  idx[2] = tmp_adr % b;
}

void UniformGrid::posToIndex(const Eigen::Vector3d& pos, Eigen::Vector3i& id) {
  for (int i = 0; i < 3; ++i) id(i) = floor((pos(i) - min_(i)) / resolution_[i]);
}

void UniformGrid::indexToPos(const Eigen::Vector3i& id, const double& inc, Eigen::Vector3d& pos) {
  // inc: 0 for min, 1 for max, 0.5 for mid point
  for (int i = 0; i < 3; ++i) pos(i) = (id(i) + inc) * resolution_[i] + min_(i);
}

void UniformGrid::activateGrids(const vector<int>& ids) {
  for (auto id : ids) {
    grid_data_[id].active_ = true;
  }
  extra_ids_ = ids;  // To avoid incomplete update
}

void UniformGrid::recordTrajectoryPosition(const Eigen::Vector3d& world_pos) {
  Eigen::Vector3d local_pos = world_pos;
  if (use_swarm_tf_) {
    const Eigen::Matrix3d rot_inv = rot_sw_.transpose();
    local_pos = rot_inv * (world_pos - trans_sw_);
  }
  if (!use_3d_) {
    // A planar HGrid is an XY column. Ground-platform odometry is normally at
    // the body origin below the single occupancy slice, so z must not prevent
    // an otherwise valid XY visit from being recorded.
    local_pos[2] = 0.5 * (min_[2] + max_[2]);
  }

  Eigen::Vector3i id;
  posToIndex(local_pos, id);
  if (insideGrid(id)) grid_data_[toAddress(id)].ever_visited_ = true;
  if (visit_margin_ <= 0.0) return;
  // Also mark the neighbouring grids whose box is within visit_margin_.
  for (int dx = -1; dx <= 1; ++dx)
    for (int dy = -1; dy <= 1; ++dy) {
      if (dx == 0 && dy == 0) continue;
      const Eigen::Vector3i nid(id[0] + dx, id[1] + dy, id[2]);
      if (!insideGrid(nid)) continue;
      double dist2 = 0.0;
      for (int k = 0; k < 2; ++k) {
        const double bmin = min_[k] + nid[k] * resolution_[k];
        const double bmax = bmin + resolution_[k];
        const double d = std::max({ 0.0, bmin - local_pos[k], local_pos[k] - bmax });
        dist2 += d * d;
      }
      if (dist2 <= visit_margin_ * visit_margin_) grid_data_[toAddress(nid)].ever_visited_ = true;
    }
}

bool UniformGrid::insideGrid(const Eigen::Vector3i& id) {
  // Check inside min max
  for (int i = 0; i < 3; ++i) {
    if (id[i] < 0 || id[i] >= grid_num_[i]) {
      return false;
    }
  }
  return true;
}

void UniformGrid::inputFrontiers(const vector<Eigen::Vector3d>& avgs,
    const vector<vector<int>>& viewpoint_unknown_voxels) {
  for (auto& grid : grid_data_) {
    grid.contained_frontier_ids_.clear();
    grid.viewpoint_unknown_gain_ = 0;
  }
  vector<std::unordered_set<int>> grid_unknown_voxels(grid_data_.size());
  Eigen::Vector3i id;
  Eigen::Matrix3d Rt = rot_sw_.transpose();
  Eigen::Vector3d t_inv = -Rt * trans_sw_;

  for (int i = 0; i < avgs.size(); ++i) {
    Eigen::Vector3d pos = avgs[i];
    if (use_swarm_tf_) {
      pos = Rt * pos + t_inv;
    }
    posToIndex(pos, id);
    if (!insideGrid(id)) continue;
    const int address = toAddress(id);
    auto& grid = grid_data_[address];
    grid.contained_frontier_ids_[i] = 1;
    if (i < static_cast<int>(viewpoint_unknown_voxels.size())) {
      auto& unique = grid_unknown_voxels[address];
      unique.insert(
          viewpoint_unknown_voxels[i].begin(), viewpoint_unknown_voxels[i].end());
      grid.viewpoint_unknown_gain_ = unique.size();
    }
  }
}

bool UniformGrid::hasEnoughUnknown(const GridInfo& grid) {
  const double nominal_voxel_count =
      resolution_[0] * resolution_[1] * resolution_[2] /
      pow(edt_->sdf_map_->getResolution(), 3);
  return min_unknown_ratio_ > 0.0
             ? nominal_voxel_count > 0.0 &&
                   double(grid.unknown_num_) / nominal_voxel_count >= min_unknown_ratio_
             : grid.unknown_num_ >= min_unknown_;
}

bool UniformGrid::isRelevant(const GridInfo& grid) {
  const bool enough_unknown = hasEnoughUnknown(grid);
  const bool has_valid_frontier = !grid.contained_frontier_ids_.empty() &&
                                  grid.viewpoint_unknown_gain_ >=
                                      min_viewpoint_unknown_gain_;
  if (exclude_visited_without_frontier_) {
    // New ground-robot policy: retain high-unknown cells that this robot has
    // not reached yet, even before a local frontier/viewpoint exists. Once
    // this robot has physically entered the cell, a missing frontier means
    // it has no currently executable residual task for this robot.
    if (frontierless_min_unknown_ratio_ > 0.0) {
      const double nominal_voxel_count =
          resolution_[0] * resolution_[1] * resolution_[2] /
          pow(edt_->sdf_map_->getResolution(), 3);
      const double unknown_ratio =
          nominal_voxel_count > 0.0 ? double(grid.unknown_num_) / nominal_voxel_count : 0.0;
      return enough_unknown &&
             (has_valid_frontier ||
                 (!grid.ever_visited_ && unknown_ratio >= frontierless_min_unknown_ratio_));
    }
    return enough_unknown && (!grid.ever_visited_ || has_valid_frontier);
  }
  if (require_frontier_for_relevance_)
    return enough_unknown && has_valid_frontier;
  return enough_unknown || has_valid_frontier;
}

void UniformGrid::getCostMatrix(const vector<Eigen::Vector3d>& positions,
    const vector<Eigen::Vector3d>& velocities, const vector<int>& prev_first_grid,
    const vector<int>& grid_ids, Eigen::MatrixXd& mat) {
}

void UniformGrid::getGridTour(const vector<int>& ids, vector<Eigen::Vector3d>& tour) {
  tour.clear();
  for (int i = 0; i < ids.size(); ++i) {
    tour.push_back(grid_data_[ids[i]].center_);
  }
}

void UniformGrid::getFrontiersInGrid(const int& grid_id, vector<int>& ftr_ids) {
  // Find frontier having more than 1/4 within the first grid
  auto& first_grid = grid_data_[grid_id];
  ftr_ids.clear();
  // for (auto pair : first_grid.frontier_cell_nums_) {
  //   ftr_ids.push_back(pair.first);
  // }
  for (auto pair : first_grid.contained_frontier_ids_) {
    ftr_ids.push_back(pair.first);
  }
}

void UniformGrid::getGridMarker(vector<Eigen::Vector3d>& pts1, vector<Eigen::Vector3d>& pts2) {

  Eigen::Vector3d p1 = min_;
  Eigen::Vector3d p2 = min_ + Eigen::Vector3d(max_[0] - min_[0], 0, 0);
  for (int i = 0; i <= grid_num_[1]; ++i) {
    Eigen::Vector3d pt1 = p1 + Eigen::Vector3d(0, resolution_[1] * i, 0);
    Eigen::Vector3d pt2 = p2 + Eigen::Vector3d(0, resolution_[1] * i, 0);
    pts1.push_back(pt1);
    pts2.push_back(pt2);
  }

  p1 = min_;
  p2 = min_ + Eigen::Vector3d(0, max_[1] - min_[1], 0);
  for (int i = 0; i <= grid_num_[0]; ++i) {
    Eigen::Vector3d pt1 = p1 + Eigen::Vector3d(resolution_[0] * i, 0, 0);
    Eigen::Vector3d pt2 = p2 + Eigen::Vector3d(resolution_[0] * i, 0, 0);
    pts1.push_back(pt1);
    pts2.push_back(pt2);
  }
  for (auto& p : pts1) p[2] = 0.5;
  for (auto& p : pts2) p[2] = 0.5;
}

}  // namespace fast_planner
