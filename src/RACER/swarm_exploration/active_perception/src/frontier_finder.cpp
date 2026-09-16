#include <active_perception/frontier_finder.h>
#include <plan_env/sdf_map.h>
#include <plan_env/raycast.h>
// #include <path_searching/astar2.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <plan_env/edt_environment.h>
#include <plan_env/multi_map_manager.h>
#include <active_perception/perception_utils.h>
#include <active_perception/graph_node.h>

// use PCL region growing segmentation
// #include <pcl/point_types.h>
// #include <pcl/search/search.h>
// #include <pcl/search/kdtree.h>
// #include <pcl/features/normal_3d.h>
// #include <pcl/segmentation/region_growing.h>
#include <pcl/filters/voxel_grid.h>

#include <Eigen/Eigenvalues>
#include <array>
#include <unordered_set>

namespace fast_planner {
FrontierFinder::FrontierFinder(const EDTEnvironment::Ptr& edt, ros::NodeHandle& nh) {
  this->edt_env_ = edt;
  int voxel_num = edt->sdf_map_->getVoxelNum();
  frontier_flag_ = vector<char>(voxel_num, 0);
  fill(frontier_flag_.begin(), frontier_flag_.end(), 0);

  nh.param("frontier/cluster_min", cluster_min_, -1);
  nh.param("frontier/cluster_size_xy", cluster_size_xy_, -1.0);
  nh.param("frontier/cluster_size_z", cluster_size_z_, -1.0);
  nh.param("frontier/min_candidate_dist", min_candidate_dist_, -1.0);
  nh.param("frontier/min_candidate_clearance", min_candidate_clearance_, -1.0);
  // Minimum distance from a viewpoint to any OCCUPIED voxel.  The ground
  // controller refuses to drive closer than its hard clearance + braking
  // margin (0.90 m) to a raw-LiDAR return, so viewpoints sampled closer to a
  // wall than that are physically unreachable and cause hover loops at doors.
  nh.param("frontier/min_candidate_obstacle_clearance",
      min_candidate_obstacle_clearance_, 0.0);
  // Radius of a viewpoint block.  Viewpoints of one frontier are sampled on
  // 1.0-1.5 m rings around its centre, so a block must cover that ring or the
  // planner just walks through the ring one sample at a time.
  nh.param("frontier/viewpoint_block_radius", viewpoint_block_radius_, 2.0);
  nh.param("frontier/candidate_dphi", candidate_dphi_, -1.0);
  nh.param("frontier/candidate_rmax", candidate_rmax_, -1.0);
  nh.param("frontier/candidate_rmin", candidate_rmin_, -1.0);
  nh.param("frontier/candidate_rnum", candidate_rnum_, -1);
  nh.param("frontier/down_sample", down_sample_, -1);
  nh.param("frontier/min_visib_num", min_visib_num_, -1);
  nh.param("frontier/min_view_finish_fraction", min_view_finish_fraction_, -1.0);
  nh.param("frontier/max_exact_cost_matrix_size", max_exact_cost_matrix_size_, 12);
  nh.param("frontier/unknown_gain_max_dist", unknown_gain_max_dist_, -1.0);
  nh.param("frontier/unknown_gain_max_rays", unknown_gain_max_rays_, 64);
  nh.param("frontier/wall_gap_max_cells", wall_gap_max_cells_, 0);
  nh.param("frontier/wall_gap_min_support_cells", wall_gap_min_support_cells_, 1);
  wall_gap_max_cells_ = std::max(0, wall_gap_max_cells_);
  wall_gap_min_support_cells_ = std::max(1, wall_gap_min_support_cells_);
  if (wall_gap_max_cells_ > 0)
    wall_gap_mask_.assign(voxel_num, 0);
  if (unknown_gain_max_dist_ <= 0.0)
    nh.param("perception_utils/max_dist", unknown_gain_max_dist_, -1.0);

  raycaster_.reset(new RayCaster);
  resolution_ = edt_env_->sdf_map_->getResolution();
  Eigen::Vector3d origin, size;
  edt_env_->sdf_map_->getRegion(origin, size);
  raycaster_->setParams(resolution_, origin);

  percep_utils_.reset(new PerceptionUtils(nh));
  ROS_WARN(
      "Frontier unknown-gain scoring (ranking only): max_dist=%.2f max_rays=%d "
      "wall_mask_radius_cells=%d",
      unknown_gain_max_dist_, unknown_gain_max_rays_, wall_gap_max_cells_);
}

FrontierFinder::~FrontierFinder() {
}

void FrontierFinder::searchFrontiers() {
  ros::Time t1 = ros::Time::now();
  tmp_frontiers_.clear();

  // Rebuild the virtual wall layer before checking whether existing frontiers
  // changed.  This is a planner-only morphological closing: it fills tiny
  // breaks in an otherwise supported wall but never overwrites SLAM log odds.
  updateWallGapMask();

  // Bounding box of updated region
  Vector3d update_min, update_max;
  edt_env_->sdf_map_->getUpdatedBox(update_min, update_max, false);

  // Bounding box of external chunks
  vector<Eigen::Vector3d> chunk_mins, chunk_maxs;
  edt_env_->sdf_map_->mm_->getChunkBoxes(chunk_mins, chunk_maxs, false);

  vector<Eigen::Vector3d> mins, maxs;
  mins.push_back(update_min);
  mins.insert(mins.end(), chunk_mins.begin(), chunk_mins.end());
  maxs.push_back(update_max);
  maxs.insert(maxs.end(), chunk_maxs.begin(), chunk_maxs.end());

  // Removed changed frontiers in updated map
  auto resetFlag = [&](list<Frontier>::iterator& iter, list<Frontier>& frontiers) {
    Eigen::Vector3i idx;
    for (auto cell : iter->cells_) {
      edt_env_->sdf_map_->posToIndex(cell, idx);
      frontier_flag_[toadr(idx)] = 0;
    }
    iter = frontiers.erase(iter);
  };

  // std::cout << "Before remove: " << frontiers_.size() << std::endl;

  removed_ids_.clear();
  int rmv_idx = 0;
  for (auto iter = frontiers_.begin(); iter != frontiers_.end();) {
    // haveOverlap(iter->box_min_, iter->box_max_, update_min, update_max)
    if (haveAnyOverlap(iter->box_min_, iter->box_max_, mins, maxs) && isFrontierChanged(*iter)) {
      resetFlag(iter, frontiers_);
      removed_ids_.push_back(rmv_idx);
    } else {
      ++rmv_idx;
      ++iter;
    }
  }
  // std::cout << "After remove: " << frontiers_.size() << std::endl;
  for (auto iter = dormant_frontiers_.begin(); iter != dormant_frontiers_.end();) {
    if (haveAnyOverlap(iter->box_min_, iter->box_max_, mins, maxs) && isFrontierChanged(*iter))
      resetFlag(iter, dormant_frontiers_);
    else
      ++iter;
  }

  // Search new frontier within box slightly inflated from updated box
  Vector3d box_min, box_max;
  edt_env_->sdf_map_->getBox(box_min, box_max);

  vector<Eigen::Vector3d> search_mins, search_maxs;
  for (int i = 0; i < mins.size(); ++i) {
    search_mins.push_back(mins[i] - Vector3d(1, 1, 0.2));
    search_maxs.push_back(maxs[i] + Vector3d(1, 1, 0.2));
    for (int k = 0; k < 3; ++k) {
      search_mins[i][k] = max(search_mins[i][k], box_min[k]);
      search_maxs[i][k] = min(search_maxs[i][k], box_max[k]);
    }
  }
  vector<Eigen::Vector3i> min_ids(mins.size()), max_ids(mins.size());
  for (int i = 0; i < mins.size(); ++i) {
    edt_env_->sdf_map_->posToIndex(search_mins[i], min_ids[i]);
    edt_env_->sdf_map_->posToIndex(search_maxs[i], max_ids[i]);
  }

  for (int i = 0; i < min_ids.size(); ++i) {
    auto min_id = min_ids[i];
    auto max_id = max_ids[i];

    for (int z = min_id(2); z <= max_id(2); ++z)
      for (int x = min_id(0); x <= max_id(0); ++x)
        for (int y = min_id(1); y <= max_id(1); ++y) {
          // Scanning the updated region to find seeds of frontiers
          Eigen::Vector3i cur(x, y, z);
          if (frontier_flag_[toadr(cur)] == 0 && knownfree(cur) && isNeighborUnknown(cur)) {
            // Expand from the seed cell to find a complete frontier cluster
            expandFrontier(cur);
          }
        }
  }

  splitLargeFrontiers(tmp_frontiers_);
}

void FrontierFinder::expandFrontier(
    const Eigen::Vector3i& first /* , const int& depth, const int& parent_id */) {

  // Data for clustering
  queue<Eigen::Vector3i> cell_queue;
  vector<Eigen::Vector3d> expanded;
  Vector3d pos;

  edt_env_->sdf_map_->indexToPos(first, pos);
  expanded.push_back(pos);
  cell_queue.push(first);
  frontier_flag_[toadr(first)] = 1;

  // Search frontier cluster based on region growing (distance clustering)
  while (!cell_queue.empty()) {
    auto cur = cell_queue.front();
    cell_queue.pop();
    auto nbrs = allNeighbors(cur);
    for (auto nbr : nbrs) {
      // Qualified cell should be inside bounding box and frontier cell not clustered
      // Check bounds before converting to a flat address.  This matters for a
      // one-voxel-high planar map: its vertical neighbors are intentionally
      // outside the map and must never index frontier_flag_.
      if (!edt_env_->sdf_map_->isInBox(nbr)) continue;
      int adr = toadr(nbr);
      if (frontier_flag_[adr] == 1 ||
          !(knownfree(nbr) && isNeighborUnknown(nbr)))
        continue;

      edt_env_->sdf_map_->indexToPos(nbr, pos);
      if (pos[2] < 0.2) continue;  // Remove noise close to ground
      expanded.push_back(pos);
      cell_queue.push(nbr);
      frontier_flag_[adr] = 1;
    }
  }
  if (expanded.size() > cluster_min_) {
    // Compute detailed info
    Frontier frontier;
    frontier.cells_ = expanded;
    computeFrontierInfo(frontier);
    tmp_frontiers_.push_back(frontier);
  }
}

void FrontierFinder::splitLargeFrontiers(list<Frontier>& frontiers) {
  list<Frontier> splits, tmps;
  for (auto it = frontiers.begin(); it != frontiers.end(); ++it) {
    // Check if each frontier needs to be split horizontally
    if (splitHorizontally(*it, splits)) {
      tmps.insert(tmps.end(), splits.begin(), splits.end());
      splits.clear();
    } else
      tmps.push_back(*it);
  }
  frontiers = tmps;
}

bool FrontierFinder::splitHorizontally(const Frontier& frontier, list<Frontier>& splits) {
  // Split a frontier into small piece if it is too large
  auto mean = frontier.average_.head<2>();
  bool need_split = false;
  for (auto cell : frontier.filtered_cells_) {
    if ((cell.head<2>() - mean).norm() > cluster_size_xy_) {
      need_split = true;
      break;
    }
  }
  if (!need_split) return false;

  // Compute principal component
  // Covariance matrix of cells
  Eigen::Matrix2d cov;
  cov.setZero();
  for (auto cell : frontier.filtered_cells_) {
    Eigen::Vector2d diff = cell.head<2>() - mean;
    cov += diff * diff.transpose();
  }
  cov /= double(frontier.filtered_cells_.size());

  // Find eigenvector corresponds to maximal eigenvector
  Eigen::EigenSolver<Eigen::Matrix2d> es(cov);
  auto values = es.eigenvalues().real();
  auto vectors = es.eigenvectors().real();
  int max_idx;
  double max_eigenvalue = -1000000;
  for (int i = 0; i < values.rows(); ++i) {
    if (values[i] > max_eigenvalue) {
      max_idx = i;
      max_eigenvalue = values[i];
    }
  }
  Eigen::Vector2d first_pc = vectors.col(max_idx);
  // std::cout << "max idx: " << max_idx << std::endl;
  // std::cout << "mean: " << mean.transpose() << ", first pc: " << first_pc.transpose() <<
  // std::endl;

  // Split the frontier into two groups along the first PC
  Frontier ftr1, ftr2;
  for (auto cell : frontier.cells_) {
    if ((cell.head<2>() - mean).dot(first_pc) >= 0)
      ftr1.cells_.push_back(cell);
    else
      ftr2.cells_.push_back(cell);
  }
  computeFrontierInfo(ftr1);
  computeFrontierInfo(ftr2);

  // Recursive call to split frontier that is still too large
  list<Frontier> splits2;
  if (splitHorizontally(ftr1, splits2)) {
    splits.insert(splits.end(), splits2.begin(), splits2.end());
    splits2.clear();
  } else
    splits.push_back(ftr1);

  if (splitHorizontally(ftr2, splits2))
    splits.insert(splits.end(), splits2.begin(), splits2.end());
  else
    splits.push_back(ftr2);

  return true;
}

bool FrontierFinder::isInBoxes(
    const vector<pair<Vector3d, Vector3d>>& boxes, const Eigen::Vector3i& idx) {
  Vector3d pt;
  edt_env_->sdf_map_->indexToPos(idx, pt);
  for (auto box : boxes) {
    // Check if contained by a box
    bool inbox = true;
    for (int i = 0; i < 3; ++i) {
      inbox = inbox && pt[i] > box.first[i] && pt[i] < box.second[i];
      if (!inbox) break;
    }
    if (inbox) return true;
  }
  return false;
}

void FrontierFinder::updateFrontierCostMatrix() {
  // std::cout << "cost mat size before remove: " << std::endl;
  // for (auto ftr : frontiers_)
  //   std::cout << "(" << ftr.costs_.size() << "," << ftr.paths_.size() << "), ";
  // std::cout << "" << std::endl;

  // std::cout << "cost mat size remove: " << std::endl;
  if (!removed_ids_.empty()) {
    // Delete path and cost for removed clusters
    for (auto it = frontiers_.begin(); it != first_new_ftr_; ++it) {
      auto cost_iter = it->costs_.begin();
      auto path_iter = it->paths_.begin();
      int iter_idx = 0;
      for (int i = 0; i < removed_ids_.size(); ++i) {
        // Step iterator to the item to be removed
        while (iter_idx < removed_ids_[i]) {
          ++cost_iter;
          ++path_iter;
          ++iter_idx;
        }
        cost_iter = it->costs_.erase(cost_iter);
        path_iter = it->paths_.erase(path_iter);
      }
      // std::cout << "(" << it->costs_.size() << "," << it->paths_.size() << "), ";
    }
    removed_ids_.clear();
  }

  const bool use_exact_paths =
      max_exact_cost_matrix_size_ <= 0 ||
      static_cast<int>(frontiers_.size()) <= max_exact_cost_matrix_size_;
  auto updateCost = [use_exact_paths](
                        const list<Frontier>::iterator& it1,
                        const list<Frontier>::iterator& it2) {
    // std::cout << "(" << it1->id_ << "," << it2->id_ << "), ";
    // Search path from old cluster's top viewpoint to new cluster'
    Viewpoint& vui = it1->viewpoints_.front();
    Viewpoint& vuj = it2->viewpoints_.front();
    vector<Vector3d> path_ij;
    double cost_ij;
    if (use_exact_paths) {
      cost_ij = ViewNode::computeCost(
          vui.pos_, vuj.pos_, vui.yaw_, vuj.yaw_, Vector3d::Zero(), 0, path_ij);
    } else {
      // The all-pairs paths are used to order frontiers, while the selected
      // next viewpoint is always checked again by obstacle-aware A*.  Avoid
      // O(N^2) A* calls when the map contains many frontier clusters.
      path_ij = { vui.pos_, vuj.pos_ };
      const double pos_cost = (vui.pos_ - vuj.pos_).norm() / ViewNode::vm_;
      double yaw_diff = fabs(vui.yaw_ - vuj.yaw_);
      yaw_diff = min(yaw_diff, 2 * M_PI - yaw_diff);
      cost_ij = max(pos_cost, yaw_diff / ViewNode::yd_);
    }
    // Insert item for both old and new clusters
    it1->costs_.push_back(cost_ij);
    it1->paths_.push_back(path_ij);
    reverse(path_ij.begin(), path_ij.end());
    it2->costs_.push_back(cost_ij);
    it2->paths_.push_back(path_ij);
  };

  // std::cout << "cost mat add: " << std::endl;
  // Compute path and cost between old and new clusters
  for (auto it1 = frontiers_.begin(); it1 != first_new_ftr_; ++it1)
    for (auto it2 = first_new_ftr_; it2 != frontiers_.end(); ++it2) updateCost(it1, it2);

  // Compute path and cost between new clusters
  for (auto it1 = first_new_ftr_; it1 != frontiers_.end(); ++it1)
    for (auto it2 = it1; it2 != frontiers_.end(); ++it2) {
      if (it1 == it2) {
        // std::cout << "(" << it1->id_ << "," << it2->id_ << "), ";
        it1->costs_.push_back(0);
        it1->paths_.push_back({});
      } else
        updateCost(it1, it2);
    }
  first_new_ftr_ = frontiers_.end();

  if (!frontiers_.empty())
    std::cout << "Frontier cost mat size: " << frontiers_.front().costs_.size() << std::endl;

  // std::cout << "cost mat size final: " << std::endl;
  // for (auto ftr : frontiers_)
  //   std::cout << "(" << ftr.costs_.size() << "," << ftr.paths_.size() << "), ";
  // std::cout << "" << std::endl;
}

void FrontierFinder::mergeFrontiers(Frontier& ftr1, const Frontier& ftr2) {
  // Merge ftr2 into ftr1
  ftr1.average_ =
      (ftr1.average_ * double(ftr1.cells_.size()) + ftr2.average_ * double(ftr2.cells_.size())) /
      (double(ftr1.cells_.size() + ftr2.cells_.size()));
  ftr1.cells_.insert(ftr1.cells_.end(), ftr2.cells_.begin(), ftr2.cells_.end());
  computeFrontierInfo(ftr1);
}

bool FrontierFinder::canBeMerged(const Frontier& ftr1, const Frontier& ftr2) {
  Vector3d merged_avg =
      (ftr1.average_ * double(ftr1.cells_.size()) + ftr2.average_ * double(ftr2.cells_.size())) /
      (double(ftr1.cells_.size() + ftr2.cells_.size()));
  // Check if it can merge two frontier without exceeding size limit
  for (auto c1 : ftr1.cells_) {
    auto diff = c1 - merged_avg;
    if (diff.head<2>().norm() > cluster_size_xy_ || diff[2] > cluster_size_z_) return false;
  }
  for (auto c2 : ftr2.cells_) {
    auto diff = c2 - merged_avg;
    if (diff.head<2>().norm() > cluster_size_xy_ || diff[2] > cluster_size_z_) return false;
  }
  return true;
}

bool FrontierFinder::haveOverlap(
    const Vector3d& min1, const Vector3d& max1, const Vector3d& min2, const Vector3d& max2) {
  // Check if two box have overlap part.  Boxes are built from voxel centres
  // (frontier cells) or from raw sensor/endpoint positions (updated box), so
  // allow half a voxel of slack.  With the planar one-voxel-high map the
  // frontier box sits at z = 0.625 (cell centre) while the updated box spans
  // z = 0.64..0.645 (flattened cloud / sensor height); the former 1e-3
  // tolerance therefore never overlapped, no frontier was ever re-checked or
  // removed by this robot's own scans, and isFrontierCovered() never fired.
  const double slack = 0.5 * resolution_ + 1e-3;
  Vector3d bmin, bmax;
  for (int i = 0; i < 3; ++i) {
    bmin[i] = max(min1[i], min2[i]);
    bmax[i] = min(max1[i], max2[i]);
    if (bmin[i] > bmax[i] + slack) return false;
  }
  return true;
}

bool FrontierFinder::haveAnyOverlap(const Vector3d& min1, const Vector3d& max1,
    const vector<Vector3d>& mins, const vector<Vector3d>& maxs) {

  for (int i = 0; i < mins.size(); ++i) {
    if (haveOverlap(min1, max1, mins[i], maxs[i])) return true;
  }
  return false;
}

bool FrontierFinder::isFrontierChanged(const Frontier& ft) {
  for (auto cell : ft.cells_) {
    Eigen::Vector3i idx;
    edt_env_->sdf_map_->posToIndex(cell, idx);
    if (!(knownfree(idx) && isNeighborUnknown(idx))) return true;
  }
  return false;
}

void FrontierFinder::computeFrontierInfo(Frontier& ftr) {
  // Compute average position and bounding box of cluster
  ftr.average_.setZero();
  ftr.box_max_ = ftr.cells_.front();
  ftr.box_min_ = ftr.cells_.front();
  for (auto cell : ftr.cells_) {
    ftr.average_ += cell;
    for (int i = 0; i < 3; ++i) {
      ftr.box_min_[i] = min(ftr.box_min_[i], cell[i]);
      ftr.box_max_[i] = max(ftr.box_max_[i], cell[i]);
    }
  }
  ftr.average_ /= double(ftr.cells_.size());

  // Compute downsampled cluster
  downsample(ftr.cells_, ftr.filtered_cells_);
}

void FrontierFinder::computeFrontiersToVisit() {
  first_new_ftr_ = frontiers_.end();
  int new_num = 0;
  int new_dormant_num = 0;
  // Try find viewpoints for each cluster and categorize them according to viewpoint number
  for (auto& tmp_ftr : tmp_frontiers_) {
    // Search viewpoints around frontier
    sampleViewpoints(tmp_ftr);
    if (!tmp_ftr.viewpoints_.empty()) {
      ++new_num;
      list<Frontier>::iterator inserted = frontiers_.insert(frontiers_.end(), tmp_ftr);
      // A frontier-cell visibility count only proves that the boundary itself
      // can be seen.  Prefer the view whose rays continue through that boundary
      // into the largest set of unique unknown voxels; use frontier visibility
      // only as a tie breaker.
      sort(inserted->viewpoints_.begin(), inserted->viewpoints_.end(),
          [](const Viewpoint& v1, const Viewpoint& v2) {
            if (v1.unknown_gain_ != v2.unknown_gain_)
              return v1.unknown_gain_ > v2.unknown_gain_;
            return v1.visib_num_ > v2.visib_num_;
          });
      if (first_new_ftr_ == frontiers_.end()) first_new_ftr_ = inserted;
    } else {
      // Find no viewpoint, move cluster to dormant list
      dormant_frontiers_.push_back(tmp_ftr);
      ++new_dormant_num;
    }
  }
  // Reset indices of frontiers
  int idx = 0;
  for (auto& ft : frontiers_) {
    ft.id_ = idx++;
  }
  // std::cout << "\nnew num: " << new_num << ", new dormant: " << new_dormant_num << std::endl;
  std::cout << "Frontier num: " << frontiers_.size() << ", " << dormant_frontiers_.size()
            << std::endl;
}

void FrontierFinder::getTopViewpointsInfo(const Vector3d& cur_pos, vector<Eigen::Vector3d>& points,
    vector<double>& yaws, vector<Eigen::Vector3d>& averages,
    vector<vector<int>>* unknown_voxel_addrs, vector<int>* unknown_gains) {
  points.clear();
  yaws.clear();
  averages.clear();
  if (unknown_voxel_addrs != nullptr) unknown_voxel_addrs->clear();
  if (unknown_gains != nullptr) unknown_gains->clear();

  for (const auto& frontier : frontiers_) {
    bool no_view = true;
    for (const auto& view : frontier.viewpoints_) {
      // Retrieve the first viewpoint that is far enough and has highest coverage
      if ((view.pos_ - cur_pos).norm() < min_candidate_dist_) continue;
      if (isViewpointBlocked(view.pos_)) continue;
      points.push_back(view.pos_);
      yaws.push_back(view.yaw_);
      averages.push_back(frontier.average_);
      if (unknown_voxel_addrs != nullptr)
        unknown_voxel_addrs->push_back(view.unknown_voxel_addrs_);
      if (unknown_gains != nullptr) unknown_gains->push_back(view.unknown_gain_);
      no_view = false;
      break;
    }
    if (no_view) {
      // Prefer the farthest valid alternative. Reusing the highest-gain
      // viewpoint at the current pose can create a one-point A* path.
      auto view = *max_element(frontier.viewpoints_.begin(), frontier.viewpoints_.end(),
          [&cur_pos](const Viewpoint& lhs, const Viewpoint& rhs) {
            return (lhs.pos_ - cur_pos).squaredNorm() <
                   (rhs.pos_ - cur_pos).squaredNorm();
          });
      points.push_back(view.pos_);
      yaws.push_back(view.yaw_);
      averages.push_back(frontier.average_);
      if (unknown_voxel_addrs != nullptr)
        unknown_voxel_addrs->push_back(view.unknown_voxel_addrs_);
      if (unknown_gains != nullptr) unknown_gains->push_back(view.unknown_gain_);
    }
  }
}

void FrontierFinder::getViewpointsInfo(const Vector3d& cur_pos, const vector<int>& ids,
    const int& view_num, const double& max_decay, vector<vector<Eigen::Vector3d>>& points,
    vector<vector<double>>& yaws) {
  points.clear();
  yaws.clear();
  for (auto id : ids) {
    // Scan all frontiers to find one with the same id
    for (auto frontier : frontiers_) {
      if (frontier.id_ == id) {
        // Get several top viewpoints that are far enough
        vector<Eigen::Vector3d> pts;
        vector<double> ys;
        const int gain_thresh =
            static_cast<int>(frontier.viewpoints_.front().unknown_gain_ * max_decay);
        for (const auto& view : frontier.viewpoints_) {
          if (pts.size() >= view_num || view.unknown_gain_ < gain_thresh) break;
          if ((view.pos_ - cur_pos).norm() < min_candidate_dist_) continue;
          if (isViewpointBlocked(view.pos_)) continue;
          pts.push_back(view.pos_);
          ys.push_back(view.yaw_);
        }
        if (pts.empty()) {
          // The high-gain subset is too close. Search all candidates before
          // accepting a zero-length target.
          for (const auto& view : frontier.viewpoints_) {
            if (pts.size() >= view_num) break;
            if ((view.pos_ - cur_pos).norm() < min_candidate_dist_) continue;
            if (isViewpointBlocked(view.pos_)) continue;
            pts.push_back(view.pos_);
            ys.push_back(view.yaw_);
          }
        }
        if (pts.empty()) {
          auto view = *max_element(frontier.viewpoints_.begin(), frontier.viewpoints_.end(),
              [&cur_pos](const Viewpoint& lhs, const Viewpoint& rhs) {
                return (lhs.pos_ - cur_pos).squaredNorm() <
                       (rhs.pos_ - cur_pos).squaredNorm();
              });
          pts.push_back(view.pos_);
          ys.push_back(view.yaw_);
        }
        points.push_back(pts);
        yaws.push_back(ys);
      }
    }
  }
}

bool FrontierFinder::getPeerFrontierViewpoints(
    const vector<Eigen::Vector3d>& remote_cells,
    const Eigen::Vector3d& suggested_pos, double suggested_yaw,
    const Eigen::Vector3d& cur_pos, int max_views,
    vector<Viewpoint>& viewpoints) {
  viewpoints.clear();
  if (remote_cells.empty() || max_views <= 0) return false;

  // Reconstruct only the portion that is still a frontier in this vehicle's
  // map.  This simultaneously rejects stale/already explored peer work and
  // ensures the free side of the boundary arrived through ChunkData.
  Frontier frontier;
  std::unordered_set<int> unique_cells;
  for (const auto& cell : remote_cells) {
    Eigen::Vector3i index;
    edt_env_->sdf_map_->posToIndex(cell, index);
    if (!edt_env_->sdf_map_->isInBox(index)) continue;
    const int address = toadr(index);
    if (!unique_cells.insert(address).second) continue;
    if (knownfree(index) && isNeighborUnknown(index))
      frontier.cells_.push_back(cell);
  }

  // countVisibleCells() requires a strict `> min_visib_num_` result, so fewer
  // cells can never produce an executable local viewpoint.
  if (static_cast<int>(frontier.cells_.size()) <= min_visib_num_) return false;

  computeFrontierInfo(frontier);
  sampleViewpoints(frontier);

  // The sender's suggestion is only another sample.  It is accepted after the
  // same local clearance, free-space and visibility checks as generated views.
  if (edt_env_->sdf_map_->isInBox(suggested_pos) &&
      edt_env_->sdf_map_->getOccupancy(suggested_pos) == SDFMap::FREE &&
      edt_env_->sdf_map_->getInflateOccupancy(suggested_pos) != 1 &&
      !isNearUnknown(suggested_pos) && !isNearObstacle(suggested_pos)) {
    vector<int> unknown_voxels;
    const int visible = evaluateViewpoint(
        suggested_pos, suggested_yaw, frontier.filtered_cells_, &unknown_voxels);
    if (visible > min_visib_num_) {
      bool duplicate = false;
      for (const auto& view : frontier.viewpoints_) {
        if ((view.pos_ - suggested_pos).norm() < 0.25) {
          duplicate = true;
          break;
        }
      }
      if (!duplicate) {
        Viewpoint suggested_view;
        suggested_view.pos_ = suggested_pos;
        suggested_view.yaw_ = suggested_yaw;
        suggested_view.visib_num_ = visible;
        suggested_view.unknown_gain_ = unknown_voxels.size();
        suggested_view.unknown_voxel_addrs_ = std::move(unknown_voxels);
        frontier.viewpoints_.push_back(std::move(suggested_view));
      }
    }
  }

  if (frontier.viewpoints_.empty()) return false;
  sort(frontier.viewpoints_.begin(), frontier.viewpoints_.end(),
      [&cur_pos](const Viewpoint& first, const Viewpoint& second) {
        if (first.unknown_gain_ != second.unknown_gain_)
          return first.unknown_gain_ > second.unknown_gain_;
        if (first.visib_num_ != second.visib_num_)
          return first.visib_num_ > second.visib_num_;
        return (first.pos_ - cur_pos).squaredNorm() <
               (second.pos_ - cur_pos).squaredNorm();
      });
  const int count =
      std::min(max_views, static_cast<int>(frontier.viewpoints_.size()));
  viewpoints.insert(viewpoints.end(), frontier.viewpoints_.begin(),
      frontier.viewpoints_.begin() + count);
  return true;
}

void FrontierFinder::getFrontiers(vector<vector<Eigen::Vector3d>>& clusters) {
  clusters.clear();
  for (auto frontier : frontiers_) clusters.push_back(frontier.cells_);
  // clusters.push_back(frontier.filtered_cells_);
}

void FrontierFinder::getDormantFrontiers(vector<vector<Vector3d>>& clusters) {
  clusters.clear();
  for (auto ft : dormant_frontiers_) clusters.push_back(ft.cells_);
}

void FrontierFinder::getFrontierBoxes(vector<pair<Eigen::Vector3d, Eigen::Vector3d>>& boxes) {
  boxes.clear();
  for (auto frontier : frontiers_) {
    Vector3d center = (frontier.box_max_ + frontier.box_min_) * 0.5;
    Vector3d scale = frontier.box_max_ - frontier.box_min_;
    boxes.push_back(make_pair(center, scale));
  }
}

void FrontierFinder::getPathForTour(
    const Vector3d& pos, const vector<int>& frontier_ids, vector<Vector3d>& path) {
  if (frontier_ids.empty()) return;

  // Make an frontier_indexer to access the frontier list easier
  vector<list<Frontier>::iterator> frontier_indexer;
  for (auto it = frontiers_.begin(); it != frontiers_.end(); ++it) frontier_indexer.push_back(it);

  // Compute the path from current pos to the first frontier
  vector<Vector3d> segment;
  ViewNode::searchPath(pos, frontier_indexer[frontier_ids[0]]->viewpoints_.front().pos_, segment);
  path.insert(path.end(), segment.begin(), segment.end());

  // Get paths of tour passing all clusters
  for (int i = 0; i < frontier_ids.size() - 1; ++i) {
    // Move to path to next cluster
    auto path_iter = frontier_indexer[frontier_ids[i]]->paths_.begin();
    int next_idx = frontier_ids[i + 1];
    for (int j = 0; j < next_idx; ++j) ++path_iter;
    path.insert(path.end(), path_iter->begin(), path_iter->end());
  }
}

void FrontierFinder::getFullCostMatrix(const Vector3d& cur_pos, const Vector3d& cur_vel,
    const Vector3d cur_yaw, Eigen::MatrixXd& mat) {
  // Use Asymmetric TSP
  int dimen = frontiers_.size();
  mat.resize(dimen + 1, dimen + 1);
  // std::cout << "mat size: " << mat.rows() << ", " << mat.cols() << std::endl;
  // Fill block for clusters
  int i = 1, j = 1;
  for (auto ftr : frontiers_) {
    for (auto cs : ftr.costs_) {
      // std::cout << "(" << i << ", " << j << ")"
      // << ", ";
      mat(i, j++) = cs;
    }
    ++i;
    j = 1;
  }
  // std::cout << "" << std::endl;

  // Fill block from current state to clusters
  mat.leftCols<1>().setZero();
  for (auto ftr : frontiers_) {
    // std::cout << "(0, " << j << ")"
    // << ", ";
    Viewpoint vj = ftr.viewpoints_.front();
    vector<Vector3d> path;
    mat(0, j++) =
        ViewNode::computeCost(cur_pos, vj.pos_, cur_yaw[0], vj.yaw_, cur_vel, cur_yaw[1], path);
  }
  // std::cout << "" << std::endl;
}

void FrontierFinder::getSwarmCostMatrix(const vector<Vector3d>& positions,
    const vector<Vector3d>& velocities, const vector<double> yaws, Eigen::MatrixXd& mat) {

  const int drone_num = positions.size();
  const int ftr_num = frontiers_.size();
  const int dimen = 1 + drone_num + ftr_num;
  mat = Eigen::MatrixXd::Zero(dimen, dimen);

  // Virtual depot to drones
  for (int i = 0; i < drone_num; ++i) {
    mat(0, 1 + i) = -1000;
    mat(1 + i, 0) = 1000;
  }
  // Virtual depot to frontiers
  for (int i = 0; i < ftr_num; ++i) {
    mat(0, 1 + drone_num + i) = 1000;
    mat(1 + drone_num + i, 0) = 0;
  }
  // Costs between drones
  for (int i = 0; i < drone_num; ++i) {
    for (int j = 0; j < drone_num; ++j) {
      mat(1 + i, 1 + j) = 10000;
    }
  }

  // Costs from drones to frontiers
  for (int i = 0; i < drone_num; ++i) {
    int j = 0;
    for (auto ftr : frontiers_) {
      Viewpoint vj = ftr.viewpoints_.front();
      vector<Vector3d> path;
      mat(1 + i, 1 + drone_num + j) =
          ViewNode::computeCost(positions[i], vj.pos_, yaws[0], vj.yaw_, velocities[i], 0.0, path);
      mat(1 + drone_num + j, 1 + i) = 0;
      ++j;
    }
  }
  // Costs between frontiers
  int i = 0, j = 0;
  for (auto ftr : frontiers_) {
    for (auto cs : ftr.costs_) {
      mat(1 + drone_num + i, 1 + drone_num + j) = cs;
      ++j;
    }
    ++i;
    j = 0;
  }
  // Diag
  for (int i = 0; i < dimen; ++i) {
    mat(i, i) = 1000;
  }

  // std::cout << "mat: " << std::endl;
  // std::cout << mat << std::endl;
}

void FrontierFinder::getSwarmCostMatrix(const vector<Vector3d>& positions,
    const vector<Vector3d>& velocities, const vector<double>& yaws, const vector<int>& ftr_ids,
    const vector<Eigen::Vector3d>& grid_pos, Eigen::MatrixXd& mat) {

  const int drone_num = positions.size();
  const int ftr_num = ftr_ids.size();
  const int all_ftr_num = frontiers_.size();
  int dimen = 1 + drone_num + ftr_num;
  if (!grid_pos.empty()) dimen += 1;

  mat = Eigen::MatrixXd::Zero(dimen, dimen);

  // The upstream implementation first constructs the complete matrix for
  // every frontier in the map, then copies this HGrid's subset.  Only the
  // current-state edges require fresh A* searches; all frontier-to-frontier
  // edges are already cached.  Build the requested subset directly so the
  // result is identical while avoiding A* calls for unrelated HGrids.
  vector<Viewpoint> selected_views;
  vector<vector<double>> selected_cost_rows;
  selected_views.reserve(ftr_num);
  selected_cost_rows.reserve(ftr_num);
  for (const int id : ftr_ids) {
    if (id < 0 || id >= all_ftr_num) {
      ROS_ERROR("Invalid frontier id %d for matrix of size %d", id, all_ftr_num);
      mat.resize(0, 0);
      return;
    }
    auto frontier = frontiers_.begin();
    std::advance(frontier, id);
    selected_views.push_back(frontier->viewpoints_.front());
    selected_cost_rows.emplace_back(frontier->costs_.begin(), frontier->costs_.end());
  }

  // Virtual depot to drones
  for (int i = 0; i < drone_num; ++i) {
    mat(0, 1 + i) = -1000;
    mat(1 + i, 0) = 1000;
  }
  // Virtual depot to frontiers
  for (int i = 0; i < ftr_num; ++i) {
    mat(0, 1 + drone_num + i) = 1000;
    mat(1 + drone_num + i, 0) = 0;
  }
  // Costs between drones
  for (int i = 0; i < drone_num; ++i) {
    for (int j = 0; j < drone_num; ++j) {
      mat(1 + i, 1 + j) = 10000;
    }
  }

  // Costs from drones to frontiers
  for (int i = 0; i < drone_num; ++i) {
    for (int j = 0; j < ftr_num; ++j) {
      vector<Vector3d> path;
      const Viewpoint& view = selected_views[j];
      mat(1 + i, 1 + drone_num + j) =
          ViewNode::computeCost(positions[i], view.pos_, yaws[0], view.yaw_,
              velocities[i], 0.0, path);
      mat(1 + drone_num + j, 1 + i) = 0;
    }
  }
  // Costs between frontiers
  for (int i = 0; i < ftr_num; ++i) {
    for (int j = 0; j < ftr_num; ++j) {
      mat(1 + drone_num + i, 1 + drone_num + j) =
          selected_cost_rows[i][ftr_ids[j]];
    }
  }
  // Diag
  for (int i = 0; i < dimen; ++i) {
    mat(i, i) = 1000;
  }

  // Consider next grid in global tour
  if (!grid_pos.empty()) {
    // Depot, 1000, -1000
    mat(0, 1 + drone_num + ftr_num) = 1000;
    mat(1 + drone_num + ftr_num, 0) = -1000;

    // Drone
    for (int i = 0; i < drone_num; ++i) {
      mat(1 + i, 1 + drone_num + ftr_num) = 1000;
      mat(1 + drone_num + ftr_num, 1 + i) = 1000;
    }

    // Frontier
    vector<Eigen::Vector3d> path;
    Eigen::Vector3d next_grid = grid_pos[0];

    for (int i = 0; i < ftr_num; ++i) {
      double cost = ViewNode::computeCost(
          next_grid, selected_views[i].pos_, 0, 0, Eigen::Vector3d::Zero(), 0, path);
      mat(1 + drone_num + i, 1 + drone_num + ftr_num) = cost;
      mat(1 + drone_num + ftr_num, 1 + drone_num + i) = cost;
    }
  }
}

int FrontierFinder::computeGainOfView(const Eigen::Vector3d& pos, const double& yaw) {
  percep_utils_->setPose(pos, yaw);

  // Compute info gain in the FOV
  Eigen::Vector3d bmin, bmax;
  percep_utils_->getFOVBoundingBox(bmin, bmax);

  int gain = 0;
  for (double sx = bmin[0]; sx <= bmax[0]; sx += 0.8) {
    for (double sy = bmin[1]; sy <= bmax[1]; sy += 0.8) {
      for (double sz = bmin[2]; sz <= bmax[2]; sz += 0.8) {
        Eigen::Vector3d sample(sx, sy, sz);
        if (percep_utils_->insideFOV(sample) &&
            edt_env_->sdf_map_->getOccupancy(sample) == SDFMap::UNKNOWN)
          ++gain;
      }
    }
  }
  return gain;
}

int FrontierFinder::deleteFrontiers(const vector<uint16_t>& ids) {
  // Remove uninformative cluster and update cost matrix of other clusters
  auto tmp_ids = ids;
  sort(tmp_ids.begin(), tmp_ids.end(), greater<int>());

  removed_ids_.clear();
  int rmv_idx = 0;
  for (auto iter = frontiers_.begin(); iter != frontiers_.end() && !tmp_ids.empty();) {
    if (iter->id_ == tmp_ids.back()) {
      // Reset flag and remove cluster from list
      iter = frontiers_.erase(iter);
      removed_ids_.push_back(rmv_idx);
      tmp_ids.pop_back();
      // Eigen::Vector3i idx;
      // for (auto cell : iter->cells_) {
      //   edt_env_->sdf_map_->posToIndex(cell, idx);
      //   frontier_flag_[toadr(idx)] = 0;
      // }
    } else {
      ++rmv_idx;
      ++iter;
    }
  }

  // Reset ids
  int idx = 0;
  for (auto& ft : frontiers_) {
    ft.id_ = idx++;
  }

  return frontiers_.size();
}

int FrontierFinder::addFrontiers(const vector<pair<Eigen::Vector3d, double>>& views) {

  first_new_ftr_ = frontiers_.end();
  for (auto v : views) {
    // Create new frontier with only viewpoint info
    Viewpoint view;
    view.pos_ = v.first;
    view.yaw_ = v.second;

    Frontier ftr;
    ftr.viewpoints_.push_back(view);

    auto inserted = frontiers_.insert(frontiers_.end(), ftr);
    if (first_new_ftr_ == frontiers_.end()) first_new_ftr_ = inserted;
  }

  int idx = 0;
  for (auto& ft : frontiers_) {
    ft.id_ = idx++;
  }
  for (auto ftr : frontiers_) {
    std::cout << ftr.viewpoints_.size() << ", ";
  }
  std::cout << "" << std::endl;
}

void FrontierFinder::findViewpoints(
    const Vector3d& sample, const Vector3d& ftr_avg, vector<Viewpoint>& vps) {
  if (!edt_env_->sdf_map_->isInBox(sample) ||
      edt_env_->sdf_map_->getInflateOccupancy(sample) == 1 || isNearUnknown(sample))
    return;

  double left_angle_, right_angle_, vertical_angle_, ray_length_;

  // Central yaw is determined by frontier's average position and sample
  auto dir = ftr_avg - sample;
  double hc = atan2(dir[1], dir[0]);

  vector<int> slice_gains;
  // Evaluate info gain of different slices
  for (double phi_h = -M_PI_2; phi_h <= M_PI_2 + 1e-3; phi_h += M_PI / 18) {
    // Compute gain of one slice
    int gain = 0;
    for (double phi_v = -vertical_angle_; phi_v <= vertical_angle_; phi_v += vertical_angle_ / 3) {
      // Find endpoint of a ray
      Vector3d end;
      end[0] = sample[0] + ray_length_ * cos(phi_v) * cos(hc + phi_h);
      end[1] = sample[1] + ray_length_ * cos(phi_v) * sin(hc + phi_h);
      end[2] = sample[2] + ray_length_ * sin(phi_v);

      // Do raycasting to check info gain
      Vector3i idx;
      raycaster_->input(sample, end);
      while (raycaster_->nextId(idx)) {
        // Hit obstacle, stop the ray
        if (edt_env_->sdf_map_->getInflateOccupancy(idx) == 1 || !edt_env_->sdf_map_->isInBox(idx))
          break;
        // Count number of unknown cells
        if (edt_env_->sdf_map_->getOccupancy(idx) == SDFMap::UNKNOWN) ++gain;
      }
    }
    slice_gains.push_back(gain);
  }

  // Sum up slices' gain to get different yaw's gain
  vector<pair<double, int>> yaw_gains;
  for (int i = 0; i < 6; ++i)  // [-90,-10]-> [10,90], delta_yaw = 20, 6 groups
  {
    double yaw = hc - M_PI_2 + M_PI / 9.0 * i + right_angle_;
    int gain = 0;
    for (int j = 2 * i; j < 2 * i + 9; ++j)  // 80 degree hFOV, 9 slices
      gain += slice_gains[j];
    yaw_gains.push_back(make_pair(yaw, gain));
  }

  // Get several yaws with highest gain
  vps.clear();
  sort(yaw_gains.begin(), yaw_gains.end(),
      [](const pair<double, int>& p1, const pair<double, int>& p2) {
        return p1.second > p2.second;
      });
  for (int i = 0; i < 3; ++i) {
    if (yaw_gains[i].second < min_visib_num_) break;
    Viewpoint vp = { sample, yaw_gains[i].first, yaw_gains[i].second };
    while (vp.yaw_ < -M_PI) vp.yaw_ += 2 * M_PI;
    while (vp.yaw_ > M_PI) vp.yaw_ -= 2 * M_PI;
    vps.push_back(vp);
  }
}

// Sample viewpoints around frontier's average position, check coverage to the frontier cells
void FrontierFinder::sampleViewpoints(Frontier& frontier) {

  // Evaluate sample viewpoints on circles, find ones that cover most cells
  for (double rc = candidate_rmin_, dr = (candidate_rmax_ - candidate_rmin_) / candidate_rnum_;
       rc <= candidate_rmax_ + 1e-3; rc += dr)
    for (double phi = -M_PI; phi < M_PI; phi += candidate_dphi_) {
      const Vector3d sample_pos = frontier.average_ + rc * Vector3d(cos(phi), sin(phi), 0);

      // Qualified viewpoint is in bounding box and in safe region
      if (!edt_env_->sdf_map_->isInBox(sample_pos) ||
          edt_env_->sdf_map_->getInflateOccupancy(sample_pos) == 1 || isNearUnknown(sample_pos) ||
          isNearObstacle(sample_pos))
        continue;

      // Compute average yaw
      auto& cells = frontier.filtered_cells_;
      Eigen::Vector3d ref_dir = (cells.front() - sample_pos).normalized();
      double avg_yaw = 0.0;
      for (int i = 1; i < cells.size(); ++i) {
        Eigen::Vector3d dir = (cells[i] - sample_pos).normalized();
        double yaw = acos(dir.dot(ref_dir));
        if (ref_dir.cross(dir)[2] < 0) yaw = -yaw;
        avg_yaw += yaw;
      }
      avg_yaw = avg_yaw / cells.size() + atan2(ref_dir[1], ref_dir[0]);
      wrapYaw(avg_yaw);
      // Keep the original frontier-visibility requirement, then continue the
      // same rays beyond the boundary.  A view of a wall-side frontier with no
      // observable unknown behind it is no longer treated as informative.
      vector<int> unknown_voxels;
      const int visib_num =
          evaluateViewpoint(sample_pos, avg_yaw, cells, &unknown_voxels);
      // Unknown gain is a ranking score only (see the viewpoint sort); a
      // low-gain but geometrically valid view is kept rather than dropped.
      if (visib_num > min_visib_num_) {
        Viewpoint vp;
        vp.pos_ = sample_pos;
        vp.yaw_ = avg_yaw;
        vp.visib_num_ = visib_num;
        vp.unknown_gain_ = unknown_voxels.size();
        vp.unknown_voxel_addrs_ = std::move(unknown_voxels);
        frontier.viewpoints_.push_back(std::move(vp));
      }
      // }
    }
}

bool FrontierFinder::isFrontierCovered() {
  Vector3d update_min, update_max;
  edt_env_->sdf_map_->getUpdatedBox(update_min, update_max);

  // Bounding box of external chunks
  vector<Eigen::Vector3d> chunk_mins, chunk_maxs;
  edt_env_->sdf_map_->mm_->getChunkBoxes(chunk_mins, chunk_maxs, false);

  vector<Eigen::Vector3d> mins, maxs;
  mins.push_back(update_min);
  mins.insert(mins.end(), chunk_mins.begin(), chunk_mins.end());
  maxs.push_back(update_max);
  maxs.insert(maxs.end(), chunk_maxs.begin(), chunk_maxs.end());

  if (!next_frontier_valid_ ||
      !haveAnyOverlap(next_frontier_.box_min_, next_frontier_.box_max_, mins, maxs))
    return false;

  const int change_thresh =
      max(1, int(min_view_finish_fraction_ * next_frontier_.cells_.size()));
  int change_num = 0;
  for (auto cell : next_frontier_.cells_) {
    Eigen::Vector3i idx;
    edt_env_->sdf_map_->posToIndex(cell, idx);
    if (!(knownfree(idx) && isNeighborUnknown(idx)) && ++change_num >= change_thresh)
      return true;
  }
  return false;
}

void FrontierFinder::setNextFrontier(const int& id) {
  next_frontier_valid_ = false;
  for (const auto& frontier : frontiers_) {
    if (frontier.id_ != id) continue;
    next_frontier_.cells_ = frontier.cells_;
    next_frontier_.box_min_ = frontier.box_min_;
    next_frontier_.box_max_ = frontier.box_max_;
    next_frontier_valid_ = true;
    return;
  }
}

bool FrontierFinder::isNearUnknown(const Eigen::Vector3d& pos) {
  const int vox_num = floor(min_candidate_clearance_ / resolution_);
  for (int x = -vox_num; x <= vox_num; ++x)
    for (int y = -vox_num; y <= vox_num; ++y)
      for (int z = -1; z <= 1; ++z) {
        Eigen::Vector3d vox;
        vox << pos[0] + x * resolution_, pos[1] + y * resolution_, pos[2] + z * resolution_;
        if (edt_env_->sdf_map_->getOccupancy(vox) == SDFMap::UNKNOWN) return true;
      }
  return false;
}

int FrontierFinder::countVisibleCells(
    const Eigen::Vector3d& pos, const double& yaw, const vector<Eigen::Vector3d>& cluster) {
  return evaluateViewpoint(pos, yaw, cluster, nullptr);
}

int FrontierFinder::evaluateViewpoint(const Eigen::Vector3d& pos, const double& yaw,
    const vector<Eigen::Vector3d>& cluster, vector<int>* unknown_voxel_addrs) {
  percep_utils_->setPose(pos, yaw);
  int visib_num = 0;
  Eigen::Vector3i idx;
  std::unordered_set<int> unique_unknown;
  const int max_gain_rays = std::max(1, unknown_gain_max_rays_);
  const int gain_stride = unknown_voxel_addrs == nullptr || cluster.empty()
                              ? 1
                              : std::max(1, static_cast<int>(std::ceil(
                                                static_cast<double>(cluster.size()) /
                                                static_cast<double>(max_gain_rays))));

  for (size_t cell_index = 0; cell_index < cluster.size(); ++cell_index) {
    const auto& cell = cluster[cell_index];
    // Check if frontier cell is inside FOV
    if (!percep_utils_->insideFOV(cell)) continue;

    // Check if frontier cell is visible (not occulded by obstacles)
    raycaster_->input(cell, pos);
    bool visib = true;
    while (raycaster_->nextId(idx)) {
      if (edt_env_->sdf_map_->getInflateOccupancy(idx) == 1 ||
          edt_env_->sdf_map_->getOccupancy(idx) == SDFMap::UNKNOWN) {
        visib = false;
        break;
      }
    }
    if (!visib) continue;
    ++visib_num;

    if (unknown_voxel_addrs == nullptr ||
        static_cast<int>(cell_index % gain_stride) != 0 ||
        unknown_gain_max_dist_ <= resolution_)
      continue;

    const Eigen::Vector3d ray = cell - pos;
    const double frontier_distance = ray.norm();
    if (frontier_distance < resolution_ ||
        frontier_distance >= unknown_gain_max_dist_ - 0.5 * resolution_)
      continue;

    const Eigen::Vector3d direction = ray / frontier_distance;
    const Eigen::Vector3d ray_start = cell + 0.51 * resolution_ * direction;
    const Eigen::Vector3d ray_end = pos + unknown_gain_max_dist_ * direction;
    raycaster_->input(ray_start, ray_end);
    while (raycaster_->nextId(idx)) {
      if (!edt_env_->sdf_map_->isInBox(idx)) break;
      const int state = edt_env_->sdf_map_->getOccupancy(idx);
      // A planner-inferred wall closure terminates the expected ray just like
      // a measured wall. Otherwise a one-cell corner crack can expose hundreds
      // of physically occluded UNKNOWN cells to the gain estimator.
      if (state == SDFMap::OCCUPIED || isWallGapClosed(idx)) break;
      if (state == SDFMap::UNKNOWN) unique_unknown.insert(toadr(idx));
    }
  }

  if (unknown_voxel_addrs != nullptr) {
    unknown_voxel_addrs->assign(unique_unknown.begin(), unique_unknown.end());
    std::sort(unknown_voxel_addrs->begin(), unknown_voxel_addrs->end());
  }
  return visib_num;
}

void FrontierFinder::downsample(
    const vector<Eigen::Vector3d>& cluster_in, vector<Eigen::Vector3d>& cluster_out) {
  // downsamping cluster
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloudf(new pcl::PointCloud<pcl::PointXYZ>);
  for (auto cell : cluster_in) cloud->points.emplace_back(cell[0], cell[1], cell[2]);

  const double leaf_size = edt_env_->sdf_map_->getResolution() * down_sample_;
  pcl::VoxelGrid<pcl::PointXYZ> sor;
  sor.setInputCloud(cloud);
  sor.setLeafSize(leaf_size, leaf_size, leaf_size);
  sor.filter(*cloudf);

  cluster_out.clear();
  for (auto pt : cloudf->points) cluster_out.emplace_back(pt.x, pt.y, pt.z);
}

void FrontierFinder::wrapYaw(double& yaw) {
  while (yaw < -M_PI) yaw += 2 * M_PI;
  while (yaw > M_PI) yaw -= 2 * M_PI;
}

Eigen::Vector3i FrontierFinder::searchClearVoxel(const Eigen::Vector3i& pt) {
  queue<Eigen::Vector3i> init_que;
  vector<Eigen::Vector3i> nbrs;
  Eigen::Vector3i cur, start_idx;
  init_que.push(pt);
  // visited_flag_[toadr(pt)] = 1;

  while (!init_que.empty()) {
    cur = init_que.front();
    init_que.pop();
    if (knownfree(cur)) {
      start_idx = cur;
      break;
    }

    nbrs = sixNeighbors(cur);
    for (auto nbr : nbrs) {
      int adr = toadr(nbr);
      // if (visited_flag_[adr] == 0)
      // {
      //   init_que.push(nbr);
      //   visited_flag_[adr] = 1;
      // }
    }
  }
  return start_idx;
}

inline vector<Eigen::Vector3i> FrontierFinder::sixNeighbors(const Eigen::Vector3i& voxel) {
  vector<Eigen::Vector3i> neighbors(6);
  Eigen::Vector3i tmp;

  tmp = voxel - Eigen::Vector3i(1, 0, 0);
  neighbors[0] = tmp;
  tmp = voxel + Eigen::Vector3i(1, 0, 0);
  neighbors[1] = tmp;
  tmp = voxel - Eigen::Vector3i(0, 1, 0);
  neighbors[2] = tmp;
  tmp = voxel + Eigen::Vector3i(0, 1, 0);
  neighbors[3] = tmp;
  tmp = voxel - Eigen::Vector3i(0, 0, 1);
  neighbors[4] = tmp;
  tmp = voxel + Eigen::Vector3i(0, 0, 1);
  neighbors[5] = tmp;

  return neighbors;
}

inline vector<Eigen::Vector3i> FrontierFinder::tenNeighbors(const Eigen::Vector3i& voxel) {
  vector<Eigen::Vector3i> neighbors(10);
  Eigen::Vector3i tmp;
  int count = 0;

  for (int x = -1; x <= 1; ++x) {
    for (int y = -1; y <= 1; ++y) {
      if (x == 0 && y == 0) continue;
      tmp = voxel + Eigen::Vector3i(x, y, 0);
      neighbors[count++] = tmp;
    }
  }
  neighbors[count++] = tmp - Eigen::Vector3i(0, 0, 1);
  neighbors[count++] = tmp + Eigen::Vector3i(0, 0, 1);
  return neighbors;
}

inline vector<Eigen::Vector3i> FrontierFinder::allNeighbors(const Eigen::Vector3i& voxel) {
  vector<Eigen::Vector3i> neighbors(26);
  Eigen::Vector3i tmp;
  int count = 0;
  for (int x = -1; x <= 1; ++x)
    for (int y = -1; y <= 1; ++y)
      for (int z = -1; z <= 1; ++z) {
        if (x == 0 && y == 0 && z == 0) continue;
        tmp = voxel + Eigen::Vector3i(x, y, z);
        neighbors[count++] = tmp;
      }
  return neighbors;
}

void FrontierFinder::updateWallGapMask() {
  // Planner-only wall mask: every OCCUPIED voxel is dilated by a square
  // (2r+1) x (2r+1) window in XY (r = wall_gap_max_cells_, so r = 2 gives the
  // requested 5 x 5 footprint).  Masked voxels are neither known-free frontier
  // cells nor unknown exploration boundaries, and unknown-gain rays terminate
  // on them like a wall.  This replaces the earlier gap-bridging / L-corner
  // closing heuristics, which could brick up partially observed doorways.
  if (wall_gap_max_cells_ <= 0) return;
  std::fill(wall_gap_mask_.begin(), wall_gap_mask_.end(), 0);

  Eigen::Vector3d origin, size;
  edt_env_->sdf_map_->getRegion(origin, size);
  const Eigen::Vector3i voxel_num(
      static_cast<int>(std::ceil(size.x() / resolution_)),
      static_cast<int>(std::ceil(size.y() / resolution_)),
      static_cast<int>(std::ceil(size.z() / resolution_)));
  const int r = wall_gap_max_cells_;

  int closed_count = 0;
  for (int z = 0; z < voxel_num.z(); ++z) {
    for (int x = 0; x < voxel_num.x(); ++x) {
      for (int y = 0; y < voxel_num.y(); ++y) {
        const Eigen::Vector3i center(x, y, z);
        if (!edt_env_->sdf_map_->isInBox(center) ||
            edt_env_->sdf_map_->getOccupancy(center) != SDFMap::OCCUPIED)
          continue;
        for (int dx = -r; dx <= r; ++dx) {
          for (int dy = -r; dy <= r; ++dy) {
            const Eigen::Vector3i cell = center + Eigen::Vector3i(dx, dy, 0);
            if (!edt_env_->sdf_map_->isInBox(cell)) continue;
            const int address = toadr(cell);
            if (wall_gap_mask_[address] == 0) {
              wall_gap_mask_[address] = 1;
              ++closed_count;
            }
          }
        }
      }
    }
  }

  ROS_INFO_THROTTLE(5.0,
      "RACER_METRIC wall_mask_dilation masked_cells=%d radius_cells=%d",
      closed_count, wall_gap_max_cells_);
}

bool FrontierFinder::isWallGapClosed(const Eigen::Vector3i& idx) const {
  if (!edt_env_->sdf_map_->isInBox(idx)) return false;
  const int address = edt_env_->sdf_map_->toAddress(idx);
  return address >= 0 && address < static_cast<int>(wall_gap_mask_.size()) &&
         wall_gap_mask_[address] != 0;
}

void FrontierFinder::blockViewpoint(const Eigen::Vector3d& pos, double until) {
  const double now = ros::Time::now().toSec();
  for (auto it = blocked_viewpoints_.begin(); it != blocked_viewpoints_.end();) {
    if (it->second <= now) it = blocked_viewpoints_.erase(it);
    else ++it;
  }
  blocked_viewpoints_.emplace_back(pos, until);
}

bool FrontierFinder::isViewpointBlocked(const Eigen::Vector3d& pos) const {
  const double now = ros::Time::now().toSec();
  for (const auto& item : blocked_viewpoints_) {
    if (item.second <= now) continue;
    if ((item.first.head<2>() - pos.head<2>()).norm() < viewpoint_block_radius_) return true;
  }
  return false;
}

bool FrontierFinder::isNearObstacle(const Eigen::Vector3d& pos) {
  // Euclidean XY clearance test against OCCUPIED voxels on the map layer(s).
  if (min_candidate_obstacle_clearance_ <= 0.0) return false;
  const int r = static_cast<int>(std::ceil(min_candidate_obstacle_clearance_ / resolution_));
  const double limit2 = min_candidate_obstacle_clearance_ * min_candidate_obstacle_clearance_;
  Eigen::Vector3i idx;
  edt_env_->sdf_map_->posToIndex(pos, idx);
  for (int dx = -r; dx <= r; ++dx) {
    for (int dy = -r; dy <= r; ++dy) {
      const double d2 = (dx * dx + dy * dy) * resolution_ * resolution_;
      if (d2 > limit2) continue;
      const Eigen::Vector3i nbr = idx + Eigen::Vector3i(dx, dy, 0);
      if (!edt_env_->sdf_map_->isInMap(nbr)) continue;
      if (edt_env_->sdf_map_->getOccupancy(nbr) == SDFMap::OCCUPIED) return true;
    }
  }
  return false;
}

inline bool FrontierFinder::isNeighborUnknown(const Eigen::Vector3i& voxel) {
  // At least one neighbor is unknown
  auto nbrs = sixNeighbors(voxel);
  for (auto nbr : nbrs) {
    if (!edt_env_->sdf_map_->isInBox(nbr)) continue;
    if (edt_env_->sdf_map_->getOccupancy(nbr) != SDFMap::UNKNOWN) continue;
    // UNKNOWN inferred to be a tiny wall break is not an exploration boundary.
    if (isWallGapClosed(nbr)) continue;
    return true;
  }
  return false;
}

inline int FrontierFinder::toadr(const Eigen::Vector3i& idx) {
  return edt_env_->sdf_map_->toAddress(idx);
}

inline bool FrontierFinder::knownfree(const Eigen::Vector3i& idx) {
  if (edt_env_->sdf_map_->getOccupancy(idx) != SDFMap::FREE) return false;
  return !isWallGapClosed(idx);
}

inline bool FrontierFinder::inmap(const Eigen::Vector3i& idx) {
  return edt_env_->sdf_map_->isInMap(idx);
}

}  // namespace fast_planner
