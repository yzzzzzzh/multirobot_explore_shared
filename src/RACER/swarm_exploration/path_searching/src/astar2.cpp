#include <path_searching/astar2.h>
#include <chrono>
#include <sstream>
#include <plan_env/sdf_map.h>

using namespace std;
using namespace Eigen;

namespace fast_planner {
Astar::Astar() {
}

Astar::~Astar() {
  for (int i = 0; i < allocate_num_; i++) delete path_node_pool_[i];
}

void Astar::init(ros::NodeHandle& nh, const EDTEnvironment::Ptr& env) {
  nh.param("astar/resolution_astar", resolution_, -1.0);
  nh.param("astar/lambda_heu", lambda_heu_, -1.0);
  nh.param("astar/max_search_time", max_search_time_, -1.0);
  nh.param("astar/allocate_num", allocate_num_, -1);

  tie_breaker_ = 1.0 + 1.0 / 1000;

  this->edt_env_ = env;

  /* ---------- map params ---------- */
  this->inv_resolution_ = 1.0 / resolution_;
  edt_env_->sdf_map_->getRegion(origin_, map_size_3d_);
  cout << "origin_: " << origin_.transpose() << endl;
  cout << "map size: " << map_size_3d_.transpose() << endl;

  path_node_pool_.resize(allocate_num_);
  for (int i = 0; i < allocate_num_; i++) {
    path_node_pool_[i] = new Node;
  }
  use_node_num_ = 0;
  iter_num_ = 0;
  early_terminate_cost_ = 0.0;
}

void Astar::setResolution(const double& res) {
  resolution_ = res;
  this->inv_resolution_ = 1.0 / resolution_;
}

int Astar::search(const Eigen::Vector3d& start_pt, const Eigen::Vector3d& end_pt, bool optimistic) {
  const int start_occupancy = edt_env_->sdf_map_->getOccupancy(start_pt);
  const int end_occupancy = edt_env_->sdf_map_->getOccupancy(end_pt);
  const bool start_inflated =
      edt_env_->sdf_map_->getInflateOccupancy(start_pt) == 1;
  const bool end_inflated =
      edt_env_->sdf_map_->getInflateOccupancy(end_pt) == 1;
  last_start_occupancy_ = start_occupancy;
  last_start_inflated_ = start_inflated;
  last_start_planner_clear_ =
      start_occupancy == SDFMap::FREE && !start_inflated;
  // Never append an endpoint inside an occupied/inflated voxel. The upstream
  // one-voxel reach test otherwise can append the exact unsafe goal after
  // reaching a free neighbour. A raw occupied start is handled below as a
  // one-way escape because the measured vehicle pose is known to be there.
  // A freshly received cloud can also lag the current pose by one map update
  // and leave only the start voxel UNKNOWN.  Treat that single voxel like an
  // unsafe start, but never allow the search to enter any other unknown voxel.
  if (end_occupancy == SDFMap::OCCUPIED || end_inflated ||
      (!optimistic && end_occupancy == SDFMap::UNKNOWN)) {
    return NO_PATH;
  }

  NodePtr cur_node = path_node_pool_[0];
  cur_node->parent = NULL;
  cur_node->position = start_pt;
  posToIndex(start_pt, cur_node->index);
  cur_node->escaping_inflation =
      start_inflated || start_occupancy == SDFMap::OCCUPIED ||
      start_occupancy == SDFMap::UNKNOWN;
  cur_node->g_score = 0.0;
  // If delayed map sharing puts the vehicle in a newly occupied/inflated
  // voxel, first run a goal-independent Dijkstra search to the nearest
  // normal-free voxel. A goal-directed heuristic can otherwise choose a
  // first step toward the obstacle (the raw-LiDAR brake then rejects it
  // forever). Once the nearest exit is popped, normal A* resumes there.
  const bool search_started_in_clearance = cur_node->escaping_inflation;
  bool inflation_escape_complete = !search_started_in_clearance;
  cur_node->f_score =
      cur_node->escaping_inflation
          ? 0.0
          : lambda_heu_ * getDiagHeu(cur_node->position, end_pt);
  if (cur_node->escaping_inflation) {
    ROS_WARN_THROTTLE(
        1.0,
        "A* start is not planner-clear (occ=%d inflated=%d); enabling one-way escape",
        start_occupancy, static_cast<int>(start_inflated));
  }

  Eigen::Vector3i end_index;
  posToIndex(end_pt, end_index);

  open_set_.push(cur_node);
  open_set_map_.insert(make_pair(cur_node->index, cur_node));
  use_node_num_ += 1;

  // max_search_time is a computation budget.  Using ROS time here makes the
  // wall-clock budget scale with Gazebo's real-time factor (and it can stop
  // advancing entirely while /clock is stalled).  A monotonic wall clock
  // keeps every A* call bounded independently of the simulation rate.
  const auto search_start = std::chrono::steady_clock::now();
  auto wallTimedOut = [&]() {
    if (max_search_time_ <= 0.0) return false;
    return std::chrono::duration<double>(
               std::chrono::steady_clock::now() - search_start)
               .count() >= max_search_time_;
  };
  auto terminateForTimeout = [&](const NodePtr& node) {
    early_terminate_cost_ =
        node->g_score + getDiagHeu(node->position, end_pt);
    return TIMEOUT;
  };

  /* ---------- search loop ---------- */
  while (!open_set_.empty()) {
    cur_node = open_set_.top();
    bool reset_after_nearest_escape = false;
    if (search_started_in_clearance && !inflation_escape_complete &&
        !cur_node->escaping_inflation) {
      // Because every node in the escape phase is ordered only by g_score,
      // this is the closest reachable exit from the inflated shell. Discard
      // alternative shell branches so the remainder cannot re-enter it.
      inflation_escape_complete = true;
      std::priority_queue<NodePtr, std::vector<NodePtr>, NodeComparator0>
          empty_queue;
      open_set_.swap(empty_queue);
      open_set_map_.clear();
      close_set_map_.clear();
      reset_after_nearest_escape = true;
    }
    bool reach_end = !cur_node->escaping_inflation &&
                     abs(cur_node->index(0) - end_index(0)) <= 1 &&
                     abs(cur_node->index(1) - end_index(1)) <= 1 &&
                     abs(cur_node->index(2) - end_index(2)) <= 1;
    if (reach_end) {
      backtrack(cur_node, end_pt);
      return REACH_END;
    }

    // Early termination if time up
    if (wallTimedOut()) return terminateForTimeout(cur_node);

    if (!reset_after_nearest_escape) {
      open_set_.pop();
      open_set_map_.erase(cur_node->index);
    }
    close_set_map_.insert(make_pair(cur_node->index, 1));
    iter_num_ += 1;

    Eigen::Vector3d cur_pos = cur_node->position;
    Eigen::Vector3d nbr_pos;
    Eigen::Vector3d step;

    for (double dx = -resolution_; dx <= resolution_ + 1e-3; dx += resolution_)
      for (double dy = -resolution_; dy <= resolution_ + 1e-3; dy += resolution_)
        for (double dz = -resolution_; dz <= resolution_ + 1e-3; dz += resolution_) {
          // A planner process can be descheduled while expanding one node.
          // Checking only at the outer while-loop allowed a nominal 5 ms
          // search to return hundreds of milliseconds later.  Bound each
          // three-dimensional neighbor expansion as well.
          if (wallTimedOut()) return terminateForTimeout(cur_node);
          step << dx, dy, dz;
          if (step.norm() < 1e-3) continue;
          nbr_pos = cur_pos + step;
          // Check safety
          if (!edt_env_->sdf_map_->isInBox(nbr_pos)) {
            // std::cout << "not in box" << std::endl;
            continue;
          }

          const int nbr_occupancy = edt_env_->sdf_map_->getOccupancy(nbr_pos);
          const bool nbr_inflated = edt_env_->sdf_map_->getInflateOccupancy(nbr_pos) == 1;
          // A vehicle can occasionally start inside the conservative
          // inflation shell after delayed map sharing.  It may leave that
          // shell through known-free voxels, but after reaching normal free
          // space it can never enter inflation again.
          const bool escape_step =
              cur_node->escaping_inflation && nbr_inflated &&
              nbr_occupancy == SDFMap::FREE;
          if (nbr_occupancy == SDFMap::OCCUPIED ||
              (nbr_inflated && !escape_step) ||
              (!optimistic && nbr_occupancy == SDFMap::UNKNOWN))
            continue;

          bool safe = true;
          Vector3d dir = nbr_pos - cur_pos;
          double len = dir.norm();
          dir.normalize();
          for (double l = 0.1; l <= len + 1e-2; l += 0.1) {
            if (wallTimedOut()) return terminateForTimeout(cur_node);
            Vector3d ckpt = cur_pos + l * dir;
            const int ckpt_occupancy = edt_env_->sdf_map_->getOccupancy(ckpt);
            const bool ckpt_inflated =
                edt_env_->sdf_map_->getInflateOccupancy(ckpt) == 1;
            Eigen::Vector3i ckpt_index;
            posToIndex(ckpt, ckpt_index);
            const bool start_voxel_checkpoint =
                cur_node->escaping_inflation &&
                (start_occupancy == SDFMap::OCCUPIED ||
                    start_occupancy == SDFMap::UNKNOWN) &&
                (ckpt_index.array() == path_node_pool_[0]->index.array()).all();
            const bool inflation_escape_checkpoint =
                cur_node->escaping_inflation && ckpt_inflated &&
                ckpt_occupancy == SDFMap::FREE;
            if ((ckpt_occupancy == SDFMap::OCCUPIED &&
                    !start_voxel_checkpoint) ||
                (ckpt_inflated && !inflation_escape_checkpoint &&
                    !start_voxel_checkpoint) ||
                (!optimistic && ckpt_occupancy == SDFMap::UNKNOWN &&
                    !start_voxel_checkpoint)) {
              // edt_env_->sdf_map_->getOccupancy(ckpt) == SDFMap::UNKNOWN
              safe = false;
              break;
            }
          }
          if (!safe) continue;

          // Check not in close set
          Eigen::Vector3i nbr_idx;
          posToIndex(nbr_pos, nbr_idx);
          if (close_set_map_.find(nbr_idx) != close_set_map_.end()) continue;

          NodePtr neighbor;
          double tmp_g_score = step.norm() + cur_node->g_score;
          auto node_iter = open_set_map_.find(nbr_idx);
          if (node_iter == open_set_map_.end()) {
            neighbor = path_node_pool_[use_node_num_];
            use_node_num_ += 1;
            if (use_node_num_ == allocate_num_) {
              cout << "run out of node pool." << endl;
              return NO_PATH;
            }
            neighbor->index = nbr_idx;
            neighbor->position = nbr_pos;
          } else if (tmp_g_score < node_iter->second->g_score) {
            neighbor = node_iter->second;
          } else
            continue;

          neighbor->parent = cur_node;
          neighbor->escaping_inflation =
              cur_node->escaping_inflation && nbr_inflated;
          neighbor->g_score = tmp_g_score;
          neighbor->f_score =
              (!inflation_escape_complete)
                  ? tmp_g_score
                  : tmp_g_score +
                        lambda_heu_ * getDiagHeu(nbr_pos, end_pt);
          open_set_.push(neighbor);
          open_set_map_[nbr_idx] = neighbor;
        }
  }
  // cout << "open set empty, no path!" << endl;
  // cout << "use node num: " << use_node_num_ << endl;
  // cout << "iter num: " << iter_num_ << endl;
  return NO_PATH;
}

double Astar::getEarlyTerminateCost() {
  return early_terminate_cost_;
}

void Astar::reset() {
  open_set_map_.clear();
  close_set_map_.clear();
  path_nodes_.clear();

  std::priority_queue<NodePtr, std::vector<NodePtr>, NodeComparator0> empty_queue;
  open_set_.swap(empty_queue);
  for (int i = 0; i < use_node_num_; i++) {
    path_node_pool_[i]->parent = NULL;
    path_node_pool_[i]->escaping_inflation = false;
  }
  use_node_num_ = 0;
  iter_num_ = 0;
}

double Astar::pathLength(const vector<Eigen::Vector3d>& path) {
  double length = 0.0;
  if (path.size() < 2) return length;
  for (int i = 0; i < path.size() - 1; ++i) length += (path[i + 1] - path[i]).norm();
  return length;
}

void Astar::backtrack(const NodePtr& end_node, const Eigen::Vector3d& end) {
  path_nodes_.push_back(end);
  path_nodes_.push_back(end_node->position);
  NodePtr cur_node = end_node;
  while (cur_node->parent != NULL) {
    cur_node = cur_node->parent;
    path_nodes_.push_back(cur_node->position);
  }
  reverse(path_nodes_.begin(), path_nodes_.end());
}

std::vector<Eigen::Vector3d> Astar::getPath() {
  return path_nodes_;
}

double Astar::getDiagHeu(const Eigen::Vector3d& x1, const Eigen::Vector3d& x2) {
  double dx = fabs(x1(0) - x2(0));
  double dy = fabs(x1(1) - x2(1));
  double dz = fabs(x1(2) - x2(2));
  double h;
  double diag = min(min(dx, dy), dz);
  dx -= diag;
  dy -= diag;
  dz -= diag;

  if (dx < 1e-4) {
    h = 1.0 * sqrt(3.0) * diag + sqrt(2.0) * min(dy, dz) + 1.0 * abs(dy - dz);
  }
  if (dy < 1e-4) {
    h = 1.0 * sqrt(3.0) * diag + sqrt(2.0) * min(dx, dz) + 1.0 * abs(dx - dz);
  }
  if (dz < 1e-4) {
    h = 1.0 * sqrt(3.0) * diag + sqrt(2.0) * min(dx, dy) + 1.0 * abs(dx - dy);
  }
  return tie_breaker_ * h;
}

double Astar::getManhHeu(const Eigen::Vector3d& x1, const Eigen::Vector3d& x2) {
  double dx = fabs(x1(0) - x2(0));
  double dy = fabs(x1(1) - x2(1));
  double dz = fabs(x1(2) - x2(2));
  return tie_breaker_ * (dx + dy + dz);
}

double Astar::getEuclHeu(const Eigen::Vector3d& x1, const Eigen::Vector3d& x2) {
  return tie_breaker_ * (x2 - x1).norm();
}

std::vector<Eigen::Vector3d> Astar::getVisited() {
  vector<Eigen::Vector3d> visited;
  for (int i = 0; i < use_node_num_; ++i) visited.push_back(path_node_pool_[i]->position);
  return visited;
}

void Astar::posToIndex(const Eigen::Vector3d& pt, Eigen::Vector3i& idx) {
  idx = ((pt - origin_) * inv_resolution_).array().floor().cast<int>();
}

}  // namespace fast_planner
