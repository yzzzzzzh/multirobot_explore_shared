
int32 from_drone_id
int32 to_drone_id

float64 stamp
uint64 expected_ego_epoch
uint64 expected_other_epoch
uint64 assignment_epoch
# Pairwise ACVRP exchanges full 3-D HGrid IDs.  int8 overflows above 127.
int32[] ego_ids
int32[] other_ids

# Optional diagnostic hint retained for wire compatibility with older local
# runs.  RACER task ownership is HGrid-ID based: absence of these arrays must
# never block an otherwise valid pairwise allocation.  The receiver normally
# regenerates viewpoints from its own merged occupancy map.
int32[] evidence_grid_ids
geometry_msgs/Point[] evidence_viewpoints
float64[] evidence_yaws
int32[] evidence_visible_cells
float64[] evidence_lease_until
uint64[] evidence_versions
