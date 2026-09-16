# Decentralized frontier assistance used only when the receiver has no
# coverable frontier in its own merged map.  Occupied/free voxels are not
# duplicated here: they are transferred by RACER's plan_env/ChunkData stream.

int32 from_drone_id
float64 stamp
uint64 frontier_epoch

# A short-lived claim prevents two idle vehicles from selecting the same peer
# frontier.  Zero means that this vehicle is not claiming a peer frontier.
uint64 claimed_signature
float64 claim_until

# Candidate arrays are aligned by index.  `reserved[i] != 0` means the source
# already selected that frontier in its current local planning round.
uint64[] signatures
uint8[] reserved
geometry_msgs/Point[] suggested_viewpoints
float64[] suggested_yaws
int32[] suggested_visible_cells

# Frontier cells are flattened to keep the ROS message compact.
# For candidate i use cells[cell_offsets[i]:cell_offsets[i + 1]].
int32[] cell_offsets
geometry_msgs/Point[] cells
