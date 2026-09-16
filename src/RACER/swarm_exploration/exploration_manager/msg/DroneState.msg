int32 drone_id

# The 3-D HGrid can contain more than 128 cells.  int8 silently wrapped valid
# IDs (for example, 129 -> -127) and corrupted distributed task ownership.
int32[] grid_ids
# Per-route and per-task ownership versions. They allow every RACER process to
# reject stale pair-opt transactions and deterministically resolve duplicate
# ownership after asynchronous delivery.
uint64 assignment_epoch
uint64[] grid_epochs

# Per-vehicle failed-task quarantine. A released task may be assigned to a
# different vehicle immediately, but not back to this vehicle before expiry.
int32[] failed_grid_ids
float64[] failed_grid_until
float64 recent_attempt_time
float64 stamp

# only used for simulation
float32[] pos
float32[] vel
float32 yaw
