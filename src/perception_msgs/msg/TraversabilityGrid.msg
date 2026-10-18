# BEV traversability map in the machine frame (base_link), REP-103 axes.
std_msgs/Header header          # stamp = source lidar scan time, NOT publish time
float32 resolution              # meters per cell
uint32 width
uint32 height
geometry_msgs/Pose origin       # pose of cell (0,0) in header.frame_id
# Row-major, length = width*height
uint8[] cost                    # 0 = free ... 254 = lethal, 255 = UNKNOWN (never treat as free)
uint8[] confidence              # 0-255, fused agreement of camera + lidar evidence
uint8[] semantic_class          # dominant RELLIS-3D class id per cell
uint32 source_flags             # bitmask: 1 = lidar, 2 = camera, 4 = remembered static evidence (<= 5 s old, ADR-0008)
