uint32 track_id                 # stable across frames until track is deleted
uint8 semantic_class            # RELLIS-3D class id
float32 class_confidence
float32 existence_probability   # consumers should gate on this, not on presence in the array
geometry_msgs/Pose pose         # box center in header frame
geometry_msgs/Vector3 dimensions
geometry_msgs/Vector3 velocity
float64[36] pose_covariance
uint32 age_frames
uint8 STATE_TENTATIVE=0
uint8 STATE_CONFIRMED=1
uint8 STATE_COASTING=2          # predicted only, no measurement this frame
uint8 track_state
