std_msgs/Header header
SensorStatus[] sensors
uint8 MODE_NOMINAL=0
uint8 MODE_LIDAR_ONLY=1
uint8 MODE_CAMERA_ONLY=2        # reduced range; consumer should cap speed
uint8 MODE_SAFE_STOP=3          # no trustworthy perception; consumer must stop
uint8 mode
float32 end_to_end_latency_ms
