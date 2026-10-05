string sensor_id
uint8 OK=0
uint8 DEGRADED=1                # e.g. low point density, overexposure, partial occlusion
uint8 FAILED=2                  # stale or missing data beyond timeout
uint8 status
float32 data_age_ms
string reason
