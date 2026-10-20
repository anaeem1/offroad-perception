# Operational Design Domain (ODD)

What this perception stack is validated for, what it is not, and how it behaves at the edges. Every limit
below links to the measurement or decision record behind it. Outside the ODD the system is not claimed to
be safe; the health monitor and contract tell the consumer when it is degraded (`interface_contract.md`).

## Machine and sensors (as validated)
| Item | Validated configuration | Source |
|---|---|---|
| Platform | Clearpath Warthog class wheeled UGV, ~1.5 m × 1.4 m | RELLIS-3D |
| Lidar | Ouster OS1-64, 10 Hz, 64 × 2048, roof-mounted (~0.85 m above `base_link`) | `config/sensors.yaml` |
| Camera | Basler acA1920 1920 × 1200 Bayer, 10 Hz, **37.7° horizontal FOV, forward only** | ADR-0002 |
| Sync | None in hardware; per-pair matching to the lidar's sweep of the camera FOV | `camera_matching_00001.json` |
| Compute | Jetson Orin NX 8 GB, JetPack 6.0, TensorRT 8.6 | `jetson_info.txt` |

## Environment
| Dimension | Inside the ODD | Outside / not validated |
|---|---|---|
| Terrain | Off-road trails and open fields: grass, dirt, mud, puddles, gravel/rubble, concrete pads, bushes, trees (RELLIS-3D classes) | Paved roads with traffic, urban scenes, construction-site machinery not represented in training data |
| Lighting | Daylight (all RELLIS-3D sequences) | Night, low sun into the camera, strong shadows not characterized |
| Weather | Dry, clear | Rain, fog, dust, snow (lidar and camera both degrade; untested) |
| Water | Shallow puddles (cost 80) | Deep water: classified lethal by semantics, but depth is unobservable |
| Slopes | Gentle (local-minimum ground estimate) | Steep slopes and step-downs can read as obstacles or be missed (ADR-0007) |
| Dynamic agents | People walking near the machine (tracked) | Vehicles: tracked poorly (recall 0.20, `tracking_00001_test.json`); fast movers untested |

## Operating limits
| Limit | Value | Why / evidence |
|---|---|---|
| Speed | **≤ 2 m/s** (RELLIS-3D median 1.4 m/s) | Fusion, memory and tracking validated only at these speeds; motion compensation untested above |
| Perception range | Useful to **~15 m**; cells beyond are mostly UNKNOWN | `grid_coverage_00001.json`, ADR-0007 |
| Near field | Lidar blind within ~3 m; covered by **5 s static grid memory** while moving | ADR-0008 |
| Camera coverage | Semantics refined only inside the forward 37.7° wedge; elsewhere lidar-only | ADR-0002 |
| Output age | Sweep-start stamp to output; bound under review: 150 ms (v0.2) vs 250 ms proposed | ADR-0005 |
| Obstacle classes | Person, vehicle, pole, log, barrier, generic object (height check) | `config/class_costs.yaml`, ADR-0003 |

## Known limitations that shape the ODD
- **Standstill blind zone:** after more than 5 s stationary, grid memory expires and the near field returns
  to UNKNOWN. An object placed while stopped cannot be seen (ADR-0008; hazard H-3).
- **Camera-only operation is not supported:** without lidar there is no geometry, so the system enters
  SAFE_STOP instead of `MODE_CAMERA_ONLY` (`health_monitor_node.cpp`).
- **Ground-truth caveat:** RELLIS-3D lidar labels mark ground where people walked as `person`, so person
  metrics are pessimistic (ADR-0009).
- **Single dataset:** all accuracy numbers come from one site and one platform. Transfer to construction sites
  needs new data for training, and new measurements of every limit above.
