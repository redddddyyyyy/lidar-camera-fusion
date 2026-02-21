# LiDAR-Camera Fusion — Project Reference

## What This Project Does

Frustum-based LiDAR-camera sensor fusion pipeline for 3D object detection and tracking.
Designed to showcase multi-sensor perception, signal processing, and real-time filtering
skills relevant to robotics/surgical robotics roles (e.g. J&J MedTech internship).

### Full Pipeline
```
Camera image → YOLOv8 → 2D bounding boxes
                               ↓
LiDAR point cloud → Frustum extraction (calibration) → 3D point clusters
                               ↓
              PCA-oriented 3D bounding box + Kalman filter tracking
                               ↓
              ROS2 node → RViz2 visualization
```

---

## Repository Structure

```
lidar-camera-fusion/
├── src/
│   ├── calibration.py       # Camera intrinsics + LiDAR-camera extrinsics, projection
│   ├── lidar_processor.py   # Point cloud I/O, RANSAC ground removal, clustering, 3D bbox
│   ├── detector.py          # YOLOv8 wrapper (falls back to mock if ultralytics missing)
│   ├── fusion.py            # Frustum fusion + Kalman filter tracker
│   └── visualization.py     # 2D overlays, BEV, fusion summary plots
├── ros2_node/
│   ├── fusion_node.py       # ROS2 node: subscribes camera+lidar, publishes to RViz
│   └── launch_fusion.py     # ROS2 launch file
├── rviz/
│   └── fusion.rviz          # Pre-configured RViz2 display config
├── tests/
│   ├── test_calibration.py  # 9 tests: projection math, distortion, YAML roundtrip
│   ├── test_fusion.py       # 13 tests: Kalman filter, IoU, SensorFusion pipeline
│   └── test_lidar_processor.py  # 14 tests: clustering, RANSAC, PCA yaw
├── run_fusion.sh            # One-shot script to run the fusion node (no paste issues)
├── main.py                  # Standalone demo (no ROS2 needed)
└── requirements.txt
```

---

## Upgrades Implemented (vs Original Repo)

### 1. O(1) Clustering Fix (`src/lidar_processor.py`)
**Problem:** BFS neighbor lookup used `np.where(unique_voxels == neighbor_id)` inside the
loop — O(N) per step → O(N²) total. Completely unusable on dense point clouds.

**Fix:** Pre-built `id_to_uidx: dict` gives O(1) per lookup. Also replaced
`queue.pop(0)` (O(N)) with `collections.deque.popleft()` (O(1)).

### 2. PCA-Based Yaw Estimation (`src/lidar_processor.py` — `BoundingBox3D.from_points`)
**Problem:** All 3D boxes had `yaw=0.0` (pure axis-aligned bounding box). Objects in the
real world are not axis-aligned — boxes were consistently wrong in orientation.

**Fix:** PCA on the XY plane of each point cluster finds the dominant orientation.
The principal eigenvector gives yaw. The box is then fit in the rotated frame and
transformed back. Tighter, oriented boxes now reflect actual object heading.

### 3. RANSAC Horizontal Plane Check (`src/lidar_processor.py` — `remove_ground`)
**Problem:** RANSAC could pick any dominant plane — walls, ramps, parked cars — and
classify it as ground.

**Fix:** Added `abs(dot(normal, Z)) >= threshold` guard. Only planes whose normal
is within `ground_normal_tol` of straight-up are accepted as ground.

### 4. Distortion Correction (`src/calibration.py` — `project_points`)
**Problem:** Projection stored `D` (distortion coefficients) but never applied them.
Just did `K @ points_norm` which ignores all lens distortion.

**Fix:** Replaced manual projection with `cv2.projectPoints()` which applies the
full radial (k1,k2,k3) + tangential (p1,p2) distortion model. Zero-op for KITTI
(D=0), correct for real cameras.

### 5. Kalman Filter Tracker (`src/fusion.py` — replaces `TemporalFusion`)
**Problem:** `TemporalFusion` used greedy IoU matching with no motion model. Fast-moving
objects lost their track immediately. No velocity estimation.

**Fix:** `KalmanTracker` with per-object `KalmanTrack`:
- **State vector:** `[x, y, z, vx, vy, vz]` (position + velocity)
- **Observation:** `[x, y, z]` (3D centre from fusion)
- **Constant-velocity model** with tuned Q (process noise) and R (measurement noise)
- **Hungarian matching** via `scipy.optimize.linear_sum_assignment` (optimal vs greedy)
- **3D distance gating** — rejects matches > `dist_threshold` metres apart
- **Velocity arrows** published as RViz markers
- `TemporalFusion = KalmanTracker` alias kept for backward compatibility

### 6. Unit Tests (`tests/` — 36 tests, all passing)
```
tests/test_calibration.py      9 tests  — projection math, distortion, YAML I/O
tests/test_fusion.py          13 tests  — Kalman predict/update/convergence, IoU, pipeline
tests/test_lidar_processor.py 14 tests  — clustering correctness, RANSAC separation, PCA yaw
```
Run with: `python3 -m pytest tests/ -v`

### 7. ROS2 Node + RViz (`ros2_node/fusion_node.py`)
Subscribes to camera + LiDAR, publishes:
- `/fusion/image_annotated`  — camera with 2D boxes + depth labels
- `/fusion/points_colored`   — depth-coloured point cloud (jet colormap)
- `/fusion/markers`          — oriented 3D bounding boxes (LINE_LIST)
- `/fusion/tracks`           — Kalman velocity arrows (ARROW markers)

---

## Running the Full Stack

### Data
```
Bag: /home/reddy/project2_easy_simulated_data
  Topics:
    /sensing/camera/camera0/image_rect_color    (sensor_msgs/Image,       545 msgs)
    /sensing/lidar/top/outlier_filtered/pointcloud (sensor_msgs/PointCloud2, 502 msgs)
  Duration: ~55 seconds
  LiDAR frame: velodyne_top_base_link
```

### Terminal 1 — Play bag (must use --clock for sim time)
```bash
source /opt/ros/humble/setup.bash
ros2 bag play /home/reddy/project2_easy_simulated_data --loop --clock
```

### Terminal 2 — KISS-ICP odometry (generates odom → velodyne_top_base_link TF)
```bash
source /opt/ros/humble/setup.bash
source /home/reddy/kiss_ws/install/setup.bash
ros2 run kiss_icp kiss_icp_node --ros-args -p use_sim_time:=true -p pointcloud_topic:=/sensing/lidar/top/outlier_filtered/pointcloud -p lidar_frame:=velodyne_top_base_link -p odom_frame:=odom -p voxel_size:=0.15 -p min_range:=1.0 -p max_range:=30.0 -p deskew:=false
```

### Terminal 3 — Fusion node
```bash
bash /home/reddy/lidar-camera-fusion/run_fusion.sh
```
(run_fusion.sh sources ROS2, cds to repo, runs node with correct topic names + use_sim_time)

### Terminal 4 — RViz
```bash
source /opt/ros/humble/setup.bash
rviz2 -d /home/reddy/lidar-camera-fusion/rviz/fusion.rviz
```
**After RViz opens:** Set Fixed Frame → `odom`

### Order matters
Start Terminal 1 first (bag), then Terminal 2 (KISS-ICP needs data to initialize),
then Terminal 3 and 4 can start in any order.

---

## Errors Encountered and Solved

### E1: NumPy 2.x compatibility
**Error:** `AttributeError: _ARRAY_API not found` / `ImportError: numpy.core.multiarray failed to import`
**Cause:** System had NumPy 2.2 but some compiled packages expected NumPy 1.x ABI.
**Fix:** `pip install "numpy<2"`

### E2: CvBridge encoding error — rgb8
**Error:** `CvBridgeError: [rgb8] is not a color format. but [rgb8] The conversion does not make sense`
**Cause:** `desired_encoding='rgb8'` on a `bgr8` image fails in some cv_bridge versions.
**Fix:** Use `desired_encoding='passthrough'` then inspect `img_msg.encoding` and
convert manually with `cv2.cvtColor`.

### E3: CvBridge encoding error — 8UC4
**Error:** `CvBridgeError: encoding specified as rgb8, but image has incompatible type 8UC4`
**Cause:** Bag camera images are BGRA (4 channels), not BGR/RGB (3 channels).
**Fix:** Added channel-count check: if `raw.shape[2] == 4`, use `COLOR_BGRA2RGB`.

### E4: PointCloud2 structured dtype cast
**Error:** `TypeError: Cannot cast array data from dtype({'names': ['x','y','z'], 'offsets':[0,4,8], 'itemsize':16}) to dtype('float32')`
**Cause:** `pc2.read_points` returns a structured numpy array with named fields and
16-byte itemsize (Velodyne adds 4 bytes padding after xyz). Direct cast to float32 fails.
**Fix:** Detect structured array via `.dtype.names`, then `np.column_stack` the
individual `['x']`, `['y']`, `['z']` fields.

### E5: RViz jerking / jumping point cloud
**Cause:** Fixed Frame was set to `velodyne_top_base_link` (the sensor frame itself).
Each new scan places the whole world relative to the moving sensor → looks like jumping.
**Fix:** Set Fixed Frame → `odom`. KISS-ICP publishes `odom → velodyne_top_base_link`
TF, so in the `odom` frame the sensor moves smoothly through a stable world.

### E6: RViz "No tf data" / point cloud invisible
**Cause:** Bag message timestamps are Oct 2025 (`1760387xxx`). KISS-ICP was publishing
TF at current wall time Feb 2026 (`1771xxx`). TF lookup for Oct 2025 timestamps found
nothing → points couldn't be placed in `odom` frame.
**Fix:** Play bag with `--clock` flag + add `use_sim_time:=true` to all nodes.
This synchronises everyone to the bag's simulated clock.

### E7: Shell paste error with multi-line --ros-args
**Error:** `rclpy._rclpy_pybind11.UnknownROSArgsError: [' ']` / `-p: command not found`
**Cause:** Copy-pasting multi-line bash commands with `\` continuations into terminal
sometimes inserts spaces or drops the backslash.
**Fix:** Created `run_fusion.sh` — all args on a single line, run with `bash run_fusion.sh`.

### E8: Mock detector returning fixed detections
**Cause:** `ultralytics` not installed → `YOLODetector` falls back to `_mock_detect()`
which always returns 3 hardcoded boxes (2 cars + 1 person at fixed relative positions).
**Fix:** `pip3 install ultralytics` — restarts detector with real YOLOv8n inference.

### E9: KISS-ICP not found as ROS2 package
**Error:** `Package 'kiss_icp' not found`
**Cause:** `kiss_icp` is built in a separate workspace (`kiss_ws`), not the system ROS2.
**Fix:** `source /home/reddy/kiss_ws/install/setup.bash` before running kiss_icp node.

### E10: Zero detections on synthetic/simulated data
**Symptom:** After installing ultralytics, mock boxes disappeared but real YOLO returned
0 detections on many frames. Node logs showed `0 det → 0 fused → 0 tracked`.
**Cause:** Default confidence threshold was `0.45`. YOLOv8n trained on real-world COCO
images has a domain gap with synthetic/simulated environments — confidence scores are
lower even when objects are clearly visible.
**Fix:** Lowered confidence to `0.2` in `run_fusion.sh` (`-p confidence:=0.2`). YOLO now
detects cars in urban sim scenes. Note: abstract/non-COCO objects (floating rocks, stylised
trees) will never be detected regardless of threshold — they are not in the label set.

---

## Remaining TODOs / Future Improvements

- [ ] **Wrong-class detections on synthetic data** — YOLO sometimes misclassifies sim
      objects (e.g. rocks detected as sports ball, buildings as nothing). Root cause is
      domain gap: YOLOv8 trained on real COCO. Options: use a larger model (`yolov8s.pt`),
      fine-tune on sim data, or accept that detection quality is limited on this bag.
- [ ] **Real calibration file** — current default calibration is approximate. Load actual
      intrinsics/extrinsics from the bag's calibration data for accurate frustum extraction.
- [ ] **YOLOv8 model size** — default uses `yolov8n.pt` (nano). Try `yolov8s.pt` or
      `yolov8m.pt` for better detection at cost of speed.
- [ ] **Point cloud in RViz 3D view** — switch PointCloud2 display topic from raw bag
      topic to `/fusion/points_colored` for depth-coloured output.
- [ ] **Intensity-based filtering** — point cloud has intensity field; currently unused.
      Intensity profiles help distinguish retroreflective surfaces (lane markings, signs).
- [ ] **SORT/DeepSORT** — upgrade Kalman tracker with appearance features for better
      re-identification after occlusion.
- [ ] **Evaluation metrics** — add KITTI-format AP evaluation to quantify detection quality.
- [ ] **ROS2 package** — wrap as proper ament_python package with setup.py so it can be
      installed and launched via `ros2 launch lidar_camera_fusion launch_fusion.py`.

---

## Key Technical Talking Points (Interview)

**Sensor fusion:** Frustum-based approach — 2D detection constrains a 3D volume (frustum),
LiDAR points inside are clustered and fitted with an oriented 3D box. Combines complementary
modalities: camera gives semantic labels, LiDAR gives metric depth.

**Calibration:** LiDAR-to-camera extrinsic T (4x4), camera intrinsic K (3x3) + distortion D.
`cv2.projectPoints` applies full radial+tangential model. Coordinate transform:
LiDAR frame → camera frame → image plane.

**PCA yaw:** Covariance of XY point distribution → eigenvectors → principal axis gives
heading. Equivalent to finding the minimum-area oriented bounding rectangle.

**Kalman filter:** Constant-velocity model. State [x,y,z,vx,vy,vz], observation [x,y,z].
Predict step: x' = Fx, P' = FPF^T + Q. Update step: innovation y = z - Hx',
Kalman gain K = P'H^T(HP'H^T + R)^-1, x = x' + Ky.
Gives velocity estimates and handles missed detections gracefully (max_age frames).

**Hungarian matching:** scipy.optimize.linear_sum_assignment on 3D distance cost matrix.
Optimal O(N³) assignment vs greedy O(N²) — matters when tracks and detections overlap.

**RANSAC ground removal:** Random 3-point plane sampling, count inliers within threshold.
Added horizontality check: reject planes where |dot(normal, Z)| < threshold (walls, ramps).
