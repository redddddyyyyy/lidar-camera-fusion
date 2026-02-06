# LiDAR-Camera Fusion with YOLO

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat&logo=python&logoColor=white)](https://python.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF?style=flat)](https://ultralytics.com/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.5+-5C3EE8?style=flat&logo=opencv&logoColor=white)](https://opencv.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Multi-sensor fusion pipeline combining **LiDAR point clouds** with **camera images** using **YOLOv8** for 3D object detection in autonomous driving scenarios.

---

## Results

### Fusion Visualization

<p align="center">
  <img src="assets/fusion_result.png" alt="Fusion Result" width="900"/>
</p>

*Top: Camera view with 2D detections and LiDAR overlay. Bottom: Bird's eye view with 3D bounding boxes.*

### Pipeline Overview

<p align="center">
  <img src="assets/pipeline.png" alt="Pipeline" width="800"/>
</p>

---

## Overview

This project implements a **frustum-based sensor fusion** approach:

1. **2D Detection** — YOLO detects objects in camera images
2. **Frustum Extraction** — Project 2D boxes to 3D frustums using calibration
3. **Point Filtering** — Extract LiDAR points within each frustum
4. **3D Estimation** — Fit 3D bounding boxes to frustum points

### Key Features

| Feature | Description |
|---------|-------------|
| **YOLOv8 Integration** | State-of-the-art 2D object detection |
| **Frustum PointNets** | 3D detection from 2D proposals |
| **KITTI Support** | Load KITTI dataset format directly |
| **Calibration Tools** | Camera intrinsics and LiDAR-camera extrinsics |
| **BEV Visualization** | Bird's eye view with 3D boxes |
| **Temporal Tracking** | IoU-based multi-object tracking |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     SENSOR FUSION PIPELINE                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Camera Image                    LiDAR Point Cloud             │
│        │                                │                       │
│        ▼                                │                       │
│   ┌─────────────┐                       │                       │
│   │   YOLOv8    │                       │                       │
│   │  Detection  │                       │                       │
│   └──────┬──────┘                       │                       │
│          │                              │                       │
│          ▼                              ▼                       │
│   2D Bounding Boxes           3D Points (x, y, z)               │
│          │                              │                       │
│          └──────────┬───────────────────┘                       │
│                     │                                           │
│                     ▼                                           │
│          ┌─────────────────────┐                                │
│          │  Calibration (K, T) │                                │
│          │  Project to Frustum │                                │
│          └──────────┬──────────┘                                │
│                     │                                           │
│                     ▼                                           │
│          ┌─────────────────────┐                                │
│          │  Frustum Points     │                                │
│          │  Extraction         │                                │
│          └──────────┬──────────┘                                │
│                     │                                           │
│                     ▼                                           │
│          ┌─────────────────────┐                                │
│          │  3D Bounding Box    │                                │
│          │  Estimation         │                                │
│          └──────────┬──────────┘                                │
│                     │                                           │
│                     ▼                                           │
│            Fused 3D Detections                                  │
│            (class, bbox_3d, distance)                           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
├── src/
│   ├── __init__.py          # Package exports
│   ├── calibration.py       # Camera & LiDAR-camera calibration
│   ├── lidar_processor.py   # Point cloud processing & clustering
│   ├── detector.py          # YOLO detection wrapper
│   ├── fusion.py            # Sensor fusion & tracking
│   └── visualization.py     # Visualization utilities
├── config/                  # Calibration files
├── data/                    # Sample data
├── scripts/                 # Utility scripts
├── main.py                  # Demo script
├── assets/                  # Result images
└── requirements.txt         # Dependencies
```

---

## Quick Start

### Installation

```bash
git clone https://github.com/redddddyyyyy/lidar-camera-fusion.git
cd lidar-camera-fusion
pip install -r requirements.txt
```

### Run Demo

```bash
# Run with synthetic demo data
python main.py --demo

# Save visualization
python main.py --demo --save assets/fusion_result.png
```

### Run with Custom Data

```bash
python main.py \
    --image data/sample/image.png \
    --lidar data/sample/velodyne.bin \
    --calib data/sample/calib.txt \
    --save output.png
```

### Run with KITTI Dataset

```bash
# Download KITTI data first
python main.py \
    --image data/kitti/image_2/000000.png \
    --lidar data/kitti/velodyne/000000.bin \
    --calib data/kitti/calib/000000.txt
```

---

## API Usage

### Basic Fusion Pipeline

```python
from src import (
    CameraCalibration, LidarCameraCalibration,
    PointCloud, YOLODetector, SensorFusion, FusionVisualizer
)
import cv2

# Load data
image = cv2.imread("image.png")
point_cloud = PointCloud.from_bin("velodyne.bin")

# Setup calibration
camera_calib = CameraCalibration.from_kitti("calib.txt")
lidar_calib = LidarCameraCalibration.from_kitti("calib.txt", camera_calib)

# Detect objects
detector = YOLODetector(model_path="yolov8n.pt", confidence_threshold=0.5)
detections = detector.detect(image)

# Fuse with LiDAR
fusion = SensorFusion(lidar_calib)
fused_objects = fusion.fuse(detections, point_cloud)

# Visualize
for obj in fused_objects:
    if obj.has_3d:
        print(f"{obj.label}: {obj.distance:.1f}m")
```

### Calibration

```python
from src import CameraCalibration, LidarCameraCalibration

# From KITTI format
camera = CameraCalibration.from_kitti("calib.txt")
lidar_cam = LidarCameraCalibration.from_kitti("calib.txt", camera)

# Default (for testing)
camera = CameraCalibration.default(image_size=(1280, 720))
lidar_cam = LidarCameraCalibration.default(camera)

# Project LiDAR points to image
pixels, depths, valid = lidar_cam.project_to_image(points)
```

### Point Cloud Processing

```python
from src import PointCloud, LidarProcessor

# Load point cloud
pc = PointCloud.from_bin("velodyne.bin")  # KITTI format
pc = PointCloud.from_npy("points.npy")    # NumPy format
pc = PointCloud.from_pcd("cloud.pcd")     # PCD format

# Filter by range
pc_filtered = pc.filter_range(
    x_range=(-40, 40),
    y_range=(0, 70),
    z_range=(-2, 10)
)

# Ground removal and clustering
processor = LidarProcessor()
non_ground, ground = processor.remove_ground(pc)
clusters = processor.cluster_points(non_ground)
```

---

## Calibration Format

### KITTI Format

```
P0: 7.215377e+02 0.000000e+00 6.095593e+02 0.000000e+00 ...
P1: 7.215377e+02 0.000000e+00 6.095593e+02 -3.875744e+02 ...
P2: 7.215377e+02 0.000000e+00 6.095593e+02 4.485728e+01 ...
P3: 7.215377e+02 0.000000e+00 6.095593e+02 -3.395242e+02 ...
R0_rect: 9.999239e-01 9.837760e-03 -7.445048e-03 ...
Tr_velo_to_cam: 7.533745e-03 -9.999714e-01 -6.166020e-04 ...
Tr_imu_to_velo: 9.999976e-01 7.553071e-04 -2.035826e-03 ...
```

### Custom YAML Format

```yaml
camera_matrix:
  rows: 3
  cols: 3
  data: [fx, 0, cx, 0, fy, cy, 0, 0, 1]
distortion_coefficients:
  data: [k1, k2, p1, p2, k3]
image_width: 1280
image_height: 720
```

---

## Supported Classes

| Class | Color | Notes |
|-------|-------|-------|
| car | Green | Primary vehicle class |
| truck | Teal | Large vehicles |
| bus | Orange | Public transport |
| motorcycle | Blue | Two-wheelers |
| bicycle | Cyan | Cyclists |
| person | Red | Pedestrians |

---

## References

- Qi, C. R., et al. "Frustum PointNets for 3D Object Detection from RGB-D Data." *CVPR*, 2018.
- Geiger, A., et al. "Are we ready for Autonomous Driving? The KITTI Vision Benchmark Suite." *CVPR*, 2012.
- Jocher, G., et al. "Ultralytics YOLOv8." https://ultralytics.com, 2023.

---

## License

MIT
