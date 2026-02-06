"""
LiDAR-Camera Fusion Demo.

Demonstrates sensor fusion pipeline with sample data or KITTI dataset.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from src import (
    CameraCalibration, LidarCameraCalibration,
    LidarProcessor, PointCloud,
    YOLODetector,
    SensorFusion, FusedObject,
    FusionVisualizer
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LiDAR-Camera Fusion Demo")

    p.add_argument("--image", type=str, default=None,
                   help="Path to camera image")
    p.add_argument("--lidar", type=str, default=None,
                   help="Path to LiDAR point cloud (.bin, .npy, or .pcd)")
    p.add_argument("--calib", type=str, default=None,
                   help="Path to calibration file (KITTI format)")
    p.add_argument("--model", type=str, default="yolov8n.pt",
                   help="YOLO model path or name")
    p.add_argument("--confidence", type=float, default=0.5,
                   help="Detection confidence threshold")
    p.add_argument("--save", type=str, default=None,
                   help="Save output visualization")
    p.add_argument("--demo", action="store_true",
                   help="Run with synthetic demo data")

    return p.parse_args()


def create_demo_data():
    """Create synthetic demo data for testing."""
    print("Creating synthetic demo data...")

    # Create a simple image (gradient with some rectangles)
    image = np.zeros((480, 640, 3), dtype=np.uint8)

    # Sky gradient
    for y in range(240):
        image[y, :] = [200 - y//3, 200 - y//3, 255 - y//4]

    # Ground
    image[240:, :] = [50, 100, 50]

    # Road
    road_mask = np.zeros((480, 640), dtype=bool)
    for y in range(240, 480):
        # Road widens toward bottom
        width = int(100 + (y - 240) * 0.8)
        center = 320
        road_mask[y, max(0, center-width):min(640, center+width)] = True

    image[road_mask] = [80, 80, 80]

    # Create synthetic LiDAR points
    np.random.seed(42)

    # Ground plane points
    x_ground = np.random.uniform(-20, 20, 5000)
    y_ground = np.random.uniform(5, 50, 5000)
    z_ground = np.random.uniform(-1.8, -1.5, 5000)  # Ground at ~-1.7m
    ground_points = np.column_stack([x_ground, y_ground, z_ground])

    # Object 1: Car at (3, 15, 0)
    car1_x = np.random.uniform(1.5, 4.5, 200)
    car1_y = np.random.uniform(13, 17, 200)
    car1_z = np.random.uniform(-1.5, 0, 200)
    car1_points = np.column_stack([car1_x, car1_y, car1_z])

    # Object 2: Car at (-5, 25, 0)
    car2_x = np.random.uniform(-7, -3, 150)
    car2_y = np.random.uniform(22, 28, 150)
    car2_z = np.random.uniform(-1.5, 0, 150)
    car2_points = np.column_stack([car2_x, car2_y, car2_z])

    # Object 3: Person at (-2, 10, 0)
    person_x = np.random.uniform(-2.5, -1.5, 50)
    person_y = np.random.uniform(8, 12, 50)
    person_z = np.random.uniform(-1.5, 0.2, 50)
    person_points = np.column_stack([person_x, person_y, person_z])

    # Combine all points
    all_points = np.vstack([ground_points, car1_points, car2_points, person_points])

    point_cloud = PointCloud(points=all_points.astype(np.float32))

    # Create calibration
    camera_calib = CameraCalibration.default(image_size=(640, 480))
    lidar_camera_calib = LidarCameraCalibration.default(camera_calib)

    return image, point_cloud, lidar_camera_calib


def run_demo():
    """Run fusion demo with synthetic data."""
    print("=" * 60)
    print("LIDAR-CAMERA FUSION DEMO")
    print("=" * 60)

    # Create demo data
    image, point_cloud, calibration = create_demo_data()

    print(f"\nImage shape: {image.shape}")
    print(f"Point cloud: {len(point_cloud)} points")

    # Initialize detector
    print("\nInitializing YOLO detector...")
    detector = YOLODetector(
        model_path="yolov8n.pt",
        confidence_threshold=0.5,
        filter_classes=True
    )

    # Detect objects
    print("Running detection...")
    detections = detector.detect(image)
    print(f"Found {len(detections)} detections")

    for det in detections:
        print(f"  - {det.label}: {det.score:.2f} at {det.bbox}")

    # Initialize fusion
    print("\nRunning sensor fusion...")
    fusion = SensorFusion(calibration)

    # Fuse detections with LiDAR
    fused_objects = fusion.fuse(detections, point_cloud, image.shape[:2])

    print(f"\nFusion results:")
    for obj in fused_objects:
        if obj.has_3d:
            print(f"  - {obj.label}: {obj.distance:.1f}m, {obj.num_points} points")
            print(f"    3D size: {obj.bbox_3d.size}")
        else:
            print(f"  - {obj.label}: No 3D info ({obj.num_points} points)")

    # Visualize
    print("\nCreating visualization...")
    visualizer = FusionVisualizer(calibration)

    fig = visualizer.create_fusion_summary(image, point_cloud, fused_objects)

    return fig, fused_objects


def run_with_data(args):
    """Run fusion with provided data files."""
    import cv2

    print("=" * 60)
    print("LIDAR-CAMERA FUSION")
    print("=" * 60)

    # Load image
    print(f"\nLoading image: {args.image}")
    image = cv2.imread(args.image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print(f"Image shape: {image.shape}")

    # Load point cloud
    print(f"Loading point cloud: {args.lidar}")
    lidar_path = Path(args.lidar)
    if lidar_path.suffix == '.bin':
        point_cloud = PointCloud.from_bin(lidar_path)
    elif lidar_path.suffix == '.npy':
        point_cloud = PointCloud.from_npy(lidar_path)
    elif lidar_path.suffix == '.pcd':
        point_cloud = PointCloud.from_pcd(lidar_path)
    else:
        raise ValueError(f"Unsupported format: {lidar_path.suffix}")
    print(f"Point cloud: {len(point_cloud)} points")

    # Load or create calibration
    if args.calib:
        print(f"Loading calibration: {args.calib}")
        camera_calib = CameraCalibration.from_kitti(args.calib)
        calibration = LidarCameraCalibration.from_kitti(args.calib, camera_calib)
    else:
        print("Using default calibration")
        camera_calib = CameraCalibration.default(image.shape[1::-1])
        calibration = LidarCameraCalibration.default(camera_calib)

    # Initialize detector
    print(f"\nInitializing YOLO detector ({args.model})...")
    detector = YOLODetector(
        model_path=args.model,
        confidence_threshold=args.confidence,
        filter_classes=True
    )

    # Detect objects
    print("Running detection...")
    detections = detector.detect(image)
    print(f"Found {len(detections)} detections")

    # Initialize fusion
    print("\nRunning sensor fusion...")
    fusion = SensorFusion(calibration)

    # Fuse detections with LiDAR
    fused_objects = fusion.fuse(detections, point_cloud, image.shape[:2])

    print(f"\nFusion results:")
    for obj in fused_objects:
        if obj.has_3d:
            print(f"  - {obj.label}: {obj.distance:.1f}m, {obj.num_points} points")
        else:
            print(f"  - {obj.label}: No 3D info")

    # Visualize
    print("\nCreating visualization...")
    visualizer = FusionVisualizer(calibration)
    fig = visualizer.create_fusion_summary(image, point_cloud, fused_objects)

    return fig, fused_objects


def main():
    args = parse_args()

    if args.demo or (args.image is None and args.lidar is None):
        fig, fused_objects = run_demo()
    else:
        if args.image is None or args.lidar is None:
            print("Error: Both --image and --lidar required (or use --demo)")
            return
        fig, fused_objects = run_with_data(args)

    if args.save:
        fig.savefig(args.save, dpi=150, bbox_inches='tight')
        print(f"\nVisualization saved to {args.save}")
    else:
        plt.show()

    print("\nDone!")


if __name__ == "__main__":
    main()
