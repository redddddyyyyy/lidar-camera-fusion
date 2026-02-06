"""
Camera and LiDAR-Camera calibration utilities.

Handles:
- Camera intrinsic parameters (focal length, principal point, distortion)
- LiDAR-Camera extrinsic transformation (rotation, translation)
- Point projection from 3D LiDAR frame to 2D image plane
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import yaml


@dataclass
class CameraCalibration:
    """
    Camera intrinsic calibration parameters.

    Attributes:
        K: 3x3 intrinsic matrix [fx, 0, cx; 0, fy, cy; 0, 0, 1]
        D: Distortion coefficients [k1, k2, p1, p2, k3]
        image_size: (width, height) in pixels
    """

    K: np.ndarray  # 3x3 intrinsic matrix
    D: np.ndarray  # Distortion coefficients
    image_size: Tuple[int, int]  # (width, height)

    @property
    def fx(self) -> float:
        return self.K[0, 0]

    @property
    def fy(self) -> float:
        return self.K[1, 1]

    @property
    def cx(self) -> float:
        return self.K[0, 2]

    @property
    def cy(self) -> float:
        return self.K[1, 2]

    @classmethod
    def from_yaml(cls, path: str | Path) -> CameraCalibration:
        """Load calibration from YAML file."""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)

        K = np.array(data['camera_matrix']['data']).reshape(3, 3)
        D = np.array(data['distortion_coefficients']['data'])
        image_size = (data['image_width'], data['image_height'])

        return cls(K=K, D=D, image_size=image_size)

    @classmethod
    def from_kitti(cls, calib_path: str | Path) -> CameraCalibration:
        """Load calibration from KITTI format file."""
        with open(calib_path, 'r') as f:
            lines = f.readlines()

        # Parse P2 (left color camera projection matrix)
        for line in lines:
            if line.startswith('P2:'):
                P2 = np.array([float(x) for x in line.strip().split()[1:]]).reshape(3, 4)
                break

        # Extract intrinsics from projection matrix
        K = P2[:, :3]
        D = np.zeros(5)  # KITTI provides rectified images
        image_size = (1242, 375)  # Standard KITTI image size

        return cls(K=K, D=D, image_size=image_size)

    @classmethod
    def default(cls, image_size: Tuple[int, int] = (1280, 720)) -> CameraCalibration:
        """Create default calibration for testing."""
        w, h = image_size
        fx = fy = 0.8 * w  # Approximate focal length
        cx, cy = w / 2, h / 2

        K = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=np.float64)

        D = np.zeros(5, dtype=np.float64)

        return cls(K=K, D=D, image_size=image_size)

    def to_yaml(self, path: str | Path) -> None:
        """Save calibration to YAML file."""
        data = {
            'camera_matrix': {
                'rows': 3,
                'cols': 3,
                'data': self.K.flatten().tolist()
            },
            'distortion_coefficients': {
                'rows': 1,
                'cols': 5,
                'data': self.D.tolist()
            },
            'image_width': self.image_size[0],
            'image_height': self.image_size[1]
        }

        with open(path, 'w') as f:
            yaml.dump(data, f)


@dataclass
class LidarCameraCalibration:
    """
    LiDAR-Camera extrinsic calibration.

    Transforms points from LiDAR frame to camera frame.

    Attributes:
        R: 3x3 rotation matrix (LiDAR to camera)
        t: 3x1 translation vector (LiDAR to camera)
        camera: Camera intrinsic calibration
    """

    R: np.ndarray  # 3x3 rotation matrix
    t: np.ndarray  # 3x1 translation vector
    camera: CameraCalibration

    @property
    def T(self) -> np.ndarray:
        """4x4 homogeneous transformation matrix."""
        T = np.eye(4)
        T[:3, :3] = self.R
        T[:3, 3] = self.t.flatten()
        return T

    @classmethod
    def from_yaml(cls, path: str | Path, camera: CameraCalibration) -> LidarCameraCalibration:
        """Load extrinsic calibration from YAML file."""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)

        R = np.array(data['rotation']['data']).reshape(3, 3)
        t = np.array(data['translation']['data']).reshape(3, 1)

        return cls(R=R, t=t, camera=camera)

    @classmethod
    def from_kitti(cls, calib_path: str | Path, camera: CameraCalibration) -> LidarCameraCalibration:
        """Load extrinsic calibration from KITTI format file."""
        with open(calib_path, 'r') as f:
            lines = f.readlines()

        # Parse Tr_velo_to_cam (Velodyne to camera transformation)
        for line in lines:
            if line.startswith('Tr_velo_to_cam:'):
                Tr = np.array([float(x) for x in line.strip().split()[1:]]).reshape(3, 4)
                break

        R = Tr[:, :3]
        t = Tr[:, 3].reshape(3, 1)

        return cls(R=R, t=t, camera=camera)

    @classmethod
    def default(cls, camera: CameraCalibration) -> LidarCameraCalibration:
        """
        Create default calibration for testing.

        Assumes LiDAR is mounted on vehicle roof, looking forward.
        Camera is mounted on front windshield.
        """
        # Rotation: LiDAR x-forward, y-left, z-up
        #           Camera x-right, y-down, z-forward
        R = np.array([
            [0, -1, 0],
            [0, 0, -1],
            [1, 0, 0]
        ], dtype=np.float64)

        # Translation: camera is ~1.5m in front and 0.5m below LiDAR
        t = np.array([[0], [-0.5], [-1.5]], dtype=np.float64)

        return cls(R=R, t=t, camera=camera)

    def to_yaml(self, path: str | Path) -> None:
        """Save extrinsic calibration to YAML file."""
        data = {
            'rotation': {
                'rows': 3,
                'cols': 3,
                'data': self.R.flatten().tolist()
            },
            'translation': {
                'rows': 3,
                'cols': 1,
                'data': self.t.flatten().tolist()
            }
        }

        with open(path, 'w') as f:
            yaml.dump(data, f)

    def project_points(self, points_lidar: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Project 3D LiDAR points to 2D image coordinates.

        Args:
            points_lidar: Nx3 array of points in LiDAR frame

        Returns:
            pixels: Nx2 array of pixel coordinates
            depths: N array of depths (z in camera frame)
        """
        N = points_lidar.shape[0]

        # Transform to camera frame
        points_cam = (self.R @ points_lidar.T + self.t).T  # Nx3

        # Filter points behind camera
        valid_mask = points_cam[:, 2] > 0.1

        # Project to image plane
        depths = points_cam[:, 2]

        # Homogeneous coordinates
        points_norm = points_cam / depths.reshape(-1, 1)  # Nx3

        # Apply intrinsics
        pixels = (self.camera.K @ points_norm.T).T[:, :2]  # Nx2

        return pixels, depths, valid_mask

    def project_to_image(
        self,
        points_lidar: np.ndarray,
        image_shape: Tuple[int, int] | None = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Project points and filter to valid image region.

        Args:
            points_lidar: Nx3 array of points in LiDAR frame
            image_shape: (height, width) or use calibration default

        Returns:
            pixels: Mx2 array of valid pixel coordinates
            depths: M array of depths
            indices: M array of original point indices
        """
        if image_shape is None:
            h, w = self.camera.image_size[1], self.camera.image_size[0]
        else:
            h, w = image_shape

        pixels, depths, front_mask = self.project_points(points_lidar)

        # Filter to image bounds
        valid_mask = (
            front_mask &
            (pixels[:, 0] >= 0) & (pixels[:, 0] < w) &
            (pixels[:, 1] >= 0) & (pixels[:, 1] < h)
        )

        indices = np.where(valid_mask)[0]

        return pixels[valid_mask], depths[valid_mask], indices

    def get_frustum_points(
        self,
        points_lidar: np.ndarray,
        bbox_2d: Tuple[int, int, int, int],
        depth_range: Tuple[float, float] = (0.5, 50.0)
    ) -> np.ndarray:
        """
        Get LiDAR points within a 2D bounding box frustum.

        Args:
            points_lidar: Nx3 array of points in LiDAR frame
            bbox_2d: (x1, y1, x2, y2) bounding box in image
            depth_range: (min_depth, max_depth) in meters

        Returns:
            Mx3 array of points within the frustum
        """
        x1, y1, x2, y2 = bbox_2d
        min_depth, max_depth = depth_range

        pixels, depths, indices = self.project_to_image(points_lidar)

        # Filter to bounding box
        in_box = (
            (pixels[:, 0] >= x1) & (pixels[:, 0] <= x2) &
            (pixels[:, 1] >= y1) & (pixels[:, 1] <= y2) &
            (depths >= min_depth) & (depths <= max_depth)
        )

        frustum_indices = indices[in_box]

        return points_lidar[frustum_indices]
