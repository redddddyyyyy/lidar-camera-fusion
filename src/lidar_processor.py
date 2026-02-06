"""
LiDAR point cloud processing utilities.

Handles:
- Point cloud loading (PCD, BIN, NPY formats)
- Ground plane removal
- Clustering and segmentation
- 3D bounding box estimation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

import numpy as np


@dataclass
class PointCloud:
    """
    Point cloud data structure.

    Attributes:
        points: Nx3 array of XYZ coordinates
        intensities: N array of intensity values (optional)
        colors: Nx3 array of RGB colors (optional)
    """

    points: np.ndarray  # Nx3
    intensities: np.ndarray | None = None  # N
    colors: np.ndarray | None = None  # Nx3

    def __len__(self) -> int:
        return len(self.points)

    def __getitem__(self, idx) -> PointCloud:
        """Index or slice the point cloud."""
        return PointCloud(
            points=self.points[idx],
            intensities=self.intensities[idx] if self.intensities is not None else None,
            colors=self.colors[idx] if self.colors is not None else None
        )

    @classmethod
    def from_bin(cls, path: str | Path) -> PointCloud:
        """Load from KITTI binary format (x, y, z, intensity)."""
        data = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
        return cls(points=data[:, :3], intensities=data[:, 3])

    @classmethod
    def from_npy(cls, path: str | Path) -> PointCloud:
        """Load from NumPy file."""
        data = np.load(path)
        if data.shape[1] == 3:
            return cls(points=data)
        elif data.shape[1] == 4:
            return cls(points=data[:, :3], intensities=data[:, 3])
        elif data.shape[1] >= 6:
            return cls(points=data[:, :3], intensities=data[:, 3] if data.shape[1] > 3 else None,
                      colors=data[:, 4:7] if data.shape[1] >= 7 else None)
        else:
            raise ValueError(f"Unexpected data shape: {data.shape}")

    @classmethod
    def from_pcd(cls, path: str | Path) -> PointCloud:
        """Load from PCD file format."""
        with open(path, 'rb') as f:
            # Read header
            header = {}
            while True:
                line = f.readline().decode('utf-8').strip()
                if line.startswith('DATA'):
                    data_format = line.split()[1]
                    break
                if ':' in line or ' ' in line:
                    parts = line.split(' ', 1)
                    if len(parts) == 2:
                        header[parts[0]] = parts[1]

            num_points = int(header.get('POINTS', header.get('WIDTH', '0')))

            if data_format == 'binary':
                # Read binary data
                data = np.frombuffer(f.read(), dtype=np.float32)
                fields = header.get('FIELDS', 'x y z').split()
                num_fields = len(fields)
                data = data.reshape(-1, num_fields)
            else:
                # Read ASCII data
                data = np.loadtxt(f, dtype=np.float32)

        points = data[:, :3]
        intensities = data[:, 3] if data.shape[1] > 3 else None

        return cls(points=points, intensities=intensities)

    def to_bin(self, path: str | Path) -> None:
        """Save to KITTI binary format."""
        if self.intensities is None:
            intensities = np.zeros(len(self.points), dtype=np.float32)
        else:
            intensities = self.intensities

        data = np.column_stack([self.points, intensities]).astype(np.float32)
        data.tofile(path)

    def to_npy(self, path: str | Path) -> None:
        """Save to NumPy file."""
        if self.intensities is not None:
            data = np.column_stack([self.points, self.intensities])
        else:
            data = self.points
        np.save(path, data)

    def filter_range(
        self,
        x_range: Tuple[float, float] = (-50, 50),
        y_range: Tuple[float, float] = (-50, 50),
        z_range: Tuple[float, float] = (-3, 10)
    ) -> PointCloud:
        """Filter points within specified ranges."""
        mask = (
            (self.points[:, 0] >= x_range[0]) & (self.points[:, 0] <= x_range[1]) &
            (self.points[:, 1] >= y_range[0]) & (self.points[:, 1] <= y_range[1]) &
            (self.points[:, 2] >= z_range[0]) & (self.points[:, 2] <= z_range[1])
        )
        return self[mask]


@dataclass
class BoundingBox3D:
    """
    3D bounding box representation.

    Attributes:
        center: (x, y, z) center position
        size: (length, width, height) dimensions
        yaw: Rotation around z-axis (radians)
        label: Object class label
        score: Detection confidence score
    """

    center: np.ndarray  # (3,)
    size: np.ndarray  # (3,) - length, width, height
    yaw: float = 0.0
    label: str = ""
    score: float = 1.0

    @property
    def corners(self) -> np.ndarray:
        """Get 8 corner points of the bounding box (8x3)."""
        l, w, h = self.size
        x, y, z = self.center

        # Corner offsets (relative to center)
        corners_local = np.array([
            [-l/2, -w/2, -h/2],
            [l/2, -w/2, -h/2],
            [l/2, w/2, -h/2],
            [-l/2, w/2, -h/2],
            [-l/2, -w/2, h/2],
            [l/2, -w/2, h/2],
            [l/2, w/2, h/2],
            [-l/2, w/2, h/2],
        ])

        # Rotation matrix around z-axis
        c, s = np.cos(self.yaw), np.sin(self.yaw)
        R = np.array([
            [c, -s, 0],
            [s, c, 0],
            [0, 0, 1]
        ])

        # Transform corners
        corners_world = (R @ corners_local.T).T + self.center

        return corners_world

    @classmethod
    def from_points(cls, points: np.ndarray, label: str = "", score: float = 1.0) -> BoundingBox3D:
        """
        Estimate bounding box from a cluster of points.

        Uses axis-aligned bounding box (no rotation estimation).
        """
        if len(points) == 0:
            return cls(
                center=np.zeros(3),
                size=np.zeros(3),
                label=label,
                score=score
            )

        min_pt = points.min(axis=0)
        max_pt = points.max(axis=0)

        center = (min_pt + max_pt) / 2
        size = max_pt - min_pt

        return cls(center=center, size=size, yaw=0.0, label=label, score=score)


class LidarProcessor:
    """
    LiDAR point cloud processing pipeline.

    Features:
    - Ground plane removal (RANSAC)
    - Euclidean clustering
    - 3D bounding box estimation
    """

    def __init__(
        self,
        ground_threshold: float = 0.2,
        cluster_tolerance: float = 0.5,
        min_cluster_size: int = 10,
        max_cluster_size: int = 10000
    ):
        """
        Initialize processor.

        Args:
            ground_threshold: Max distance from ground plane (m)
            cluster_tolerance: Euclidean clustering distance (m)
            min_cluster_size: Minimum points per cluster
            max_cluster_size: Maximum points per cluster
        """
        self.ground_threshold = ground_threshold
        self.cluster_tolerance = cluster_tolerance
        self.min_cluster_size = min_cluster_size
        self.max_cluster_size = max_cluster_size

    def remove_ground(self, pc: PointCloud) -> Tuple[PointCloud, PointCloud]:
        """
        Remove ground plane points using RANSAC.

        Args:
            pc: Input point cloud

        Returns:
            (non_ground_points, ground_points)
        """
        points = pc.points
        best_inliers = []
        best_plane = None
        n_iterations = 100

        for _ in range(n_iterations):
            # Sample 3 random points
            idx = np.random.choice(len(points), 3, replace=False)
            p1, p2, p3 = points[idx]

            # Compute plane normal
            v1 = p2 - p1
            v2 = p3 - p1
            normal = np.cross(v1, v2)

            if np.linalg.norm(normal) < 1e-6:
                continue

            normal = normal / np.linalg.norm(normal)
            d = -np.dot(normal, p1)

            # Count inliers
            distances = np.abs(np.dot(points, normal) + d)
            inliers = np.where(distances < self.ground_threshold)[0]

            if len(inliers) > len(best_inliers):
                best_inliers = inliers
                best_plane = (normal, d)

        # Separate ground and non-ground points
        ground_mask = np.zeros(len(points), dtype=bool)
        ground_mask[best_inliers] = True

        ground_pc = pc[ground_mask]
        non_ground_pc = pc[~ground_mask]

        return non_ground_pc, ground_pc

    def cluster_points(self, pc: PointCloud) -> List[PointCloud]:
        """
        Cluster points using simple grid-based approach.

        Args:
            pc: Input point cloud (ideally with ground removed)

        Returns:
            List of point cloud clusters
        """
        points = pc.points

        if len(points) == 0:
            return []

        # Simple voxel-based clustering
        voxel_size = self.cluster_tolerance

        # Voxelize points
        voxel_coords = np.floor(points / voxel_size).astype(int)

        # Create unique voxel IDs
        voxel_ids = (
            voxel_coords[:, 0] * 1000000 +
            voxel_coords[:, 1] * 1000 +
            voxel_coords[:, 2]
        )

        # Group points by connected voxels (simplified)
        unique_voxels, inverse = np.unique(voxel_ids, return_inverse=True)

        # Build clusters from connected components
        clusters = []
        visited = set()

        for start_idx in range(len(unique_voxels)):
            if start_idx in visited:
                continue

            # BFS to find connected voxels
            cluster_voxels = {start_idx}
            queue = [start_idx]

            while queue:
                current = queue.pop(0)
                current_coord = voxel_coords[inverse == current][0]

                # Check 26-connected neighbors
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        for dz in [-1, 0, 1]:
                            if dx == 0 and dy == 0 and dz == 0:
                                continue

                            neighbor_coord = current_coord + np.array([dx, dy, dz])
                            neighbor_id = (
                                neighbor_coord[0] * 1000000 +
                                neighbor_coord[1] * 1000 +
                                neighbor_coord[2]
                            )

                            # Find if neighbor exists
                            neighbor_idx = np.where(unique_voxels == neighbor_id)[0]
                            if len(neighbor_idx) > 0:
                                neighbor_idx = neighbor_idx[0]
                                if neighbor_idx not in visited and neighbor_idx not in cluster_voxels:
                                    cluster_voxels.add(neighbor_idx)
                                    queue.append(neighbor_idx)

            visited.update(cluster_voxels)

            # Get points in this cluster
            cluster_mask = np.isin(inverse, list(cluster_voxels))
            cluster_points = points[cluster_mask]

            # Filter by size
            if self.min_cluster_size <= len(cluster_points) <= self.max_cluster_size:
                clusters.append(PointCloud(points=cluster_points))

        return clusters

    def estimate_bounding_boxes(
        self,
        clusters: List[PointCloud],
        labels: List[str] | None = None,
        scores: List[float] | None = None
    ) -> List[BoundingBox3D]:
        """
        Estimate 3D bounding boxes for each cluster.

        Args:
            clusters: List of point cloud clusters
            labels: Optional labels for each cluster
            scores: Optional confidence scores

        Returns:
            List of 3D bounding boxes
        """
        boxes = []

        for i, cluster in enumerate(clusters):
            label = labels[i] if labels else "object"
            score = scores[i] if scores else 1.0

            box = BoundingBox3D.from_points(cluster.points, label=label, score=score)
            boxes.append(box)

        return boxes

    def process(self, pc: PointCloud) -> Tuple[List[PointCloud], List[BoundingBox3D]]:
        """
        Full processing pipeline.

        Args:
            pc: Input point cloud

        Returns:
            (clusters, bounding_boxes)
        """
        # Remove ground
        non_ground, _ = self.remove_ground(pc)

        # Cluster
        clusters = self.cluster_points(non_ground)

        # Estimate boxes
        boxes = self.estimate_bounding_boxes(clusters)

        return clusters, boxes
