"""
LiDAR point cloud processing utilities.

Handles:
- Point cloud loading (PCD, BIN, NPY formats)
- Ground plane removal (RANSAC with horizontal-plane validation)
- Voxel-based clustering with O(1) BFS neighbor lookup
- PCA-based 3D bounding box estimation with yaw angle
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

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

            if data_format == 'binary':
                data = np.frombuffer(f.read(), dtype=np.float32)
                fields = header.get('FIELDS', 'x y z').split()
                data = data.reshape(-1, len(fields))
            else:
                data = np.loadtxt(f, dtype=np.float32)

        points = data[:, :3]
        intensities = data[:, 3] if data.shape[1] > 3 else None
        return cls(points=points, intensities=intensities)

    def to_bin(self, path: str | Path) -> None:
        """Save to KITTI binary format."""
        intensities = self.intensities if self.intensities is not None else np.zeros(len(self.points), dtype=np.float32)
        data = np.column_stack([self.points, intensities]).astype(np.float32)
        data.tofile(path)

    def to_npy(self, path: str | Path) -> None:
        """Save to NumPy file."""
        data = np.column_stack([self.points, self.intensities]) if self.intensities is not None else self.points
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
        yaw: Rotation around z-axis (radians), estimated via PCA
        label: Object class label
        score: Detection confidence score
    """

    center: np.ndarray  # (3,)
    size: np.ndarray    # (3,) - length, width, height
    yaw: float = 0.0
    label: str = ""
    score: float = 1.0

    @property
    def corners(self) -> np.ndarray:
        """Get 8 corner points of the bounding box (8x3)."""
        l, w, h = self.size
        corners_local = np.array([
            [-l/2, -w/2, -h/2], [ l/2, -w/2, -h/2],
            [ l/2,  w/2, -h/2], [-l/2,  w/2, -h/2],
            [-l/2, -w/2,  h/2], [ l/2, -w/2,  h/2],
            [ l/2,  w/2,  h/2], [-l/2,  w/2,  h/2],
        ])
        c, s = np.cos(self.yaw), np.sin(self.yaw)
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        return (R @ corners_local.T).T + self.center

    @classmethod
    def from_points(cls, points: np.ndarray, label: str = "", score: float = 1.0) -> BoundingBox3D:
        """
        Estimate bounding box from a cluster of points using PCA.

        PCA on the XY plane gives the dominant orientation (yaw) of the
        cluster, producing a tighter, oriented box compared to AABB.
        """
        if len(points) == 0:
            return cls(center=np.zeros(3), size=np.zeros(3), label=label, score=score)

        if len(points) < 3:
            # Fall back to AABB for very small clusters
            min_pt = points.min(axis=0)
            max_pt = points.max(axis=0)
            return cls(
                center=(min_pt + max_pt) / 2,
                size=max_pt - min_pt,
                yaw=0.0, label=label, score=score
            )

        # --- PCA on XY plane to find dominant orientation ---
        xy = points[:, :2]
        xy_centered = xy - xy.mean(axis=0)
        cov = np.cov(xy_centered.T)                    # 2x2 covariance
        eigvals, eigvecs = np.linalg.eigh(cov)         # sorted ascending
        principal_axis = eigvecs[:, -1]                # largest eigenvector
        yaw = float(np.arctan2(principal_axis[1], principal_axis[0]))

        # --- Rotate points into the PCA frame ---
        c, s = np.cos(-yaw), np.sin(-yaw)
        R2d = np.array([[c, -s], [s, c]])
        xy_rot = (R2d @ xy_centered.T).T

        # --- Compute oriented box extents ---
        xy_min = xy_rot.min(axis=0)
        xy_max = xy_rot.max(axis=0)
        z_min, z_max = points[:, 2].min(), points[:, 2].max()

        length = float(xy_max[0] - xy_min[0])
        width  = float(xy_max[1] - xy_min[1])
        height = float(z_max - z_min)

        # Center in world frame
        center_rot = np.array([(xy_min[0] + xy_max[0]) / 2,
                                (xy_min[1] + xy_max[1]) / 2])
        c2, s2 = np.cos(yaw), np.sin(yaw)
        R2d_inv = np.array([[c2, -s2], [s2, c2]])
        center_xy = R2d_inv @ center_rot + xy.mean(axis=0)
        center_z  = (z_min + z_max) / 2
        center    = np.array([center_xy[0], center_xy[1], center_z])

        return cls(
            center=center,
            size=np.array([length, width, height]),
            yaw=yaw,
            label=label,
            score=score
        )


class LidarProcessor:
    """
    LiDAR point cloud processing pipeline.

    Improvements over original:
    - RANSAC ground removal validates that the detected plane is roughly
      horizontal (normal aligned with Z), rejecting walls/ramps.
    - Voxel BFS clustering uses a dict for O(1) neighbor lookup instead
      of the original O(N) np.where scan per BFS step.
    - 3D bounding boxes use PCA-based yaw estimation (see BoundingBox3D).
    """

    def __init__(
        self,
        ground_threshold: float = 0.2,
        cluster_tolerance: float = 0.5,
        min_cluster_size: int = 10,
        max_cluster_size: int = 10000,
        ground_normal_tol: float = 0.3,   # max |dot(normal, Z)| deviation from 1
    ):
        self.ground_threshold = ground_threshold
        self.cluster_tolerance = cluster_tolerance
        self.min_cluster_size = min_cluster_size
        self.max_cluster_size = max_cluster_size
        self.ground_normal_tol = ground_normal_tol

    # ------------------------------------------------------------------
    # Ground removal
    # ------------------------------------------------------------------

    def remove_ground(self, pc: PointCloud) -> Tuple[PointCloud, PointCloud]:
        """
        Remove ground plane using RANSAC.

        Adds a horizontality check: only planes whose normal is within
        `ground_normal_tol` of straight-up (Z axis) are accepted.
        This prevents walls or ramps from being mistaken for ground.

        Returns:
            (non_ground_pc, ground_pc)
        """
        points = pc.points
        best_inliers: np.ndarray = np.array([], dtype=int)
        n_iterations = 100
        z_up = np.array([0.0, 0.0, 1.0])

        for _ in range(n_iterations):
            idx = np.random.choice(len(points), 3, replace=False)
            p1, p2, p3 = points[idx]

            normal = np.cross(p2 - p1, p3 - p1)
            norm_len = np.linalg.norm(normal)
            if norm_len < 1e-6:
                continue
            normal /= norm_len

            # Reject non-horizontal planes (walls, ramps, etc.)
            if abs(np.dot(normal, z_up)) < (1.0 - self.ground_normal_tol):
                continue

            d = -np.dot(normal, p1)
            distances = np.abs(points @ normal + d)
            inliers = np.where(distances < self.ground_threshold)[0]

            if len(inliers) > len(best_inliers):
                best_inliers = inliers

        ground_mask = np.zeros(len(points), dtype=bool)
        ground_mask[best_inliers] = True

        return pc[~ground_mask], pc[ground_mask]

    # ------------------------------------------------------------------
    # Clustering  (O(1) BFS — fixed from original O(N²))
    # ------------------------------------------------------------------

    def cluster_points(self, pc: PointCloud) -> List[PointCloud]:
        """
        Cluster non-ground points via voxel-based connected-component BFS.

        Key fix: neighbor lookup uses a pre-built dict {voxel_id -> array_index}
        giving O(1) per BFS step instead of the original O(N) np.where scan.
        """
        points = pc.points
        if len(points) == 0:
            return []

        voxel_size = self.cluster_tolerance

        # Integer voxel coordinates per point
        voxel_coords = np.floor(points / voxel_size).astype(np.int64)

        # Encode (ix, iy, iz) → single int64 key.
        # Using large primes avoids collisions for typical outdoor scenes.
        PRIME_Y = np.int64(1_000_003)
        PRIME_Z = np.int64(1_000_000_007)
        voxel_ids = (
            voxel_coords[:, 0].astype(np.int64) * PRIME_Y * PRIME_Z +
            voxel_coords[:, 1].astype(np.int64) * PRIME_Z +
            voxel_coords[:, 2].astype(np.int64)
        )

        # unique voxel ids → index into unique_voxels array
        unique_ids, inverse = np.unique(voxel_ids, return_inverse=True)

        # O(1) lookup: voxel_id → position in unique_ids
        id_to_uidx: Dict[int, int] = {int(uid): i for i, uid in enumerate(unique_ids)}

        # Voxel coordinate for each unique voxel (needed for neighbor offsets)
        # Pick the coordinate of the first point that lands in each voxel
        first_point_per_voxel = np.zeros((len(unique_ids), 3), dtype=np.int64)
        assigned = np.zeros(len(unique_ids), dtype=bool)
        for pt_idx, uidx in enumerate(inverse):
            if not assigned[uidx]:
                first_point_per_voxel[uidx] = voxel_coords[pt_idx]
                assigned[uidx] = True

        # BFS over voxels
        visited = np.zeros(len(unique_ids), dtype=bool)
        clusters: List[PointCloud] = []

        for start_uidx in range(len(unique_ids)):
            if visited[start_uidx]:
                continue

            cluster_uidxs: List[int] = []
            queue: deque = deque([start_uidx])
            visited[start_uidx] = True

            while queue:
                cur = queue.popleft()
                cluster_uidxs.append(cur)
                cx, cy, cz = first_point_per_voxel[cur]

                # 26-connected neighbors — O(1) dict lookup each
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        for dz in (-1, 0, 1):
                            if dx == 0 and dy == 0 and dz == 0:
                                continue
                            nid = int(
                                (cx + dx) * PRIME_Y * PRIME_Z +
                                (cy + dy) * PRIME_Z +
                                (cz + dz)
                            )
                            nidx = id_to_uidx.get(nid)
                            if nidx is not None and not visited[nidx]:
                                visited[nidx] = True
                                queue.append(nidx)

            # Collect all points belonging to this cluster
            cluster_mask = np.isin(inverse, cluster_uidxs)
            cluster_pts = points[cluster_mask]

            if self.min_cluster_size <= len(cluster_pts) <= self.max_cluster_size:
                clusters.append(PointCloud(points=cluster_pts))

        return clusters

    # ------------------------------------------------------------------
    # Bounding box estimation
    # ------------------------------------------------------------------

    def estimate_bounding_boxes(
        self,
        clusters: List[PointCloud],
        labels: List[str] | None = None,
        scores: List[float] | None = None
    ) -> List[BoundingBox3D]:
        """Estimate PCA-oriented 3D bounding boxes for each cluster."""
        boxes = []
        for i, cluster in enumerate(clusters):
            label = labels[i] if labels else "object"
            score = scores[i] if scores else 1.0
            boxes.append(BoundingBox3D.from_points(cluster.points, label=label, score=score))
        return boxes

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def process(self, pc: PointCloud) -> Tuple[List[PointCloud], List[BoundingBox3D]]:
        """
        Full processing pipeline: ground removal → clustering → box estimation.

        Returns:
            (clusters, bounding_boxes)
        """
        non_ground, _ = self.remove_ground(pc)
        clusters = self.cluster_points(non_ground)
        boxes = self.estimate_bounding_boxes(clusters)
        return clusters, boxes
