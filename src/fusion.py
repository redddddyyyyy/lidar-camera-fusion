"""
LiDAR-Camera sensor fusion module.

Combines:
- 2D object detections from YOLO
- 3D point cloud data from LiDAR
- Camera-LiDAR calibration

To produce 3D object detections with accurate depth and dimensions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple, Dict

import numpy as np

from .calibration import LidarCameraCalibration
from .lidar_processor import PointCloud, BoundingBox3D, LidarProcessor
from .detector import Detection


@dataclass
class FusedObject:
    """
    Fused detection combining 2D and 3D information.

    Attributes:
        detection_2d: Original YOLO detection
        bbox_3d: 3D bounding box from LiDAR points
        points: LiDAR points belonging to this object
        distance: Distance to object center (meters)
        num_points: Number of LiDAR points
    """

    detection_2d: Detection
    bbox_3d: BoundingBox3D | None = None
    points: np.ndarray = field(default_factory=lambda: np.array([]))
    distance: float = 0.0
    num_points: int = 0

    @property
    def label(self) -> str:
        return self.detection_2d.label

    @property
    def confidence(self) -> float:
        return self.detection_2d.score

    @property
    def has_3d(self) -> bool:
        return self.bbox_3d is not None and self.num_points > 0

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        result = {
            'label': self.label,
            'confidence': self.confidence,
            'bbox_2d': self.detection_2d.bbox,
            'distance': self.distance,
            'num_points': self.num_points,
        }

        if self.has_3d:
            result['center_3d'] = self.bbox_3d.center.tolist()
            result['size_3d'] = self.bbox_3d.size.tolist()
            result['yaw'] = self.bbox_3d.yaw

        return result


class SensorFusion:
    """
    LiDAR-Camera fusion pipeline.

    Pipeline:
    1. Detect objects in image using YOLO
    2. Project LiDAR points to image plane
    3. For each 2D detection, extract LiDAR points in frustum
    4. Estimate 3D bounding box from frustum points
    """

    def __init__(
        self,
        calibration: LidarCameraCalibration,
        min_points_threshold: int = 5,
        depth_range: Tuple[float, float] = (0.5, 80.0),
        frustum_expansion: float = 0.1
    ):
        """
        Initialize fusion module.

        Args:
            calibration: LiDAR-Camera calibration
            min_points_threshold: Minimum LiDAR points for valid 3D box
            depth_range: (min, max) depth range in meters
            frustum_expansion: Expand 2D bbox by this fraction
        """
        self.calibration = calibration
        self.min_points_threshold = min_points_threshold
        self.depth_range = depth_range
        self.frustum_expansion = frustum_expansion

        self.lidar_processor = LidarProcessor()

    def fuse(
        self,
        detections: List[Detection],
        point_cloud: PointCloud,
        image_shape: Tuple[int, int] | None = None
    ) -> List[FusedObject]:
        """
        Fuse 2D detections with LiDAR point cloud.

        Args:
            detections: List of 2D detections from YOLO
            point_cloud: LiDAR point cloud
            image_shape: (height, width) of image

        Returns:
            List of fused objects with 3D information
        """
        fused_objects = []

        for detection in detections:
            # Expand bounding box slightly
            bbox = self._expand_bbox(detection.bbox, image_shape)

            # Get frustum points
            frustum_points = self.calibration.get_frustum_points(
                point_cloud.points,
                bbox,
                depth_range=self.depth_range
            )

            # Create fused object
            fused = FusedObject(
                detection_2d=detection,
                points=frustum_points,
                num_points=len(frustum_points)
            )

            # Estimate 3D box if enough points
            if len(frustum_points) >= self.min_points_threshold:
                # Remove outliers using simple distance filtering
                filtered_points = self._filter_outliers(frustum_points)

                if len(filtered_points) >= self.min_points_threshold:
                    # Estimate 3D bounding box
                    bbox_3d = BoundingBox3D.from_points(
                        filtered_points,
                        label=detection.label,
                        score=detection.score
                    )

                    fused.bbox_3d = bbox_3d
                    fused.points = filtered_points
                    fused.num_points = len(filtered_points)
                    fused.distance = np.linalg.norm(bbox_3d.center)

            fused_objects.append(fused)

        return fused_objects

    def _expand_bbox(
        self,
        bbox: Tuple[int, int, int, int],
        image_shape: Tuple[int, int] | None
    ) -> Tuple[int, int, int, int]:
        """Expand bounding box by frustum_expansion fraction."""
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1

        dx = int(w * self.frustum_expansion)
        dy = int(h * self.frustum_expansion)

        x1 = max(0, x1 - dx)
        y1 = max(0, y1 - dy)
        x2 = x2 + dx
        y2 = y2 + dy

        if image_shape is not None:
            img_h, img_w = image_shape
            x2 = min(img_w, x2)
            y2 = min(img_h, y2)

        return (x1, y1, x2, y2)

    def _filter_outliers(
        self,
        points: np.ndarray,
        std_ratio: float = 2.0
    ) -> np.ndarray:
        """
        Filter outlier points using statistical approach.

        Removes points that are > std_ratio standard deviations
        from the median in any dimension.
        """
        if len(points) < 4:
            return points

        median = np.median(points, axis=0)
        std = np.std(points, axis=0)

        # Avoid division by zero
        std = np.maximum(std, 0.1)

        # Compute distances from median in each dimension
        dists = np.abs(points - median) / std

        # Keep points within threshold in all dimensions
        mask = np.all(dists < std_ratio, axis=1)

        return points[mask]

    def fuse_with_clustering(
        self,
        detections: List[Detection],
        point_cloud: PointCloud,
        image_shape: Tuple[int, int] | None = None
    ) -> List[FusedObject]:
        """
        Alternative fusion using LiDAR clustering.

        1. Cluster LiDAR points
        2. Match clusters to 2D detections based on projection
        """
        # Remove ground and cluster
        non_ground, _ = self.lidar_processor.remove_ground(point_cloud)
        clusters = self.lidar_processor.cluster_points(non_ground)

        # Project cluster centers to image
        cluster_projections = []
        for cluster in clusters:
            center = cluster.points.mean(axis=0)
            pixel, depth, valid = self.calibration.project_points(center.reshape(1, 3))
            if valid[0]:
                cluster_projections.append({
                    'cluster': cluster,
                    'pixel': pixel[0],
                    'depth': depth[0],
                    'center': center
                })

        # Match detections to clusters
        fused_objects = []

        for detection in detections:
            best_match = None
            best_score = 0

            x1, y1, x2, y2 = detection.bbox
            bbox_center = ((x1 + x2) / 2, (y1 + y2) / 2)

            for proj in cluster_projections:
                px, py = proj['pixel']

                # Check if cluster center projects inside bbox
                if x1 <= px <= x2 and y1 <= py <= y2:
                    # Score based on distance to bbox center
                    dist = np.sqrt((px - bbox_center[0])**2 + (py - bbox_center[1])**2)
                    score = 1.0 / (1.0 + dist)

                    if score > best_score:
                        best_score = score
                        best_match = proj

            # Create fused object
            if best_match is not None:
                bbox_3d = BoundingBox3D.from_points(
                    best_match['cluster'].points,
                    label=detection.label,
                    score=detection.score
                )

                fused = FusedObject(
                    detection_2d=detection,
                    bbox_3d=bbox_3d,
                    points=best_match['cluster'].points,
                    distance=best_match['depth'],
                    num_points=len(best_match['cluster'].points)
                )
            else:
                fused = FusedObject(
                    detection_2d=detection,
                    num_points=0
                )

            fused_objects.append(fused)

        return fused_objects


class TemporalFusion:
    """
    Temporal fusion for tracking objects across frames.

    Uses simple IoU-based tracking.
    """

    def __init__(self, max_age: int = 5, min_hits: int = 3, iou_threshold: float = 0.3):
        """
        Initialize temporal fusion.

        Args:
            max_age: Maximum frames to keep track without detection
            min_hits: Minimum detections before track is confirmed
            iou_threshold: IoU threshold for matching
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold

        self.tracks: List[Dict] = []
        self.next_id = 0

    def update(self, fused_objects: List[FusedObject]) -> List[FusedObject]:
        """
        Update tracks with new fused objects.

        Returns fused objects with track IDs assigned.
        """
        # Match new detections to existing tracks
        matched, unmatched_dets, unmatched_tracks = self._match(fused_objects)

        # Update matched tracks
        for track_idx, det_idx in matched:
            self.tracks[track_idx]['object'] = fused_objects[det_idx]
            self.tracks[track_idx]['hits'] += 1
            self.tracks[track_idx]['age'] = 0

        # Create new tracks for unmatched detections
        for det_idx in unmatched_dets:
            self.tracks.append({
                'id': self.next_id,
                'object': fused_objects[det_idx],
                'hits': 1,
                'age': 0
            })
            self.next_id += 1

        # Age unmatched tracks
        for track_idx in unmatched_tracks:
            self.tracks[track_idx]['age'] += 1

        # Remove old tracks
        self.tracks = [t for t in self.tracks if t['age'] < self.max_age]

        # Return confirmed tracks
        confirmed = [t['object'] for t in self.tracks if t['hits'] >= self.min_hits]

        return confirmed

    def _match(self, fused_objects: List[FusedObject]):
        """Match detections to tracks using IoU."""
        if not self.tracks or not fused_objects:
            unmatched_dets = list(range(len(fused_objects)))
            unmatched_tracks = list(range(len(self.tracks)))
            return [], unmatched_dets, unmatched_tracks

        # Compute IoU matrix
        iou_matrix = np.zeros((len(self.tracks), len(fused_objects)))

        for t, track in enumerate(self.tracks):
            for d, det in enumerate(fused_objects):
                iou_matrix[t, d] = self._iou_2d(
                    track['object'].detection_2d.bbox,
                    det.detection_2d.bbox
                )

        # Greedy matching
        matched = []
        unmatched_dets = set(range(len(fused_objects)))
        unmatched_tracks = set(range(len(self.tracks)))

        while True:
            max_iou = self.iou_threshold
            best_match = None

            for t in unmatched_tracks:
                for d in unmatched_dets:
                    if iou_matrix[t, d] > max_iou:
                        max_iou = iou_matrix[t, d]
                        best_match = (t, d)

            if best_match is None:
                break

            matched.append(best_match)
            unmatched_tracks.discard(best_match[0])
            unmatched_dets.discard(best_match[1])

        return matched, list(unmatched_dets), list(unmatched_tracks)

    @staticmethod
    def _iou_2d(box1: Tuple, box2: Tuple) -> float:
        """Compute 2D IoU."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

        return inter / (area1 + area2 - inter) if (area1 + area2 - inter) > 0 else 0
