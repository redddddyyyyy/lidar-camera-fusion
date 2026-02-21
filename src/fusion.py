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


class KalmanTrack:
    """
    Single-object Kalman filter track.

    State vector:  x = [px, py, pz, vx, vy, vz]   (position + velocity)
    Observation:   z = [px, py, pz]                 (3-D centre from fusion)

    Constant-velocity model with dt=1 (one frame).  All units in metres.
    """

    # State-transition matrix F  (constant-velocity)
    _F = np.array([
        [1, 0, 0, 1, 0, 0],
        [0, 1, 0, 0, 1, 0],
        [0, 0, 1, 0, 0, 1],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1],
    ], dtype=np.float64)

    # Observation matrix H  (we only observe position)
    _H = np.array([
        [1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
    ], dtype=np.float64)

    # Process noise Q  (tuned: position noise 0.1 m, velocity noise 1 m/s²)
    _Q = np.diag([0.01, 0.01, 0.01, 1.0, 1.0, 1.0]).astype(np.float64)

    # Measurement noise R  (LiDAR centroid uncertainty ~0.1 m per axis)
    _R = np.diag([0.1, 0.1, 0.1]).astype(np.float64)

    def __init__(self, fused_obj: FusedObject, track_id: int):
        self.track_id = track_id
        self.label = fused_obj.label
        self.hits = 1
        self.age = 0               # frames since last matched detection

        # Initial state from first 3-D centre; velocity = 0
        if fused_obj.has_3d:
            pos = fused_obj.bbox_3d.center.copy()
        else:
            pos = np.zeros(3)

        self.x = np.array([pos[0], pos[1], pos[2], 0.0, 0.0, 0.0], dtype=np.float64)
        self.P = np.eye(6, dtype=np.float64) * 10.0   # initial uncertainty

        # Keep the most recent FusedObject for downstream use
        self.last_fused: FusedObject = fused_obj

    # ------------------------------------------------------------------ #
    # Kalman predict / update
    # ------------------------------------------------------------------ #

    def predict(self) -> np.ndarray:
        """Predict next state (call once per frame before update)."""
        self.x = self._F @ self.x
        self.P = self._F @ self.P @ self._F.T + self._Q
        self.age += 1
        return self.x[:3]          # predicted position

    def update(self, fused_obj: FusedObject) -> None:
        """Correct state with a new matched detection."""
        if not fused_obj.has_3d:
            # No 3-D info — skip Kalman update but keep the track alive
            self.hits += 1
            self.age = 0
            self.last_fused = fused_obj
            return

        z = fused_obj.bbox_3d.center.astype(np.float64)   # (3,)

        # Innovation
        y = z - self._H @ self.x
        S = self._H @ self.P @ self._H.T + self._R
        K = self.P @ self._H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ self._H) @ self.P

        self.hits += 1
        self.age = 0
        self.last_fused = fused_obj

    @property
    def position(self) -> np.ndarray:
        """Current (filtered) 3-D position estimate."""
        return self.x[:3].copy()

    @property
    def velocity(self) -> np.ndarray:
        """Current velocity estimate (m/frame)."""
        return self.x[3:].copy()


class KalmanTracker:
    """
    Multi-object Kalman filter tracker.

    Replaces the original IoU-only TemporalFusion with:
    - Per-object Kalman filters for position + velocity estimation
    - Hungarian (optimal) assignment via scipy.optimize.linear_sum_assignment
    - 3-D distance gating on matched pairs (ignores clearly wrong matches)
    - Graceful fallback to 2-D IoU when a detection has no 3-D box

    Public API mirrors the old TemporalFusion.update() signature.
    """

    def __init__(
        self,
        max_age: int = 5,
        min_hits: int = 3,
        dist_threshold: float = 5.0,   # max 3-D distance (m) for a valid match
        iou_threshold: float = 0.3,    # 2-D IoU fallback threshold
    ):
        self.max_age = max_age
        self.min_hits = min_hits
        self.dist_threshold = dist_threshold
        self.iou_threshold = iou_threshold

        self.tracks: List[KalmanTrack] = []
        self.next_id = 0

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def update(self, fused_objects: List[FusedObject]) -> List[FusedObject]:
        """
        Update all tracks with new detections for this frame.

        Returns the list of confirmed FusedObjects (hits >= min_hits),
        with their Kalman-filtered 3-D centres written back in.
        """
        # 1. Predict step for every existing track
        for track in self.tracks:
            track.predict()

        # 2. Match detections → tracks
        matched, unmatched_dets, unmatched_tracks = self._match(fused_objects)

        # 3. Update matched tracks
        for t_idx, d_idx in matched:
            self.tracks[t_idx].update(fused_objects[d_idx])

        # 4. Spawn new tracks for unmatched detections
        for d_idx in unmatched_dets:
            self.tracks.append(KalmanTrack(fused_objects[d_idx], self.next_id))
            self.next_id += 1

        # 5. Remove stale tracks
        self.tracks = [t for t in self.tracks if t.age < self.max_age]

        # 6. Return confirmed tracks; patch FusedObject centre with Kalman estimate
        confirmed: List[FusedObject] = []
        for track in self.tracks:
            if track.hits >= self.min_hits:
                obj = track.last_fused
                if obj.has_3d:
                    # Overwrite centre with Kalman-filtered position
                    obj.bbox_3d.center = track.position
                    obj.distance = float(np.linalg.norm(track.position))
                confirmed.append(obj)

        return confirmed

    # ------------------------------------------------------------------ #
    # Matching
    # ------------------------------------------------------------------ #

    def _match(
        self,
        detections: List[FusedObject]
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        Hungarian matching between current tracks and new detections.

        Cost matrix uses 3-D Euclidean distance when both sides have 3-D boxes,
        falling back to negative 2-D IoU otherwise.
        """
        from scipy.optimize import linear_sum_assignment

        if not self.tracks or not detections:
            return [], list(range(len(detections))), list(range(len(self.tracks)))

        T, D = len(self.tracks), len(detections)
        cost = np.full((T, D), fill_value=1e6, dtype=np.float64)

        for t_idx, track in enumerate(self.tracks):
            for d_idx, det in enumerate(detections):
                if det.has_3d:
                    # 3-D distance between Kalman prediction and detection centre
                    cost[t_idx, d_idx] = float(
                        np.linalg.norm(track.position - det.bbox_3d.center)
                    )
                else:
                    # Fallback: convert IoU to cost (lower = better)
                    iou = _iou_2d(track.last_fused.detection_2d.bbox,
                                  det.detection_2d.bbox)
                    cost[t_idx, d_idx] = 1.0 - iou

        row_ind, col_ind = linear_sum_assignment(cost)

        matched, unmatched_dets, unmatched_tracks = [], [], []
        matched_t, matched_d = set(), set()

        for t_idx, d_idx in zip(row_ind, col_ind):
            det = detections[d_idx]
            if det.has_3d and cost[t_idx, d_idx] > self.dist_threshold:
                continue   # gate: too far apart
            if not det.has_3d and cost[t_idx, d_idx] > (1.0 - self.iou_threshold):
                continue   # gate: IoU too low
            matched.append((t_idx, d_idx))
            matched_t.add(t_idx)
            matched_d.add(d_idx)

        unmatched_tracks = [i for i in range(T) if i not in matched_t]
        unmatched_dets   = [i for i in range(D) if i not in matched_d]

        return matched, unmatched_dets, unmatched_tracks


# ------------------------------------------------------------------ #
# Shared helper (used by KalmanTracker and SensorFusion)
# ------------------------------------------------------------------ #

def _iou_2d(box1: Tuple, box2: Tuple) -> float:
    """Compute 2D IoU between two (x1,y1,x2,y2) boxes."""
    x1 = max(box1[0], box2[0]);  y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2]);  y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    a2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    denom = a1 + a2 - inter
    return inter / denom if denom > 0 else 0.0


# Keep the old name as an alias so existing code that imports TemporalFusion
# continues to work during the transition.
TemporalFusion = KalmanTracker
