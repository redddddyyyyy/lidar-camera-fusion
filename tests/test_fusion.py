"""Tests for fusion.py — Kalman tracker, IoU helper, SensorFusion pipeline."""
import numpy as np
import pytest
from src.calibration import CameraCalibration, LidarCameraCalibration
from src.detector import Detection
from src.lidar_processor import BoundingBox3D, PointCloud
from src.fusion import FusedObject, SensorFusion, KalmanTracker, KalmanTrack, _iou_2d


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _make_detection(bbox=(100, 100, 200, 200), label="car", score=0.9):
    return Detection(bbox=bbox, label=label, class_id=2, score=score)


def _make_fused(bbox=(100, 100, 200, 200), center=None, label="car"):
    det = _make_detection(bbox=bbox, label=label)
    if center is not None:
        bbox3d = BoundingBox3D(
            center=np.array(center, dtype=float),
            size=np.array([4.0, 2.0, 1.5]),
            yaw=0.0, label=label
        )
        return FusedObject(detection_2d=det, bbox_3d=bbox3d,
                           points=np.random.randn(20, 3),
                           num_points=20,
                           distance=float(np.linalg.norm(center)))
    return FusedObject(detection_2d=det)


# ------------------------------------------------------------------
# _iou_2d helper
# ------------------------------------------------------------------

def test_iou_identical_boxes():
    box = (10, 10, 50, 50)
    assert _iou_2d(box, box) == pytest.approx(1.0)


def test_iou_non_overlapping():
    assert _iou_2d((0, 0, 10, 10), (20, 20, 30, 30)) == pytest.approx(0.0)


def test_iou_partial_overlap():
    # Two 10×10 boxes sharing a 5×10 region → IoU = 50/150
    iou = _iou_2d((0, 0, 10, 10), (5, 0, 15, 10))
    assert iou == pytest.approx(50 / 150)


# ------------------------------------------------------------------
# KalmanTrack
# ------------------------------------------------------------------

def test_kalman_track_predict_moves_position():
    obj = _make_fused(center=[10.0, 0.0, 0.0])
    track = KalmanTrack(obj, track_id=0)
    # Give it a velocity by hand
    track.x[3] = 1.0   # vx = 1 m/frame
    pos_before = track.position.copy()
    track.predict()
    pos_after = track.position
    assert pos_after[0] > pos_before[0], "X position should increase after predict with vx>0"


def test_kalman_track_update_reduces_error():
    obj = _make_fused(center=[0.0, 0.0, 5.0])
    track = KalmanTrack(obj, track_id=0)
    # Displace state intentionally
    track.x[:3] = [3.0, 3.0, 3.0]
    # Observe the true position
    obs_obj = _make_fused(center=[0.0, 0.0, 5.0])
    error_before = np.linalg.norm(track.x[:3] - np.array([0.0, 0.0, 5.0]))
    track.update(obs_obj)
    error_after = np.linalg.norm(track.x[:3] - np.array([0.0, 0.0, 5.0]))
    assert error_after < error_before, "Kalman update should pull state toward measurement"


def test_kalman_track_velocity_estimated():
    """After several observations of a moving object, velocity should be non-zero."""
    obj0 = _make_fused(center=[0.0, 0.0, 5.0])
    track = KalmanTrack(obj0, track_id=0)
    for t in range(1, 8):
        track.predict()
        obs = _make_fused(center=[float(t), 0.0, 5.0])
        track.update(obs)
    # After 7 frames of 1 m/frame motion, vx should be close to 1
    assert abs(track.velocity[0] - 1.0) < 0.4, \
        f"Expected vx≈1 m/frame, got {track.velocity[0]:.3f}"


# ------------------------------------------------------------------
# KalmanTracker
# ------------------------------------------------------------------

def test_kalman_tracker_spawns_tracks():
    tracker = KalmanTracker(max_age=5, min_hits=1)
    objs = [_make_fused(center=[0, 0, 5]), _make_fused(center=[5, 0, 10])]
    confirmed = tracker.update(objs)
    assert len(confirmed) == 2


def test_kalman_tracker_confirms_after_min_hits():
    tracker = KalmanTracker(max_age=5, min_hits=3)
    obj = _make_fused(center=[0, 0, 5])
    # 1st and 2nd frame: below min_hits
    assert len(tracker.update([obj])) == 0
    assert len(tracker.update([obj])) == 0
    # 3rd frame: should be confirmed
    assert len(tracker.update([obj])) == 1


def test_kalman_tracker_removes_stale_tracks():
    tracker = KalmanTracker(max_age=2, min_hits=1)
    obj = _make_fused(center=[0, 0, 5])
    tracker.update([obj])
    # No detections for max_age frames
    tracker.update([])
    tracker.update([])
    assert len(tracker.tracks) == 0, "Stale track should be removed"


def test_kalman_tracker_matches_same_object():
    """Same object in two consecutive frames should reuse the same track."""
    tracker = KalmanTracker(max_age=5, min_hits=1)
    obj1 = _make_fused(center=[0.0, 0.0, 5.0])
    obj2 = _make_fused(center=[0.1, 0.0, 5.0])   # slight movement
    tracker.update([obj1])
    n_tracks_before = len(tracker.tracks)
    tracker.update([obj2])
    n_tracks_after = len(tracker.tracks)
    assert n_tracks_after == n_tracks_before, "Close object should reuse existing track"


# ------------------------------------------------------------------
# SensorFusion pipeline (smoke test with synthetic data)
# ------------------------------------------------------------------

@pytest.fixture
def default_fusion():
    cam = CameraCalibration.default(image_size=(640, 480))
    calib = LidarCameraCalibration.default(cam)
    return SensorFusion(calib, min_points_threshold=3)


def test_sensor_fusion_returns_list(default_fusion):
    detections = [_make_detection()]
    pc = PointCloud(points=np.random.uniform(-5, 5, (500, 3)).astype(np.float32))
    result = default_fusion.fuse(detections, pc, image_shape=(480, 640))
    assert isinstance(result, list)


def test_sensor_fusion_empty_detections(default_fusion):
    pc = PointCloud(points=np.random.randn(100, 3).astype(np.float32))
    result = default_fusion.fuse([], pc)
    assert result == []


def test_sensor_fusion_empty_pointcloud(default_fusion):
    detections = [_make_detection()]
    pc = PointCloud(points=np.empty((0, 3), dtype=np.float32))
    result = default_fusion.fuse(detections, pc)
    assert len(result) == 1
    assert not result[0].has_3d
