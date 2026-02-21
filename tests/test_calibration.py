"""Tests for calibration.py — projection math and distortion correction."""
import numpy as np
import pytest
from src.calibration import CameraCalibration, LidarCameraCalibration


@pytest.fixture
def default_calib():
    cam = CameraCalibration.default(image_size=(640, 480))
    return LidarCameraCalibration.default(cam)


# ------------------------------------------------------------------
# CameraCalibration
# ------------------------------------------------------------------

def test_intrinsic_properties():
    cam = CameraCalibration.default(image_size=(640, 480))
    assert cam.fx == cam.K[0, 0]
    assert cam.fy == cam.K[1, 1]
    assert cam.cx == cam.K[0, 2]
    assert cam.cy == cam.K[1, 2]


def test_default_calibration_shape():
    cam = CameraCalibration.default()
    assert cam.K.shape == (3, 3)
    assert cam.D.shape == (5,)


def test_yaml_roundtrip(tmp_path):
    cam = CameraCalibration.default(image_size=(1280, 720))
    path = tmp_path / "cam.yaml"
    cam.to_yaml(path)
    cam2 = CameraCalibration.from_yaml(path)
    np.testing.assert_allclose(cam.K, cam2.K)
    np.testing.assert_allclose(cam.D, cam2.D)
    assert cam.image_size == cam2.image_size


# ------------------------------------------------------------------
# LidarCameraCalibration — projection
# ------------------------------------------------------------------

def test_project_points_front_facing(default_calib):
    """A point in front of the camera must project with positive depth.

    Default R maps LiDAR-x → camera-z, so a point with positive LiDAR-x
    lands in front of the camera.
    """
    pts = np.array([[10.0, 0.0, 0.0]])   # 10 m ahead in LiDAR (x-forward) frame
    pixels, depths, valid = default_calib.project_points(pts)
    assert valid[0], "Point in front of camera should be valid"
    assert depths[0] > 0


def test_project_points_behind_camera(default_calib):
    """A point behind the camera (negative z in camera frame) must be invalid."""
    # Transform a point so it ends up behind the camera
    # In default calibration R maps LiDAR-x → camera-z, so negative LiDAR-x → negative camera-z
    pts = np.array([[-20.0, 0.0, 0.0]])
    pixels, depths, valid = default_calib.project_points(pts)
    assert not valid[0], "Point behind camera must be flagged as invalid"


def test_project_points_output_shapes(default_calib):
    pts = np.random.randn(50, 3)
    pixels, depths, valid = default_calib.project_points(pts)
    assert pixels.shape == (50, 2)
    assert depths.shape == (50,)
    assert valid.shape == (50,)


def test_distortion_zero_coefficients(default_calib):
    """With D=0, cv2.projectPoints must agree with the manual pinhole formula."""
    # Default R maps LiDAR-x → camera-z; use positive LiDAR-x so depth > 0
    pts = np.array([[5.0, -1.0, 0.0]])

    # Manual pinhole (no distortion)
    pts_cam = (default_calib.R @ pts.T + default_calib.t).T
    depth = pts_cam[0, 2]
    px_ideal = default_calib.camera.K @ (pts_cam[0] / depth)
    expected = px_ideal[:2]

    pixels, depths, valid = default_calib.project_points(pts)
    assert valid[0], f"Point should be in front of camera (depth={depths[0]:.2f})"
    np.testing.assert_allclose(pixels[0], expected, atol=0.5)


def test_frustum_points_inside_box(default_calib):
    """Points that project inside a large bounding box should be returned.

    Default R maps LiDAR-x → camera-z, so points with positive LiDAR-x are
    in front of the camera.
    """
    np.random.seed(42)
    # Positive LiDAR-x (5–15 m), small y/z variation → all land in front
    pts = np.random.uniform(low=[5, -2, -1], high=[15, 2, 1], size=(200, 3))
    frustum = default_calib.get_frustum_points(
        pts, bbox_2d=(0, 0, 640, 480), depth_range=(0.5, 80.0)
    )
    assert len(frustum) > 0


def test_extrinsic_T_matrix(default_calib):
    T = default_calib.T
    assert T.shape == (4, 4)
    np.testing.assert_allclose(T[:3, :3], default_calib.R)
    np.testing.assert_allclose(T[:3, 3], default_calib.t.flatten())
    np.testing.assert_allclose(T[3], [0, 0, 0, 1])
