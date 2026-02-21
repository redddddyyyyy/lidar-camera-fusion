"""Tests for lidar_processor.py — clustering, RANSAC, PCA yaw estimation."""
import numpy as np
import pytest
from src.lidar_processor import PointCloud, BoundingBox3D, LidarProcessor


# ------------------------------------------------------------------
# PointCloud
# ------------------------------------------------------------------

def test_pointcloud_len():
    pc = PointCloud(points=np.random.randn(100, 3))
    assert len(pc) == 100


def test_pointcloud_slice():
    pts = np.arange(30).reshape(10, 3).astype(float)
    pc = PointCloud(points=pts, intensities=np.ones(10))
    sliced = pc[:5]
    assert sliced.points.shape == (5, 3)
    assert sliced.intensities.shape == (5,)


def test_pointcloud_filter_range():
    pts = np.array([[0, 0, 0], [100, 0, 0], [0, 0, 5]], dtype=float)
    pc = PointCloud(points=pts)
    filtered = pc.filter_range(x_range=(-50, 50), y_range=(-50, 50), z_range=(-3, 10))
    assert len(filtered) == 2   # (100,0,0) and (0,0,5)→z=5 is inside, (100,…) outside x


def test_pointcloud_bin_roundtrip(tmp_path):
    pts = np.random.randn(100, 3).astype(np.float32)
    pc = PointCloud(points=pts, intensities=np.ones(100, dtype=np.float32))
    path = tmp_path / "test.bin"
    pc.to_bin(path)
    pc2 = PointCloud.from_bin(path)
    np.testing.assert_allclose(pc.points, pc2.points, atol=1e-5)


# ------------------------------------------------------------------
# BoundingBox3D — PCA yaw estimation
# ------------------------------------------------------------------

def test_bbox3d_from_points_empty():
    box = BoundingBox3D.from_points(np.empty((0, 3)))
    assert np.allclose(box.center, 0)
    assert np.allclose(box.size, 0)


def test_bbox3d_from_points_axis_aligned():
    """An axis-aligned rectangular cluster should yield yaw ≈ 0 or π/2."""
    pts = np.column_stack([
        np.random.uniform(0, 4, 200),   # long axis along X
        np.random.uniform(0, 1, 200),
        np.zeros(200)
    ])
    box = BoundingBox3D.from_points(pts)
    # Yaw should be close to 0 (or π which is equivalent)
    assert abs(np.cos(2 * box.yaw)) > 0.7, f"Unexpected yaw={box.yaw:.3f}"


def test_bbox3d_from_points_rotated():
    """A cluster rotated 45° should have yaw ≈ 45°."""
    angle = np.pi / 4
    n = 300
    # Generate a 4m × 1m box at 45°
    local_x = np.random.uniform(-2, 2, n)
    local_y = np.random.uniform(-0.5, 0.5, n)
    c, s = np.cos(angle), np.sin(angle)
    world_x = c * local_x - s * local_y
    world_y = s * local_x + c * local_y
    pts = np.column_stack([world_x, world_y, np.zeros(n)])

    box = BoundingBox3D.from_points(pts)
    # Yaw should be close to ±45° (modulo π ambiguity)
    yaw_mod = box.yaw % np.pi
    assert abs(yaw_mod - angle) < 0.15 or abs(yaw_mod - (np.pi - angle)) < 0.15, \
        f"Expected yaw≈45°, got {np.degrees(box.yaw):.1f}°"


def test_bbox3d_corners_shape():
    box = BoundingBox3D(
        center=np.array([1.0, 2.0, 0.5]),
        size=np.array([4.0, 2.0, 1.5]),
        yaw=0.3
    )
    assert box.corners.shape == (8, 3)


def test_bbox3d_size_positive():
    pts = np.random.randn(50, 3)
    box = BoundingBox3D.from_points(pts)
    assert np.all(box.size >= 0)


# ------------------------------------------------------------------
# LidarProcessor — clustering (O(1) BFS fix)
# ------------------------------------------------------------------

@pytest.fixture
def processor():
    return LidarProcessor(
        ground_threshold=0.2,
        cluster_tolerance=0.5,
        min_cluster_size=5,
        max_cluster_size=5000,
    )


def _make_two_blobs(n=200):
    """Two spatially separated point clusters (should yield 2 clusters)."""
    blob1 = np.random.randn(n, 3) * 0.3 + np.array([0.0, 0.0, 0.0])
    blob2 = np.random.randn(n, 3) * 0.3 + np.array([5.0, 0.0, 0.0])
    pts = np.vstack([blob1, blob2])
    return PointCloud(points=pts)


def test_clustering_finds_two_blobs(processor):
    pc = _make_two_blobs(200)
    clusters = processor.cluster_points(pc)
    assert len(clusters) == 2, f"Expected 2 clusters, got {len(clusters)}"


def test_clustering_empty_input(processor):
    pc = PointCloud(points=np.empty((0, 3)))
    clusters = processor.cluster_points(pc)
    assert clusters == []


def test_clustering_size_filter(processor):
    """Clusters below min_size or above max_size should be filtered out."""
    proc = LidarProcessor(min_cluster_size=50, max_cluster_size=100, cluster_tolerance=0.5)
    # 10-point blob: below min → filtered
    tiny_blob = np.random.randn(10, 3) * 0.1
    pc = PointCloud(points=tiny_blob)
    clusters = proc.cluster_points(pc)
    assert len(clusters) == 0


# ------------------------------------------------------------------
# LidarProcessor — RANSAC ground removal
# ------------------------------------------------------------------

def test_ground_removal_separates_ground(processor):
    """Most ground points (z ≈ 0) should be classified as ground."""
    np.random.seed(0)
    ground = np.column_stack([
        np.random.uniform(-10, 10, 1000),
        np.random.uniform(-10, 10, 1000),
        np.random.uniform(-0.1, 0.1, 1000)
    ])
    objects = np.column_stack([
        np.random.uniform(-5, 5, 100),
        np.random.uniform(-5, 5, 100),
        np.random.uniform(0.5, 2.0, 100)
    ])
    pc = PointCloud(points=np.vstack([ground, objects]))
    non_ground, ground_pc = processor.remove_ground(pc)
    # Most of the 1000 ground points should be removed
    assert len(ground_pc) > 600


def test_ground_removal_rejects_vertical_plane(processor):
    """A vertical wall should NOT be classified as ground."""
    np.random.seed(1)
    # Vertical plane at x=5 — normal points in X, not Z
    wall = np.column_stack([
        np.full(500, 5.0) + np.random.randn(500) * 0.05,
        np.random.uniform(-5, 5, 500),
        np.random.uniform(-2, 3, 500)
    ])
    # Real ground
    ground = np.column_stack([
        np.random.uniform(-10, 10, 500),
        np.random.uniform(-10, 10, 500),
        np.random.uniform(-0.1, 0.1, 500)
    ])
    pc = PointCloud(points=np.vstack([wall, ground]))
    non_ground, ground_pc = processor.remove_ground(pc)
    # Wall points should NOT dominate the ground set
    assert len(ground_pc) < 800, "Vertical wall incorrectly classified as ground"
