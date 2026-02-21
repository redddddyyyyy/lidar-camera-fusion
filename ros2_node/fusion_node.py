#!/usr/bin/env python3
"""
LiDAR-Camera Fusion ROS 2 Node.

Subscribes to:
  /camera/image_raw          (sensor_msgs/Image)
  /velodyne_points           (sensor_msgs/PointCloud2)

Publishes:
  /fusion/image_annotated    (sensor_msgs/Image)       — 2D boxes + depth labels
  /fusion/points_colored     (sensor_msgs/PointCloud2) — depth-coloured point cloud
  /fusion/markers            (visualization_msgs/MarkerArray) — 3D oriented boxes
  /fusion/tracks             (visualization_msgs/MarkerArray) — Kalman track velocity arrows

All topics are viewable directly in RViz2 without extra plugins.

Usage (after sourcing ROS2 and installing package deps):
    python3 ros2_node/fusion_node.py

Or as a ROS2 run target if installed as a package:
    ros2 run lidar_camera_fusion fusion_node
"""

from __future__ import annotations

import sys
import os
import time
from typing import Optional

import numpy as np

# ── ROS 2 imports ──────────────────────────────────────────────────────────────
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from message_filters import ApproximateTimeSynchronizer, Subscriber

from sensor_msgs.msg import Image, PointCloud2, PointField
from std_msgs.msg import Header, ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, Vector3
import sensor_msgs_py.point_cloud2 as pc2
from cv_bridge import CvBridge

# ── Project imports ─────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.calibration import CameraCalibration, LidarCameraCalibration
from src.lidar_processor import PointCloud, LidarProcessor
from src.detector import YOLODetector
from src.fusion import SensorFusion, KalmanTracker, FusedObject


# ── Colour palette (BGR for OpenCV, RGB for RViz markers) ──────────────────────
CLASS_COLOURS_BGR = {
    'car':        (0,   255,   0),
    'truck':      (0,   200, 100),
    'bus':        (0,   150, 255),
    'motorcycle': (255, 100,   0),
    'bicycle':    (255, 255,   0),
    'person':     (0,     0, 255),
}

CLASS_COLOURS_RGBA = {
    k: ColorRGBA(r=b/255.0, g=g/255.0, b=r/255.0, a=0.6)
    for k, (b, g, r) in CLASS_COLOURS_BGR.items()
}
DEFAULT_COLOUR = ColorRGBA(r=0.5, g=0.5, b=0.5, a=0.5)


class FusionNode(Node):
    """
    ROS 2 node that runs the full LiDAR-camera fusion pipeline every frame
    and publishes results for RViz2 visualisation.
    """

    def __init__(self):
        super().__init__('lidar_camera_fusion')

        # ── Parameters ─────────────────────────────────────────────────────────
        self.declare_parameter('camera_topic',   '/camera/image_raw')
        self.declare_parameter('lidar_topic',    '/velodyne_points')
        self.declare_parameter('calib_yaml',     '')           # path to extrinsic YAML
        self.declare_parameter('yolo_model',     'yolov8n.pt')
        self.declare_parameter('confidence',     0.45)
        self.declare_parameter('min_points',     5)
        self.declare_parameter('sync_slop',      0.1)          # seconds
        self.declare_parameter('lidar_frame',    'velodyne_top_base_link')
        self.declare_parameter('camera_frame',   'camera_link')

        cam_topic   = self.get_parameter('camera_topic').value
        lidar_topic = self.get_parameter('lidar_topic').value
        calib_yaml  = self.get_parameter('calib_yaml').value
        yolo_model  = self.get_parameter('yolo_model').value
        confidence  = self.get_parameter('confidence').value
        min_points  = self.get_parameter('min_points').value
        sync_slop   = self.get_parameter('sync_slop').value
        self.lidar_frame  = self.get_parameter('lidar_frame').value

        # ── Calibration ────────────────────────────────────────────────────────
        if calib_yaml:
            cam_calib = CameraCalibration.from_yaml(calib_yaml)
            self.calib = LidarCameraCalibration.from_yaml(calib_yaml, cam_calib)
            self.get_logger().info(f'Loaded calibration from {calib_yaml}')
        else:
            cam_calib = CameraCalibration.default(image_size=(640, 480))
            self.calib = LidarCameraCalibration.default(cam_calib)
            self.get_logger().warn('No calibration file — using default (testing only)')

        # ── Pipeline components ─────────────────────────────────────────────────
        self.detector = YOLODetector(
            model_path=yolo_model,
            confidence_threshold=confidence,
            filter_classes=True
        )
        self.fusion   = SensorFusion(self.calib, min_points_threshold=min_points)
        self.tracker  = KalmanTracker(max_age=8, min_hits=2, dist_threshold=4.0)
        self.bridge   = CvBridge()

        # Frame counter for marker IDs
        self._frame = 0

        # ── QoS ────────────────────────────────────────────────────────────────
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            depth=5
        )

        # ── Subscribers (time-synchronised) ────────────────────────────────────
        self._img_sub   = Subscriber(self, Image,        cam_topic,   qos_profile=sensor_qos)
        self._lidar_sub = Subscriber(self, PointCloud2,  lidar_topic, qos_profile=sensor_qos)

        self._sync = ApproximateTimeSynchronizer(
            [self._img_sub, self._lidar_sub],
            queue_size=10,
            slop=sync_slop
        )
        self._sync.registerCallback(self._callback)

        # ── Publishers ─────────────────────────────────────────────────────────
        self._pub_image   = self.create_publisher(Image,       '/fusion/image_annotated',  10)
        self._pub_cloud   = self.create_publisher(PointCloud2, '/fusion/points_colored',   10)
        self._pub_boxes   = self.create_publisher(MarkerArray, '/fusion/markers',           10)
        self._pub_tracks  = self.create_publisher(MarkerArray, '/fusion/tracks',            10)

        self.get_logger().info(
            f'FusionNode ready — camera: {cam_topic}  lidar: {lidar_topic}'
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Main callback
    # ──────────────────────────────────────────────────────────────────────────

    def _callback(self, img_msg: Image, lidar_msg: PointCloud2) -> None:
        t0 = time.monotonic()
        self._frame += 1
        stamp = img_msg.header.stamp

        # ── Convert ROS messages to numpy ──────────────────────────────────────
        # Use passthrough to get whatever encoding the bag uses (bgr8, rgb8, etc.)
        # Convert to 3-channel RGB regardless of source encoding.
        import cv2 as _cv2
        raw = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding='passthrough')
        enc = img_msg.encoding.lower()
        if raw.ndim == 2:                        # mono / grayscale
            image = _cv2.cvtColor(raw, _cv2.COLOR_GRAY2RGB)
        elif raw.shape[2] == 4:                  # BGRA or RGBA (8UC4)
            if 'rgb' in enc:
                image = _cv2.cvtColor(raw, _cv2.COLOR_RGBA2RGB)
            else:
                image = _cv2.cvtColor(raw, _cv2.COLOR_BGRA2RGB)
        elif 'bgr' in enc:                       # BGR
            image = _cv2.cvtColor(raw, _cv2.COLOR_BGR2RGB)
        elif 'bayer' in enc:
            image = _cv2.cvtColor(raw, _cv2.COLOR_BayerBG2RGB)
        else:
            image = raw                          # already RGB
        h, w  = image.shape[:2]

        # read_points returns a structured array with named fields + possible padding.
        # Explicitly stack x/y/z to get a clean (N,3) float32 array.
        pts_struct = np.array(
            list(pc2.read_points(lidar_msg, field_names=('x', 'y', 'z'), skip_nans=True))
        )
        if pts_struct.size == 0:
            pts_raw = np.empty((0, 3), dtype=np.float32)
        elif pts_struct.dtype.names:   # structured array — extract fields
            pts_raw = np.column_stack([
                pts_struct['x'].astype(np.float32),
                pts_struct['y'].astype(np.float32),
                pts_struct['z'].astype(np.float32),
            ])
        else:                          # plain numeric array — just reshape
            pts_raw = pts_struct.astype(np.float32).reshape(-1, 3)

        pc = PointCloud(points=pts_raw)

        # ── Run fusion pipeline ────────────────────────────────────────────────
        detections   = self.detector.detect(image)
        fused_objs   = self.fusion.fuse(detections, pc, image_shape=(h, w))
        tracked_objs = self.tracker.update(fused_objs)

        dt = time.monotonic() - t0
        self.get_logger().info(
            f'[{self._frame}] {len(detections)} det → {len(fused_objs)} fused '
            f'→ {len(tracked_objs)} tracked  ({dt*1000:.1f} ms)',
            throttle_duration_sec=1.0
        )

        # ── Publish results ────────────────────────────────────────────────────
        self._publish_annotated_image(image, fused_objs, stamp, img_msg.header.frame_id)
        self._publish_colored_cloud(pc, stamp)
        self._publish_3d_boxes(tracked_objs, stamp)
        self._publish_velocity_arrows(tracked_objs, stamp)

    # ──────────────────────────────────────────────────────────────────────────
    # Publishers
    # ──────────────────────────────────────────────────────────────────────────

    def _publish_annotated_image(
        self,
        image: np.ndarray,
        objects: list[FusedObject],
        stamp,
        frame_id: str
    ) -> None:
        """Draw 2D detections and distance labels on the camera image."""
        import cv2
        vis = image.copy()

        for obj in objects:
            x1, y1, x2, y2 = obj.detection_2d.bbox
            colour = CLASS_COLOURS_BGR.get(obj.label, (128, 128, 128))
            cv2.rectangle(vis, (x1, y1), (x2, y2), colour, 2)

            label = f'{obj.label}'
            if obj.has_3d:
                label += f' {obj.distance:.1f}m'
            cv2.putText(vis, label, (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 2)

        msg = self.bridge.cv2_to_imgmsg(vis, encoding='rgb8')
        msg.header.stamp    = stamp
        msg.header.frame_id = frame_id
        self._pub_image.publish(msg)

    def _publish_colored_cloud(self, pc: PointCloud, stamp) -> None:
        """Publish point cloud coloured by depth for RViz."""
        pts = pc.points
        if len(pts) == 0:
            return

        depths  = np.linalg.norm(pts, axis=1)
        d_norm  = np.clip(depths / 50.0, 0.0, 1.0)       # normalise 0–50 m

        # Jet-like colour map: near=blue, far=red
        r = np.clip(1.5 - np.abs(d_norm - 1.0) * 4, 0, 1)
        g = np.clip(1.5 - np.abs(d_norm - 0.5) * 4, 0, 1)
        b = np.clip(1.5 - np.abs(d_norm - 0.0) * 4, 0, 1)

        rgb_packed = (
            (r * 255).astype(np.uint32) << 16 |
            (g * 255).astype(np.uint32) << 8  |
            (b * 255).astype(np.uint32)
        ).view(np.float32)

        cloud_data = np.column_stack([pts, rgb_packed])

        fields = [
            PointField(name='x',   offset=0,  datatype=PointField.FLOAT32, count=1),
            PointField(name='y',   offset=4,  datatype=PointField.FLOAT32, count=1),
            PointField(name='z',   offset=8,  datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1),
        ]

        msg = pc2.create_cloud(
            Header(stamp=stamp, frame_id=self.lidar_frame),
            fields,
            cloud_data.astype(np.float32).tolist()
        )
        self._pub_cloud.publish(msg)

    def _publish_3d_boxes(self, objects: list[FusedObject], stamp) -> None:
        """Publish oriented 3-D bounding boxes as LINE_LIST markers."""
        arr = MarkerArray()

        # Clear previous markers
        clear = Marker()
        clear.action = Marker.DELETEALL
        arr.markers.append(clear)

        for i, obj in enumerate(objects):
            if not obj.has_3d:
                continue

            box = obj.bbox_3d
            corners = box.corners          # (8, 3)
            colour  = CLASS_COLOURS_RGBA.get(obj.label, DEFAULT_COLOUR)

            # 12 edges of a box: 4 bottom, 4 top, 4 verticals
            edges = [
                (0,1),(1,2),(2,3),(3,0),   # bottom face
                (4,5),(5,6),(6,7),(7,4),   # top face
                (0,4),(1,5),(2,6),(3,7),   # vertical edges
            ]

            m = Marker()
            m.header.stamp    = stamp
            m.header.frame_id = self.lidar_frame
            m.ns              = 'boxes'
            m.id              = i
            m.type            = Marker.LINE_LIST
            m.action          = Marker.ADD
            m.scale.x         = 0.05      # line width (m)
            m.color           = colour
            m.lifetime        = rclpy.duration.Duration(seconds=0.2).to_msg()

            for a, b in edges:
                pa, pb = corners[a], corners[b]
                m.points.append(Point(x=float(pa[0]), y=float(pa[1]), z=float(pa[2])))
                m.points.append(Point(x=float(pb[0]), y=float(pb[1]), z=float(pb[2])))

            # Label
            label_m = Marker()
            label_m.header     = m.header
            label_m.ns         = 'labels'
            label_m.id         = i + 1000
            label_m.type       = Marker.TEXT_VIEW_FACING
            label_m.action     = Marker.ADD
            label_m.pose.position = Point(
                x=float(box.center[0]),
                y=float(box.center[1]),
                z=float(box.center[2]) + float(box.size[2]) / 2 + 0.3
            )
            label_m.scale.z    = 0.4
            label_m.color      = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
            label_m.text       = f'{obj.label}\n{obj.distance:.1f}m'
            label_m.lifetime   = m.lifetime

            arr.markers.extend([m, label_m])

        self._pub_boxes.publish(arr)

    def _publish_velocity_arrows(self, objects: list[FusedObject], stamp) -> None:
        """Publish Kalman velocity estimates as ARROW markers."""
        arr = MarkerArray()

        clear = Marker()
        clear.action = Marker.DELETEALL
        arr.markers.append(clear)

        for i, obj in enumerate(objects):
            if not obj.has_3d:
                continue

            # Retrieve velocity from the track that owns this object
            track = next(
                (t for t in self.tracker.tracks if t.last_fused is obj),
                None
            )
            if track is None:
                continue

            vel = track.velocity           # m/frame
            speed = np.linalg.norm(vel)
            if speed < 0.05:               # skip nearly-stationary objects
                continue

            c = obj.bbox_3d.center
            m = Marker()
            m.header.stamp    = stamp
            m.header.frame_id = self.lidar_frame
            m.ns              = 'velocities'
            m.id              = i
            m.type            = Marker.ARROW
            m.action          = Marker.ADD
            m.scale           = Vector3(x=float(speed * 2), y=0.15, z=0.15)
            m.color           = ColorRGBA(r=1.0, g=1.0, b=0.0, a=0.9)
            m.lifetime        = rclpy.duration.Duration(seconds=0.2).to_msg()
            m.points.append(Point(x=float(c[0]),               y=float(c[1]),               z=float(c[2])))
            m.points.append(Point(x=float(c[0]+vel[0]*2),     y=float(c[1]+vel[1]*2),     z=float(c[2]+vel[2]*2)))
            arr.markers.append(m)

        self._pub_tracks.publish(arr)


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = FusionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
