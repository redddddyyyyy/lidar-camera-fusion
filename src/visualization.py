"""
Visualization utilities for LiDAR-Camera fusion.

Provides:
- 2D detection overlay on images
- 3D bounding box projection onto images
- Bird's eye view (BEV) visualization
- Point cloud coloring by depth/intensity
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.collections import LineCollection
import matplotlib.cm as cm

from .calibration import LidarCameraCalibration
from .lidar_processor import PointCloud, BoundingBox3D
from .detector import Detection, YOLODetector
from .fusion import FusedObject


class FusionVisualizer:
    """Visualization tools for sensor fusion results."""

    # Color scheme
    COLORS = {
        'car': (0, 255, 0),
        'truck': (0, 200, 100),
        'bus': (0, 150, 255),
        'motorcycle': (255, 100, 0),
        'bicycle': (255, 255, 0),
        'person': (255, 0, 0),
        'default': (128, 128, 128),
    }

    def __init__(self, calibration: LidarCameraCalibration):
        """
        Initialize visualizer.

        Args:
            calibration: LiDAR-Camera calibration for projections
        """
        self.calibration = calibration

    def draw_detections_2d(
        self,
        image: np.ndarray,
        detections: List[Detection],
        show_labels: bool = True,
        thickness: int = 2
    ) -> np.ndarray:
        """
        Draw 2D bounding boxes on image.

        Args:
            image: RGB image (H, W, 3)
            detections: List of 2D detections
            show_labels: Whether to show class labels
            thickness: Box line thickness

        Returns:
            Image with drawn detections
        """
        import cv2

        result = image.copy()

        for det in detections:
            x1, y1, x2, y2 = det.bbox
            color = self.COLORS.get(det.label, self.COLORS['default'])

            # Draw box
            cv2.rectangle(result, (x1, y1), (x2, y2), color, thickness)

            # Draw label
            if show_labels:
                label_text = f"{det.label}: {det.score:.2f}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                text_thickness = 1

                (text_w, text_h), _ = cv2.getTextSize(
                    label_text, font, font_scale, text_thickness
                )

                # Background rectangle for text
                cv2.rectangle(
                    result,
                    (x1, y1 - text_h - 4),
                    (x1 + text_w, y1),
                    color,
                    -1
                )

                # Text
                cv2.putText(
                    result,
                    label_text,
                    (x1, y1 - 2),
                    font,
                    font_scale,
                    (255, 255, 255),
                    text_thickness
                )

        return result

    def draw_fused_objects(
        self,
        image: np.ndarray,
        fused_objects: List[FusedObject],
        show_distance: bool = True,
        show_3d_box: bool = True
    ) -> np.ndarray:
        """
        Draw fused detection results.

        Shows 2D boxes with distance and projected 3D boxes.

        Args:
            image: RGB image
            fused_objects: List of fused objects
            show_distance: Show distance annotation
            show_3d_box: Project and draw 3D bounding box

        Returns:
            Annotated image
        """
        import cv2

        result = image.copy()

        for obj in fused_objects:
            det = obj.detection_2d
            x1, y1, x2, y2 = det.bbox
            color = self.COLORS.get(det.label, self.COLORS['default'])

            # Draw 2D box
            cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)

            # Draw 3D box if available
            if show_3d_box and obj.has_3d:
                result = self._draw_3d_box(result, obj.bbox_3d, color)

            # Draw label with distance
            if show_distance and obj.has_3d:
                label_text = f"{det.label}: {obj.distance:.1f}m"
            else:
                label_text = f"{det.label}: {det.score:.2f}"

            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(result, label_text, (x1, y1 - 5), font, 0.5, color, 2)

        return result

    def _draw_3d_box(
        self,
        image: np.ndarray,
        bbox_3d: BoundingBox3D,
        color: Tuple[int, int, int]
    ) -> np.ndarray:
        """Draw projected 3D bounding box on image."""
        import cv2

        # Get 3D corners
        corners_3d = bbox_3d.corners  # 8x3

        # Project to image
        pixels, depths, valid = self.calibration.project_points(corners_3d)

        if not np.all(valid):
            return image

        # Convert to integer pixel coordinates
        pixels = pixels.astype(int)

        # Draw 12 edges of the box
        # Bottom face: 0-1-2-3-0
        # Top face: 4-5-6-7-4
        # Vertical edges: 0-4, 1-5, 2-6, 3-7
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),  # Bottom
            (4, 5), (5, 6), (6, 7), (7, 4),  # Top
            (0, 4), (1, 5), (2, 6), (3, 7),  # Vertical
        ]

        for i, j in edges:
            pt1 = tuple(pixels[i])
            pt2 = tuple(pixels[j])
            cv2.line(image, pt1, pt2, color, 2)

        return image

    def draw_lidar_on_image(
        self,
        image: np.ndarray,
        point_cloud: PointCloud,
        color_by: str = "depth",
        point_size: int = 2,
        max_depth: float = 50.0
    ) -> np.ndarray:
        """
        Overlay LiDAR points on image.

        Args:
            image: RGB image
            point_cloud: LiDAR point cloud
            color_by: "depth" or "intensity"
            point_size: Size of drawn points
            max_depth: Maximum depth for colormap normalization

        Returns:
            Image with LiDAR overlay
        """
        import cv2

        result = image.copy()

        # Project points
        pixels, depths, indices = self.calibration.project_to_image(
            point_cloud.points,
            image.shape[:2]
        )

        if len(pixels) == 0:
            return result

        # Get colors
        if color_by == "depth":
            values = depths / max_depth
        elif color_by == "intensity" and point_cloud.intensities is not None:
            values = point_cloud.intensities[indices]
            values = values / (values.max() + 1e-6)
        else:
            values = depths / max_depth

        # Apply colormap
        colors = plt.cm.jet(values)[:, :3] * 255

        # Draw points
        for pixel, color in zip(pixels.astype(int), colors.astype(int)):
            cv2.circle(result, tuple(pixel), point_size, tuple(color.tolist()), -1)

        return result

    def create_bev_plot(
        self,
        point_cloud: PointCloud,
        fused_objects: List[FusedObject] | None = None,
        x_range: Tuple[float, float] = (-30, 30),
        y_range: Tuple[float, float] = (0, 60),
        resolution: float = 0.1
    ) -> plt.Figure:
        """
        Create bird's eye view visualization.

        Args:
            point_cloud: LiDAR point cloud
            fused_objects: Optional list of fused objects
            x_range: X-axis range (left-right)
            y_range: Y-axis range (forward)
            resolution: Meters per pixel

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 12))

        # Filter points to range
        points = point_cloud.points
        mask = (
            (points[:, 0] >= x_range[0]) & (points[:, 0] <= x_range[1]) &
            (points[:, 1] >= y_range[0]) & (points[:, 1] <= y_range[1])
        )
        points = points[mask]

        # Create BEV image
        bev_width = int((x_range[1] - x_range[0]) / resolution)
        bev_height = int((y_range[1] - y_range[0]) / resolution)

        # Map points to BEV pixels
        px = ((points[:, 0] - x_range[0]) / resolution).astype(int)
        py = ((y_range[1] - points[:, 1]) / resolution).astype(int)  # Flip Y

        # Clip to bounds
        px = np.clip(px, 0, bev_width - 1)
        py = np.clip(py, 0, bev_height - 1)

        # Create density image
        bev = np.zeros((bev_height, bev_width), dtype=np.float32)
        np.add.at(bev, (py, px), 1)

        # Log scale for better visualization
        bev = np.log1p(bev)

        ax.imshow(bev, cmap='gray', origin='upper',
                 extent=[x_range[0], x_range[1], y_range[0], y_range[1]])

        # Draw fused objects
        if fused_objects:
            for obj in fused_objects:
                if obj.has_3d:
                    box = obj.bbox_3d
                    color = np.array(self.COLORS.get(obj.label, self.COLORS['default'])) / 255

                    # Draw box outline
                    corners = box.corners[:4, :2]  # Bottom 4 corners, XY only
                    corners = np.vstack([corners, corners[0]])  # Close polygon

                    ax.plot(corners[:, 0], corners[:, 1], '-', color=color, linewidth=2)

                    # Label
                    ax.text(box.center[0], box.center[1],
                           f"{obj.label}\n{obj.distance:.1f}m",
                           fontsize=8, ha='center', color='white',
                           bbox=dict(boxstyle='round', facecolor=color, alpha=0.7))

        # Draw ego vehicle
        ego_corners = np.array([[-1, -2], [1, -2], [1, 0], [0, 1], [-1, 0], [-1, -2]])
        ax.fill(ego_corners[:, 0], ego_corners[:, 1], 'blue', alpha=0.7)

        ax.set_xlabel('X (m) - Left/Right')
        ax.set_ylabel('Y (m) - Forward')
        ax.set_title("Bird's Eye View")
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def create_fusion_summary(
        self,
        image: np.ndarray,
        point_cloud: PointCloud,
        fused_objects: List[FusedObject]
    ) -> plt.Figure:
        """
        Create comprehensive fusion visualization.

        Shows:
        - Top: Image with detections and LiDAR overlay
        - Bottom left: Bird's eye view
        - Bottom right: Detection statistics

        Returns:
            Matplotlib figure
        """
        fig = plt.figure(figsize=(16, 12))

        # Top: Camera view with fusion results
        ax1 = fig.add_subplot(2, 2, (1, 2))
        vis_image = self.draw_fused_objects(image, fused_objects)
        vis_image = self.draw_lidar_on_image(vis_image, point_cloud, point_size=1)
        ax1.imshow(vis_image)
        ax1.set_title('Camera View with LiDAR Overlay')
        ax1.axis('off')

        # Bottom left: BEV
        ax2 = fig.add_subplot(2, 2, 3)

        # Simple BEV plot
        points = point_cloud.points
        mask = (points[:, 1] > 0) & (points[:, 1] < 60) & \
               (np.abs(points[:, 0]) < 30)
        ax2.scatter(points[mask, 0], points[mask, 1], c=points[mask, 2],
                   cmap='viridis', s=0.5, alpha=0.5)

        for obj in fused_objects:
            if obj.has_3d:
                color = np.array(self.COLORS.get(obj.label, self.COLORS['default'])) / 255
                corners = obj.bbox_3d.corners[:4, :2]
                corners = np.vstack([corners, corners[0]])
                ax2.plot(corners[:, 0], corners[:, 1], '-', color=color, linewidth=2)

        ax2.set_xlim(-30, 30)
        ax2.set_ylim(0, 60)
        ax2.set_xlabel('X (m)')
        ax2.set_ylabel('Y (m)')
        ax2.set_title("Bird's Eye View")
        ax2.set_aspect('equal')
        ax2.grid(True, alpha=0.3)

        # Bottom right: Statistics
        ax3 = fig.add_subplot(2, 2, 4)
        ax3.axis('off')

        stats_text = "Detection Summary\n"
        stats_text += "=" * 30 + "\n\n"
        stats_text += f"Total Objects: {len(fused_objects)}\n"
        stats_text += f"With 3D Info: {sum(1 for o in fused_objects if o.has_3d)}\n\n"

        # Count by class
        class_counts = {}
        for obj in fused_objects:
            label = obj.label
            if label not in class_counts:
                class_counts[label] = {'total': 0, 'with_3d': 0, 'min_dist': float('inf')}
            class_counts[label]['total'] += 1
            if obj.has_3d:
                class_counts[label]['with_3d'] += 1
                class_counts[label]['min_dist'] = min(
                    class_counts[label]['min_dist'], obj.distance
                )

        stats_text += "By Class:\n"
        for label, counts in class_counts.items():
            dist_str = f"{counts['min_dist']:.1f}m" if counts['with_3d'] > 0 else "N/A"
            stats_text += f"  {label}: {counts['total']} ({counts['with_3d']} 3D) - nearest: {dist_str}\n"

        ax3.text(0.1, 0.9, stats_text, transform=ax3.transAxes,
                fontsize=12, verticalalignment='top', fontfamily='monospace')

        plt.tight_layout()
        return fig
