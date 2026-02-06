"""
YOLO object detection wrapper.

Supports:
- YOLOv8 (Ultralytics)
- YOLOv5
- Custom model loading
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np


@dataclass
class Detection:
    """
    2D object detection result.

    Attributes:
        bbox: (x1, y1, x2, y2) bounding box coordinates
        label: Class label string
        class_id: Class index
        score: Detection confidence [0, 1]
    """

    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    label: str
    class_id: int
    score: float

    @property
    def center(self) -> Tuple[float, float]:
        """Bounding box center."""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    @property
    def width(self) -> int:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> int:
        return self.bbox[3] - self.bbox[1]

    @property
    def area(self) -> int:
        return self.width * self.height


class YOLODetector:
    """
    YOLO-based object detector.

    Wraps Ultralytics YOLOv8 for inference.
    """

    # COCO classes relevant for autonomous driving
    VEHICLE_CLASSES = {'car', 'truck', 'bus', 'motorcycle', 'bicycle'}
    PEDESTRIAN_CLASSES = {'person'}
    RELEVANT_CLASSES = VEHICLE_CLASSES | PEDESTRIAN_CLASSES

    def __init__(
        self,
        model_path: str | Path = "yolov8n.pt",
        confidence_threshold: float = 0.5,
        nms_threshold: float = 0.45,
        device: str = "auto",
        filter_classes: bool = True
    ):
        """
        Initialize YOLO detector.

        Args:
            model_path: Path to YOLO weights or model name (e.g., "yolov8n.pt")
            confidence_threshold: Minimum detection confidence
            nms_threshold: NMS IoU threshold
            device: "auto", "cpu", "cuda", or "cuda:0"
            filter_classes: If True, only return driving-relevant classes
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.device = device
        self.filter_classes = filter_classes

        self.model = None
        self.class_names = []

        self._load_model()

    def _load_model(self) -> None:
        """Load YOLO model."""
        try:
            from ultralytics import YOLO
            self.model = YOLO(str(self.model_path))

            if self.device != "auto":
                self.model.to(self.device)

            # Get class names
            self.class_names = self.model.names

        except ImportError:
            print("Warning: ultralytics not installed. Using mock detector.")
            print("Install with: pip install ultralytics")
            self.model = None
            self.class_names = {
                0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle',
                4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck'
            }

    def detect(self, image: np.ndarray) -> List[Detection]:
        """
        Run detection on image.

        Args:
            image: RGB image as numpy array (H, W, 3)

        Returns:
            List of Detection objects
        """
        if self.model is None:
            return self._mock_detect(image)

        # Run inference
        results = self.model(
            image,
            conf=self.confidence_threshold,
            iou=self.nms_threshold,
            verbose=False
        )

        detections = []

        for result in results:
            boxes = result.boxes

            if boxes is None:
                continue

            for i in range(len(boxes)):
                # Get box coordinates
                xyxy = boxes.xyxy[i].cpu().numpy()
                x1, y1, x2, y2 = map(int, xyxy)

                # Get class and confidence
                cls_id = int(boxes.cls[i].cpu().numpy())
                conf = float(boxes.conf[i].cpu().numpy())

                # Get label
                label = self.class_names.get(cls_id, f"class_{cls_id}")

                # Filter to relevant classes
                if self.filter_classes and label not in self.RELEVANT_CLASSES:
                    continue

                detection = Detection(
                    bbox=(x1, y1, x2, y2),
                    label=label,
                    class_id=cls_id,
                    score=conf
                )
                detections.append(detection)

        return detections

    def _mock_detect(self, image: np.ndarray) -> List[Detection]:
        """Mock detection for testing without YOLO installed."""
        h, w = image.shape[:2]

        # Return some fake detections for testing
        mock_detections = [
            Detection(
                bbox=(int(w * 0.3), int(h * 0.4), int(w * 0.5), int(h * 0.7)),
                label="car",
                class_id=2,
                score=0.85
            ),
            Detection(
                bbox=(int(w * 0.6), int(h * 0.5), int(w * 0.75), int(h * 0.8)),
                label="car",
                class_id=2,
                score=0.72
            ),
            Detection(
                bbox=(int(w * 0.1), int(h * 0.3), int(w * 0.15), int(h * 0.6)),
                label="person",
                class_id=0,
                score=0.68
            ),
        ]

        return mock_detections

    def detect_batch(self, images: List[np.ndarray]) -> List[List[Detection]]:
        """
        Run detection on a batch of images.

        Args:
            images: List of RGB images

        Returns:
            List of detection lists (one per image)
        """
        all_detections = []

        for image in images:
            detections = self.detect(image)
            all_detections.append(detections)

        return all_detections

    @staticmethod
    def get_class_color(label: str) -> Tuple[int, int, int]:
        """Get visualization color for a class."""
        colors = {
            'car': (0, 255, 0),       # Green
            'truck': (0, 200, 100),   # Teal
            'bus': (0, 150, 255),     # Orange
            'motorcycle': (255, 100, 0),  # Blue
            'bicycle': (255, 255, 0), # Cyan
            'person': (0, 0, 255),    # Red
        }
        return colors.get(label, (128, 128, 128))  # Gray default


class DetectionFilter:
    """
    Filter and post-process detections.
    """

    @staticmethod
    def filter_by_score(
        detections: List[Detection],
        min_score: float = 0.5
    ) -> List[Detection]:
        """Filter detections by confidence score."""
        return [d for d in detections if d.score >= min_score]

    @staticmethod
    def filter_by_area(
        detections: List[Detection],
        min_area: int = 100,
        max_area: int = 500000
    ) -> List[Detection]:
        """Filter detections by bounding box area."""
        return [d for d in detections if min_area <= d.area <= max_area]

    @staticmethod
    def filter_by_class(
        detections: List[Detection],
        allowed_classes: set[str]
    ) -> List[Detection]:
        """Filter detections by class label."""
        return [d for d in detections if d.label in allowed_classes]

    @staticmethod
    def nms(
        detections: List[Detection],
        iou_threshold: float = 0.5
    ) -> List[Detection]:
        """Apply non-maximum suppression."""
        if not detections:
            return []

        # Sort by score (descending)
        detections = sorted(detections, key=lambda d: d.score, reverse=True)

        keep = []

        while detections:
            best = detections.pop(0)
            keep.append(best)

            detections = [
                d for d in detections
                if DetectionFilter._iou(best.bbox, d.bbox) < iou_threshold
            ]

        return keep

    @staticmethod
    def _iou(box1: Tuple, box2: Tuple) -> float:
        """Compute IoU between two boxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        inter_area = max(0, x2 - x1) * max(0, y2 - y1)

        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

        union_area = box1_area + box2_area - inter_area

        return inter_area / union_area if union_area > 0 else 0
