"""OpenCV utilities for finding the center of a bright track corridor."""

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass
class CorridorDetection:
    """Result of corridor detection in full-image pixel coordinates."""

    near_center: tuple[int, int]
    far_center: tuple[int, int]
    centers: list[tuple[int, int]]
    spans: list[tuple[int, int, int]]
    confidence: float
    mask: np.ndarray
    roi_top: int


class CorridorDetector:
    """Segment and track a low-saturation, bright driving corridor."""

    def __init__(
        self,
        roi_top_fraction: float = 0.35,
        max_saturation: int = 85,
        min_value: int = 105,
        morphology_kernel: int = 5,
        min_component_area: int = 350,
        near_row_fraction: float = 0.86,
        far_row_fraction: float = 0.38,
        sample_row_count: int = 8,
        row_band_height: int = 5,
        min_corridor_width: int = 8,
    ) -> None:
        self.roi_top_fraction = roi_top_fraction
        self.max_saturation = max_saturation
        self.min_value = min_value
        self.morphology_kernel = max(1, morphology_kernel)
        self.min_component_area = min_component_area
        self.near_row_fraction = near_row_fraction
        self.far_row_fraction = far_row_fraction
        self.sample_row_count = max(2, sample_row_count)
        self.row_band_height = max(1, row_band_height)
        self.min_corridor_width = max(2, min_corridor_width)

    def detect(self, bgr_image: np.ndarray) -> Optional[CorridorDetection]:
        """Return corridor geometry, or ``None`` when it is not visible."""
        if bgr_image is None or bgr_image.ndim != 3:
            return None

        image_height, image_width = bgr_image.shape[:2]
        roi_top = int(np.clip(
            self.roi_top_fraction * image_height,
            0,
            image_height - 2,
        ))
        roi = bgr_image[roi_top:, :]
        roi_height = roi.shape[0]

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(
            hsv,
            np.array([0, 0, self.min_value], dtype=np.uint8),
            np.array([179, self.max_saturation, 255], dtype=np.uint8),
        )

        kernel_size = self.morphology_kernel
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

        corridor_mask = self._select_corridor_component(mask, image_width)
        if corridor_mask is None:
            return None

        near_row = int(np.clip(
            self.near_row_fraction * (roi_height - 1),
            0,
            roi_height - 1,
        ))
        far_row = int(np.clip(
            self.far_row_fraction * (roi_height - 1),
            0,
            roi_height - 1,
        ))
        if far_row >= near_row:
            far_row = max(0, near_row - 1)

        sample_rows = np.linspace(
            near_row,
            far_row,
            self.sample_row_count,
        ).astype(int)

        expected_x = image_width / 2.0
        centers_roi: list[tuple[int, int]] = []
        spans_roi: list[tuple[int, int, int]] = []
        for row in sample_rows:
            run = self._choose_run(corridor_mask, row, expected_x)
            if run is None:
                continue
            left, right = run
            center_x = int(round((left + right) / 2.0))
            centers_roi.append((center_x, int(row)))
            spans_roi.append((left, right, int(row)))
            expected_x = center_x

        minimum_rows = max(3, int(np.ceil(self.sample_row_count * 0.5)))
        if len(centers_roi) < minimum_rows:
            return None

        confidence = len(centers_roi) / float(self.sample_row_count)
        centers = [(x, y + roi_top) for x, y in centers_roi]
        spans = [(left, right, y + roi_top) for left, right, y in spans_roi]

        full_mask = np.zeros((image_height, image_width), dtype=np.uint8)
        full_mask[roi_top:, :] = corridor_mask
        return CorridorDetection(
            near_center=centers[0],
            far_center=centers[-1],
            centers=centers,
            spans=spans,
            confidence=confidence,
            mask=full_mask,
            roi_top=roi_top,
        )

    def _select_corridor_component(
        self,
        mask: np.ndarray,
        image_width: int,
    ) -> Optional[np.ndarray]:
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        if count <= 1:
            return None

        image_center = image_width / 2.0
        mask_height = mask.shape[0]
        bottom_band_top = int(mask_height * 0.72)
        best_label = None
        best_score = -np.inf

        for label in range(1, count):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < self.min_component_area:
                continue

            component_y = int(stats[label, cv2.CC_STAT_TOP])
            component_h = int(stats[label, cv2.CC_STAT_HEIGHT])
            component_bottom = component_y + component_h - 1
            if component_bottom < bottom_band_top:
                continue

            bottom_pixels = np.argwhere(
                labels[bottom_band_top:, :] == label
            )
            if bottom_pixels.size == 0:
                continue
            closest_x_distance = float(np.min(np.abs(
                bottom_pixels[:, 1] - image_center
            )))
            score = area - (12.0 * closest_x_distance)
            if score > best_score:
                best_score = score
                best_label = label

        if best_label is None:
            return None
        return np.where(labels == best_label, 255, 0).astype(np.uint8)

    def _choose_run(
        self,
        mask: np.ndarray,
        row: int,
        expected_x: float,
    ) -> Optional[tuple[int, int]]:
        half_band = self.row_band_height // 2
        start = max(0, row - half_band)
        stop = min(mask.shape[0], row + half_band + 1)
        row_mask = np.any(mask[start:stop, :] > 0, axis=0).astype(np.uint8)

        padded = np.pad(row_mask, (1, 1), constant_values=0)
        changes = np.diff(padded.astype(np.int8))
        starts = np.where(changes == 1)[0]
        stops = np.where(changes == -1)[0] - 1

        candidates = []
        for left, right in zip(starts, stops):
            width = int(right - left + 1)
            if width < self.min_corridor_width:
                continue
            center = (left + right) / 2.0
            candidates.append((abs(center - expected_x), -width, int(left), int(right)))

        if not candidates:
            return None
        _, _, left, right = min(candidates)
        return left, right
