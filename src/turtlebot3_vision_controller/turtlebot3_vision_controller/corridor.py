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
    """Track either a bright floor or the gap between two bright walls."""

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
        detection_mode: str = 'bright_corridor',
        min_wall_width: int = 3,
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
        if detection_mode not in (
            'bright_corridor',
            'dark_corridor',
            'white_walls',
        ):
            raise ValueError(
                'detection_mode must be bright_corridor, dark_corridor, '
                'or white_walls'
            )
        self.detection_mode = detection_mode
        self.min_wall_width = max(1, min_wall_width)
        # The compact lab course can push one boundary outside the camera
        # image in a bend.  Retain a nominal pixel gap for that brief case.
        self._last_gap_width: Optional[float] = None
        self._last_center_x: Optional[float] = None

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

        if self.detection_mode in ('bright_corridor', 'dark_corridor'):
            if self.detection_mode == 'dark_corridor':
                mask = cv2.bitwise_not(mask)
            detection_mask = self._select_corridor_component(
                mask,
                image_width,
            )
            if detection_mask is None:
                return None
        else:
            detection_mask = mask

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

        expected_x = (
            self._last_center_x
            if self._last_center_x is not None
            else image_width / 2.0
        )
        centers_roi: list[tuple[int, int]] = []
        spans_roi: list[tuple[int, int, int]] = []
        bounded_rows = 0
        for row in sample_rows:
            if self.detection_mode == 'white_walls':
                run = self._choose_wall_gap(
                    detection_mask,
                    row,
                    expected_x,
                )
            else:
                run = self._choose_run(
                    detection_mask,
                    row,
                    expected_x,
                )
            if run is None:
                continue
            left, right = run
            if self.detection_mode == 'white_walls':
                self._last_gap_width = float(right - left + 1)
            if (
                self.detection_mode == 'dark_corridor'
                and left <= 1
                and right >= image_width - 2
            ):
                # Unbounded dark floor is not a valid lab corridor. This
                # prevents the exterior floor from becoming a false track.
                continue
            center_x = int(round((left + right) / 2.0))
            centers_roi.append((center_x, int(row)))
            spans_roi.append((left, right, int(row)))
            if left > 1 and right < image_width - 2:
                bounded_rows += 1
            expected_x = center_x

        minimum_rows = max(3, int(np.ceil(self.sample_row_count * 0.5)))
        if len(centers_roi) < minimum_rows:
            return None
        if self.detection_mode == 'dark_corridor' and bounded_rows < 2:
            return None

        # The near sample is the best estimate of the robot's current lane
        # center.  Do not let a far divider overwrite it for one-wall holds.
        if centers_roi:
            self._last_center_x = float(centers_roi[0][0])

        confidence = len(centers_roi) / float(self.sample_row_count)
        centers = [(x, y + roi_top) for x, y in centers_roi]
        spans = [(left, right, y + roi_top) for left, right, y in spans_roi]

        full_mask = np.zeros((image_height, image_width), dtype=np.uint8)
        full_mask[roi_top:, :] = detection_mask
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
            candidates.append((
                abs(center - expected_x),
                -width,
                int(left),
                int(right),
            ))

        if not candidates:
            return None
        _, _, left, right = min(candidates)
        return left, right

    def _choose_wall_gap(
        self,
        mask: np.ndarray,
        row: int,
        expected_x: float,
    ) -> Optional[tuple[int, int]]:
        """Return the free-space gap bounded by adjacent white wall runs."""
        half_band = self.row_band_height // 2
        start = max(0, row - half_band)
        stop = min(mask.shape[0], row + half_band + 1)
        row_mask = np.any(mask[start:stop, :] > 0, axis=0).astype(np.uint8)

        padded = np.pad(row_mask, (1, 1), constant_values=0)
        changes = np.diff(padded.astype(np.int8))
        starts = np.where(changes == 1)[0]
        stops = np.where(changes == -1)[0] - 1
        wall_runs = [
            (int(left), int(right))
            for left, right in zip(starts, stops)
            if right - left + 1 >= self.min_wall_width
        ]

        candidates = []
        for left_wall, right_wall in zip(wall_runs, wall_runs[1:]):
            gap_left = left_wall[1] + 1
            gap_right = right_wall[0] - 1
            gap_width = gap_right - gap_left + 1
            if gap_width < self.min_corridor_width:
                continue

            center = (gap_left + gap_right) / 2.0
            contains_expected = gap_left <= expected_x <= gap_right
            # Prefer the gap containing the previous center estimate. This
            # prevents distant divider walls from stealing the detection.
            candidates.append((
                0 if contains_expected else 1,
                abs(center - expected_x),
                -gap_width,
                gap_left,
                gap_right,
            ))

        if not candidates:
            # If only one wall is visible, use the previous gap width and the
            # image edge as a temporary virtual boundary.  This keeps the
            # robot moving away from the visible wall instead of stopping at
            # every bend where the opposite wall leaves the frame.
            if len(wall_runs) != 1:
                return None
            wall_left, wall_right = wall_runs[0]
            # Only infer a missing boundary when the visible wall itself is
            # clipped by the image edge.  An isolated wall in the middle of
            # the frame is not enough evidence to drive.
            if wall_left != 0 and wall_right != mask.shape[1] - 1:
                return None
            if wall_left == 0 and wall_right == mask.shape[1] - 1:
                return None
            gap_width = self._last_gap_width
            if gap_width is None:
                gap_width = mask.shape[1] * 0.65
            # Do not infer a turn from the clipped wall location.  Preserve
            # the last midpoint until both boundaries are visible again.
            center = self._last_center_x
            if center is None:
                center = expected_x
            gap_left = int(round(center - gap_width / 2.0))
            gap_right = int(round(center + gap_width / 2.0 - 1.0))
            return max(0, gap_left), min(mask.shape[1] - 1, gap_right)
        _, _, _, left, right = min(candidates)
        return left, right
