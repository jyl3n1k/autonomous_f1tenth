import cv2
import numpy as np

from turtlebot3_vision_controller.corridor import CorridorDetector


def make_track(offset_near=0, offset_far=0):
    image = np.zeros((240, 320, 3), dtype=np.uint8)
    image[:, :] = (0, 190, 245)  # Yellow exterior in BGR.
    polygon = np.array(
        [
            [105 + offset_far, 80],
            [215 + offset_far, 80],
            [265 + offset_near, 239],
            [55 + offset_near, 239],
        ],
        dtype=np.int32,
    )
    cv2.fillPoly(image, [polygon], (205, 205, 205))
    return image


def test_detects_centered_corridor():
    detection = CorridorDetector().detect(make_track())

    assert detection is not None
    assert abs(detection.near_center[0] - 160) < 5
    assert abs(detection.far_center[0] - 160) < 5
    assert detection.confidence >= 0.75


def test_detects_corridor_heading_to_right():
    detection = CorridorDetector().detect(
        make_track(offset_near=0, offset_far=35)
    )

    assert detection is not None
    assert detection.far_center[0] > detection.near_center[0]


def test_returns_none_without_visible_corridor():
    yellow_image = np.zeros((240, 320, 3), dtype=np.uint8)
    yellow_image[:, :] = (0, 190, 245)

    assert CorridorDetector().detect(yellow_image) is None
