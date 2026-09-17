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


def make_white_wall_track(offset_near=0, offset_far=0, right_wall=True):
    image = np.zeros((240, 320, 3), dtype=np.uint8)
    image[:, :] = (70, 70, 70)
    left_wall = np.array(
        [
            [92 + offset_far, 70],
            [112 + offset_far, 70],
            [65 + offset_near, 239],
            [38 + offset_near, 239],
        ],
        dtype=np.int32,
    )
    cv2.fillPoly(image, [left_wall], (245, 245, 245))
    if right_wall:
        right_wall_polygon = np.array(
            [
                [208 + offset_far, 70],
                [228 + offset_far, 70],
                [282 + offset_near, 239],
                [255 + offset_near, 239],
            ],
            dtype=np.int32,
        )
        cv2.fillPoly(image, [right_wall_polygon], (245, 245, 245))
    return image


def make_dark_floor_track(offset_near=0, offset_far=0):
    image = np.full((240, 320, 3), 245, dtype=np.uint8)
    polygon = np.array(
        [
            [115 + offset_far, 70],
            [205 + offset_far, 70],
            [270 + offset_near, 239],
            [50 + offset_near, 239],
        ],
        dtype=np.int32,
    )
    cv2.fillPoly(image, [polygon], (155, 155, 155))
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


def test_white_wall_mode_centers_between_barriers():
    detector = CorridorDetector(
        detection_mode='white_walls',
        min_value=165,
        max_saturation=55,
        min_wall_width=2,
    )

    detection = detector.detect(make_white_wall_track())

    assert detection is not None
    assert abs(detection.near_center[0] - 160) < 5
    assert abs(detection.far_center[0] - 160) < 5
    assert detection.confidence >= 0.75


def test_white_wall_mode_tracks_gap_heading_right():
    detector = CorridorDetector(
        detection_mode='white_walls',
        min_value=165,
        max_saturation=55,
        min_wall_width=2,
    )

    detection = detector.detect(
        make_white_wall_track(offset_near=0, offset_far=30)
    )

    assert detection is not None
    assert detection.far_center[0] > detection.near_center[0]


def test_white_wall_mode_stops_when_one_boundary_is_missing():
    detector = CorridorDetector(
        detection_mode='white_walls',
        min_value=165,
        max_saturation=55,
        min_wall_width=2,
    )

    assert detector.detect(
        make_white_wall_track(right_wall=False)
    ) is None


def test_dark_corridor_mode_follows_lab_floor():
    detector = CorridorDetector(
        detection_mode='dark_corridor',
        min_value=165,
        max_saturation=55,
    )

    detection = detector.detect(make_dark_floor_track(offset_far=25))

    assert detection is not None
    assert abs(detection.near_center[0] - 160) < 5
    assert detection.far_center[0] > detection.near_center[0]
