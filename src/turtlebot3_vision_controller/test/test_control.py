"""Regression tests for continuous tracking and command safety."""

from types import SimpleNamespace

import numpy as np
import pytest
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.time import Time
from sensor_msgs.msg import Imu

from turtlebot3_vision_controller.controller import VisionCorridorController
from turtlebot3_vision_controller.path_tracking import ClosedPath, slew


def test_target_is_continuous_across_vertex():
    path = ClosedPath([(0, 0), (1, 0), (1, 1), (0, 1)])
    before = path.target((0.999, 0), 0.2)
    after = path.target((1, 0.001), 0.2)
    assert np.linalg.norm(after - before) < 0.003


def test_target_crosses_lap_seam_without_reversing():
    path = ClosedPath([(0, 0), (1, 0), (1, 1), (0, 1)])
    before = path.target((0, 0.01), 0.2)
    after = path.target((0.01, 0), 0.2)
    assert path.progress == pytest.approx(4.01)
    assert np.linalg.norm(after - before) < 0.03


def test_nearby_return_lane_does_not_steal_target():
    path = ClosedPath([(0, 0), (3, 0), (3, 0.3), (0, 0.3)])
    path.target((1.5, 0), 0.2)
    target = path.target((1.5, 0.2), 0.2)
    assert target[1] == 0.0
    assert target[0] > 1.5


@pytest.mark.parametrize('fps', [10, 30, 60])
def test_acceleration_limit_is_independent_of_frame_rate(fps):
    value = 0.0
    for _ in range(fps):
        value = slew(value, 2.0, 0.5, 1.0 / fps)
    assert value == pytest.approx(0.5)


def controller_without_ros():
    node = VisionCorridorController.__new__(VisionCorridorController)
    node._previous_linear_command = 0.2
    node._previous_angular_command = 0.0
    node._max_linear_speed = 0.2
    node._max_linear_acceleration = 0.2
    node._max_angular_acceleration = 1.5
    node._use_lidar_safety = True
    node._lidar_stop_distance = 0.14
    node._lidar_slow_distance = 0.4
    node._sensor_timeout = 0.6
    node._waypoint_origin_yaw = 0.0
    node._reverse_route = False
    return node


def test_emergency_stop_bypasses_acceleration_and_minimum_speed():
    node = controller_without_ros()
    node._front_clearance = lambda: 0.13
    command = Twist()
    command.linear.x = 0.2
    command.angular.z = 1.0
    result = node._limit_command(command, 1 / 30)
    assert result.linear.x == 0.0
    assert result.angular.z == pytest.approx(0.05)


def test_slowdown_can_go_below_minimum_cruise_speed():
    node = controller_without_ros()
    node._front_clearance = lambda: 0.15
    command = Twist()
    command.linear.x = 0.2
    result = node._limit_command(command, 1 / 30)
    assert 0 < result.linear.x < 0.01


def test_stop_clears_history_before_restarting():
    node = controller_without_ros()
    sent = []
    node._command_publisher = SimpleNamespace(publish=sent.append)
    node._previous_lateral_error = 0.5
    node._previous_frame_time = Time(seconds=1)
    node._recovery_direction = 1.0
    node._publish_stop()
    assert sent[-1].linear.x == 0.0
    assert node._previous_linear_command == 0.0
    assert node._previous_lateral_error == 0.0
    assert node._previous_frame_time is None
    assert node._recovery_direction is None


def test_missing_stale_or_future_sensor_data_is_not_fresh():
    node = controller_without_ros()
    node.get_clock = lambda: SimpleNamespace(now=lambda: Time(seconds=2))
    assert not node._sensor_fresh(None)
    assert not node._sensor_fresh(Time(seconds=1))
    assert not node._sensor_fresh(Time(seconds=3))
    assert node._sensor_fresh(Time(seconds=1.5))


def test_imu_corrects_heading_of_wheel_distance():
    node = controller_without_ros()
    node.get_clock = lambda: SimpleNamespace(now=lambda: Time(seconds=2))
    node._use_imu_heading = True
    node._last_imu_time = Time(seconds=2)
    node._imu_yaw = 0.0
    node._previous_imu_yaw = 0.0
    node._previous_odometry_pose = np.zeros(2)
    node._fused_position = np.zeros(2)
    message = Odometry()
    # Wheels falsely report a northward heading, while the IMU points east.
    message.pose.pose.position.y = 0.1
    message.pose.pose.orientation.z = np.sin(np.pi / 4)
    message.pose.pose.orientation.w = np.cos(np.pi / 4)
    node._odometry_callback(message)
    np.testing.assert_allclose(node._fused_position, [0.1, 0.0], atol=1e-8)


def test_imu_heading_is_relative_to_start_and_wraps_at_pi():
    node = controller_without_ros()
    node.get_clock = lambda: SimpleNamespace(now=lambda: Time(seconds=2))
    node._imu_origin = None
    for yaw in (np.pi - 0.1, -np.pi + 0.1):
        message = Imu()
        message.orientation.z = np.sin(yaw / 2)
        message.orientation.w = np.cos(yaw / 2)
        node._imu_callback(message)
    assert node._imu_yaw == pytest.approx(0.2)


def test_queued_camera_callback_cannot_restart_stopped_controller():
    node = controller_without_ros()
    node._stopping = True
    # A queued callback must return before touching the image or publishing.
    node._image_callback(None)


def test_route_command_obeys_lidar_stop():
    node = controller_without_ros()
    node._path = None
    node._waypoint_origin_x = 2.2
    node._waypoint_origin_y = 0.65
    node._waypoint_lookahead = 0.22
    node._max_angular_speed = 1.0
    node._turn_slowdown = 0.75
    node._use_imu_heading = False
    node._latest_odometry = Odometry()
    node._latest_odometry.pose.pose.orientation.w = 1.0
    node._front_clearance = lambda: 0.13
    assert node._waypoint_command(1 / 30).linear.x == 0.0


@pytest.mark.parametrize('reverse,spawn_yaw,world_target_x', [
    (False, 0.0, 2.42), (True, np.pi, 1.98),
])
def test_route_direction_matches_rotated_spawn(reverse, spawn_yaw, world_target_x):
    node = controller_without_ros()
    node._path = None
    node._waypoint_origin_x = 2.2
    node._waypoint_origin_y = 0.65
    node._waypoint_origin_yaw = spawn_yaw
    node._reverse_route = reverse
    node._waypoint_lookahead = 0.22
    node._max_angular_speed = 1.0
    node._turn_slowdown = 0.75
    node._use_imu_heading = False
    node._latest_odometry = Odometry()
    node._latest_odometry.pose.pose.orientation.w = 1.0
    node._front_clearance = lambda: 1.0
    command = node._waypoint_command(1 / 30)
    assert command.linear.x > 0
    assert command.angular.z == pytest.approx(0, abs=1e-8)
    target = node._path.target((0, 0), 0.22)
    c, s = np.cos(spawn_yaw), np.sin(spawn_yaw)
    world_target = np.array([[c, -s], [s, c]]) @ target + [2.2, 0.65]
    np.testing.assert_allclose(world_target, [world_target_x, 0.65], atol=1e-8)
