"""ROS 2 node that keeps TurtleBot3 centered in a visible track corridor."""

from typing import Optional

import cv2
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.signals import SignalHandlerOptions
from sensor_msgs.msg import Image, Imu, LaserScan

from .corridor import CorridorDetection, CorridorDetector
from .path_tracking import ClosedPath, slew


class VisionCorridorController(Node):
    """Drive a TurtleBot3 by following the center of the visible corridor."""

    def __init__(self) -> None:
        super().__init__('turtlebot3_vision_controller')

        self._declare_parameters()
        self._bridge = CvBridge()
        self._detector = CorridorDetector(
            detection_mode=self._parameter('detection_mode'),
            roi_top_fraction=self._parameter('roi_top_fraction'),
            max_saturation=self._parameter('max_saturation'),
            min_value=self._parameter('min_value'),
            morphology_kernel=self._parameter('morphology_kernel'),
            min_component_area=self._parameter('min_component_area'),
            near_row_fraction=self._parameter('near_row_fraction'),
            far_row_fraction=self._parameter('far_row_fraction'),
            sample_row_count=self._parameter('sample_row_count'),
            row_band_height=self._parameter('row_band_height'),
            min_corridor_width=self._parameter('min_corridor_width'),
            min_wall_width=self._parameter('min_wall_width'),
        )

        self._kp_lateral = float(self._parameter('kp_lateral'))
        self._kd_lateral = float(self._parameter('kd_lateral'))
        self._kp_heading = float(self._parameter('kp_heading'))
        self._kp_heading_right = float(self._parameter('kp_heading_right'))
        self._heading_deadband = float(self._parameter('heading_deadband'))
        self._max_linear_speed = float(self._parameter('max_linear_speed'))
        self._min_linear_speed = float(self._parameter('min_linear_speed'))
        self._max_angular_speed = float(self._parameter('max_angular_speed'))
        self._turn_slowdown = float(self._parameter('turn_slowdown'))
        self._steering_smoothing = float(self._parameter('steering_smoothing'))
        self._minimum_confidence = float(self._parameter('minimum_confidence'))
        self._image_timeout = float(self._parameter('image_timeout'))
        self._publish_debug = bool(self._parameter('publish_debug'))
        self._use_lab_waypoints = bool(self._parameter('use_lab_waypoints'))
        self._waypoint_recovery = bool(self._parameter('waypoint_recovery'))
        self._waypoint_origin_x = float(self._parameter('waypoint_origin_x'))
        self._waypoint_origin_y = float(self._parameter('waypoint_origin_y'))
        self._waypoint_origin_yaw = float(self._parameter('waypoint_origin_yaw'))
        self._reverse_route = bool(self._parameter('reverse_route'))
        self._waypoint_lookahead = float(self._parameter('waypoint_lookahead'))
        self._max_linear_acceleration = float(
            self._parameter('max_linear_acceleration'))
        self._max_angular_acceleration = float(
            self._parameter('max_angular_acceleration'))
        self._sensor_timeout = float(self._parameter('sensor_timeout'))
        self._path = None
        self._use_imu_heading = bool(self._parameter('use_imu_heading'))
        self._imu_yaw = None
        self._imu_origin = None
        self._last_imu_time = None
        self._fused_position = np.zeros(2)
        self._previous_odometry_pose = None
        self._previous_imu_yaw = None
        self._use_lidar_safety = bool(self._parameter('use_lidar_safety'))
        self._lidar_stop_distance = float(
            self._parameter('lidar_stop_distance')
        )
        self._lidar_slow_distance = float(
            self._parameter('lidar_slow_distance')
        )
        self._lidar_corner_gain = float(self._parameter('lidar_corner_gain'))
        self._lidar_center_gain = float(self._parameter('lidar_center_gain'))
        if min(self._waypoint_lookahead, self._max_linear_acceleration,
               self._max_angular_acceleration, self._sensor_timeout) <= 0:
            raise ValueError('Lookahead, acceleration limits and timeout must be positive')
        if not 0 <= self._lidar_stop_distance < self._lidar_slow_distance:
            raise ValueError('LiDAR distances must satisfy 0 <= stop < slow')
        self._control_mode = 'WAITING'
        self._stopping = False

        camera_topic = str(self._parameter('camera_topic'))
        scan_topic = str(self._parameter('scan_topic'))
        command_topic = str(self._parameter('command_topic'))
        debug_image_topic = str(self._parameter('debug_image_topic'))
        debug_mask_topic = str(self._parameter('debug_mask_topic'))

        self._command_publisher = self.create_publisher(
            Twist,
            command_topic,
            10,
        )
        self._debug_image_publisher = self.create_publisher(
            Image,
            debug_image_topic,
            10,
        )
        self._debug_mask_publisher = self.create_publisher(
            Image,
            debug_mask_topic,
            10,
        )
        self._image_subscription = self.create_subscription(
            Image,
            camera_topic,
            self._image_callback,
            qos_profile_sensor_data,
        )
        self._latest_scan: Optional[LaserScan] = None
        self._scan_subscription = self.create_subscription(
            LaserScan,
            scan_topic,
            self._scan_callback,
            qos_profile_sensor_data,
        )
        self._latest_odometry: Optional[Odometry] = None
        self._last_scan_time = None
        self._last_odometry_time = None
        self._recovery_direction: Optional[float] = None
        self._odometry_subscription = self.create_subscription(
            Odometry,
            '/turtlebot3/odometry',
            self._odometry_callback,
            qos_profile_sensor_data,
        )
        self._imu_subscription = self.create_subscription(
            Imu, str(self._parameter('imu_topic')), self._imu_callback,
            qos_profile_sensor_data)

        self._previous_lateral_error = 0.0
        self._previous_angular_command = 0.0
        self._previous_linear_command = 0.0
        self._previous_frame_time: Optional[rclpy.time.Time] = None
        self._last_image_time: Optional[rclpy.time.Time] = None
        self._watchdog_timer = self.create_timer(0.1, self._watchdog_callback)
        self._watchdog_stopped = False

        self.get_logger().info(
            f'Mode: {"route-assisted" if self._use_lab_waypoints else "camera"}; '
            f'camera: {camera_topic}; publishing to '
            f'{command_topic}'
        )

    def _declare_parameters(self) -> None:
        parameters = {
            'camera_topic': '/turtlebot3/camera/image_raw',
            'scan_topic': '/turtlebot3/scan',
            'command_topic': '/turtlebot3/cmd_vel',
            'debug_image_topic': '/turtlebot3/vision/debug_image',
            'debug_mask_topic': '/turtlebot3/vision/corridor_mask',
            'publish_debug': True,
            'use_lab_waypoints': False,
            'waypoint_recovery': False,
            'waypoint_origin_x': 1.52,
            'waypoint_origin_y': 0.65,
            'waypoint_origin_yaw': 0.0,
            'reverse_route': False,
            'waypoint_lookahead': 0.22,
            'use_imu_heading': False,
            'imu_topic': '/turtlebot3/imu',
            'max_linear_acceleration': 0.20,
            'max_angular_acceleration': 1.50,
            'sensor_timeout': 0.60,
            'detection_mode': 'bright_corridor',
            'roi_top_fraction': 0.35,
            'max_saturation': 85,
            'min_value': 105,
            'morphology_kernel': 5,
            'min_component_area': 350,
            'near_row_fraction': 0.86,
            'far_row_fraction': 0.38,
            'sample_row_count': 8,
            'row_band_height': 5,
            'min_corridor_width': 8,
            'min_wall_width': 3,
            'kp_lateral': 1.15,
            'kd_lateral': 0.08,
            'kp_heading': 1.35,
            'kp_heading_right': 1.35,
            'heading_deadband': 0.0,
            'steering_smoothing': 0.35,
            'max_angular_speed': 1.50,
            'max_linear_speed': 0.16,
            'min_linear_speed': 0.05,
            'turn_slowdown': 0.72,
            'minimum_confidence': 0.50,
            'image_timeout': 0.50,
            'use_lidar_safety': False,
            'lidar_stop_distance': 0.20,
            'lidar_slow_distance': 0.48,
            'lidar_corner_gain': 1.20,
            'lidar_center_gain': 0.0,
        }
        for name, default in parameters.items():
            self.declare_parameter(name, default)

    def _parameter(self, name: str):
        return self.get_parameter(name).value

    def _scan_callback(self, message: LaserScan) -> None:
        self._latest_scan = message
        self._last_scan_time = self.get_clock().now()

    def _odometry_callback(self, message: Odometry) -> None:
        self._latest_odometry = message
        self._last_odometry_time = self.get_clock().now()
        pose = message.pose.pose
        current = np.array([pose.position.x, pose.position.y])
        if self._use_imu_heading and self._sensor_fresh(self._last_imu_time):
            if self._previous_odometry_pose is not None:
                delta = current - self._previous_odometry_pose
                wheel_yaw = self._yaw(pose.orientation)
                signed_distance = float(np.dot(
                    delta, (np.cos(wheel_yaw), np.sin(wheel_yaw))))
                previous_yaw = (self._imu_yaw if self._previous_imu_yaw is None
                                else self._previous_imu_yaw)
                change = np.arctan2(np.sin(self._imu_yaw - previous_yaw),
                                    np.cos(self._imu_yaw - previous_yaw))
                midpoint_yaw = previous_yaw + change / 2.0
                self._fused_position += signed_distance * np.array(
                    [np.cos(midpoint_yaw), np.sin(midpoint_yaw)])
            self._previous_imu_yaw = self._imu_yaw
        self._previous_odometry_pose = current

    @staticmethod
    def _yaw(orientation) -> float:
        q = orientation
        return float(np.arctan2(2 * (q.w * q.z + q.x * q.y),
                               1 - 2 * (q.y * q.y + q.z * q.z)))

    def _imu_callback(self, message: Imu) -> None:
        if message.orientation_covariance[0] < 0:
            return
        yaw = self._yaw(message.orientation)
        if self._imu_origin is None:
            self._imu_origin = yaw
        self._imu_yaw = float(np.arctan2(np.sin(yaw - self._imu_origin),
                                       np.cos(yaw - self._imu_origin)))
        self._last_imu_time = self.get_clock().now()

    def _image_callback(self, message: Image) -> None:
        if self._stopping:
            return
        now = self.get_clock().now()
        self._last_image_time = now
        self._watchdog_stopped = False

        try:
            frame = self._bridge.imgmsg_to_cv2(
                message,
                desired_encoding='bgr8',
            )
        except Exception as error:  # CvBridge exposes several exception types.
            self.get_logger().error(f'Could not convert camera image: {error}')
            self._publish_stop()
            return

        detection = self._detector.detect(frame)
        delta_time = self._frame_delta_seconds(now)
        if self._use_lidar_safety and not self._sensor_fresh(self._last_scan_time):
            self._publish_stop()
            return
        if self._use_imu_heading and not self._sensor_fresh(self._last_imu_time):
            self._publish_stop()
            return
        if self._use_lab_waypoints:
            if not self._sensor_fresh(self._last_odometry_time):
                self._publish_stop()
                return
            self._control_mode = 'ROUTE'
            command = self._waypoint_command(delta_time)
            self._command_publisher.publish(command)
            if self._publish_debug:
                self._publish_debug_images(frame, detection, message, command)
            return
        if (
            detection is None
            or detection.confidence < self._minimum_confidence
        ):
            if (self._waypoint_recovery
                    and self._sensor_fresh(self._last_odometry_time)):
                self._control_mode = 'ROUTE RECOVERY'
                command = self._waypoint_command(delta_time)
                self._command_publisher.publish(command)
                if self._publish_debug:
                    self._publish_debug_images(frame, detection, message, command)
                return
            recovery_command = self._lidar_recovery_command()
            if recovery_command is None:
                self._publish_stop()
            else:
                self._control_mode = 'LIDAR RECOVERY'
                recovery_command = self._limit_command(recovery_command, delta_time)
                recovery_command.linear.x = 0.0
                self._previous_linear_command = 0.0
                self._command_publisher.publish(recovery_command)
            if self._publish_debug:
                self._publish_debug_images(
                    frame,
                    detection,
                    message,
                    recovery_command,
                )
            return

        # A valid corridor reacquisition ends any held LiDAR recovery turn.
        self._control_mode = 'CAMERA'
        self._recovery_direction = None
        command, lateral_error, heading_error = self._command_from_detection(
            detection,
            frame.shape[1],
            delta_time,
        )
        self._command_publisher.publish(command)

        if self._publish_debug:
            self._publish_debug_images(
                frame,
                detection,
                message,
                command,
                lateral_error,
                heading_error,
            )

    def _frame_delta_seconds(self, now: rclpy.time.Time) -> float:
        if self._previous_frame_time is None:
            delta_time = 1.0 / 30.0
        else:
            delta_time = (now - self._previous_frame_time).nanoseconds / 1e9
            delta_time = float(np.clip(delta_time, 1e-3, 0.25))
        self._previous_frame_time = now
        return delta_time

    def _sensor_fresh(self, received) -> bool:
        if received is None:
            return False
        age = (self.get_clock().now() - received).nanoseconds / 1e9
        return 0.0 <= age <= self._sensor_timeout

    def _limit_command(self, command: Twist, delta_time: float) -> Twist:
        command.angular.z = slew(
            self._previous_angular_command, command.angular.z,
            self._max_angular_acceleration, delta_time)
        command.linear.x = slew(
            self._previous_linear_command, command.linear.x,
            self._max_linear_acceleration, delta_time)
        if self._use_lidar_safety:
            # Safety reduction is a hard cap, including an immediate stop;
            # neither the minimum cruise speed nor smoothing may override it.
            command.linear.x = min(command.linear.x,
                                   self._max_linear_speed * self._lidar_speed_scale())
        self._previous_linear_command = command.linear.x
        self._previous_angular_command = command.angular.z
        return command

    def _command_from_detection(
        self,
        detection: CorridorDetection,
        image_width: int,
        delta_time: float,
    ) -> tuple[Twist, float, float]:
        half_width = max(1.0, image_width / 2.0)
        image_center = image_width / 2.0
        lateral_error = (
            detection.near_center[0] - image_center
        ) / half_width
        heading_error = (
            detection.far_center[0] - detection.near_center[0]
        ) / half_width
        heading_control = float(np.sign(heading_error)) * max(
            0.0,
            abs(heading_error) - self._heading_deadband,
        )
        lateral_derivative = (
            lateral_error - self._previous_lateral_error
        ) / delta_time
        lateral_gain = self._kp_lateral * (
            1.0 - min(0.75, abs(heading_control) * 3.0)
        )
        heading_gain = (
            self._kp_heading_right
            if heading_control > 0.0
            else self._kp_heading
        )

        raw_angular = -(
            lateral_gain * lateral_error
            + heading_gain * heading_control
            + self._kd_lateral * lateral_derivative
        )
        raw_angular = float(np.clip(
            raw_angular,
            -self._max_angular_speed,
            self._max_angular_speed,
        ))
        alpha = 1.0 - (1.0 - float(np.clip(
            self._steering_smoothing, 0.0, 1.0))) ** (30.0 * delta_time)
        angular = (
            alpha * raw_angular
            + (1.0 - alpha) * self._previous_angular_command
        )

        if self._use_lidar_safety:
            angular = self._lidar_assisted_angular(angular)

        turn_ratio = abs(angular) / max(self._max_angular_speed, 1e-6)
        linear = self._max_linear_speed * (
            1.0 - self._turn_slowdown * turn_ratio
        )
        curve_scale = max(0.35, 1.0 - 3.0 * abs(heading_control))
        linear *= curve_scale
        linear *= 0.65 + 0.35 * detection.confidence
        linear = float(np.clip(
            linear,
            self._min_linear_speed,
            self._max_linear_speed,
        ))
        if self._use_lidar_safety:
            safety_scale = self._lidar_speed_scale()
            if safety_scale <= 0.0:
                linear = 0.0
            else:
                linear *= safety_scale

        command = Twist()
        command.linear.x = linear
        command.angular.z = angular
        self._previous_lateral_error = lateral_error
        command = self._limit_command(command, delta_time)
        return command, lateral_error, heading_error

    def _waypoint_command(self, delta_time: float) -> Twist:
        """Follow the surveyed lab centerline using spawn-relative odometry."""
        world_points = (
            (1.52, .65), (1.75, .65), (1.95, .65), (2.18, .65),
            (2.40, .65), (2.62, .65), (2.82, .65), (3.03, .68),
            (3.22, .70), (3.40, .78), (3.55, .90), (3.68, 1.08),
            (3.70, 1.28), (3.68, 1.50), (3.72, 1.70), (3.72, 1.92),
            (3.72, 2.15), (3.72, 2.35), (3.68, 2.55), (3.65, 2.75),
            (3.60, 2.95), (3.60, 3.15), (3.53, 3.18), (3.32, 3.18),
            (3.12, 3.20), (2.92, 3.25), (2.72, 3.28), (2.50, 3.28),
            (2.30, 3.30), (2.10, 3.28), (1.88, 3.30), (1.72, 3.45),
            (1.72, 3.65), (1.85, 3.82), (2.05, 3.85), (2.20, 3.90),
            (2.35, 4.05), (2.55, 4.10), (2.78, 4.10), (2.95, 4.18),
            (3.12, 4.30), (3.30, 4.38), (3.45, 4.50), (3.53, 4.68),
            (3.55, 4.88), (3.45, 5.05), (3.30, 5.20), (3.15, 5.32),
            (2.95, 5.35), (2.75, 5.35), (2.53, 5.35), (2.33, 5.43),
            (2.18, 5.52), (2.02, 5.38), (1.83, 5.32), (1.62, 5.28),
            (1.42, 5.22), (1.25, 5.15), (1.10, 5.00), (.90, 5.00),
            (.80, 4.95), (.88, 4.78), (.88, 4.58), (.88, 4.38),
            (.85, 4.18), (.82, 3.98), (.82, 3.75), (.82, 3.53),
            (.82, 3.32), (.82, 3.12), (.80, 2.95), (.85, 2.75),
            (.85, 2.53), (.85, 2.33), (.85, 2.12), (.85, 1.90),
            (.85, 1.68), (.85, 1.45), (.80, 1.25), (.85, 1.10),
            (1.00, .95), (1.15, .80), (1.35, .73), (1.52, .65),
        )
        if self._path is None:
            points = np.asarray(world_points, dtype=float)
            if self._reverse_route:
                points = points[::-1]
            points = points - [self._waypoint_origin_x, self._waypoint_origin_y]
            # Both wheel odometry and IMU heading start in the spawn frame.
            # Rotate the world route into that frame as well as translating it.
            c, s = np.cos(self._waypoint_origin_yaw), np.sin(self._waypoint_origin_yaw)
            self._path = ClosedPath(points @ np.array([[c, -s], [s, c]]))
        pose = self._latest_odometry.pose.pose
        x, y = pose.position.x, pose.position.y
        yaw = self._yaw(pose.orientation)
        if self._use_imu_heading:
            x, y = self._fused_position
            yaw = self._imu_yaw
        target = self._path.target((x, y), self._waypoint_lookahead)
        dx, dy = target[0] - x, target[1] - y
        lateral = -np.sin(yaw) * dx + np.cos(yaw) * dy
        curvature = 2.0 * lateral / max(dx * dx + dy * dy, 1e-6)
        linear = self._max_linear_speed / (
            1.0 + self._turn_slowdown * 0.25 * abs(curvature))
        linear = min(linear, self._max_angular_speed / max(abs(curvature), 1e-6))
        angular = linear * curvature
        command = Twist()
        command.linear.x = float(linear)
        command.angular.z = float(angular)
        return self._limit_command(command, delta_time)

    def _scan_sector_min(self, start_angle: float, end_angle: float) -> float:
        """Return the nearest finite LiDAR return in an angular sector."""
        scan = self._latest_scan
        if scan is None or not scan.ranges or scan.angle_increment == 0.0:
            return float('inf')

        angles = (
            scan.angle_min
            + np.arange(len(scan.ranges)) * scan.angle_increment
        )
        # Gazebo's TurtleBot scan is encoded from 0 to 2*pi. Normalize it so
        # sectors crossing the robot's right side can use conventional
        # negative angles.
        angles = (angles + np.pi) % (2.0 * np.pi) - np.pi
        ranges = np.asarray(scan.ranges, dtype=float)
        valid = (
            np.isfinite(ranges)
            & (ranges >= max(0.0, scan.range_min))
            & (ranges <= scan.range_max)
            & (angles >= start_angle)
            & (angles <= end_angle)
        )
        if not np.any(valid):
            return float('inf')
        return float(np.min(ranges[valid]))

    def _front_clearance(self) -> float:
        # Keep this narrow on the compact lab track; a wide cone includes the
        # close side barriers and falsely reports them as frontal obstacles.
        return self._scan_sector_min(np.deg2rad(-10), np.deg2rad(10))

    def _lidar_speed_scale(self) -> float:
        """Slow continuously, and stop completely, near a frontal wall."""
        front = self._front_clearance()
        if not np.isfinite(front):
            return 1.0
        if front <= self._lidar_stop_distance:
            return 0.0
        distance_span = max(
            1e-6,
            self._lidar_slow_distance - self._lidar_stop_distance,
        )
        return float(np.clip(
            (front - self._lidar_stop_distance) / distance_span,
            0.0,
            1.0,
        ))

    def _lidar_assisted_angular(self, vision_angular: float) -> float:
        """Bias the vision command toward the open side of a tight corner."""
        front = self._front_clearance()
        left = self._scan_sector_min(np.deg2rad(55), np.deg2rad(105))
        right = self._scan_sector_min(np.deg2rad(-105), np.deg2rad(-55))
        if not np.isfinite(left) and not np.isfinite(right):
            return vision_angular

        assisted = vision_angular
        if np.isfinite(left) and np.isfinite(right):
            center_correction = self._lidar_center_gain * (left - right)
            assisted += float(np.clip(center_correction, -0.35, 0.35))

        if not np.isfinite(front) or front >= self._lidar_slow_distance:
            return float(np.clip(
                assisted,
                -self._max_angular_speed,
                self._max_angular_speed,
            ))

        # Positive angular.z turns left. Prefer the side with more clearance.
        left_score = left if np.isfinite(left) else self._lidar_slow_distance
        right_score = right if np.isfinite(right) else self._lidar_slow_distance
        direction = float(np.sign(left_score - right_score))
        proximity = 1.0 - float(np.clip(
            (front - self._lidar_stop_distance)
            / max(1e-6, self._lidar_slow_distance - self._lidar_stop_distance),
            0.0,
            1.0,
        ))
        assisted += (
            direction * self._lidar_corner_gain * proximity
        )
        return float(np.clip(
            assisted,
            -self._max_angular_speed,
            self._max_angular_speed,
        ))

    def _lidar_recovery_command(self) -> Optional[Twist]:
        """Turn toward free space until the camera corridor reappears."""
        if not self._use_lidar_safety or self._latest_scan is None:
            return None

        front = self._front_clearance()
        left = self._scan_sector_min(np.deg2rad(20), np.deg2rad(95))
        right = self._scan_sector_min(np.deg2rad(-95), np.deg2rad(-20))
        if not np.isfinite(left) and not np.isfinite(right):
            if not np.isfinite(front):
                return None
            direction = 1.0 if self._previous_angular_command >= 0.0 else -1.0
        else:
            left_score = (
                left if np.isfinite(left) else self._lidar_slow_distance
            )
            right_score = (
                right if np.isfinite(right) else self._lidar_slow_distance
            )
            direction = 1.0 if left_score >= right_score else -1.0

        if self._recovery_direction is None:
            self._recovery_direction = direction
        direction = self._recovery_direction

        command = Twist()
        command.angular.z = direction * min(0.65, self._max_angular_speed)
        return command

    def _publish_debug_images(
        self,
        frame: np.ndarray,
        detection: Optional[CorridorDetection],
        source_message: Image,
        command: Optional[Twist] = None,
        lateral_error: float = 0.0,
        heading_error: float = 0.0,
    ) -> None:
        annotated = frame.copy()
        height, width = annotated.shape[:2]
        cv2.line(
            annotated,
            (width // 2, 0),
            (width // 2, height - 1),
            (0, 0, 255),
            1,
        )

        mask = np.zeros((height, width), dtype=np.uint8)
        status = 'LOST - STOPPED'
        status_color = (0, 0, 255)
        if detection is not None:
            mask = detection.mask
            cv2.line(
                annotated,
                (0, detection.roi_top),
                (width - 1, detection.roi_top),
                (255, 255, 0),
                1,
            )
            overlay = np.zeros_like(annotated)
            overlay[:, :, 1] = mask
            annotated = cv2.addWeighted(annotated, 1.0, overlay, 0.22, 0.0)
            for left, right, row in detection.spans:
                cv2.line(annotated, (left, row), (right, row), (255, 0, 0), 1)
            for center in detection.centers:
                cv2.circle(annotated, center, 3, (0, 255, 0), -1)
            cv2.line(
                annotated,
                detection.near_center,
                detection.far_center,
                (0, 255, 255),
                2,
            )
        if command is not None:
            confidence = detection.confidence if detection is not None else 0.0
            status = (
                f'{self._control_mode} v={command.linear.x:.2f} '
                f'w={command.angular.z:.2f} conf={confidence:.2f}'
            )
            status_color = (0, 255, 0)

        cv2.putText(
            annotated,
            status,
            (6, 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            status_color,
            1,
            cv2.LINE_AA,
        )

        debug_message = self._bridge.cv2_to_imgmsg(annotated, encoding='bgr8')
        debug_message.header = source_message.header
        mask_message = self._bridge.cv2_to_imgmsg(mask, encoding='mono8')
        mask_message.header = source_message.header
        self._debug_image_publisher.publish(debug_message)
        self._debug_mask_publisher.publish(mask_message)

    def _watchdog_callback(self) -> None:
        if self._last_image_time is None:
            return
        image_age = (
            self.get_clock().now() - self._last_image_time
        ).nanoseconds / 1e9
        if image_age > self._image_timeout and not self._watchdog_stopped:
            self.get_logger().warning(
                f'No camera frame for {image_age:.2f}s; stopping robot.'
            )
            self._publish_stop()
            self._watchdog_stopped = True

    def _publish_stop(self) -> None:
        self._control_mode = 'STOPPED'
        self._command_publisher.publish(Twist())
        self._previous_angular_command = 0.0
        self._previous_linear_command = 0.0
        self._previous_lateral_error = 0.0
        self._previous_frame_time = None
        self._recovery_direction = None

    def stop(self) -> None:
        """Publish multiple zero commands before shutting down."""
        self._stopping = True
        if not rclpy.ok():
            return
        for _ in range(3):
            self._publish_stop()


def main(args=None) -> None:
    """Run the vision corridor controller node."""
    # Keep the ROS context alive through KeyboardInterrupt so stop() can
    # deliver zero commands before destroying the node.
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    node = VisionCorridorController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
