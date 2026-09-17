"""ROS 2 node that keeps TurtleBot3 centered in a visible track corridor."""

from typing import Optional

import cv2
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, LaserScan

from .corridor import CorridorDetection, CorridorDetector


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
        self._max_linear_speed = float(self._parameter('max_linear_speed'))
        self._min_linear_speed = float(self._parameter('min_linear_speed'))
        self._max_angular_speed = float(self._parameter('max_angular_speed'))
        self._turn_slowdown = float(self._parameter('turn_slowdown'))
        self._steering_smoothing = float(self._parameter('steering_smoothing'))
        self._minimum_confidence = float(self._parameter('minimum_confidence'))
        self._image_timeout = float(self._parameter('image_timeout'))
        self._publish_debug = bool(self._parameter('publish_debug'))
        self._use_lidar_safety = bool(self._parameter('use_lidar_safety'))
        self._lidar_stop_distance = float(
            self._parameter('lidar_stop_distance')
        )
        self._lidar_slow_distance = float(
            self._parameter('lidar_slow_distance')
        )
        self._lidar_corner_gain = float(self._parameter('lidar_corner_gain'))

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

        self._previous_lateral_error = 0.0
        self._previous_angular_command = 0.0
        self._previous_frame_time: Optional[rclpy.time.Time] = None
        self._last_image_time: Optional[rclpy.time.Time] = None
        self._watchdog_timer = self.create_timer(0.1, self._watchdog_callback)
        self._watchdog_stopped = False

        self.get_logger().info(
            f'Following corridor from {camera_topic}; publishing to '
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
        }
        for name, default in parameters.items():
            self.declare_parameter(name, default)

    def _parameter(self, name: str):
        return self.get_parameter(name).value

    def _scan_callback(self, message: LaserScan) -> None:
        self._latest_scan = message

    def _image_callback(self, message: Image) -> None:
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
        if (
            detection is None
            or detection.confidence < self._minimum_confidence
        ):
            self._publish_stop()
            if self._publish_debug:
                self._publish_debug_images(frame, detection, message)
            return

        delta_time = self._frame_delta_seconds(now)
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
        lateral_derivative = (
            lateral_error - self._previous_lateral_error
        ) / delta_time

        raw_angular = -(
            self._kp_lateral * lateral_error
            + self._kp_heading * heading_error
            + self._kd_lateral * lateral_derivative
        )
        raw_angular = float(np.clip(
            raw_angular,
            -self._max_angular_speed,
            self._max_angular_speed,
        ))
        alpha = float(np.clip(self._steering_smoothing, 0.0, 1.0))
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
        linear *= 0.65 + 0.35 * detection.confidence
        linear = float(np.clip(
            linear,
            self._min_linear_speed,
            self._max_linear_speed,
        ))
        if self._use_lidar_safety:
            linear *= self._lidar_speed_scale()

        command = Twist()
        command.linear.x = linear
        command.angular.z = angular
        self._previous_lateral_error = lateral_error
        self._previous_angular_command = angular
        return command, lateral_error, heading_error

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
        return self._scan_sector_min(np.deg2rad(-24), np.deg2rad(24))

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
        if not np.isfinite(front) or front >= self._lidar_slow_distance:
            return vision_angular

        left = self._scan_sector_min(np.deg2rad(18), np.deg2rad(82))
        right = self._scan_sector_min(np.deg2rad(-82), np.deg2rad(-18))
        if not np.isfinite(left) and not np.isfinite(right):
            return vision_angular

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
        assisted = vision_angular + (
            direction * self._lidar_corner_gain * proximity
        )
        return float(np.clip(
            assisted,
            -self._max_angular_speed,
            self._max_angular_speed,
        ))

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
                status = (
                    f'v={command.linear.x:.2f} w={command.angular.z:.2f} '
                    f'lat={lateral_error:+.2f} head={heading_error:+.2f} '
                    f'conf={detection.confidence:.2f}'
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
        self._command_publisher.publish(Twist())
        self._previous_angular_command = 0.0

    def stop(self) -> None:
        """Publish multiple zero commands before shutting down."""
        if not rclpy.ok():
            return
        for _ in range(3):
            self._publish_stop()


def main(args=None) -> None:
    """Run the vision corridor controller node."""
    rclpy.init(args=args)
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
