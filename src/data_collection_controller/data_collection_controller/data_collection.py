import csv
import math
import os
from datetime import datetime
from threading import Thread

from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)
from sensor_msgs.msg import LaserScan
from sshkeyboard import listen_keyboard
from std_msgs.msg import Float32MultiArray, String


class DataCollection(Node):
    SECTOR_CENTERS_DEG = tuple(range(0, 360, 10))
    SECTOR_HALF_WIDTH_DEG = 3.0
    SECTOR_PERCENTILE = 10

    def __init__(self):
        super().__init__('data_collection_node')

        scan_qos = QoSProfile(
            depth=10,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
        )
        status_qos = QoSProfile(
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        )

        self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            scan_qos,
        )
        self.create_subscription(
            Twist,
            '/cmd_vel',
            self.cmd_vel_callback,
            10,
        )
        self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10,
        )
        self.sector_publisher = self.create_publisher(
            Float32MultiArray,
            '/data_collection/lidar_sectors',
            10,
        )
        self.status_publisher = self.create_publisher(
            String,
            '/data_collection/status',
            status_qos,
        )

        self.expert_linear_velocity = 0.0
        self.expert_angular_velocity = 0.0
        self.measured_linear_velocity = 0.0
        self.measured_angular_velocity = 0.0
        self.measured_linear_acceleration = 0.0
        self.received_expert_command = False
        self.received_odometry = False
        self.received_acceleration = False
        self.scan_timestamp_ns = None
        self.odom_timestamp_ns = None
        self.expert_command_timestamp_ns = None
        self.previous_linear_velocity = None
        self.previous_odom_timestamp_ns = None
        self.sector_ranges = None

        self.csv_file = None
        self.csv_writer = None
        self.csv_filename = None
        self.is_recording = False
        self.publish_status('Not recording')

    def publish_status(self, status):
        message = String()
        message.data = status
        self.status_publisher.publish(message)

    def initialize_csv(self):
        timestamp = datetime.now().strftime('%m%d%Y_%H%M%S_%f')
        output_directory = os.path.join(
            'data_collection_controller',
            'data_collection',
        )
        os.makedirs(output_directory, exist_ok=True)

        self.csv_filename = os.path.join(
            output_directory,
            f'robot_data_{timestamp}.csv',
        )
        self.csv_file = open(self.csv_filename, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(
            [
                'scan_timestamp_ns',
                'odom_timestamp_ns',
                'expert_command_timestamp_ns',
                *(
                    f'range_{angle:03d}_deg'
                    for angle in self.SECTOR_CENTERS_DEG
                ),
                'measured_linear_velocity',
                'measured_angular_velocity',
                'measured_linear_acceleration',
                'expert_linear_velocity',
                'expert_angular_velocity',
            ]
        )
        self.csv_file.flush()
        self.get_logger().info(
            f'Initialized CSV logger: {self.csv_filename}'
        )

    def key_press(self, key):
        if key == 'r' and not self.is_recording:
            self.initialize_csv()
            self.is_recording = True
            status = f'Recording: {os.path.basename(self.csv_filename)}'
            self.publish_status(status)
            self.get_logger().info('Started recording')
        elif key == 'q':
            self.is_recording = False
            self.publish_status('Not recording')
            self.get_logger().info('Stopped recording')
            if rclpy.ok():
                rclpy.shutdown()

    def key_release(self, key):
        pass

    def cmd_vel_callback(self, msg):
        self.expert_linear_velocity = float(msg.linear.x)
        self.expert_angular_velocity = float(msg.angular.z)
        self.expert_command_timestamp_ns = self.get_clock().now().nanoseconds
        self.received_expert_command = True

    def odom_callback(self, msg):
        new_linear_velocity = float(msg.twist.twist.linear.x)
        new_angular_velocity = float(msg.twist.twist.angular.z)
        stamp = msg.header.stamp
        new_timestamp_ns = stamp.sec * 1_000_000_000 + stamp.nanosec
        if new_timestamp_ns == 0:
            new_timestamp_ns = self.get_clock().now().nanoseconds

        if (
            self.previous_linear_velocity is not None
            and self.previous_odom_timestamp_ns is not None
            and new_timestamp_ns > self.previous_odom_timestamp_ns
        ):
            time_difference = (
                new_timestamp_ns - self.previous_odom_timestamp_ns
            ) / 1_000_000_000.0
            self.measured_linear_acceleration = (
                new_linear_velocity - self.previous_linear_velocity
            ) / time_difference
            self.received_acceleration = True

        self.measured_linear_velocity = new_linear_velocity
        self.measured_angular_velocity = new_angular_velocity
        self.odom_timestamp_ns = new_timestamp_ns
        self.previous_linear_velocity = new_linear_velocity
        self.previous_odom_timestamp_ns = new_timestamp_ns
        self.received_odometry = True

    def scan_callback(self, msg):
        stamp = msg.header.stamp
        self.scan_timestamp_ns = stamp.sec * 1_000_000_000 + stamp.nanosec
        if self.scan_timestamp_ns == 0:
            self.scan_timestamp_ns = self.get_clock().now().nanoseconds

        ranges = np.asarray(msg.ranges, dtype=float)
        beam_angles_deg = np.degrees(
            msg.angle_min + np.arange(len(ranges)) * msg.angle_increment
        ) % 360.0

        fallback_range = (
            float(msg.range_max)
            if math.isfinite(msg.range_max)
            else 3.5
        )
        sector_ranges = []

        for center_deg in self.SECTOR_CENTERS_DEG:
            angular_distance = np.abs(
                (beam_angles_deg - center_deg + 180.0) % 360.0 - 180.0
            )
            sector_values = ranges[
                angular_distance <= self.SECTOR_HALF_WIDTH_DEG
            ]
            valid_values = sector_values[
                np.isfinite(sector_values)
                & (sector_values >= msg.range_min)
                & (sector_values <= msg.range_max)
            ]

            if valid_values.size:
                sector_range = float(
                    np.percentile(valid_values, self.SECTOR_PERCENTILE)
                )
            else:
                sector_range = fallback_range

            sector_ranges.append(sector_range)

        self.sector_ranges = sector_ranges
        sector_message = Float32MultiArray()
        sector_message.data = sector_ranges
        self.sector_publisher.publish(sector_message)
        self.log_scan()

    def log_scan(self):
        if (
            not self.is_recording
            or not self.received_expert_command
            or not self.received_odometry
            or not self.received_acceleration
            or self.sector_ranges is None
            or self.csv_writer is None
            or self.csv_file is None
        ):
            return

        self.csv_writer.writerow(
            [
                self.scan_timestamp_ns,
                self.odom_timestamp_ns,
                self.expert_command_timestamp_ns,
                *self.sector_ranges,
                self.measured_linear_velocity,
                self.measured_angular_velocity,
                self.measured_linear_acceleration,
                self.expert_linear_velocity,
                self.expert_angular_velocity,
            ]
        )
        self.csv_file.flush()

    def cleanup(self):
        self.is_recording = False
        if self.csv_file is not None and not self.csv_file.closed:
            self.csv_file.close()
        self.destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = DataCollection()

    keyboard_thread = Thread(
        target=listen_keyboard,
        kwargs={
            'on_press': node.key_press,
            'on_release': node.key_release,
        },
        daemon=True,
    )
    keyboard_thread.start()

    print('Press r to start recording, q to quit.')

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.cleanup()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
