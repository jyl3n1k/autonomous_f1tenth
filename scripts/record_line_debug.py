#!/usr/bin/env python3
"""Save the latest camera/debug frame while the isolated lap test runs."""
from pathlib import Path
import cv2
from cv_bridge import CvBridge
import rclpy
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

rclpy.init()
node = rclpy.create_node('line_debug_recorder')
bridge = CvBridge()
out = Path(__file__).resolve().parents[1] / 'artifacts/black_line'
def save(message, name):
    cv2.imwrite(str(out / (name + '.png')),
                bridge.imgmsg_to_cv2(message, desired_encoding='bgr8'))
subscriptions = [node.create_subscription(Image, topic,
    lambda message, name=name: save(message, name), qos_profile_sensor_data)
    for topic, name in [('/turtlebot3/camera/image_raw', 'camera'),
                        ('/turtlebot3/vision/debug_image', 'debug')]]
try:
    rclpy.spin(node)
except KeyboardInterrupt:
    pass
finally:
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()
