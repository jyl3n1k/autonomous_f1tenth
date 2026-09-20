#!/usr/bin/env python3
"""Measure the controller in a separately launched lab-track simulation.

Run with the same ROS_DOMAIN_ID as the test simulator. This starts a controller
and therefore requires that no other controller publishes on its cmd_vel topic.
Writes odometry, commands and LiDAR clearance; stops the robot on exit.
"""

import argparse
import csv
import json
import math
from pathlib import Path
import time

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from rclpy.executors import ExternalShutdownException
from rclpy.qos import qos_profile_sensor_data
from rclpy.signals import SignalHandlerOptions
from tf2_msgs.msg import TFMessage

from turtlebot3_vision_controller.controller import VisionCorridorController


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--duration', type=float, default=240.0)
    parser.add_argument('--wall-timeout', type=float, default=600.0)
    parser.add_argument('--laps', type=int, default=0,
                        help='Stop after this many actual start-line crossings')
    parser.add_argument('--reverse-route', action='store_true')
    parser.add_argument('--origin-yaw', type=float,
                        help='Must match the simulator spawn_yaw (radians)')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', default=str(
        Path(__file__).resolve().parents[1]
        / 'src/turtlebot3_vision_controller/config/lab_track.yaml'))
    args = parser.parse_args()
    ros_args = ['--ros-args', '--params-file', args.config]
    if args.reverse_route:
        ros_args += ['-p', 'reverse_route:=true']
    if args.origin_yaw is not None:
        ros_args += ['-p', f'waypoint_origin_yaw:={args.origin_yaw}']
    rclpy.init(args=ros_args,
               signal_handler_options=SignalHandlerOptions.NO)
    node = VisionCorridorController()
    rows, commands = [], []
    latest = Twist()
    truth = None
    recording = True

    def on_truth(message):
        nonlocal truth
        # This test world has one dynamic model, followed by its links.
        # Garden's Pose_V bridge leaves TF frame names empty.
        if message.transforms:
            truth = message.transforms[0]

    def on_command(message):
        nonlocal latest
        latest = message
        if recording:
            commands.append((node.get_clock().now().nanoseconds / 1e9,
                             message.linear.x, message.angular.z))

    subscription = node.create_subscription(
        Twist, '/turtlebot3/cmd_vel', on_command, qos_profile_sensor_data)
    truth_subscription = node.create_subscription(
        TFMessage, '/world/empty/dynamic_pose/info', on_truth, qos_profile_sensor_data)
    wall_start = time.monotonic()
    start = last_sample = last_print = None
    distance = 0.0
    laps = 0
    armed = False
    lap_distance = 0.0
    previous_forward = 0.0
    c, s = math.cos(node._waypoint_origin_yaw), math.sin(node._waypoint_origin_yaw)
    rotation = np.array([[c, -s], [s, c]])
    try:
        while time.monotonic() - wall_start < args.wall_timeout:
            rclpy.spin_once(node, timeout_sec=0.1)
            odom = node._latest_odometry
            if odom is None or node._last_image_time is None or truth is None:
                continue
            stamp = odom.header.stamp
            now = stamp.sec + stamp.nanosec / 1e9
            if start is None:
                start = now
            if last_sample is not None and now - last_sample < 0.1:
                continue
            last_sample = now
            pose = odom.pose.pose
            # Store positions in world-aligned axes relative to the spawn
            # translation, including when odometry starts facing backwards.
            x, y = rotation @ [pose.position.x, pose.position.y]
            yaw = (2 * math.atan2(pose.orientation.z, pose.orientation.w)
                   + node._waypoint_origin_yaw)
            yaw = math.atan2(math.sin(yaw), math.cos(yaw))
            tx = truth.transform.translation.x - node._waypoint_origin_x
            ty = truth.transform.translation.y - node._waypoint_origin_y
            q = truth.transform.rotation
            tyaw = math.atan2(2 * (q.w * q.z + q.x * q.y),
                             1 - 2 * (q.y * q.y + q.z * q.z))
            if rows:
                distance += math.hypot(tx - rows[-1][12], ty - rows[-1][13])
            if math.hypot(tx, ty) > 1.0:
                armed = True
            forward, lateral = c * tx + s * ty, -s * tx + c * ty
            heading_error = tyaw - node._waypoint_origin_yaw
            heading_error = math.atan2(math.sin(heading_error), math.cos(heading_error))
            if (armed and previous_forward < 0 <= forward
                    and abs(lateral) < 0.20 and abs(heading_error) < 0.5
                    and distance - lap_distance > 10.0):
                laps += 1
                armed = False
                lap_distance = distance
            previous_forward = forward
            clearance = node._scan_sector_min(-math.pi, math.pi)
            estimate = (rotation @ node._fused_position if node._use_imu_heading
                        else (x, y))
            rows.append((now - start, x, y, yaw, latest.linear.x,
                         latest.angular.z, odom.twist.twist.linear.x,
                         odom.twist.twist.angular.z, clearance,
                         node._front_clearance(), distance, laps, tx, ty, tyaw,
                         estimate[0], estimate[1]))
            if last_print is None or now - last_print >= 10:
                last_print = now
                print(f't={now-start:.1f} truth=({tx:.2f},{ty:.2f}) '
                      f'odom_error={math.hypot(x-tx,y-ty):.3f} '
                      f'estimate_error={math.hypot(estimate[0]-tx,estimate[1]-ty):.3f} '
                      f'v={latest.linear.x:.3f} w={latest.angular.z:.3f} '
                      f'clearance={clearance:.3f} distance={distance:.2f} '
                      f'laps={laps}', flush=True)
            if now - start >= args.duration:
                break
            if args.laps and laps >= args.laps:
                break
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        recording = False
        node.stop()
        # Allow the bridge to receive the stop before tearing down DDS.
        try:
            for _ in range(5):
                if rclpy.ok():
                    rclpy.spin_once(node, timeout_sec=0.05)
        except (KeyboardInterrupt, ExternalShutdownException):
            pass
        node.destroy_subscription(subscription)
        node.destroy_subscription(truth_subscription)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(('time', 'x', 'y', 'yaw', 'command_v', 'command_w',
                             'actual_v', 'actual_w', 'scan_min', 'front',
                             'distance', 'laps', 'truth_x', 'truth_y', 'truth_yaw',
                             'estimate_x', 'estimate_y'))
            writer.writerows(rows)
        command_path = args.output.with_suffix('.commands.csv')
        with command_path.open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(('time', 'v', 'w'))
            writer.writerows(commands)
        summary = {'samples': len(rows), 'laps': laps, 'distance_m': distance,
                   'sim_seconds': rows[-1][0] if rows else 0.0}
        if len(commands) > 2 and rows:
            data = np.asarray(commands)
            dt = np.diff(data[:, 0])
            valid = dt > 1e-4
            dw = np.abs(np.diff(data[:, 2]))[valid]
            summary.update(
                angular_step_p95=float(np.percentile(dw, 95)),
                angular_step_max=float(np.max(dw)),
                angular_acceleration_p95=float(np.percentile(dw / dt[valid], 95)),
                angular_acceleration_max=float(np.max(dw / dt[valid])),
                mean_speed=float(np.mean(data[:, 1])),
                minimum_lidar_range=float(min(row[8] for row in rows)),
                max_odometry_error=float(max(math.hypot(row[1]-row[12],
                                                       row[2]-row[13])
                                             for row in rows)),
                max_estimate_error=float(max(math.hypot(row[15]-row[12],
                                                       row[16]-row[13])
                                             for row in rows)))
        args.output.with_suffix('.json').write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2), flush=True)
    if args.laps:
        if laps < args.laps:
            raise SystemExit(f'Only {laps} of {args.laps} requested laps completed.')
    elif not rows or rows[-1][0] < args.duration - 0.2:
        raise SystemExit('Simulation did not reach the requested duration.')


if __name__ == '__main__':
    main()
