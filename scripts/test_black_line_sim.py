#!/usr/bin/env python3
"""Run an isolated Fortress black-line lap test with ground-truth measurement.

Source ROS and the workspace before running. Camera steering has no access to
ground truth. Only the evaluator uses it to count actual completed laps.
"""
import os
import json
from pathlib import Path
import signal
import subprocess
import time
import xml.etree.ElementTree as ET
import numpy as np
from check_lab_clearance import check

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/black_line'
OUT.mkdir(parents=True, exist_ok=True)
env = os.environ.copy()
env.update(ROS_DOMAIN_ID='87', IGN_PARTITION='black_line_test',
           GZ_PARTITION='black_line_test', LIBGL_ALWAYS_SOFTWARE='1',
           IGN_GAZEBO_RESOURCE_PATH=str(ROOT / 'src/environments/models'))
tree = ET.parse(ROOT / 'src/environments/worlds/lab_track.sdf')
world = tree.getroot().find('world')
world.find('physics/max_step_size').text = '0.004'
model = ET.parse(ROOT / 'src/environments/models/turtlebot3_burger_cam/model.sdf').getroot().find('model')
model.set('name', 'turtlebot3')
model.find('pose').text = '2.20 0.65 0.01 0 0 0'
# Exercise reduced camera cadence while keeping software rendering practical.
model.find("link[@name='camera_link']/sensor/update_rate").text = '10'
world.append(model)
for plugin in world.findall('.//plugin'):
    plugin.set('filename', plugin.get('filename').replace('gz-sim-', 'ignition-gazebo-'))
    plugin.set('name', plugin.get('name').replace('gz::sim::', 'ignition::gazebo::'))
    engine = plugin.find('render_engine')
    if engine is not None:
        engine.text = 'ogre2'
tree.write(OUT / 'test_world.sdf')
processes, logs = [], []
def launch(name, command):
    log = (OUT / (name + '.log')).open('w')
    logs.append(log)
    process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log,
                               stderr=subprocess.STDOUT, start_new_session=True)
    processes.append(process)
    return process
try:
    launch('simulator', ['ign', 'gazebo', '-s', '-r', '--headless-rendering', str(OUT / 'test_world.sdf')])
    launch('bridge', ['ros2', 'run', 'ros_gz_bridge', 'parameter_bridge',
        '/clock@rosgraph_msgs/msg/Clock[ignition.msgs.Clock',
        '/turtlebot3/odometry@nav_msgs/msg/Odometry[ignition.msgs.Odometry',
        '/turtlebot3/scan@sensor_msgs/msg/LaserScan[ignition.msgs.LaserScan',
        '/turtlebot3/camera/image_raw@sensor_msgs/msg/Image[ignition.msgs.Image',
        '/turtlebot3/cmd_vel@geometry_msgs/msg/Twist]ignition.msgs.Twist',
        '/world/empty/dynamic_pose/info@tf2_msgs/msg/TFMessage[ignition.msgs.Pose_V'])
    evaluator = launch('evaluation', ['python3', '-u', 'scripts/evaluate_lab_controller.py',
        '--duration', '350', '--wall-timeout', '1800', '--laps', '1',
        '--output', str(OUT / 'lap.csv')])
    launch('recorder', ['python3', 'scripts/record_line_debug.py'])
    result = evaluator.wait()
    data = np.genfromtxt(OUT / 'lap.csv', delimiter=',', names=True)
    if data.size:
        xy = np.column_stack((data['truth_x'] + 2.20, data['truth_y'] + .65))
        clearance = np.full(len(xy), np.inf)
        for box in world.findall("model[@name='lab_track_walls']/link/collision"):
            x, y, _, _, _, yaw = map(float, box.findtext('pose').split())
            sx, sy, _ = map(float, box.findtext('geometry/box/size').split())
            local = (xy - [x, y]) @ np.array([
                [np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
            clearance = np.minimum(clearance, np.linalg.norm(
                np.maximum(np.abs(local) - [sx/2, sy/2], 0), axis=1))
        summary_path = OUT / 'lap.json'
        summary = json.loads(summary_path.read_text())
        # A 0.13 m circle encloses the Burger collision footprint. Positive
        # clearance is a conservative geometric no-contact check.
        summary['minimum_robot_envelope_clearance_m'] = float(clearance.min() - .13)
        summary['test_engine'] = 'Gazebo Fortress / Ogre2 software, 0.004 s step, 10 Hz camera'
        summary_path.write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2), flush=True)
        if not check(OUT / 'lap.csv'):
            raise SystemExit('Padded robot footprint approached/intersected a wall.')
    raise SystemExit(result)
finally:
    for process in reversed(processes):
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGINT)
    for process in processes:
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
    for log in logs:
        log.close()
