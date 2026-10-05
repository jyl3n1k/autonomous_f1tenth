#!/usr/bin/env python3
"""Validate the route and prepare /tmp/lab_centerline_camera.sdf.

On Fortress run with LIBGL_ALWAYS_SOFTWARE=1 ign gazebo -s -r, then
subscribe to /world/empty/model/overview_camera/link/camera_link/sensor/overview/image
to activate capture. PNG frames are saved in /tmp/lab_centerline_capture.
"""
import importlib.util
import shutil
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    'lab', ROOT / 'src/environments/scripts/generate_lab_track.py')
lab = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lab)
tree = ET.parse(ROOT / 'src/environments/worlds/lab_track.sdf')
world = tree.getroot().find('world')
# Fortress uses the older plugin names; only adapt the temporary capture world.
if not Path('/usr/lib/x86_64-linux-gnu/gz-sim-8').exists() and shutil.which('ign'):
    for plugin in world.findall('plugin'):
        plugin.set('filename', plugin.get('filename').replace('gz-sim-', 'ignition-gazebo-'))
        plugin.set('name', plugin.get('name').replace('gz::sim::', 'ignition::gazebo::'))
        engine = plugin.find('render_engine')
        if engine is not None:
            engine.text = 'ogre'
points = np.array(lab._sample_path(lab.CENTERLINE_POINTS, True))
minimum = float('inf')
for box in world.findall("model[@name='lab_track_walls']/link/collision"):
    x, y, _, _, _, yaw = map(float, box.findtext('pose').split())
    sx, sy, _ = map(float, box.findtext('geometry/box/size').split())
    delta = points - [x, y]
    local = delta @ np.array([[np.cos(yaw), -np.sin(yaw)],
                             [np.sin(yaw), np.cos(yaw)]])
    distance = np.linalg.norm(np.maximum(np.abs(local) - [sx/2, sy/2], 0), axis=1)
    minimum = min(minimum, float(distance.min()) - lab.CENTERLINE_WIDTH/2)
assert minimum > .10, f'Line approaches a wall: {minimum:.3f} m'
print(f'Minimum paint-edge clearance to walls: {minimum:.3f} m')
model = ET.SubElement(world, 'model', name='overview_camera')
ET.SubElement(model, 'static').text = 'true'
ET.SubElement(model, 'pose').text = '2.2 3.2 9 0 1.57079632679 1.57079632679'
link = ET.SubElement(model, 'link', name='camera_link')
sensor = ET.SubElement(link, 'sensor', name='overview', type='camera')
ET.SubElement(sensor, 'always_on').text = 'true'
ET.SubElement(sensor, 'update_rate').text = '1'
camera = ET.SubElement(sensor, 'camera')
ET.SubElement(camera, 'horizontal_fov').text = '0.80'
image = ET.SubElement(camera, 'image')
ET.SubElement(image, 'width').text = '1200'
ET.SubElement(image, 'height').text = '1600'
ET.SubElement(image, 'format').text = 'R8G8B8'
clip = ET.SubElement(camera, 'clip')
ET.SubElement(clip, 'near').text = '0.1'
ET.SubElement(clip, 'far').text = '30'
save = ET.SubElement(camera, 'save', enabled='true')
ET.SubElement(save, 'path').text = '/tmp/lab_centerline_capture'
tree.write('/tmp/lab_centerline_camera.sdf', encoding='utf-8', xml_declaration=True)
