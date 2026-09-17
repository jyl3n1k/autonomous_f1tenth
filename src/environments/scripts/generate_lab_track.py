#!/usr/bin/env python3
"""Generate a TurtleBot-scale Gazebo world based on the lab track.

The dimensions intentionally remain in this source file so that measured lab
dimensions can replace the initial photo/LiDAR-based estimates later.
"""

from __future__ import annotations

import math
from pathlib import Path
from xml.etree import ElementTree as ET


WORLD_NAME = 'empty'  # cartrack.launch.py bridges services under this name.
FLOOR_SIZE = (4.40, 6.40)
LANE_WIDTH = 0.72
WALL_THICKNESS = 0.09
# Both sensors must see the barriers: the TurtleBot LDS is around 0.17 m and
# the camera optical center is around 0.20 m above the floor.
WALL_HEIGHT = 0.25
# Keep curved barriers visually smooth at TurtleBot scale.  Twenty-four
# samples puts adjacent box centers roughly 2--5 cm apart through the tight
# bends while preserving the measured control-point geometry.
SAMPLES_PER_CONTROL_POINT = 24

# Barrier centerlines copied from the topology in the reference layout: one
# outer boundary, two open divider fingers, and one lower inner loop. Defining
# the barriers directly avoids the extra paired loops produced by offsetting a
# closed driving centerline.
BARRIER_PATHS = (
    (
        'outer',
        (
            (0.72, 0.42),
            (0.38, 0.82),
            (0.39, 5.42),
            (0.78, 5.96),
            (2.20, 6.05),
            (3.60, 5.98),
            (4.00, 5.55),
            (4.02, 4.78),
            (3.73, 4.33),
            (3.43, 4.12),
            (3.55, 3.72),
            (3.86, 3.34),
            (4.03, 2.70),
            (4.02, 1.03),
            (3.68, 0.48),
            (2.25, 0.39),
        ),
        True,
    ),
    (
        'upper_finger',
        (
            (1.35, 3.48),
            (1.32, 3.93),
            (1.60, 4.38),
            (2.12, 4.63),
            (2.78, 4.66),
            (3.25, 4.64),
        ),
        False,
    ),
    (
        'middle_finger',
        (
            (2.02, 3.55),
            (2.48, 3.55),
            (3.02, 3.55),
            (3.43, 3.57),
        ),
        False,
    ),
    (
        'inner_loop',
        (
            (1.36, 3.43),
            (1.38, 2.75),
            (1.34, 1.35),
            (1.68, 0.98),
            (2.42, 0.91),
            (3.08, 0.98),
            (3.39, 1.37),
            (3.38, 2.31),
            (3.11, 2.79),
            (2.53, 3.02),
            (1.72, 3.00),
        ),
        True,
    ),
)

RECOMMENDED_SPAWN = (1.52, 0.66, 0.0)


def _catmull_rom_point(p0, p1, p2, p3, t):
    """Return a point on a uniform Catmull-Rom spline."""
    t2 = t * t
    t3 = t2 * t
    values = []
    for index in range(2):
        value = 0.5 * (
            (2.0 * p1[index])
            + (-p0[index] + p2[index]) * t
            + (
                2.0 * p0[index]
                - 5.0 * p1[index]
                + 4.0 * p2[index]
                - p3[index]
            ) * t2
            + (
                -p0[index]
                + 3.0 * p1[index]
                - 3.0 * p2[index]
                + p3[index]
            ) * t3
        )
        values.append(value)
    return tuple(values)


def _sample_path(control_points, closed):
    points = []
    count = len(control_points)
    segment_count = count if closed else count - 1
    for index in range(segment_count):
        if closed:
            p0 = control_points[(index - 1) % count]
            p1 = control_points[index]
            p2 = control_points[(index + 1) % count]
            p3 = control_points[(index + 2) % count]
        else:
            p0 = control_points[max(0, index - 1)]
            p1 = control_points[index]
            p2 = control_points[index + 1]
            p3 = control_points[min(count - 1, index + 2)]
        for sample in range(SAMPLES_PER_CONTROL_POINT):
            points.append(_catmull_rom_point(
                p0,
                p1,
                p2,
                p3,
                sample / SAMPLES_PER_CONTROL_POINT,
            ))
    if not closed:
        points.append(control_points[-1])
    return points


def _subelement(parent, tag, text=None, **attributes):
    element = ET.SubElement(parent, tag, attributes)
    if text is not None:
        element.text = str(text)
    return element


def _add_material(visual, ambient, diffuse, emissive=None):
    material = _subelement(visual, 'material')
    _subelement(material, 'ambient', ambient)
    _subelement(material, 'diffuse', diffuse)
    _subelement(material, 'specular', '0.15 0.15 0.15 1')
    if emissive is not None:
        _subelement(material, 'emissive', emissive)


def _add_box_geometry(parent, size):
    geometry = _subelement(parent, 'geometry')
    box = _subelement(geometry, 'box')
    _subelement(box, 'size', size)


def _add_floor(world):
    model = _subelement(world, 'model', name='lab_floor')
    _subelement(model, 'static', 'true')
    _subelement(
        model,
        'pose',
        f'{FLOOR_SIZE[0] / 2} {FLOOR_SIZE[1] / 2} -0.01 0 0 0',
    )
    link = _subelement(model, 'link', name='floor_link')
    collision = _subelement(link, 'collision', name='floor_collision')
    _add_box_geometry(collision, f'{FLOOR_SIZE[0]} {FLOOR_SIZE[1]} 0.02')
    visual = _subelement(link, 'visual', name='floor_visual')
    _add_box_geometry(visual, f'{FLOOR_SIZE[0]} {FLOOR_SIZE[1]} 0.02')
    _add_material(visual, '0.28 0.29 0.30 1', '0.32 0.33 0.34 1')


def _add_wall_segment(link, path_name, index, start, end):
    delta_x = end[0] - start[0]
    delta_y = end[1] - start[1]
    length = math.hypot(delta_x, delta_y) + WALL_THICKNESS * 0.55
    center_x = (start[0] + end[0]) / 2.0
    center_y = (start[1] + end[1]) / 2.0
    yaw = math.atan2(delta_y, delta_x)
    pose = f'{center_x:.5f} {center_y:.5f} {WALL_HEIGHT / 2:.5f} 0 0 {yaw:.7f}'
    size = f'{length:.5f} {WALL_THICKNESS:.5f} {WALL_HEIGHT:.5f}'

    collision = _subelement(
        link,
        'collision',
        name=f'{path_name}_collision_{index:03d}',
    )
    _subelement(collision, 'pose', pose)
    _add_box_geometry(collision, size)

    visual = _subelement(
        link,
        'visual',
        name=f'{path_name}_visual_{index:03d}',
    )
    _subelement(visual, 'pose', pose)
    _add_box_geometry(visual, size)
    _add_material(
        visual,
        '0.92 0.92 0.90 1',
        '0.98 0.98 0.96 1',
        '0.72 0.72 0.70 1',
    )


def _add_walls(world):
    model = _subelement(world, 'model', name='lab_track_walls')
    _subelement(model, 'static', 'true')
    link = _subelement(model, 'link', name='walls')
    for path_name, control_points, closed in BARRIER_PATHS:
        path = _sample_path(control_points, closed)
        segment_count = len(path) if closed else len(path) - 1
        for index in range(segment_count):
            start = path[index]
            end = path[(index + 1) % len(path)]
            _add_wall_segment(link, path_name, index, start, end)


def _build_world():
    # Gazebo Fortress / ROS 2 Humble validates worlds against SDF 1.7.
    sdf = ET.Element('sdf', {'version': '1.7'})
    world = _subelement(sdf, 'world', name=WORLD_NAME)

    physics = _subelement(world, 'physics', name='1ms', type='ignored')
    _subelement(physics, 'max_step_size', '0.001')
    _subelement(physics, 'real_time_factor', '1.0')

    for filename, name in (
        ('gz-sim-physics-system', 'gz::sim::systems::Physics'),
        ('gz-sim-sensors-system', 'gz::sim::systems::Sensors'),
        ('gz-sim-user-commands-system', 'gz::sim::systems::UserCommands'),
        ('gz-sim-scene-broadcaster-system', 'gz::sim::systems::SceneBroadcaster'),
    ):
        plugin = _subelement(world, 'plugin', filename=filename, name=name)
        if filename == 'gz-sim-sensors-system':
            _subelement(plugin, 'render_engine', 'ogre2')

    scene = _subelement(world, 'scene')
    _subelement(scene, 'ambient', '0.75 0.75 0.75')
    _subelement(scene, 'background', '0.55 0.57 0.60')

    light = _subelement(world, 'light', type='directional', name='lab_light')
    _subelement(light, 'cast_shadows', 'true')
    _subelement(light, 'pose', '2.2 3.2 5.0 0 0 0')
    _subelement(light, 'diffuse', '1 1 1 1')
    _subelement(light, 'specular', '0.2 0.2 0.2 1')
    _subelement(light, 'direction', '-0.25 0.15 -1.0')

    _add_floor(world)
    _add_walls(world)
    ET.indent(sdf, space='  ')
    return ET.ElementTree(sdf)


def main():
    output = Path(__file__).resolve().parents[1] / 'worlds' / 'lab_track.sdf'
    world = _build_world()
    world.write(output, encoding='utf-8', xml_declaration=True)
    print(f'Wrote {output}')
    print(f'Floor: {FLOOR_SIZE[0]:.2f} m x {FLOOR_SIZE[1]:.2f} m')
    print(f'Clear lane width: {LANE_WIDTH:.2f} m')
    spawn_x, spawn_y, spawn_yaw = RECOMMENDED_SPAWN
    print(
        f'Recommended spawn: x={spawn_x:.2f} '
        f'y={spawn_y:.2f} yaw={spawn_yaw:.4f}'
    )


if __name__ == '__main__':
    main()
