#!/usr/bin/env python3
"""Generate compact, fixed-width TurtleBot training tracks."""

import importlib.util
import math
from pathlib import Path
import struct


ROOT = Path(__file__).resolve().parents[1]
WAYPOINTS_FILE = ROOT / 'src/environments/environments/waypoints.py'
MESH_DIRECTORY = ROOT / 'src/environments/meshes'
WORLD_DIRECTORY = ROOT / 'src/environments/worlds'

TRACKS = {
    'turtlebot_track_01': ('track_01_1m', 0.60),
    'turtlebot_track_05': ('track_05_1m', 0.50),
}

CORRIDOR_WIDTH = 0.50
MARGIN = 0.70
GRID_SIZE = 0.05
CURB_HEIGHT = 0.12


def load_waypoints():
    spec = importlib.util.spec_from_file_location(
        'track_waypoints', WAYPOINTS_FILE
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.waypoints


def transform(points, scale):
    minimum_x = min(point.x for point in points)
    minimum_y = min(point.y for point in points)
    return [
        (
            (point.x - minimum_x) * scale + MARGIN + CORRIDOR_WIDTH / 2,
            (point.y - minimum_y) * scale + MARGIN + CORRIDOR_WIDTH / 2,
        )
        for point in points
    ]


def distance_to_centerline(x, y, points):
    minimum_squared_distance = math.inf
    for start, end in zip(points, points[1:] + points[:1]):
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        squared_length = dx * dx + dy * dy
        if squared_length == 0.0:
            projection = 0.0
        else:
            projection = max(
                0.0,
                min(
                    1.0,
                    ((x - start[0]) * dx + (y - start[1]) * dy)
                    / squared_length,
                ),
            )
        nearest_x = start[0] + projection * dx
        nearest_y = start[1] + projection * dy
        squared_distance = (x - nearest_x) ** 2 + (y - nearest_y) ** 2
        minimum_squared_distance = min(
            minimum_squared_distance, squared_distance
        )
    return math.sqrt(minimum_squared_distance)


def triangle(normal, a, b, c):
    return struct.pack('<12fH', *(normal + a + b + c), 0)


def write_mesh(path, points):
    maximum_x = max(point[0] for point in points) + MARGIN + CORRIDOR_WIDTH / 2
    maximum_y = max(point[1] for point in points) + MARGIN + CORRIDOR_WIDTH / 2
    columns = math.ceil(maximum_x / GRID_SIZE)
    rows = math.ceil(maximum_y / GRID_SIZE)
    occupied = []
    half_width = CORRIDOR_WIDTH / 2

    for row in range(rows):
        occupied_row = []
        y = (row + 0.5) * GRID_SIZE
        for column in range(columns):
            x = (column + 0.5) * GRID_SIZE
            occupied_row.append(
                distance_to_centerline(x, y, points) > half_width
            )
        occupied.append(occupied_row)

    triangles = []
    for row in range(rows):
        for column in range(columns):
            if not occupied[row][column]:
                continue
            x0 = column * GRID_SIZE
            x1 = (column + 1) * GRID_SIZE
            y0 = row * GRID_SIZE
            y1 = (row + 1) * GRID_SIZE
            z0 = 0.0
            z1 = CURB_HEIGHT

            triangles.extend(
                [
                    triangle(
                        (0.0, 0.0, 1.0),
                        (x0, y0, z1),
                        (x1, y0, z1),
                        (x1, y1, z1),
                    ),
                    triangle(
                        (0.0, 0.0, 1.0),
                        (x0, y0, z1),
                        (x1, y1, z1),
                        (x0, y1, z1),
                    ),
                    triangle(
                        (0.0, 0.0, -1.0),
                        (x0, y1, z0),
                        (x1, y1, z0),
                        (x1, y0, z0),
                    ),
                    triangle(
                        (0.0, 0.0, -1.0),
                        (x0, y1, z0),
                        (x1, y0, z0),
                        (x0, y0, z0),
                    ),
                ]
            )

            neighbours = (
                (-1, 0, (-1.0, 0.0, 0.0), (x0, y1), (x0, y0)),
                (1, 0, (1.0, 0.0, 0.0), (x1, y0), (x1, y1)),
                (0, -1, (0.0, -1.0, 0.0), (x0, y0), (x1, y0)),
                (0, 1, (0.0, 1.0, 0.0), (x1, y1), (x0, y1)),
            )
            for dc, dr, normal, first, second in neighbours:
                neighbour_column = column + dc
                neighbour_row = row + dr
                exposed = (
                    neighbour_column < 0
                    or neighbour_column >= columns
                    or neighbour_row < 0
                    or neighbour_row >= rows
                    or not occupied[neighbour_row][neighbour_column]
                )
                if exposed:
                    a = (first[0], first[1], z0)
                    b = (second[0], second[1], z0)
                    c = (second[0], second[1], z1)
                    d = (first[0], first[1], z1)
                    triangles.extend(
                        [triangle(normal, a, b, c), triangle(normal, a, c, d)]
                    )

    header = b'TurtleBot fixed-width track'.ljust(80, b'\0')
    path.write_bytes(
        header + struct.pack('<I', len(triangles)) + b''.join(triangles)
    )
    return columns * GRID_SIZE, rows * GRID_SIZE, len(triangles)


def world_xml(name, width, height):
    return f'''<sdf version="1.6">
  <world name="empty">
    <physics name="1ms" type="ignored">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>10.0</real_time_factor>
    </physics>
    <plugin filename="gz-sim-physics-system"
            name="gz::sim::systems::Physics"/>
    <plugin filename="gz-sim-sensors-system" name="gz::sim::systems::Sensors">
      <render_engine>ogre2</render_engine>
    </plugin>
    <plugin filename="gz-sim-user-commands-system"
            name="gz::sim::systems::UserCommands"/>
    <plugin filename="gz-sim-scene-broadcaster-system"
            name="gz::sim::systems::SceneBroadcaster"/>
    <scene>
      <ambient>1.0 1.0 1.0</ambient>
      <background>0.8 0.8 0.8</background>
      <sky/>
    </scene>
    <model name="base_plate">
      <pose>{width / 2:.3f} {height / 2:.3f} -0.005 0 0 0</pose>
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box><size>{width:.3f} {height:.3f} 0.01</size></box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box><size>{width:.3f} {height:.3f} 0.01</size></box>
          </geometry>
          <material>
            <ambient>0.6 0.6 0.6 1</ambient>
            <diffuse>0.6 0.6 0.6 1</diffuse>
          </material>
        </visual>
      </link>
    </model>
    <light type="directional" name="sun">
      <pose>0 0 10 0 0 0</pose>
      <diffuse>1 1 1 1</diffuse>
      <specular>0.5 0.5 0.5 1</specular>
      <direction>-0.5 0.1 -0.9</direction>
    </light>
    <model name="{name}">
      <static>true</static>
      <link name="track">
        <visual name="track_visual">
          <geometry><mesh><uri>../meshes/{name}.stl</uri></mesh></geometry>
          <material>
            <ambient>1 0.4 0 1</ambient>
            <diffuse>1 0.4 0.01 1</diffuse>
          </material>
        </visual>
        <collision name="track_collision">
          <geometry><mesh><uri>../meshes/{name}.stl</uri></mesh></geometry>
        </collision>
      </link>
    </model>
  </world>
</sdf>
'''


def main():
    all_waypoints = load_waypoints()
    for output_name, (source_name, scale) in TRACKS.items():
        source_points = all_waypoints[source_name]
        points = transform(source_points, scale)
        mesh_path = MESH_DIRECTORY / f'{output_name}.stl'
        width, height, count = write_mesh(mesh_path, points)
        world_path = WORLD_DIRECTORY / f'{output_name}.sdf'
        world_path.write_text(world_xml(output_name, width, height))
        lap_length = sum(
            math.dist(start, end)
            for start, end in zip(points, points[1:] + points[:1])
        )
        print(
            f'{output_name}: {lap_length:.2f} m, '
            f'{CORRIDOR_WIDTH:.2f} m corridor, '
            f'{width:.2f} x {height:.2f} m, {count} triangles'
        )


if __name__ == '__main__':
    main()
