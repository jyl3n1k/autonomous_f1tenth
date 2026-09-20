#!/usr/bin/env python3
"""Plot recorded lab driving against the actual Gazebo wall geometry."""

import argparse
from pathlib import Path
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--baseline', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run = np.genfromtxt(args.run, names=True, delimiter=',')
    baseline = (np.genfromtxt(args.baseline, names=True, delimiter=',')
                if args.baseline else None)
    root = Path(__file__).resolve().parents[1]
    world = ET.parse(root / 'src/environments/worlds/lab_track.sdf')
    fig = plt.figure(figsize=(12, 9), layout='constrained')
    grid = fig.add_gridspec(2, 2, width_ratios=[1, 1.2])
    course = fig.add_subplot(grid[:, 0])
    course.set_facecolor('#374151')
    for collision in world.findall('.//model[@name="lab_track_walls"]//collision'):
        x, y, _, _, _, yaw = map(float, collision.findtext('pose').split())
        length, width, _ = map(float, collision.findtext('geometry/box/size').split())
        corners = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * [length/2, width/2]
        rotation = np.array([[np.cos(yaw), -np.sin(yaw)],
                             [np.sin(yaw), np.cos(yaw)]])
        course.add_patch(Polygon(corners @ rotation.T + [x, y],
                                 color='#e5e7eb', linewidth=0))
    course.plot(run['x'] + 2.2, run['y'] + 0.65, '--',
                color='#f59e0b', linewidth=1, alpha=0.8, label='Wheel odometry')
    course.plot(run['truth_x'] + 2.2, run['truth_y'] + 0.65,
                color='#34d399', linewidth=1.7, label='Actual robot path')
    course.plot(2.2, 0.65, 'o', color='#38bdf8', label='Start')
    course.set(xlim=(0, 4.4), ylim=(0, 6.4), aspect='equal',
               xlabel='World x (m)', ylabel='World y (m)',
               title=f"Actual motion: {int(run['laps'][-1])} completed laps")
    course.legend(loc='upper left', fontsize=8)
    for index, (field, label) in enumerate([
            ('command_v', 'Forward command (m/s)'),
            ('command_w', 'Turning command (rad/s)')]):
        axis = fig.add_subplot(grid[index, 1])
        if baseline is not None:
            axis.plot(baseline['time'], baseline[field], color='#dc2626',
                      linewidth=0.7, alpha=0.55, label='Before')
        axis.plot(run['time'], run[field], color='#059669', linewidth=1,
                  label='After')
        axis.set(xlabel='Simulation time (s)', ylabel=label)
        axis.grid(alpha=0.2)
        axis.legend(loc='upper right')
    fig.suptitle('TurtleBot lab controller — Gazebo evaluation', fontsize=15)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    print(args.output)


if __name__ == '__main__':
    main()
