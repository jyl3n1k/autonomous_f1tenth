#!/usr/bin/env python3
"""Check a padded Burger footprint against SDF wall boxes using ground truth."""
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np


def check(run):
    data = np.genfromtxt(run, delimiter=',', names=True)
    xy = np.column_stack((data['truth_x'] + 2.20, data['truth_y'] + .65))
    yaw = data['truth_yaw']
    forward = np.column_stack((np.cos(yaw), np.sin(yaw)))
    left = np.column_stack((-np.sin(yaw), np.cos(yaw)))
    # Encloses the base, wheel cylinders, LDS and caster collision shapes.
    # Add 15 mm for sampling intervals, small pitch/roll and geometry rounding.
    center = xy - .031 * forward
    hx, hy = .071 + .015, .090 + .015
    root = Path(__file__).resolve().parents[1]
    world = ET.parse(root / 'src/environments/worlds/lab_track.sdf')
    clearance = np.full(len(xy), np.inf)
    for wall in world.findall(".//model[@name='lab_track_walls']/link/collision"):
        x, y, _, _, _, angle = map(float, wall.findtext('pose').split())
        sx, sy, _ = map(float, wall.findtext('geometry/box/size').split())
        a = np.array([np.cos(angle), np.sin(angle)])
        b = np.array([-np.sin(angle), np.cos(angle)])
        gaps = []
        for axis in (forward, left, a, b):
            support = (hx * np.abs(np.sum(forward * axis, axis=1))
                       + hy * np.abs(np.sum(left * axis, axis=1))
                       + sx/2 * np.abs(np.sum(a * axis, axis=-1))
                       + sy/2 * np.abs(np.sum(b * axis, axis=-1)))
            gaps.append(np.abs(np.sum((center - [x, y]) * axis, axis=1)) - support)
        clearance = np.minimum(clearance, np.max(gaps, axis=0))
    path = Path(run).with_suffix('.json')
    summary = json.loads(path.read_text())
    summary['minimum_padded_footprint_separation_m'] = float(clearance.min())
    summary['footprint_padding_m'] = .015
    summary['footprint_check_passed'] = bool(np.all(clearance > 0))
    path.write_text(json.dumps(summary, indent=2))
    print(f'Padded footprint minimum separation: {clearance.min():.4f} m')
    return bool(np.all(clearance > 0))


if __name__ == '__main__':
    import sys
    raise SystemExit(0 if check(sys.argv[1]) else 1)
