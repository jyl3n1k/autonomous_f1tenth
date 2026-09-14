#!/usr/bin/env python3
"""Render a simulator-style preview of the generated lab-track SDF."""

from __future__ import annotations

import math
from pathlib import Path
from xml.etree import ElementTree as ET

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


ROOT = Path(__file__).resolve().parents[1]
WORLD = ROOT / "src" / "environments" / "worlds" / "lab_track.sdf"
OUTPUT = ROOT / "lab_track_preview.png"


def parse_numbers(text: str | None, expected: int) -> list[float]:
    values = [float(value) for value in (text or "").split()]
    if len(values) != expected:
        raise ValueError(f"Expected {expected} values, got {values}")
    return values


def box_faces(center: list[float], size: list[float], yaw: float):
    hx, hy, hz = (dimension / 2.0 for dimension in size)
    local = [
        (-hx, -hy, -hz), (hx, -hy, -hz), (hx, hy, -hz), (-hx, hy, -hz),
        (-hx, -hy, hz), (hx, -hy, hz), (hx, hy, hz), (-hx, hy, hz),
    ]
    cosine, sine = math.cos(yaw), math.sin(yaw)
    vertices = [
        (
            center[0] + cosine * x - sine * y,
            center[1] + sine * x + cosine * y,
            center[2] + z,
        )
        for x, y, z in local
    ]
    return [
        [vertices[index] for index in face]
        for face in (
            (0, 1, 2, 3), (4, 5, 6, 7),
            (0, 1, 5, 4), (1, 2, 6, 5),
            (2, 3, 7, 6), (3, 0, 4, 7),
        )
    ]


def main() -> None:
    tree = ET.parse(WORLD)
    root = tree.getroot()

    fig = plt.figure(figsize=(10.5, 12), dpi=160, facecolor="#8b9098")
    ax = fig.add_subplot(111, projection="3d", computed_zorder=False)
    ax.set_facecolor("#8b9098")

    for model in root.findall(".//model"):
        model_pose = parse_numbers(model.findtext("pose", "0 0 0 0 0 0"), 6)
        is_floor = model.get("name") == "lab_floor"
        for visual in model.findall(".//visual"):
            pose = parse_numbers(visual.findtext("pose", "0 0 0 0 0 0"), 6)
            size_node = visual.find("geometry/box/size")
            if size_node is None:
                continue
            size = parse_numbers(size_node.text, 3)
            center = [model_pose[i] + pose[i] for i in range(3)]
            faces = box_faces(center, size, model_pose[5] + pose[5])
            if is_floor:
                color, edge, width, zorder = "#4d5155", "#3d4145", 0.25, 1
            else:
                # Gazebo does not draw polygon outlines, so omit them here too.
                color, edge, width, zorder = "#f4f3ed", "none", 0.0, 4
            collection = Poly3DCollection(
                faces,
                facecolors=color,
                edgecolors=edge,
                linewidths=width,
                zorder=zorder,
            )
            ax.add_collection3d(collection)

    # A subtle spawn marker gives scale and orientation without obscuring the track.
    spawn_x, spawn_y = 1.52, 0.66
    ax.scatter([spawn_x], [spawn_y], [0.025], s=34, color="#f2c14e", depthshade=False)
    ax.quiver(
        spawn_x, spawn_y, 0.028, -0.24, 0, 0,
        color="#f2c14e", linewidth=2.2, arrow_length_ratio=0.32,
    )

    ax.set_xlim(-0.15, 4.55)
    ax.set_ylim(-0.15, 6.55)
    ax.set_zlim(-0.02, 1.1)
    ax.set_box_aspect((4.7, 6.7, 2.15))
    ax.view_init(elev=58, azim=-54)
    ax.set_proj_type("ortho")
    ax.set_axis_off()
    ax.set_title(
        "LAB TRACK  •  4.4 m × 6.4 m  •  0.72 m lane",
        color="white",
        fontsize=14,
        fontweight="semibold",
        pad=5,
    )
    fig.text(
        0.5,
        0.045,
        "Gazebo world preview   •   yellow marker = recommended TurtleBot spawn",
        ha="center",
        color="#eef0f2",
        fontsize=9,
    )
    plt.subplots_adjust(left=0, right=1, bottom=0.07, top=0.94)
    fig.savefig(OUTPUT, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.12)
    print(OUTPUT)


if __name__ == "__main__":
    main()
