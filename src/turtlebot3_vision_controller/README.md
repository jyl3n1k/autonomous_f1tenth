# TurtleBot3 vision corridor controller

This package uses the simulated TurtleBot3 camera to follow either a bright
driving surface or the gap between two bright white barriers. It estimates
the corridor center at several look-ahead rows and publishes conservative
`geometry_msgs/Twist` commands. It does not collect training data.

## Run

Launch a narrow track and TurtleBot3 first:

```bash
ros2 launch environments cartrack.launch.py \
  track:=narrow_track_01 \
  robot_model:=turtlebot3_burger_cam \
  car_name:=turtlebot3 \
  spawn_x:=19.0 \
  spawn_y:=2.9 \
  spawn_yaw:=0.0
```

The spawn pose above is on the driving surface of `narrow_track_01`. The
launch defaults to `(0, 0)`, which is beside this track and is not a valid
place to start the vision controller.

In a second sourced terminal:

```bash
ros2 launch turtlebot3_vision_controller vision_controller.launch.py
```

For `lab_track`, use the lab-specific world and controller configuration:

```bash
ros2 launch environments cartrack.launch.py \
  track:=lab_track \
  robot_model:=turtlebot3_burger_cam \
  car_name:=turtlebot3 \
  spawn_x:=2.20 \
  spawn_y:=0.65 \
  spawn_yaw:=0.0
```

Then launch its controller configuration:

```bash
ros2 launch turtlebot3_vision_controller vision_controller.launch.py \
  config:=$(ros2 pkg prefix turtlebot3_vision_controller)/share/\
turtlebot3_vision_controller/config/lab_track.yaml
```

The lab configuration follows the black painted centerline using camera pixels.
Stored waypoints, odometry steering, IMU heading and wall-based recovery are
disabled. LiDAR only reduces forward speed near obstacles; losing the line
stops the robot. The narrow-track configuration still follows its bright floor.

Start on the line at the documented pose, facing along it. Restart the simulator
and controller after updating the world. Rebuild with:

```bash
colcon build --packages-select environments turtlebot3_vision_controller --symlink-install
source install/setup.bash
```

The black-pixel threshold is `black_max_value` (HSV value, 0–255). Debug images
overlay detected paint in green. Sensor timeouts stop motion and LiDAR braking
bypasses acceleration smoothing. The 0.14 m stop threshold is above the
scanner's 0.12 m minimum range.

For testing without a desktop renderer, append `headless_rendering:=true` to
the environment launch command. Keep only one Gazebo server per `GZ_PARTITION`
and one controller per command topic.

Inspect what the controller detects:

```bash
ros2 run rqt_image_view rqt_image_view \
  /turtlebot3/vision/debug_image
```

With the narrow-track configuration, the green overlay covers the drivable
surface. With the lab configuration, it covers the black centerline. In
both modes, blue horizontal lines should span the free corridor and green
points should follow its center. Stop the controller with Ctrl+C before
teleoperating the robot.

## Tuning order

1. Adjust `min_value` and `max_saturation` until only the track floor is green.
2. Adjust `roi_top_fraction`, `near_row_fraction`, and `far_row_fraction` so
   the sampled rows remain on the visible track.
3. Tune `kp_heading`, then `kp_lateral`; add only enough `kd_lateral` to reduce
   oscillation.
4. Increase `max_linear_speed` only after all six narrow tracks run reliably.

All defaults are in `config/narrow_track.yaml`.

## Repeat the simulation test

For an isolated black-line test on Gazebo Fortress (including simulator,
bridges, camera recordings and a ground-truth lap count), run:

```bash
source /opt/ros/humble/setup.bash
source install/setup.bash
python3 scripts/test_black_line_sim.py
```

It uses domain 87, a 10 Hz camera, a 4 ms physics step and software Ogre2
rendering. Results are written to `artifacts/black_line/lap.json` and `lap.csv`.
It fails if a lap is not completed or the robot's rectangular collision envelope,
expanded by 15 mm, intersects wall geometry at any recorded position. Ground truth is used only
for evaluation, never steering.

Recorded black-line result: one 16.10 m lap in 142.22 simulation seconds,
with at least 22 mm separation after the 15 mm footprint padding. The existing
45 mm painted line was sufficient. This validates the documented start pose
and test conditions; other starts and lighting have not been tested.

Use a separate `ROS_DOMAIN_ID` and `GZ_PARTITION` in **every** test terminal,
then launch the lab simulator at the documented start pose. Do not launch the
normal controller as well: the evaluation script starts one itself.

```bash
export ROS_DOMAIN_ID=76
export GZ_PARTITION=lab_evaluation
source install/setup.bash
```

In another terminal with the same environment, bridge simulator ground truth
for measurement (it is not used by the controller):

```bash
ros2 run ros_gz_bridge parameter_bridge \
  '/world/empty/dynamic_pose/info@tf2_msgs/msg/TFMessage[gz.msgs.Pose_V'
```

Then, in another identically configured terminal:

```bash
python3 scripts/evaluate_lab_controller.py --duration 250 --output /tmp/lab-run.csv
python3 scripts/plot_lab_evaluation.py /tmp/lab-run.csv --output /tmp/lab-run.png
```

The test writes CSV commands/trajectory and a JSON summary, and stops the
robot on exit. Lap counting uses actual simulator position, not wheel odometry;
the latter can keep advancing while the robot is pressed against a wall.
The ground-truth bridge assumes this lab world's single dynamic robot.
