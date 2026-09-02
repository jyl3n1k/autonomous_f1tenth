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

For `lab_track`, select the white-wall configuration instead:

```bash
ros2 launch turtlebot3_vision_controller vision_controller.launch.py \
  config:=$(ros2 pkg prefix turtlebot3_vision_controller)/share/\
turtlebot3_vision_controller/config/lab_track.yaml
```

Inspect what the controller detects:

```bash
ros2 run rqt_image_view rqt_image_view \
  /turtlebot3/vision/debug_image
```

With the narrow-track configuration, the green overlay covers the drivable
surface. With the lab configuration, it covers the two white barriers. In
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
