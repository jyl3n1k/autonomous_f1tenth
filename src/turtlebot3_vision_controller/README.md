# TurtleBot3 vision corridor controller

This package uses the simulated TurtleBot3 camera to segment the bright,
low-saturation driving surface from the yellow track exterior. It estimates
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

Inspect what the controller detects:

```bash
ros2 run rqt_image_view rqt_image_view \
  /turtlebot3/vision/debug_image
```

The green overlay should cover the drivable corridor, blue horizontal lines
should span it, and green points should follow its center. Stop the controller
with Ctrl+C before teleoperating the robot.

## Tuning order

1. Adjust `min_value` and `max_saturation` until only the track floor is green.
2. Adjust `roi_top_fraction`, `near_row_fraction`, and `far_row_fraction` so
   the sampled rows remain on the visible track.
3. Tune `kp_heading`, then `kp_lateral`; add only enough `kd_lateral` to reduce
   oscillation.
4. Increase `max_linear_speed` only after all six narrow tracks run reliably.

All defaults are in `config/narrow_track.yaml`.
