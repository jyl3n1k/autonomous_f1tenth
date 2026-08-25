# Autonomous F1tenth
Using reinforcement learning techniques to drive the f1tenth vehicle platform.

### Dependencies
| Dependencies | Version |
| ----------- | ----------- |
| Gazebo | [Garden](https://gazebosim.org/docs/garden/install_ubuntu_src) |
| ROS2 | [Humble Hawksbill](https://docs.ros.org/en/humble/Installation.html) |
| colcon | [ROS2](https://colcon.readthedocs.io/en/released/user/installation.html) |
| CARES RL | [Link](https://github.com/UoA-CARES/cares_reinforcement_learning) |
| f1tenth | [Link](https://github.com/UoA-CARES/f1tenth) |

We source build Gazebo Garden, and use a forked `gz-sim`. To use the forked `gz-sim` run the following command before building Gazebo
```
rm -rdf gz-sim
git clone https://github.com/UoA-CARES/gz-sim.git
```

If running on the physical car, install additional dependency
```
sudo apt-get install -y ros-humble-urg-node
```

# Installation Instructions
Follow these instructions to run/test this repository on your local machine.

### Locally
Ensure you have installed the dependencies outlined above.

Clone the repository
```
git clone --recurse-submodules https://github.com/UoA-CARES/autonomous_f1tenth.git
```

Install dependencies using `rosdep`
```
cd autonomous_f1tenth/
rosdep update -y
rosdep install --from-paths src --ignore-src -r -y --rosdistro humble
```

### Using Docker (Recommended)

Use this command to run the docker, it will automatically pull the image if not found locally:
```
docker run --rm -it --network host --gpus all -e DISPLAY -e GZ_PARTITION=12 -e ROS_DOMAIN_ID=12 -v "$PWD/rl_logs:/ws/rl_logs" caresrl/autonomous_f1tenth:latest bash
```
**Note: it is important to have a different GZ_PARITION/ROS_DOMAIN_ID for every container you plan on running**

# Runnning the Simulations
There are several environments that are available. Refer to the wiki for more detailed information.

This repository provides two functions:
1. **Training** a reinforcement learning agent on a particular _environment_
2. **Testing** (evaluating) your trained agent on a particular _environment_

To control **aspects** of the **training/testing** – eg. environment, hyperparameters, model paths – edit the `src/reinforcement_learning/config/train.yaml` and `src/reinforcement_learning/config/test.yaml` files respectively.

An example of the yaml that controls training is shown below:
```
train:
  ros__parameters:
    environment: 'CarBeat' # CarGoal, CarWall, CarBlock, CarTrack, CarBeat
    max_steps_exploration: 5000
    track: 'track_2' # track_1, track_2, track_3 -> only applies for CarTrack
    max_steps_exploration: 1000
    max_steps_training: 1000000
    reward_range: 1.0
    collision_range: 0.2
    observation_mode: 'no_position'
    # gamma: 0.95
    # tau: 0.005
    # g: 5
    # batch_size: 32
    # buffer_size: 1000000
    # actor_lr: 0.0001
    # critic_lr: 0.001
    # max_steps: 10
    # step_length: 0.25
    # seed: 123
```

Once you have configured your desired experiment you can run the following to launch your training/testing:
```bash
colcon build
. install/setup.bash
ros2 launch reinforcement_learning train.launch.py # or test.launch.py if you were testing your agents
```
Refer to this [link](https://docs.ros.org/en/foxy/Tutorials/Beginner-Client-Libraries/Colcon-Tutorial.html) for more information on `colcon build` in `ros2`

## TurtleBot3 Burger camera model

The `CarTrack` environment can launch either the F1TENTH vehicle or a TurtleBot3
Burger camera model. To train with the TurtleBot, set these parameters in
`src/reinforcement_learning/config/train.yaml`:

```yaml
environment: 'CarTrack'
robot_model: 'turtlebot3_burger_cam'
car_name: 'turtlebot3'
```

Use the same values in `test.yaml` when evaluating a TurtleBot policy. The
existing lidar and odometry observations, track-progress reward, reset service,
and stepping service are reused. TurtleBot actions are interpreted as linear
velocity and angular velocity and use the limits in
`src/environments/config/config.yaml`.

The camera is published on `/turtlebot3/camera/image_raw`, with camera metadata
on `/turtlebot3/camera/camera_info`. Camera pixels are not currently part of the
RL observation; the policy continues to use lidar and odometry, so TurtleBot
policies should be trained separately from F1TENTH policies. The upstream
fisheye sensor is represented as an 80-degree perspective RGB camera because
Gazebo Garden's wide-angle camera requires Ogre 1.x while these worlds use
Ogre2.

The vendored TurtleBot model and meshes originate from ROBOTIS-GIT's
`turtlebot3_simulations` project and retain its Apache-2.0 license under
`src/environments/models/LICENSE.turtlebot3_simulations`.
