import os
from ament_index_python import get_package_share_directory
from launch_ros.actions import Node 
from launch import LaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.actions import (
    AppendEnvironmentVariable,
    IncludeLaunchDescription,
    DeclareLaunchArgument,
    OpaqueFunction,
)
from launch.substitutions import LaunchConfiguration

def launch(context, *args, **kwargs):
    pkg_ros_gz_sim = get_package_share_directory('ros_gz_sim')
    pkg_environments = get_package_share_directory('environments')

    track = LaunchConfiguration('track').perform(context)
    car_name = LaunchConfiguration('car_name').perform(context)
    robot_model = LaunchConfiguration('robot_model').perform(context)
    spawn_x = LaunchConfiguration('spawn_x').perform(context)
    spawn_y = LaunchConfiguration('spawn_y').perform(context)
    spawn_yaw = LaunchConfiguration('spawn_yaw').perform(context)
    
    gz_sim = IncludeLaunchDescription(
        launch_description_source=PythonLaunchDescriptionSource(
            os.path.join(pkg_ros_gz_sim, 'launch', 'gz_sim.launch.py')),
        launch_arguments={
            'gz_args': f'-s -r {pkg_environments}/worlds/{track}.sdf',
        }.items()
    )

    reset = Node(
        package='environments',
        executable='F1TenthReset',
        parameters=[{
            'env_name': 'car_track',
            'robot_z': 0.01 if robot_model == 'turtlebot3_burger_cam' else 0.0,
        }],
        output='screen',
        emulate_tty=True,
    )
    goal = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=[
            '-name', 'goal',
            '-file', os.path.join(pkg_environments, 'sdf', 'goal.sdf'),
            '-x', '1.0',
            '-y', '1.0',
            '-z', '1.0',
        ],
        output='screen',
    )

    if robot_model == 'f1tenth':
        pkg_f1tenth_bringup = get_package_share_directory('f1tenth_bringup')
        robot = IncludeLaunchDescription(
            launch_description_source=PythonLaunchDescriptionSource(
                os.path.join(pkg_f1tenth_bringup, 'simulation_bringup.launch.py')),
            launch_arguments={
                'name': car_name,
                'world': 'empty'
            }.items()
        )
        return [gz_sim, robot, goal, reset]

    if robot_model == 'turtlebot3_burger_cam':
        if car_name != 'turtlebot3':
            raise RuntimeError(
                "The TurtleBot model currently requires car_name='turtlebot3' "
                "because its Gazebo topics are namespaced in the SDF."
            )

        model_path = os.path.join(
            pkg_environments,
            'models',
            'turtlebot3_burger_cam',
            'model.sdf',
        )
        bridge_path = os.path.join(
            pkg_environments,
            'config',
            'turtlebot3_burger_cam_bridge.yaml',
        )

        robot = Node(
            package='ros_gz_sim',
            executable='create',
            arguments=[
                '-name', car_name,
                '-file', model_path,
                '-x', spawn_x,
                '-y', spawn_y,
                '-z', '0.01',
                '-Y', spawn_yaw,
            ],
            output='screen',
        )
        robot_bridge = Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            arguments=[
                '--ros-args',
                '-p',
                f'config_file:={bridge_path}',
            ],
            output='screen',
        )
        image_bridge = Node(
            package='ros_gz_image',
            executable='image_bridge',
            arguments=['/turtlebot3/camera/image_raw'],
            output='screen',
        )
        return [gz_sim, robot, robot_bridge, image_bridge, goal, reset]

    raise RuntimeError(
        f"Unsupported robot_model '{robot_model}'. "
        "Expected 'f1tenth' or 'turtlebot3_burger_cam'."
    )

def generate_launch_description():

    track_arg = DeclareLaunchArgument(
        'track',
        default_value='track_1'
    )

    car_name = DeclareLaunchArgument(
        'car_name',
        default_value='f1tenth'
    )

    robot_model = DeclareLaunchArgument(
        'robot_model',
        default_value='f1tenth'
    )

    spawn_x = DeclareLaunchArgument(
        'spawn_x',
        default_value='0.0'
    )

    spawn_y = DeclareLaunchArgument(
        'spawn_y',
        default_value='0.0'
    )

    spawn_yaw = DeclareLaunchArgument(
        'spawn_yaw',
        default_value='0.0'
    )

    pkg_environments = get_package_share_directory('environments')
    model_resource_path = AppendEnvironmentVariable(
        'GZ_SIM_RESOURCE_PATH',
        os.path.join(pkg_environments, 'models'),
    )

    service_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        output='screen',
        arguments=[
            f'/world/empty/control@ros_gz_interfaces/srv/ControlWorld',
            f'/world/empty/create@ros_gz_interfaces/srv/SpawnEntity',
            f'/world/empty/remove@ros_gz_interfaces/srv/DeleteEntity',
            f'/world/empty/set_pose@ros_gz_interfaces/srv/SetEntityPose',
            f'/world/empty/clock@rosgraph_msgs/msg/Clock@gz.msgs.Clock'
        ],
        remappings=[
            (f'/world/empty/clock', f'/clock'),
        ],
    )

    stepping_service = Node(
            package='environments',
            executable='SteppingService',
            output='screen',
            emulate_tty=True,
    )

    return LaunchDescription([
        model_resource_path,
        track_arg,
        car_name,
        robot_model,
        spawn_x,
        spawn_y,
        spawn_yaw,
        OpaqueFunction(function=launch),
        service_bridge,
        stepping_service,
])
