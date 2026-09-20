import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    package_share = get_package_share_directory(
        'turtlebot3_vision_controller'
    )
    default_config = os.path.join(
        package_share,
        'config',
        'narrow_track.yaml',
    )

    config_argument = DeclareLaunchArgument(
        'config',
        default_value=default_config,
        description='Controller parameter YAML file',
    )
    controller = Node(
        package='turtlebot3_vision_controller',
        executable='vision_controller',
        name='turtlebot3_vision_controller',
        output='screen',
        parameters=[LaunchConfiguration('config'), {
            'reverse_route': ParameterValue(
                LaunchConfiguration('reverse_route'), value_type=bool),
            'waypoint_origin_yaw': ParameterValue(
                LaunchConfiguration('waypoint_origin_yaw'), value_type=float),
        }],
    )

    return LaunchDescription([
        config_argument,
        DeclareLaunchArgument(
            'reverse_route', default_value='false',
            description='Follow the stored route in the opposite direction'),
        DeclareLaunchArgument(
            'waypoint_origin_yaw', default_value='0.0',
            description='Must match the simulator spawn_yaw, in radians'),
        controller,
    ])
