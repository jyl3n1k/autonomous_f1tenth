import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


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
        parameters=[LaunchConfiguration('config')],
    )

    return LaunchDescription([config_argument, controller])
