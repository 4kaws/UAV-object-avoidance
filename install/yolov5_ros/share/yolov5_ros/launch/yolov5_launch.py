from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='yolov5_ros',
            executable='object_detector',
            name='yolov5_node',
            output='screen',
            parameters=[{'param_name': 'param_value'}]
        )
    ])
