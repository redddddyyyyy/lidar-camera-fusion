"""
ROS 2 launch file — starts the fusion node and RViz together.

Usage:
    source /opt/ros/humble/setup.bash
    python3 ros2_node/launch_fusion.py

Or via ros2 launch (if installed as a package):
    ros2 launch lidar_camera_fusion launch_fusion.py
"""

import os
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    rviz_cfg = os.path.join(repo_dir, 'rviz', 'fusion.rviz')

    return LaunchDescription([
        DeclareLaunchArgument('camera_topic', default_value='/camera/image_raw'),
        DeclareLaunchArgument('lidar_topic',  default_value='/velodyne_points'),
        DeclareLaunchArgument('calib_yaml',   default_value=''),
        DeclareLaunchArgument('yolo_model',   default_value='yolov8n.pt'),
        DeclareLaunchArgument('confidence',   default_value='0.45'),
        DeclareLaunchArgument('lidar_frame',  default_value='velodyne'),

        Node(
            package='lidar_camera_fusion',
            executable='fusion_node',
            name='lidar_camera_fusion',
            parameters=[{
                'camera_topic': LaunchConfiguration('camera_topic'),
                'lidar_topic':  LaunchConfiguration('lidar_topic'),
                'calib_yaml':   LaunchConfiguration('calib_yaml'),
                'yolo_model':   LaunchConfiguration('yolo_model'),
                'confidence':   LaunchConfiguration('confidence'),
                'lidar_frame':  LaunchConfiguration('lidar_frame'),
            }],
            output='screen',
        ),

        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', rviz_cfg],
            output='screen',
        ),
    ])
