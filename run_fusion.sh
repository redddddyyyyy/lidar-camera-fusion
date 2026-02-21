#!/bin/bash
source /opt/ros/humble/setup.bash
cd "$(dirname "$0")"
python3 ros2_node/fusion_node.py --ros-args -p use_sim_time:=true -p camera_topic:=/sensing/camera/camera0/image_rect_color -p lidar_topic:=/sensing/lidar/top/outlier_filtered/pointcloud -p lidar_frame:=velodyne_top_base_link -p confidence:=0.2
