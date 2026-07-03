#!/usr/bin/env python3
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource


def generate_launch_description():
    rmp_pkg = get_package_share_directory("rb10_rmpflow_rviz")
    base_launch = os.path.join(rmp_pkg, "launch", "rb10_rmpflow.launch.py")

    return LaunchDescription([
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(base_launch),
            launch_arguments={
                "cb_simulation": "false",
                "use_proximity_bridge": "true",
                "proximity_surface_visualization": "false",
                "surface_patch_enabled": "false",
                "surface_patch_collision_memory_enabled": "false",
                "use_interactive_goal": "false",
                "use_rmp_goal_logger": "false",
                "use_rmpflow_trace_logger": "false",
                "record_data": "true",
                "auto_start_recording": "true",
                "recording_output_prefix": "raw_distance",
            }.items(),
        ),
    ])
