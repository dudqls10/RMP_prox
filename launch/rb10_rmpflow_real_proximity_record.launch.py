#!/usr/bin/env python3
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def _create_distance_recorder(context):
    source = LaunchConfiguration("recording_range_source").perform(context).strip().lower()
    if source not in {"proximity", "raw"}:
        raise RuntimeError(
            "recording_range_source must be 'proximity' or 'raw', "
            f"got '{source}'"
        )

    topic_prefix = "/proximity_distance" if source == "proximity" else "/raw_distance"
    range_unit = "millimeters" if source == "proximity" else "raw"
    default_prefix = "proximity_distance" if source == "proximity" else "raw_distance"
    configured_prefix = LaunchConfiguration("recording_output_prefix").perform(context).strip()

    return [Node(
        package="rb10_rmpflow_rviz",
        executable="rmp_data_recorder.py",
        name="rmp_data_recorder",
        output="screen",
        parameters=[{
            "mode": "real",
            "range_unit": range_unit,
            "auto_start": ParameterValue(
                LaunchConfiguration("auto_start_recording"), value_type=bool
            ),
            "recording_rate": ParameterValue(
                LaunchConfiguration("recording_rate"), value_type=float
            ),
            "output_directory": LaunchConfiguration("recording_output_directory"),
            "output_prefix": configured_prefix or default_prefix,
            "joint_state_topic": LaunchConfiguration("recording_joint_state_topic"),
            "range_topics": [f"{topic_prefix}{index}" for index in range(1, 21)],
        }],
    )]


def generate_launch_description():
    rmp_pkg = get_package_share_directory("rb10_rmpflow_rviz")
    base_launch = os.path.join(rmp_pkg, "launch", "rb10_rmpflow.launch.py")

    return LaunchDescription([
        DeclareLaunchArgument(
            "recording_range_source",
            default_value="proximity",
            description=(
                "Record physical /proximity_distance* values for CSV replay, or raw for "
                "sensor-signal analysis."
            ),
        ),
        DeclareLaunchArgument("auto_start_recording", default_value="true"),
        DeclareLaunchArgument("recording_rate", default_value="100.0"),
        DeclareLaunchArgument(
            "recording_output_directory",
            default_value=os.path.expanduser("~/ros2_ws/data/rmp_datasets"),
        ),
        DeclareLaunchArgument(
            "recording_output_prefix",
            default_value="",
            description="Empty selects proximity_distance or raw_distance from the source.",
        ),
        DeclareLaunchArgument(
            "recording_joint_state_topic",
            default_value="/joint_states",
        ),
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
                "record_data": "false",
            }.items(),
        ),
        OpaqueFunction(function=_create_distance_recorder),
    ])
