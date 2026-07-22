#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def _bool_config(name):
    value = LaunchConfiguration(name)
    return ParameterValue(
        PythonExpression(['True if "', value, '" == "true" else False']),
        value_type=bool,
    )


def generate_launch_description():
    replay_node = Node(
        package="rb10_rmpflow_rviz",
        executable="fake_proximity_scenario_v2.py",
        name="fake_proximity_scenario_v2",
        output="screen",
        parameters=[{
            "csv_path": LaunchConfiguration("csv_path"),
            "timestamp_column": LaunchConfiguration("timestamp_column"),
            "distance_column_prefix": LaunchConfiguration("distance_column_prefix"),
            "input_unit": LaunchConfiguration("input_unit"),
            "playback_rate": ParameterValue(
                LaunchConfiguration("playback_rate"), value_type=float
            ),
            "publish_rate_hz": ParameterValue(
                LaunchConfiguration("publish_rate_hz"), value_type=float
            ),
            "start_offset_s": ParameterValue(
                LaunchConfiguration("start_offset_s"), value_type=float
            ),
            "duration_s": ParameterValue(
                LaunchConfiguration("duration_s"), value_type=float
            ),
            "start_delay_s": ParameterValue(
                LaunchConfiguration("start_delay_s"), value_type=float
            ),
            "start_trigger_topic": LaunchConfiguration("start_trigger_topic"),
            "loop": _bool_config("loop"),
            "interpolate_ranges": _bool_config("interpolate_ranges"),
            "inactive_range_m": ParameterValue(
                LaunchConfiguration("inactive_range_m"), value_type=float
            ),
            "minimum_valid_range_m": ParameterValue(
                LaunchConfiguration("minimum_valid_range_m"), value_type=float
            ),
            "maximum_valid_range_m": ParameterValue(
                LaunchConfiguration("maximum_valid_range_m"), value_type=float
            ),
            "output_range_scale": 0.001,
            "proximity_topic_prefix": "/fake_proximity_distance",
            "raw_topic_prefix": "/fake_raw_distance",
            "publish_raw_topics": True,
            "publish_rmp_flag": True,
            "rmp_flag_topic": "/RMP_flag",
        }],
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            "csv_path",
            description="Physical proximity-distance CSV to replay.",
        ),
        DeclareLaunchArgument("timestamp_column", default_value="timestamp_unix"),
        DeclareLaunchArgument(
            "distance_column_prefix",
            default_value="proximity_distance",
            description=(
                "Prefix for numbered distance columns 1..20, or a template "
                "containing {index}."
            ),
        ),
        DeclareLaunchArgument(
            "input_unit",
            default_value="auto",
            description="auto, millimeters, or meters.",
        ),
        DeclareLaunchArgument("playback_rate", default_value="1.0"),
        DeclareLaunchArgument(
            "publish_rate_hz",
            default_value="0.0",
            description="0 uses the CSV median sample rate multiplied by playback_rate.",
        ),
        DeclareLaunchArgument("start_offset_s", default_value="0.0"),
        DeclareLaunchArgument(
            "duration_s",
            default_value="0.0",
            description="0 replays from start_offset_s to the end of the CSV.",
        ),
        DeclareLaunchArgument("start_delay_s", default_value="1.0"),
        DeclareLaunchArgument(
            "start_trigger_topic",
            default_value="",
            description=(
                "Optional PoseStamped topic. When set, replay time starts on its "
                "first message and start_delay_s is measured from that trigger."
            ),
        ),
        DeclareLaunchArgument("loop", default_value="false"),
        DeclareLaunchArgument(
            "interpolate_ranges",
            default_value="false",
            description="Linearly interpolate physical ranges between CSV rows.",
        ),
        DeclareLaunchArgument("inactive_range_m", default_value="0.90"),
        DeclareLaunchArgument("minimum_valid_range_m", default_value="0.001"),
        DeclareLaunchArgument("maximum_valid_range_m", default_value="10.0"),
        replay_node,
    ])
