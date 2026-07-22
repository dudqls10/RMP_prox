#!/usr/bin/env python3
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def _include_canonical_validation_launch(context):
    profile = LaunchConfiguration("canonical_profile").perform(context).strip().lower()
    if profile not in {"normal", "forced_stuck"}:
        raise RuntimeError(
            'Launch argument "canonical_profile" must be "normal" or "forced_stuck".'
        )

    # forced_stuck deliberately makes canonical stuck confidence permissive. It
    # verifies state transitions and logging; it is not a controller tuning profile.
    if profile == "forced_stuck":
        canonical_profile_args = {
            "tangent_escape_rmp_progress_low_threshold": "1.0",
            "tangent_escape_rmp_progress_ok_threshold": "2.0",
            "tangent_escape_rmp_still_speed_threshold": "1.0",
            "tangent_escape_rmp_moving_speed_threshold": "10.0",
            "tangent_escape_rmp_intent_on_speed": "0.000001",
            "tangent_escape_rmp_intent_full_speed": "0.00001",
            "tangent_escape_rmp_minimum_drive_duration": "0.1",
            "tangent_escape_rmp_command_test_distance": "0.001",
            "tangent_escape_rmp_minimum_move_ratio": "0.5",
        }
    else:
        canonical_profile_args = {
            "tangent_escape_rmp_progress_low_threshold": "0.001",
            "tangent_escape_rmp_progress_ok_threshold": "0.01",
            "tangent_escape_rmp_still_speed_threshold": "0.003",
            "tangent_escape_rmp_moving_speed_threshold": "0.02",
            "tangent_escape_rmp_intent_on_speed": "0.005",
            "tangent_escape_rmp_intent_full_speed": "0.03",
            "tangent_escape_rmp_minimum_drive_duration": "0.2",
            "tangent_escape_rmp_command_test_distance": "0.01",
            "tangent_escape_rmp_minimum_move_ratio": "0.2",
        }

    package_share = get_package_share_directory("rb10_rmpflow_rviz")
    base_launch = os.path.join(package_share, "launch", "rb10_rmpflow_fake_proximity.launch.py")
    launch_arguments = {
        "use_rviz": LaunchConfiguration("use_rviz"),
        "control_rate": "100.0",
        "goal_x": LaunchConfiguration("goal_x"),
        "goal_y": LaunchConfiguration("goal_y"),
        "goal_z": LaunchConfiguration("goal_z"),
        "start_fake_proximity": LaunchConfiguration("start_fake_proximity"),
        "fake_scenario": LaunchConfiguration("fake_scenario"),
        "fake_sensor_name": LaunchConfiguration("fake_sensor_name"),
        "fake_active_sensor_names": LaunchConfiguration("fake_active_sensor_names"),
        "fake_range_m": LaunchConfiguration("fake_range_m"),
        "fake_start_range_m": LaunchConfiguration("fake_start_range_m"),
        "fake_end_range_m": LaunchConfiguration("fake_end_range_m"),
        "fake_start_s": LaunchConfiguration("fake_start_s"),
        "fake_duration_s": LaunchConfiguration("fake_duration_s"),
        "fake_hold_s": LaunchConfiguration("fake_hold_s"),
        "enable_tangent_escape_rmp": LaunchConfiguration("enable_tangent_escape_rmp"),
        "tangent_escape_rmp_acceleration_model": "canonical_velocity",
        "tangent_escape_rmp_max_accel": LaunchConfiguration(
            "tangent_escape_rmp_max_accel"
        ),
        "tangent_escape_rmp_normal_tolerance": LaunchConfiguration(
            "tangent_escape_rmp_normal_tolerance"
        ),
        "tangent_escape_rmp_blocked_memory_penalty_weight": LaunchConfiguration(
            "tangent_escape_rmp_blocked_memory_penalty_weight"
        ),
        "publish_tangent_escape_rmp_data": "true",
        "publish_tangent_escape_geometry_debug": "true",
        "use_rmpflow_trace_logger": "true",
        "rmpflow_trace_log_rate": LaunchConfiguration("rmpflow_trace_log_rate"),
        "rmpflow_trace_log_directory": LaunchConfiguration(
            "rmpflow_trace_log_directory"
        ),
        **canonical_profile_args,
    }
    return [
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(base_launch),
            launch_arguments=launch_arguments.items(),
        )
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            "canonical_profile",
            default_value="normal",
            description=(
                "normal checks real detector behavior; forced_stuck deliberately forces "
                "canonical stuck-confidence transitions for validation."
            ),
        ),
        DeclareLaunchArgument("use_rviz", default_value="true"),
        DeclareLaunchArgument(
            "enable_tangent_escape_rmp",
            default_value="true",
            description="Enable the Escape leaf in this dedicated validation launch.",
        ),
        DeclareLaunchArgument(
            "start_fake_proximity",
            default_value="false",
            description="Keep false to inject fake sensors from a separate terminal.",
        ),
        DeclareLaunchArgument("goal_x", default_value="0.6"),
        DeclareLaunchArgument("goal_y", default_value="-0.4"),
        DeclareLaunchArgument("goal_z", default_value="0.6"),
        DeclareLaunchArgument("fake_scenario", default_value="approach_retreat"),
        DeclareLaunchArgument("fake_sensor_name", default_value="tof_W"),
        DeclareLaunchArgument("fake_active_sensor_names", default_value=""),
        DeclareLaunchArgument("fake_range_m", default_value="0.06"),
        DeclareLaunchArgument("fake_start_range_m", default_value="0.50"),
        DeclareLaunchArgument("fake_end_range_m", default_value="0.50"),
        DeclareLaunchArgument("fake_start_s", default_value="2.0"),
        DeclareLaunchArgument("fake_duration_s", default_value="12.0"),
        DeclareLaunchArgument("fake_hold_s", default_value="6.0"),
        DeclareLaunchArgument("tangent_escape_rmp_max_accel", default_value="0.6"),
        DeclareLaunchArgument(
            "tangent_escape_rmp_normal_tolerance",
            default_value="0.20",
        ),
        DeclareLaunchArgument(
            "tangent_escape_rmp_blocked_memory_penalty_weight",
            default_value="2.0",
        ),
        DeclareLaunchArgument("rmpflow_trace_log_rate", default_value="100.0"),
        DeclareLaunchArgument(
            "rmpflow_trace_log_directory",
            default_value=os.path.expanduser("~/ros2_ws/log/rmpflow_trace"),
        ),
        OpaqueFunction(function=_include_canonical_validation_launch),
    ])
