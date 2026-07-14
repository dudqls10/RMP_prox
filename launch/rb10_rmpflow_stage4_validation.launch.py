#!/usr/bin/env python3
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def _include_validation_launch(context):
    profile = LaunchConfiguration("stage4_profile").perform(context).strip().lower()
    if profile not in {"normal", "forced_stuck"}:
        raise RuntimeError(
            'Launch argument "stage4_profile" must be "normal" or "forced_stuck".'
        )

    # forced_stuck deliberately makes the detector permissive. It verifies the
    # supervisor state machine and logging; it is not a controller tuning profile.
    if profile == "forced_stuck":
        supervisor_args = {
            "tangent_escape_rmp_stuck_activation_threshold": "0.05",
            "tangent_escape_rmp_stuck_velocity_threshold": "10.0",
            "tangent_escape_rmp_stuck_progress_threshold": "10.0",
            "tangent_escape_rmp_stuck_time_threshold": "0.25",
            "tangent_escape_rmp_stuck_metric_boost": "1.10",
            "tangent_escape_rmp_stuck_accel_boost": "1.02",
            "tangent_escape_rmp_blocked_memory_update_duration": "0.70",
            "tangent_escape_rmp_blocked_memory_progress_threshold": "10.0",
            "tangent_escape_rmp_blocked_memory_clearance_improvement": "10.0",
        }
    else:
        supervisor_args = {
            "tangent_escape_rmp_stuck_activation_threshold": "0.5",
            "tangent_escape_rmp_stuck_velocity_threshold": "0.01",
            "tangent_escape_rmp_stuck_progress_threshold": "0.005",
            "tangent_escape_rmp_stuck_time_threshold": "0.6",
            "tangent_escape_rmp_stuck_metric_boost": "1.2",
            "tangent_escape_rmp_stuck_accel_boost": "1.05",
            "tangent_escape_rmp_blocked_memory_update_duration": "0.8",
            "tangent_escape_rmp_blocked_memory_progress_threshold": "0.008",
            "tangent_escape_rmp_blocked_memory_clearance_improvement": "0.01",
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
        "enable_tangent_escape_filter": "false",
        "enable_tangent_escape_rmp": "false",
        "tangent_escape_rmp_leaf_mode": LaunchConfiguration(
            "tangent_escape_rmp_leaf_mode"
        ),
        "tangent_escape_rmp_metric_scalar": LaunchConfiguration(
            "tangent_escape_rmp_metric_scalar"
        ),
        "tangent_escape_rmp_damping_gain": LaunchConfiguration(
            "tangent_escape_rmp_damping_gain"
        ),
        "tangent_escape_rmp_position_gain": LaunchConfiguration(
            "tangent_escape_rmp_position_gain"
        ),
        "tangent_escape_rmp_escape_length": LaunchConfiguration(
            "tangent_escape_rmp_escape_length"
        ),
        "tangent_escape_rmp_max_accel": LaunchConfiguration(
            "tangent_escape_rmp_max_accel"
        ),
        "tangent_escape_rmp_softmax_beta": LaunchConfiguration(
            "tangent_escape_rmp_softmax_beta"
        ),
        "tangent_escape_rmp_supervisor_enabled": "true",
        "tangent_escape_rmp_supervisor_dt": "0.01",
        "tangent_escape_rmp_branch_hold_weight": "0.5",
        "tangent_escape_rmp_branch_hold_duration": "0.6",
        "tangent_escape_rmp_branch_hold_max_adjacent_risk": "0.9",
        "tangent_escape_rmp_blocked_memory_penalty_weight": "2.0",
        "tangent_escape_rmp_blocked_memory_decay_time": "6.0",
        "tangent_escape_rmp_recovery_duration": "0.5",
        "publish_tangent_escape_rmp_data": "true",
        "publish_tangent_escape_filter_debug": "false",
        "publish_tangent_escape_geometry_debug": "true",
        "use_rmpflow_trace_logger": "true",
        "rmpflow_trace_log_rate": LaunchConfiguration("rmpflow_trace_log_rate"),
        "rmpflow_trace_log_directory": LaunchConfiguration(
            "rmpflow_trace_log_directory"
        ),
        **supervisor_args,
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
            "stage4_profile",
            default_value="normal",
            description=(
                "normal checks real detector behavior; forced_stuck deliberately forces "
                "stuck/memory transitions for state-machine validation."
            ),
        ),
        DeclareLaunchArgument("use_rviz", default_value="true"),
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
        DeclareLaunchArgument("tangent_escape_rmp_metric_scalar", default_value="150.0"),
        DeclareLaunchArgument("tangent_escape_rmp_damping_gain", default_value="4.0"),
        DeclareLaunchArgument("tangent_escape_rmp_position_gain", default_value="16.0"),
        DeclareLaunchArgument("tangent_escape_rmp_escape_length", default_value="0.06"),
        DeclareLaunchArgument("tangent_escape_rmp_max_accel", default_value="0.6"),
        DeclareLaunchArgument("tangent_escape_rmp_softmax_beta", default_value="4.0"),
        DeclareLaunchArgument(
            "tangent_escape_rmp_leaf_mode",
            default_value="stable_hybrid_gds",
        ),
        DeclareLaunchArgument("rmpflow_trace_log_rate", default_value="100.0"),
        DeclareLaunchArgument(
            "rmpflow_trace_log_directory",
            default_value=os.path.expanduser("~/ros2_ws/log/rmpflow_trace"),
        ),
        OpaqueFunction(function=_include_validation_launch),
    ])
