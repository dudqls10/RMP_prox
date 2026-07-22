#!/usr/bin/env python3

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    RegisterEventHandler,
    Shutdown,
    TimerAction,
)
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    package_share = get_package_share_directory("rb10_rmpflow_rviz")
    launch_directory = os.path.join(package_share, "launch")
    config_directory = os.path.join(package_share, "config")

    controller_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(launch_directory, "rb10_rmpflow_fake_proximity.launch.py")
        ),
        launch_arguments={
            "use_rviz": LaunchConfiguration("use_rviz"),
            "use_interactive_goal": "false",
            "use_obstacles": "false",
            "use_proximity_bridge": "true",
            "start_fake_proximity": "false",
            "initialize_goal_from_first_state": "true",
            "rmp_flag_gate_enabled": "false",
            "enable_tangent_escape_rmp": LaunchConfiguration(
                "enable_tangent_escape_rmp"
            ),
            "tangent_escape_rmp_metric_scalar": LaunchConfiguration(
                "tangent_escape_rmp_metric_scalar"
            ),
            "tangent_escape_rmp_prevent_weight": LaunchConfiguration(
                "tangent_escape_rmp_prevent_weight"
            ),
            "damping_rmp_accel_d_gain": LaunchConfiguration(
                "damping_rmp_accel_d_gain"
            ),
            "damping_rmp_metric_scalar": LaunchConfiguration(
                "damping_rmp_metric_scalar"
            ),
            "damping_rmp_inertia": LaunchConfiguration("damping_rmp_inertia"),
            "preserve_joint_accel_direction": LaunchConfiguration(
                "preserve_joint_accel_direction"
            ),
            "publish_tangent_escape_rmp_data": "true",
            "publish_tangent_escape_dual_solve_data": "true",
            "publish_leaf_ablation_data": "true",
            "publish_debug_state": "true",
            "publish_tangent_escape_geometry_debug": "true",
            "publish_stability_certificate_data": "false",
            "use_stability_certificate_visualizer": "false",
            "proximity_surface_visualization": "false",
            "use_rmpflow_trace_logger": "true",
            "rmpflow_trace_log_rate": LaunchConfiguration("trace_rate_hz"),
            "rmpflow_trace_log_directory": LaunchConfiguration("output_directory"),
            "rmpflow_trace_output_prefix": LaunchConfiguration("output_prefix"),
        }.items(),
    )

    proximity_replay = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(launch_directory, "fake_proximity_scenario_v2.launch.py")
        ),
        launch_arguments={
            "csv_path": LaunchConfiguration("proximity_csv"),
            "timestamp_column": "timestamp_unix",
            "distance_column_prefix": "proximity_distance",
            "input_unit": "meters",
            "playback_rate": "1.0",
            "publish_rate_hz": "100.0",
            "start_offset_s": "0.0",
            "duration_s": "0.0",
            "start_delay_s": "0.0",
            "start_trigger_topic": "/goal_pose",
            "loop": "true",
            "interpolate_ranges": "true",
            "inactive_range_m": "0.90",
        }.items(),
    )

    goal_runner = Node(
        package="rb10_rmpflow_rviz",
        executable="experiment_goal_runner.py",
        name="leaf_ablation_goal_runner",
        output="screen",
        arguments=[
            "--poses-file",
            os.path.join(config_directory, "experiment_goal_poses.yaml"),
            "--trajectory",
            LaunchConfiguration("trajectory"),
            "--fixed-goal-duration-sec",
            LaunchConfiguration("goal_interval_sec"),
            "--publish-rate",
            "20.0",
            "--no-record",
        ],
    )
    delayed_goal_runner = TimerAction(
        period=LaunchConfiguration("runner_start_delay_sec"),
        actions=[goal_runner],
    )
    shutdown_after_runner = RegisterEventHandler(
        OnProcessExit(
            target_action=goal_runner,
            on_exit=[Shutdown(reason="fixed leaf-ablation trajectory completed")],
        )
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("use_rviz", default_value="false"),
            DeclareLaunchArgument(
                "enable_tangent_escape_rmp",
                default_value="false",
            ),
            DeclareLaunchArgument(
                "tangent_escape_rmp_metric_scalar",
                default_value="50000.0",
            ),
            DeclareLaunchArgument(
                "tangent_escape_rmp_prevent_weight",
                default_value="0.5",
            ),
            DeclareLaunchArgument("damping_rmp_accel_d_gain", default_value="120.0"),
            DeclareLaunchArgument("damping_rmp_metric_scalar", default_value="0.0"),
            DeclareLaunchArgument("damping_rmp_inertia", default_value="300.0"),
            DeclareLaunchArgument(
                "preserve_joint_accel_direction",
                default_value="true",
            ),
            DeclareLaunchArgument(
                "proximity_csv",
                default_value=os.path.join(
                    config_directory,
                    "leaf_ablation_tof_w_proximity.csv",
                ),
            ),
            DeclareLaunchArgument(
                "trajectory",
                default_value="escape_tuning_validation",
            ),
            DeclareLaunchArgument("goal_interval_sec", default_value="4.0"),
            DeclareLaunchArgument("runner_start_delay_sec", default_value="3.0"),
            DeclareLaunchArgument("trace_rate_hz", default_value="100.0"),
            DeclareLaunchArgument(
                "output_directory",
                default_value=os.path.expanduser("~/ros2_ws/log/rmpflow_trace"),
            ),
            DeclareLaunchArgument(
                "output_prefix",
                default_value="leaf_ablation_fixed",
            ),
            controller_launch,
            proximity_replay,
            delayed_goal_runner,
            shutdown_after_runner,
        ]
    )
