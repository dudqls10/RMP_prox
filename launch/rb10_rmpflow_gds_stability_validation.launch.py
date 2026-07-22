#!/usr/bin/env python3

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    package_share = get_package_share_directory("rb10_rmpflow_rviz")
    base_launch = os.path.join(
        package_share,
        "launch",
        "rb10_rmpflow_fake_proximity.launch.py",
    )

    include = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(base_launch),
        launch_arguments={
            "use_rviz": LaunchConfiguration("use_rviz"),
            "start_fake_proximity": LaunchConfiguration("start_fake_proximity"),
            "fake_scenario": LaunchConfiguration("fake_scenario"),
            "fake_sensor_name": LaunchConfiguration("fake_sensor_name"),
            "fake_range_m": LaunchConfiguration("fake_range_m"),
            "enable_tangent_escape_rmp": LaunchConfiguration(
                "enable_tangent_escape_rmp"
            ),
            "collision_policy": "paper_gds",
            "joint_limit_policy": "paper_gds",
            "escape_stability_guard_enabled": "true",
            "escape_stability_tank_capacity": LaunchConfiguration(
                "escape_stability_tank_capacity"
            ),
            "publish_stability_certificate_data": "true",
            "publish_tangent_escape_dual_solve_data": "true",
            "publish_debug_state": "true",
            "use_stability_certificate_visualizer": "true",
            "use_rmpflow_trace_logger": "true",
            # Use one physically consistent measured tangent state in the
            # continuous-time model certificate (qdot = dq/dt).
            "measured_position_feedback_blend": "1.0",
            "measured_velocity_feedback_blend": "1.0",
            "use_velocity_feedback_in_solver": "true",
            # These settings make the remaining built-in leaves fit the
            # constant-metric GDS proof profile.
            "root_solve_offset": "0.0",
            "target_rmp_min_metric_alpha": "1.0",
            "target_rmp_proximity_metric_boost_scalar": "1.0",
            "axis_target_rmp_proximity_metric_boost_scalar": "1.0",
            "damping_rmp_metric_scalar": "0.0",
            "joint_velocity_cap_enabled": "false",
            # Loosen configurable post-solve limits in this simulation-only
            # profile.  Unconditional joint/domain command guards still exist
            # downstream and are explicitly outside the solver certificate.
            "max_joint_accel": LaunchConfiguration("max_joint_accel"),
            "command_guard_max_step_rad": "10.0",
            "command_guard_max_velocity_rad_s": "100.0",
            "predictive_joint_limit_guard": "false",
        }.items(),
    )

    route_runner = TimerAction(
        period=3.0,
        actions=[
            Node(
                package="rb10_rmpflow_rviz",
                executable="experiment_goal_runner.py",
                name="gds_stability_route_runner",
                output="screen",
                condition=IfCondition(LaunchConfiguration("start_goal_runner")),
                arguments=[
                    "--trajectory",
                    "RMP_No_Plate",
                    "--no-record",
                    "--goal-timeout-sec",
                    LaunchConfiguration("goal_timeout_sec"),
                ],
            )
        ],
    )

    return LaunchDescription([
        DeclareLaunchArgument("use_rviz", default_value="true"),
        DeclareLaunchArgument("start_fake_proximity", default_value="true"),
        DeclareLaunchArgument("fake_scenario", default_value="single"),
        DeclareLaunchArgument("fake_sensor_name", default_value="tof_W"),
        DeclareLaunchArgument("fake_range_m", default_value="0.06"),
        DeclareLaunchArgument("enable_tangent_escape_rmp", default_value="true"),
        DeclareLaunchArgument("start_goal_runner", default_value="true"),
        DeclareLaunchArgument("goal_timeout_sec", default_value="8.0"),
        DeclareLaunchArgument(
            "max_joint_accel",
            default_value="1000000.0",
            description=(
                "Per-joint acceleration limit included in the solver "
                "certificate; use 10.0 for bounded-command validation"
            ),
        ),
        DeclareLaunchArgument(
            "escape_stability_tank_capacity",
            default_value="0.25",
            description="Finite Escape energy budget; the tank starts full.",
        ),
        include,
        route_runner,
    ])
