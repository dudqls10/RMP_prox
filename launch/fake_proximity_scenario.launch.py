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
    fake_scenario = LaunchConfiguration("fake_scenario")
    fake_sensor_name = LaunchConfiguration("fake_sensor_name")
    fake_active_sensor_names = LaunchConfiguration("fake_active_sensor_names")
    fake_range_m = LaunchConfiguration("fake_range_m")
    fake_start_range_m = LaunchConfiguration("fake_start_range_m")
    fake_end_range_m = LaunchConfiguration("fake_end_range_m")
    fake_inactive_range_m = LaunchConfiguration("fake_inactive_range_m")
    fake_start_s = LaunchConfiguration("fake_start_s")
    fake_duration_s = LaunchConfiguration("fake_duration_s")
    fake_period_s = LaunchConfiguration("fake_period_s")
    fake_hold_s = LaunchConfiguration("fake_hold_s")
    fake_random_count = LaunchConfiguration("fake_random_count")
    fake_random_seed = LaunchConfiguration("fake_random_seed")
    fake_random_sensor_count = LaunchConfiguration("fake_random_sensor_count")
    fake_random_allow_repeats = LaunchConfiguration("fake_random_allow_repeats")
    fake_publish_rate = LaunchConfiguration("fake_publish_rate")

    fake_proximity = Node(
        package="rb10_rmpflow_rviz",
        executable="fake_proximity_scenario.py",
        name="fake_proximity_scenario",
        output="screen",
        parameters=[{
            "publish_rate_hz": ParameterValue(fake_publish_rate, value_type=float),
            "scenario": fake_scenario,
            "sensor_name": fake_sensor_name,
            "active_sensor_names_csv": fake_active_sensor_names,
            "range_m": ParameterValue(fake_range_m, value_type=float),
            "approach_start_range_m": ParameterValue(fake_start_range_m, value_type=float),
            "approach_end_range_m": ParameterValue(fake_end_range_m, value_type=float),
            "inactive_range_m": ParameterValue(fake_inactive_range_m, value_type=float),
            "start_s": ParameterValue(fake_start_s, value_type=float),
            "duration_s": ParameterValue(fake_duration_s, value_type=float),
            "period_s": ParameterValue(fake_period_s, value_type=float),
            "hold_s": ParameterValue(fake_hold_s, value_type=float),
            "random_count": ParameterValue(fake_random_count, value_type=int),
            "random_seed": ParameterValue(fake_random_seed, value_type=int),
            "random_sensor_count": ParameterValue(fake_random_sensor_count, value_type=int),
            "random_allow_repeats": _bool_config("fake_random_allow_repeats"),
            "range_scale": 0.001,
            "proximity_topic_prefix": "/fake_proximity_distance",
            "raw_topic_prefix": "/fake_raw_distance",
            "publish_raw_topics": True,
            "publish_rmp_flag": True,
            "rmp_flag_topic": "/RMP_flag",
        }],
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            "fake_scenario",
            default_value="approach_retreat",
            description=(
                "Fake ToF scenario: single, pulse, cycle, wall, approach, "
                "approach_retreat, random, random_pulse, off."
            ),
        ),
        DeclareLaunchArgument("fake_sensor_name", default_value="tof6_1_R"),
        DeclareLaunchArgument(
            "fake_active_sensor_names",
            default_value="",
            description="Comma-separated sensor names, or 'all'. Empty means fake_sensor_name.",
        ),
        DeclareLaunchArgument(
            "fake_range_m",
            default_value="0.05",
            description="Closest active range in meters.",
        ),
        DeclareLaunchArgument(
            "fake_start_range_m",
            default_value="0.50",
            description="Initial range for approach or approach_retreat in meters.",
        ),
        DeclareLaunchArgument(
            "fake_end_range_m",
            default_value="0.50",
            description="Final range for approach_retreat in meters.",
        ),
        DeclareLaunchArgument("fake_inactive_range_m", default_value="0.90"),
        DeclareLaunchArgument("fake_start_s", default_value="0.0"),
        DeclareLaunchArgument("fake_duration_s", default_value="10.0"),
        DeclareLaunchArgument("fake_period_s", default_value="10.0"),
        DeclareLaunchArgument("fake_hold_s", default_value="3.0"),
        DeclareLaunchArgument(
            "fake_random_count",
            default_value="5",
            description="Number of random fake sensor events when fake_scenario:=random.",
        ),
        DeclareLaunchArgument(
            "fake_random_seed",
            default_value="1",
            description="Random seed for fake_scenario:=random. Use -1 for time-based seed.",
        ),
        DeclareLaunchArgument(
            "fake_random_sensor_count",
            default_value="1",
            description="Number of sensors activated in each random event.",
        ),
        DeclareLaunchArgument(
            "fake_random_allow_repeats",
            default_value="false",
            description="Allow sensor names to repeat before the random sensor pool is exhausted.",
        ),
        DeclareLaunchArgument("fake_publish_rate", default_value="50.0"),
        fake_proximity,
    ])
