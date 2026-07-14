#!/usr/bin/env python3
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def _validate_params_file(context):
    params_path = LaunchConfiguration("params_file").perform(context)
    if not os.path.isfile(params_path):
        raise RuntimeError(
            f'Launch argument "params_file" must point to an existing YAML file: {params_path}'
        )
    return []


def _bool_config(name):
    value = LaunchConfiguration(name)
    return ParameterValue(
        PythonExpression(['True if "', value, '" == "true" else False']),
        value_type=bool,
    )


def generate_launch_description():
    rmp_pkg = get_package_share_directory("rb10_rmpflow_rviz")
    config_dir = os.path.join(rmp_pkg, "config")
    default_params_path = os.path.join(config_dir, "params.yaml")
    rviz_config_path = os.path.join(config_dir, "rb10_rmpflow.rviz")
    urdf_path = os.path.join(rmp_pkg, "urdf", "rb10_1300e.urdf")
    with open(urdf_path, "r", encoding="utf-8") as stream:
        robot_description = stream.read()
    fake_sensor_frames = [
        "tof6_1_L",
        "tof6_1_F",
        "tof6_1_R",
        "tof6_1_U",
        "tof_S",
        "tof_E",
        "tof_N",
        "tof_W",
        "tof3_1_S",
        "tof3_1_W",
        "tof3_1_N",
        "tof3_1_E",
        "tof2_1_E",
        "tof2_1_S",
        "tof2_1_W",
        "tof2_1_N",
        "tof2_E",
        "tof2_S",
        "tof2_W",
        "tof2_N",
    ]
    fake_topic_indices = [
        1,
        2,
        3,
        4,
        6,
        7,
        8,
        5,
        12,
        9,
        10,
        11,
        17,
        18,
        19,
        20,
        16,
        15,
        14,
        13,
    ]
    fake_bridge_range_topics = [
        f"/fake_proximity_distance{index}" for index in fake_topic_indices
    ]
    fake_raw_range_topics = [f"/fake_raw_distance{index}" for index in range(1, 21)]

    params_file = LaunchConfiguration("params_file")
    use_rviz = LaunchConfiguration("use_rviz")
    rviz_config = LaunchConfiguration("rviz_config")
    control_rate = LaunchConfiguration("control_rate")
    goal_x = LaunchConfiguration("goal_x")
    goal_y = LaunchConfiguration("goal_y")
    goal_z = LaunchConfiguration("goal_z")
    command_mode = LaunchConfiguration("command_mode")
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
    start_fake_proximity = LaunchConfiguration("start_fake_proximity")
    tangent_escape_rmp_leaf_mode = LaunchConfiguration("tangent_escape_rmp_leaf_mode")
    tangent_escape_rmp_metric_scalar = LaunchConfiguration("tangent_escape_rmp_metric_scalar")
    tangent_escape_rmp_damping_gain = LaunchConfiguration("tangent_escape_rmp_damping_gain")
    tangent_escape_rmp_position_gain = LaunchConfiguration("tangent_escape_rmp_position_gain")
    tangent_escape_rmp_escape_length = LaunchConfiguration("tangent_escape_rmp_escape_length")
    tangent_escape_rmp_collision_accel_scale = LaunchConfiguration(
        "tangent_escape_rmp_collision_accel_scale"
    )
    tangent_escape_rmp_max_accel = LaunchConfiguration("tangent_escape_rmp_max_accel")
    tangent_escape_rmp_softmax_beta = LaunchConfiguration("tangent_escape_rmp_softmax_beta")
    tangent_escape_rmp_supervisor_dt = LaunchConfiguration("tangent_escape_rmp_supervisor_dt")
    tangent_escape_rmp_branch_hold_weight = LaunchConfiguration(
        "tangent_escape_rmp_branch_hold_weight"
    )
    tangent_escape_rmp_branch_hold_duration = LaunchConfiguration(
        "tangent_escape_rmp_branch_hold_duration"
    )
    tangent_escape_rmp_branch_hold_max_adjacent_risk = LaunchConfiguration(
        "tangent_escape_rmp_branch_hold_max_adjacent_risk"
    )
    tangent_escape_rmp_stable_mode_normal_tolerance = LaunchConfiguration(
        "tangent_escape_rmp_stable_mode_normal_tolerance"
    )
    tangent_escape_rmp_stuck_activation_threshold = LaunchConfiguration(
        "tangent_escape_rmp_stuck_activation_threshold"
    )
    tangent_escape_rmp_stuck_velocity_threshold = LaunchConfiguration(
        "tangent_escape_rmp_stuck_velocity_threshold"
    )
    tangent_escape_rmp_stuck_progress_threshold = LaunchConfiguration(
        "tangent_escape_rmp_stuck_progress_threshold"
    )
    tangent_escape_rmp_stuck_time_threshold = LaunchConfiguration(
        "tangent_escape_rmp_stuck_time_threshold"
    )
    tangent_escape_rmp_stuck_metric_boost = LaunchConfiguration(
        "tangent_escape_rmp_stuck_metric_boost"
    )
    tangent_escape_rmp_stuck_accel_boost = LaunchConfiguration(
        "tangent_escape_rmp_stuck_accel_boost"
    )
    tangent_escape_rmp_blocked_memory_update_duration = LaunchConfiguration(
        "tangent_escape_rmp_blocked_memory_update_duration"
    )
    tangent_escape_rmp_blocked_memory_progress_threshold = LaunchConfiguration(
        "tangent_escape_rmp_blocked_memory_progress_threshold"
    )
    tangent_escape_rmp_blocked_memory_clearance_improvement = LaunchConfiguration(
        "tangent_escape_rmp_blocked_memory_clearance_improvement"
    )
    tangent_escape_rmp_blocked_memory_penalty_weight = LaunchConfiguration(
        "tangent_escape_rmp_blocked_memory_penalty_weight"
    )
    tangent_escape_rmp_blocked_memory_decay_time = LaunchConfiguration(
        "tangent_escape_rmp_blocked_memory_decay_time"
    )
    tangent_escape_rmp_recovery_duration = LaunchConfiguration(
        "tangent_escape_rmp_recovery_duration"
    )
    publish_tangent_escape_rmp_data = LaunchConfiguration("publish_tangent_escape_rmp_data")
    use_rmpflow_trace_logger = LaunchConfiguration("use_rmpflow_trace_logger")
    rmpflow_trace_log_rate = LaunchConfiguration("rmpflow_trace_log_rate")
    rmpflow_trace_log_directory = LaunchConfiguration("rmpflow_trace_log_directory")
    rmpflow_trace_console_summary = LaunchConfiguration("rmpflow_trace_console_summary")
    surface_patch_enabled = LaunchConfiguration("surface_patch_enabled")
    surface_patch_collision_memory_enabled = LaunchConfiguration(
        "surface_patch_collision_memory_enabled"
    )
    proximity_surface_visualization = LaunchConfiguration("proximity_surface_visualization")

    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="screen",
        parameters=[{
            "robot_description": robot_description,
            "publish_frequency": 50.0,
        }],
        remappings=[
            ("robot_description", "/rb10/robot_description"),
        ],
    )

    rmpflow_controller = Node(
        package="rb10_rmpflow_rviz",
        executable="rmpflow_controller",
        name="rmpflow_controller",
        output="screen",
        parameters=[
            params_file,
            {
                "backend_mode": "simulation",
                "command_mode": command_mode,
                "control_rate": ParameterValue(control_rate, value_type=float),
                "publish_joint_states": True,
                "joint_state_publish_topic": "/joint_states",
                "joint_state_topic": "/joint_states",
                "publish_position_command": True,
                "position_command_state_topic": "/rmp_position_command",
                "position_command_topic": "/position_controllers/commands",
                "publish_target_q": True,
                "target_q_topic": "/target_q",
                "publish_rmp_accel_debug": True,
                "rmp_joint_accel_topic": "/rmp_joint_accel",
                "rmp_tcp_accel_topic": "/rmp_tcp_accel",
                "initialize_goal_from_first_state": False,
                "goal_x": ParameterValue(goal_x, value_type=float),
                "goal_y": ParameterValue(goal_y, value_type=float),
                "goal_z": ParameterValue(goal_z, value_type=float),
                "safety_stop_on_min_z": False,
                "rmp_flag_gate_enabled": False,
                "enable_tangent_escape_filter": _bool_config("enable_tangent_escape_filter"),
                "enable_tangent_escape_rmp": _bool_config("enable_tangent_escape_rmp"),
                "tangent_escape_rmp_leaf_mode": tangent_escape_rmp_leaf_mode,
                "tangent_escape_rmp_metric_scalar": ParameterValue(
                    tangent_escape_rmp_metric_scalar,
                    value_type=float,
                ),
                "tangent_escape_rmp_damping_gain": ParameterValue(
                    tangent_escape_rmp_damping_gain,
                    value_type=float,
                ),
                "tangent_escape_rmp_position_gain": ParameterValue(
                    tangent_escape_rmp_position_gain,
                    value_type=float,
                ),
                "tangent_escape_rmp_escape_length": ParameterValue(
                    tangent_escape_rmp_escape_length,
                    value_type=float,
                ),
                "tangent_escape_rmp_collision_accel_scale": ParameterValue(
                    tangent_escape_rmp_collision_accel_scale,
                    value_type=float,
                ),
                "tangent_escape_rmp_max_accel": ParameterValue(
                    tangent_escape_rmp_max_accel,
                    value_type=float,
                ),
                "tangent_escape_rmp_softmax_beta": ParameterValue(
                    tangent_escape_rmp_softmax_beta,
                    value_type=float,
                ),
                "tangent_escape_rmp_supervisor_enabled": _bool_config(
                    "tangent_escape_rmp_supervisor_enabled"
                ),
                "tangent_escape_rmp_supervisor_dt": ParameterValue(
                    tangent_escape_rmp_supervisor_dt,
                    value_type=float,
                ),
                # The RMP supervisor reuses the historical filter parameter name.
                "tangent_escape_filter_branch_hold_weight": ParameterValue(
                    tangent_escape_rmp_branch_hold_weight,
                    value_type=float,
                ),
                "tangent_escape_rmp_branch_hold_duration": ParameterValue(
                    tangent_escape_rmp_branch_hold_duration,
                    value_type=float,
                ),
                "tangent_escape_rmp_branch_hold_max_adjacent_risk": ParameterValue(
                    tangent_escape_rmp_branch_hold_max_adjacent_risk,
                    value_type=float,
                ),
                "tangent_escape_rmp_stable_mode_normal_tolerance": ParameterValue(
                    tangent_escape_rmp_stable_mode_normal_tolerance,
                    value_type=float,
                ),
                "tangent_escape_rmp_stuck_activation_threshold": ParameterValue(
                    tangent_escape_rmp_stuck_activation_threshold,
                    value_type=float,
                ),
                "tangent_escape_rmp_stuck_velocity_threshold": ParameterValue(
                    tangent_escape_rmp_stuck_velocity_threshold,
                    value_type=float,
                ),
                "tangent_escape_rmp_stuck_progress_threshold": ParameterValue(
                    tangent_escape_rmp_stuck_progress_threshold,
                    value_type=float,
                ),
                "tangent_escape_rmp_stuck_time_threshold": ParameterValue(
                    tangent_escape_rmp_stuck_time_threshold,
                    value_type=float,
                ),
                "tangent_escape_rmp_stuck_metric_boost": ParameterValue(
                    tangent_escape_rmp_stuck_metric_boost,
                    value_type=float,
                ),
                "tangent_escape_rmp_stuck_accel_boost": ParameterValue(
                    tangent_escape_rmp_stuck_accel_boost,
                    value_type=float,
                ),
                "tangent_escape_rmp_blocked_memory_update_duration": ParameterValue(
                    tangent_escape_rmp_blocked_memory_update_duration,
                    value_type=float,
                ),
                "tangent_escape_rmp_blocked_memory_progress_threshold": ParameterValue(
                    tangent_escape_rmp_blocked_memory_progress_threshold,
                    value_type=float,
                ),
                "tangent_escape_rmp_blocked_memory_clearance_improvement": ParameterValue(
                    tangent_escape_rmp_blocked_memory_clearance_improvement,
                    value_type=float,
                ),
                "tangent_escape_rmp_blocked_memory_penalty_weight": ParameterValue(
                    tangent_escape_rmp_blocked_memory_penalty_weight,
                    value_type=float,
                ),
                "tangent_escape_rmp_blocked_memory_decay_time": ParameterValue(
                    tangent_escape_rmp_blocked_memory_decay_time,
                    value_type=float,
                ),
                "tangent_escape_rmp_recovery_duration": ParameterValue(
                    tangent_escape_rmp_recovery_duration,
                    value_type=float,
                ),
                "publish_tangent_escape_rmp_data": _bool_config(
                    "publish_tangent_escape_rmp_data"
                ),
                "publish_tangent_escape_filter_data": True,
                "publish_tangent_escape_filter_candidate_data": True,
                "publish_tangent_escape_filter_debug": _bool_config(
                    "publish_tangent_escape_filter_debug"
                ),
                "publish_tangent_escape_geometry_debug": _bool_config(
                    "publish_tangent_escape_geometry_debug"
                ),
                "tangent_escape_geometry_data_topic": "/tangent_escape_geometry_data",
                "tangent_escape_geometry_marker_topic": "tangent_escape_geometry_markers",
            },
        ],
    )

    fake_proximity = Node(
        package="rb10_rmpflow_rviz",
        executable="fake_proximity_scenario.py",
        name="fake_proximity_scenario",
        output="screen",
        condition=IfCondition(start_fake_proximity),
        parameters=[{
            "publish_rate_hz": ParameterValue(fake_publish_rate, value_type=float),
            "scenario": fake_scenario,
            "sensor_name": fake_sensor_name,
            "active_sensor_names_csv": fake_active_sensor_names,
            "range_m": ParameterValue(fake_range_m, value_type=float),
            "inactive_range_m": ParameterValue(fake_inactive_range_m, value_type=float),
            "approach_start_range_m": ParameterValue(fake_start_range_m, value_type=float),
            "approach_end_range_m": ParameterValue(fake_end_range_m, value_type=float),
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

    proximity_obstacle_bridge = Node(
        package="rb10_rmpflow_rviz",
        executable="proximity_obstacle_bridge",
        name="proximity_obstacle_bridge",
        output="screen",
        parameters=[
            params_file,
            {
                "publish_collision_obstacles": True,
                "obstacle_topic": "/obstacles",
                "visualization_obstacle_topic": "/obstacle_markers",
                "surface_patch_enabled": _bool_config("surface_patch_enabled"),
                "surface_patch_collision_memory_enabled": _bool_config(
                    "surface_patch_collision_memory_enabled"
                ),
                "surface_patch_fixed_visualization": _bool_config(
                    "proximity_surface_visualization"
                ),
                "range_topics": fake_bridge_range_topics,
                "sensor_frames": fake_sensor_frames,
                "sensor_enabled": [True] * 20,
                "trigger_distances": [0.55] * 20,
                "joint_state_topic": "/joint_states",
                "elbow_tof3_1_w_ignore_enabled": False,
            },
        ],
    )

    rmpflow_trace_logger = Node(
        package="rb10_rmpflow_rviz",
        executable="rmpflow_trace_logger.py",
        name="rmpflow_trace_logger",
        output="screen",
        condition=IfCondition(use_rmpflow_trace_logger),
        parameters=[{
            "log_rate_hz": ParameterValue(rmpflow_trace_log_rate, value_type=float),
            "console_summary": _bool_config("rmpflow_trace_console_summary"),
            "output_directory": rmpflow_trace_log_directory,
            "rmp_flag_topic": "/RMP_flag",
            "external_goal_topic": "/RMP_goal",
            "controller_goal_topic": "/goal_pose",
            "joint_state_topic": "/joint_states",
            "command_topic": "/position_controllers/commands",
            "target_q_topic": "/target_q",
            "target_metric_topic": "/target_metric",
            "debug_state_topic": "/rmp_debug_state",
            "rmp_ee_pose_topic": "/rmp_ee_pose",
            "rmp_joint_accel_topic": "/rmp_joint_accel",
            "rmp_tcp_accel_topic": "/rmp_tcp_accel",
            "tangent_escape_filter_data_topic": "/tangent_escape_filter_data",
            "tangent_escape_filter_candidate_data_topic": "/tangent_escape_filter_candidates",
            "tangent_escape_rmp_data_topic": "/tangent_escape_rmp_data",
            "tangent_escape_dual_solve_topic": "/tangent_escape_dual_solve",
            "tangent_escape_geometry_data_topic": "/tangent_escape_geometry_data",
            "obstacle_marker_topic": "/obstacles",
            "repulsion_metric_marker_topic": "/repulsion_metric_markers",
            "tcp_accel_marker_topic": "/tcp_accel_marker",
            "range_scale": 0.001,
            "minimum_hold_distance": 0.05,
            "trigger_distance": 0.29,
            "range_topics": fake_raw_range_topics,
        }],
    )

    rviz = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        condition=IfCondition(use_rviz),
        arguments=["-d", rviz_config],
    )

    return LaunchDescription([
        DeclareLaunchArgument("params_file", default_value=default_params_path),
        DeclareLaunchArgument("use_rviz", default_value="true"),
        DeclareLaunchArgument("rviz_config", default_value=rviz_config_path),
        DeclareLaunchArgument("control_rate", default_value="100.0"),
        DeclareLaunchArgument("command_mode", default_value="velocity"),
        DeclareLaunchArgument("goal_x", default_value="0.6"),
        DeclareLaunchArgument("goal_y", default_value="-0.4"),
        DeclareLaunchArgument("goal_z", default_value="0.6"),
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
        DeclareLaunchArgument("fake_range_m", default_value="0.05"),
        DeclareLaunchArgument("fake_start_range_m", default_value="0.50"),
        DeclareLaunchArgument("fake_end_range_m", default_value="0.50"),
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
        DeclareLaunchArgument(
            "start_fake_proximity",
            default_value="false",
            description="Start fake proximity publisher from this launch. Usually keep false and run it from another terminal.",
        ),
        DeclareLaunchArgument("enable_tangent_escape_filter", default_value="false"),
        DeclareLaunchArgument("enable_tangent_escape_rmp", default_value="false"),
        DeclareLaunchArgument("tangent_escape_rmp_leaf_mode", default_value="collision_scaled"),
        DeclareLaunchArgument("tangent_escape_rmp_metric_scalar", default_value="150.0"),
        DeclareLaunchArgument("tangent_escape_rmp_damping_gain", default_value="4.0"),
        DeclareLaunchArgument("tangent_escape_rmp_position_gain", default_value="16.0"),
        DeclareLaunchArgument("tangent_escape_rmp_escape_length", default_value="0.06"),
        DeclareLaunchArgument(
            "tangent_escape_rmp_collision_accel_scale",
            default_value="0.001",
        ),
        DeclareLaunchArgument("tangent_escape_rmp_max_accel", default_value="0.6"),
        DeclareLaunchArgument("tangent_escape_rmp_softmax_beta", default_value="4.0"),
        DeclareLaunchArgument("tangent_escape_rmp_supervisor_enabled", default_value="true"),
        DeclareLaunchArgument("tangent_escape_rmp_supervisor_dt", default_value="0.01"),
        DeclareLaunchArgument("tangent_escape_rmp_branch_hold_weight", default_value="0.5"),
        DeclareLaunchArgument("tangent_escape_rmp_branch_hold_duration", default_value="0.6"),
        DeclareLaunchArgument(
            "tangent_escape_rmp_branch_hold_max_adjacent_risk",
            default_value="0.9",
        ),
        DeclareLaunchArgument(
            "tangent_escape_rmp_stable_mode_normal_tolerance",
            default_value="0.20",
        ),
        DeclareLaunchArgument(
            "tangent_escape_rmp_stuck_activation_threshold",
            default_value="0.5",
        ),
        DeclareLaunchArgument(
            "tangent_escape_rmp_stuck_velocity_threshold",
            default_value="0.01",
        ),
        DeclareLaunchArgument(
            "tangent_escape_rmp_stuck_progress_threshold",
            default_value="0.005",
        ),
        DeclareLaunchArgument("tangent_escape_rmp_stuck_time_threshold", default_value="0.6"),
        DeclareLaunchArgument("tangent_escape_rmp_stuck_metric_boost", default_value="1.2"),
        DeclareLaunchArgument("tangent_escape_rmp_stuck_accel_boost", default_value="1.05"),
        DeclareLaunchArgument(
            "tangent_escape_rmp_blocked_memory_update_duration",
            default_value="0.8",
        ),
        DeclareLaunchArgument(
            "tangent_escape_rmp_blocked_memory_progress_threshold",
            default_value="0.008",
        ),
        DeclareLaunchArgument(
            "tangent_escape_rmp_blocked_memory_clearance_improvement",
            default_value="0.01",
        ),
        DeclareLaunchArgument(
            "tangent_escape_rmp_blocked_memory_penalty_weight",
            default_value="2.0",
        ),
        DeclareLaunchArgument(
            "tangent_escape_rmp_blocked_memory_decay_time",
            default_value="6.0",
        ),
        DeclareLaunchArgument("tangent_escape_rmp_recovery_duration", default_value="0.5"),
        DeclareLaunchArgument("publish_tangent_escape_rmp_data", default_value="true"),
        DeclareLaunchArgument("publish_tangent_escape_filter_debug", default_value="true"),
        DeclareLaunchArgument("publish_tangent_escape_geometry_debug", default_value="true"),
        DeclareLaunchArgument("surface_patch_enabled", default_value="false"),
        DeclareLaunchArgument("surface_patch_collision_memory_enabled", default_value="false"),
        DeclareLaunchArgument("proximity_surface_visualization", default_value="false"),
        DeclareLaunchArgument("use_rmpflow_trace_logger", default_value="true"),
        DeclareLaunchArgument("rmpflow_trace_log_rate", default_value="100.0"),
        DeclareLaunchArgument(
            "rmpflow_trace_log_directory",
            default_value=os.path.expanduser("~/ros2_ws/log/rmpflow_trace"),
        ),
        DeclareLaunchArgument("rmpflow_trace_console_summary", default_value="false"),
        OpaqueFunction(function=_validate_params_file),
        robot_state_publisher,
        rmpflow_controller,
        fake_proximity,
        proximity_obstacle_bridge,
        rmpflow_trace_logger,
        rviz,
    ])
