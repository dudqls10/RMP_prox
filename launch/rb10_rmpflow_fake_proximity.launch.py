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
    use_interactive_goal = LaunchConfiguration("use_interactive_goal")
    use_obstacles = LaunchConfiguration("use_obstacles")
    use_proximity_bridge = LaunchConfiguration("use_proximity_bridge")
    initialize_goal_from_first_state = LaunchConfiguration(
        "initialize_goal_from_first_state"
    )
    rmp_flag_gate_enabled = LaunchConfiguration("rmp_flag_gate_enabled")
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
    collision_policy = LaunchConfiguration("collision_policy")
    joint_limit_policy = LaunchConfiguration("joint_limit_policy")
    escape_stability_tank_capacity = LaunchConfiguration(
        "escape_stability_tank_capacity"
    )
    root_solve_offset = LaunchConfiguration("root_solve_offset")
    max_joint_accel = LaunchConfiguration("max_joint_accel")
    preserve_joint_accel_direction = LaunchConfiguration(
        "preserve_joint_accel_direction"
    )
    target_rmp_min_metric_alpha = LaunchConfiguration(
        "target_rmp_min_metric_alpha"
    )
    target_rmp_proximity_metric_boost_scalar = LaunchConfiguration(
        "target_rmp_proximity_metric_boost_scalar"
    )
    axis_target_rmp_proximity_metric_boost_scalar = LaunchConfiguration(
        "axis_target_rmp_proximity_metric_boost_scalar"
    )
    damping_rmp_metric_scalar = LaunchConfiguration("damping_rmp_metric_scalar")
    damping_rmp_accel_d_gain = LaunchConfiguration("damping_rmp_accel_d_gain")
    damping_rmp_inertia = LaunchConfiguration("damping_rmp_inertia")
    measured_position_feedback_blend = LaunchConfiguration(
        "measured_position_feedback_blend"
    )
    measured_velocity_feedback_blend = LaunchConfiguration(
        "measured_velocity_feedback_blend"
    )
    command_guard_max_step_rad = LaunchConfiguration("command_guard_max_step_rad")
    command_guard_max_velocity_rad_s = LaunchConfiguration(
        "command_guard_max_velocity_rad_s"
    )
    tangent_escape_rmp_max_accel = LaunchConfiguration("tangent_escape_rmp_max_accel")
    tangent_escape_rmp_acceleration_model = LaunchConfiguration(
        "tangent_escape_rmp_acceleration_model"
    )
    tangent_escape_rmp_metric_scalar = LaunchConfiguration(
        "tangent_escape_rmp_metric_scalar"
    )
    tangent_escape_rmp_clearance_margin = LaunchConfiguration(
        "tangent_escape_rmp_clearance_margin"
    )
    tangent_escape_rmp_influence_distance = LaunchConfiguration(
        "tangent_escape_rmp_influence_distance"
    )
    tangent_escape_rmp_velocity_gain = LaunchConfiguration(
        "tangent_escape_rmp_velocity_gain"
    )
    tangent_escape_rmp_prevent_weight = LaunchConfiguration(
        "tangent_escape_rmp_prevent_weight"
    )
    tangent_escape_rmp_prevent_speed = LaunchConfiguration(
        "tangent_escape_rmp_prevent_speed"
    )
    tangent_escape_rmp_recovery_speed = LaunchConfiguration(
        "tangent_escape_rmp_recovery_speed"
    )
    tangent_escape_rmp_max_speed = LaunchConfiguration(
        "tangent_escape_rmp_max_speed"
    )
    tangent_escape_rmp_desired_velocity_time_constant = LaunchConfiguration(
        "tangent_escape_rmp_desired_velocity_time_constant"
    )
    tangent_escape_rmp_drive_ramp_duration = LaunchConfiguration(
        "tangent_escape_rmp_drive_ramp_duration"
    )
    tangent_escape_rmp_handoff_duration = LaunchConfiguration(
        "tangent_escape_rmp_handoff_duration"
    )
    tangent_escape_rmp_release_stop_speed = LaunchConfiguration(
        "tangent_escape_rmp_release_stop_speed"
    )
    tangent_escape_rmp_release_hold_speed = LaunchConfiguration(
        "tangent_escape_rmp_release_hold_speed"
    )
    tangent_escape_rmp_progress_low_threshold = LaunchConfiguration(
        "tangent_escape_rmp_progress_low_threshold"
    )
    tangent_escape_rmp_progress_ok_threshold = LaunchConfiguration(
        "tangent_escape_rmp_progress_ok_threshold"
    )
    tangent_escape_rmp_still_speed_threshold = LaunchConfiguration(
        "tangent_escape_rmp_still_speed_threshold"
    )
    tangent_escape_rmp_moving_speed_threshold = LaunchConfiguration(
        "tangent_escape_rmp_moving_speed_threshold"
    )
    tangent_escape_rmp_intent_on_speed = LaunchConfiguration(
        "tangent_escape_rmp_intent_on_speed"
    )
    tangent_escape_rmp_intent_full_speed = LaunchConfiguration(
        "tangent_escape_rmp_intent_full_speed"
    )
    tangent_escape_rmp_minimum_drive_duration = LaunchConfiguration(
        "tangent_escape_rmp_minimum_drive_duration"
    )
    tangent_escape_rmp_command_test_distance = LaunchConfiguration(
        "tangent_escape_rmp_command_test_distance"
    )
    tangent_escape_rmp_minimum_move_ratio = LaunchConfiguration(
        "tangent_escape_rmp_minimum_move_ratio"
    )
    tangent_escape_rmp_normal_tolerance = LaunchConfiguration(
        "tangent_escape_rmp_normal_tolerance"
    )
    tangent_escape_rmp_blocked_memory_penalty_weight = LaunchConfiguration(
        "tangent_escape_rmp_blocked_memory_penalty_weight"
    )
    publish_tangent_escape_rmp_data = LaunchConfiguration("publish_tangent_escape_rmp_data")
    publish_tangent_escape_dual_solve_data = LaunchConfiguration(
        "publish_tangent_escape_dual_solve_data"
    )
    publish_debug_state = LaunchConfiguration("publish_debug_state")
    use_rmpflow_trace_logger = LaunchConfiguration("use_rmpflow_trace_logger")
    rmpflow_trace_log_rate = LaunchConfiguration("rmpflow_trace_log_rate")
    rmpflow_trace_log_directory = LaunchConfiguration("rmpflow_trace_log_directory")
    rmpflow_trace_output_prefix = LaunchConfiguration("rmpflow_trace_output_prefix")
    rmpflow_trace_console_summary = LaunchConfiguration("rmpflow_trace_console_summary")
    proximity_surface_visualization = LaunchConfiguration("proximity_surface_visualization")
    use_stability_certificate_visualizer = LaunchConfiguration(
        "use_stability_certificate_visualizer"
    )

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
                "root_solve_offset": ParameterValue(
                    root_solve_offset,
                    value_type=float,
                ),
                "max_joint_accel": ParameterValue(
                    max_joint_accel,
                    value_type=float,
                ),
                "preserve_joint_accel_direction": _bool_config(
                    "preserve_joint_accel_direction"
                ),
                "target_rmp_min_metric_alpha": ParameterValue(
                    target_rmp_min_metric_alpha,
                    value_type=float,
                ),
                "target_rmp_proximity_metric_boost_scalar": ParameterValue(
                    target_rmp_proximity_metric_boost_scalar,
                    value_type=float,
                ),
                "axis_target_rmp_proximity_metric_boost_scalar": ParameterValue(
                    axis_target_rmp_proximity_metric_boost_scalar,
                    value_type=float,
                ),
                "damping_rmp_metric_scalar": ParameterValue(
                    damping_rmp_metric_scalar,
                    value_type=float,
                ),
                "damping_rmp_accel_d_gain": ParameterValue(
                    damping_rmp_accel_d_gain,
                    value_type=float,
                ),
                "damping_rmp_inertia": ParameterValue(
                    damping_rmp_inertia,
                    value_type=float,
                ),
                "measured_position_feedback_blend": ParameterValue(
                    measured_position_feedback_blend,
                    value_type=float,
                ),
                "measured_velocity_feedback_blend": ParameterValue(
                    measured_velocity_feedback_blend,
                    value_type=float,
                ),
                "use_velocity_feedback_in_solver": _bool_config(
                    "use_velocity_feedback_in_solver"
                ),
                "command_guard_max_step_rad": ParameterValue(
                    command_guard_max_step_rad,
                    value_type=float,
                ),
                "command_guard_max_velocity_rad_s": ParameterValue(
                    command_guard_max_velocity_rad_s,
                    value_type=float,
                ),
                "predictive_joint_limit_guard": _bool_config(
                    "predictive_joint_limit_guard"
                ),
                "graph.joint_velocity_cap.enabled": _bool_config(
                    "joint_velocity_cap_enabled"
                ),
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
                "initialize_goal_from_first_state": ParameterValue(
                    initialize_goal_from_first_state,
                    value_type=bool,
                ),
                "goal_x": ParameterValue(goal_x, value_type=float),
                "goal_y": ParameterValue(goal_y, value_type=float),
                "goal_z": ParameterValue(goal_z, value_type=float),
                "safety_stop_on_min_z": False,
                "rmp_flag_gate_enabled": ParameterValue(
                    rmp_flag_gate_enabled,
                    value_type=bool,
                ),
                "enable_tangent_escape_rmp": _bool_config("enable_tangent_escape_rmp"),
                "tangent_escape_rmp_acceleration_model": tangent_escape_rmp_acceleration_model,
                "tangent_escape_rmp_metric_scalar": ParameterValue(
                    tangent_escape_rmp_metric_scalar,
                    value_type=float,
                ),
                "tangent_escape_rmp_clearance_margin": ParameterValue(
                    tangent_escape_rmp_clearance_margin,
                    value_type=float,
                ),
                "collision_policy": collision_policy,
                "joint_limit_policy": joint_limit_policy,
                "escape_stability_guard_enabled": _bool_config(
                    "escape_stability_guard_enabled"
                ),
                "escape_stability_tank_capacity": ParameterValue(
                    escape_stability_tank_capacity,
                    value_type=float,
                ),
                "publish_stability_certificate_data": _bool_config(
                    "publish_stability_certificate_data"
                ),
                "tangent_escape_rmp_max_accel": ParameterValue(
                    tangent_escape_rmp_max_accel,
                    value_type=float,
                ),
                "tangent_escape_rmp_influence_distance": ParameterValue(
                    tangent_escape_rmp_influence_distance,
                    value_type=float,
                ),
                "tangent_escape_rmp_velocity_gain": ParameterValue(
                    tangent_escape_rmp_velocity_gain,
                    value_type=float,
                ),
                "tangent_escape_rmp_prevent_weight": ParameterValue(
                    tangent_escape_rmp_prevent_weight,
                    value_type=float,
                ),
                "tangent_escape_rmp_prevent_speed": ParameterValue(
                    tangent_escape_rmp_prevent_speed,
                    value_type=float,
                ),
                "tangent_escape_rmp_recovery_speed": ParameterValue(
                    tangent_escape_rmp_recovery_speed,
                    value_type=float,
                ),
                "tangent_escape_rmp_max_speed": ParameterValue(
                    tangent_escape_rmp_max_speed,
                    value_type=float,
                ),
                "tangent_escape_rmp_desired_velocity_time_constant": ParameterValue(
                    tangent_escape_rmp_desired_velocity_time_constant,
                    value_type=float,
                ),
                "tangent_escape_rmp_drive_ramp_duration": ParameterValue(
                    tangent_escape_rmp_drive_ramp_duration,
                    value_type=float,
                ),
                "tangent_escape_rmp_handoff_duration": ParameterValue(
                    tangent_escape_rmp_handoff_duration,
                    value_type=float,
                ),
                "tangent_escape_rmp_release_stop_speed": ParameterValue(
                    tangent_escape_rmp_release_stop_speed,
                    value_type=float,
                ),
                "tangent_escape_rmp_release_hold_speed": ParameterValue(
                    tangent_escape_rmp_release_hold_speed,
                    value_type=float,
                ),
                "tangent_escape_rmp_progress_low_threshold": ParameterValue(
                    tangent_escape_rmp_progress_low_threshold,
                    value_type=float,
                ),
                "tangent_escape_rmp_progress_ok_threshold": ParameterValue(
                    tangent_escape_rmp_progress_ok_threshold,
                    value_type=float,
                ),
                "tangent_escape_rmp_still_speed_threshold": ParameterValue(
                    tangent_escape_rmp_still_speed_threshold,
                    value_type=float,
                ),
                "tangent_escape_rmp_moving_speed_threshold": ParameterValue(
                    tangent_escape_rmp_moving_speed_threshold,
                    value_type=float,
                ),
                "tangent_escape_rmp_intent_on_speed": ParameterValue(
                    tangent_escape_rmp_intent_on_speed,
                    value_type=float,
                ),
                "tangent_escape_rmp_intent_full_speed": ParameterValue(
                    tangent_escape_rmp_intent_full_speed,
                    value_type=float,
                ),
                "tangent_escape_rmp_minimum_drive_duration": ParameterValue(
                    tangent_escape_rmp_minimum_drive_duration,
                    value_type=float,
                ),
                "tangent_escape_rmp_command_test_distance": ParameterValue(
                    tangent_escape_rmp_command_test_distance,
                    value_type=float,
                ),
                "tangent_escape_rmp_minimum_move_ratio": ParameterValue(
                    tangent_escape_rmp_minimum_move_ratio,
                    value_type=float,
                ),
                "tangent_escape_rmp_normal_tolerance": ParameterValue(
                    tangent_escape_rmp_normal_tolerance,
                    value_type=float,
                ),
                "tangent_escape_rmp_blocked_memory_penalty_weight": ParameterValue(
                    tangent_escape_rmp_blocked_memory_penalty_weight,
                    value_type=float,
                ),
                "publish_tangent_escape_rmp_data": _bool_config(
                    "publish_tangent_escape_rmp_data"
                ),
                "publish_tangent_escape_dual_solve_data": _bool_config(
                    "publish_tangent_escape_dual_solve_data"
                ),
                "tangent_escape_dual_solve_topic": "/tangent_escape_dual_solve",
                "publish_leaf_ablation_data": _bool_config(
                    "publish_leaf_ablation_data"
                ),
                "leaf_ablation_topic": "/rmp_leaf_ablation",
                "publish_debug_state": _bool_config("publish_debug_state"),
                "publish_tangent_escape_geometry_debug": _bool_config(
                    "publish_tangent_escape_geometry_debug"
                ),
                "tangent_escape_geometry_data_topic": "/tangent_escape_geometry_data",
                "tangent_escape_geometry_marker_topic": "tangent_escape_geometry_markers",
            },
        ],
    )

    interactive_goal = Node(
        package="rb10_rmpflow_rviz",
        executable="interactive_goal",
        name="interactive_goal",
        output="screen",
        parameters=[
            params_file,
            {
                "joint_state_topic": "/joint_states",
                "lock_orientation_to_tcp": False,
            },
        ],
        condition=IfCondition(use_interactive_goal),
    )

    obstacle_manager = Node(
        package="rb10_rmpflow_rviz",
        executable="obstacle_manager",
        name="obstacle_manager",
        output="screen",
        parameters=[params_file],
        condition=IfCondition(PythonExpression([
            '"', use_obstacles, '" == "true" and "',
            use_proximity_bridge, '" != "true"',
        ])),
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
        condition=IfCondition(use_proximity_bridge),
        parameters=[
            params_file,
            {
                "publish_collision_obstacles": True,
                "obstacle_topic": "/obstacles",
                "visualization_obstacle_topic": "/obstacle_markers",
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
            "output_prefix": rmpflow_trace_output_prefix,
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
            "tangent_escape_rmp_data_topic": "/tangent_escape_rmp_data",
            "tangent_escape_dual_solve_topic": "/tangent_escape_dual_solve",
            "leaf_ablation_topic": "/rmp_leaf_ablation",
            "tangent_escape_geometry_data_topic": "/tangent_escape_geometry_data",
            "stability_certificate_data_topic": "/rmp_stability_certificate",
            "obstacle_marker_topic": "/obstacles",
            "repulsion_metric_marker_topic": "/repulsion_metric_markers",
            "tcp_accel_marker_topic": "/tcp_accel_marker",
            "range_scale": 0.001,
            "minimum_hold_distance": 0.05,
            "trigger_distance": 0.29,
            "range_topics": fake_raw_range_topics,
        }],
    )

    stability_certificate_visualizer = Node(
        package="rb10_rmpflow_rviz",
        executable="stability_certificate_visualizer.py",
        name="stability_certificate_visualizer",
        output="screen",
        condition=IfCondition(use_stability_certificate_visualizer),
        parameters=[{
            "certificate_topic": "/rmp_stability_certificate",
            "marker_topic": "/rmp_stability_certificate_marker",
            "frame_id": "base_link",
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
        DeclareLaunchArgument(
            "use_interactive_goal",
            default_value="false",
            description="Start the RViz interactive goal publisher.",
        ),
        DeclareLaunchArgument(
            "use_obstacles",
            default_value="false",
            description=(
                "Start the interactive cylindrical obstacle manager when the "
                "proximity bridge is disabled."
            ),
        ),
        DeclareLaunchArgument(
            "use_proximity_bridge",
            default_value="true",
            description=(
                "Convert fake proximity ranges into /obstacles. Set false to "
                "use the interactive cylindrical obstacle manager instead."
            ),
        ),
        DeclareLaunchArgument(
            "initialize_goal_from_first_state",
            default_value="false",
            description=(
                "Hold the controller at the initial simulated pose until an "
                "interactive goal is supplied."
            ),
        ),
        DeclareLaunchArgument(
            "rmp_flag_gate_enabled",
            default_value="false",
            description=(
                "Hold the simulation state while /RMP_flag is inactive. Enable "
                "this for repeatable tuning trials that start from initial_q."
            ),
        ),
        DeclareLaunchArgument("control_rate", default_value="200.0"),
        DeclareLaunchArgument("root_solve_offset", default_value="0.001"),
        DeclareLaunchArgument("max_joint_accel", default_value="10.0"),
        DeclareLaunchArgument(
            "preserve_joint_accel_direction",
            default_value="true",
            description=(
                "Scale the complete qdd vector when any joint exceeds the "
                "limit; false restores independent per-joint clipping."
            ),
        ),
        DeclareLaunchArgument(
            "target_rmp_min_metric_alpha",
            default_value="0.1",
        ),
        DeclareLaunchArgument(
            "target_rmp_proximity_metric_boost_scalar",
            default_value="2.0",
        ),
        DeclareLaunchArgument(
            "axis_target_rmp_proximity_metric_boost_scalar",
            default_value="10.0",
        ),
        DeclareLaunchArgument(
            "damping_rmp_metric_scalar",
            default_value="0.0",
        ),
        DeclareLaunchArgument(
            "damping_rmp_accel_d_gain",
            default_value="120.0",
        ),
        DeclareLaunchArgument(
            "damping_rmp_inertia",
            default_value="300.0",
        ),
        DeclareLaunchArgument(
            "joint_velocity_cap_enabled",
            default_value="true",
        ),
        DeclareLaunchArgument(
            "command_guard_max_step_rad",
            default_value="0.00436332313",
        ),
        DeclareLaunchArgument(
            "command_guard_max_velocity_rad_s",
            default_value="1.0",
        ),
        DeclareLaunchArgument(
            "predictive_joint_limit_guard",
            default_value="true",
        ),
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
        DeclareLaunchArgument("enable_tangent_escape_rmp", default_value="true"),
        DeclareLaunchArgument(
            "tangent_escape_rmp_metric_scalar",
            default_value="50000.0",
            description="Independent Tangent Escape metric scalar.",
        ),
        DeclareLaunchArgument(
            "tangent_escape_rmp_clearance_margin",
            default_value="0.0",
            description="Independent Tangent Escape clearance margin in metres.",
        ),
        DeclareLaunchArgument(
            "collision_policy",
            default_value="lula_canonical",
            description="lula_canonical preserves the production formula; paper_gds enables the structured-GDS leaf.",
        ),
        DeclareLaunchArgument(
            "joint_limit_policy",
            default_value="lula_canonical",
            description="lula_canonical preserves the production formula; paper_gds enables the logit structured-GDS leaf.",
        ),
        DeclareLaunchArgument(
            "measured_position_feedback_blend",
            default_value="1.0",
        ),
        DeclareLaunchArgument(
            "measured_velocity_feedback_blend",
            default_value="0.6",
        ),
        DeclareLaunchArgument(
            "use_velocity_feedback_in_solver",
            default_value="true",
        ),
        DeclareLaunchArgument(
            "escape_stability_guard_enabled",
            default_value="false",
            description="Limit positive Escape interconnection energy with a finite tank.",
        ),
        DeclareLaunchArgument(
            "escape_stability_tank_capacity",
            default_value="0.25",
            description="Finite Escape energy budget; the tank starts full.",
        ),
        DeclareLaunchArgument(
            "publish_stability_certificate_data",
            default_value="true",
        ),
        DeclareLaunchArgument(
            "use_stability_certificate_visualizer",
            default_value="true",
        ),
        DeclareLaunchArgument(
            "tangent_escape_rmp_acceleration_model",
            default_value="risk_damped",
            description=(
                "Select risk_damped for production experiments; "
                "canonical_velocity is retained only for explicit A/B rollback."
            ),
        ),
        DeclareLaunchArgument(
            "tangent_escape_rmp_max_accel",
            default_value="0.6",
            description=(
                "Escape task-space acceleration saturation. risk_damped "
                "requires a positive value; max_joint_accel independently "
                "limits the resolved joint acceleration."
            ),
        ),
        DeclareLaunchArgument(
            "tangent_escape_rmp_influence_distance",
            default_value="0.24",
        ),
        DeclareLaunchArgument("tangent_escape_rmp_velocity_gain", default_value="20.0"),
        DeclareLaunchArgument("tangent_escape_rmp_prevent_weight", default_value="0.5"),
        DeclareLaunchArgument("tangent_escape_rmp_prevent_speed", default_value="0.018"),
        DeclareLaunchArgument("tangent_escape_rmp_recovery_speed", default_value="0.05"),
        DeclareLaunchArgument("tangent_escape_rmp_max_speed", default_value="0.05"),
        DeclareLaunchArgument(
            "tangent_escape_rmp_desired_velocity_time_constant",
            default_value="0.22",
        ),
        DeclareLaunchArgument(
            "tangent_escape_rmp_drive_ramp_duration",
            default_value="0.25",
        ),
        DeclareLaunchArgument(
            "tangent_escape_rmp_handoff_duration",
            default_value="0.25",
        ),
        DeclareLaunchArgument(
            "tangent_escape_rmp_release_stop_speed",
            default_value="0.01",
        ),
        DeclareLaunchArgument(
            "tangent_escape_rmp_release_hold_speed",
            default_value="0.06",
        ),
        DeclareLaunchArgument(
            "tangent_escape_rmp_progress_low_threshold",
            default_value="0.001",
        ),
        DeclareLaunchArgument(
            "tangent_escape_rmp_progress_ok_threshold",
            default_value="0.02",
        ),
        DeclareLaunchArgument(
            "tangent_escape_rmp_still_speed_threshold",
            default_value="0.01",
        ),
        DeclareLaunchArgument(
            "tangent_escape_rmp_moving_speed_threshold",
            default_value="0.08",
        ),
        DeclareLaunchArgument("tangent_escape_rmp_intent_on_speed", default_value="0.005"),
        DeclareLaunchArgument("tangent_escape_rmp_intent_full_speed", default_value="0.03"),
        DeclareLaunchArgument(
            "tangent_escape_rmp_minimum_drive_duration",
            default_value="0.8",
        ),
        DeclareLaunchArgument(
            "tangent_escape_rmp_command_test_distance",
            default_value="0.002",
        ),
        DeclareLaunchArgument(
            "tangent_escape_rmp_minimum_move_ratio",
            default_value="0.30",
        ),
        DeclareLaunchArgument(
            "tangent_escape_rmp_normal_tolerance",
            default_value="0.20",
        ),
        DeclareLaunchArgument(
            "tangent_escape_rmp_blocked_memory_penalty_weight",
            default_value="2.0",
        ),
        DeclareLaunchArgument("publish_tangent_escape_rmp_data", default_value="true"),
        DeclareLaunchArgument(
            "publish_tangent_escape_dual_solve_data",
            default_value="false",
            description="Publish same-state Escape ON/OFF counterfactual acceleration data.",
        ),
        DeclareLaunchArgument(
            "publish_leaf_ablation_data",
            default_value="true",
            description="Publish frozen-policy per-leaf removal diagnostics in fake mode.",
        ),
        DeclareLaunchArgument("publish_debug_state", default_value="false"),
        DeclareLaunchArgument("publish_tangent_escape_geometry_debug", default_value="true"),
        DeclareLaunchArgument("proximity_surface_visualization", default_value="false"),
        DeclareLaunchArgument("use_rmpflow_trace_logger", default_value="true"),
        DeclareLaunchArgument("rmpflow_trace_log_rate", default_value="100.0"),
        DeclareLaunchArgument(
            "rmpflow_trace_log_directory",
            default_value=os.path.expanduser("~/ros2_ws/log/rmpflow_trace"),
        ),
        DeclareLaunchArgument("rmpflow_trace_output_prefix", default_value="rmpflow_trace"),
        DeclareLaunchArgument("rmpflow_trace_console_summary", default_value="false"),
        OpaqueFunction(function=_validate_params_file),
        robot_state_publisher,
        rmpflow_controller,
        interactive_goal,
        obstacle_manager,
        fake_proximity,
        proximity_obstacle_bridge,
        rmpflow_trace_logger,
        stability_certificate_visualizer,
        rviz,
    ])
