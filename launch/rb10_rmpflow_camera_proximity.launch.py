#!/usr/bin/env python3
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    rmp_pkg = get_package_share_directory("rb10_rmpflow_rviz")
    test_launch_path = os.path.join(rmp_pkg, "launch", "rb10_rmpflow_test.launch.py")
    camera_rviz_config_path = os.path.join(
        rmp_pkg,
        "config",
        "rb10_rmpflow_camera_obstacles.rviz",
    )

    use_camera_obstacles = LaunchConfiguration("use_camera_obstacles")
    camera_cloud_topic = LaunchConfiguration("camera_cloud_topic")
    camera_marker_topic = LaunchConfiguration("camera_marker_topic")
    camera_collision_output_topic = LaunchConfiguration("camera_collision_output_topic")
    camera_rviz_output_topic = LaunchConfiguration("camera_rviz_output_topic")
    camera_obstacle_output_frame = LaunchConfiguration("camera_obstacle_output_frame")
    camera_default_radius_m = LaunchConfiguration("camera_default_radius_m")
    camera_max_cloud_spheres = LaunchConfiguration("camera_max_cloud_spheres")
    camera_cloud_stride = LaunchConfiguration("camera_cloud_stride")
    camera_stale_timeout_sec = LaunchConfiguration("camera_stale_timeout_sec")
    proximity_raw_obstacle_topic = LaunchConfiguration("proximity_raw_obstacle_topic")

    base_test_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(test_launch_path),
        launch_arguments={
            "use_obstacles": "false",
            "use_proximity_bridge": "true",
            "proximity_surface_visualization": "false",
            "proximity_collision_obstacle_topic": proximity_raw_obstacle_topic,
            "proximity_visualization_obstacle_topic": camera_rviz_output_topic,
            "rviz_config": camera_rviz_config_path,
        }.items(),
    )

    camera_obstacle_bridge = Node(
        package="rb10_rmpflow_rviz",
        executable="camera_body_sphere_cloud_bridge.py",
        name="camera_body_sphere_cloud_bridge",
        output="screen",
        condition=IfCondition(use_camera_obstacles),
        parameters=[{
            "cloud_topic": camera_cloud_topic,
            "camera_marker_topic": camera_marker_topic,
            "additional_obstacle_topics": [proximity_raw_obstacle_topic],
            "collision_obstacle_topic": camera_collision_output_topic,
            "rviz_marker_topic": camera_rviz_output_topic,
            "output_frame": camera_obstacle_output_frame,
            "default_radius_m": ParameterValue(camera_default_radius_m, value_type=float),
            "max_cloud_spheres": ParameterValue(camera_max_cloud_spheres, value_type=int),
            "cloud_stride": ParameterValue(camera_cloud_stride, value_type=int),
            "stale_timeout_sec": ParameterValue(camera_stale_timeout_sec, value_type=float),
            "marker_namespace": "camera_obstacles",
            "relay_camera_markers_to_rviz": False,
            "subscribe_camera_marker_topic": False,
        }],
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            "use_camera_obstacles",
            default_value="true",
            description="Convert camera body sphere PointCloud2 into RMPflow collision obstacles.",
        ),
        DeclareLaunchArgument(
            "camera_cloud_topic",
            default_value="/rmp_camera/obstacle_body_sphere_cloud",
            description="PointCloud2 topic containing camera body obstacle sphere centers for RMPflow.",
        ),
        DeclareLaunchArgument(
            "camera_marker_topic",
            default_value="/rmp_camera/obstacle_body_spheres",
            description="MarkerArray topic from the camera pipeline used for RViz visualization.",
        ),
        DeclareLaunchArgument(
            "camera_collision_output_topic",
            default_value="/obstacles",
            description="Combined camera+proximity MarkerArray topic consumed by rmpflow_controller.",
        ),
        DeclareLaunchArgument(
            "camera_rviz_output_topic",
            default_value="/obstacle_markers",
            description="MarkerArray topic displayed by the existing RViz obstacle display.",
        ),
        DeclareLaunchArgument(
            "camera_obstacle_output_frame",
            default_value="base_link",
            description="Frame for camera obstacles before publishing to RMPflow; requires TF if different from the cloud frame.",
        ),
        DeclareLaunchArgument(
            "camera_default_radius_m",
            default_value="0.08",
            description="Fallback radius for each camera PointCloud2 sphere when no radius field exists.",
        ),
        DeclareLaunchArgument(
            "camera_max_cloud_spheres",
            default_value="128",
            description="Maximum number of camera sphere markers forwarded to RMPflow per cloud.",
        ),
        DeclareLaunchArgument(
            "camera_cloud_stride",
            default_value="1",
            description="Use every Nth PointCloud2 point when building camera obstacle spheres.",
        ),
        DeclareLaunchArgument(
            "camera_stale_timeout_sec",
            default_value="0.50",
            description="Drop camera/proximity obstacle messages older than this timeout.",
        ),
        DeclareLaunchArgument(
            "proximity_raw_obstacle_topic",
            default_value="/proximity_obstacles_raw",
            description="Intermediate proximity MarkerArray topic merged with camera obstacles.",
        ),
        base_test_launch,
        camera_obstacle_bridge,
    ])
