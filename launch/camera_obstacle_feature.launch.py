#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            "depth_topic",
            default_value="/camera/camera/aligned_depth_to_color/image_raw",
            description="Aligned depth image topic.",
        ),
        DeclareLaunchArgument(
            "camera_info_topic",
            default_value="/camera/camera/aligned_depth_to_color/camera_info",
            description="CameraInfo topic matching the aligned depth image.",
        ),
        DeclareLaunchArgument(
            "body_marker_topic",
            default_value="/camera/body_obstacle_markers",
            description="RViz MarkerArray topic for the camera-derived body obstacle.",
        ),
        DeclareLaunchArgument(
            "body_marker_shape",
            default_value="sphere",
            description='Body obstacle marker shape: "sphere" or "cylinder".',
        ),
        DeclareLaunchArgument(
            "publish_collision_obstacles",
            default_value="false",
            description="Publish the camera body obstacle into the RMPflow collision obstacle topic.",
        ),
        DeclareLaunchArgument(
            "collision_obstacle_topic",
            default_value="/obstacles",
            description="RMPflow collision obstacle MarkerArray topic.",
        ),
        DeclareLaunchArgument(
            "obstacle_output_frame",
            default_value="base_link",
            description="Frame used when publishing collision obstacles.",
        ),
        DeclareLaunchArgument(
            "min_depth_m",
            default_value="0.30",
            description="Ignore points closer than this distance.",
        ),
        DeclareLaunchArgument(
            "max_depth_m",
            default_value="1.50",
            description="Ignore points farther than this distance.",
        ),
        DeclareLaunchArgument(
            "sample_step",
            default_value="8",
            description="Pixel stride for depth search. Larger values are faster.",
        ),
        DeclareLaunchArgument(
            "cluster_depth_window_m",
            default_value="0.35",
            description="Depth band behind the nearest surface used to form the body obstacle.",
        ),
        DeclareLaunchArgument(
            "min_cluster_points",
            default_value="20",
            description="Minimum sampled depth points needed before publishing a body obstacle.",
        ),
        DeclareLaunchArgument(
            "body_min_radius_m",
            default_value="0.18",
            description="Minimum body obstacle cylinder radius.",
        ),
        DeclareLaunchArgument(
            "body_max_radius_m",
            default_value="0.45",
            description="Maximum body obstacle cylinder radius.",
        ),
        DeclareLaunchArgument(
            "body_min_height_m",
            default_value="0.50",
            description="Minimum body obstacle cylinder height.",
        ),
        DeclareLaunchArgument(
            "body_max_height_m",
            default_value="1.80",
            description="Maximum body obstacle cylinder height.",
        ),
        DeclareLaunchArgument(
            "roi_x_min",
            default_value="0.05",
            description="Normalized ROI left bound.",
        ),
        DeclareLaunchArgument(
            "roi_x_max",
            default_value="0.95",
            description="Normalized ROI right bound.",
        ),
        DeclareLaunchArgument(
            "roi_y_min",
            default_value="0.05",
            description="Normalized ROI top bound.",
        ),
        DeclareLaunchArgument(
            "roi_y_max",
            default_value="0.95",
            description="Normalized ROI bottom bound.",
        ),
        Node(
            package="rb10_rmpflow_rviz",
            executable="camera_obstacle_feature_node.py",
            name="camera_obstacle_feature_node",
            output="screen",
            parameters=[{
                "depth_topic": LaunchConfiguration("depth_topic"),
                "camera_info_topic": LaunchConfiguration("camera_info_topic"),
                "body_marker_topic": LaunchConfiguration("body_marker_topic"),
                "body_marker_shape": LaunchConfiguration("body_marker_shape"),
                "publish_collision_obstacles": LaunchConfiguration("publish_collision_obstacles"),
                "collision_obstacle_topic": LaunchConfiguration("collision_obstacle_topic"),
                "obstacle_output_frame": LaunchConfiguration("obstacle_output_frame"),
                "min_depth_m": LaunchConfiguration("min_depth_m"),
                "max_depth_m": LaunchConfiguration("max_depth_m"),
                "sample_step": LaunchConfiguration("sample_step"),
                "cluster_depth_window_m": LaunchConfiguration("cluster_depth_window_m"),
                "min_cluster_points": LaunchConfiguration("min_cluster_points"),
                "body_min_radius_m": LaunchConfiguration("body_min_radius_m"),
                "body_max_radius_m": LaunchConfiguration("body_max_radius_m"),
                "body_min_height_m": LaunchConfiguration("body_min_height_m"),
                "body_max_height_m": LaunchConfiguration("body_max_height_m"),
                "roi_x_min": LaunchConfiguration("roi_x_min"),
                "roi_x_max": LaunchConfiguration("roi_x_max"),
                "roi_y_min": LaunchConfiguration("roi_y_min"),
                "roi_y_max": LaunchConfiguration("roi_y_max"),
            }],
        ),
    ])
