#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            "color_topic",
            default_value="/camera/camera/color/image_raw",
            description="RGB image topic used by MediaPipe Pose.",
        ),
        DeclareLaunchArgument(
            "depth_topic",
            default_value="/camera/camera/aligned_depth_to_color/image_raw",
            description="Aligned depth image topic used to lift pose landmarks to 3D.",
        ),
        DeclareLaunchArgument(
            "camera_info_topic",
            default_value="/camera/camera/aligned_depth_to_color/camera_info",
            description="CameraInfo topic matching the aligned depth image.",
        ),
        DeclareLaunchArgument(
            "marker_topic",
            default_value="/camera/human_pose_obstacles",
            description="RViz MarkerArray topic for MediaPipe joint and limb obstacles.",
        ),
        DeclareLaunchArgument(
            "visibility_threshold",
            default_value="0.65",
            description="Minimum MediaPipe landmark visibility before using a joint.",
        ),
        DeclareLaunchArgument(
            "landmark_set",
            default_value="major",
            description='Landmarks to publish: "major" uses shoulders/elbows/wrists/hips/knees/ankles; "full" uses all MediaPipe pose landmarks.',
        ),
        DeclareLaunchArgument(
            "min_valid_joints",
            default_value="0",
            description="Minimum valid 3D joints required before publishing pose obstacles.",
        ),
        DeclareLaunchArgument(
            "min_core_joints",
            default_value="1",
            description="Minimum visible core joints among shoulders/hips before publishing pose obstacles.",
        ),
        DeclareLaunchArgument(
            "max_joint_depth_deviation_m",
            default_value="0.35",
            description="Reject valid joints that are this far from the median joint depth. Set 0 to disable.",
        ),
        DeclareLaunchArgument(
            "min_depth_m",
            default_value="0.30",
            description="Ignore depth values closer than this distance.",
        ),
        DeclareLaunchArgument(
            "max_depth_m",
            default_value="2.50",
            description="Ignore depth values farther than this distance.",
        ),
        DeclareLaunchArgument(
            "joint_radius_m",
            default_value="0.08",
            description="Sphere radius for visible pose joints.",
        ),
        DeclareLaunchArgument(
            "limb_radius_m",
            default_value="0.10",
            description="Sphere radius for limb samples between visible joints.",
        ),
        DeclareLaunchArgument(
            "limb_spacing_m",
            default_value="0.16",
            description="Spacing between limb spheres.",
        ),
        DeclareLaunchArgument(
            "max_limb_spheres",
            default_value="12",
            description="Maximum number of spheres used per limb segment.",
        ),
        DeclareLaunchArgument(
            "publish_limbs",
            default_value="true",
            description="Publish limb spheres when both endpoint joints are valid.",
        ),
        DeclareLaunchArgument(
            "publish_body_obstacle",
            default_value="true",
            description="Publish one larger body obstacle sphere from visible MediaPipe joints.",
        ),
        DeclareLaunchArgument(
            "body_radius_m",
            default_value="0.25",
            description="Body obstacle sphere radius.",
        ),
        DeclareLaunchArgument(
            "body_min_joints",
            default_value="1",
            description="Minimum valid joints required to publish the body obstacle sphere.",
        ),
        DeclareLaunchArgument(
            "publish_segmentation_obstacles",
            default_value="true",
            description="Publish person-mask depth obstacle spheres from MediaPipe segmentation.",
        ),
        DeclareLaunchArgument(
            "segmentation_mode",
            default_value="compact",
            description='Segmentation obstacle mode: "compact" publishes a few large band spheres; "dense" samples many mask pixels.',
        ),
        DeclareLaunchArgument(
            "segmentation_threshold",
            default_value="0.65",
            description="Minimum person-mask probability before sampling a pixel.",
        ),
        DeclareLaunchArgument(
            "segmentation_stride_px",
            default_value="24",
            description="Pixel stride between sampled person-mask obstacle spheres.",
        ),
        DeclareLaunchArgument(
            "segmentation_compact_bands",
            default_value="3",
            description="Number of vertical mask bands used in compact segmentation mode.",
        ),
        DeclareLaunchArgument(
            "segmentation_marker_radius_m",
            default_value="0.18",
            description="Sphere radius in meters for person-mask obstacle samples.",
        ),
        DeclareLaunchArgument(
            "max_segmentation_markers",
            default_value="80",
            description="Maximum person-mask obstacle spheres per frame.",
        ),
        DeclareLaunchArgument(
            "segmentation_min_markers",
            default_value="1",
            description="Minimum valid depth samples required to publish segmentation obstacles.",
        ),
        DeclareLaunchArgument(
            "process_every_n",
            default_value="1",
            description="Run MediaPipe on every Nth color image.",
        ),
        DeclareLaunchArgument(
            "marker_lifetime_s",
            default_value="1.0",
            description="Marker lifetime in seconds. Larger values reduce flicker.",
        ),
        Node(
            package="rb10_rmpflow_rviz",
            executable="mediapipe_pose_obstacle_node.py",
            name="mediapipe_pose_obstacle_node",
            output="screen",
            parameters=[{
                "color_topic": LaunchConfiguration("color_topic"),
                "depth_topic": LaunchConfiguration("depth_topic"),
                "camera_info_topic": LaunchConfiguration("camera_info_topic"),
                "marker_topic": LaunchConfiguration("marker_topic"),
                "visibility_threshold": ParameterValue(
                    LaunchConfiguration("visibility_threshold"), value_type=float),
                "landmark_set": LaunchConfiguration("landmark_set"),
                "min_valid_joints": ParameterValue(
                    LaunchConfiguration("min_valid_joints"), value_type=int),
                "min_core_joints": ParameterValue(
                    LaunchConfiguration("min_core_joints"), value_type=int),
                "max_joint_depth_deviation_m": ParameterValue(
                    LaunchConfiguration("max_joint_depth_deviation_m"), value_type=float),
                "min_depth_m": ParameterValue(
                    LaunchConfiguration("min_depth_m"), value_type=float),
                "max_depth_m": ParameterValue(
                    LaunchConfiguration("max_depth_m"), value_type=float),
                "joint_radius_m": ParameterValue(
                    LaunchConfiguration("joint_radius_m"), value_type=float),
                "limb_radius_m": ParameterValue(
                    LaunchConfiguration("limb_radius_m"), value_type=float),
                "limb_spacing_m": ParameterValue(
                    LaunchConfiguration("limb_spacing_m"), value_type=float),
                "max_limb_spheres": ParameterValue(
                    LaunchConfiguration("max_limb_spheres"), value_type=int),
                "publish_limbs": ParameterValue(
                    LaunchConfiguration("publish_limbs"), value_type=bool),
                "publish_body_obstacle": ParameterValue(
                    LaunchConfiguration("publish_body_obstacle"), value_type=bool),
                "body_radius_m": ParameterValue(
                    LaunchConfiguration("body_radius_m"), value_type=float),
                "body_min_joints": ParameterValue(
                    LaunchConfiguration("body_min_joints"), value_type=int),
                "publish_segmentation_obstacles": ParameterValue(
                    LaunchConfiguration("publish_segmentation_obstacles"), value_type=bool),
                "segmentation_mode": LaunchConfiguration("segmentation_mode"),
                "segmentation_threshold": ParameterValue(
                    LaunchConfiguration("segmentation_threshold"), value_type=float),
                "segmentation_stride_px": ParameterValue(
                    LaunchConfiguration("segmentation_stride_px"), value_type=int),
                "segmentation_compact_bands": ParameterValue(
                    LaunchConfiguration("segmentation_compact_bands"), value_type=int),
                "segmentation_marker_radius_m": ParameterValue(
                    LaunchConfiguration("segmentation_marker_radius_m"), value_type=float),
                "max_segmentation_markers": ParameterValue(
                    LaunchConfiguration("max_segmentation_markers"), value_type=int),
                "segmentation_min_markers": ParameterValue(
                    LaunchConfiguration("segmentation_min_markers"), value_type=int),
                "process_every_n": ParameterValue(
                    LaunchConfiguration("process_every_n"), value_type=int),
                "marker_lifetime_s": ParameterValue(
                    LaunchConfiguration("marker_lifetime_s"), value_type=float),
            }],
        ),
    ])
