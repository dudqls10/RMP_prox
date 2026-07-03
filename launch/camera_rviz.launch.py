#!/usr/bin/env python3

import os
import tempfile

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, IncludeLaunchDescription, OpaqueFunction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _topic(*parts: str) -> str:
    cleaned = [part.strip("/") for part in parts if part and part.strip("/")]
    if not cleaned:
        return "/"
    return "/" + "/".join(cleaned)


def _default_rgb_topic(camera_driver: str, namespace: str, camera_name: str) -> str:
    if camera_driver == "realsense2_camera":
        return _topic(namespace, camera_name, "color", "image_raw")
    return _topic(namespace, "image_raw")


def _default_depth_topic(camera_driver: str, namespace: str, camera_name: str) -> str:
    if camera_driver == "realsense2_camera":
        return _topic(namespace, camera_name, "depth", "image_rect_raw")
    return _topic(namespace, "depth", "image_raw")


def _default_pointcloud_topic(camera_driver: str, namespace: str, camera_name: str) -> str:
    if camera_driver == "realsense2_camera":
        return _topic(namespace, camera_name, "depth", "color", "points")
    return ""


def _usb_cam_pixel_format(pixel_format: str) -> str:
    normalized = pixel_format.strip().lower()
    if normalized in ("yuyv", "yuyv2rgb"):
        return "yuyv2rgb"
    if normalized in ("mjpg", "mjpeg", "mjpeg2rgb"):
        return "mjpeg2rgb"
    return normalized


def _bool_value(value: str) -> bool:
    return value.strip().lower() in ("1", "true", "yes", "on")


def _write_rviz_config(
    rgb_topic: str,
    depth_topic: str,
    pointcloud_topic: str,
    nearest_marker_topic: str,
    body_marker_topic: str,
    human_pose_marker_topic: str,
    fixed_frame: str,
) -> str:
    pointcloud_display = ""
    if pointcloud_topic:
        pointcloud_display = f"""    - Alpha: 1
      Autocompute Intensity Bounds: true
      Autocompute Value Bounds:
        Max Value: 10
        Min Value: -10
        Value: true
      Axis: Z
      Channel Name: intensity
      Class: rviz_default_plugins/PointCloud2
      Color: 255; 255; 255
      Color Transformer: RGB8
      Decay Time: 0
      Enabled: true
      Invert Rainbow: false
      Max Color: 255; 255; 255
      Max Intensity: 4096
      Min Color: 0; 0; 0
      Min Intensity: 0
      Name: CameraPointCloud
      Position Transformer: XYZ
      Selectable: true
      Size (Pixels): 3
      Size (m): 0.01
      Style: Flat Squares
      Topic:
        Depth: 5
        Durability Policy: Volatile
        Filter size: 10
        History Policy: Keep Last
        Reliability Policy: Best Effort
        Value: {pointcloud_topic}
      Use Fixed Frame: true
      Use rainbow: true
      Value: true
"""

    rviz_config = f"""Panels:
  - Class: rviz_common/Displays
    Name: Displays
  - Class: rviz_common/Selection
    Name: Selection
  - Class: rviz_common/Tool Properties
    Name: Tool Properties
  - Class: rviz_common/Views
    Name: Views

Visualization Manager:
  Class: ""
  Displays:
    - Alpha: 0.5
      Cell Size: 1
      Class: rviz_default_plugins/Grid
      Color: 160; 160; 164
      Enabled: true
      Name: Grid
      Normal Cell Count: 0
      Offset:
        X: 0
        Y: 0
        Z: 0
      Plane: XY
      Plane Cell Count: 10
      Reference Frame: <Fixed Frame>
      Value: true
{pointcloud_display.rstrip()}
    - Class: rviz_default_plugins/Image
      Enabled: true
      Max Value: 1
      Median window: 5
      Min Value: 0
      Name: RGBImage
      Normalize Range: true
      Topic:
        Depth: 5
        Durability Policy: Volatile
        History Policy: Keep Last
        Reliability Policy: Reliable
        Value: {rgb_topic}
      Value: true
    - Class: rviz_default_plugins/Image
      Enabled: true
      Max Value: 5000
      Median window: 5
      Min Value: 0
      Name: DepthImage
      Normalize Range: true
      Topic:
        Depth: 5
        Durability Policy: Volatile
        History Policy: Keep Last
        Reliability Policy: Reliable
        Value: {depth_topic}
      Value: true
    - Class: rviz_default_plugins/Marker
      Enabled: true
      Name: NearestObstaclePoint
      Namespaces:
        camera_obstacle_feature: true
      Topic:
        Depth: 5
        Durability Policy: Volatile
        History Policy: Keep Last
        Reliability Policy: Reliable
        Value: {nearest_marker_topic}
      Value: true
    - Class: rviz_default_plugins/MarkerArray
      Enabled: true
      Name: CameraBodyObstacle
      Namespaces:
        camera_body_obstacles: true
      Topic:
        Depth: 5
        Durability Policy: Volatile
        History Policy: Keep Last
        Reliability Policy: Reliable
        Value: {body_marker_topic}
      Value: true
    - Class: rviz_default_plugins/MarkerArray
      Enabled: true
      Name: HumanPoseObstacles
      Namespaces:
        human_pose_body: true
        human_pose_joints: true
        human_pose_limbs: true
        human_segmentation_obstacles: true
      Topic:
        Depth: 5
        Durability Policy: Volatile
        History Policy: Keep Last
        Reliability Policy: Reliable
        Value: {human_pose_marker_topic}
      Value: true
    - Class: rviz_default_plugins/TF
      Enabled: true
      Frame Timeout: 15
      Frames:
        All Enabled: true
      Marker Scale: 0.25
      Name: TF
      Show Arrows: true
      Show Axes: true
      Show Names: true
      Update Interval: 0
      Value: true
  Enabled: true
  Global Options:
    Background Color: 48; 48; 48
    Fixed Frame: {fixed_frame}
    Frame Rate: 30
  Name: root
  Tools:
    - Class: rviz_default_plugins/Interact
      Hide Inactive Objects: true
    - Class: rviz_default_plugins/MoveCamera
    - Class: rviz_default_plugins/Select
    - Class: rviz_default_plugins/FocusCamera
    - Class: rviz_default_plugins/Measure
      Line color: 128; 128; 0
  Transformation:
    Current:
      Class: rviz_default_plugins/TF
  Value: true
  Views:
    Current:
      Class: rviz_default_plugins/Orbit
      Distance: 1.5
      Enable Stereo Rendering:
        Stereo Eye Separation: 0.06
        Stereo Focal Distance: 1
        Swap Stereo Eyes: false
        Value: false
      Focal Point:
        X: 0
        Y: 0
        Z: 0
      Focal Shape Fixed Size: true
      Focal Shape Size: 0.05
      Invert Z Axis: false
      Name: Current View
      Near Clip Distance: 0.01
      Pitch: 0.5
      Target Frame: <Fixed Frame>
      Value: Orbit (rviz_default_plugins)
      Yaw: 0.8
    Saved: ~

Window Geometry:
  Displays:
    collapsed: false
  Height: 900
  Hide Left Dock: false
  Hide Right Dock: false
  Selection:
    collapsed: false
  Tool Properties:
    collapsed: false
  Views:
    collapsed: false
  Width: 1400
  X: 80
  Y: 60
"""

    rviz_file = os.path.join(tempfile.gettempdir(), "rb10_camera_rviz.rviz")
    with open(rviz_file, "w", encoding="utf-8") as stream:
        stream.write(rviz_config)
    return rviz_file


def _launch_setup(context, *args, **kwargs):
    camera_driver = LaunchConfiguration("camera_driver").perform(context)
    camera_namespace = LaunchConfiguration("camera_namespace").perform(context)
    camera_name = LaunchConfiguration("camera_name").perform(context)
    video_device = LaunchConfiguration("video_device").perform(context)
    camera_frame = LaunchConfiguration("camera_frame").perform(context)
    fixed_frame = LaunchConfiguration("fixed_frame").perform(context)
    image_width = int(LaunchConfiguration("image_width").perform(context))
    image_height = int(LaunchConfiguration("image_height").perform(context))
    framerate = int(LaunchConfiguration("framerate").perform(context))
    pixel_format = LaunchConfiguration("pixel_format").perform(context)
    output_encoding = LaunchConfiguration("output_encoding").perform(context)
    rgb_topic = LaunchConfiguration("rgb_topic").perform(context)
    depth_topic = LaunchConfiguration("depth_topic").perform(context)
    pointcloud_topic = LaunchConfiguration("pointcloud_topic").perform(context)
    nearest_marker_topic = LaunchConfiguration("nearest_marker_topic").perform(context)
    body_marker_topic = LaunchConfiguration("body_marker_topic").perform(context)
    human_pose_marker_topic = LaunchConfiguration("human_pose_marker_topic").perform(context)
    obstacle_camera_info_topic = LaunchConfiguration("obstacle_camera_info_topic").perform(context)
    pose_color_topic = LaunchConfiguration("pose_color_topic").perform(context)
    pose_depth_topic = LaunchConfiguration("pose_depth_topic").perform(context)
    pose_camera_info_topic = LaunchConfiguration("pose_camera_info_topic").perform(context)
    pose_visibility_threshold = float(
        LaunchConfiguration("pose_visibility_threshold").perform(context))
    pose_landmark_set = LaunchConfiguration("pose_landmark_set").perform(context)
    pose_min_valid_joints = int(LaunchConfiguration("pose_min_valid_joints").perform(context))
    pose_min_core_joints = int(LaunchConfiguration("pose_min_core_joints").perform(context))
    pose_max_joint_depth_deviation = LaunchConfiguration(
        "pose_max_joint_depth_deviation").perform(context)
    pose_max_joint_depth_deviation = float(pose_max_joint_depth_deviation)
    pose_joint_radius = float(LaunchConfiguration("pose_joint_radius").perform(context))
    pose_limb_radius = float(LaunchConfiguration("pose_limb_radius").perform(context))
    pose_limb_spacing = float(LaunchConfiguration("pose_limb_spacing").perform(context))
    pose_max_limb_spheres = int(LaunchConfiguration("pose_max_limb_spheres").perform(context))
    pose_publish_limbs = _bool_value(LaunchConfiguration("pose_publish_limbs").perform(context))
    pose_publish_body_obstacle = _bool_value(
        LaunchConfiguration("pose_publish_body_obstacle").perform(context))
    pose_body_radius = float(LaunchConfiguration("pose_body_radius").perform(context))
    pose_body_min_joints = int(LaunchConfiguration("pose_body_min_joints").perform(context))
    pose_publish_segmentation_obstacles = _bool_value(
        LaunchConfiguration("pose_publish_segmentation_obstacles").perform(context))
    pose_segmentation_mode = LaunchConfiguration("pose_segmentation_mode").perform(context)
    pose_segmentation_threshold = float(
        LaunchConfiguration("pose_segmentation_threshold").perform(context))
    pose_segmentation_stride_px = int(
        LaunchConfiguration("pose_segmentation_stride_px").perform(context))
    pose_segmentation_compact_bands = int(
        LaunchConfiguration("pose_segmentation_compact_bands").perform(context))
    pose_segmentation_marker_radius = float(
        LaunchConfiguration("pose_segmentation_marker_radius").perform(context))
    pose_max_segmentation_markers = int(
        LaunchConfiguration("pose_max_segmentation_markers").perform(context))
    pose_segmentation_min_markers = int(
        LaunchConfiguration("pose_segmentation_min_markers").perform(context))
    pose_marker_lifetime = float(LaunchConfiguration("pose_marker_lifetime").perform(context))
    align_depth = LaunchConfiguration("align_depth").perform(context)
    pointcloud = LaunchConfiguration("pointcloud").perform(context)
    color_profile = LaunchConfiguration("realsense_color_profile").perform(context)
    depth_profile = LaunchConfiguration("realsense_depth_profile").perform(context)
    pointcloud_enabled = pointcloud.lower() in ("1", "true", "yes", "on")

    if not fixed_frame:
        fixed_frame = "camera_depth_optical_frame" if camera_driver == "realsense2_camera" else "base_link"
    if not rgb_topic:
        rgb_topic = _default_rgb_topic(camera_driver, camera_namespace, camera_name)
    if not depth_topic:
        depth_topic = _default_depth_topic(camera_driver, camera_namespace, camera_name)
    if not pointcloud_topic and pointcloud_enabled:
        pointcloud_topic = _default_pointcloud_topic(camera_driver, camera_namespace, camera_name)
    if not obstacle_camera_info_topic:
        if camera_driver == "realsense2_camera" and "aligned_depth_to_color" in depth_topic:
            obstacle_camera_info_topic = _topic(
                camera_namespace, camera_name, "aligned_depth_to_color", "camera_info")
        elif camera_driver == "realsense2_camera":
            obstacle_camera_info_topic = _topic(camera_namespace, camera_name, "depth", "camera_info")
        else:
            obstacle_camera_info_topic = _topic(camera_namespace, "camera_info")
    if not pose_color_topic:
        pose_color_topic = rgb_topic
    if not pose_depth_topic:
        pose_depth_topic = depth_topic
    if not pose_camera_info_topic:
        pose_camera_info_topic = obstacle_camera_info_topic
    rviz_config_path = _write_rviz_config(
        rgb_topic,
        depth_topic,
        pointcloud_topic,
        nearest_marker_topic,
        body_marker_topic,
        human_pose_marker_topic,
        fixed_frame,
    )

    actions = []

    if camera_driver == "v4l2_camera":
        actions.append(
            Node(
                package="v4l2_camera",
                executable="v4l2_camera_node",
                namespace=camera_namespace,
                name=camera_name,
                output="screen",
                parameters=[{
                    "video_device": video_device,
                    "image_size": [image_width, image_height],
                    "time_per_frame": [1, framerate],
                    "pixel_format": pixel_format,
                    "output_encoding": output_encoding,
                    "camera_frame_id": camera_frame,
                }],
            )
        )
    elif camera_driver == "usb_cam":
        actions.append(
            Node(
                package="usb_cam",
                executable="usb_cam_node_exe",
                namespace=camera_namespace,
                name=camera_name,
                output="screen",
                parameters=[{
                    "video_device": video_device,
                    "image_width": image_width,
                    "image_height": image_height,
                    "framerate": float(framerate),
                    "pixel_format": _usb_cam_pixel_format(pixel_format),
                    "frame_id": camera_frame,
                    "camera_name": camera_name,
                }],
            )
        )
    elif camera_driver == "realsense2_camera":
        realsense_launch = os.path.join(
            get_package_share_directory("realsense2_camera"),
            "launch",
            "rs_launch.py",
        )
        realsense_arguments = {
            "camera_name": camera_name,
            "camera_namespace": camera_namespace,
            "enable_color": "true",
            "enable_depth": "true",
            "enable_gyro": "false",
            "enable_accel": "false",
            "enable_motion": "false",
            "align_depth.enable": align_depth,
            "pointcloud.enable": pointcloud,
            "publish_tf": "true",
            "base_frame_id": camera_frame,
        }
        if color_profile:
            realsense_arguments["rgb_camera.color_profile"] = color_profile
        if depth_profile:
            realsense_arguments["depth_module.depth_profile"] = depth_profile

        actions.append(
            GroupAction(
                scoped=True,
                forwarding=False,
                actions=[
                    IncludeLaunchDescription(
                        PythonLaunchDescriptionSource(realsense_launch),
                        launch_arguments=realsense_arguments.items(),
                    )
                ],
            )
        )
    else:
        raise RuntimeError(
            'camera_driver must be "v4l2_camera", "usb_cam", or "realsense2_camera"; '
            f'got "{camera_driver}"'
        )

    actions.extend([
        Node(
            package="rb10_rmpflow_rviz",
            executable="camera_obstacle_feature_node.py",
            name="camera_obstacle_feature_node",
            output="screen",
            condition=IfCondition(LaunchConfiguration("use_obstacle_feature")),
            parameters=[{
                "depth_topic": depth_topic,
                "camera_info_topic": obstacle_camera_info_topic,
                "marker_topic": nearest_marker_topic,
                "body_marker_topic": body_marker_topic,
                "body_marker_shape": "sphere",
                "publish_collision_obstacles": False,
            }],
        ),
        Node(
            package="rb10_rmpflow_rviz",
            executable="mediapipe_pose_obstacle_node.py",
            name="mediapipe_pose_obstacle_node",
            output="screen",
            condition=IfCondition(LaunchConfiguration("use_human_pose_obstacles")),
            parameters=[{
                "color_topic": pose_color_topic,
                "depth_topic": pose_depth_topic,
                "camera_info_topic": pose_camera_info_topic,
                "marker_topic": human_pose_marker_topic,
                "visibility_threshold": pose_visibility_threshold,
                "landmark_set": pose_landmark_set,
                "min_valid_joints": pose_min_valid_joints,
                "min_core_joints": pose_min_core_joints,
                "max_joint_depth_deviation_m": pose_max_joint_depth_deviation,
                "joint_radius_m": pose_joint_radius,
                "limb_radius_m": pose_limb_radius,
                "limb_spacing_m": pose_limb_spacing,
                "max_limb_spheres": pose_max_limb_spheres,
                "publish_limbs": pose_publish_limbs,
                "publish_body_obstacle": pose_publish_body_obstacle,
                "body_radius_m": pose_body_radius,
                "body_min_joints": pose_body_min_joints,
                "publish_segmentation_obstacles": pose_publish_segmentation_obstacles,
                "segmentation_mode": pose_segmentation_mode,
                "segmentation_threshold": pose_segmentation_threshold,
                "segmentation_stride_px": pose_segmentation_stride_px,
                "segmentation_compact_bands": pose_segmentation_compact_bands,
                "segmentation_marker_radius_m": pose_segmentation_marker_radius,
                "max_segmentation_markers": pose_max_segmentation_markers,
                "segmentation_min_markers": pose_segmentation_min_markers,
                "marker_lifetime_s": pose_marker_lifetime,
            }],
        ),
        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            name="camera_static_tf",
            output="screen",
            condition=IfCondition(LaunchConfiguration("publish_static_tf")),
            arguments=[
                "--x", LaunchConfiguration("camera_x"),
                "--y", LaunchConfiguration("camera_y"),
                "--z", LaunchConfiguration("camera_z"),
                "--roll", LaunchConfiguration("camera_roll"),
                "--pitch", LaunchConfiguration("camera_pitch"),
                "--yaw", LaunchConfiguration("camera_yaw"),
                "--frame-id", fixed_frame,
                "--child-frame-id", camera_frame,
            ],
        ),
        Node(
            package="rviz2",
            executable="rviz2",
            name="camera_rviz2",
            output="screen",
            condition=IfCondition(LaunchConfiguration("use_rviz")),
            arguments=["-d", rviz_config_path],
        ),
    ])

    return actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            "camera_driver",
            default_value="v4l2_camera",
            description='Camera driver package: "v4l2_camera", "usb_cam", or "realsense2_camera".',
        ),
        DeclareLaunchArgument(
            "camera_namespace",
            default_value="camera",
            description="Namespace for camera topics. Default image topic is /camera/image_raw.",
        ),
        DeclareLaunchArgument(
            "camera_name",
            default_value="camera",
            description="ROS node name for the camera driver.",
        ),
        DeclareLaunchArgument(
            "video_device",
            default_value="/dev/video0",
            description="Linux camera device path.",
        ),
        DeclareLaunchArgument(
            "image_width",
            default_value="640",
            description="Requested camera image width.",
        ),
        DeclareLaunchArgument(
            "image_height",
            default_value="480",
            description="Requested camera image height.",
        ),
        DeclareLaunchArgument(
            "framerate",
            default_value="30",
            description="Requested camera framerate.",
        ),
        DeclareLaunchArgument(
            "pixel_format",
            default_value="YUYV",
            description="Camera pixel format. v4l2_camera commonly uses YUYV; usb_cam maps MJPG to mjpeg2rgb.",
        ),
        DeclareLaunchArgument(
            "output_encoding",
            default_value="rgb8",
            description="v4l2_camera output image encoding.",
        ),
        DeclareLaunchArgument(
            "rgb_topic",
            default_value="",
            description="RGB image topic to show in RViz. Empty means choose from camera_driver.",
        ),
        DeclareLaunchArgument(
            "depth_topic",
            default_value="",
            description="Depth image topic to show in RViz. Empty means choose from camera_driver.",
        ),
        DeclareLaunchArgument(
            "pointcloud_topic",
            default_value="",
            description="PointCloud2 topic to show in RViz. Empty means choose from camera_driver.",
        ),
        DeclareLaunchArgument(
            "nearest_marker_topic",
            default_value="/camera/nearest_obstacle_marker",
            description="Marker topic for the nearest depth obstacle point.",
        ),
        DeclareLaunchArgument(
            "body_marker_topic",
            default_value="/camera/body_obstacle_markers",
            description="MarkerArray topic for the camera-derived body obstacle.",
        ),
        DeclareLaunchArgument(
            "human_pose_marker_topic",
            default_value="/camera/human_pose_obstacles",
            description="MarkerArray topic for MediaPipe human joint and limb obstacles.",
        ),
        DeclareLaunchArgument(
            "use_obstacle_feature",
            default_value="false",
            description="Start the depth-only camera obstacle feature node for RViz visualization.",
        ),
        DeclareLaunchArgument(
            "use_human_pose_obstacles",
            default_value="false",
            description="Start MediaPipe human pose obstacle visualization.",
        ),
        DeclareLaunchArgument(
            "obstacle_camera_info_topic",
            default_value="",
            description="CameraInfo topic for obstacle feature extraction. Empty chooses from depth_topic.",
        ),
        DeclareLaunchArgument(
            "pose_color_topic",
            default_value="",
            description="Color topic for MediaPipe pose. Empty uses rgb_topic.",
        ),
        DeclareLaunchArgument(
            "pose_depth_topic",
            default_value="",
            description="Aligned depth topic for MediaPipe pose. Empty uses depth_topic.",
        ),
        DeclareLaunchArgument(
            "pose_camera_info_topic",
            default_value="",
            description="CameraInfo topic for MediaPipe pose. Empty chooses from depth_topic.",
        ),
        DeclareLaunchArgument(
            "pose_visibility_threshold",
            default_value="0.65",
            description="Minimum MediaPipe landmark visibility for pose obstacle visualization.",
        ),
        DeclareLaunchArgument(
            "pose_landmark_set",
            default_value="major",
            description='MediaPipe pose landmarks to publish: "full" or "major".',
        ),
        DeclareLaunchArgument(
            "pose_min_valid_joints",
            default_value="0",
            description="Minimum valid 3D joints before publishing pose obstacles.",
        ),
        DeclareLaunchArgument(
            "pose_min_core_joints",
            default_value="1",
            description="Minimum shoulder/hip joints before publishing pose obstacles.",
        ),
        DeclareLaunchArgument(
            "pose_max_joint_depth_deviation",
            default_value="0.35",
            description="Reject pose joints far from median depth. 0 disables this filter.",
        ),
        DeclareLaunchArgument(
            "pose_joint_radius",
            default_value="0.08",
            description="Sphere radius in meters for visible pose joints.",
        ),
        DeclareLaunchArgument(
            "pose_limb_radius",
            default_value="0.10",
            description="Sphere radius in meters for pose limb samples.",
        ),
        DeclareLaunchArgument(
            "pose_limb_spacing",
            default_value="0.16",
            description="Spacing between pose limb spheres.",
        ),
        DeclareLaunchArgument(
            "pose_max_limb_spheres",
            default_value="12",
            description="Maximum pose limb spheres per segment.",
        ),
        DeclareLaunchArgument(
            "pose_publish_limbs",
            default_value="true",
            description="Publish pose limb spheres between visible joints.",
        ),
        DeclareLaunchArgument(
            "pose_publish_body_obstacle",
            default_value="true",
            description="Publish one larger body obstacle sphere from visible MediaPipe joints.",
        ),
        DeclareLaunchArgument(
            "pose_body_radius",
            default_value="0.25",
            description="Body obstacle sphere radius in meters.",
        ),
        DeclareLaunchArgument(
            "pose_body_min_joints",
            default_value="1",
            description="Minimum valid joints required to publish the body obstacle sphere.",
        ),
        DeclareLaunchArgument(
            "pose_publish_segmentation_obstacles",
            default_value="true",
            description="Publish person-mask depth obstacle spheres from MediaPipe segmentation.",
        ),
        DeclareLaunchArgument(
            "pose_segmentation_mode",
            default_value="compact",
            description='Segmentation obstacle mode: "compact" publishes a few large band spheres; "dense" samples many mask pixels.',
        ),
        DeclareLaunchArgument(
            "pose_segmentation_threshold",
            default_value="0.65",
            description="Minimum person-mask probability before sampling a pixel.",
        ),
        DeclareLaunchArgument(
            "pose_segmentation_stride_px",
            default_value="24",
            description="Pixel stride between sampled person-mask obstacle spheres.",
        ),
        DeclareLaunchArgument(
            "pose_segmentation_compact_bands",
            default_value="3",
            description="Number of vertical mask bands used in compact segmentation mode.",
        ),
        DeclareLaunchArgument(
            "pose_segmentation_marker_radius",
            default_value="0.18",
            description="Sphere radius in meters for person-mask obstacle samples.",
        ),
        DeclareLaunchArgument(
            "pose_max_segmentation_markers",
            default_value="80",
            description="Maximum person-mask obstacle spheres per frame.",
        ),
        DeclareLaunchArgument(
            "pose_segmentation_min_markers",
            default_value="1",
            description="Minimum valid depth samples required to publish segmentation obstacles.",
        ),
        DeclareLaunchArgument(
            "pose_marker_lifetime",
            default_value="1.0",
            description="Human pose marker lifetime in seconds.",
        ),
        DeclareLaunchArgument(
            "align_depth",
            default_value="true",
            description="Enable RealSense depth-to-color alignment filter.",
        ),
        DeclareLaunchArgument(
            "pointcloud",
            default_value="false",
            description="Enable RealSense pointcloud publication.",
        ),
        DeclareLaunchArgument(
            "realsense_color_profile",
            default_value="",
            description='RealSense color profile such as "640x480x30". Empty uses driver default.',
        ),
        DeclareLaunchArgument(
            "realsense_depth_profile",
            default_value="",
            description='RealSense depth profile such as "640x480x30". Empty uses driver default.',
        ),
        DeclareLaunchArgument(
            "camera_frame",
            default_value="camera_link",
            description="Frame id stamped on camera images.",
        ),
        DeclareLaunchArgument(
            "fixed_frame",
            default_value="",
            description=(
                "RViz fixed frame and parent frame for temporary camera TF. "
                "Empty uses camera_depth_optical_frame for RealSense, base_link otherwise."
            ),
        ),
        DeclareLaunchArgument(
            "publish_static_tf",
            default_value="false",
            description="Publish a temporary fixed_frame -> camera_frame transform.",
        ),
        DeclareLaunchArgument("camera_x", default_value="0.0"),
        DeclareLaunchArgument("camera_y", default_value="0.0"),
        DeclareLaunchArgument("camera_z", default_value="0.0"),
        DeclareLaunchArgument("camera_roll", default_value="0.0"),
        DeclareLaunchArgument("camera_pitch", default_value="0.0"),
        DeclareLaunchArgument("camera_yaw", default_value="0.0"),
        DeclareLaunchArgument(
            "use_rviz",
            default_value="true",
            description="Start RViz with an Image display pointed at the camera topic.",
        ),
        OpaqueFunction(function=_launch_setup),
    ])
