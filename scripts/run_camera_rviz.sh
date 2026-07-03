#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
ROS_DISTRO_NAME="${ROS_DISTRO:-humble}"

DEVICE="/dev/video0"
WIDTH="640"
HEIGHT="480"
FPS="30"
PIXEL_FORMAT="YUYV"
OUTPUT_ENCODING="rgb8"
DRIVER="auto"
CAMERA_NAMESPACE="camera"
CAMERA_NAME="camera"
RGB_TOPIC=""
DEPTH_TOPIC=""
POINTCLOUD_TOPIC=""
ALIGN_DEPTH="true"
POINTCLOUD="false"
POINTCLOUD_REQUESTED="false"
FIXED_FRAME=""
PUBLISH_STATIC_TF="false"
RS_COLOR_PROFILE=""
RS_DEPTH_PROFILE=""
USE_RVIZ="true"
VISUALIZE_OBSTACLES="false"
VISUALIZE_POSE_OBSTACLES="false"

usage() {
  cat <<'EOF'
Usage:
  scripts/run_camera_rviz.sh [options]

Options:
  --device PATH         Camera device path. Default: /dev/video0
  --width PX           Image width. Default: 640
  --height PX          Image height. Default: 480
  --fps FPS            Framerate. Default: 30
  --format FORMAT      Pixel format. Default: YUYV. For usb_cam, MJPG maps to mjpeg2rgb.
  --encoding ENCODING  v4l2 output encoding. Default: rgb8
  --driver DRIVER      auto, v4l2_camera, usb_cam, or realsense2_camera. Default: auto
  --namespace NAME     Camera namespace. Default: camera
  --camera-name NAME   Camera node/name. Default: camera
  --rgb-topic TOPIC    RGB image topic to show in RViz. Default depends on driver.
  --depth-topic TOPIC  Depth image topic to show in RViz. Default depends on driver.
  --points-topic TOPIC PointCloud2 topic to show in RViz. Default depends on driver.
  --aligned-depth      For RealSense, show /aligned_depth_to_color/image_raw.
  --pointcloud         For RealSense, enable depth/color pointcloud.
  --obstacles          Start depth-only obstacle feature visualization in RViz.
  --no-obstacles       Do not start obstacle feature visualization. Default.
  --pose-obstacles     Start MediaPipe human joint/limb obstacle visualization in RViz.
  --no-pose-obstacles  Do not start MediaPipe human pose visualization. Default.
  --fast               RealSense fast view: 640x480x30 RGB/depth. Keeps pointcloud if --pointcloud is set.
  --rs-color-profile P RealSense color profile, e.g. 640x480x30.
  --rs-depth-profile P RealSense depth profile, e.g. 640x480x30.
  --no-rviz           Start camera topics only, without RViz rendering.
  --rviz              Start RViz. Default.
  --fixed-frame FRAME  RViz fixed frame. Default: camera_depth_optical_frame for RealSense.
  --publish-static-tf  Publish temporary fixed-frame -> camera_link TF.
  --no-static-tf       Do not publish temporary camera TF. Default.
  -h, --help           Show this help.

Examples:
  scripts/run_camera_rviz.sh
  scripts/run_camera_rviz.sh --device /dev/video2 --format MJPG
  scripts/run_camera_rviz.sh --driver realsense2_camera
  scripts/run_camera_rviz.sh --driver realsense2_camera --fast
  scripts/run_camera_rviz.sh --driver realsense2_camera --fast --obstacles
  scripts/run_camera_rviz.sh --driver realsense2_camera --fast --pointcloud --obstacles
  scripts/run_camera_rviz.sh --driver realsense2_camera --fast --pointcloud --pose-obstacles
  scripts/run_camera_rviz.sh --driver realsense2_camera --fast --no-rviz
  scripts/run_camera_rviz.sh --width 1280 --height 720 --fps 30
EOF
}

topic_join() {
  local result=""
  local part
  for part in "$@"; do
    part="${part#/}"
    part="${part%/}"
    [[ -z "$part" ]] && continue
    result="${result}/${part}"
  done
  [[ -n "$result" ]] && printf '%s\n' "$result" || printf '/\n'
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --device)
      DEVICE="${2:?--device needs a value}"
      shift 2
      ;;
    --width)
      WIDTH="${2:?--width needs a value}"
      shift 2
      ;;
    --height)
      HEIGHT="${2:?--height needs a value}"
      shift 2
      ;;
    --fps)
      FPS="${2:?--fps needs a value}"
      shift 2
      ;;
    --format)
      PIXEL_FORMAT="${2:?--format needs a value}"
      shift 2
      ;;
    --encoding)
      OUTPUT_ENCODING="${2:?--encoding needs a value}"
      shift 2
      ;;
    --driver)
      DRIVER="${2:?--driver needs a value}"
      shift 2
      ;;
    --namespace)
      CAMERA_NAMESPACE="${2:?--namespace needs a value}"
      shift 2
      ;;
    --camera-name)
      CAMERA_NAME="${2:?--camera-name needs a value}"
      shift 2
      ;;
    --rgb-topic)
      RGB_TOPIC="${2:?--rgb-topic needs a value}"
      shift 2
      ;;
    --depth-topic)
      DEPTH_TOPIC="${2:?--depth-topic needs a value}"
      shift 2
      ;;
    --points-topic)
      POINTCLOUD_TOPIC="${2:?--points-topic needs a value}"
      POINTCLOUD="true"
      POINTCLOUD_REQUESTED="true"
      shift 2
      ;;
    --aligned-depth)
      DEPTH_TOPIC="$(topic_join "$CAMERA_NAMESPACE" "$CAMERA_NAME" aligned_depth_to_color image_raw)"
      ALIGN_DEPTH="true"
      shift
      ;;
    --pointcloud)
      POINTCLOUD="true"
      POINTCLOUD_REQUESTED="true"
      shift
      ;;
    --obstacles)
      VISUALIZE_OBSTACLES="true"
      shift
      ;;
    --no-obstacles)
      VISUALIZE_OBSTACLES="false"
      shift
      ;;
    --pose-obstacles)
      VISUALIZE_POSE_OBSTACLES="true"
      shift
      ;;
    --no-pose-obstacles)
      VISUALIZE_POSE_OBSTACLES="false"
      shift
      ;;
    --fast)
      RS_COLOR_PROFILE="640x480x30"
      RS_DEPTH_PROFILE="640x480x30"
      if [[ "$POINTCLOUD_REQUESTED" != "true" ]]; then
        POINTCLOUD="false"
        POINTCLOUD_TOPIC=""
      fi
      ALIGN_DEPTH="true"
      DEPTH_TOPIC="$(topic_join "$CAMERA_NAMESPACE" "$CAMERA_NAME" aligned_depth_to_color image_raw)"
      shift
      ;;
    --rs-color-profile)
      RS_COLOR_PROFILE="${2:?--rs-color-profile needs a value}"
      shift 2
      ;;
    --rs-depth-profile)
      RS_DEPTH_PROFILE="${2:?--rs-depth-profile needs a value}"
      shift 2
      ;;
    --no-rviz)
      USE_RVIZ="false"
      shift
      ;;
    --rviz)
      USE_RVIZ="true"
      shift
      ;;
    --fixed-frame)
      FIXED_FRAME="${2:?--fixed-frame needs a value}"
      shift 2
      ;;
    --publish-static-tf)
      PUBLISH_STATIC_TF="true"
      shift
      ;;
    --no-static-tf)
      PUBLISH_STATIC_TF="false"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

set +u
source "/opt/ros/${ROS_DISTRO_NAME}/setup.bash"

if [[ -f "$WORKSPACE_DIR/install/setup.bash" ]]; then
  source "$WORKSPACE_DIR/install/setup.bash"
fi
set -u

if ! ros2 pkg prefix rb10_rmpflow_rviz >/dev/null 2>&1; then
  echo "Package rb10_rmpflow_rviz is not built/sourced yet." >&2
  echo "Run:" >&2
  echo "  cd $WORKSPACE_DIR" >&2
  echo "  colcon build --packages-select rb10_rmpflow_rviz --symlink-install" >&2
  echo "  source install/setup.bash" >&2
  exit 1
fi

driver_available() {
  ros2 pkg prefix "$1" >/dev/null 2>&1
}

if [[ "$DRIVER" == "auto" ]]; then
  if driver_available v4l2_camera; then
    DRIVER="v4l2_camera"
  elif driver_available usb_cam; then
    DRIVER="usb_cam"
  elif driver_available realsense2_camera; then
    DRIVER="realsense2_camera"
  else
    echo "No supported ROS camera driver is installed." >&2
    echo "Recommended install:" >&2
    echo "  sudo apt install ros-${ROS_DISTRO_NAME}-v4l2-camera" >&2
    echo "Then run this script again." >&2
    exit 1
  fi
fi

if [[ "$DRIVER" != "v4l2_camera" && "$DRIVER" != "usb_cam" && "$DRIVER" != "realsense2_camera" ]]; then
  echo "Unsupported --driver value: $DRIVER" >&2
  echo "Use auto, v4l2_camera, usb_cam, or realsense2_camera." >&2
  exit 2
fi

if [[ "$DRIVER" != "realsense2_camera" && ! -e "$DEVICE" ]]; then
  echo "Camera device not found: $DEVICE" >&2
  echo "Plug in the camera and check available devices:" >&2
  echo "  ls -l /dev/video*" >&2
  exit 1
fi

if ! driver_available "$DRIVER"; then
  echo "ROS package '$DRIVER' is not installed." >&2
  if [[ "$DRIVER" == "v4l2_camera" ]]; then
    echo "Install it with:" >&2
    echo "  sudo apt install ros-${ROS_DISTRO_NAME}-v4l2-camera" >&2
  elif [[ "$DRIVER" == "usb_cam" ]]; then
    echo "Install it with:" >&2
    echo "  sudo apt install ros-${ROS_DISTRO_NAME}-usb-cam" >&2
  else
    echo "Install it with:" >&2
    echo "  sudo apt install ros-${ROS_DISTRO_NAME}-realsense2-camera" >&2
  fi
  exit 1
fi

if [[ -z "$RGB_TOPIC" ]]; then
  if [[ "$DRIVER" == "realsense2_camera" ]]; then
    RGB_TOPIC="$(topic_join "$CAMERA_NAMESPACE" "$CAMERA_NAME" color image_raw)"
  else
    RGB_TOPIC="$(topic_join "$CAMERA_NAMESPACE" image_raw)"
  fi
fi

if [[ -z "$DEPTH_TOPIC" ]]; then
  if [[ "$DRIVER" == "realsense2_camera" ]]; then
    DEPTH_TOPIC="$(topic_join "$CAMERA_NAMESPACE" "$CAMERA_NAME" depth image_rect_raw)"
  else
    DEPTH_TOPIC="$(topic_join "$CAMERA_NAMESPACE" depth image_raw)"
  fi
fi

if [[ -z "$POINTCLOUD_TOPIC" && "$POINTCLOUD" == "true" ]]; then
  if [[ "$DRIVER" == "realsense2_camera" ]]; then
    POINTCLOUD_TOPIC="$(topic_join "$CAMERA_NAMESPACE" "$CAMERA_NAME" depth color points)"
  fi
fi

if [[ -z "$FIXED_FRAME" ]]; then
  if [[ "$DRIVER" == "realsense2_camera" ]]; then
    FIXED_FRAME="camera_depth_optical_frame"
  else
    FIXED_FRAME="base_link"
  fi
fi

echo "Launching camera pipeline"
echo "  driver: $DRIVER"
if [[ "$DRIVER" == "realsense2_camera" ]]; then
  echo "  device: RealSense driver auto-detect"
else
  echo "  device: $DEVICE"
fi
echo "  rgb:    $RGB_TOPIC"
echo "  depth:  $DEPTH_TOPIC"
echo "  points: $POINTCLOUD_TOPIC"
echo "  obstacle visualization: $VISUALIZE_OBSTACLES"
echo "  pose obstacle visualization: $VISUALIZE_POSE_OBSTACLES"
echo "  frame:  $FIXED_FRAME"
echo "  rviz:   $USE_RVIZ"
if [[ "$DRIVER" == "realsense2_camera" ]]; then
  echo "  color profile: ${RS_COLOR_PROFILE:-driver default}"
  echo "  depth profile: ${RS_DEPTH_PROFILE:-driver default}"
fi
echo "  size:   ${WIDTH}x${HEIGHT}@${FPS}"

launch_args=(
  "camera_driver:=$DRIVER"
  "camera_namespace:=$CAMERA_NAMESPACE"
  "camera_name:=$CAMERA_NAME"
  "video_device:=$DEVICE"
  "image_width:=$WIDTH"
  "image_height:=$HEIGHT"
  "framerate:=$FPS"
  "pixel_format:=$PIXEL_FORMAT"
  "output_encoding:=$OUTPUT_ENCODING"
  "rgb_topic:=$RGB_TOPIC"
  "depth_topic:=$DEPTH_TOPIC"
  "align_depth:=$ALIGN_DEPTH"
  "pointcloud:=$POINTCLOUD"
  "use_obstacle_feature:=$VISUALIZE_OBSTACLES"
  "use_human_pose_obstacles:=$VISUALIZE_POSE_OBSTACLES"
  "fixed_frame:=$FIXED_FRAME"
  "publish_static_tf:=$PUBLISH_STATIC_TF"
  "use_rviz:=$USE_RVIZ"
)

if [[ -n "$POINTCLOUD_TOPIC" ]]; then
  launch_args+=("pointcloud_topic:=$POINTCLOUD_TOPIC")
fi
if [[ -n "$RS_COLOR_PROFILE" ]]; then
  launch_args+=("realsense_color_profile:=$RS_COLOR_PROFILE")
fi
if [[ -n "$RS_DEPTH_PROFILE" ]]; then
  launch_args+=("realsense_depth_profile:=$RS_DEPTH_PROFILE")
fi

exec ros2 launch rb10_rmpflow_rviz camera_rviz.launch.py "${launch_args[@]}"
