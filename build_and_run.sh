#!/bin/bash
# Build and run RB10 RMPflow RViz simulation

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(dirname "$SCRIPT_DIR")"

echo "========================================"
echo "RB10 RMPflow RViz Simulation"
echo "========================================"

# Check if ROS 2 is sourced
if [ -z "$ROS_DISTRO" ]; then
    echo "Error: ROS 2 is not sourced. Please source your ROS 2 installation."
    echo "  source /opt/ros/humble/setup.bash"
    exit 1
fi

echo "ROS 2 Distro: $ROS_DISTRO"

# Setup rmp2 Python path
export PYTHONPATH="$WORKSPACE_DIR/rmp2:$PYTHONPATH"
echo "Added rmp2 to PYTHONPATH"

# Check if TensorFlow is available
python3 -c "import tensorflow as tf; print(f'TensorFlow version: {tf.__version__}')" 2>/dev/null || {
    echo "Warning: TensorFlow not found. RMPflow will use fallback PD controller."
    echo "To install TensorFlow: pip install tensorflow"
}

# Build the package
echo ""
echo "Building rb10_rmpflow_rviz package..."
cd "$WORKSPACE_DIR"

# Create colcon workspace structure if needed
if [ ! -f "$WORKSPACE_DIR/src" ]; then
    mkdir -p "$WORKSPACE_DIR/colcon_ws/src"
    ln -sf "$SCRIPT_DIR" "$WORKSPACE_DIR/colcon_ws/src/rb10_rmpflow_rviz" 2>/dev/null || true
fi

# Build
cd "$WORKSPACE_DIR"
colcon build --packages-select rb10_rmpflow_rviz --symlink-install

# Source the workspace
source "$WORKSPACE_DIR/install/setup.bash"

echo ""
echo "Build complete!"
echo ""
echo "========================================"
echo "Starting RViz simulation..."
echo "========================================"
echo ""
echo "Controls:"
echo "  - Drag green sphere: Move goal position"
echo "  - Drag red sphere(s): Move obstacles"
echo ""

# Run the launch file
ros2 launch rb10_rmpflow_rviz rb10_rmpflow.launch.py "$@"
